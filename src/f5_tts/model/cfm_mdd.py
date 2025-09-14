"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint,odeint_jacobian,odeint_jacobian_wrong,odeint_dist,odeint_jacobian_aabb,odeint_jacobian_aabb_fix,odeint_jacobian_hut

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)
import pdb,sys
from torch.distributions import Normal


class CFM_MDD(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler_mdd"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        
        cond = cond.to(next(self.parameters()).dtype)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        # if not exists(lens):
        #     lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
            
        ##MDD check
        assert torch.all(duration == duration[0])
        #assert cond_seq_len == duration[0] 
        assert lens is None
            
        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # cond_mask should be all 1 for MDD
        #cond_mask = lens_to_mask(duration)
        #assert cond_mask[0].sum() == cond_seq_len
        #if edit_mask is not None:
        #    cond_mask = cond_mask & edit_mask
        
        cond_mask = edit_mask

        duration = torch.maximum((text != -1).sum(dim=-1), duration)  # MDD: duration at least text
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        ##MDD, we don't pad anything than mask the target segment
        cond_orig = cond.clone()
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            
        self.odeint_kwargs["method"]="euler"
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory, cond_orig
    
    
    @torch.no_grad()
    def sample_from_start(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        start
    ):
        self.eval()
        # raw wave
        # if cond.ndim == 2:
        #     cond = self.mel_spec(cond)
        #     cond = cond.permute(0, 2, 1)
        #     assert cond.shape[-1] == self.num_channels
        
        cond = cond.to(next(self.parameters()).dtype)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        # if not exists(lens):
        #     lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
        
        ##MDD check
        #assert torch.all(duration == duration[0])
        #assert cond_seq_len == duration[0] 
        assert lens is None
        
        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
  
        step_cond = cond

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        y0 = start
        #y0 = torch.zeros_like(start)
        #y0 = torch.randn_like(start)
        t_start = 0

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            
        self.odeint_kwargs["method"]="euler"

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        #out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def compute_prob(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
        t_inerval=None,
    ):
        self.eval()
        # check methods
        assert self.odeint_kwargs["method"] == "euler_mdd"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass
        
        # #test
        # y1=y1[:10]
        
        trajectory, jacob_trace = odeint_jacobian(fn, y1, t, cond_mask.squeeze(-1), t_inerval, cfg_strength,  **self.odeint_kwargs)
        y0 = trajectory[-1]
        ##inverse jacob for forward pass
        jacob_trace = jacob_trace.flip([0])
        self.transformer.clear_cache()
        ##NULL
        trajectory_null, jacob_trace_null = odeint_jacobian(fn_null, y1, t, cond_mask.squeeze(-1), t_inerval, cfg_strength, **self.odeint_kwargs)
        y0_null = trajectory_null[-1]
        jacob_trace_null = jacob_trace_null.flip([0])
        self.transformer.clear_cache()
        
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = norm.log_prob(y0).sum(-1)
        log_prob_y0_null = norm.log_prob(y0_null).sum(-1)
        ## manual Euler continous change of variable
        ## the last jacob is not needed for forward pass
        for i, (jacob_t, jacob_t_null) in enumerate(zip(jacob_trace[:-1], jacob_trace_null[:-1])):
            log_prob_y0 += -jacob_t*dt[i]
            log_prob_y0_null += -jacob_t_null*dt[i]
        return log_prob_y0, log_prob_y0_null
    
    ##  Jacobian replace 1, the length
    def compute_traLen(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods, it does not matter 
        assert self.odeint_kwargs["method"] == "euler_mdd" or self.odeint_kwargs["method"] == "euler"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
  
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass
        
        # #test
        # y1=y1[:10]
        self.odeint_kwargs["method"] = "euler_mdd"
        opt_dist, step_dist = odeint_dist(fn, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        ##NULL
        opt_dist_null, step_dist_null = odeint_dist(fn_null, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        ##distance based log-prob
        ##check if opt_dist_null == opt_dist? No!
        log_prob_y0 = torch.log(opt_dist/step_dist)
        log_prob_y0_null = torch.log(opt_dist_null/step_dist_null)
    
        return log_prob_y0, log_prob_y0_null
    
    ##  Jacobian replace 2, cos similarity
    def compute_similarity(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods, it does not matter 
        assert self.odeint_kwargs["method"] == "euler_mdd" or self.odeint_kwargs["method"] == "euler"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                #x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                #text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                text_null = torch.ones(mel_target.shape[0], mel_target.shape[1], dtype=torch.long, device=device)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
        t_f = torch.linspace(0, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        
  
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            t_f = t_f + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_f) - 1 + t_f)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass
        
        # #test
        # y1=y1[:10]
        self.odeint_kwargs["method"] = "euler"
        solutions = odeint(fn, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled_y0 = solutions[-1]
        directions = ((solutions - sampled_y0).flip([0]))[1:]
        ##NULL
        solutions_null = odeint(fn_null, sampled_y0, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        directions_null = (solutions_null - sampled_y0)[1:]
        ##similarity based log-prob, t-accumulated cos-similarity diffence
        optimal_dir = y1 - sampled_y0
        cos = nn.CosineSimilarity(dim=-1)
        sim = 0
        sim_null = 0
        for i, (dir, dir_null) in enumerate(zip(directions, directions_null)):
            sim += cos(optimal_dir, dir)*dt[i]
            sim_null += cos(optimal_dir, dir_null)*dt[i]

        return sim, sim_null
  
    ##  Jacobian replace 3, distance end-point
    def compute_anchor_dist(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods, it does not matter 
        assert self.odeint_kwargs["method"] == "euler_mdd" or self.odeint_kwargs["method"] == "euler"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                #x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                #text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                text_null = torch.ones(mel_target.shape[0], mel_target.shape[1], dtype=torch.long, device=device)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
        t_f = torch.linspace(0, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        
  
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            t_f = t_f + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_f) - 1 + t_f)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass
        self.odeint_kwargs["method"] = "euler"
        solutions = odeint(fn, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled_y0 = solutions[-1]
        ##null forward pass
        solutions_null = odeint(fn_null, sampled_y0, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        null_points = solutions_null[-1] 
        ##similarity based on anchor point
        y0_anchor = torch.zeros_like(y1)
        solutions_anchor = odeint(fn, y0_anchor, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        anchor_points = solutions_anchor[-1]
        
        ###cos-similarity, already normalized, otherwise we could use L2(normalized with the length of y1 or dist(y1,anchor))
        cos = nn.CosineSimilarity(dim=-1) 
        dist = cos(anchor_points, y1)
        dist_null = cos(anchor_points, null_points)

        return dist, dist_null
    
    ##  Jacobian replace 4, distance l2
    def compute_l2_dist(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods, it does not matter 
        assert self.odeint_kwargs["method"] == "euler_mdd" or self.odeint_kwargs["method"] == "euler"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                #x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                #text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                text_null = torch.ones(mel_target.shape[0], mel_target.shape[1], dtype=torch.long, device=device)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
        t_f = torch.linspace(0, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        
  
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            t_f = t_f + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_f) - 1 + t_f)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass
        self.odeint_kwargs["method"] = "euler"
        solutions = odeint(fn, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled_y0 = solutions[-1]

        ##null forward pass
        solutions_null = odeint(fn_null, sampled_y0, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        null_points = solutions_null[-1] 
        ##similarity based on anchor point
        y0_anchor = torch.zeros_like(y1)
        solutions_anchor = odeint(fn, y0_anchor, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        anchor_points = solutions_anchor[-1]
        
        ###l2-distance, normalized using the anchor distance
        anc_len = torch.norm(anchor_points, p=2, dim=-1)
        dist = anc_len/torch.norm(y1-anchor_points, p=2, dim=-1)
        dist_null = anc_len/torch.norm(null_points-anchor_points, p=2, dim=-1)

        return dist, dist_null
    
    ##  Jacobian replace 5, noJac+l2-distance fix
    def compute_noJac_l2_fix(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods, it does not matter 
        assert self.odeint_kwargs["method"] == "euler_mdd" or self.odeint_kwargs["method"] == "euler"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                #x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                #text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                text_null = torch.ones(mel_target.shape[0], mel_target.shape[1], dtype=torch.long, device=device)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
        t_f = torch.linspace(0, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        
  
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            t_f = t_f + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_f) - 1 + t_f)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass
        self.odeint_kwargs["method"] = "euler"
        solutions = odeint(fn, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled_y0 = solutions[-1]
        ##null forward pass
        solutions_null = odeint(fn_null, sampled_y0, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        null_points = solutions_null[-1] 
        ##similarity based on anchor point
        y0_anchor = torch.zeros_like(y1)
        solutions_anchor = odeint(fn, y0_anchor, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        anchor_points = solutions_anchor[-1]
        
        ##start point
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = norm.log_prob(sampled_y0).sum(-1)
             
        ###l2-distance fixing, normalized with anc_len
        anc_len = torch.norm(anchor_points, p=2, dim=-1)
        dist = torch.norm(y1-anchor_points, p=2, dim=-1)/anc_len
        dist_null = torch.norm(null_points-anchor_points, p=2, dim=-1)/anc_len
        return log_prob_y0*dist, log_prob_y0*dist_null
    
    ##  Jacobian replace 6, noJac+cos-distance fix
    def compute_noJac_cos_fix(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods, it does not matter 
        assert self.odeint_kwargs["method"] == "euler_mdd" or self.odeint_kwargs["method"] == "euler"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                #x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                #text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                text_null = torch.ones(mel_target.shape[0], mel_target.shape[1], dtype=torch.long, device=device)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
        t_f = torch.linspace(0, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
  
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            t_f = t_f + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_f) - 1 + t_f)
        
        ##backward pass
        self.odeint_kwargs["method"] = "euler"
        solutions = odeint(fn, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled_y0 = solutions[-1]
        ##null forward pass
        solutions_null = odeint(fn_null, sampled_y0, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        null_points = solutions_null[-1] 
        ##similarity based on anchor point, forward
        y0_anchor = torch.zeros_like(y1)
        solutions_anchor = odeint(fn, y0_anchor, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        anchor_points = solutions_anchor[-1]
        
        ##start point
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device)) 
        #log_prob_y0 = norm.log_prob(sampled_y0).sum(-1)
        log_prob_y0 = norm.log_prob(sampled_y0).mean(-1)   ##nomalized by the dims, should we do it?  
        
        ###cos-distance fixing
        cos = nn.CosineSimilarity(dim=-1) 
        dist = cos(anchor_points, y1) + 1
        dist_null = cos(anchor_points, null_points) + 1
        assert (dist <= 0).sum() == 0 
        assert (dist_null <= 0).sum() == 0
        
        return log_prob_y0+dist.log(), log_prob_y0+dist_null.log()
    
    ##  Jacobian replace 7, noJac+OT-path+start-point fix
    def compute_noJac_OT_start_fix(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods, it does not matter 
        assert self.odeint_kwargs["method"] == "euler_mdd" or self.odeint_kwargs["method"] == "euler"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                #x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                #text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                text_null = torch.ones(mel_target.shape[0], mel_target.shape[1], dtype=torch.long, device=device)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
        t_f = torch.linspace(0, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
  
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            t_f = t_f + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_f) - 1 + t_f)
        
        self.odeint_kwargs["method"] = "euler"
        solutions = odeint(fn, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled_y0 = solutions[-1]
        directions = ((solutions - sampled_y0).flip([0]))[1:]
        ##NULL,different from r2, here we go back from y1 using fn_null
        solutions_null = odeint(fn_null, y1, t, **self.odeint_kwargs)
        y0_null = solutions_null[-1]
        self.transformer.clear_cache()
        ##[solutions_null - y0_null] or [solutions_null - sampled_y0]? the latter is relative so I think the first makes more sense
        directions_null = ((solutions_null - y0_null).flip([0]))[1:]
        ###start-prob-log
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        ### Nomalization mean
        # log_prob_y0 = norm.log_prob(sampled_y0).mean(-1)
        # log_prob_y0_null = norm.log_prob(y0_null).mean(-1)
        ### Nomalization mean X step_size
        # log_prob_y0 = norm.log_prob(sampled_y0).mean(-1)*(steps)
        # log_prob_y0_null = norm.log_prob(y0_null).mean(-1)*(steps)
        ## NoStart
        log_prob_y0 = 0
        log_prob_y0_null = 0
        ##similarity based log-prob, t-accumulated cos-similarity diffence
        optimal_dir = y1 - sampled_y0
        optimal_dir_null = y1 - y0_null
        cos = nn.CosineSimilarity(dim=-1)
        sim = log_prob_y0
        sim_null = log_prob_y0_null
        for i, (dir, dir_null) in enumerate(zip(directions, directions_null)):
            sim += cos(optimal_dir, dir)*dt[i]
            sim_null += cos(optimal_dir_null, dir_null)*dt[i]

        return sim, sim_null
    
    ## Jacobian replace 8, recon cos-similarity
    def compute_recon_cos(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
        remove_first_t_back=False,
    ):
        self.eval()
        # check methods, it does not matter 
        assert self.odeint_kwargs["method"] == "euler_mdd" or self.odeint_kwargs["method"] == "euler"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
                #x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                #text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                text_null = torch.ones(mel_target.shape[0], mel_target.shape[1], dtype=torch.long, device=device)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        t_f = t.flip(0)
        
        if remove_first_t_back == True:
            t = t[1:]  
        ##reversed dt for forward ODE
        #dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
    
        self.odeint_kwargs["method"] = "euler"
        
        solutions = odeint(fn, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled_y0 = solutions[-1]
        solutions_forward = odeint(fn, sampled_y0, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        recon = solutions_forward[-1]
        ##NULL
        solutions_null = odeint(fn_null, y1, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled_y0_null = solutions_null[-1]
        solutions_forward_null = odeint(fn_null, sampled_y0_null, t_f, **self.odeint_kwargs)
        self.transformer.clear_cache()
        recon_null = solutions_forward_null[-1]
        #compute the cos-sim
        cos = nn.CosineSimilarity(dim=-1)
        sim = cos(y1, recon)
        sim_null = cos(y1, recon_null)
        return sim, sim_null
    
    def compute_prob_wrong(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods
        assert self.odeint_kwargs["method"] == "euler_mdd"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)    
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred 

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass
        trajectory, jacob_trace = odeint_jacobian_wrong(fn, y1, t, cond_mask.squeeze(-1), cfg_strength, **self.odeint_kwargs)
        y0 = trajectory[-1]
        ##inverse jacob for forward pass
        jacob_trace = jacob_trace.flip([0])
        self.transformer.clear_cache()
        ##NULL
        trajectory_null, jacob_trace_null = odeint_jacobian_wrong(fn_null, y1, t, cond_mask.squeeze(-1), cfg_strength, **self.odeint_kwargs)
        y0_null = trajectory_null[-1]
        jacob_trace_null = jacob_trace_null.flip([0])
        self.transformer.clear_cache()
        
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = norm.log_prob(y0).sum(-1)
        log_prob_y0_null = norm.log_prob(y0_null).sum(-1)
        ## manual Euler continous change of variable
        ## the last jacob is not needed for forward pass
        for i, (jacob_t, jacob_t_null) in enumerate(zip(jacob_trace[:-1], jacob_trace_null[:-1])):
            log_prob_y0 += -jacob_t*dt[i]
            log_prob_y0_null += -jacob_t_null*dt[i]
        return log_prob_y0, log_prob_y0_null
    
    def compute_prob_noJac(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods, it does not matter 
        assert self.odeint_kwargs["method"] == "euler_mdd" or self.odeint_kwargs["method"] == "euler"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)        
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        
        ##test
        # step_cond=step_cond[:10]
        # text=text[:10]
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
                #x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)
  
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass
        
        # #test
        # y1=y1[:10]
        self.odeint_kwargs["method"] = "euler"
        trajectory = odeint(fn, y1, t, **self.odeint_kwargs)
        y0 = trajectory[-1]
        self.transformer.clear_cache()
        ##NULL
        trajectory_null = odeint(fn_null, y1, t, **self.odeint_kwargs)
        y0_null = trajectory_null[-1]
        self.transformer.clear_cache()
        
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = norm.log_prob(y0).sum(-1)
        log_prob_y0_null = norm.log_prob(y0_null).sum(-1)

        return log_prob_y0, log_prob_y0_null
  
    def compute_prob_aabb(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods
        assert self.odeint_kwargs["method"] == "euler_mdd"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)    
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            # predict flow
            # nonlocal step_cond
            # nonlocal text
            # if step_cond.shape[0] != x.shape[0]:
            #     dup = x.shape[0] / step_cond.shape[0]
            #     step_cond = step_cond.repeat(int(dup),1,1)
            # if text.shape[0] != x.shape[0]:
            #     dup = x.shape[0] / text.shape[0]
            #     text = text.repeat(int(dup),1)
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred 

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass 
        trajectory, jacob_trace = odeint_jacobian_aabb(fn, y1, t, **self.odeint_kwargs)
        y0 = trajectory[-1]
        ##inverse jacob for forward pass
        jacob_trace = jacob_trace.flip([0])
        self.transformer.clear_cache()
        ##NULL
        trajectory_null, jacob_trace_null = odeint_jacobian_aabb(fn_null, y1, t, **self.odeint_kwargs)
        y0_null = trajectory_null[-1]
        jacob_trace_null = jacob_trace_null.flip([0])
        self.transformer.clear_cache()
        
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = norm.log_prob(y0).sum(-1)
        log_prob_y0_null = norm.log_prob(y0_null).sum(-1)
        ## manual Euler continous change of variable
        ## the last jacob is not needed for forward pass
        for i, (jacob_t, jacob_t_null) in enumerate(zip(jacob_trace, jacob_trace_null)):
            log_prob_y0 += -jacob_t*dt[i]
            log_prob_y0_null += -jacob_t_null*dt[i]
        return log_prob_y0, log_prob_y0_null
    
    def compute_prob_aabb_fixed(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
        t_interval=None,
    ):
        self.eval()
        # check methods
        assert self.odeint_kwargs["method"] == "euler_mdd"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)    
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            # predict flow
            # nonlocal step_cond
            # nonlocal text
            # if step_cond.shape[0] != x.shape[0]:
            #     dup = x.shape[0] / step_cond.shape[0]
            #     step_cond = step_cond.repeat(int(dup),1,1)
            # if text.shape[0] != x.shape[0]:
            #     dup = x.shape[0] / text.shape[0]
            #     text = text.repeat(int(dup),1)
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred 

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        
        ##reversed dt for forward ODE
        dt = [ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)]
        
        ##backward pass 
        trajectory, jacob_trace = odeint_jacobian_aabb_fix(fn, y1, t, t_interval, **self.odeint_kwargs)
        y0 = trajectory[-1]
        ##inverse jacob for forward pass
        jacob_trace = jacob_trace.flip([0])
        self.transformer.clear_cache()
        ##NULL
        trajectory_null, jacob_trace_null = odeint_jacobian_aabb_fix(fn_null, y1, t, t_interval, **self.odeint_kwargs)
        y0_null = trajectory_null[-1]
        jacob_trace_null = jacob_trace_null.flip([0])
        self.transformer.clear_cache()
        
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = norm.log_prob(y0).sum(-1)
        log_prob_y0_null = norm.log_prob(y0_null).sum(-1)
        ## manual Euler continous change of variable
        ## the last jacob is not needed for forward pass
        ## do we need to negate?
        for i, (jacob_t, jacob_t_null) in enumerate(zip(jacob_trace, jacob_trace_null)):
            log_prob_y0 -= jacob_t*dt[i]
            log_prob_y0_null -= jacob_t_null*dt[i]
        return log_prob_y0, log_prob_y0_null
    
    def compute_prob_hut(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
        n_samples=None,
    ):
        self.eval()
        # check methods
        assert self.odeint_kwargs["method"] == "euler_mdd"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)    
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        cond_mask = cond_mask.squeeze(-1)
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            # predict flow
            # nonlocal step_cond
            # nonlocal text
            # if step_cond.shape[0] != x.shape[0]:
            #     dup = x.shape[0] / step_cond.shape[0]
            #     step_cond = step_cond.repeat(int(dup),1,1)
            # if text.shape[0] != x.shape[0]:
            #     dup = x.shape[0] / text.shape[0]
            #     text = text.repeat(int(dup),1)
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred 

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            # def fn_null(t, x):
            #     null_pred = self.transformer(
            #         x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            #     )
            #     if cfg_strength < 1e-5:
            #         return null_pred 

            #     null_null_pred = self.transformer(
            #         x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            #     )
            #     return null_pred + (null_pred - null_null_pred) * cfg_strength
            
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred 
        else: 
            #sys.exit("not supported in current version")
            def fn_null(t, x):
                text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        ##backward pass 
        trajectory, trace_phoneme = odeint_jacobian_hut(fn, y1, t, cond_mask, n_samples, **self.odeint_kwargs)
        y0 = trajectory[-1]
        ##inverse jacob for forward pass
        trace_phoneme = trace_phoneme.flip([0])
        self.transformer.clear_cache()
        ##NULL
        trajectory_null, trace_phoneme_null = odeint_jacobian_hut(fn_null, y1, t, cond_mask, n_samples, **self.odeint_kwargs)
        y0_null = trajectory_null[-1]
        trace_phoneme_null = trace_phoneme_null.flip([0])
        self.transformer.clear_cache()
        
        ##compute gop
        ##reversed dt for forward ODE
        dt = torch.tensor([ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)], device=device)
        frame_num = torch.tensor( [ (~cond_mask[i]).sum() for i in range(cond_mask.shape[0])],device=device, dtype=torch.long)
        ##start point
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = (norm.log_prob(y0).sum(-1))
        log_prob_y0_null = (norm.log_prob(y0_null).sum(-1))
        ###get phoneme-level probs
        gop = torch.tensor([log_prob_y0[i, ~(cond_mask[i])].sum(-1) for i in range(batch_size)], device=device )/frame_num
        gop_null = torch.tensor([log_prob_y0_null[i, ~(cond_mask[i])].sum(-1) for i in range(batch_size)], device=device )/frame_num
        ## manual Euler continous change of variable, hut approximation at phoneme-level
        gop_fix = (trace_phoneme*dt[:,None]).sum(dim=0)/frame_num
        gop_null_fix = (trace_phoneme_null*dt[:,None]).sum(dim=0)/frame_num
        pdb.set_trace()
        gop -= gop_fix
        gop_null -= gop_null_fix
        # for i, (jacob_t, jacob_t_null) in enumerate(zip(trace_phoneme, trace_phoneme_null)):
        #     gop -= jacob_t*dt[i]
        #     gop_null -= jacob_t_null*dt[i]
        ##take the average over frames
        #frame_num = torch.tensor( [ (~cond_mask[i]).sum() for i in range(cond_mask.shape[0])],device=device, dtype=torch.long)
        #return gop/frame_num, gop_null/frame_num
        return gop, gop_null
    
    def compute_prob_hut_start(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list_orig=None,
        n_samples=None,
    ):
        self.eval()
        # check methods
        assert self.odeint_kwargs["method"] == "euler_mdd"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)   
        mel_target = mel_target.to(next(self.parameters()).dtype) 
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        step_cond = torch.where(
            cond_mask[...,None], mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        cond_mask = cond_mask
        ##cond_mask for computing the gop
        cond_mask_orig = torch.stack(phoneme_mask_list_orig)
        # neural ode
        def fn(t, x):
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred 

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
    
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        ##backward pass 
        trajectory, trace_phoneme = odeint_jacobian_hut(fn, y1, t, cond_mask_orig, n_samples, **self.odeint_kwargs)
        y0 = trajectory[-1]
        ##inverse jacob for forward pass
        trace_phoneme = trace_phoneme.flip([0])
        self.transformer.clear_cache()

        ##compute gop
        ##reversed dt for forward ODE
        dt = torch.tensor([ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)], device=device)
        frame_num = torch.tensor( [ (~cond_mask[i]).sum() for i in range(cond_mask.shape[0])],device=device, dtype=torch.long)
        ##start point
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = (norm.log_prob(y0).sum(-1))/100
        ###get phoneme-level probs
        gop = torch.tensor([log_prob_y0[i, ~(cond_mask[i])].sum(-1) for i in range(batch_size)], device=device )/frame_num
        ## manual Euler continous change of variable, hut approximation at phoneme-level
        gop_fix = (trace_phoneme*dt[:,None]).sum(dim=0)/(100*frame_num)
        gop -= gop_fix     
   
        return gop, y0
    
    def compute_prob_hut_recon(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list_orig=None,
        n_samples=None,
    ):
        self.eval()
        # check methods
        assert self.odeint_kwargs["method"] == "euler_mdd"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)   
        mel_target = mel_target.to(next(self.parameters()).dtype) 
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        # ##mask the text
        text=torch.tensor([[1]], device=device)
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##cond_mask for computing the gop
        cond_mask = torch.stack(phoneme_mask_list_orig)
        
        ##MDD, we don't pad anything than masking the target segment)
        step_cond = torch.where(
            cond_mask[...,None], mel_target, torch.zeros_like(mel_target)
        )  

      
        # neural ode
        def fn(t, x):
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=None, drop_audio_cond=False, drop_text=True, cache=True
            )
            if cfg_strength < 1e-5:
                return pred 

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=None, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
    
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        t_forward = t.flip(0)
        ## remove the first (t=1)
        #t = t[1:]
        ##backward pass 
        trajectory, trace_phoneme = odeint_jacobian_hut(fn, y1, t, cond_mask, n_samples, **self.odeint_kwargs)
        y0 = trajectory[-1]
        ##inverse jacob for forward pass
        trace_phoneme = trace_phoneme.flip([0])
        self.transformer.clear_cache()

        ##back 2
        # self.odeint_kwargs["method"]="euler"
        # trajectory_2 = odeint(fn, y1, t, **self.odeint_kwargs)
        # self.transformer.clear_cache()
        # y0_2 = trajectory_2[-1]
        # pdb.set_trace()
        # y0 = y0_2
        cos = nn.CosineSimilarity(dim=-1)
        ##compute gop
        ##reversed dt for forward ODE
        dt = torch.tensor([ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)], device=device)
        frame_num = torch.tensor( [ (~cond_mask[i]).sum() for i in range(cond_mask.shape[0])],device=device, dtype=torch.long)
        ##start point
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = (norm.log_prob(y0).sum(-1))/100
        ###get phoneme-level probs
        gop = torch.tensor([log_prob_y0[i, ~(cond_mask[i])].sum(-1) for i in range(batch_size)], device=device )/frame_num
        ## manual Euler continous change of variable, hut approximation at phoneme-level
        gop_fix = (trace_phoneme*dt[:,None]).sum(dim=0)/(100*frame_num)
        gop -= gop_fix
        print(f"GOP-1st:{gop}")

        
        self.odeint_kwargs["method"]="euler"
        trajectory = odeint(fn, y0, t.flip(0), **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled = trajectory[-1]

        ## back again
        self.odeint_kwargs["method"] = "euler_mdd"  
        trajectory, trace_phoneme = odeint_jacobian_hut(fn, sampled, t, cond_mask, n_samples, **self.odeint_kwargs)
        y0_final = trajectory[-1]
        self.transformer.clear_cache()
        
        log_prob_y0 = (norm.log_prob(y0_final).sum(-1))/100
        ###get phoneme-level probs
        gop = torch.tensor([log_prob_y0[i, ~(cond_mask[i])].sum(-1) for i in range(batch_size)], device=device )/frame_num
        ## manual Euler continous change of variable, hut approximation at phoneme-level
        gop_fix = (trace_phoneme*dt[:,None]).sum(dim=0)/(100*frame_num)
        gop -= gop_fix
        print(f"GOP-2ed:{gop}")
        
        #pdb.set_trace()
        self.odeint_kwargs["method"]="euler"
        trajectory = odeint(fn, y0_final, t.flip(0), **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled2 = trajectory[-1]
        
        ## back again
        self.odeint_kwargs["method"] = "euler_mdd"  
        trajectory, trace_phoneme = odeint_jacobian_hut(fn, sampled2, t, cond_mask, n_samples, **self.odeint_kwargs)
        y0_final2 = trajectory[-1]
        self.transformer.clear_cache()
        
        log_prob_y0 = (norm.log_prob(y0_final2).sum(-1))/100
        ###get phoneme-level probs
        gop = torch.tensor([log_prob_y0[i, ~(cond_mask[i])].sum(-1) for i in range(batch_size)], device=device )/frame_num
        ## manual Euler continous change of variable, hut approximation at phoneme-level
        gop_fix = (trace_phoneme*dt[:,None]).sum(dim=0)/(100*frame_num)
        gop -= gop_fix
        print(f"GOP-3rd:{gop}")
        
        #pdb.set_trace()
        self.odeint_kwargs["method"]="euler"
        trajectory = odeint(fn, y0_final2, t.flip(0), **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled3 = trajectory[-1]
        
        pdb.set_trace()

        #return gop, y0, y0_final, y0_final2 
        return gop, sampled, sampled2, sampled3
    
    def compute_prob_hut_mask(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
        phoneme_mask_list_orig=None,
        n_samples=None,
    ):
        self.eval()
        # check methods
        assert self.odeint_kwargs["method"] == "euler_mdd"
        # raw wave
        if mel_target[0].ndim != 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        ## To tensors
        mel_target = torch.stack(mel_target)   
        mel_target = mel_target.to(next(self.parameters()).dtype) 
        batch_size, seq_len, device = *mel_target.shape[:2], mel_target.device
        
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                sys.exit("MDD must have a vocab file")
            assert text.shape[0] == batch_size

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        else:
            sys.exit("MDD: duariont must be a int")
        
        max_t_len = torch.max((text != -1).sum(dim=-1))  # MDD: duration at least text

        ##MDD check
        assert torch.all(duration == duration[0])
        assert seq_len == duration[0]
        assert lens is None
        assert duration[0] <= max_duration
        assert duration[0] >= max_t_len         
        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == seq_len
   
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            #test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            sys.exit("MDD: not supported.")
            
        ##MDD, we don't pad anything than masking the target segment)
        cond_mask = torch.stack(phoneme_mask_list).unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, mel_target, torch.zeros_like(mel_target)
        )  
        att_mask = None
        cond_mask = cond_mask.squeeze(-1)
        ##cond_mask for computing the gop
        cond_mask_orig = torch.stack(phoneme_mask_list_orig)
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            # predict flow
            # nonlocal step_cond
            # nonlocal text
            # if step_cond.shape[0] != x.shape[0]:
            #     dup = x.shape[0] / step_cond.shape[0]
            #     step_cond = step_cond.repeat(int(dup),1,1)
            # if text.shape[0] != x.shape[0]:
            #     dup = x.shape[0] / text.shape[0]
            #     text = text.repeat(int(dup),1)
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
            )
            if cfg_strength < 1e-5:
                return pred 

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            )
            return pred + (pred - null_pred) * cfg_strength
        
        if diff_symbol == None:  ## all ZERO cases, check class TextEmbedding
            # def fn_null(t, x):
            #     null_pred = self.transformer(
            #         x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
            #     )
            #     if cfg_strength < 1e-5:
            #         return null_pred 

            #     null_null_pred = self.transformer(
            #         x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
            #     )
            #     return null_pred + (null_pred - null_null_pred) * cfg_strength
            
            def fn_null(t, x):
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=False, drop_text=True, cache=True
                )
                return null_pred 
        else: 
            #sys.exit("not supported in current version")
            def fn_null(t, x):
                text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=False, drop_text=False, cache=True
                )
                return null_pred
        # speech input
        y1 = mel_target
        t_start = 1
        t = torch.linspace(t_start, 0, steps + 1, device=self.device, dtype=step_cond.dtype)

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        ##remove some t
        t = t[1:]
        ##backward pass 
        trajectory, trace_phoneme = odeint_jacobian_hut(fn, y1, t, cond_mask_orig, n_samples, **self.odeint_kwargs)
        y0 = trajectory[-1]
        ##inverse jacob for forward pass
        trace_phoneme = trace_phoneme.flip([0])
        self.transformer.clear_cache()
        ##NULL
        trajectory_null, trace_phoneme_null = odeint_jacobian_hut(fn_null, y1, t, cond_mask_orig, n_samples, **self.odeint_kwargs)
        y0_null = trajectory_null[-1]
        trace_phoneme_null = trace_phoneme_null.flip([0])
        self.transformer.clear_cache()
        
        ##compute gop
        ##reversed dt for forward ODE
        dt = torch.tensor([ t[len(t)-i-2] - t[len(t)-i-1] for i in range(len(t)-1)], device=device)
        frame_num = torch.tensor( [ (~cond_mask[i]).sum() for i in range(cond_mask.shape[0])],device=device, dtype=torch.long)
        ##start point
        norm = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0],device=device))
        log_prob_y0 = (norm.log_prob(y0).sum(-1))
        log_prob_y0_null = (norm.log_prob(y0_null).sum(-1))
        ###get phoneme-level probs
        gop = torch.tensor([log_prob_y0[i, ~(cond_mask[i])].sum(-1) for i in range(batch_size)], device=device )/frame_num
        gop_null = torch.tensor([log_prob_y0_null[i, ~(cond_mask[i])].sum(-1) for i in range(batch_size)], device=device )/frame_num
        ## manual Euler continous change of variable, hut approximation at phoneme-level
        gop_fix = (trace_phoneme*dt[:,None]).sum(dim=0)/frame_num
        gop_null_fix = (trace_phoneme_null*dt[:,None]).sum(dim=0)/frame_num
        gop -= gop_fix
        gop_null -= gop_null_fix
        # for i, (jacob_t, jacob_t_null) in enumerate(zip(trace_phoneme, trace_phoneme_null)):
        #     gop -= jacob_t*dt[i]
        #     gop_null -= jacob_t_null*dt[i]
        ##take the average over frames
        #frame_num = torch.tensor( [ (~cond_mask[i]).sum() for i in range(cond_mask.shape[0])],device=device, dtype=torch.long)
        #return gop/frame_num, gop_null/frame_num
        return gop, gop_null
    
    
    # def forward(
    #     self,
    #     inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
    #     text: int["b nt"] | list[str],  # noqa: F722
    #     *,
    #     lens: int["b"] | None = None,  # noqa: F821
    #     noise_scheduler: str | None = None,
    # ):
    #     # handle raw wave
    #     if inp.ndim == 2:
    #         inp = self.mel_spec(inp)
    #         inp = inp.permute(0, 2, 1)
    #         assert inp.shape[-1] == self.num_channels

    #     batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

    #     # handle text as string
    #     if isinstance(text, list):
    #         if exists(self.vocab_char_map):
    #             text = list_str_to_idx(text, self.vocab_char_map).to(device)
    #         else:
    #             text = list_str_to_tensor(text).to(device)
    #         assert text.shape[0] == batch

    #     # lens and mask
    #     if not exists(lens):
    #         lens = torch.full((batch,), seq_len, device=device)

    #     mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

    #     # get a random span to mask out for training conditionally
    #     frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
    #     rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

    #     if exists(mask):
    #         rand_span_mask &= mask

    #     # mel is x1
    #     x1 = inp

    #     # x0 is gaussian noise
    #     x0 = torch.randn_like(x1)

    #     # time step
    #     time = torch.rand((batch,), dtype=dtype, device=self.device)
    #     # TODO. noise_scheduler

    #     # sample xt (φ_t(x) in the paper)
    #     t = time.unsqueeze(-1).unsqueeze(-1)
    #     φ = (1 - t) * x0 + t * x1
    #     flow = x1 - x0

    #     # only predict what is within the random mask span for infilling
    #     cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

    #     # transformer and cfg training with a drop rate
    #     drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
    #     if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
    #         drop_audio_cond = True
    #         drop_text = True
    #     else:
    #         drop_text = False

    #     # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
    #     # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
    #     pred = self.transformer(
    #         x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
    #     )

    #     # flow matching loss
    #     loss = F.mse_loss(pred, flow, reduction="none")
    #     loss = loss[rand_span_mask]

    #     return loss.mean(), cond, pred
