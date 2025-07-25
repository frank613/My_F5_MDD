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
from torchdiffeq import odeint,odeint_jacobian

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
            method="euler"  # 'midpoint'
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
        
        self.norm = m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

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
        pdb.set_trace()
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
        assert cond_seq_len == duration[0]
        assert lens is None
            
        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # cond_mask should be all 1 for MDD
        cond_mask = lens_to_mask(duration)
        assert cond_mask[0].sum() == cond_seq_len
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

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

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory, cond_orig

    def compute_prob(
        self,
        mel_target: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        max_duration=4096,
        duplicate_test=False,
        diff_symbol=None,
        phoneme_mask_list=None,
    ):
        self.eval()
        # check methods
        assert self.odeint_kwargs["method"] == "euler"
        self.odeint_kwargs["method"] = "euler_mdd"
        # raw wave
        if mel_target[0].ndim == 2:
            # cond = self.mel_spec(cond)
            # cond = cond.permute(0, 2, 1)
            # assert cond.shape[-1] == self.num_channels
            sys.exit("MDD: input must be mel")
        
        dtype = next(self.parameters()).dtype
        
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
        
        max_t_len = torch.maximum((text != -1).sum(dim=-1))  # MDD: duration at least text

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
                    x=x, cond=step_cond, text=text, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
                )
                return null_pred
        else: 
            def fn_null(t, x):
                text_null = torch.ones_like(text)*self.vocab_char_map[diff_symbol]
                null_pred = self.transformer(
                    x=x, cond=step_cond, text=text_null, time=t, mask=att_mask, drop_audio_cond=True, drop_text=True, cache=True
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
        pdb.set_trace()
        
        ##backward pass
        trajectory, jacob_trace = odeint_jacobian(fn, y1, t, **self.odeint_kwargs)
        y0 = trajectory[-1]
        ##inverse jacob for forward pass
        jacob_trace = jacob_trace.flip([0])
        
        self.transformer.clear_cache()
        
        trajectory_null, jacob_trace_null = odeint_jacobian(fn_null, y1, t, **self.odeint_kwargs)
        y0_null = trajectory_null[-1]
        ##inverse jacob for forward pass
        jacob_trace_null = jacob_trace_null.flip([0])
        
        self.transformer.clear_cache()
        
   
        log_prob_y0 = self.norm.log_prob(y0)
        log_prob_y0_null = self.norm.log_prob(y0_null)
        ## manual Euler continous change of variable
        ## the last jacob is not needed for forward pass
        for i, jacob_t, jacob_t_null in enumerate(zip(jacob_trace[:-1], jacob_trace_null[-1])):
            log_prob_y0 += -jacob_t*dt[i]
            log_prob_y0_null += jacob_t_null*dt[i]

        return log_prob_y0, log_prob_y0_null
    
    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
