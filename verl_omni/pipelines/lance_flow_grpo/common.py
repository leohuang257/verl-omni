# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for Lance FlowGRPO adapters.

Lance's inherited t2i serving path defaults to BAGEL's sampling values
(shift 3.0, 50 steps); the released Lance checkpoints use shift 3.5 and 30
steps (vllm-omni ``LanceDefaults``), so both adapters build their sigma
schedules through :func:`setup_lance_sigmas` and the rollout adapter injects
the Lance values explicitly.
"""

import torch

from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

# Lance sampling constants (vllm-omni pinned ``LanceDefaults``).
LANCE_TIMESTEP_SHIFT = 3.5
LANCE_NUM_TIMESTEPS = 30

# CFG defaults from the pinned Lance t2i serving path (``LanceDefaults`` +
# ``LancePipeline.forward``); rollout and training must use the same values.
LANCE_FLOWGRPO_CFG_DEFAULTS = {
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 1.0,
    "cfg_interval": (0.4, 1.0),
    "cfg_renorm_type": "global",
    "cfg_renorm_min": 0.0,
}


def maybe_to_cpu(value):
    """Move a single value to CPU if it is a ``torch.Tensor``; else return unchanged."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def lance_time_shift(shift: float, t):
    """SD3-style time shift: ``shift * t / (1 + (shift - 1) * t)``.

    Works with both ``torch.Tensor`` and ``numpy.ndarray``.
    """
    return (shift * t) / (1 + (shift - 1) * t)


def vllm_omni_num_timesteps(lance_num_timesteps: int) -> int:
    """Map official Lance step count to vllm-omni 0.22 generate_image input."""
    return lance_num_timesteps - 1 if lance_num_timesteps > 1 else lance_num_timesteps


def setup_lance_sigmas(
    scheduler: FlowMatchSDEDiscreteScheduler,
    num_steps: int,
    shift: float = LANCE_TIMESTEP_SHIFT,
    device: str | None = None,
) -> list[float]:
    """Compute shifted sigmas and configure the scheduler for Lance.

    Used verbatim by both the rollout and training adapters so schedule
    parity is structural.  Returns the sigma list (terminal 0 dropped).
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    # Warmup may pass one step; keep one non-terminal sigma for dummy runs.
    schedule_points = max(num_steps, 2)
    t = torch.linspace(1, 0, schedule_points, dtype=torch.float32, device=device or "cpu")
    t_shifted = lance_time_shift(shift, t)
    sigmas = t_shifted[:-1].tolist()

    scheduler.set_shift(1.0)  # identity — sigmas already shifted
    if device is not None:
        scheduler.set_timesteps(sigmas=sigmas, timesteps=sigmas, device=device)
    else:
        scheduler.set_timesteps(sigmas=sigmas, timesteps=sigmas)
    scheduler.set_begin_index(0)
    return sigmas
