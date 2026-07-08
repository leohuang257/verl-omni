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

"""Lance (MoT) training-side adapter for FlowGRPO.

Registered as ``OmniLanceForConditionalGeneration`` in the DiffusionModelBase
registry.  Like BAGEL, Lance takes raw token IDs (instead of prompt_embeds)
and applies 3-branch CFG with renormalization matching the rollout pipeline;
unlike BAGEL, positions are Qwen2.5-VL multimodal (t, h, w) triples and the
sigma schedule uses Lance's timestep shift 3.5.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import LANCE_FLOWGRPO_CFG_DEFAULTS, LANCE_TIMESTEP_SHIFT, setup_lance_sigmas
from .lance_model import LanceForTraining, get_flattened_position_ids

logger = logging.getLogger(__name__)

# micro-batch key holding step-independent model inputs (see prepare_model_inputs)
_STEP_INDEPENDENT_CACHE_KEY = "_lance_step_independent_inputs"


@DiffusionModelBase.register("OmniLanceForConditionalGeneration", algorithm="flow_grpo")
class LanceDiffusion(DiffusionModelBase):
    """DiffusionModelBase wrapper for ``LanceForTraining`` (MoT)."""

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        logger.info("Loading LanceForTraining from %s", model_config.local_path)
        return LanceForTraining.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)

    @classmethod
    def configure_train_mode(cls, module):
        """Match Lance/BAGEL MoT train-mode flags while gradients stay enabled."""
        inner = module.module if hasattr(module, "module") else module
        if not hasattr(inner, "layers"):
            return
        inner.training = False
        for layer in inner.layers:
            layer_inner = layer.module if hasattr(layer, "module") else layer
            layer_inner.training = False
            if hasattr(layer_inner, "self_attn"):
                layer_inner.self_attn.training = False

    @classmethod
    def configure_trainable_params(cls, module, model_config):
        """Freeze all params except the generation (``moe_gen``) pathway."""
        for name, param in module.named_parameters():
            param.requires_grad = "moe_gen" in name

        # cast all trainable parameters to fp32
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        # Build on GPU so scheduler buffers are comparable with cuda timesteps in FSDP forward.
        scheduler = FlowMatchSDEDiscreteScheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        # Honor a timestep_shift override so the training schedule matches a
        # rollout-side override (set both sides together, as for the CFG
        # params)::
        #
        #     +actor_rollout_ref.rollout.pipeline.timestep_shift=3.0
        #     +actor_rollout_ref.model.pipeline.timestep_shift=3.0
        #
        # The rollout builds sigmas from ``extra_args["timestep_shift"]``; a
        # mismatched shift here would make the trainer's exact-equality
        # timestep lookup fail (or silently corrupt dt/sigma_prev).
        shift = float(getattr(model_config.pipeline, "timestep_shift", LANCE_TIMESTEP_SHIFT))
        setup_lance_sigmas(scheduler, model_config.pipeline.num_inference_steps, shift=shift, device=device)

    @classmethod
    def _get_latent_pos_ids(cls, model_config: DiffusionModelConfig, module, device) -> torch.Tensor:
        """Compute latent position IDs from model config / image dimensions."""
        config = module.config
        latent_ds = config.latent_patch_size * config.vae_downsample
        img_h = model_config.pipeline.height // latent_ds
        img_w = model_config.pipeline.width // latent_ds
        # Clamp to max_latent_size
        img_h = min(img_h, config.max_latent_size)
        img_w = min(img_w, config.max_latent_size)
        H_px = img_h * latent_ds
        W_px = img_w * latent_ds
        pos_ids = get_flattened_position_ids(H_px, W_px, latent_ds, config.max_latent_size)
        return pos_ids.to(device)

    @classmethod
    def _prompt_token_ids_to_batch(
        cls,
        micro_batch: TensorDict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad Lance-native ``prompt_token_ids`` from the data pipeline.

        Args:
            micro_batch: Batch containing pre-tokenized ``prompt_token_ids``,
                one sequence per sample, already in Lance format
                (``[<|im_start|>] caption [<|im_end|>]``).
            device: Target device for the padded tensors.

        Returns:
            ``(text_token_ids, text_attention_mask)`` with shape ``(B, max_len)``.
        """
        prompt_token_ids = micro_batch["prompt_token_ids"]
        if isinstance(prompt_token_ids, NonTensorStack):
            prompt_token_ids = [
                tu.unwrap_non_tensor_data(prompt_token_ids[i]) for i in range(micro_batch.batch_size[0])
            ]
        else:
            prompt_token_ids = tu.unwrap_non_tensor_data(prompt_token_ids)

        if isinstance(prompt_token_ids, torch.Tensor) and prompt_token_ids.ndim == 1:
            prompt_token_ids = [prompt_token_ids]

        B = len(prompt_token_ids)
        max_len = max(len(ids) for ids in prompt_token_ids)
        text_token_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        text_attention_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for i, ids in enumerate(prompt_token_ids):
            n = len(ids)
            text_token_ids[i, :n] = torch.as_tensor(ids, dtype=torch.long, device=device)
            text_attention_mask[i, :n] = True
        return text_token_ids, text_attention_mask

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, dict]:
        B = latents.shape[0]
        device = latents.device

        hidden_states = latents[:, step]
        timestep = timesteps[:, step]

        # Token padding and latent position IDs are step-independent; compute
        # them once per micro-batch instead of at every denoising step.
        cached = tu.get_non_tensor_data(micro_batch, _STEP_INDEPENDENT_CACHE_KEY, default=None)
        if cached is None or cached["text_token_ids"].device != device:
            text_token_ids, text_attention_mask = cls._prompt_token_ids_to_batch(micro_batch, device)
            latent_pos_ids = cls._get_latent_pos_ids(model_config, module, device)
            latent_pos_ids = latent_pos_ids.unsqueeze(0).expand(B, -1)
            cached = {
                "text_token_ids": text_token_ids,
                "text_attention_mask": text_attention_mask,
                "latent_pos_ids": latent_pos_ids,
            }
            tu.assign_non_tensor(micro_batch, **{_STEP_INDEPENDENT_CACHE_KEY: cached})
        text_token_ids = cached["text_token_ids"]
        text_attention_mask = cached["text_attention_mask"]
        latent_pos_ids = cached["latent_pos_ids"]

        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "text_token_ids": text_token_ids,
            "text_attention_mask": text_attention_mask,
            "latent_pos_ids": latent_pos_ids,
        }

        # For Lance, the text-unconditional pass uses text_token_ids=None
        # (rollout keeps the cfg_text KV cache empty for t2i).
        negative_model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "text_token_ids": None,
            "latent_pos_ids": latent_pos_ids,
        }

        return model_inputs, negative_model_inputs

    @staticmethod
    def _get_cfg_params(model_config: DiffusionModelConfig) -> dict:
        """Resolve CFG params, falling back to Lance serving defaults.

        Override via Hydra (set both rollout and model sides together)::

            +actor_rollout_ref.model.pipeline.cfg_text_scale=4.0
            +actor_rollout_ref.model.pipeline.cfg_img_scale=1.0

        Returns:
            Dict with ``cfg_text_scale``, ``cfg_img_scale``,
            ``cfg_renorm_type``, ``cfg_renorm_min``,
            ``cfg_interval_low``, ``cfg_interval_high``.
        """
        p = model_config.pipeline
        cfg_interval = getattr(p, "cfg_interval", LANCE_FLOWGRPO_CFG_DEFAULTS["cfg_interval"])
        if isinstance(cfg_interval, list | tuple) and len(cfg_interval) == 2:
            interval_low, interval_high = float(cfg_interval[0]), float(cfg_interval[1])
        else:
            interval_low, interval_high = 0.0, 1.0
        return {
            "cfg_text_scale": float(getattr(p, "cfg_text_scale", LANCE_FLOWGRPO_CFG_DEFAULTS["cfg_text_scale"])),
            "cfg_img_scale": float(getattr(p, "cfg_img_scale", LANCE_FLOWGRPO_CFG_DEFAULTS["cfg_img_scale"])),
            "cfg_renorm_type": str(getattr(p, "cfg_renorm_type", LANCE_FLOWGRPO_CFG_DEFAULTS["cfg_renorm_type"])),
            "cfg_renorm_min": float(getattr(p, "cfg_renorm_min", LANCE_FLOWGRPO_CFG_DEFAULTS["cfg_renorm_min"])),
            "cfg_interval_low": interval_low,
            "cfg_interval_high": interval_high,
        }

    @staticmethod
    def _combine_cfg(
        v_t: torch.Tensor,
        cfg_text_v_t: torch.Tensor,
        cfg_img_v_t: Optional[torch.Tensor],
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_renorm_type: str,
        cfg_renorm_min: float,
    ) -> torch.Tensor:
        """Byte-identical port of vllm-omni's ``Bagel._combine_cfg`` (shared by Lance).

        Applies 3-branch CFG with global/channel renormalization so the
        training velocity matches the rollout trajectory exactly.

        Args:
            v_t: Gen-branch velocity ``(B, L, D)``.
            cfg_text_v_t: Text-unconditional velocity.
            cfg_img_v_t: Image-unconditional velocity (or ``None``).
            cfg_text_scale: Text CFG scale (e.g. 4.0).
            cfg_img_scale: Image CFG scale (1.0 disables the branch).
            cfg_renorm_type: ``"global"`` or ``"channel"``.
            cfg_renorm_min: Minimum renorm clamp.

        Returns:
            CFG-combined velocity of shape ``(B, L, D)``.
        """
        if cfg_renorm_type == "text_channel":
            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
            norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
            scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            v_t_text = v_t_text_ * scale
            if cfg_img_scale > 1.0 and cfg_img_v_t is not None:
                return cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
            return v_t_text

        v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
        if cfg_img_scale > 1.0 and cfg_img_v_t is not None:
            v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
        else:
            v_t_ = v_t_text_

        if cfg_renorm_type == "global":
            # The rollout handles one image per request, so its "global"
            # renorm is global over latent tokens/channels for each sample.
            # Training is batched; keep samples independent instead of
            # mixing the whole micro-batch into one scalar norm.
            norm_dims = tuple(range(1, v_t.ndim))
            norm_v_t = torch.linalg.vector_norm(v_t, dim=norm_dims, keepdim=True)
            norm_v_t_ = torch.linalg.vector_norm(v_t_, dim=norm_dims, keepdim=True)
        elif cfg_renorm_type == "channel":
            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
            norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
        else:
            raise NotImplementedError(f"cfg_renorm_type={cfg_renorm_type!r} is not supported")

        scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        return v_t_ * scale

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        # Gen branch (text-conditional).
        noise_pred = module(**model_inputs)[0]

        # Apply CFG matching the rollout so the importance-sampling ratio is
        # unbiased. Rollout uses cfg_text_scale=4.0 + global renorm gated to
        # the Lance cfg_interval.
        cfg = cls._get_cfg_params(model_config)
        # Gate the CFG interval per sample: trajectories from different
        # rollout GPUs carry different SDE windows, so sigma at the same
        # sliced step index can differ across the micro-batch and samples may
        # straddle the interval boundary (Lance's cfg_interval_low is 0.4).
        sigma_now = timesteps[:, step]  # (B,)
        in_cfg_interval = (sigma_now > cfg["cfg_interval_low"]) & (sigma_now <= cfg["cfg_interval_high"])
        apply_cfg = cfg["cfg_text_scale"] > 1.0 and bool(in_cfg_interval.any())

        if apply_cfg:
            assert negative_model_inputs is not None, (
                "Lance CFG requires negative_model_inputs (text-unconditional branch)."
            )
            # cfg_text branch: text_token_ids=None -> empty text context.
            cfg_text_pred = module(**negative_model_inputs)[0]
            # For text2img, no input image was supplied to drop, so the
            # cfg_img branch is identical to the gen branch and we can
            # reuse ``noise_pred`` instead of running a third forward.
            cfg_img_pred = noise_pred if cfg["cfg_img_scale"] > 1.0 else None

            combined = cls._combine_cfg(
                v_t=noise_pred,
                cfg_text_v_t=cfg_text_pred,
                cfg_img_v_t=cfg_img_pred,
                cfg_text_scale=cfg["cfg_text_scale"],
                cfg_img_scale=cfg["cfg_img_scale"],
                cfg_renorm_type=cfg["cfg_renorm_type"],
                cfg_renorm_min=cfg["cfg_renorm_min"],
            )
            if bool(in_cfg_interval.all()):
                noise_pred = combined
            else:
                # _combine_cfg renormalizes per sample, so selecting rows is
                # exact for the in-interval samples.
                mask = in_cfg_interval.view(-1, *([1] * (noise_pred.dim() - 1)))
                noise_pred = torch.where(mask, combined, noise_pred)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
            include_logprob_normalizer=False,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
