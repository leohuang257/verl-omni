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

"""LanceForTraining – FSDP-compatible Lance MoT module for flow-matching training.

Ported from vllm-omni's Lance integration (BAGEL's Qwen2-MoT core).  Lance
deltas vs the BAGEL training port: Qwen2.5-VL multimodal RoPE (per-token
``(t, h, w)`` positions, ``mrope_section=[16, 24, 24]``, latent layout
mirroring the serving position table — see ``_build_position_ids``), GQA
with q/k/v biases and QK-norm on both experts, Wan2.2 latent geometry
(48 channels, downsample 16, 64×64 max grid), and fail-closed checkpoint
loading (every source key must map, every parameter must be filled; only
the unused AR head ``language_model.lm_head.weight`` is dropped).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from verl_omni.pipelines.non_diffusers_model_base import NonDiffusersModelBase

# ===================================================================
#  Config
# ===================================================================

# Lance constants that upstream keeps in code rather than in any shipped
# JSON (verified against the released ``Lance_3B`` checkpoint and vllm-omni's
# pinned ``LanceDefaults``).
_LANCE_LATENT_PATCH_SIZE = 1
_LANCE_MAX_LATENT_SIZE = 64
_LANCE_LATENT_CHANNEL = 48
_LANCE_VAE_DOWNSAMPLE = 16

_IMAGE_CKPT_DIR = "Lance_3B"


@dataclass
class LanceTrainingConfig:
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 128000
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])
    # Lance-specific latent geometry (Wan2.2 VAE)
    latent_patch_size: int = _LANCE_LATENT_PATCH_SIZE
    max_latent_size: int = _LANCE_MAX_LATENT_SIZE
    latent_channel: int = _LANCE_LATENT_CHANNEL
    vae_downsample: int = _LANCE_VAE_DOWNSAMPLE
    # The sampling timestep shift is not a model property: the adapters read
    # it from ``common.LANCE_TIMESTEP_SHIFT`` / the pipeline config.
    start_of_image_id: int = 151652  # <|vision_start|>
    end_of_image_id: int = 151653  # <|vision_end|>

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} not divisible by num_attention_heads {self.num_attention_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} not divisible by "
                f"num_key_value_heads {self.num_key_value_heads}"
            )
        if 2 * sum(self.mrope_section) != self.head_dim:
            raise ValueError(
                f"mrope_section {self.mrope_section} must sum to head_dim/2 = {self.head_dim // 2} "
                f"(got {sum(self.mrope_section)})"
            )
        if self.latent_patch_size <= 0 or self.max_latent_size <= 0 or self.latent_channel <= 0:
            raise ValueError("latent geometry values must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_latent_dim(self) -> int:
        return self.latent_patch_size**2 * self.latent_channel

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(asdict(self), f, indent=4, sort_keys=True)

    @classmethod
    def from_model_path(cls, model_path: str) -> LanceTrainingConfig:
        """Parse config from a Lance checkpoint directory.

        Accepts either the bundled repo root (containing ``Lance_3B/``) or the
        checkpoint subdirectory itself (containing ``llm_config.json``).
        """
        ckpt_dir = resolve_lance_checkpoint_dir(model_path)
        cfg_path = os.path.join(ckpt_dir, "llm_config.json")
        with open(cfg_path) as f:
            llm = json.load(f)

        rope_scaling = llm.get("rope_scaling") or {}
        mrope_section = rope_scaling.get("mrope_section", [16, 24, 24])

        return cls(
            hidden_size=llm.get("hidden_size", 2048),
            intermediate_size=llm.get("intermediate_size", 11008),
            num_hidden_layers=llm.get("num_hidden_layers", 36),
            num_attention_heads=llm.get("num_attention_heads", 16),
            num_key_value_heads=llm.get("num_key_value_heads", 2),
            vocab_size=llm.get("vocab_size", 151936),
            rms_norm_eps=llm.get("rms_norm_eps", 1e-6),
            rope_theta=llm.get("rope_theta", 1_000_000.0),
            max_position_embeddings=llm.get("max_position_embeddings", 128000),
            mrope_section=list(mrope_section),
            start_of_image_id=llm.get("vision_start_token_id", 151652),
            end_of_image_id=llm.get("vision_end_token_id", 151653),
        )


def resolve_lance_checkpoint_dir(model_path: str) -> str:
    """Resolve the ``Lance_3B`` checkpoint directory from a user-supplied path."""
    model_path = os.path.expanduser(model_path)
    if os.path.isfile(os.path.join(model_path, "llm_config.json")):
        return model_path
    subdir = os.path.join(model_path, _IMAGE_CKPT_DIR)
    if os.path.isfile(os.path.join(subdir, "llm_config.json")):
        return subdir
    raise FileNotFoundError(
        f"No Lance checkpoint found at {model_path!r}: expected llm_config.json "
        f"either directly or under {_IMAGE_CKPT_DIR}/."
    )


def get_flattened_position_ids(img_h: int, img_w: int, patch_size: int, max_num_patches_per_side: int) -> torch.Tensor:
    """Compute flattened 2-D position IDs for latent patches (``hi * max_side + wi``)."""
    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


# ===================================================================
#  Transformer building blocks
# ===================================================================


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class LanceMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ===================================================================
#  mRoPE helpers
# ===================================================================


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(q, k, cos, sin):
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


class LanceRotaryEmbedding(nn.Module):
    """Qwen2.5-VL multimodal rotary embedding.

    Consumes 3-D position ids ``(B, 3, S)`` with rows ``(t, h, w)`` and
    assembles the per-section rotary basis exactly as the serving-side
    rotary does (per-axis frequencies split by ``mrope_section * 2`` with
    cyclic axis assignment ``i % 3``).  Frequencies use the standard
    default-rope basis.
    """

    def __init__(self, config: LanceTrainingConfig):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = list(config.mrope_section)

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Generate cos/sin for ``(B, 3, S)`` multimodal position ids.

        Returns:
            cos, sin: float32 tensors of shape ``(B, S, head_dim)``.
        """
        if position_ids.ndim != 3 or position_ids.shape[1] != 3:
            raise ValueError(f"expected multimodal position ids of shape (B, 3, S), got {tuple(position_ids.shape)}")
        B = position_ids.shape[0]
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(B, 3, -1, 1).to(position_ids.device)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, 3, S, head_dim)
        cos_per_axis = emb.cos()
        sin_per_axis = emb.sin()
        sec_full = self.mrope_section * 2
        cos_split = cos_per_axis.split(sec_full, dim=-1)
        sin_split = sin_per_axis.split(sec_full, dim=-1)
        cos = torch.cat([c[:, i % 3] for i, c in enumerate(cos_split)], dim=-1)
        sin = torch.cat([s[:, i % 3] for i, s in enumerate(sin_split)], dim=-1)
        return cos, sin


# ===================================================================
#  MoT Attention & Layer
# ===================================================================


class LanceMoTAttention(nn.Module):
    """MoT attention with separate understanding and generation projections."""

    def __init__(self, config: LanceTrainingConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm_moe_gen = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_mask: Tensor,
        latent_mask: Tensor,
        L_ctx: int = 0,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, L, _ = hidden_states.shape
        text_idx = text_mask.nonzero(as_tuple=True)
        latent_idx = latent_mask.nonzero(as_tuple=True)

        q = hidden_states.new_zeros(B, L, self.num_heads * self.head_dim)
        k = hidden_states.new_zeros(B, L, self.num_kv_heads * self.head_dim)
        v = hidden_states.new_zeros(B, L, self.num_kv_heads * self.head_dim)

        text_hs = hidden_states[text_idx]
        q[text_idx] = self.q_proj(text_hs)
        k[text_idx] = self.k_proj(text_hs)
        v[text_idx] = self.v_proj(text_hs)

        latent_hs = hidden_states[latent_idx]
        q[latent_idx] = self.q_proj_moe_gen(latent_hs)
        k[latent_idx] = self.k_proj_moe_gen(latent_hs)
        v[latent_idx] = self.v_proj_moe_gen(latent_hs)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)
        v = v.view(B, L, self.num_kv_heads, self.head_dim)

        # QK-norm + RoPE in float32 (cast to bf16 only for SDPA).
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        q_normed = q.new_zeros(q.shape)
        k_normed = k.new_zeros(k.shape)
        q_normed[text_idx] = self.q_norm(q[text_idx])
        k_normed[text_idx] = self.k_norm(k[text_idx])
        q_normed[latent_idx] = self.q_norm_moe_gen(q[latent_idx])
        k_normed[latent_idx] = self.k_norm_moe_gen(k[latent_idx])

        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        q_normed, k_normed = _apply_rotary_emb(q_normed, k_normed, cos, sin)

        q_normed = q_normed.to(torch.bfloat16)
        k_normed = k_normed.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k_normed = k_normed.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, L, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, L, self.num_heads, self.head_dim)

        q_normed = q_normed.transpose(1, 2)  # (B, H, L, D)
        k_normed = k_normed.transpose(1, 2)
        v = v.transpose(1, 2)

        if L_ctx > 0:
            if key_padding_mask is not None and not key_padding_mask.all():
                # Zero-padded text keys must not leak into either branch:
                # the serving side packs prompts so padded keys do not exist.
                text_key_mask = key_padding_mask[:, :L_ctx].view(B, 1, 1, L_ctx)
                causal_mask = torch.ones(L_ctx, L_ctx, dtype=torch.bool, device=hidden_states.device).tril()
                text_out = F.scaled_dot_product_attention(
                    q_normed[:, :, :L_ctx],
                    k_normed[:, :, :L_ctx],
                    v[:, :, :L_ctx],
                    attn_mask=text_key_mask & causal_mask.view(1, 1, L_ctx, L_ctx),
                    is_causal=False,
                )
                img_attn_mask = key_padding_mask.view(B, 1, 1, L)
                img_out = F.scaled_dot_product_attention(
                    q_normed[:, :, L_ctx:],
                    k_normed,
                    v,
                    attn_mask=img_attn_mask,
                    is_causal=False,
                )
            else:
                text_out = F.scaled_dot_product_attention(
                    q_normed[:, :, :L_ctx],
                    k_normed[:, :, :L_ctx],
                    v[:, :, :L_ctx],
                    is_causal=True,
                )
                img_out = F.scaled_dot_product_attention(
                    q_normed[:, :, L_ctx:],
                    k_normed,
                    v,
                    is_causal=False,
                )
            attn_out = torch.cat([text_out, img_out], dim=2)
        else:
            attn_out = F.scaled_dot_product_attention(q_normed, k_normed, v, is_causal=False)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)

        out = hidden_states.new_zeros(B, L, self.hidden_size)
        out[text_idx] = self.o_proj(attn_out[text_idx].to(self.o_proj.weight.dtype))
        out[latent_idx] = self.o_proj_moe_gen(attn_out[latent_idx].to(self.o_proj_moe_gen.weight.dtype))
        return out


class LanceMoTLayer(nn.Module):
    def __init__(self, config: LanceTrainingConfig):
        super().__init__()
        self.self_attn = LanceMoTAttention(config)
        self.mlp = LanceMLP(config.hidden_size, config.intermediate_size)
        self.mlp_moe_gen = LanceMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_mask: Tensor,
        latent_mask: Tensor,
        L_ctx: int = 0,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with MoT-routed layernorm, attention, and MLP.

        Args:
            hidden_states: ``(B, L, D)`` input sequence.
            cos: ``(B, L, head_dim)`` rotary cosines, shared by all layers.
            sin: ``(B, L, head_dim)`` rotary sines, shared by all layers.
            text_mask: Bool mask — True for text pathway.
            latent_mask: Bool mask — True for gen pathway.
            L_ctx: Text context length for the causal split.
            key_padding_mask: ``(B, L)`` — True at valid keys.

        Returns:
            Output of shape ``(B, L, D)``.
        """
        text_idx = text_mask.nonzero(as_tuple=True)
        latent_idx = latent_mask.nonzero(as_tuple=True)

        normed = hidden_states.new_zeros(hidden_states.shape)
        normed[text_idx] = self.input_layernorm(hidden_states[text_idx])
        normed[latent_idx] = self.input_layernorm_moe_gen(hidden_states[latent_idx])

        attn_out = self.self_attn(
            normed,
            cos,
            sin,
            text_mask,
            latent_mask,
            L_ctx,
            key_padding_mask=key_padding_mask,
        )
        hidden_states = hidden_states + attn_out

        residual = hidden_states
        mlp_out = hidden_states.new_zeros(hidden_states.shape)
        mlp_out[text_idx] = self.mlp(self.post_attention_layernorm(hidden_states[text_idx]))
        mlp_out[latent_idx] = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(hidden_states[latent_idx]))
        hidden_states = residual + mlp_out
        return hidden_states


# ===================================================================
#  Position embedding helpers
# ===================================================================


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_dim = freq_dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        pos_embed = _get_2d_sincos_pos_embed(hidden_size, max_num_patch_per_side)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float(), requires_grad=False)

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.pos_embed[position_ids]


# ===================================================================
#  Main module: LanceForTraining
# ===================================================================


class LanceForTraining(NonDiffusersModelBase):
    """Standalone Lance MoT module for FlowGRPO FSDP training.

    ``_no_split_modules`` enables layer-level FSDP sharding so that
    ``layered_summon`` finds ``layers.N`` for rollout weight sync.
    """

    _no_split_modules = ["LanceMoTLayer"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: LanceTrainingConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LanceMoTLayer(config) for _ in range(config.num_hidden_layers)])
        # One shared rotary module: position ids are identical across layers,
        # so cos/sin are computed once per forward and passed into each layer.
        self.rotary_emb = LanceRotaryEmbedding(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.vae2llm = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, config.patch_latent_dim)
        self.latent_pos_embed = PositionEmbedding(config.max_latent_size, config.hidden_size)

    # ------------------------------------------------------------------
    #  mRoPE position table
    # ------------------------------------------------------------------

    def _build_position_ids(
        self,
        latent_pos_ids: Tensor,
        text_attention_mask: Optional[Tensor],
        L_ctx: int,
        B: int,
        device: torch.device,
    ) -> Tensor:
        """Build the per-token ``(t, h, w)`` position table for the sequence.

        Mirrors the serving pipeline's latent position layout: with ``P`` the
        per-sample rope counter after the text prefix (0 for the
        text-unconditional branch),

        * text token ``j`` sits at ``(j, j, j)``,
        * ``start_of_image`` at ``(P, P, P)``,
        * latent ``(hi, wi)`` at ``(P+1, P+1+hi, P+1+wi)``,
        * ``end_of_image`` at ``(P+max(h,w)+1, …)``.

        ``hi``/``wi`` are recovered from the flattened 2-D latent position ids
        (``hi * max_latent_size + wi``).

        Returns:
            Long tensor of shape ``(B, 3, L_total)``.
        """
        max_side = self.config.max_latent_size
        hi = (latent_pos_ids // max_side).long()  # (B, L_latent)
        wi = (latent_pos_ids % max_side).long()
        h = hi.max(dim=1).values + 1  # (B,)
        w = wi.max(dim=1).values + 1
        max_hw = torch.maximum(h, w)

        if L_ctx > 0:
            if text_attention_mask is not None:
                P = text_attention_mask.sum(dim=-1).long()  # (B,) true text lengths
            else:
                P = torch.full((B,), L_ctx, dtype=torch.long, device=device)
        else:
            P = torch.zeros(B, dtype=torch.long, device=device)

        L_latent = latent_pos_ids.shape[1]
        L_total = L_ctx + 1 + L_latent + 1
        pos = torch.zeros(B, 3, L_total, dtype=torch.long, device=device)

        if L_ctx > 0:
            ctx = torch.arange(L_ctx, device=device).view(1, 1, L_ctx).expand(B, 3, L_ctx)
            pos[:, :, :L_ctx] = ctx

        soi = L_ctx
        lat0 = L_ctx + 1
        eoi = L_ctx + 1 + L_latent

        pos[:, :, soi] = P.view(B, 1).expand(B, 3)
        pos[:, 0, lat0:eoi] = (P + 1).view(B, 1)
        pos[:, 1, lat0:eoi] = (P + 1).view(B, 1) + hi
        pos[:, 2, lat0:eoi] = (P + 1).view(B, 1) + wi
        end_p = P + max_hw + 1
        pos[:, :, eoi] = end_p.view(B, 1).expand(B, 3)
        return pos

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        text_token_ids: Optional[Tensor],
        latent_pos_ids: Tensor,
        **kwargs,
    ) -> tuple[Tensor]:
        """Forward pass.

        Args:
            hidden_states: ``(B, L_latent, patch_latent_dim)`` noisy latent patches.
            timestep: ``(B,)`` diffusion timestep.
            text_token_ids: ``(B, L_text)`` token IDs, or ``None`` for the CFG
                text-unconditional branch.
            latent_pos_ids: ``(B, L_latent)`` flattened 2-D position indices.
            text_attention_mask: ``(B, L_text)`` bool mask (via ``**kwargs``).

        Returns:
            Tuple of ``(velocity,)`` — velocity prediction of shape
            ``(B, L_latent, patch_latent_dim)``.
        """
        text_attention_mask = kwargs.pop("text_attention_mask", None)
        if text_token_ids is not None and text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(device=text_token_ids.device, dtype=torch.bool)
            text_lengths = text_attention_mask.sum(dim=-1)
            if text_lengths.numel() > 0:
                text_length = int(text_lengths.max().item())
                if text_length > 0:
                    text_token_ids = text_token_ids[:, :text_length]
                    text_attention_mask = text_attention_mask[:, :text_length]
                else:
                    text_token_ids = None
                    text_attention_mask = None

        B = hidden_states.shape[0]
        L_latent = hidden_states.shape[1]
        dev = hidden_states.device

        # 1. Embed text context
        if text_token_ids is not None:
            text_embeds = self.embed_tokens(text_token_ids)
            L_ctx = text_embeds.shape[1]
        else:
            L_ctx = 0
            text_attention_mask = None

        # 2. SOI / EOI boundary tokens
        soi_ids = torch.full((B, 1), self.config.start_of_image_id, dtype=torch.long, device=dev)
        eoi_ids = torch.full((B, 1), self.config.end_of_image_id, dtype=torch.long, device=dev)
        soi_emb = self.embed_tokens(soi_ids)
        eoi_emb = self.embed_tokens(eoi_ids)

        # 3. Latent projection
        t_emb = self.time_embedder(timestep)
        pos_emb = self.latent_pos_embed(latent_pos_ids)
        latent_embeds = self.vae2llm(hidden_states) + t_emb.unsqueeze(1) + pos_emb
        latent_embeds = latent_embeds.to(soi_emb.dtype)

        # 4. Sequence: [text?, soi, latent_0..N, eoi]
        L_total = L_ctx + 1 + L_latent + 1
        if L_ctx > 0:
            sequence = torch.cat([text_embeds, soi_emb, latent_embeds, eoi_emb], dim=1)
        else:
            sequence = torch.cat([soi_emb, latent_embeds, eoi_emb], dim=1)

        # 5. MoT routing masks
        #    text pathway: text_ctx + soi + eoi
        #    gen pathway:  latent tokens only
        text_mask = torch.zeros(B, L_total, dtype=torch.bool, device=dev)
        text_mask[:, : L_ctx + 1] = True  # text + soi
        text_mask[:, -1] = True  # eoi
        latent_mask = ~text_mask

        # 6. mRoPE (t, h, w) positions — cos/sin computed once for all layers
        position_ids = self._build_position_ids(latent_pos_ids, text_attention_mask, L_ctx, B, dev)
        cos, sin = self.rotary_emb(position_ids)

        # Key padding mask: zero-padded text tokens in uneven micro-batches
        # must not attend to image queries.  ``None`` keeps the flash backend.
        if L_ctx > 0 and text_attention_mask is not None and not bool(text_attention_mask.all()):
            key_padding_mask = text_attention_mask.new_ones(B, L_total, dtype=torch.bool)
            key_padding_mask[:, :L_ctx] = text_attention_mask
        else:
            key_padding_mask = None

        # 7. Transformer layers (split attention: text causal + image full)
        for layer in self.layers:

            def _layer_fn(seq, cos_, sin_, text_mask_, latent_mask_, kpm, *, _layer=layer):
                return _layer(seq, cos_, sin_, text_mask_, latent_mask_, L_ctx, key_padding_mask=kpm)

            sequence = self._checkpointed_call(_layer_fn, sequence, cos, sin, text_mask, latent_mask, key_padding_mask)

        # 8. Final norm with MoT routing
        normed = sequence.new_zeros(sequence.shape)
        t_idx = text_mask.nonzero(as_tuple=True)
        l_idx = latent_mask.nonzero(as_tuple=True)
        normed[t_idx] = self.norm(sequence[t_idx])
        normed[l_idx] = self.norm_moe_gen(sequence[l_idx])

        # 9. Extract latent output
        latent_output = normed[:, L_ctx + 1 : L_ctx + 1 + L_latent, :]
        velocity = self.llm2vae(latent_output)

        return (velocity,)

    # ------------------------------------------------------------------
    #  Checkpoint loading (fail-closed)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype=torch.bfloat16) -> LanceForTraining:
        """Load pretrained weights from a Lance checkpoint directory.

        Accepts the bundled repo root (containing ``Lance_3B/``) or the
        checkpoint subdirectory itself.  Loading is fail-closed: any source
        key outside the defined remap, any non-allowlisted dropped key, and
        any unfilled module parameter raises instead of warning.
        """
        from safetensors.torch import load_file

        config = LanceTrainingConfig.from_model_path(model_path)
        ckpt_dir = resolve_lance_checkpoint_dir(model_path)
        ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
        # mmap-backed load: tensors stay on the OS page cache (shared across
        # concurrent workers loading the same checkpoint) until touched.
        state_dict = load_file(ckpt_path)

        if "latent_pos_embed.pos_embed" in state_dict:
            actual_len = state_dict["latent_pos_embed.pos_embed"].shape[0]
            grid = int(actual_len**0.5)
            if grid * grid == actual_len and grid != config.max_latent_size:
                config.max_latent_size = grid

        # Build parameters on the meta device and attach the checkpoint
        # tensors via ``assign=True`` — avoids materializing a second full
        # copy of the 3B model in host RAM per worker.
        with torch.device("meta"):
            model = cls(config)
        mapped, dropped, unmapped = _map_checkpoint_to_training(state_dict)
        if unmapped:
            raise RuntimeError(
                f"Lance checkpoint contains {len(unmapped)} keys outside the defined remap "
                f"(first 10: {sorted(unmapped)[:10]}). Refusing to load."
            )
        unexpected_dropped = dropped - CHECKPOINT_DROPPED_KEY_ALLOWLIST
        if unexpected_dropped:
            raise RuntimeError(
                f"Lance checkpoint keys dropped outside the allowlist: {sorted(unexpected_dropped)[:10]}"
            )

        missing, unexpected = model.load_state_dict(mapped, strict=False, assign=True)
        if missing:
            raise RuntimeError(
                f"Lance checkpoint is missing {len(missing)} required parameters "
                f"(first 10: {sorted(missing)[:10]}). Refusing to load."
            )
        if unexpected:
            raise RuntimeError(
                f"Lance checkpoint has {len(unexpected)} keys with no matching parameter "
                f"(first 10: {sorted(unexpected)[:10]}). Refusing to load."
            )

        model = model.to(torch_dtype)
        # The non-persistent rotary inv_freq buffer is not in the checkpoint
        # (still on the meta device); rebuild it *after* the dtype cast so it
        # stays fp32.  A bf16 inv_freq would dephase the mRoPE cos/sin from
        # the pinned rollout, which computes rotary frequencies in fp32.
        model.rotary_emb = LanceRotaryEmbedding(config)
        return model


# Keys deliberately not loaded: the AR head is unused by the velocity path
# (the released checkpoint ships it untied despite ``tie_word_embeddings``).
CHECKPOINT_DROPPED_KEY_ALLOWLIST = frozenset({"language_model.lm_head.weight"})

_TOP_LEVEL_PREFIXES = ("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")
_LLM_PREFIX = "language_model.model."


def _map_checkpoint_to_training(state_dict: dict[str, Tensor]) -> tuple[dict[str, Tensor], set[str], set[str]]:
    """Map released-checkpoint keys to ``LanceForTraining`` parameter names.

    Returns:
        ``(mapped, dropped, unmapped)`` where ``dropped`` holds source keys
        that are deliberately discarded and ``unmapped`` holds source keys
        that matched no remap rule (a non-empty ``unmapped`` must fail the
        load).
    """
    mapped: dict[str, Tensor] = {}
    dropped: set[str] = set()
    unmapped: set[str] = set()
    for src_key, tensor in state_dict.items():
        if src_key.startswith(_LLM_PREFIX):
            mapped[src_key[len(_LLM_PREFIX) :]] = tensor
        elif src_key in CHECKPOINT_DROPPED_KEY_ALLOWLIST:
            dropped.add(src_key)
        elif src_key.startswith(_TOP_LEVEL_PREFIXES):
            mapped[src_key] = tensor
        else:
            unmapped.add(src_key)
    return mapped, dropped, unmapped


def _map_training_to_checkpoint(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Inverse of :func:`_map_checkpoint_to_training` (checkpoint layout out).

    Used by round-trip tests; ``language_model.lm_head.weight`` is not
    reconstructed (it is never loaded).
    """
    out: dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith(_TOP_LEVEL_PREFIXES):
            out[key] = tensor
        else:
            out[_LLM_PREFIX + key] = tensor
    return out
