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

"""CPU tests for the Lance FlowGRPO pipeline package.

Covers registry dispatch, sigma-schedule parity with the rollout denoise
loop, mRoPE parity with the pinned vllm-omni Lance implementation, the
fail-closed checkpoint remap on a tiny-random checkpoint, SDE-window
selection and trajectory slicing, and scheduler float32 enforcement.
"""

import json
import os

import pytest
import torch

from verl_omni.pipelines.lance_flow_grpo.common import (
    LANCE_TIMESTEP_SHIFT,
    setup_lance_sigmas,
    vllm_omni_num_timesteps,
)
from verl_omni.pipelines.lance_flow_grpo.lance_model import (
    CHECKPOINT_DROPPED_KEY_ALLOWLIST,
    LanceForTraining,
    LanceRotaryEmbedding,
    LanceTrainingConfig,
    _map_checkpoint_to_training,
    _map_training_to_checkpoint,
    get_flattened_position_ids,
)
from verl_omni.pipelines.lance_flow_grpo.vllm_omni_rollout_adapter import _extract_prompt_text, _pick_sde_window
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler


def _tiny_config(**overrides) -> LanceTrainingConfig:
    kwargs = dict(
        hidden_size=48,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=200,
        mrope_section=[2, 2, 2],
        latent_channel=8,
        max_latent_size=8,
        start_of_image_id=100,
        end_of_image_id=101,
    )
    kwargs.update(overrides)
    return LanceTrainingConfig(**kwargs)


# ---------------------------------------------------------------------------
#  Registry dispatch
# ---------------------------------------------------------------------------


class TestRegistryDispatch:
    def test_training_registry_resolves(self):
        from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion
        from verl_omni.pipelines.model_base import DiffusionModelBase

        assert DiffusionModelBase._registry[("OmniLanceForConditionalGeneration", "flow_grpo")] is LanceDiffusion

    def test_rollout_registry_resolves(self):
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase

        path = VllmOmniPipelineBase.get_pipeline_path("OmniLanceForConditionalGeneration", "flow_grpo")
        assert path == "verl_omni.pipelines.lance_flow_grpo.vllm_omni_rollout_adapter.LancePipelineWithLogProb"

    def test_unknown_architecture_has_no_silent_fallback(self):
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase

        assert VllmOmniPipelineBase.get_class("OmniLanceForConditionalGeneration", "no_such_algo") is None
        assert VllmOmniPipelineBase.get_pipeline_path("NoSuchArchitecture", "flow_grpo") is None


# ---------------------------------------------------------------------------
#  Sigma schedule
# ---------------------------------------------------------------------------


class TestSchedule:
    def test_default_shift_is_lance(self):
        assert LANCE_TIMESTEP_SHIFT == 3.5

    def test_sigmas_match_rollout_denoise_loop(self):
        """setup_lance_sigmas(N) must equal the sigmas the pinned denoise loop
        produces for ``generate_image(num_timesteps=vllm_omni_num_timesteps(N))``.
        """
        num_steps = 15
        scheduler = FlowMatchSDEDiscreteScheduler()
        sigmas = setup_lance_sigmas(scheduler, num_steps)

        # Pinned Bagel.generate_image: linspace(1, 0, n+1) -> shift -> drop terminal.
        n_rollout = vllm_omni_num_timesteps(num_steps)
        t = torch.linspace(1, 0, n_rollout + 1)
        t = LANCE_TIMESTEP_SHIFT * t / (1 + (LANCE_TIMESTEP_SHIFT - 1) * t)
        rollout_sigmas = t[:-1]

        assert len(sigmas) == n_rollout
        assert torch.allclose(torch.tensor(sigmas), rollout_sigmas)

    def test_step_count_compensation(self):
        assert vllm_omni_num_timesteps(15) == 14
        assert vllm_omni_num_timesteps(2) == 1
        assert vllm_omni_num_timesteps(1) == 1

    def test_rejects_nonpositive_steps(self):
        with pytest.raises(ValueError):
            setup_lance_sigmas(FlowMatchSDEDiscreteScheduler(), 0)

    def test_set_timesteps_honors_pipeline_timestep_shift(self):
        """A rollout-side timestep_shift override must reach the training
        scheduler too, or the trainer's exact-equality timestep lookup fails."""
        from types import SimpleNamespace

        from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion

        num_steps = 15
        overridden = FlowMatchSDEDiscreteScheduler()
        model_config = SimpleNamespace(pipeline=SimpleNamespace(num_inference_steps=num_steps, timestep_shift=3.0))
        LanceDiffusion.set_timesteps(overridden, model_config, "cpu")
        expected = setup_lance_sigmas(FlowMatchSDEDiscreteScheduler(), num_steps, shift=3.0)
        assert torch.allclose(overridden.timesteps, torch.tensor(expected))

        default = FlowMatchSDEDiscreteScheduler()
        model_config = SimpleNamespace(pipeline=SimpleNamespace(num_inference_steps=num_steps))
        LanceDiffusion.set_timesteps(default, model_config, "cpu")
        expected = setup_lance_sigmas(FlowMatchSDEDiscreteScheduler(), num_steps)
        assert torch.allclose(default.timesteps, torch.tensor(expected))


# ---------------------------------------------------------------------------
#  mRoPE parity with the pinned rollout implementation
# ---------------------------------------------------------------------------


class TestMRoPEParity:
    def test_latent_position_table_matches_pinned_pipeline(self):
        from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel

        class _FakeBagel:
            latent_downsample = 16
            _per_token_mrope_for_vae_latent = LanceBagel._per_token_mrope_for_vae_latent

        grid_h, grid_w, text_len = 3, 5, 7
        rollout_tbl = _FakeBagel()._per_token_mrope_for_vae_latent([(grid_h * 16, grid_w * 16)], [text_len])

        cfg = _tiny_config()
        model = LanceForTraining(cfg)
        lat_pos = get_flattened_position_ids(grid_h * 16, grid_w * 16, 16, cfg.max_latent_size).unsqueeze(0)
        mask = torch.ones(1, text_len, dtype=torch.bool)
        trainer_tbl = model._build_position_ids(lat_pos, mask, L_ctx=text_len, B=1, device=torch.device("cpu"))

        assert torch.equal(trainer_tbl[0, :, text_len:], rollout_tbl)

    def test_unconditional_table_anchored_at_zero(self):
        """The CFG text-unconditional branch keeps an empty KV cache in the
        rollout, so its latent table is anchored at rope position 0."""
        from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel

        class _FakeBagel:
            latent_downsample = 16
            _per_token_mrope_for_vae_latent = LanceBagel._per_token_mrope_for_vae_latent

        rollout_tbl = _FakeBagel()._per_token_mrope_for_vae_latent([(4 * 16, 4 * 16)], [0])

        cfg = _tiny_config()
        model = LanceForTraining(cfg)
        lat_pos = get_flattened_position_ids(4 * 16, 4 * 16, 16, cfg.max_latent_size).unsqueeze(0)
        trainer_tbl = model._build_position_ids(lat_pos, None, L_ctx=0, B=1, device=torch.device("cpu"))

        assert torch.equal(trainer_tbl[0], rollout_tbl)

    def test_rotary_matches_pinned_rotary(self):
        from vllm_omni.diffusion.models.bagel.bagel_transformer import BagelRotaryEmbedding

        cfg = _tiny_config()

        class _PinCfg:
            rope_scaling = {"rope_type": "mrope", "mrope_section": list(cfg.mrope_section)}
            rope_parameters = None
            rope_theta = cfg.rope_theta
            hidden_size = cfg.hidden_size
            num_attention_heads = cfg.num_attention_heads

        pin_rope = BagelRotaryEmbedding(_PinCfg())
        trainer_rope = LanceRotaryEmbedding(cfg)

        pos = torch.stack(
            [
                torch.arange(10),
                torch.arange(10) + 3,
                torch.arange(10) + 7,
            ]
        ).unsqueeze(0)
        cos_pin, sin_pin = pin_rope(torch.zeros(1, dtype=torch.float32), pos)
        cos_tr, sin_tr = trainer_rope(pos)
        assert torch.equal(cos_pin, cos_tr)
        assert torch.equal(sin_pin, sin_tr)

    def test_rotary_rejects_scalar_positions(self):
        cfg = _tiny_config()
        rope = LanceRotaryEmbedding(cfg)
        with pytest.raises(ValueError):
            rope(torch.arange(10).unsqueeze(0))


# ---------------------------------------------------------------------------
#  Forward semantics
# ---------------------------------------------------------------------------


class TestForward:
    def _run(self, model, cfg, text_ids, mask, B=2, grid=4):
        lat_pos = get_flattened_position_ids(grid * 16, grid * 16, 16, cfg.max_latent_size)
        lat_pos = lat_pos.unsqueeze(0).expand(B, -1)
        torch.manual_seed(0)
        x = torch.randn(B, grid * grid, cfg.patch_latent_dim)
        t = torch.full((B,), 0.7)
        with torch.no_grad():
            (v,) = model(
                hidden_states=x,
                timestep=t,
                text_token_ids=text_ids,
                latent_pos_ids=lat_pos,
                text_attention_mask=mask,
            )
        return v

    def test_velocity_shape_and_conditioning(self):
        cfg = _tiny_config()
        model = LanceForTraining(cfg).eval()
        ids = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        ids[0, :3] = torch.tensor([102, 5, 103])
        mask[0, :3] = True
        ids[1, :5] = torch.tensor([102, 5, 6, 7, 103])
        mask[1, :5] = True

        v = self._run(model, cfg, ids, mask)
        v_uncond = self._run(model, cfg, None, None)
        assert v.shape == (2, 16, cfg.patch_latent_dim)
        assert torch.isfinite(v).all()
        assert not torch.allclose(v, v_uncond)

    def test_padding_invariance(self):
        """Extra zero padding on the text side must not change the velocity."""
        cfg = _tiny_config()
        model = LanceForTraining(cfg).eval()
        ids = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        ids[0, :3] = torch.tensor([102, 5, 103])
        mask[0, :3] = True
        ids[1, :5] = torch.tensor([102, 5, 6, 7, 103])
        mask[1, :5] = True

        ids_padded = torch.zeros(2, 9, dtype=torch.long)
        mask_padded = torch.zeros(2, 9, dtype=torch.bool)
        ids_padded[:, :5] = ids
        mask_padded[:, :5] = mask

        v = self._run(model, cfg, ids, mask)
        v_padded = self._run(model, cfg, ids_padded, mask_padded)
        assert torch.allclose(v, v_padded, atol=1e-5)

    def test_incompatible_config_raises(self):
        with pytest.raises(ValueError):
            _tiny_config(hidden_size=50)  # not divisible by 4 heads
        with pytest.raises(ValueError):
            _tiny_config(mrope_section=[2, 2, 1])  # does not sum to head_dim/2
        with pytest.raises(ValueError):
            _tiny_config(num_key_value_heads=3)  # heads not divisible


# ---------------------------------------------------------------------------
#  Fail-closed checkpoint loading (tiny-random)
# ---------------------------------------------------------------------------


def _write_tiny_checkpoint(tmp_path, cfg: LanceTrainingConfig, mutate=None):
    """Write a tiny-random checkpoint in the released Lance_3B layout."""
    from safetensors.torch import save_file

    model = LanceForTraining(cfg)
    state = _map_training_to_checkpoint({k: v for k, v in model.state_dict().items()})
    for v in state.values():
        if v.is_floating_point():
            v.normal_(std=0.02)
    state["language_model.lm_head.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02
    if mutate is not None:
        mutate(state)

    ckpt_dir = tmp_path / "Lance_3B"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(dict(state), str(ckpt_dir / "model.safetensors"))

    llm_config = {
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_scaling": {"type": "mrope", "mrope_section": list(cfg.mrope_section)},
        "vision_start_token_id": cfg.start_of_image_id,
        "vision_end_token_id": cfg.end_of_image_id,
        "tie_word_embeddings": True,
    }
    with open(ckpt_dir / "llm_config.json", "w") as f:
        json.dump(llm_config, f)
    return tmp_path, state


class TestCheckpointLoading:
    def test_tiny_random_round_trip(self, tmp_path):
        cfg = _tiny_config()
        # Constants not carried in llm_config.json must match the tiny model.
        root, src_state = _write_tiny_checkpoint(tmp_path, cfg)

        # from_pretrained reads latent geometry from Lance constants; patch the
        # tiny geometry through the pos-embed grid inference + explicit config.
        loaded_cfg = LanceTrainingConfig.from_model_path(str(root))
        assert loaded_cfg.num_hidden_layers == cfg.num_hidden_layers
        assert loaded_cfg.mrope_section == cfg.mrope_section

        mapped, dropped, unmapped = _map_checkpoint_to_training(src_state)
        assert unmapped == set()
        assert dropped == set(CHECKPOINT_DROPPED_KEY_ALLOWLIST)

        model = LanceForTraining(cfg)
        # latent channel differs from the 3B default; ensure clean load.
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        assert missing == []
        assert unexpected == []

        # Round trip: training state -> checkpoint layout -> equal tensors.
        rebuilt = _map_training_to_checkpoint(model.state_dict())
        for key, tensor in rebuilt.items():
            assert torch.equal(tensor, src_state[key]), key
        assert set(src_state) - set(rebuilt) == set(CHECKPOINT_DROPPED_KEY_ALLOWLIST)

    def test_renamed_forward_critical_key_raises(self, tmp_path):
        cfg = _tiny_config()

        def rename(state):
            old = "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight"
            state["language_model.model.layers.0.self_attn.q_proj_gen.weight"] = state.pop(old)

        root, _ = _write_tiny_checkpoint(tmp_path, cfg, mutate=rename)
        with pytest.raises(RuntimeError):
            _load_tiny(root, cfg)

    def test_missing_key_raises(self, tmp_path):
        cfg = _tiny_config()

        def drop(state):
            state.pop("vae2llm.bias")

        root, _ = _write_tiny_checkpoint(tmp_path, cfg, mutate=drop)
        with pytest.raises(RuntimeError):
            _load_tiny(root, cfg)

    def test_unknown_extra_key_raises(self, tmp_path):
        cfg = _tiny_config()

        def add(state):
            state["mystery.weight"] = torch.zeros(2, 2)

        root, _ = _write_tiny_checkpoint(tmp_path, cfg, mutate=add)
        with pytest.raises(RuntimeError):
            _load_tiny(root, cfg)

    def test_from_pretrained_keeps_rotary_inv_freq_fp32(self, tmp_path):
        """The ``model.to(torch_dtype)`` cast in from_pretrained must not leave
        a downcast inv_freq behind: the pinned rollout computes rotary
        frequencies in fp32, and bf16 frequencies dephase the mRoPE cos/sin
        from the rollout trajectory."""
        # Use the 3B latent-geometry constants so from_pretrained's config
        # (which reads only llm_config.json) matches the tiny checkpoint.
        cfg = _tiny_config(latent_channel=48, max_latent_size=64)
        root, _ = _write_tiny_checkpoint(tmp_path, cfg)

        model = LanceForTraining.from_pretrained(str(root), torch_dtype=torch.bfloat16)
        assert model.rotary_emb.inv_freq.dtype == torch.float32

        pos = torch.stack([torch.arange(10), torch.arange(10) + 3, torch.arange(10) + 7]).unsqueeze(0)
        cos, sin = model.rotary_emb(pos)
        ref_cos, ref_sin = LanceRotaryEmbedding(cfg)(pos)
        assert torch.equal(cos, ref_cos)
        assert torch.equal(sin, ref_sin)


def _load_tiny(root, cfg: LanceTrainingConfig):
    """Run the fail-closed load against a tiny checkpoint dir.

    Mirrors ``LanceForTraining.from_pretrained`` but constructs the module
    from the tiny config (the 3B latent constants don't apply to tiny tests).
    """
    from safetensors.torch import load_file

    ckpt_path = os.path.join(str(root), "Lance_3B", "model.safetensors")
    state_dict = load_file(ckpt_path)
    model = LanceForTraining(cfg)
    mapped, dropped, unmapped = _map_checkpoint_to_training(state_dict)
    if unmapped:
        raise RuntimeError(f"unmapped keys: {sorted(unmapped)[:5]}")
    if dropped - CHECKPOINT_DROPPED_KEY_ALLOWLIST:
        raise RuntimeError(f"unexpected dropped keys: {sorted(dropped)[:5]}")
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"missing={missing[:5]} unexpected={unexpected[:5]}")
    return model


# ---------------------------------------------------------------------------
#  Prompt text extraction
# ---------------------------------------------------------------------------


class TestPromptExtraction:
    def test_native_caption_starting_with_user_is_not_stripped(self):
        """Lance's native t2i format has no role header; a caption that merely
        begins with the word "user" must survive intact."""
        decoded = "<|im_start|>user interface design of a mobile app<|im_end|>"
        assert _extract_prompt_text(decoded) == "user interface design of a mobile app"

    def test_chat_template_role_header_is_stripped(self):
        decoded = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n"
        assert _extract_prompt_text(decoded) == "a cat"

    def test_plain_wrapped_caption(self):
        assert _extract_prompt_text("<|im_start|>a red panda<|im_end|>") == "a red panda"


# ---------------------------------------------------------------------------
#  Training-side CFG gating and input caching
# ---------------------------------------------------------------------------


class TestTrainingAdapterForward:
    def test_cfg_interval_gated_per_sample(self):
        """Samples from different SDE windows can straddle the cfg_interval
        boundary within one micro-batch; each sample must get the velocity the
        rollout actually produced for its own sigma."""
        from types import SimpleNamespace

        from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion

        scheduler = FlowMatchSDEDiscreteScheduler()
        setup_lance_sigmas(scheduler, 10)
        sigma_in = scheduler.timesteps[0]  # 1.0 — inside (0.4, 1.0]
        sigma_out = scheduler.timesteps[-1]  # ~0.30 — outside
        assert sigma_in > 0.4 and sigma_out <= 0.4

        B, L, D = 2, 4, 8
        torch.manual_seed(0)
        latents = torch.randn(B, 2, L, D, dtype=torch.float32)
        timesteps = torch.stack([sigma_in.view(1), sigma_out.view(1)])  # (B, 1)

        class _ConstModule:
            """Gen branch (text conditioned) predicts 2.0, uncond 1.0."""

            def __call__(self, **kwargs):
                hs = kwargs["hidden_states"]
                val = 2.0 if kwargs.get("text_token_ids") is not None else 1.0
                return (torch.full_like(hs, val),)

        # cfg_renorm_min=1.0 disables the renorm shrink so the CFG-combined
        # velocity (5.0) is distinguishable from the gen branch (2.0).
        model_config = SimpleNamespace(
            pipeline=SimpleNamespace(cfg_renorm_min=1.0),
            algo=SimpleNamespace(noise_level=0.7, sde_type="sde"),
        )
        model_inputs = {"hidden_states": latents[:, 0], "text_token_ids": torch.ones(B, 3, dtype=torch.long)}
        negative_model_inputs = {"hidden_states": latents[:, 0], "text_token_ids": None}

        log_prob, prev_sample_mean, _, _ = LanceDiffusion.forward_and_sample_previous_step(
            module=_ConstModule(),
            scheduler=scheduler,
            model_config=model_config,
            model_inputs=model_inputs,
            negative_model_inputs=negative_model_inputs,
            scheduler_inputs={"all_latents": latents, "all_timesteps": timesteps},
            step=0,
        )

        # Sample 0 (in interval): CFG-combined 1 + 4*(2-1) = 5.  Sample 1
        # (below the interval): raw gen-branch velocity 2.
        expected_v = torch.stack([torch.full((L, D), 5.0), torch.full((L, D), 2.0)])
        ref_log_prob, ref_mean, _, _ = scheduler.sample_previous_step(
            sample=latents[:, 0],
            model_output=expected_v,
            timestep=timesteps[:, 0],
            noise_level=0.7,
            prev_sample=latents[:, 1],
            sde_type="sde",
            return_logprobs=True,
            return_sqrt_dt=True,
            include_logprob_normalizer=False,
        )[1:]
        assert torch.allclose(prev_sample_mean, ref_mean)
        assert torch.allclose(log_prob, ref_log_prob)

    def test_prepare_model_inputs_caches_step_independent_tensors(self):
        """Token padding and latent position IDs must be computed once per
        micro-batch, not at every denoising step."""
        from types import SimpleNamespace

        from tensordict import TensorDict

        from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion

        B, T, L, D = 2, 3, 16, 8
        micro_batch = TensorDict({"prompt_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}, batch_size=[B])
        latents = torch.randn(B, T + 1, L, D)
        timesteps = torch.rand(B, T)
        module = SimpleNamespace(config=SimpleNamespace(latent_patch_size=1, vae_downsample=16, max_latent_size=8))
        model_config = SimpleNamespace(pipeline=SimpleNamespace(height=64, width=64))

        def _prepare(step):
            return LanceDiffusion.prepare_model_inputs(
                module=module,
                model_config=model_config,
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=None,
                prompt_embeds_mask=None,
                negative_prompt_embeds=None,
                negative_prompt_embeds_mask=None,
                micro_batch=micro_batch,
                step=step,
            )

        first, first_neg = _prepare(0)
        second, _ = _prepare(1)
        assert second["text_token_ids"] is first["text_token_ids"]
        assert second["text_attention_mask"] is first["text_attention_mask"]
        assert second["latent_pos_ids"] is first["latent_pos_ids"]
        assert first_neg["text_token_ids"] is None
        assert torch.equal(first["hidden_states"], latents[:, 0])
        assert torch.equal(second["hidden_states"], latents[:, 1])
        assert first["text_attention_mask"].all()


# ---------------------------------------------------------------------------
#  SDE window + trajectory slicing
# ---------------------------------------------------------------------------


class TestSdeWindowAndSlicing:
    def test_window_disabled(self):
        assert _pick_sde_window(None, None, seed=0) is None
        assert _pick_sde_window(0, (0, 7), seed=0) is None

    def test_window_without_range(self):
        assert _pick_sde_window(2, None, seed=0) == (0, 2)

    def test_window_seeding_is_per_gpu_not_per_request(self):
        """The window is a function of the seed only: one GPU (one LOCAL_RANK
        seed) must produce the same window for every request it serves."""
        windows = {_pick_sde_window(2, (0, 7), seed=3) for _ in range(32)}
        assert len(windows) == 1

        # Different GPU ranks are allowed to (and here do) pick different windows.
        all_windows = {_pick_sde_window(2, (0, 7), seed=s) for s in range(16)}
        assert len(all_windows) > 1

    def test_window_fits_range(self):
        for seed in range(20):
            begin, end = _pick_sde_window(2, (0, 7), seed=seed)
            assert 0 <= begin <= 5
            assert end - begin == 2

    def test_trajectory_slicing_lengths(self):
        """Slicing must keep window+1 latents and window timesteps."""
        num_steps, hw, dim = 14, 16, 8
        traj_latents = torch.randn(num_steps + 1, hw, dim, dtype=torch.float32)
        traj_timesteps = torch.rand(num_steps, dtype=torch.float32)

        begin, end = 3, 5
        sliced_latents = traj_latents[begin : end + 1]
        sliced_timesteps = traj_timesteps[begin:end]

        window = end - begin
        assert sliced_latents.shape[0] == window + 1
        assert sliced_timesteps.shape[0] == window
        assert torch.equal(sliced_latents[0], traj_latents[begin])
        assert torch.equal(sliced_latents[-1], traj_latents[end])


# ---------------------------------------------------------------------------
#  Scheduler float32 enforcement
# ---------------------------------------------------------------------------


class TestSchedulerDtype:
    def test_bf16_inputs_rejected(self):
        scheduler = FlowMatchSDEDiscreteScheduler()
        setup_lance_sigmas(scheduler, 10)
        sample = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        model_output = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        with pytest.raises(AssertionError):
            scheduler.sample_previous_step(
                sample=sample,
                model_output=model_output,
                timestep=scheduler.timesteps[0],
                noise_level=0.7,
                sde_type="sde",
            )

    def test_fp32_inputs_accepted(self):
        scheduler = FlowMatchSDEDiscreteScheduler()
        sigmas = setup_lance_sigmas(scheduler, 10)
        sample = torch.randn(1, 4, 8, dtype=torch.float32)
        model_output = torch.randn(1, 4, 8, dtype=torch.float32)
        out = scheduler.sample_previous_step(
            sample=sample,
            model_output=model_output,
            timestep=torch.tensor([sigmas[0]]),
            noise_level=0.7,
            sde_type="sde",
            return_logprobs=False,
        )
        prev_sample = out[0]
        assert prev_sample.dtype == torch.float32
        assert torch.isfinite(prev_sample).all()


# ---------------------------------------------------------------------------
#  LoRA collection prefix mismatch
# ---------------------------------------------------------------------------


class TestLoraCollection:
    def test_misconfigured_prefix_collects_nothing(self):
        from verl_omni.utils.fsdp_utils import _layered_summon_lora_params_diffusers

        cfg = _tiny_config()
        model = LanceForTraining(cfg)
        collected = _layered_summon_lora_params_diffusers(model, layer_prefixes=["transformer_blocks."])
        assert len(collected) == 0

    def test_zero_collected_params_raises(self, monkeypatch):
        """A prefix mismatch yields an empty collection (previous test); an
        empty collection must abort weight sync with ``RuntimeError`` rather
        than silently syncing nothing."""
        from collections import OrderedDict

        from verl_omni.utils import fsdp_utils

        monkeypatch.setattr(fsdp_utils, "_collect_lora_params_with_adapter", lambda *args, **kwargs: OrderedDict())

        cfg = _tiny_config()
        model = LanceForTraining(cfg)
        with pytest.raises(RuntimeError, match="collected 0 parameters"):
            fsdp_utils.collect_lora_params(
                model,
                layered_summon=True,
                base_sync_done=True,
                is_diffusers=True,
                adapter_name="lance_lora",
                layer_prefixes=["wrong_prefix."],
            )


# ---------------------------------------------------------------------------
#  Data preprocessing token format
# ---------------------------------------------------------------------------


def _load_lance_pickscore_module():
    """Load the data-process script without executing its __main__ body."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lance_pickscore",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "examples",
            "flowgrpo_trainer",
            "data_process",
            "lance_pickscore.py",
        ),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakePickscoreTokenizer:
    def __init__(self, caption_ids):
        self._caption_ids = caption_ids

    def convert_tokens_to_ids(self, tok):
        return {"<|im_start|>": 151644, "<|im_end|>": 151645}[tok]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(self._caption_ids)


class TestPromptFormat:
    def test_lance_prompt_wrap(self):
        module = _load_lance_pickscore_module()
        ids = module.tokenize_lance_prompt(_FakePickscoreTokenizer([1, 2, 3]), "a cat")
        assert ids == [151644, 1, 2, 3, 151645]

    def test_lance_prompt_truncation_keeps_end_marker(self):
        """Truncation must drop caption tokens, never <|im_end|>: the rollout
        re-wraps the decoded caption with a fresh <|im_end|>, so a stored
        sequence missing the marker diverges from the rollout conditioning."""
        module = _load_lance_pickscore_module()
        ids = module.tokenize_lance_prompt(_FakePickscoreTokenizer(range(1, 11)), "long caption", max_length=6)
        assert ids == [151644, 1, 2, 3, 4, 151645]

    def test_lance_prompt_rejects_tiny_max_length(self):
        module = _load_lance_pickscore_module()
        with pytest.raises(ValueError):
            module.tokenize_lance_prompt(_FakePickscoreTokenizer([1]), "a cat", max_length=1)
