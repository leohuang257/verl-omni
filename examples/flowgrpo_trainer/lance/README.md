# Lance 3B FlowGRPO training

[Lance](https://huggingface.co/bytedance-research/Lance) is a 3B native
unified multimodal model (BAGEL-lineage Qwen2.5-VL Mixture-of-Transformers)
supporting image and video understanding, generation, and editing.  Like
BAGEL, Lance is a **non-diffusers** model — it cannot be loaded by diffusers
and uses its own weight-loading path via ``NonDiffusersModelBase``.  See
[docs/contributing/integrating_a_non_diffusers_model.md](../../docs/contributing/integrating_a_non_diffusers_model.md)
for the integration architecture.

This recipe covers the text-to-image path (``Lance_3B`` checkpoint) with LoRA
on the generation-expert pathway (the ``*_moe_gen`` projections) and the
PickScore reward.  Rollouts go through the pinned ``vllm-omni`` ``LancePipeline``
with no upstream serving changes.

Lance deltas handled by the ``lance_flow_grpo`` package (vs the BAGEL recipe):

- **Timestep shift 3.5, 30 serving steps** (``LanceDefaults``): the inherited
  t2i serving path would otherwise default to BAGEL's shift 3.0 / 50 steps;
  the rollout adapter injects the Lance values explicitly and both adapters
  share ``setup_lance_sigmas``.
- **Qwen2.5-VL mRoPE**: latent tokens carry per-token ``(t, h, w)`` positions
  (``mrope_section=[16, 24, 24]``); the trainer mirrors the serving
  pipeline's latent position table exactly.
- **Wan2.2 latent geometry**: 48 latent channels, latent downsample 16,
  64×64 max latent grid (up to 1024×1024 output).
- **Fail-closed checkpoint loading**: every checkpoint key must map onto the
  training module; only the unused AR head
  (``language_model.lm_head.weight``) is deliberately dropped.

## Prerequisites

- Install VeRL-Omni (see [docs/start/install.md](../../docs/start/install.md)).

- 4 GPUs.  Run commands from the repository root.

- Download the bundled checkpoint (image side):

  ```bash
  hf download bytedance-research/Lance \
    --include "Lance_3B/*" --include "Qwen2.5-VL-ViT/*" \
    --include "Wan2.2_VAE.pth" --include "config.json" \
    --local-dir ~/models/Lance
  ```

## PickScore training

### Prepare the dataset

Download the raw PickScore prompts (``train.txt`` / ``test.txt``) from
[flow_grpo](https://github.com/yifan123/flow_grpo/tree/main/dataset/pickscore)
into ``~/data/pickscore``, then pre-tokenize them in Lance's native t2i
format (``[<|im_start|>] caption [<|im_end|>]`` — the serving path applies no
chat template):

```bash
python3 examples/flowgrpo_trainer/data_process/lance_pickscore.py \
  --model_path ~/models/Lance \
  --input_dir ~/data/pickscore \
  --output_dir ~/data/pickscore/lance
```

### Run training

```bash
bash examples/flowgrpo_trainer/lance/run_lance_pickscore_lora.sh
```

Set ``LANCE_MODEL_PATH`` if the checkpoint is not at ``~/models/Lance``.
The run requires the explicit
``+actor_rollout_ref.model.architecture=OmniLanceForConditionalGeneration``
override (already in the script); without it registry resolution fails —
there is no silent fallback.

The PickScore reward model
([yuvalkirstain/PickScore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1))
is downloaded on first use.

### LoRA notes

The LoRA targets are the seven gen-expert projections per layer
(``q/k/v/o_proj_moe_gen``, ``mlp_moe_gen.{gate,up,down}_proj``).  On the
rollout side these fold into fused modules (``qkv_proj_moe_gen``: 3 packed
slices; ``mlp_moe_gen.gate_up_proj``: 2 packed slices), discovered through
the model's ``stacked_params_mapping``.  Always target the MLP projections
with the ``mlp_moe_gen.`` parent prefix — a bare ``gate_proj`` pattern would
also match the understanding expert.

``fsdp_layer_prefixes=['layers.']`` matches ``LanceForTraining``'s decoder
stack and is required for layered-summon LoRA weight sync.

## Tests

CPU tests for the package:

```bash
pytest tests/pipelines/test_lance_flow_grpo_on_cpu.py
```
