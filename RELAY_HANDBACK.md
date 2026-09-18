# Relay-warmstarted I-DLM — handback

Branch: `relay-2step-impl` (from `origin/relay-2step` @ `7b97d66`). `main` was not touched.

## What was implemented

Two-pass K=2 training so I-DLM can consume Relay-style hidden-state warmstarts on rejected draft positions.

| Piece | Where |
| --- | --- |
| CPU p/q verify + vectorized `a_b` (cumprod) + `o_b = s_b + a_b + 2` | `training/llama_factory_sdar/src/llamafactory/train/sft/relay_verify.py` |
| Anchor-parameterized mask (slots ≠ positions) | `relay_mask.py` and `training/model/Qwen3-8B-b7-allmasked/modeling_sdar.py` |
| Packed step-2 canvas + carry map | `relay_canvas.py` |
| Two-pass step helpers | `relay_step.py` |
| Trainer: pass-1 combined loss, no-grad verify, pass-2 mask-CE only, 0.5× scaling, `relay_rollout_only` | `trainer.py`; flag in `finetuning_args.py` |
| Zero-init `relay_layer_norm`, mask-gated `emb += LN(h)`, last-layer-only `outputs.relay_h_last` | b7 `modeling_sdar.py` only (b1/b2/b3 left pristine) |
| Matched LoRA yamls (arms A/B/C) | `training/llama_factory_sdar/examples/train_idlm/qwen3_8b_b7-*-lora.yaml` |
| Tiny-model tests | `training/tests/` |

**Injection interface.** `relay_h` and `relay_mask` are length-`L` (noisy half). `relay_h_last` is the last decoder layer over the full `2L` layout; the trainer splits halves. `h` is detached before injection (Phase-2 BPTT can drop the `.detach()`). Clean slots are never written.

**Trainer structure.** Vanilla (`relay_enable: false`) still calls `super().training_step` once. Relay uses a single override that calls `compute_loss` + `accelerator.backward` twice. That avoids double-counting HF token counters / callbacks / grad-accum that would come from `super().training_step()` twice. Each pass is multiplied by `0.5` (and then by `1/grad_accum`) so logged + gradient magnitude is comparable to the one-pass vanilla arm. Pass 2 does **not** add clean CE.

## Environment

Discovered on the Unity login node (no GPU):

| Item | Value |
| --- | --- |
| Python | `/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/envs/relay/bin/python` (3.10.20) |
| torch | 2.12.0+cu130 |
| `flex_attention` | importable |
| transformers | 4.53.1 |
| pytest | 9.1.1 |
| CUDA in this session | **unavailable** (`torch.cuda.is_available() == False`) |
| `srun` | present; GPU tests were not queued (login-node / non-interactive allocation not used) |
| `idlm` conda env | not used; `relay` already had torch + flex |

GPU-only tests are `pytest.mark.skipif(not torch.cuda.is_available())`.

## Tests

Run from `training/tests` with the `relay` env:

```bash
/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/envs/relay/bin/python -m pytest -q
```

| Test | Result |
| --- | --- |
| `test_verify.py` (p=q accept, p=0 reject, all/first/mid-block `a_b`, prompt/pad, property vs Python loop, anchors) | pass |
| `test_anchor_formula.py` (plan §4.3 worked example, all three cases) | pass |
| `test_mask.py` (uniform anchors ≡ legacy `block_diff_mask`; ragged isolation / clean visibility) | pass |
| `test_canvas.py` (formula positions, clamp/drop, carry map, prompt guard, rollout_only) | pass |
| `test_injection.py` (zero-init identity, clean-half never injected, gamma=ones changes carry slots only, grads on LN in arm C / absent in arm B) | pass |
| `test_integration.py::test_tiny_relay_smoke_loss_decreases` (50 steps, arm C) | pass — late-window loss < early-window; accept metrics in [0, 1] |
| `test_integration.py::test_tiny_rollout_only_smoke` (50 steps, arm B) | pass |
| `test_integration.py` vanilla equivalence / determinism | pass |
| `test_integration.py::test_gpu_short_relay_run` | **skipped** (no GPU on the login node) |

**33 passed, 1 skipped** on 2026-09-18 (CPU).

## Ablation configs (matched)

All three: `open_thoughts3_sample`, `block_length: 7`, `finetuning_type: lora`, `lora_rank: 128`, `cutoff_len: 4096`, batch 1, grad accum 4, lr `2e-4` cosine, warmup 0.03, bf16, `seed: 42`, `max_steps: 500`.

| Arm | File | Flags |
| --- | --- | --- |
| A vanilla | `qwen3_8b_b7-vanilla-lora.yaml` | `relay_enable: false` |
| B rollout-only | `qwen3_8b_b7-rollout-lora.yaml` | `relay_enable: true`, `relay_rollout_only: true` |
| C relay | `qwen3_8b_b7-relay-lora.yaml` | `relay_enable: true`; `additional_target: relay_layer_norm` |

`qwen3_8b_b7-relay.yaml` is an alias of arm C (the old `finetuning_type: full` was fixed).

### Launch (human)

1. Put the 8B weights (released I-DLM-8B / Ram baseline checkpoint) into a directory that also contains the b7 `modeling_sdar.py` / `config.json` (`block_size: 7`), **or** point `MODEL_DIR` at a copy of `training/model/Qwen3-8B-b7-allmasked` after copying `.safetensors` in.
2. On a GPU node (`srun`/`sbatch`, A100):

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=...
export HUGGING_FACE_HUB_TOKEN=...
export MODEL_DIR=/path/to/Qwen3-8B-b7-allmasked-WITH-WEIGHTS

cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/I-DLM/training
bash run_train_b7_lora.sh vanilla   # arm A
bash run_train_b7_lora.sh rollout   # arm B  (~2× step time vs A)
bash run_train_b7_lora.sh relay     # arm C  (~2× step time vs A)
```

Equivalent `torchrun` (from `training/llama_factory_sdar`):

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
torchrun --nnodes=1 --nproc_per_node=1 --master_addr=localhost --master_port=29500 \
  src/llamafactory/launcher.py ./examples/train_idlm/qwen3_8b_b7-vanilla-lora.yaml \
  model_name_or_path="$MODEL_DIR"
```

`per_device_train_batch_size` must stay 1 (SDAR `[noisy\|clean]` concat). Scale with `NPROC` / `gradient_accumulation_steps`.

### Same-steps vs same-compute

- **Same steps:** all arms `max_steps: 500`. B/C do two forwards+backwards per step → ~2× wall time and ~2× tokens-seen-through-the-model vs A.
- **Same compute:** compare A at 500 steps to B/C at 250 steps (or A at 1000 to B/C at 500). Prefer reporting both.

### Training-time metrics to compare

Logged every `logging_steps`:

- `relay/mean_accept_rate` — fraction of verifiable drafts accepted (before first reject is *not* this; this is raw accept bits on valid positions)
- `relay/mean_a_b` — mean committed prefix length in `[0, B]`
- `relay/frac_all_accept`, `relay/frac_zero_accept`
- `relay/pass2_supervised` — fraction of packed slots with a real label
- plus the existing `train/task_loss`, `train/clean_ce_loss`, `train/combined_loss`

Carried-suffix acceptance (the 21.4% / 10% frozen-model Slack number) is an **inference** metric and is not computed here.

## Design notes / inference cross-check

`o_b = s_b + a_b + 2` is used uniformly for reject and all-accept (plan §4.3). Overlapping canvases are packed by slot; RoPE uses drafting position ids; the mask is slot-indexed so two slots with the same position do not attend each other.

I-DLM inference (`inference/sglang/.../idlm_blockN.py`) was consulted; no 1-off vs that code was introduced. If a later inference-side audit finds a discrepancy, update `test_anchor_formula.py` and this paragraph.

Vanilla GPU training still uses `create_block_mask` + flex attention. CPU / tests materialize a dense bool mask and take SDPA (`is_causal=False`). Fused Triton CE is CUDA-only; CPU uses standard CE / `1/p_mask`.

## Known limitations (do not block)

- **8B weights path** is a placeholder. Do not download weights from this session; a human must point `MODEL_DIR` at the Unity checkpoint Ram used.
- **No 8B run was launched.**
- **SGLang relay injection** (real MATH-500 / HumanEval TPF) is out of scope.
- **BPTT** through carried `h` is Phase 2. The interface already accepts a non-detached tensor; the trainer detaches.
- **LoRA is not mask-gated.** All arms use plain `lora_target: all` for consistency. Gating is an inference-side later question.
- **GPU flex + Triton verify** was not executed here (login node). Re-run `pytest` on a GPU allocation to clear the skip.
- `training/model/Qwen3-8B-b7-allmasked/` is a copy of b3 + relay edits. Do not copy those edits back into b1/b2/b3 (vanilla arm stays bitwise-old modeling aside from using the b7 config's `block_size: 7`).

## Deferred

- Inference-side relay injection
- BPTT / injection-depth ablation / relaxed-acceptance jitter
- Resolving the exact baseline checkpoint on Unity
