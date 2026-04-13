# Training

This directory contains the training code for I-DLM, built on a modified [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) framework and [SDAR](https://github.com/JetAstra/SDAR) codebase.

## Overview

I-DLM training converts a pretrained AR model (e.g., Qwen3-8B) into a diffusion language model that preserves introspective consistency. The key training procedure:

1. Construct input as `[MASK, ..., MASK | clean_x1, ..., clean_xL]` (all-masked + clean concatenation)
2. Apply strict causal attention uniformly across both regions
3. Compute CE loss on all non-padding positions (both noisy and clean)
4. Add weighted CE on the clean region with Dream-shift-aligned labels

The combined loss: `L = CE_noisy + alpha * CE_clean`

## Environment Setup

```bash
conda create -n idlm python=3.10
conda activate idlm

# Install PyTorch >= 2.5.0 (flex_attention required; match CUDA to your driver)
# See https://pytorch.org for the correct command for your system
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install training dependencies (includes deepspeed, wandb, etc.)
cd llama_factory_sdar
pip install -r requirements.txt

# Install FlashAttention (required for fused RMS norm in SDAR model)
pip install flash-attn --no-build-isolation
```

## Directory Structure

```
training/
├── run_train_b1-allmasked_idlm_sample.sh        # Block-1 training launch script
├── run_train_b2-allmasked_idlm_sample.sh        # Block-2 training launch script
├── run_train_b3-allmasked_idlm_sample.sh        # Block-3 training launch script
├── model/
│   ├── Qwen3-8B-b1-allmasked/                # SDAR model config, block_size=1
│   ├── Qwen3-8B-b2-allmasked/                # SDAR model config, block_size=2
│   └── Qwen3-8B-b3-allmasked/                # SDAR model config, block_size=3
└── llama_factory_sdar/
    ├── requirements.txt                       # Python dependencies
    ├── examples/
    │   ├── train_idlm/
    │   │   ├── qwen3_8b_b1-allmasked_sample.yaml  # Block-1 config
    │   │   ├── qwen3_8b_b2-allmasked_sample.yaml  # Block-2 config
    │   │   └── qwen3_8b_b3-allmasked_sample.yaml  # Block-3 config (auto-balanced)
    │   └── deepspeed/                         # DeepSpeed ZeRO configs
    └── src/llamafactory/                      # Core training code
```

## Model Configs

The `model/` directory contains SDAR model configurations (architecture, tokenizer, custom modeling code) **without weights**. You need to either:
- Download weights from HuggingFace (links in main README)
- Or place your own pretrained SDAR checkpoint `.safetensors` files in the model directory

Each model config specifies `block_size` in `config.json`:
- `Qwen3-8B-b1-allmasked`: `block_size=1` (1 mask per block, generates 2 tokens per diffusion step)
- `Qwen3-8B-b2-allmasked`: `block_size=2` (2 masks per block, generates 3 tokens per diffusion step)
- `Qwen3-8B-b3-allmasked`: `block_size=3` (3 masks per block, generates 4 tokens per diffusion step)

> **Note**: `block_size` in `config.json` is the SDAR model architecture parameter (number of mask tokens per block). `block_length` in the YAML config is the corresponding data loader parameter — both should match for correct training.

> **Note**: The block-1 and block-2 sample configs use fixed `ce_alpha` weighting. The block-3 config enables `loss_auto_balance: true`, which automatically scales the clean CE loss to match the task loss magnitude — this is recommended for larger block sizes.

## Usage

### 1. Configure

Edit the YAML config to set your paths:

**Block-1** (`examples/train_idlm/qwen3_8b_b1-allmasked_sample.yaml`):
```yaml
model_name_or_path: <path-to-Qwen3-8B-b1-allmasked-with-weights>
dataset: open_thoughts3_sample   # sample dataset; replace with your own
output_dir: <output-dir>
```

**Block-2** (`examples/train_idlm/qwen3_8b_b2-allmasked_sample.yaml`):
```yaml
model_name_or_path: <path-to-Qwen3-8B-b2-allmasked-with-weights>
dataset: open_thoughts3_sample   # sample dataset; replace with your own
output_dir: <output-dir>
```

> **Note**: The configs ship with `open_thoughts3_sample` ([open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M), 1.2M reasoning traces across math, code, and science) as a **sample dataset for demonstration only** — this is not the dataset used in the paper. To use your own dataset, add an entry to `data/dataset_info.json` and update the `dataset` field in the YAML config.

### 2. Set credentials

Export required environment variables before launching:
```bash
export WANDB_API_KEY="your-key"
export WANDB_ENTITY="your-entity"
export HUGGING_FACE_HUB_TOKEN="your-token"
```

### 3. Launch training

```bash
# Block-1 (1 mask token, ready for N=2 ISD stride)
bash run_train_b1-allmasked_idlm_sample.sh

# Block-2 (2 mask tokens, ready for N=3 ISD stride)
bash run_train_b2-allmasked_idlm_sample.sh

# Block-3 (3 mask tokens, ready for N=4 ISD stride, auto-balanced loss)
bash run_train_b3-allmasked_idlm_sample.sh
```

All scripts launch 8-GPU distributed training via `torchrun` with DeepSpeed ZeRO-2.

## Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `model_name_or_path` | Path to SDAR model with weights | - |
| `block_length` | Number of tokens per diffusion block | 1 |
| `ce_alpha` | Weight for clean-region CE loss (alpha in `CE_noisy + alpha * CE_clean`) | 0.2 |
| `idlm_loss_type` | Loss type (`ar` = CE on clean tokens with Dream shift) | `ar` |
| `loss_auto_balance` | Auto-scale clean CE loss to match task loss magnitude | false |
| `cutoff_len` | Maximum sequence length | 4096 |
| `pretokenized` | Whether dataset is pre-tokenized | false |
| `learning_rate` | Learning rate | 1e-5 |
| `per_device_train_batch_size` | Batch size per GPU | 1 |
| `gradient_accumulation_steps` | Gradient accumulation | 4 |

## Training Details

- **Hardware**: 8x H100 GPUs
- **Precision**: bf16
- **Optimizer**: AdamW with cosine scheduler, 3% warmup
- **Effective batch size**: 32 (1 per device x 4 accumulation x 8 GPUs)
- **DeepSpeed**: ZeRO Stage 2

> **Note**: Only `per_device_train_batch_size=1` is supported. The SDAR model's internal `[noisy | clean]` concatenation assumes a single sample per device. Use `gradient_accumulation_steps` to increase effective batch size.
