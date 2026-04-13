# I-DLM Inference

Serving system for I-DLM with Introspective Strided Decoding (ISD), built on [SGLang](https://github.com/sgl-project/sglang). Supports CUDA graphs, continuous batching, paged KV cache, and OpenAI-compatible API.

## Installation

### Prerequisites

- Python 3.12
- CUDA 12.9
- PyTorch 2.9.1
- GPU with compute capability ≥ 8.0 (A100 / H100 recommended)

### Install

```bash
cd inference
bash install.sh
```

Or with conda (recommended, handles Python + PyTorch automatically):

```bash
bash setup_env.sh   # Creates 'idlm' conda env and installs everything
conda activate idlm
```

## Quick Start

### Launch Server

```bash
python -m sglang.launch_server \
  --model-path yifanyu/I-DLM-8B \
  --dllm-algorithm IDLMBlockN \
  --dllm-algorithm-config configs/idlm_blockN4_config.yaml \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.85 --max-running-requests 32 \
  --attention-backend flashinfer --dtype bfloat16
```

### Send Requests

```python
import requests

response = requests.post("http://localhost:30000/v1/chat/completions", json={
    "model": "default",
    "messages": [{"role": "user", "content": "What is 2+3?"}],
    "max_tokens": 512,
    "temperature": 1.0,
})
print(response.json()["choices"][0]["message"]["content"])
```

### Multi-GPU Serving (8x TP=1)

```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
    --model-path yifanyu/I-DLM-8B \
    --dllm-algorithm IDLMBlockN \
    --dllm-algorithm-config configs/idlm_blockN4_config.yaml \
    --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dtype bfloat16 \
    --port $((30000+i)) &
done
```

### LoRA Serving

Use a standard Qwen3-8B base model with a DLLM LoRA adapter:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --lora-paths idlm=yifanyu/I-DLM-8B-lora-r128 \
  --dllm-algorithm IDLMBlockN \
  --dllm-algorithm-config configs/idlm_blockN4_config.yaml \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.85 --max-running-requests 32 \
  --attention-backend flashinfer --dtype bfloat16
```

Send requests using the LoRA adapter name as the model:

```python
response = requests.post("http://localhost:30000/v1/chat/completions", json={
    "model": "idlm",  # LoRA adapter name
    "messages": [{"role": "user", "content": "What is 2+3?"}],
    "max_tokens": 512,
})
```

## Available Models

### Full Checkpoints

| Model | HuggingFace ID | N | Recommended Config |
|-------|---------------|---|-------------------|
| I-DLM-8B | [`yifanyu/I-DLM-8B`](https://huggingface.co/yifanyu/I-DLM-8B) | 4 | `idlm_blockN4_config.yaml` |
| I-DLM-32B | [`yifanyu/I-DLM-32B`](https://huggingface.co/yifanyu/I-DLM-32B) | 4 | `idlm_blockN4_config.yaml` |

### LoRA Adapters

| Adapter | HuggingFace ID | Base Model | N |
|---------|---------------|------------|---|
| I-DLM-8B LoRA | [`yifanyu/I-DLM-8B-lora-r128`](https://huggingface.co/yifanyu/I-DLM-8B-lora-r128) | `Qwen/Qwen3-8B` | 4 |

## Algorithm Configurations

| Config | N | block_size |
|--------|---|-----------|
| `idlm_blockN2_config.yaml` | 2 | 3 |
| `idlm_blockN3_config.yaml` | 3 | 5 |
| `idlm_blockN4_config.yaml` | 4 | 7 |
| `idlm_blockN5_config.yaml` | 5 | 9 |
| `idlm_blockN8_config.yaml` | 8 | 15 |
| `idlm_blockN16_config.yaml` | 16 | 31 |

**Key parameters** in config files:

```yaml
block_size: 7           # Total tokens per forward (2*N - 1)
gen_block_size: 4       # New tokens generated per step (N)
confidence_threshold: 0.0  # Acceptance threshold (0 = accept all)
temperature: 1.0        # Sampling temperature
top_k: 50
top_p: 0.95
use_spec_verify: true   # Enable speculative verification
```

## Benchmarks

### Throughput on MATH-500 (1× H100 80GB, tok/s)

I-DLM-8B N=4, per-request TPS vs AR baseline.
Settings: bf16, burst mode, max_tokens=2048.

| Concurrency | AR (Qwen3-8B) | **I-DLM-8B N=4 (ours)** | Speedup |
|-------------|--------------|--------------------------|---------|
| 1 | 142 | **326** | **2.30×** |
| 2 | 132 | **275** | **2.08×** |
| 4 | 130 | **305** | **2.35×** |
| 8 | 119 | **256** | **2.15×** |
| 16 | 123 | **237** | **1.93×** |
| 32 | 111 | **201** | **1.81×** |
| 64 | 93 | **125** | **1.34×** |

### Quality (I-DLM-8B)

**Knowledge & Reasoning**

| Benchmark | I-DLM-8B |
|-----------|----------|
| ARC-C | 95.8 |
| MMLU | 82.4 |
| MMLU-Pro | 73.1 |
| GPQA-Diamond | 55.6 |
| GPQA | 54.9 |

**Math**

| Benchmark | I-DLM-8B |
|-----------|----------|
| GSM8K | 95.0 |
| MATH-500 | 96.8 |
| MathBench | 89.1 |
| AIME-24 | 69.6 |
| AIME-25 | 60.8 |

**Code**

| Benchmark | I-DLM-8B |
|-----------|----------|
| HumanEval | 93.3 |
| MBPP | 92.2 |
| LCB-v6 | 45.7 |

**Instruction Following**

| Benchmark | I-DLM-8B |
|-----------|----------|
| IFEval | 84.7 |

## Evaluation

Benchmark scripts are in `eval/`:

```bash
# Example: evaluate MATH-500
python eval/eval_math500.py \
  --ports 30000 30001 30002 30003 30004 30005 30006 30007 \
  --max-tokens 32768

# Example: evaluate GSM8K
python eval/eval_gsm8k.py --ports 30000 --max-tokens 32768
```

Available benchmarks: ARC-C, MMLU, MMLU-Pro, GPQA (Diamond/Main), IFEval, GSM8K, MATH-500, MathBench, AIME-24/25, HumanEval, MBPP, LiveCodeBench-v6.

## Directory Structure

```
inference/
├── README.md               # This file
├── install.sh              # Install bundled SGLang with I-DLM support
├── setup_env.sh            # Full env setup (conda + PyTorch + install)
├── sglang/                 # SGLang with ISD algorithm (bundled)
├── configs/                # ISD algorithm configurations
│   ├── idlm_blockN2_config.yaml
│   ├── idlm_blockN3_config.yaml
│   ├── idlm_blockN4_config.yaml
│   ├── idlm_blockN5_config.yaml
│   ├── idlm_blockN8_config.yaml
│   └── idlm_blockN16_config.yaml
└── eval/                   # Benchmark evaluation scripts
```

## License

BSD 3-Clause License. See [LICENSE](../LICENSE) for details.
