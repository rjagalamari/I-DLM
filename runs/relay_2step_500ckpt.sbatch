#!/bin/bash
#SBATCH --job-name=relay_2step_500ckpt
#SBATCH --partition=superpod-a100
#SBATCH --qos=long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --output=relay_2step_500ckpt-%j.out
#SBATCH --error=relay_2step_500ckpt-%j.err

set -eo pipefail
echo "started $(date) on $(hostname)"

export IDLM=/work/pi_mccallum_umass_edu/rjagalamari_umass_edu/idlm
export L5=$IDLM/all_ladder_experiments/ladder_5_hidden_relay

TAG=relay_2step_500ckpt
ROOT=$L5/relay-2step
RUNS=$L5/relay-2step/runs
LORA=$L5/relay-2step/runs/relay_2step/checkpoint-500

module load cuda/12.4.1
source $IDLM/.venv_idlm_infer/bin/activate
export HF_HOME=$IDLM/hf_cache
cd "$ROOT"
mkdir -p inference/results/$TAG
echo "adapter: $LORA"

# ==================== MATH-500 ====================
python -m sglang.launch_server \
  --model-path jianstreet/I-DLM-8B-N8-full-autobalance \
  --lora-paths sdar=$LORA \
  --dllm-algorithm IDLMBlockN \
  --dllm-algorithm-config inference/configs/idlm_blockN8_config.yaml \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.85 --max-running-requests 8 \
  --attention-backend flashinfer --dtype bfloat16 \
  --disable-cuda-graph \
  --random-seed 358866389 --port 30000 \
  > $RUNS/server_${TAG}_math500.log 2>&1 &
SRV_PID=$!
for i in $(seq 1 120); do
    curl -s http://localhost:30000/health > /dev/null 2>&1 && break
    sleep 10
done

echo "--- relay load check ---"
grep -E "\[RELAY\]" $RUNS/server_${TAG}_math500.log || echo "NO [RELAY] LINES"

python inference/eval/eval_math500.py \
  --ports 30000 --max-tokens 32768 --timeout 10800 \
  --output-dir inference/results/$TAG --tag ${TAG}_math500

echo "--- relay probe ---"
grep -E "\[RELAY-PROBE\]" $RUNS/server_${TAG}_math500.log || echo "NO PROBE LINE"
echo "--- stats math500 ---"
grep -oE "accept=[0-9.]+%|tok/fwd: *[0-9.]+" $RUNS/server_${TAG}_math500.log | tail -4 || true

kill $SRV_PID 2>/dev/null || true
sleep 15

# ==================== HumanEval ====================
python -m sglang.launch_server \
  --model-path jianstreet/I-DLM-8B-N8-full-autobalance \
  --lora-paths sdar=$LORA \
  --dllm-algorithm IDLMBlockN \
  --dllm-algorithm-config inference/configs/idlm_blockN8_config.yaml \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.85 --max-running-requests 8 \
  --attention-backend flashinfer --dtype bfloat16 \
  --disable-cuda-graph \
  --random-seed 358866389 --port 30000 \
  > $RUNS/server_${TAG}_humaneval.log 2>&1 &
SRV_PID=$!
for i in $(seq 1 120); do
    curl -s http://localhost:30000/health > /dev/null 2>&1 && break
    sleep 10
done

python inference/eval/eval_humaneval.py \
  --ports 30000 --max-tokens 32768 --timeout 10800 \
  --output-dir inference/results/$TAG --tag ${TAG}_humaneval

echo "--- stats humaneval ---"
grep -oE "accept=[0-9.]+%|tok/fwd: *[0-9.]+" $RUNS/server_${TAG}_humaneval.log | tail -4 || true

kill $SRV_PID 2>/dev/null || true
echo "finished $(date)"
