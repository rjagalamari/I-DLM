#!/bin/bash
#SBATCH --job-name=relay-2step-res
#SBATCH --partition=superpod-a100
#SBATCH --qos=long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail
echo "started $(date) on $(hostname)"
nvidia-smi

export IDLM=/work/pi_mccallum_umass_edu/rjagalamari_umass_edu/idlm
export R2=$IDLM/all_ladder_experiments/ladder_5_hidden_relay/relay-2step
LF=$R2/training/llama_factory_sdar

grep -q "api.wandb.ai" "$HOME/.netrc" || { echo "no wandb creds"; exit 1; }

export WANDB_RUN_ID=1q5z6xg7
export WANDB_RESUME=allow
export WANDB_PROJECT=IDLM
export WANDB_DIR=$HOME/wandb
mkdir -p "$WANDB_DIR"

module load cuda/12.4.1
source $IDLM/.venv_idlm/bin/activate
export PYTHONPATH="$LF/src:${PYTHONPATH:-}"
export HF_HOME=$IDLM/hf_cache
export DS_ACCELERATOR=cuda
export CC=gcc

CACHE="/tmp/${USER}/idlm_${SLURM_JOB_ID}"
mkdir -p "$CACHE"/{triton,inductor,cuda}
export TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export CUDA_CACHE_PATH="$CACHE/cuda"

rm -rf $IDLM/hf_cache/modules/transformers_modules/I-DLM-8B-N8-relay


CKPT=$(ls -dv $R2/runs/relay_2step/checkpoint-* | tail -1)
[ -n "$CKPT" ] || { echo "NO CHECKPOINT"; exit 1; }
echo "resuming from $CKPT"
cd $LF
python src/llamafactory/launcher.py \
    examples/train_idlm/lora_with_hidden_relay.yaml \
    save_steps=10 \
    resume_from_checkpoint="$CKPT"

echo "finished $(date)"
