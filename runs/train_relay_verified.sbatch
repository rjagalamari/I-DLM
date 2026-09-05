#!/bin/bash
#SBATCH --job-name=relay-verified
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
export VN=$IDLM/all_ladder_experiments/ladder_5_hidden_relay/verified_new
LF=$VN/training/llama_factory_sdar

grep -q "api.wandb.ai" "$HOME/.netrc" || { echo "no wandb creds"; exit 1; }
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

# model_n8/modeling_sdar.py is edited; a stale cached copy would shadow it
rm -rf $HF_HOME/modules/transformers_modules/model_n8

cd $LF
python src/llamafactory/launcher.py \
    examples/train_idlm/relay_verified.yaml

echo "finished $(date)"
