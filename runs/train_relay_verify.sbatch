#!/bin/bash
#SBATCH --job-name=relay_brandnew_verify
#SBATCH --partition=superpod-a100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=relay_brandnew_verify-%j.out
#SBATCH --error=relay_brandnew_verify-%j.err

set -eo pipefail
echo "started $(date) on $(hostname)"
nvidia-smi

export IDLM=/work/pi_mccallum_umass_edu/rjagalamari_umass_edu/idlm
export ROOT=$IDLM/all_ladder_experiments/ladder_5_hidden_relay/new_5_experiments/relay_brandnew_model_verify
export LF=$ROOT/training/llama_factory_sdar

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

rm -rf $IDLM/hf_cache/modules/transformers_modules/Qwen3-8B-b7-allmasked
cd $LF
TORCHELASTIC_ERROR_FILE=$ROOT/torch_error.json torchrun --redirects 3 --tee 3 --nproc_per_node 2 --master_port 29517 \
    src/llamafactory/launcher.py \
    examples/train_idlm/qwen3_8b_b7-relay.yaml \
    run_name=relay_brandnew_verify

echo "finished $(date)"
