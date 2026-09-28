#!/bin/bash
#SBATCH --job-name=relay_debug
#SBATCH --partition=superpod-a100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=relay_debug-%j.out
#SBATCH --error=relay_debug-%j.err

set -eo pipefail
export IDLM=/work/pi_mccallum_umass_edu/rjagalamari_umass_edu/idlm
export ROOT=$IDLM/all_ladder_experiments/ladder_5_hidden_relay/new_5_experiments/relay_brandnew_model_verify
export LF=$ROOT/training/llama_factory_sdar

module load cuda/12.4.1
source $IDLM/.venv_idlm/bin/activate
export PYTHONPATH="$LF/src:${PYTHONPATH:-}"
export HF_HOME=$IDLM/hf_cache
export DS_ACCELERATOR=cuda
export CC=gcc
export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1

rm -rf $IDLM/hf_cache/modules/transformers_modules/Qwen3-8B-b7-allmasked
cd $LF
python -u src/llamafactory/launcher.py \
    examples/train_idlm/qwen3_8b_b7-relay.yaml \
    deepspeed=null \
    max_steps=3 \
    max_samples=96 \
    report_to=none \
    run_name=relay_debug
