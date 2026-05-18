#!/bin/bash
# DeltaAI Sinii replication — Qwen2.5-1.5B + GSM8K, activation steering.
# Submit with: sbatch scripts/sinii-replicate.sbatch
#
# Assumes this branch (deltaai-replicate) is cloned at:
#   /projects/bgye/$USER/steering-reasoning/
# Mounts the repo root as /proj inside apptainer, and /work/nvme/bgye/$USER as /nvme.
#
# Pre-reqs:
#   - apptainer container env already validated (interactive Block 2)
#   - bin/helpers/modify_bias_sitepackages.sh already run against the installed vllm
#   - WANDB_API_KEY, HF_TOKEN, WANDB_ENTITY exported in the submitting shell
#   - Dry run (Block 4) passed

#SBATCH --partition=ghx4
#SBATCH --account=bgye-dtai-gh
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --time=08:00:00
#SBATCH --job-name=sinii-replicate
#SBATCH --output=/projects/bgye/%u/steering-reasoning/logs/%j.out

set -euo pipefail

# Required env from submitting shell
: "${WANDB_API_KEY:?WANDB_API_KEY must be exported before sbatch}"
: "${HF_TOKEN:?HF_TOKEN must be exported before sbatch}"
: "${WANDB_ENTITY:?WANDB_ENTITY must be exported before sbatch}"

REPO=/projects/bgye/$USER/steering-reasoning
NVME=/work/nvme/bgye/$USER
CONTAINER=/sw/user/NGC_containers/pytorch_24.09-py3.sif

mkdir -p "$REPO/logs"

apptainer exec --nv \
  --bind "$REPO":/proj \
  --bind "$NVME":/nvme \
  --env WANDB_API_KEY="$WANDB_API_KEY" \
  --env HF_TOKEN="$HF_TOKEN" \
  --env WANDB_ENTITY="$WANDB_ENTITY" \
  --env HF_HOME=/nvme/hf_cache \
  --env TRANSFORMERS_CACHE=/nvme/hf_cache \
  "$CONTAINER" \
  bash -c '
    set -euo pipefail
    cd /proj

    # Sed-patch entity placeholder into a temp config so the tracked file stays clean.
    SRC=/proj/configs/train/rl/qwen2.5-1.5b/gsm8k/steering-deltaai.yml
    TMPCFG=$(mktemp --suffix=.yml)
    sed "s|entity: \"PATCH_ME_AT_RUNTIME\"|entity: \"$WANDB_ENTITY\"|" "$SRC" > "$TMPCFG"
    echo "Patched config -> $TMPCFG"

    # Single-node Ray head for the rollout coordinator.
    ray start --head --port=6379

    export PYTHONPATH=/proj:$PYTHONPATH

    python -m steering_reasoning.train.rl.main \
      --config_path "$TMPCFG" \
      --seed 0 \
      --is_dist false
  '
