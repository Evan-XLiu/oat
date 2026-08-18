#!/usr/bin/env bash

# Shared paths for OAT development on NUS Vanda.
# This file intentionally does not set a PBS billing project. Until `hpc project`
# lists a real CFP project, jobs should use the automatically assigned personal
# allocation by omitting `#PBS -P`.

export OAT_WORKSPACE="/scratch/${USER}/bimanual_tokenizer"
export OAT_REPO="${OAT_WORKSPACE}/oat"
export OAT_VENV="${OAT_WORKSPACE}/.venv"
export OAT_IMAGE="/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif"

# Keep caches and generated data off the 40 GB home directory.
export UV_CACHE_DIR="${OAT_WORKSPACE}/.cache/uv"
export HF_HOME="${OAT_WORKSPACE}/.cache/huggingface"
export XDG_CACHE_HOME="${OAT_WORKSPACE}/.cache/xdg"
export WANDB_DIR="${OAT_WORKSPACE}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p \
  "${UV_CACHE_DIR}" \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" \
  "${OAT_WORKSPACE}/logs"
