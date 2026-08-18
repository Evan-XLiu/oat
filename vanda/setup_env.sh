#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [[ "$(hostname)" != gn-a40-* && "$(hostname)" != cn-* ]]; then
  echo "Run this script inside a Vanda interactive compute job, not on the login node." >&2
  echo "Example: qsub -I -l select=1:ngpus=1 -l walltime=02:00:00" >&2
  exit 1
fi

module load msingularity

singularity exec --nv -e \
  --bind "${OAT_WORKSPACE}:${OAT_WORKSPACE}" \
  "${OAT_IMAGE}" \
  bash -lc "
    set -euo pipefail
    cd '${OAT_REPO}'
    python --version
    python -m venv '${OAT_VENV}'
    '${OAT_VENV}/bin/python' -m pip install --upgrade pip uv
    VIRTUAL_ENV='${OAT_VENV}' \
      UV_CACHE_DIR='${UV_CACHE_DIR}' \
      '${OAT_VENV}/bin/uv' sync --frozen --active
    '${OAT_VENV}/bin/python' -m pip install --editable . --no-deps
  "

echo "Environment created at ${OAT_VENV}"
echo "Submit vanda/preflight.pbs before starting a long training run."
