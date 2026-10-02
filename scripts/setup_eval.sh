#!/usr/bin/env bash
# Install the upstream evaluation dependency into a dedicated conda environment.
# Set VLMEVALKIT_DIR to reuse an existing checkout without changing its revision.
# TORCH_INDEX_URL defaults to CUDA 12.4; choose a wheel channel for your driver.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EGOTOOLS_EVAL_ENV:-egotools_eval}"
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-${REPO_ROOT}/third_party/VLMEvalKit}"
VLMEVALKIT_REV="${VLMEVALKIT_REV:-e7d64cfa8f6036e1d00e21522aaee0102544ea25}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; see docs/EVALUATION.md for installation into an existing environment." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required to clone VLMEvalKit." >&2
  exit 1
fi
if [[ ! -e "${VLMEVALKIT_DIR}" ]]; then
  mkdir -p "$(dirname "${VLMEVALKIT_DIR}")"
  git clone https://github.com/open-compass/VLMEvalKit.git "${VLMEVALKIT_DIR}"
  git -C "${VLMEVALKIT_DIR}" checkout "${VLMEVALKIT_REV}"
elif [[ ! -f "${VLMEVALKIT_DIR}/run.py" ]]; then
  echo "VLMEVALKIT_DIR does not contain run.py: ${VLMEVALKIT_DIR}" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
  conda env create -n "${ENV_NAME}" -f "${REPO_ROOT}/configs/evaluation/environment.yml"
fi
conda activate "${ENV_NAME}"
# Conda otherwise permits ~/.local Python packages to satisfy requirements and
# shadow this environment's packages. Keep installs and future activations
# independent of that unrelated user site.
conda env config vars set -n "${ENV_NAME}" PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
python -m pip install "torch==2.6.0" "torchvision==0.21.0" --index-url "${TORCH_INDEX_URL}"
python -m pip install -c "${REPO_ROOT}/configs/evaluation/constraints.txt" -e "${VLMEVALKIT_DIR}" -e "${REPO_ROOT}[eval]"
python "${SCRIPT_DIR}/verify_eval.py"
echo "Activate with: conda activate ${ENV_NAME}"
echo "VLMEvalKit checkout: ${VLMEVALKIT_DIR}"
