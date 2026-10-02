#!/usr/bin/env bash
# Install the training stack of the paper run into the active Python 3.10 environment:
# pinned dependencies, MS-Swift at 85ce1be with the EgoTools video-entry patch, and FlashAttention.
# Set MS_SWIFT_DIR to reuse an existing checkout; it must be at MS_SWIFT_REV.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MS_SWIFT_DIR="${MS_SWIFT_DIR:-${REPO_ROOT}/third_party/ms-swift}"
MS_SWIFT_REV="${MS_SWIFT_REV:-85ce1be91aad75b1c86424190d4efac1b9896cb3}"
PATCH="${REPO_ROOT}/configs/training/ms-swift-85ce1be-qwen-video-window.patch"

if [[ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
  echo "Activate a Python 3.10 virtual or conda environment first; see docs/TRAINING.md." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required to clone MS-Swift." >&2
  exit 1
fi
if [[ ! -e "${MS_SWIFT_DIR}" ]]; then
  mkdir -p "$(dirname "${MS_SWIFT_DIR}")"
  git clone https://github.com/modelscope/ms-swift.git "${MS_SWIFT_DIR}"
  git -C "${MS_SWIFT_DIR}" checkout "${MS_SWIFT_REV}"
fi
if [[ "$(git -C "${MS_SWIFT_DIR}" rev-parse HEAD)" != "${MS_SWIFT_REV}" ]]; then
  echo "${MS_SWIFT_DIR} is not at ${MS_SWIFT_REV}; the paper run used that revision." >&2
  exit 1
fi
# Apply once; an already patched checkout reverse-applies cleanly.
if git -C "${MS_SWIFT_DIR}" apply --check "${PATCH}" 2>/dev/null; then
  git -C "${MS_SWIFT_DIR}" apply "${PATCH}"
elif ! git -C "${MS_SWIFT_DIR}" apply --check --reverse "${PATCH}" 2>/dev/null; then
  echo "Cannot apply ${PATCH} to ${MS_SWIFT_DIR}; check out ${MS_SWIFT_REV} first." >&2
  exit 1
fi

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -r "${REPO_ROOT}/configs/training/requirements-cu124.txt"
python -m pip install -e "${MS_SWIFT_DIR}"
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e "${REPO_ROOT}"
python - <<'PY'
import inspect

from swift.template.templates.qwen import Qwen2VLTemplate

if "isinstance(video, dict)" not in inspect.getsource(Qwen2VLTemplate.replace_tag):
    raise SystemExit("MS-Swift is installed without the EgoTools video-entry patch")
print("MS-Swift video-entry patch: OK")
PY
echo "MS-Swift checkout: ${MS_SWIFT_DIR} @ ${MS_SWIFT_REV} + configs/training/$(basename "${PATCH}")"
