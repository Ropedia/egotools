#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: scripts/train.sh --dataset TRAIN.jsonl --output-dir DIR [options]

Train the Qwen3-VL-8B language model with the EgoTools paper recipe.

Options:
  --dataset PATH     Local ms-swift JSONL (or set DATASET).
  --model ID_OR_PATH Base model; default: Qwen/Qwen3-VL-8B-Instruct (or MODEL).
  --output-dir DIR   Training output directory (or set OUTPUT_DIR).
  --dry-run          Print the environment and command without running training.
                    Does not require ms-swift, GPUs, or downloaded data/models.
  -h, --help         Show this help.
  -- ARGS...        Append explicit ms-swift overrides for another experiment.

Runtime controls: CUDA_VISIBLE_DEVICES, NPROC_PER_NODE, MASTER_PORT,
DATASET_NUM_PROC, DATALOADER_NUM_WORKERS, SAVE_STEPS, SAVE_TOTAL_LIMIT,
SWIFT_PYTHON (interpreter used to check the MS-Swift patch; default python).
See docs/TRAINING.md for dependencies, video settings, and dataset availability.
HELP
}

fail() { echo "Error: $*" >&2; exit 2; }

DATASET="${DATASET:-}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
dry_run=false
extra_args=()

while (($#)); do
  case "$1" in
    --dataset|--model|--output-dir)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
      case "$1" in
        --dataset) DATASET="$2" ;;
        --model) MODEL="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
      esac
      shift 2
      ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra_args=("$@"); break ;;
    *) fail "unknown option: $1 (use --help, or -- before ms-swift arguments)" ;;
  esac
done

[[ -n "$DATASET" ]] || fail "provide --dataset or DATASET"
[[ -n "$OUTPUT_DIR" ]] || fail "provide --output-dir or OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29621}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ms-swift passes these settings to qwen-vl-utils. Both bounds are needed:
# FPS_MAX_FRAMES alone would leave shorter videos at fewer than 64 samples.
# The decoder still limits sampling to the number of frames in the source.
export FPS="${FPS:-2}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-64}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}"
# The paper run fixed each video frame at 128 visual tokens (min = max).
export VIDEO_MIN_TOKEN_NUM="${VIDEO_MIN_TOKEN_NUM:-128}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

swift_bin="${SWIFT_BIN:-swift}"
command_args=(
  "$swift_bin" sft
  --model "$MODEL"
  --model_type qwen3_vl
  --use_hf true
  --dataset "$DATASET"
  --output_dir "$OUTPUT_DIR"
  --split_dataset_ratio 0
  --load_from_cache_file false
  --dataset_num_proc "${DATASET_NUM_PROC:-16}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}"
  --tuner_type full
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --torch_dtype bfloat16
  --attn_impl flash_attn
  --gradient_checkpointing true
  --vit_gradient_checkpointing false
  --max_length 8192
  --truncation_strategy delete
  --num_train_epochs 1
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 16
  --optim adamw_torch
  --learning_rate 2.3e-6
  --lr_scheduler_type constant
  --warmup_ratio 0
  --weight_decay 0.1
  --adam_beta1 0.9
  --adam_beta2 0.95
  --deepspeed zero3
  --seed 42
  --data_seed 42
  --logging_steps 5
  --save_steps "${SAVE_STEPS:-300}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-4}"
  --save_only_model true
  --report_to tensorboard
)
# Bash 3.2 treats expansion of an empty array as unbound under `set -u`.
if ((${#extra_args[@]})); then
  command_args+=("${extra_args[@]}")
fi

print_command() {
  local name
  printf 'env'
  for name in CUDA_VISIBLE_DEVICES NPROC_PER_NODE MASTER_PORT OMP_NUM_THREADS \
    TOKENIZERS_PARALLELISM PYTORCH_CUDA_ALLOC_CONF FPS FPS_MIN_FRAMES \
    FPS_MAX_FRAMES VIDEO_MIN_TOKEN_NUM VIDEO_MAX_TOKEN_NUM IMAGE_MAX_TOKEN_NUM FORCE_QWENVL_VIDEO_READER; do
    printf ' %q' "$name=${!name}"
  done
  printf ' %q' "${command_args[@]}"
  printf '\n'
}

if "$dry_run"; then
  print_command
  exit 0
fi

[[ -f "$DATASET" ]] || fail "dataset does not exist: $DATASET"
command -v "$swift_bin" >/dev/null 2>&1 || fail "ms-swift is not installed; see docs/TRAINING.md"
# Dict video entries ({"video": ...}, optionally with video_start/video_end) load only with
# the patched MS-Swift from scripts/setup_train.sh; unpatched revisions fail on every such row.
if grep -q -F '{"video":' "$DATASET"; then
  "${SWIFT_PYTHON:-python}" - 2>/dev/null <<'PY' || fail "dataset uses dict video entries, but MS-Swift (checked with ${SWIFT_PYTHON:-python}) lacks the EgoTools video-entry patch; run scripts/setup_train.sh"
import inspect

from swift.template.templates.qwen import Qwen2VLTemplate

raise SystemExit(0 if "isinstance(video, dict)" in inspect.getsource(Qwen2VLTemplate.replace_tag) else 1)
PY
fi
mkdir -p "$OUTPUT_DIR"
# pipefail preserves a failed training process's exit status through tee.
{ print_command; "${command_args[@]}"; } 2>&1 | tee -a "$OUTPUT_DIR/train.log"
