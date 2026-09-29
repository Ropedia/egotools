# Training EgoTools-8B

The reference recipe starts from
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), updates
the full language-model component, and freezes the vision encoder and multimodal
aligner. It accepts a local dataset, a Hub model ID or local model directory,
and an output directory.

Inspect the effective command without a GPU or installed training framework:

```bash
bash scripts/train.sh \
  --dataset /path/to/train.swift.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/egotools-8b \
  --dry-run
```

`--dry-run` prints the effective environment and shell-quoted command. It does
not load data or models, create output directories, or start training. Remove
it after preparing the environment and data.

## Environment

The supplied dependency recipe targets Linux x86_64, Python 3.11, and CUDA 12.4.
The paper configuration uses eight GPUs. Building FlashAttention also requires
a compatible CUDA toolkit and C++ compiler.

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -r configs/training/requirements-cu124.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e .
python -m pip check
swift sft --help
```

MS-Swift is installed from upstream revision
`44c92c7cea08bf3b6e9f9b05ab182b6e81b0a7c7` (reported version `4.2.0.dev0`).
The launcher uses this revision's `swift sft` CLI, including `--tuner_type`,
`--torch_dtype`, and `--freeze_aligner`. `--use_hf true` selects Hugging Face for
model downloads; local model directories are also accepted.

The pinned CLI's `swift sft --help` shows a preliminary backend parser. To see
the full argument list without loading a model:

```bash
python -c "from swift.pipelines import sft_main; sft_main(['--help'])"
```

[`requirements-cu124.txt`](../configs/training/requirements-cu124.txt) pins the
main dependencies rather than a complete environment lock. It uses
`datasets==3.6.0` to satisfy the pinned framework's `datasets>=3.0,<4.0`
requirement. It is a compatible installation recipe, not an exact export of the
original training environment. Other CUDA platforms need matching PyTorch and
torchvision wheels.

## Prepare Training Inputs

The paper mixture contains **184,679 examples**. Its final resource location is
pending in [the resource configuration](../configs/resources.yaml).
[Historical resources](DATA.md#historical-resources) provide earlier data for
inspection and software checks. The launcher requires `--dataset` or `DATASET`
explicitly.

Use MS-Swift JSONL with `messages` and the referenced media columns. Video rows
use `videos` and `<video>` placeholders; image rows use `images` and `<image>`
placeholders. Both media types may appear in a single row:

```json
{"messages":[{"role":"user","content":"<video>Which tool is being used?"},{"role":"assistant","content":"A screwdriver."}],"videos":["/path/to/clips/example.mp4"]}
```

The JSONL and all referenced media must be present on the training host.
Relative paths are resolved from the launch working directory, not automatically
from the JSONL directory. Use absolute media paths, or launch from the directory
that contains the referenced `videos/` and `images/` paths. When launching outside
the code checkout, invoke `scripts/train.sh` by its absolute path.

Perform source filtering and any intended clip construction on the original
annotations before preparing model inputs. The paper protocol excludes benchmark
source videos and all of their derivatives from training; see
[Data Processing](DATA_PROCESSING.md) for the source-split tools.

Create a uniform model-input copy:

```bash
egotools-prepare-sft /path/to/original.jsonl \
  --output /path/to/train.swift.jsonl
```

The converter streams records in order and preserves `messages`, `videos`,
`images`, `audios`, `tools`, and `objects` when present. It removes provenance
columns from the copy and retains the original JSONL. Invalid JSON or malformed
messages stop conversion with a line number, without publishing an incomplete
output.

This conversion is required for the historical bundle: its heterogeneous nested
`metadata` causes the pinned loader to fail. Conversion also removes the original
`start_frame` and `end_frame` fields, which the pinned MS-Swift SFT loader discards.
Neither the converter nor the launcher crops videos using those fields.
[Historical training instructions](DATA.md#historical-training-data) explain the
base and incremental downloads and their shared media root.

## Paper Recipe

The settings are summarized in
[`qwen3_vl_8b_full_sft.yaml`](../configs/training/qwen3_vl_8b_full_sft.yaml).
That file is a human-readable reference; the executable MS-Swift arguments live
in [`scripts/train.sh`](../scripts/train.sh).

| Setting | Value |
| --- | --- |
| Base model | Qwen3-VL-8B-Instruct |
| GPUs | 8 |
| Precision | BF16 |
| Trainable modules | Full language model |
| Frozen modules | Vision encoder and aligner |
| Epochs | 1 |
| Effective batch | 128 (1 per device × 16 accumulation × 8 GPUs) |
| Optimizer | AdamW (`adamw_torch`) |
| Learning rate | `2.3e-6`, constant |
| Warmup | 0 |
| Weight decay | 0.1 |
| Adam betas | 0.9, 0.95 |
| Sequence length | 8192 |
| Video sampling | 2 FPS target, minimum and maximum 64 frames |
| Visual budget | `VIDEO_MAX_TOKEN_NUM=128`; `IMAGE_MAX_TOKEN_NUM=1024` |
| Distributed optimizer | DeepSpeed ZeRO-3, no offload |
| Attention | FlashAttention |
| Gradient checkpointing | Enabled for the language model |
| Seed / data seed | 42 / 42 |

Run the default recipe:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
bash scripts/train.sh \
  --dataset /path/to/train.swift.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/egotools-8b
```

The reported 1,443 optimizer steps describe the paper run. The launcher trains
for one epoch and does not hard-code that step count. `--truncation_strategy
delete` discards over-length examples, so inspect the retained example and step
counts when changing data or media preparation.

Both `FPS_MIN_FRAMES=64` and `FPS_MAX_FRAMES=64` are required for the fixed-frame
setting. The pinned decoder still limits samples to the available source frames,
rounded to an even number, and does not duplicate frames to reach 64 for shorter
sources. The launcher uses Decord and does not modify clips.

## Runtime Overrides and Checkpoints

Command-line values override the `DATASET`, `MODEL`, and `OUTPUT_DIR` environment
variables. The launcher also accepts `DATASET_NUM_PROC`, `DATALOADER_NUM_WORKERS`,
`MASTER_PORT`, `SAVE_STEPS`, and `SAVE_TOTAL_LIMIT`. Set both `CUDA_VISIBLE_DEVICES`
and `NPROC_PER_NODE` when changing GPU count; the effective batch changes unless
accumulation is adjusted.

Append explicit MS-Swift overrides after `--` for a separate experiment:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
bash scripts/train.sh \
  --dataset /path/to/small.swift.jsonl \
  --output-dir outputs/training-smoke \
  -- --max_steps 1 --gradient_accumulation_steps 1
```

This changes the paper configuration; full-parameter 8B training may still
exceed a single GPU's memory. Video environment variables can also be overridden.
The launcher does not silently reduce frames or batch size after an error.

Logs are appended to `OUTPUT_DIR/train.log`. A failed MS-Swift process propagates
a failing exit status through the logging pipeline. The default
`--save_only_model true` omits optimizer and scheduler state. For training-state
resumption, start with `-- --save_only_model false` and resume from a checkpoint
that includes those states.

## Validation Scope

Earlier validation loaded all 172,118 prepared historical rows and completed one
optimizer step with Qwen3-VL-2B-Instruct on one RTX 6000 Ada. That test used SDPA,
accumulation 1, and no checkpoint saving; it did not exercise FlashAttention,
8B training, or state resumption. The documented Python 3.11 environment resolved
dependencies, while the runtime test used a temporary Python 3.10 environment.

Full reproduction requires the final 184,679-example data and media, the final
checkpoint for comparison, the intended hardware, and a complete training run.
Measured logs, environment details, and limits are recorded in
[Validation](VALIDATION.md). The earlier 116,031-example checkpoint is described
under [Historical checkpoint](DATA.md#historical-checkpoint).
