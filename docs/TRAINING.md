# Training EgoTools-8B

The reference recipe starts from
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), updates
the full language-model component in one SFT stage, and freezes the vision encoder
and multimodal aligner. The reported settings are in the paper's
[Training and Evaluation Details appendix](https://arxiv.org/abs/2609.39378).
The launcher accepts a local dataset, a Hub model ID or local model directory,
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

The paper run used Linux x86_64, Python 3.10, CUDA 12.4 wheels, and eight
A100-40GB GPUs. [`scripts/setup_train.sh`](../scripts/setup_train.sh) installs
that stack into the active environment: the pinned dependencies, MS-Swift at
revision `85ce1be91aad75b1c86424190d4efac1b9896cb3` (reported version
`4.2.0.dev0`) with the EgoTools video-entry patch, FlashAttention 2.8.3, and this
package. Building FlashAttention requires a compatible CUDA toolkit and C++
compiler.

```bash
python3.10 -m venv .venv-train
source .venv-train/bin/activate
bash scripts/setup_train.sh
swift sft --help
```

The script clones MS-Swift into `third_party/ms-swift/` (set `MS_SWIFT_DIR` to
reuse a checkout at the same revision) and applies
[`ms-swift-85ce1be-qwen-video-window.patch`](../configs/training/ms-swift-85ce1be-qwen-video-window.patch).
The patch lets a video entry be a dictionary, such as
`{"video": "clips/a.mp4", "video_start": 12.0, "video_end": 18.5}`, and passes the
clip window to `qwen-vl-utils`. Every video row of the final training data uses
this form; unpatched MS-Swift revisions fail on these rows, and
`scripts/train.sh` refuses to start on such data without the patch. Revision
`85ce1be` also adds support for `datasets` 4.x, which the paper run used (4.8.4).

[`requirements-cu124.txt`](../configs/training/requirements-cu124.txt) pins the
direct dependencies of the paper run, and
[`original-env-freeze.txt`](../configs/training/original-env-freeze.txt) lists
every package version of that environment. Other CUDA platforms need matching
PyTorch and torchvision wheels.

The launcher uses this revision's `swift sft` CLI, including `--tuner_type`,
`--torch_dtype`, and `--freeze_aligner`. `--use_hf true` selects Hugging Face for
model downloads; local model directories are also accepted. The pinned CLI's
`swift sft --help` shows a preliminary backend parser. To see the full argument
list without loading a model:

```bash
python -c "from swift.pipelines import sft_main; sft_main(['--help'])"
```

## Prepare Training Inputs

### Final Training Data

The paper mixture contains **184,679 examples**. Its model-input JSONL and all
referenced media (3,729 videos and 7,595 images, about 107 GB) are in the
`sft/` folder of [EgoTools-Data](https://huggingface.co/datasets/ropedia-ai/egotools-data/tree/main/sft),
configured as `training` in [`configs/resources.yaml`](../configs/resources.yaml):

```bash
egotools-download training --output-dir data/egotools-data
```

The command prints the local `sft/` path. It contains `train_184679.swift.jsonl`,
a 128-row `smoke_128.swift.jsonl`, the media under
`data_final_v4_sft_v5_902_20260625/`, and `provenance/` with the paper run's
original JSONL, arguments, and logs. Media paths are relative to `sft/`, so
launch from that directory. The JSONL already contains only `messages`, `videos`, and `images`;
`egotools-prepare-sft` is not needed.

| Rows | Video entry | Images |
| ---: | --- | ---: |
| 174,679 | `{"video", "video_start", "video_end"}` clip window | 0 |
| 5,000 | `{"video"}` pre-cut clip, next-action question | 1 (current frame) |
| 5,000 | none, single-image open-ended question | 1 |

`provenance/undecodable_rows.json` lists 29 rows whose clip window starts after
the end of the source video. MS-Swift replaces each with a randomly drawn row,
as it did in the paper run.

### Other Inputs

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
source videos and all of their derivatives from training; use the
[source-video exclusion tool](DATA_PROCESSING.md#exclude-benchmark-source-videos-from-training).
The experimental temporal split retains other intervals from the same source
video and does not implement the paper's split. Preparing model inputs below
does not perform either split.

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
Neither the converter nor the launcher crops videos using those row-level
fields; a clip window must be given inside the video entry, as in the final
training data.
[Historical training instructions](DATA.md#historical-training-data) explain the
base and incremental downloads and their shared media root.

## Paper Recipe

[`scripts/train.sh`](../scripts/train.sh) implements the settings reported in
the paper's **Training and Evaluation Details** appendix:

| Setting | Value |
| --- | --- |
| Base model | Qwen3-VL-8B-Instruct |
| GPUs | 8 |
| Precision | BF16 |
| Trainable modules | Full language model |
| Frozen modules | Vision encoder and aligner |
| Epochs | 1 |
| Effective batch | 128 (1 per device × 16 accumulation × 8 GPUs) |
| Optimizer | AdamW |
| Learning rate | `2.3e-6`, constant |
| Warmup | 0 |
| Sequence length | 8192 |
| Video sampling | 64 uniformly sampled frames, with a 2 FPS target |
| Image budget | At most 1024 visual tokens per image |
| Distributed optimizer | DeepSpeed ZeRO-3 |
| Attention | FlashAttention |
| Gradient checkpointing | Enabled |

The launcher also supplies implementation settings that the paper does not
report: `VIDEO_MIN_TOKEN_NUM=VIDEO_MAX_TOKEN_NUM=128`, the `adamw_torch` backend, weight decay `0.1`,
Adam betas `(0.9, 0.95)`, seeds `42 / 42`, Decord decoding, and ZeRO-3 without
offload. These are the current code defaults, not additional paper specifications.
The video-frame token setting is separate from `IMAGE_MAX_TOKEN_NUM=1024`;
the paper does not specify a separate video-frame token budget.

In the pinned MS-Swift implementation, `--tuner_type full` and the three
`--freeze_*` flags train the language model and freeze the complete visual tower,
including its merger and DeepStack projectors. `--torch_dtype bfloat16` enables
BF16 training, and `--attn_impl flash_attn` selects FlashAttention 2. MS-Swift
passes the image/video budgets and FPS bounds to `qwen-vl-utils`.

Run the default recipe on the final data from its `sft/` folder (smoke-test
first with `--dataset smoke_128.swift.jsonl -- --max_steps 2`):

```bash
cd data/egotools-data/sft
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
bash /path/to/EgoTools/scripts/train.sh \
  --dataset train_184679.swift.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir /path/to/outputs/egotools-8b
```

The launcher trains for one epoch and does not hard-code a step count; on eight
GPUs the final data gives the paper run's 1,443 optimizer steps. `--truncation_strategy
delete` is another implementation choice, not a reported paper setting; it
discards over-length examples. Inspect the retained example and step counts
when changing data or media preparation.

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
  --output-dir outputs/one-step-training \
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

For the final data, the following were checked against the paper run: the
downloaded JSONL has 184,679 rows and differs from the trained file only by a
removed absolute path prefix (both files are in `provenance/`); every referenced
media file is present; the patched `qwen.py` is byte-identical to the file used
in training; and the launcher arguments match the run's recorded arguments,
except checkpoint frequency and retention and the DataLoader worker count. A full
GPU training run has not been repeated with this repository.

An earlier runtime check loaded all 172,118 prepared
historical rows and completed one optimizer step with Qwen3-VL-2B-Instruct on
one example and one RTX 6000 Ada. That test used SDPA, accumulation 1, and no
checkpoint saving; it did not exercise FlashAttention, 8B training, or state
resumption. It ran in a temporary Python 3.10 environment with the earlier
MS-Swift pin. GPU training has not been rerun after the code reorganization.

Full reproduction still requires the intended hardware, a complete training
run, and comparison with the released checkpoint.
The [historical checkpoint](DATA.md#historical-checkpoint) comes from an earlier
116,031-example run and does not establish reproduction of the final model.
