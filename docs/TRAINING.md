# Training EgoTools-8B

The paper recipe starts from
[`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
and updates the full language-model component while freezing the visual encoder
and multimodal aligner. The launcher accepts local data, a Hub model ID or local
model directory, and an output directory. It can be inspected without a GPU:

```bash
bash training/train_qwen3_vl_8b.sh \
  --dataset /path/to/final.swift.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/egotools-8b \
  --dry-run
```

`--dry-run` prints the effective environment and a shell-quoted command. It does
not import ms-swift, download a model, read data, create the output directory, or
start training. Remove `--dry-run` to execute the same command after setup.

## Environment

Use Linux x86_64 and Python 3.11 for the supplied CUDA 12.4 dependencies. The
paper configuration uses eight GPUs. The driver must support CUDA 12.4, and
building FlashAttention requires a compatible CUDA toolkit and C++ compiler.

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -r training/requirements-cu124.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check
swift sft --help
```

The pinned upstream `swift sft --help` prints its preliminary backend parser.
To inspect the complete argument list without loading a model:

```bash
python -c "from swift.pipelines import sft_main; sft_main(['--help'])"
```

`ms-swift` is an external dependency, installed from its public Git repository at
revision `44c92c7cea08bf3b6e9f9b05ab182b6e81b0a7c7` (reported version
`4.2.0.dev0`). The launcher uses that revision's `swift sft` CLI, including
`--tuner_type`, `--torch_dtype`, and `--freeze_aligner`. `--use_hf true` selects
Hugging Face for downloading model IDs. A local model directory is also accepted.

`training/requirements-cu124.txt` pins the main dependencies and is not a complete
environment lock. It uses `datasets==3.6.0` because the pinned ms-swift revision
requires `datasets>=3.0,<4.0`; the previously captured development environment
listed `datasets==4.8.4`, which conflicts with that requirement. This is a
compatible installation recipe, not an exact export of the original training
environment. Other CUDA platforms need matching PyTorch and torchvision wheels.

## Data and media paths

The complete paper mixture has **184,679 examples**. The currently configured
training download is an earlier staging bundle; it cannot reproduce the final
paper model by itself. See [the repository README](../README.md) for current
resource availability. The launcher does not silently choose a training dataset:
provide `--dataset` or `DATASET` yourself.

Use ms-swift JSONL with `messages` and the media columns referenced by each row.
Video examples use `videos` and `<video>` placeholders. The final mixture also
contains single-image examples, which use `images` and `<image>` placeholders,
and examples that include both media types. For example:

```json
{"messages":[{"role":"user","content":"<video>Which tool is being used?"},{"role":"assistant","content":"A screwdriver."}],"videos":["/path/to/clips/example.mp4"]}
```

The JSONL and every referenced clip or image must be available on the training
host. Absolute paths are the simplest portable choice after downloading and
extracting the media. Relative media paths are resolved by the runtime from the
launch working directory; they are not automatically made relative to the JSONL
file. For the staging bundle, merge the base and delta downloads into the same
directory as described in [DATA.md](DATA.md), then launch from that directory so
the `videos/train/...` references resolve. Invoke this repository's launcher by
its absolute path if running outside the code checkout. Preserve the source-video
split when preparing data: benchmark source
videos and all their derivatives are excluded from training.

Before loading the staging JSONL with ms-swift, create a model-input copy:

```bash
python training/prepare_sft.py \
  /path/to/training-root/sft/v4_all_backfilled/final.jsonl \
  --output /path/to/training-root/train.swift.jsonl
```

Use `train.swift.jsonl` as the launcher's `--dataset`. Directly loading the
published `final.jsonl` fails in Hugging Face Datasets because its nested
`metadata` fields differ across batches. The converter streams all records in
their original order and preserves `messages`, `videos`, `images`, `audios`,
`tools`, and `objects` when present. It removes provenance columns from the copy;
the original JSONL is retained. Invalid JSON or malformed messages stop the
conversion with a line number, and an incomplete output is not published.

The pinned ms-swift SFT loader discards the original top-level `start_frame` and
`end_frame` fields; the converter removes them along with `_index` and `metadata`.
Neither the converter nor this training launcher crops videos using those span
fields. Retain the original annotations for provenance and perform any source
exclusion or intended clip construction before creating the model-input copy.

## Paper recipe

The human-readable reference is
[`configs/training/qwen3_vl_8b_full_sft.yaml`](../configs/training/qwen3_vl_8b_full_sft.yaml).
It is not a ms-swift configuration file; the executable arguments are in
[`training/train_qwen3_vl_8b.sh`](../training/train_qwen3_vl_8b.sh).

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

The paper reports 1,443 optimizer steps for the final mixture. This is a reported
result, not a hard-coded stopping point: the launcher trains for one epoch.
`--truncation_strategy delete` discards examples that exceed the sequence limit,
so the retained example count and resulting step count must be checked in the
training log when using another dataset or media preparation.

Both `FPS_MIN_FRAMES=64` and `FPS_MAX_FRAMES=64` are needed for the paper's fixed
64-frame setting. A maximum alone samples fewer frames from videos shorter than
32 seconds at the target 2 FPS. The pinned `qwen-vl-utils` decoder still clamps
the sample count to the available source frames (rounded to an even number);
it does not duplicate frames to reach 64 on a source containing fewer than 64
frames. Exact fixed-frame reproduction therefore also depends on the supplied
media. The launcher uses the `decord` reader and does not modify clips.

Run the default recipe with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
bash training/train_qwen3_vl_8b.sh \
  --dataset /path/to/final.swift.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/egotools-8b
```

The equivalent environment variables `DATASET`, `MODEL`, and `OUTPUT_DIR` remain
supported; command-line values take precedence. The launcher also accepts
`DATASET_NUM_PROC`, `DATALOADER_NUM_WORKERS`, `MASTER_PORT`, `SAVE_STEPS`, and
`SAVE_TOTAL_LIMIT`. Set both `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` if changing
the GPU count. That changes the effective batch unless accumulation is adjusted.

For an explicit experiment, append additional ms-swift arguments after `--`:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
bash training/train_qwen3_vl_8b.sh \
  --dataset /path/to/small.swift.jsonl \
  --output-dir outputs/smoke-test \
  -- --max_steps 1 --gradient_accumulation_steps 1
```

This example tests a different configuration; it is not the paper run, and
full-parameter training may still exceed the memory of a single GPU. Video
environment variables can likewise be overridden explicitly. The launcher does
not automatically reduce frames or batch size after an out-of-memory error.

Logs are appended to `OUTPUT_DIR/train.log`, and a failing ms-swift process
returns a failing exit status through the logging pipeline. By default,
`--save_only_model true` omits optimizer and scheduler state. If a run needs exact
training-state resumption, start it with `-- --save_only_model false` and resume
from a checkpoint that actually includes those states.

## Existing public checkpoint and verification limits

The currently public
[`egotools-dev/egotools-8b-v3_3`](https://huggingface.co/egotools-dev/egotools-8b-v3_3)
is an earlier **116,031-example** run. Its published `args.json` reports an
effective batch of 128 through per-device batch 2 and accumulation 8, and its
repository root is an intermediate checkpoint. It is available for smoke testing
and inspection; it is not the final 184,679-example paper model.

The release was also exercised with the actual pinned ms-swift framework:

- The raw public 172,118-row JSONL failed Arrow loading because of heterogeneous
  provenance metadata. The prepared copy loaded all 172,118 rows with strict
  dataset validation and unchanged message and video values.
- The released launcher completed one full language-model optimizer step on
  **Qwen3-VL-2B-Instruct**, using one RTX 6000 Ada GPU and one public training
  example with its referenced 99-frame video. The run retained BF16, ZeRO-3,
  frozen vision encoder/aligner, 64 sampled frames, an 8,192-token maximum, and
  the `2.3e-6` learning rate. It reported finite loss `2.73193479`, gradient norm
  `78.9930412`, step `1/1`, and peak allocated memory `35.84 GiB`, and exited 0.
- That bounded test explicitly used SDPA, accumulation 1, `--max_steps 1`, and
  `--save_strategy no`. The original FlashAttention setting failed with a missing
  `flash_attn` dependency in the available environment. Checkpoint saving and
  resumption were not exercised.
- The runtime test used a temporary Python 3.10 environment that reused installed
  PyTorch 2.6.0 / CUDA 12.4 and installed the pinned training framework packages.
  A clean installation of the documented Python 3.11 environment and a
  FlashAttention build were not exercised; dependency resolution for that
  environment did succeed.

The 2B one-step test verifies the training software path; it does not reproduce
the 8B paper model or its accuracy. Full reproduction still requires the final
184,679-example data and media, the final checkpoint for comparison, the intended
hardware, and a full training run. Shell integration tests separately cover
quoting, argument overrides, logging, and failure propagation.
