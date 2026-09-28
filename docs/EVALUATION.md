# Evaluation

EgoTools evaluates video multiple-choice questions with options A through H.
The runner preserves the development benchmark's prompt and answer-extraction
heuristic. It reports micro-averaged accuracy, extraction coverage, per-`qtype`
accuracy, and research-track accuracy when the manifest includes track labels:
`AC` (Affordance & Causality), `PG` (Perception & Grounding), `PD` (Procedural
Dynamics), and `SR` (Spatial Reasoning).

## Prepare the benchmark

Install the lightweight utilities first:

```bash
python -m pip install -e '.[eval]'
```

See [DATA.md](DATA.md) for the external resources and their current status. The
currently published 902-question staging bundle is a delta over an earlier
video bundle, and its original `index` column contains duplicates. Download
both bundles, then prepare a local manifest with unique sequential indices.
`qa_id`, question content, answers, and media paths are preserved.

```bash
egotools-download benchmark-base \
  --output-dir data/hf/benchmark/v5_686_plus_jskim216
egotools-download benchmark --output-dir data/hf
egotools-prepare-manifest \
  data/hf/benchmark/v5_686_plus_jskim216/final.tsv \
  --output data/prepared/manifest.tsv
python eval/scripts/materialize_manifest_dataset.py \
  --manifest data/prepared/manifest.tsv \
  --source-root data/hf/benchmark/v5_686_plus_jskim216 \
  --output-dir data/benchmark \
  --asset-mode full
egotools-validate-manifest data/benchmark/manifest.tsv \
  --expected-rows 902 --asset-root data/benchmark --asset-mode full
```

The materializer links the downloaded asset directories into the runnable
root; `--link-mode copy` is available when symlinks are unsuitable. The source
assets remain external to Git. Existing unrelated output directories are
preserved. A Hugging Face cache root with multiple snapshots requires selecting
the intended snapshot explicitly.

The merged staging resources cover all 902 full-video references. Three
nonempty clip paths are missing from the public resources checked for this
release. Use `--video-mode full` for that bundle. Clip mode uses `clip_video`
and falls back to `video` only when the clip field is empty; it reports a
missing file when a nonempty clip path cannot be found. Do not interpret the
staging bundle as the final 1,000-question paper benchmark.

For another manifest, use the same preparation and materialization commands
with its local path and asset root. `--bench-version` is an output label; it
can also select a version subfolder under `--dataset-root`.

## CPU mock check

A mock check exercises manifest reading, the unchanged MCQ prompt, prediction
output, and scoring without downloading a model or decoding videos:

```bash
python eval/scripts/run_eval.py \
  --dataset-root data/benchmark \
  --model mock --limit 10
```

`--mock-mode fixed --mock-letter A` gives a simple deterministic baseline.
Mock metrics verify the software path and are not model evaluation results.

## Install VLMEvalKit

The repository includes an adapter and launcher; VLMEvalKit remains an external
dependency. The setup script clones upstream revision
`e7d64cfa8f6036e1d00e21522aaee0102544ea25` into the ignored
`eval/VLMEvalKit/` directory and installs it into a dedicated conda environment.
An existing checkout is reused without changing its revision.

```bash
bash eval/setup/setup_env.sh
conda activate egotools_eval
```

The recipe uses PyTorch 2.6.0, torchvision 0.21.0, and Transformers 5.6.2.
The default PyTorch wheel channel is CUDA 12.4; choose a channel supported by
your driver using `TORCH_INDEX_URL`. CPU wheels can install the framework for
adapter checks, but the Qwen inference adapter requires a GPU. Set
`EGOTOOLS_EVAL_ENV` to choose another conda environment name and
`VLMEVALKIT_DIR` to reuse an external checkout.

If you already manage an inference environment, install the dependency there:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git /path/to/VLMEvalKit
git -C /path/to/VLMEvalKit checkout e7d64cfa8f6036e1d00e21522aaee0102544ea25
python -m pip install -c eval/setup/constraints.txt \
  -e /path/to/VLMEvalKit -e '.[eval]'
python eval/setup/verify_env.py
```

Install the appropriate PyTorch wheels first. Some other VLMEvalKit model
families need their own environments and upstream dependencies. This repository
does not bundle those environments or the development VITA-specific patches.

## Run a model

Check the manifest, media references, sampling configuration, and upstream
entry-point location before model loading:

```bash
python eval/scripts/run_eval.py \
  --dataset-root data/benchmark \
  --model Qwen3-VL-8B-Instruct \
  --vlmevalkit-dir /path/to/VLMEvalKit \
  --nframe 64 --limit 10 --dry-run
```

`--dry-run` neither downloads nor initializes a model and does not verify GPU
inference. Omit `--vlmevalkit-dir` when setup installed the default checkout or
an editable installation whose source includes `run.py`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python eval/scripts/run_eval.py \
  --dataset-root data/benchmark \
  --model Qwen3-VL-8B-Instruct \
  --nframe 64 --nproc-per-node 8
```

The launcher uses the current interpreter and `torch.distributed.run` for
multiple processes. It passes the dataset's frame count to VLMEvalKit. The
default is 64 uniformly sampled frames; `--fps 1` selects a frame-rate protocol
instead, and `--nframe 0` delegates sampling to the model's defaults. Upstream
adapters may differ in how they sample or limit video frames, so a shared frame
count alone does not make all model inputs identical. Audio and proprietary
model protocols should follow the settings documented in the paper.

Use a trained checkpoint with its compatible upstream architecture preset:

```bash
python eval/scripts/run_eval.py \
  --dataset-root data/benchmark \
  --model Qwen3-VL-8B-Instruct \
  --model-path /path/to/merged-checkpoint --nframe 64
```

Known Hugging Face model presets use the normal local cache and download a
missing snapshot when needed. `--no-auto-download-model` requires the snapshot
to be cached. A `--model-path` directory bypasses model downloads. API models
use their upstream authentication environment variables and should normally
run with one process. Pass additional upstream options after `--extra`;
`--data`, `--model`, `--work-dir`, and `--config` are controlled by this runner.

Qwen3-VL uses Transformers when vLLM is unavailable. The adapter selects SDPA
instead of upstream's required Flash Attention 2; set `EGOTOOLS_QWEN3VL_ATTN`
or `EGOTOOLS_QWEN2VL_ATTN` to change the attention implementation. MiMo presets
use the Transformers backend, and the Qwen2.5-Omni `ForVideo` preset evaluates
visual inputs with audio disabled, matching the existing development recipe.
Video decode failures are surfaced; this public runner does not rewrite media.

Outputs go to `eval/results/<run-id>/` by default. Mock runs write raw and
scored TSVs plus `results.json`. Real runs keep VLMEvalKit's prediction table,
write its adjacent `*_egotools_score.json`, and aggregate `results.json` after
successful inference. `--results-dir` changes the output parent and
`EGOTOOLS_RUN_ID` can name a particular run. Use a fresh run ID when changing
the model checkpoint, dataset, or input protocol to avoid upstream result reuse.

## Score existing predictions

The standalone scorer accepts TSV, CSV, and JSONL. Each prediction needs
`qa_id` or `index`, plus `prediction`. Pair it with the exact manifest used for
inference (including an intentionally limited subset, if applicable):

```bash
egotools-score predictions.tsv \
  --manifest data/benchmark/manifest.tsv \
  --output-dir outputs/model-name
```

It writes `metrics.json` and `predictions_scored.tsv`. By default, missing predictions, duplicate rows,
and inconsistent identities are reported as errors. `--allow-partial` explicitly
scores only the available predictions and reports their manifest coverage.

The historical extractor first looks for a standalone A-H letter, including
answers with explanations, then recognizes answer prefixes and option-text
matches. The option-text fallback uses containment and token overlap; it is a
heuristic, not an LLM judge or a strict final-answer parser. Unextracted outputs
score zero. Both the standalone scorer and the VLMEvalKit adapter use the same
extraction helper, preserving the development evaluation behavior.
