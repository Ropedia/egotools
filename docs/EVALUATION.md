# Evaluation

EgoTools evaluates video multiple-choice questions with options A through H.
The runner reports micro-averaged accuracy, answer-extraction coverage,
per-`qtype` accuracy, and research-track accuracy when track labels are present:

| Track | Capability |
| --- | --- |
| `AC` | Affordance & Causality |
| `PG` | Perception & Grounding |
| `PD` | Procedural Dynamics |
| `SR` | Spatial Reasoning |

## Offline Example

Install the evaluation utilities from the repository root:

```bash
python -m pip install -e '.[eval]'

egotools-evaluate \
  --dataset-root examples/benchmark --bench-version example \
  --model mock --smoke --mock-mode fixed --mock-letter A \
  --results-dir outputs/example
```

This reads the two synthetic manifest rows, builds prompts, writes predictions,
and scores the fixed answer `A` at 1/2. It does not decode videos or run a model.
Results are written to `outputs/example/<run-id>/results.json` alongside raw and
scored TSVs. The same command can check another manifest's software path without
media, using `--limit` to select a subset if desired.

## Prepare a Benchmark

Obtain a manifest and its media as described in [Data](DATA.md). The default
resource mapping reserves the final release locations; [historical downloads](DATA.md#historical-benchmark)
require the explicit preview configuration.

Prepare sequential row indices, then pair the manifest with its asset root:

```bash
egotools-prepare-manifest /path/to/download/final.tsv \
  --output data/prepared/manifest.tsv

egotools-materialize \
  --manifest data/prepared/manifest.tsv \
  --source-root /path/to/download \
  --output-dir data/benchmark \
  --asset-mode full

egotools-validate-manifest data/benchmark/manifest.tsv \
  --asset-root data/benchmark --asset-mode full
```

Preparation preserves `qa_id`, questions, options, answers, and media paths.
Materialization links the downloaded asset directories into the runnable root;
`--link-mode copy` is available when symlinks are unsuitable. Existing unrelated
output content is preserved, and conflicting manifests or asset directories
cause an error. A Hub cache root with multiple snapshots requires selecting the
intended snapshot explicitly.

For the earlier 902-question bundle, the source root after the download commands
in [Data](DATA.md#historical-benchmark) is
`data/huggingface/benchmark/v5_686_plus_jskim216`. It covers all full-video
references but has three missing non-empty clip paths. Use `--video-mode full`
for that bundle. Clip mode selects `clip_video`, falling back to `video` only
when the clip field is empty; a non-empty missing clip is an error.

`--dataset-root` can point directly to a directory containing `manifest.tsv`, or
to its parent when `--bench-version` selects a version subdirectory. The benchmark
version also labels the outputs; it does not verify a dataset's provenance.

## Install VLMEvalKit

Real model inference uses an external VLMEvalKit checkout. The setup script
creates a dedicated conda environment and clones upstream revision
`e7d64cfa8f6036e1d00e21522aaee0102544ea25` into the ignored
`third_party/VLMEvalKit/` directory:

```bash
bash scripts/setup_eval.sh
conda activate egotools_eval
```

An existing checkout is reused without changing its revision. Set
`VLMEVALKIT_DIR` to reuse another checkout and `EGOTOOLS_EVAL_ENV` to choose a
conda environment name. The script sets `PYTHONNOUSERSITE=1` in that environment
so unrelated user-site packages cannot override the installed dependencies.

The recipe uses Python 3.10, PyTorch 2.6.0, torchvision 0.21.0, and Transformers
5.6.2. CUDA 12.4 is the default PyTorch wheel channel; set `TORCH_INDEX_URL` for
another driver-compatible channel. CPU wheels can support adapter checks, but
the Qwen inference adapter requires a GPU.

<details>
<summary>Install into an existing inference environment</summary>

Install driver-compatible PyTorch wheels first, then:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git /path/to/VLMEvalKit
git -C /path/to/VLMEvalKit checkout e7d64cfa8f6036e1d00e21522aaee0102544ea25
export PYTHONNOUSERSITE=1
python -m pip install -c configs/evaluation/constraints.txt \
  -e /path/to/VLMEvalKit -e '.[eval]'
python scripts/verify_eval.py
```

Keep `PYTHONNOUSERSITE=1` set when running that environment. For conda,
`conda env config vars set -n YOUR_ENV PYTHONNOUSERSITE=1` persists it for future
activations. Other VLMEvalKit model families may need separate upstream
dependencies; the supplied environment targets the Qwen3-VL recipe.

</details>

On the tested Linux x86_64 / Python 3.10 installation, `pip check` reported a
Decord 0.6.0 wheel metadata warning about its internal Python 3.6 tag. Decord
imports and 64-frame decoding succeeded in that environment. This remains an
upstream packaging warning, not a clean `pip check` result. See
[Validation](VALIDATION.md) for the exact runtime checks.

## Run a Model

Validate the manifest, media references, sampling settings, and upstream
entry-point location before loading a model:

```bash
egotools-evaluate \
  --dataset-root data/benchmark \
  --model Qwen3-VL-8B-Instruct \
  --nframe 64 --limit 10 --dry-run
```

`--dry-run` neither downloads nor initializes a model and does not verify GPU
inference. If the checkout is outside the default location, add
`--vlmevalkit-dir /path/to/VLMEvalKit` or set `VLMEVALKIT_DIR`. An editable
VLMEvalKit installation can also supply its source checkout containing `run.py`.

Run on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 egotools-evaluate \
  --dataset-root data/benchmark \
  --model Qwen3-VL-8B-Instruct --nframe 64 \
  --results-dir outputs/qwen3-vl-8b
```

For distributed inference, specify both the visible devices and process count:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 egotools-evaluate \
  --dataset-root data/benchmark \
  --model Qwen3-VL-8B-Instruct \
  --nframe 64 --nproc-per-node 8
```

The launcher uses the active interpreter and `torch.distributed.run` for multiple
processes. It passes the requested frame count to VLMEvalKit. The default is
64 uniformly sampled frames; `--fps 1` selects a frame-rate protocol, while
`--nframe 0` delegates sampling to the model's defaults. Upstream adapters may
sample or limit frames differently, so a shared frame count does not establish
identical inputs across models. Audio and proprietary-model protocols should
follow the paper's settings.

Use a local trained checkpoint with its compatible architecture preset:

```bash
egotools-evaluate \
  --dataset-root data/benchmark \
  --model Qwen3-VL-8B-Instruct \
  --model-path /path/to/merged-checkpoint --nframe 64
```

Known Hugging Face presets use the local cache and download missing snapshots.
`--no-auto-download-model` requires the snapshot to be cached. A local
`--model-path` bypasses model downloads. API models use upstream authentication
variables and should normally run with one process. Additional upstream options
can follow `--extra`; `--data`, `--model`, `--work-dir`, and `--config` are
controlled by the EgoTools runner.

Qwen3-VL uses Transformers when vLLM is unavailable. The adapter selects SDPA;
`EGOTOOLS_QWEN3VL_ATTN` and `EGOTOOLS_QWEN2VL_ATTN` override the relevant
attention implementation. MiMo presets use Transformers. The Qwen2.5-Omni
`ForVideo` preset evaluates visual inputs with audio disabled, following the
existing recipe. Decode failures are surfaced rather than rewriting media.

## Results and Scoring

Outputs default to `outputs/evaluation/<run-id>/`. `--results-dir` changes the
parent directory, and `EGOTOOLS_RUN_ID` supplies a run directory name. Use a fresh
run ID when changing the checkpoint, dataset, or input protocol to avoid
upstream prediction reuse.

| Output | Contents |
| --- | --- |
| `results.json` | Aggregated metrics for the run |
| Raw and scored TSVs | Mock predictions and extracted answers |
| VLMEvalKit prediction table | Real-model responses in the upstream format |
| `*_egotools_score.json` | Metrics next to a real-model prediction table |

The standalone scorer accepts TSV, CSV, and JSONL. Each prediction needs
`qa_id` or `index`, plus `prediction`. Pair it with the exact inference manifest,
including any intentionally selected subset:

```bash
egotools-score predictions.tsv \
  --manifest data/benchmark/manifest.tsv \
  --output-dir outputs/model-name
```

It writes `metrics.json` and `predictions_scored.tsv`. Missing predictions,
duplicate rows, and inconsistent identities are errors by default.
`--allow-partial` explicitly scores available predictions and reports manifest
coverage.

The shared answer extractor first looks for standalone A–H letters, then answer
prefixes and option-text matches. Text fallback uses containment and token
overlap. This is the historical evaluation heuristic, not an LLM judge or a
strict final-answer parser; unextracted outputs score zero. The standalone scorer
and VLMEvalKit adapter use the same helper.
