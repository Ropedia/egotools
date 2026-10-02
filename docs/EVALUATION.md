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

The defaults follow **Experimental Setup** and **Training and Evaluation
Details** in the [paper](https://arxiv.org/abs/2609.39378). Full benchmark results
use its released 1,000 questions and curated track labels (363 AC, 236 PG,
222 PD, 179 SR); a smaller `--limit` run measures only that subset.

## Install VLMEvalKit

Model inference uses an external VLMEvalKit checkout. The setup script
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
upstream packaging warning, not a clean `pip check` result.

## Prepare a Benchmark

Obtain a manifest and its media as described in [Data](DATA.md). A directory
that already contains `manifest.tsv` and its referenced media is ready to use.
[Historical downloads](DATA.md#historical-benchmark) require the explicit preview
configuration.

To combine a separate manifest and media root, use one command:

```bash
egotools-prepare-benchmark \
  --manifest /path/to/download/manifest.tsv \
  --assets /path/to/download \
  --output data/benchmark \
  --asset-mode full
```

The command validates questions and media before writing a single `manifest.tsv`
and linking asset directories into the runnable root. It accepts TSV, CSV, and
JSONL and preserves indices and question data. Add `--reindex` only to replace
missing or repeated row indices with sequential ones; stable `qa_id` values stay
unchanged. Use `--link-mode copy` when symlinks are unsuitable. Existing unrelated
output content is preserved, and conflicting manifests or asset directories
cause an error. A Hub cache root with multiple snapshots requires selecting the
intended snapshot explicitly.

The evaluation entry point requires all eight A–H options to be nonempty.
Use the released option order: the construction pipeline already shuffles it
deterministically from `qa_id`. Do not reshuffle options at inference time.

Use `--video-mode full` for the [historical bundle](DATA.md#historical-benchmark).
Clip mode selects `clip_video`, falling back to `video` only when the clip field
is empty; a non-empty missing clip is an error.

`--dataset-root` can point directly to a directory containing `manifest.tsv`, or
to its parent when `--bench-version` selects a version subdirectory. The benchmark
version also labels the outputs; it does not verify a dataset's provenance.

## Run a Model

Run on one GPU with 64 sampled frames:

```bash
CUDA_VISIBLE_DEVICES=0 egotools-evaluate \
  --dataset-root data/benchmark \
  --model Qwen3-VL-8B-Instruct --nframe 64 \
  --results-dir outputs/qwen3-vl-8b
```

Add `--limit 10` for a small run. `--dry-run` checks the manifest, media,
sampling settings, and upstream entry point without downloading or initializing
a model; it does not test GPU inference. If the checkout is outside the default
location, add
`--vlmevalkit-dir /path/to/VLMEvalKit` or set `VLMEVALKIT_DIR`. An editable
VLMEvalKit installation can also supply its source checkout containing `run.py`.

For distributed inference, specify both the visible devices and process count:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 egotools-evaluate \
  --dataset-root data/benchmark \
  --model Qwen3-VL-8B-Instruct \
  --nframe 64 --nproc-per-node 8
```

The launcher uses the active interpreter and `torch.distributed.run` for multiple
processes. Without an explicit sampling override, it selects the paper's input
setting for each model:

| Model | Sampling | Audio in the paper |
| --- | --- | --- |
| EgoTools-8B, Qwen3-VL Instruct, other visual baselines | 64 frames | No |
| Qwen3-VL-4B/8B-Thinking | 512 frames | No |
| Qwen2.5-Omni-7B-Instruct | 64 frames | Yes |
| Gemini-3-Flash, Gemini-3.1-Pro/Flash-Lite | 1 FPS video | Yes |

Explicit `--nframe` or `--fps` values override these defaults, while
`--nframe 0` delegates sampling to the model's defaults. Upstream adapters may
sample or limit frames differently, so a shared frame count does not establish
identical inputs across models. Audio and proprietary-model protocols should
follow the paper's settings.

Use a local trained checkpoint with its compatible architecture preset:

```bash
egotools-evaluate \
  --dataset-root data/benchmark \
  --model EgoTools-8B \
  --model-path /path/to/merged-checkpoint --nframe 64
```

Known Hugging Face presets use the local cache and download missing snapshots.
`--no-auto-download-model` requires the snapshot to be cached. A local
`--model-path` bypasses model downloads. API models use upstream authentication
variables and should normally run with one process. Additional upstream options
can follow `--extra`; `--data`, `--model`, `--work-dir`, and `--config` are
controlled by the EgoTools runner. `EgoTools-8B` uses the Qwen3-VL-8B architecture
with the paper's `ropedia-ai/egotools-8b` checkpoint location; supplying
`--model-path` uses your local checkpoint instead.

Qwen3-VL uses Transformers when vLLM is unavailable. The adapter selects SDPA;
`EGOTOOLS_QWEN3VL_ATTN` and `EGOTOOLS_QWEN2VL_ATTN` override the relevant
attention implementation. MiMo presets use Transformers. The Qwen2.5-Omni
`ForVideo` preset retains upstream audio support. Gemini's video-and-audio path
requires `GOOGLE_API_BACKEND=genai`; the upstream `vertex` path converts video
to images and does not reproduce this input protocol. The API version names and
generation defaults come from the adapters; the paper does not specify their
exact preview revisions, temperatures, or thinking budgets.

Audio-enabled runs must use MP4s containing only synchronized **non-narration
audio**. The runner does not identify or remove narrated speech. Narration audio,
corrected narrations, dense captions, and annotation metadata are excluded from
evaluation inputs. The prompt uses only the question and A–H options, together
with the selected video. Use the prepared benchmark videos, not raw recordings
with narration tracks. Decode failures are surfaced rather than rewriting media.

The Qwen baseline and EgoTools-8B main protocol passes MP4s to the model adapter.
The paper's deterministic, pre-extracted-frame input ablations are a different
pipeline; selecting 64 frames alone does not reproduce those ablation results.

## Results and Scoring

Outputs default to `outputs/evaluation/<run-id>/`. `--results-dir` changes the
parent directory, and `EGOTOOLS_RUN_ID` supplies a run directory name. Use a fresh
run ID when changing the checkpoint, dataset, or input protocol to avoid
upstream prediction reuse.

| Output | Contents |
| --- | --- |
| `results.json` | Aggregated metrics and requested `input_settings` (video mode, frame count, FPS) |
| VLMEvalKit prediction table | Model responses in the upstream format |
| `*_egotools_score.json` | Metrics next to the prediction table |

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
and VLMEvalKit adapter share answer extraction and metric calculation.

## Validation Scope

On October 2, 2026, the reorganized code completed Qwen3-VL-8B-Instruct inference
on two historical benchmark questions from one downloaded video, using 64
frames, BF16/SDPA, and one RTX 6000 Ada 48 GB. The run exited successfully,
produced two predictions, and scored 0/2 with 100% answer extraction. The
standalone scorer installed from the wheel reproduced the same metrics.

The 114 local tests, lint checks, distribution build, and installed command
entry points also passed. These checks cover the implementation and a small
real-model run. The final benchmark, EgoTools checkpoint, and complete paper
results remain unverified. External EgoSchema/EgoThink runners and
proprietary-model evaluations are not reproduced by this repository.
