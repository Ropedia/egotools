<div align="center">

# EgoTools: Towards Tool-Centric Reasoning

**in Real-World Egocentric Videos**

<a href="">Paper</a> ·
<a href="https://ropedia.github.io/egotools/">Project Page</a> ·
<a href="">Dataset</a> ·
<a href="">Models</a>

</div>

## Overview

**EgoTools** studies how people select, use, and reason about tools in real-world
egocentric videos. It brings together a video corpus with hierarchical and
tool-centric annotations, a diagnostic benchmark, and an instruction-tuned
vision-language model.

| Component | Description |
| --- | --- |
| **EgoTools-Data** | Approximately 100 hours of egocentric video, with 361,332 hierarchical captions and 6,519 tool-centric narrations |
| **EgoTools-Bench** | 1,000 eight-choice questions across Affordance & Causality, Perception & Grounding, Procedural Dynamics, and Spatial Reasoning |
| **EgoTools-8B** | Qwen3-VL-8B-Instruct fine-tuned on 184,679 examples, with training and benchmark data separated by source video |

This repository provides data preparation, benchmark evaluation, and the training
recipe. Final paper, dataset, and model links are pending; the checks completed
so far and their scope are recorded in [Validation](docs/VALIDATION.md).

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[eval]'
```

This installs the dataset utilities, CPU mock evaluator, and scorer. GPU model
inference uses [VLMEvalKit](docs/EVALUATION.md#install-vlmevalkit), and training
uses a separate [MS-Swift environment](docs/TRAINING.md#environment).

<details>
<summary>Optional dependencies</summary>

```bash
# Gemini-assisted annotation tools
python -m pip install -e '.[data]'

# Development checks
python -m pip install -e '.[test]'
python -m pytest -q
```

</details>

## Quick Start

The two synthetic questions in [`examples/`](examples/) demonstrate the manifest,
prediction, and metric formats. They run locally without downloading data,
weights, or videos:

```bash
egotools-validate-manifest examples/benchmark/manifest.tsv --expected-rows 2

egotools-evaluate \
  --dataset-root examples/benchmark --bench-version example \
  --model mock --smoke --mock-mode fixed --mock-letter A \
  --results-dir outputs/example

egotools-score examples/predictions.jsonl \
  --manifest examples/benchmark/manifest.tsv \
  --output-dir outputs/scoring
```

The fixed-answer mock scores **1/2**; the supplied example predictions score
**2/2**. These are format demonstrations, with no video decoding or model
inference. Inspect `outputs/example/<run-id>/results.json` and
`outputs/scoring/metrics.json` for the resulting metrics.

## Data

The benchmark uses a TSV manifest with relative video paths, a question, options
`A` through `H`, a gold answer, and stable question/source identifiers. Media and
model weights are downloaded separately and remain outside this repository.

- [Data format and downloads](docs/DATA.md)
- [Benchmark construction and source-video split tools](docs/DATA_PROCESSING.md)

The default [resource configuration](configs/resources.yaml) reserves the final
release locations. Earlier experimental resources are documented separately in
[Historical resources](docs/DATA.md#historical-resources).

## Evaluation

After preparing a benchmark and installing the inference environment, run an
upstream model preset:

```bash
egotools-evaluate \
  --dataset-root /path/to/benchmark \
  --model Qwen3-VL-8B-Instruct --nframe 64 \
  --results-dir outputs/qwen3-vl-8b
```

Use `--model-path /path/to/checkpoint` to evaluate a local checkpoint with the
same architecture. The evaluator reports overall accuracy, answer-extraction
coverage, and accuracy by question type and research track. See the
[Evaluation guide](docs/EVALUATION.md) for environment setup, media preparation,
distributed inference, and scoring existing predictions.

## Training

The reference recipe fine-tunes the language-model component of Qwen3-VL-8B,
while freezing the vision encoder and multimodal aligner. Inspect the command
without requiring a GPU or downloading a model:

```bash
bash scripts/train.sh \
  --dataset /path/to/train.swift.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/egotools-8b \
  --dry-run
```

Follow the [Training guide](docs/TRAINING.md) to prepare inputs and install the
pinned dependencies. The human-readable settings are in
[`configs/training/`](configs/training/).

## Repository Structure

```text
configs/                  Resource mappings and evaluation/training environments
docs/                     Data, evaluation, training, and validation guides
examples/                 Synthetic inputs for the offline quick start
scripts/                  Environment setup and training launchers
src/egotools/
├── data/                 Benchmark construction, source splits, and SFT preparation
├── evaluation/           Runner, dataset adapter, materialization, and metrics
├── resources.py          Resource download configuration and CLI
├── manifest.py           Manifest validation
├── prepare.py            Manifest preparation
├── answers.py            Shared multiple-choice answer extraction
└── score.py              Prediction-file scoring
tests/                    Unit and integration checks
third_party/              Notices for adapted upstream code
```

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff). The final paper
link and BibTeX entry will be added when available.

## Acknowledgments

EgoTools builds on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
[MS-Swift](https://github.com/modelscope/ms-swift), and
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit). See
[Third-party software](docs/THIRD_PARTY.md) for attribution and dependency
revisions.

## License

The project license is pending. Third-party components retain their upstream
licenses; release prerequisites are tracked in the
[Release checklist](docs/RELEASE_CHECKLIST.md).
