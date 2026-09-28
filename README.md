# EgoTools

Official code for **EgoTools: Towards Tool-Centric Reasoning in Real-World
Egocentric Videos**.

EgoTools combines a 100.36-hour egocentric tool-use corpus, a 1,000-question
diagnostic benchmark, and an 8B reference model. This repository contains only
the public data-processing, training, and evaluation code. Dataset media and
model weights are hosted on Hugging Face and are intentionally excluded from
Git history.

> **Private release candidate.** The code version is `1.0.0`, but no `v1.0.0`
> tag or public GitHub Release has been created yet. The final code license and
> final paper-aligned Hugging Face resources remain to be supplied before the
> repository is made public. See [the release checklist](docs/RELEASE_CHECKLIST.md).

## Resources

| Resource | Location | Current status |
| --- | --- | --- |
| Project page and paper PDF | <https://ropedia.github.io/egotools/> | Available |
| Public model checkpoint | <https://huggingface.co/egotools-dev/egotools-8b-v3_3> | Earlier 116,031-example checkpoint; not the final 184,679-example paper model |
| Public data bundle | <https://huggingface.co/datasets/egotools-dev/egotools_v4_backfilled_sft_v5_902_20260623> | Earlier 172,118-example SFT / 902-QA bundle; not the final paper release |
| Base model | <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct> | Available |

The resource mapping lives in [`configs/resources.yaml`](configs/resources.yaml).
All download commands accept explicit repository overrides so the code can be
used immediately when the final paper-aligned Hub repositories are published.

## Repository layout

```text
configs/                    Resource and training configuration
data_processing/            Benchmark construction and leakage-safe split tools
docs/                        Data, training, evaluation, and release documentation
eval/                        VLMEvalKit adapter and evaluation runner
src/egotools/                Lightweight download, validation, and scoring CLIs
tests/                       Unit and smoke tests
training/                    Paper training launcher and dependency specification
```

The repository deliberately does **not** contain video/audio media, annotation
submissions, participant or annotator metadata, internal review systems,
experiment caches, model weights, copied third-party repositories, or private
development history.

## Quick start

Create a lightweight environment for downloads, validation, and scoring:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,eval,test]'
```

Download the current benchmark metadata (no GPU or media download required):

```bash
egotools-download benchmark --metadata-only --output-dir data/huggingface
```

The historical 902-question bundle has duplicate integer row indices. Create
a TSV with sequential indices, preserving every question and its `qa_id`, then
validate it:

```bash
egotools-prepare-manifest \
  data/huggingface/benchmark/v5_686_plus_jskim216/final.tsv \
  --output data/benchmark/manifest.tsv
egotools-validate-manifest data/benchmark/manifest.tsv --expected-rows 902
```

Exercise prompt construction and scoring with a deterministic mock model:

```bash
python eval/scripts/run_eval.py \
  --dataset-root data/benchmark --bench-version preview902 \
  --model mock --smoke --limit 10
```

This mock run does not decode videos or measure model accuracy. Real inference
needs the base and incremental media downloads described in
[`docs/EVALUATION.md`](docs/EVALUATION.md).

Score a complete model prediction file using the same answer parser as the
evaluation adapter:

```bash
egotools-score predictions.tsv \
  --manifest data/benchmark/manifest.tsv \
  --output-dir outputs/my-model
```

For full model inference through VLMEvalKit, follow
[`docs/EVALUATION.md`](docs/EVALUATION.md). For the paper training recipe, see
[`docs/TRAINING.md`](docs/TRAINING.md).

## Reproducibility boundary

The paper reports a final 184,679-example training mixture and a 1,000-question
benchmark consisting of 900 human-authored questions and 100 human-verified
spatial questions. The currently public Hub bundle predates that final version.
The public bundle combines assets from two Hub repositories. Their file lists
cover all 902 `video` references; three optional `clip_video` files are absent.
The code has been tested on CPU and with existing framework imports. Full GPU
training and paper-model evaluation have not been rerun for this candidate.

For local checks:

```bash
python -m pytest -q
```

[`docs/VALIDATION.md`](docs/VALIDATION.md) records the checks and limitations.
The installable Python wheel provides the four `egotools-*` utilities. Clone
this repository (or use the source distribution) for training, data-processing,
and VLMEvalKit entry points.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The final
bibliographic entry will be added when the archival paper identifier is
available.

## License

The project license has not yet been selected by the authors. Until a `LICENSE`
file is added, this private release candidate is not ready for public reuse or
redistribution. Third-party components remain governed by their own licenses;
see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
