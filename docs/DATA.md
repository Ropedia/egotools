# Data and benchmark format

Media and annotations are distributed separately through Hugging Face. This
code repository contains no dataset payload, participant records, or private
review logs. Repository access and permitted uses follow the terms attached
to the corresponding dataset.

## Paper release and current resources

The final paper describes 646 rectified egocentric videos totaling 100.36 hours,
361,332 hierarchical captions, 6,519 tool-centric narrations, 184,679
source-video-disjoint instruction-tuning examples, and 1,000 eight-way
benchmark questions. The benchmark combines 900 human-authored questions with
100 human-verified spatial questions from the 3D annotation pipeline.

The current `configs/resources.yaml` points to earlier resources: 172,118
training examples and 902 benchmark questions. The historical training assets
use temporal exclusion and must not be described as the final paper's
source-video-disjoint split. Raw fisheye recordings, participant identities,
and internal review records are outside the code release.

The June bundle is an addition to earlier media collections. Download the
benchmark base into the directory where the later manifest expects its media,
then download the addition into the enclosing Hub root:

```bash
egotools-download benchmark-base \
  --output-dir data/huggingface/benchmark/v5_686_plus_jskim216
egotools-download benchmark --output-dir data/huggingface

egotools-prepare-manifest \
  data/huggingface/benchmark/v5_686_plus_jskim216/final.tsv \
  --output data/huggingface/benchmark/v5_686_plus_jskim216/manifest.tsv

egotools-validate-manifest \
  data/huggingface/benchmark/v5_686_plus_jskim216/manifest.tsv \
  --expected-rows 902 \
  --asset-root data/huggingface/benchmark/v5_686_plus_jskim216 \
  --asset-mode full
```

`benchmark-base` points to `egotools-dev/egotools_bench_v5_20260505`; the later
bundle is `egotools-dev/egotools_v4_backfilled_sft_v5_902_20260623`.
The historical `final.tsv` repeats some integer indices. Preparation writes
sequential indices in a separate TSV while preserving `qa_id`, questions,
answers, and media paths. The combined Hub file inventory covers all 902
full-video references and 899 of 902 clip references; clip-mode experiments
must account for those three missing clips. Inventory coverage does not
establish that every media file decodes correctly.

Training likewise needs both base and added media at the same snapshot root:

```bash
egotools-download training-base --output-dir data/huggingface-training
egotools-download training --output-dir data/huggingface-training
```

`training-base` selects `videos/train/` from
`egotools-dev/egotools_train_time_excluded_20260504`. `training` downloads the
later `sft/v4_all_backfilled/` annotations and added `videos/train/` assets.
Training media references resolve from that shared root, not from the nested
JSONL directory. See [TRAINING.md](TRAINING.md) for launcher usage.

## Benchmark manifest

Evaluation consumes a UTF-8 TSV with one row per question:

| Column | Meaning |
| --- | --- |
| `index` | Unique row index |
| `video` | Relative full-video asset path |
| `question` | Multiple-choice question |
| `A` ... `H` | Option columns; unused options may be empty in historical data |
| `answer` | Gold letter selecting a non-empty option |
| `qa_id` | Unique question identifier |
| `canonical_video_id` | Stable source-video identifier |
| `clip_video` | Optional relative question-clip path |
| `qtype` | Optional fine-grained question type |
| `research_track_id` | Optional track label; final paper labels are `AC`, `PG`, `PD`, `SR` |

The validator requires the `A` through `H` columns, at least two non-empty
options, and a non-empty gold option. This accommodates the older mixed-choice
bundle. New output from the eight-choice builder has eight non-empty options.
Do not infer that passing the generic validator establishes the paper's exact
question count, option count, or human review status.

Package media paths are relative to the asset root. For a builder-produced
package this is the directory containing `manifest.tsv`. The validator rejects
absolute paths and parent traversal in media fields; `--asset-root` and
`--asset-mode` additionally check the selected files exist. Generated JSONL
uses selected QA fields and does not copy raw annotator/editor identity fields.
Local build and review reports can contain source paths and should remain local.

## Split construction

The final paper protocol separates training and benchmark examples by canonical
source video before constructing instructions and questions.
`data_processing/training/filter_train_eval_overlap.py` implements whole-source
exclusion. Its inputs must share canonical IDs; clip filenames alone cannot
establish source independence. Records whose source identity is unknown are
excluded and counted.

The historical `build_train_temporal_split.py` retains temporal complements of
reserved evaluation intervals. It remains available for reproducing that
experimental protocol, with an explicit distinction from whole-source
exclusion. It requires complete evaluation timestamps and constant-frame-rate
media for frame-indexed annotations.

The processing commands, schemas, review status behavior, and track-label
limitations are documented in [data_processing/README.md](../data_processing/README.md).
