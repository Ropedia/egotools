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

The benchmark base and June addition require manual Hugging Face access
approval. Request access on both dataset pages while signed into the account
that will download them, then run `hf auth login` locally with that approved
account. The training base is ungated, but the training addition uses the same
manually gated June repository. Public file listings do not establish download
access: an anonymous attempt to download a benchmark-base MP4 returned HTTP 401.

The June bundle is an addition to earlier media collections. After access is
approved, download the benchmark base into the directory where the later
manifest expects its media, then download the addition into the enclosing
Hub root:

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
A full metadata scan found 172,118 training rows and 405 unique relative video
references. All 405 paths appear in the combined base/addition Hub file
inventory. This verifies paths and counts; it does not verify all video bytes
or decoding. Training media references resolve from that shared root, not
from the nested JSONL directory. See [TRAINING.md](TRAINING.md) for model-input
preparation and launcher usage.

Run any source-ID filtering on the original JSONL first; model-input
preparation removes the metadata that source filtering needs. The distributed
JSONL has heterogeneous nested metadata and failed the pinned MS-Swift loader
after 5,766 examples. Prepare a uniform model-input JSONL before training:

```bash
python training/prepare_sft.py \
  data/huggingface-training/sft/v4_all_backfilled/final.jsonl \
  --output data/huggingface-training/train.swift.jsonl
```

The prepared file was loaded successfully for all 172,118 rows by the pinned
MS-Swift dataset loader. Preparation preserves model message/video inputs; it
does not apply `start_frame`/`end_frame`, crop media, or establish a source split.
Using a filtered input changes the resulting row count. The example above
prepares the historical mixture exactly as distributed.

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
exclusion. It reads `canonical_video_id` or `metadata.canonical_video_id`;
legacy `video_id` fields are used only with an explicit `--source-id-map` JSON
object mapping those IDs to canonical IDs. Clip filenames alone cannot
establish source independence. Records whose source identity is unknown are
excluded and counted.

The historical 172,118-row SFT file does not provide canonical IDs for 116,031
rows. Those rows carry legacy `metadata.video_id` values in another namespace.
Only 56,087 rows have explicit canonical IDs. Running the corrected filter on
the full SFT file against the current 902-question benchmark retained those
56,087 rows and excluded 116,031 unresolved rows. No known canonical IDs in
that retained subset matched benchmark IDs. This is a partial, conservatively
filtered dataset, not a reproduction of the 184,679-example paper mixture.
A complete source-ID crosswalk is needed to decide the remaining rows; the
release does not infer one from truncated or transformed filenames. Joining
explicit `video_id`/`canonical_video_id` pairs already present in the same SFT
file provides 158 source mappings. A second actual run with that partial
mapping retained 91,331 rows and still excluded 80,787 unresolved rows; the
available metadata therefore does not supply a complete crosswalk.

The historical `build_train_temporal_split.py` retains temporal complements of
reserved evaluation intervals. It remains available for reproducing that
experimental protocol, with an explicit distinction from whole-source
exclusion. It requires complete evaluation timestamps and constant-frame-rate
media for frame-indexed annotations.

The processing commands, schemas, review status behavior, and track-label
limitations are documented in [data_processing/README.md](../data_processing/README.md).


## What can be reconstructed from the available files

The following checks used the actual Hub metadata, not only synthetic fixtures.
They distinguish consuming a packaged dataset from reconstructing its curation.

| Step | Required inputs | Current availability and actual check |
| --- | --- | --- |
| Prepare the historical benchmark | Current `final.tsv` | Available after repository access; 902 rows and unique `qa_id` values; integer indices need preparation |
| Use historical training media paths | SFT JSONL plus base/addition media | 172,118 rows scanned; all 405 referenced paths are listed in the combined repositories |
| Normalize existing reviewed QA | QA texts, options, canonical IDs | The older public `normalized.jsonl` has 686 eight-option rows, but 42 lack canonical IDs; direct normalization fails on those records |
| Rebuild a benchmark from original source annotations | Per-source `annotation_manifest.json`, source filenames/IDs, media | Per-source manifests are absent from the configured repositories; all 686 rows of the older normalized file have empty `source_file` and `source_id`; source-based build fails |
| Reapply human review decisions | Append-only review edit log | The required edit log is absent from the configured repositories |
| Suggest research tracks | QA texts/options | The rule-based script ran on all 686 older normalized rows; this does not reproduce the paper's curated track labels |
| Reconstruct the temporal split | Complete benchmark source timestamps, canonical source IDs, original frame-indexed annotations | The required complete timestamp and source-ID inputs are absent from the configured repositories |
| Reconstruct the final paper mixture | Final split membership, complete source-ID mapping, generation inputs/recipe | The configured resources predate the final paper mixture; the released utilities and historical data do not reproduce its 184,679 rows |

Some historical summaries mention internal source files and an old pending
upload state. The measured Hub inventory is the evidence for current file
presence; internal path strings in those summaries are not usable downloads.
The source-processing CLI examples in `data_processing/README.md` document how
to use those tools when the required inputs are supplied. They are not a claim
that an external user can currently rebuild the complete dataset from the
configured Hub resources.
