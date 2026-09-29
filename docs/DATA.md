# Data and Benchmark Format

EgoTools media, annotations, and model weights are distributed separately from
the code. The paper describes 646 egocentric videos totaling approximately
100 hours, 361,332 hierarchical captions, 6,519 tool-centric narrations,
184,679 source-video-disjoint instruction-tuning examples, and a 1,000-question
benchmark. The benchmark contains 900 human-authored questions and
100 human-verified spatial questions.

The final dataset and model locations are not yet set in
[`configs/resources.yaml`](../configs/resources.yaml). Earlier resources and
instructions for using them appear under [Historical resources](#historical-resources).
Those resources predate the paper's final training and benchmark versions.

## Resource Configuration

`egotools-download` reads `configs/resources.yaml` in a source checkout or its
packaged copy in an installed wheel. Resource locations can be supplied in a
local YAML file or overridden explicitly. For a repository available to you:

```bash
egotools-download benchmark \
  --repo-id YOUR_ORG/YOUR_DATASET --subdir benchmark \
  --metadata-only --output-dir data/huggingface
```

Replace the repository and subdirectory with the actual resource location;
`--subdir ''` selects the repository root. Omit `--metadata-only` to include
media, use `--revision` to pin a Hub revision, and use `--dry-run` to inspect the
download plan. Gated resources require access approval and `hf auth login` with
an approved account. Their own licenses and access terms apply.

## Benchmark Manifest

Evaluation consumes a UTF-8 TSV with one row per question:

| Column | Meaning |
| --- | --- |
| `index` | Unique row index |
| `video` | Relative full-video asset path |
| `question` | Multiple-choice question |
| `A` ... `H` | Option columns |
| `answer` | Gold letter selecting a non-empty option |
| `qa_id` | Unique question identifier |
| `canonical_video_id` | Stable source-video identifier |
| `clip_video` | Optional relative question-clip path |
| `qtype` | Optional fine-grained question type |
| `research_track_id` | Optional research track: `AC`, `PG`, `PD`, or `SR` |

A runnable dataset has this layout:

```text
benchmark/
├── manifest.tsv
├── videos/               # paths referenced by the video column
└── clips/                # optional paths referenced by clip_video
```

The validator requires the `A` through `H` columns, at least two non-empty
options, and a non-empty gold option. It accepts older mixed-choice data;
passing validation does not establish that a manifest is the final eight-choice
paper benchmark. The eight-choice builder produces eight non-empty options.
The [synthetic example](../examples/benchmark/manifest.tsv) illustrates the
schema without including video files.

```bash
egotools-validate-manifest /path/to/benchmark/manifest.tsv \
  --asset-root /path/to/benchmark --asset-mode full
```

Media paths are relative to the asset root. The validator rejects absolute paths
and parent traversal. `--asset-root` additionally checks that the selected files
exist. To combine a separate manifest and media directory, use
`egotools-materialize` as described in [Evaluation](EVALUATION.md#prepare-a-benchmark).

## Source-Video Separation

The paper separates training and benchmark data by canonical source video
before constructing instructions and questions. The
[`egotools.data.splits.filter_train_eval_overlap`](../src/egotools/data/splits/filter_train_eval_overlap.py)
module implements whole-source exclusion. It reads `canonical_video_id` or
`metadata.canonical_video_id`. Legacy `video_id` fields require an explicit
`--source-id-map` JSON object mapping those IDs to canonical IDs; filenames alone
do not establish source independence. Records with unresolved source identities
are excluded and counted.

Run source filtering on original annotations before preparing model-input SFT
JSONL, because SFT preparation removes provenance metadata. Generated benchmark
JSONL copies selected QA fields rather than raw annotator/editor identity fields.
Local build and review reports may contain source paths and should remain local.

The temporal split utility retains complements of reserved evaluation intervals.
It supports the earlier experimental protocol and is distinct from whole-source
exclusion. Frame-indexed annotations require complete evaluation timestamps and
constant-frame-rate media. See [Data Processing](DATA_PROCESSING.md) for these
commands, input schemas, and review-status handling.

## Historical Resources

The following repositories were used for earlier experiments and software
validation. Their mapping is retained in
[`configs/resources.preview.yaml`](../configs/resources.preview.yaml), which
must be selected explicitly with `--config`. They are not the final
184,679-example / 1,000-question paper release.

| Resource | Repository | Scope |
| --- | --- | --- |
| Added training and benchmark data | [egotools_v4_backfilled_sft_v5_902_20260623](https://huggingface.co/datasets/egotools-dev/egotools_v4_backfilled_sft_v5_902_20260623) | 172,118-example SFT and 902-question benchmark; additions to earlier media |
| Benchmark base | [egotools_bench_v5_20260505](https://huggingface.co/datasets/egotools-dev/egotools_bench_v5_20260505) | Earlier benchmark media |
| Training base | [egotools_train_time_excluded_20260504](https://huggingface.co/datasets/egotools-dev/egotools_train_time_excluded_20260504) | Earlier training media and temporal-exclusion protocol |
| Model checkpoint | [egotools-8b-v3_3](https://huggingface.co/egotools-dev/egotools-8b-v3_3) | Earlier 116,031-example training run |

The benchmark base and June addition require manual Hub access approval. Request
access on both dataset pages and authenticate with the approved account. The
training base is ungated, but the training addition uses the gated June
repository. A public file listing does not grant permission to download files.

### Historical Benchmark

Download the base into the directory expected by the later manifest, then add
the June bundle at the enclosing root:

```bash
egotools-download benchmark-base \
  --config configs/resources.preview.yaml \
  --output-dir data/huggingface/benchmark/v5_686_plus_jskim216

egotools-download benchmark \
  --config configs/resources.preview.yaml \
  --output-dir data/huggingface

egotools-prepare-manifest \
  data/huggingface/benchmark/v5_686_plus_jskim216/final.tsv \
  --output data/huggingface/benchmark/v5_686_plus_jskim216/manifest.tsv

egotools-validate-manifest \
  data/huggingface/benchmark/v5_686_plus_jskim216/manifest.tsv \
  --expected-rows 902 \
  --asset-root data/huggingface/benchmark/v5_686_plus_jskim216 \
  --asset-mode full
```

The historical `final.tsv` repeats some integer indices. Preparation writes
sequential indices in a separate TSV while preserving `qa_id`, questions,
answers, and media paths. The combined Hub inventory covers all 902 full-video
references and 899 of 902 clip references. Use full-video evaluation for this
bundle; a non-empty missing clip path causes clip-mode evaluation to fail.
File inventory checks do not establish that every video decodes correctly.

### Historical Training Data

Merge the training base and additional media into one root:

```bash
egotools-download training-base \
  --config configs/resources.preview.yaml \
  --output-dir data/huggingface-training

egotools-download training \
  --config configs/resources.preview.yaml \
  --output-dir data/huggingface-training

egotools-prepare-sft \
  data/huggingface-training/sft/v4_all_backfilled/final.jsonl \
  --output data/huggingface-training/train.swift.jsonl
```

The 172,118-row JSONL references 405 unique relative video paths, all present in
the combined repository inventories. Media paths resolve from the shared root,
not the nested annotation directory. Run source filtering first if needed;
the command above prepares the historical mixture as distributed.

The raw JSONL contains heterogeneous nested metadata and failed the pinned
MS-Swift dataset loader after 5,766 examples. The prepared copy loaded all
172,118 rows successfully. Preparation preserves model message and media inputs;
it does not crop clips using `start_frame`/`end_frame` or establish a source
split. See [Training](TRAINING.md#prepare-training-inputs) for launch-directory
and input-format details.

### Historical Checkpoint

```bash
egotools-download model \
  --config configs/resources.preview.yaml \
  --output-dir checkpoints/preview
```

The earlier checkpoint's `args.json` reports effective batch 128 through
per-device batch 2 and accumulation 8. Its repository root is an intermediate
checkpoint from a 116,031-example run. It can be inspected or used with the
Qwen3-VL-8B preset; it does not reproduce the final paper model.

### Reconstruction Limits

Consuming these packaged files is different from rebuilding their curation.
Measured checks on the historical metadata found:

| Task | Available evidence and missing inputs |
| --- | --- |
| Prepare the 902-question manifest | Unique `qa_id` values; integer indices need preparation |
| Verify training media references | 172,118 rows and 405 relative paths; inventories cover all paths, without full video decoding |
| Normalize earlier reviewed QA | The 686-row eight-option `normalized.jsonl` lacks canonical IDs for 42 rows, causing direct normalization to fail |
| Rebuild source-based QA | Required per-source `annotation_manifest.json` files are absent; all 686 normalized rows have empty `source_file` and `source_id` |
| Reapply human review decisions | Required append-only review edit log is absent |
| Reconstruct temporal exclusions | Complete benchmark timestamps, source IDs, and original frame-indexed annotations are absent |
| Reconstruct the final paper mixture | Final split membership, a complete source-ID crosswalk, and generation inputs are unavailable in these earlier bundles |

For source filtering, only 56,087 of the 172,118 training rows contain explicit
canonical IDs; the other 116,031 contain legacy IDs in a different namespace.
The conservative whole-source filter retained 56,087 rows and excluded the
unresolved rows. A partial crosswalk derived from 158 explicit ID pairs
increased the retained subset to 91,331 rows, leaving 80,787 unresolved. Neither
result reconstructs the paper mixture. A zero-overlap result that ignores
unresolved IDs is not evidence of complete source separation.

Rule-based track assignment was exercised on the older 686-row QA file, but its
output is not the paper's curated labeling. The processing examples describe how
to run the tools when the required inputs are supplied. [Validation](VALIDATION.md)
records the measured checks and their limits.
