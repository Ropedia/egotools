# Data and Benchmark Format

EgoTools media, annotations, and model weights are distributed separately from
the code. The paper describes 646 egocentric videos totaling approximately
100 hours, 361,332 hierarchical captions, 6,519 tool-centric narrations,
184,679 source-video-disjoint instruction-tuning examples, and a 1,000-question
benchmark. The benchmark contains 900 human-authored questions and
100 human-verified spatial questions.

The paper links to [EgoTools-Data](https://huggingface.co/datasets/ropedia-ai/egotools-data)
and [EgoTools-8B](https://huggingface.co/ropedia-ai/egotools-8b). As of October 2,
2026, the public data repository contains 131 recording episodes under
`raw/<uuid>/ep1/`, with videos, captions, and sensor annotations, and the final
SFT export under `sft/` (see [Training](TRAINING.md#final-training-data)).
`raw/manifest.json` is an asset inventory; the repository does not supply the
benchmark question table required by the commands below. The model repository returned HTTP 401
to an anonymous metadata request, so public checkpoint access was not verified.
Earlier runnable resources appear under [Historical resources](#historical-resources).

## Resource Configuration

`egotools-download` reads `configs/resources.yaml` in a source checkout or its
packaged copy in an installed wheel. Resource locations can be supplied in a
local YAML file or overridden explicitly. The model mapping is
`ropedia-ai/egotools-8b`, and the training mapping is the `sft/` folder of
EgoTools-Data described in [Training](TRAINING.md#final-training-data); the benchmark
mapping remains blank pending its question-table export. With access to the model
repository, download it with:

```bash
egotools-download model --output-dir models/egotools-8b
```

For a dataset repository available to you:

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

The paper uses the rectified front-left RGB stream as the canonical view:
1024 × 1024 pixels at 20 FPS. Point the manifest at those prepared benchmark
videos. The multi-camera recording assets in the current public repository
are not a substitute for the paper's processed video and question release.

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
paper benchmark. The eight-choice builder produces eight non-empty options,
and `egotools-evaluate` requires all eight for the paper evaluation protocol.

```bash
egotools-validate-manifest /path/to/benchmark/manifest.tsv \
  --asset-root /path/to/benchmark --asset-mode full
```

Media paths are relative to the asset root. The validator rejects absolute paths
and parent traversal. `--asset-root` additionally checks that the selected files
exist. To combine a separate manifest and media directory, use
`egotools-prepare-benchmark` as described in [Evaluation](EVALUATION.md#prepare-a-benchmark).

## Source-Video Separation

The paper separates training and benchmark data by canonical source video.
Filter original annotations before `egotools-prepare-sft`, which removes the
provenance needed for filtering. See [Data Processing](DATA_PROCESSING.md) for
canonical-ID mappings, unresolved-source handling, and the distinct experimental
temporal-split protocol. Keep local build and review reports out of data releases;
they may contain source paths.

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

egotools-prepare-benchmark \
  --manifest data/huggingface/benchmark/v5_686_plus_jskim216/final.tsv \
  --assets data/huggingface/benchmark/v5_686_plus_jskim216 \
  --output data/benchmark --reindex
```

The historical `final.tsv` repeats some integer indices. `--reindex` writes
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

The raw JSONL's heterogeneous metadata requires `egotools-prepare-sft` before
loading with MS-Swift. This conversion does not crop clips or establish a source
split. See [Training](TRAINING.md#prepare-training-inputs) for input-format and
launch-directory details.

### Historical Checkpoint

```bash
egotools-download model \
  --config configs/resources.preview.yaml \
  --output-dir checkpoints/preview
```

This checkpoint comes from an earlier 116,031-example run. Use it with the
Qwen3-VL-8B preset; it is not the final paper model.

### Reconstruction Limits

The historical bundles can be consumed as packaged, but lack inputs needed to
reconstruct their curation or the final paper release:

| Task | Missing or incomplete inputs |
| --- | --- |
| Normalize or rebuild source-based QA | Complete canonical IDs, `source_file`/`source_id`, and per-source `annotation_manifest.json` files |
| Reapply human review | Append-only review edit log |
| Reconstruct temporal exclusions | Complete benchmark timestamps, source IDs, and original frame-indexed annotations |
| Reconstruct the final paper mixture | Final split membership, complete source-ID crosswalk, and generation inputs |

Source filtering excludes unresolved IDs; zero overlap in the remaining subset
does not establish complete source separation. Rule-based track assignments are
curation suggestions, not the paper's final manually assigned labels. See
[Data Processing](DATA_PROCESSING.md) for commands to use when the required
source inputs are available.
