# Data processing

These scripts expose the reusable QA and split-processing steps from the
research workspace. The repository contains code and prompt templates; source
annotations, review logs, manifests, participant information, and media are
supplied separately. Run the commands from a source checkout. Every input and
output path is explicit; no private directory layout is assumed. The currently
configured Hub resources do not include every source input needed to recreate
the original curation. See the availability table in `docs/DATA.md` before
attempting source reconstruction.

Offline normalization, track assignment, review extraction, and benchmark
construction use the Python standard library. For optional Gemini processing,
install `python -m pip install -e '.[data]'` and export `GEMINI_API_KEY`. An
explicit `--env-file` is also supported. Video frame sampling and temporal
trimming require `ffmpeg` and `ffprobe` on `PATH`.

## Normalize questions to eight options

Each input JSONL row is a flat object or `{ "qa_id": ..., "source_file": ...,
"record": {...} }`. The QA object supplies:

| Field | Meaning |
| --- | --- |
| `qa_id` | Unique question ID |
| `canonical_video_id` | Source-video ID shared by all derived examples |
| `annotation_id` | Annotation ID used for clip lookup |
| `source_file` | Source filename such as `source001.json` |
| `question` | Question text |
| `answer` | Correct answer text |
| `distractors` | List of incorrect answer texts |

Flat rows with `normalized_options` and a zero-based `correct_index` are also
accepted. For nested rows, source IDs may be carried in `record._video`.

```bash
python -m data_processing.benchmark_construction.normalize_to_8choice \
  --input-jsonl /path/to/input/qa.jsonl \
  --output-dir /path/to/normalized \
  --mode checks_only
```

`checks_only` makes no API calls. It writes valid existing eight-option rows
to `normalized.jsonl`, rejected rows to `skipped.jsonl`, and reasons and counts
to separate report files. It does not invent missing options. `--dry-run`
reports counts without writing files. The output directory must be outside the
input file's directory tree, preserving the existing source-data safeguard.

`--mode auto` uses Gemini to augment, validate, or reduce options. Optional
`--reasons-csv` supplies `qa_id`, `decision`, and `reason` review flags. Video
evidence uses explicit `clip_path` or `source_video_path` fields, resolved
relative to `--media-root` or the input JSONL directory. `clip_path` denotes
an already trimmed clip; source timestamps apply to `source_video_path`.
Missing video evidence falls back to text-only processing. `--resume` skips
question IDs already written; use a new output directory after changing a QA.
Generated questions still need human review. `--relaxed` explicitly retains
some quality violations and records them in the output.

## Build and review a benchmark

Media lookup uses a directory containing one manifest per source:

```text
sources/
  source001/
    annotation_manifest.json
    full.mp4
    clip001.mp4
```

For a QA with `source_file="source001.json"`, the corresponding
`annotation_manifest.json` can contain:

```json
{
  "canonical_video_id": "video001",
  "source_video_path": "full.mp4",
  "annotations": [
    {"annotation_id": "a001", "clip_path": "clip001.mp4"}
  ]
}
```

Paths in this source manifest are absolute or relative to the manifest's own
directory. Output manifests contain relative media paths.

```bash
python -m data_processing.benchmark_construction.build_benchmark \
  --normalized-run /path/to/normalized \
  --workspace-sources-root /path/to/sources \
  --output-root /path/to/builds \
  --version review --date 20260101
```

The package contains `manifest.tsv`, a JSONL with selected QA fields,
`videos/`, `clips/`, and local build reports. Missing full videos exclude the
corresponding questions; missing clips are reported and leave `clip_video`
empty. `--link-mode copy` is the default and creates a self-contained package;
symlinks depend on the original media. `--dry-run` estimates media size without
writing. The existing 600 GiB materialization limit is retained and exposed as
`--max-copy-gib` for the target disk. Local build reports contain diagnostic
source paths and are not release annotations.

Review extraction takes that package's JSONL and an append-only edit JSONL.
Edits are keyed by `qa_id`, with `saved_at_utc`, optional `edited_question`,
`edited_options`, zero-based `edited_correct_index`, and `review_status`
(`clean`, `pending`, or `drop`). The latest timestamp wins; file order breaks
ties. The extractor preserves status and omits annotator/editor identity
fields. It writes validation results and exits nonzero if required QA/media
checks fail.

```bash
python -m data_processing.benchmark_construction.extract_final_reviewed_qa \
  --manifest /path/to/builds/review_20260101/manifest.jsonl \
  --package-dir /path/to/builds/review_20260101 \
  --edits /path/to/edits.jsonl \
  --output /path/to/reviewed.jsonl \
  --report /path/to/review-report.json

python -m data_processing.benchmark_construction.assign_benchmark_research_tracks \
  --input /path/to/reviewed.jsonl \
  --output /path/to/tracked.jsonl \
  --report /path/to/track-report.json \
  --exclude-dropped

python -m data_processing.benchmark_construction.build_benchmark \
  --input-jsonl /path/to/tracked.jsonl \
  --workspace-sources-root /path/to/sources \
  --output-root /path/to/builds \
  --version reviewed --date 20260101
```

The rule-based track script produces descriptive taxonomy IDs such as
`tool_perception_state_grounding` and `spatial_intelligence`, plus subtracks
and capability axes. These are research curation suggestions; they are not
an automatic reconstruction of the final paper's curated `AC`, `PG`, `PD`,
and `SR` labels. The builder preserves whichever track fields the input
contains and excludes records marked dropped or `keep_for_benchmark=false`.
Pending or unreviewed status alone does not exclude a row: select the intended
review statuses before a final build.

## Exclude benchmark source videos from training

This is the source-video-disjoint split used for the paper protocol:

```bash
python -m data_processing.training.filter_train_eval_overlap \
  --eval-dir /path/to/benchmark \
  --input-jsonl /path/to/training/train.jsonl \
  --output-root /path/to/train-disjoint
```

The evaluation directory supplies `manifest.tsv`. Training rows use the same
canonical source ID in `canonical_video_id` or
`metadata.canonical_video_id`. A legacy `video_id` or `metadata.video_id` is
ignored unless `--source-id-map /path/to/mapping.json` explicitly maps it to a
canonical ID. The mapping is a JSON object such as
`{"legacy-source-001": "canonical-video-001"}` and must come from source
metadata. Filename or clip-ID inference is not performed. Rows without a
resolved canonical source ID are excluded and counted.

In the distributed historical 172,118-row SFT file, 116,031 rows lack canonical
IDs; the default filter excludes them and retains 56,087 rows with known IDs.
This measured result is a partial filtered subset, not the paper training
mixture. See `docs/DATA.md` for the missing source inputs and access requirements.
Repeat `--input-jsonl` for files with distinct basenames. Media references in
retained records remain unchanged.

Optional `--captions-dir` and `--videos-dir` copy source assets named
`<canonical_video_id>.json`, `<canonical_video_id>.mp4`, and `<ID>_clips/`.
Only assets for canonical IDs retained in the input records are copied.
Use a fresh asset output directory so files from an earlier split cannot remain
in the result. Generated training JSONL and reports are written directly under
`--output-root`.

## Experimental temporal exclusion

`build_train_temporal_split.py` supports the historical temporal protocol.
It excludes reserved intervals while retaining other intervals of the same
source. This is not source-video separation and does not reproduce the final
paper training split.

```bash
python -m data_processing.training.build_train_temporal_split \
  --eval-dir /path/to/benchmark \
  --benchmark-full /path/to/qa-timestamps.jsonl \
  --captions-dir /path/to/captions \
  --videos-dir /path/to/videos \
  --input-jsonl /path/to/train.jsonl \
  --output-root /path/to/temporal-split
```

The timestamp input is JSONL or a JSON array with `qa_id`,
`clip_start_seconds`, and `clip_end_seconds` for every benchmark QA. Filenames
and training source IDs use canonical video IDs. Caption JSON contains a
`segments` list with one-based inclusive `start_frame`/`end_frame`; training
rows carry the same fields at the top level or in `metadata`. Constant frame
rate is required. Unknown source IDs, missing ranges on overlapping sources,
and intervals that cross reserved evaluation spans are excluded.

Retained intervals of at least one second are re-encoded to H.264/AAC for
accurate boundaries; videos with no reserved intervals are symlinked. Rows are
updated to `videos/train/` paths relative to the output root, and frame indices
are shifted for trimmed segments. Cleaned captions retain their original
source frame coordinates. Reports include source-to-segment mappings. The
output is for local processing; symlinks must be materialized for distribution.
