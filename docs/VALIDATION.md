# Code candidate validation

Checked on 2026-09-28. These checks validate the published code paths; they do
not constitute a rerun of the paper's model experiments.

## Checks performed

- `python -m pytest -q`: 57 tests passed in a fresh lightweight Python 3.13
  environment. Tests use synthetic records and subprocesses, including a real
  FFmpeg trim of generated long-GOP video.
- `ruff check .`: passed. Python package wheel and source distribution build
  successfully. The wheel includes the resource configuration and retained
  third-party license notice.
  A separate wheel installation, run outside the source checkout, loaded the
  bundled resource mapping and exercised all four command-line entry points.
- The downloader fetched the existing public benchmark metadata. Its original
  902-question TSV failed validation because integer indices repeat. Preparing
  a separate TSV with sequential indices preserved the questions and `qa_id`
  values; the resulting 902-row manifest passed validation.
- A complete CPU mock run on that manifest emitted the fixed letter `A` for
  all 902 questions. The adapter and independent scoring CLI both returned
  **129/902 (14.301552%)**, matching the manifest's count of gold `A` answers.
  No video decoding or model inference is involved in this result.
- Public Hub file inventories were checked without downloading the full media
  collection. Combining the base and incremental benchmark repositories covers
  all 902 `video` references and 899 of 902 `clip_video` references. File presence
  does not establish decode quality.
- Training shell syntax, no-dependency dry-run, argument quoting, command replay,
  environment overrides, and failure propagation through logging passed.
  Linux x86_64 / Python 3.11 dependency resolution succeeded for 155 packages.
  This resolution was diagnostic output, not a new lock file.
- The pinned Qwen video-sampling implementation was exercised on CPU. With a
  10-second 30-FPS source, the maximum-only setting gives 20 samples, whereas
  minimum and maximum 64 gives 64. A source containing just 30 frames remains
  clamped to 30 frames.
- VLMEvalKit registration was exercised in an existing Python 3.10 evaluation
  environment against a pristine export of the pinned upstream Git revision.
  The real `VideoBaseDataset` and the custom 64-frame dataset alias loaded.
  The same integration decoded eight frames from a generated two-second MP4
  and built the EgoTools prompt. The upstream launcher help path also ran.

## Remaining verification

No complete fresh CUDA installation, FlashAttention build, full SFT run,
external model API call, or paper-model GPU evaluation was performed. The final
184,679-example training data, 1,000-question benchmark, and matching checkpoint
are still needed for that comparison.

The repository provides the EgoTools benchmark runner. It does not bundle the
historical machine-specific launch queues, saved experiment results, or external
EgoSchema/EgoThink benchmark runners. The descriptive track-assignment utility
does not recreate the final paper's manually curated track labels.

See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the remaining author decisions
and final resource links before the planned public release.
