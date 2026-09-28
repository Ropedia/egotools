# Measured reproduction status

Checked on 2026-09-28. This candidate supports some tested execution paths, but
does not yet reproduce the final paper. Unit tests and mock predictions alone
are not evidence that the real data, frameworks, training, and evaluation work
together. The checks below distinguish those cases.

## Actual execution

| Path | What was actually run | Result and limit |
| --- | --- | --- |
| Source checkout and lightweight install | A new GitHub clone, separate Python 3.10 venv, README installation and commands | Installation, all 57 original tests, metadata download, 902-row preparation, validation, and mock evaluation passed |
| Current code tests | Synthetic regression and subprocess tests after the real-data fixes | 67 tests passed; Ruff and shell syntax checks passed |
| Python distribution | Wheel and source distribution; wheel installed outside the checkout | Configuration loading and the four utility entry points passed |
| Dataset access | Anonymous download of an actual benchmark MP4, followed by authorized download | Anonymous request returned HTTP 401; the approved local account downloaded both test videos |
| Evaluation setup | New conda environment, independent clone of pinned VLMEvalKit, actual setup script | Required imports and CUDA passed; user-site isolation was corrected and four borrowed packages were installed into the environment |
| Real model inference | Qwen3-VL-8B-Instruct, two downloaded benchmark videos, 64 frames, one RTX 6000 Ada 48 GB, BF16/SDPA | Both predictions completed; 0/2 correct and 100% answer extraction. This tiny sample establishes execution, not benchmark quality |
| Result aggregation | The actual upstream prediction table and its model-root symlink | Initially failed by counting the symlink as another prediction; corrected aggregation processed the same predictions successfully |
| Full SFT ingestion | Actual pinned ms-swift loader on all 172,118 public records | Original file failed after 5,766 rows because nested metadata schemas differ; the prepared file loaded all 172,118 rows in 3.979 seconds |
| SFT conversion | Streaming conversion of all public records, followed by row-by-row comparison | Messages, videos, and order matched for all 172,118 records; non-model provenance columns removed |
| Actual training | Qwen3-VL-2B-Instruct, one public example/video, one GPU, BF16/SDPA, ZeRO-3, 64 frames, one full-language-model optimizer step | Exit 0; loss 2.73193479, gradient norm 78.9930412, peak 35.84 GiB. This used a Python 3.10 dependency overlay, not the full documented Python 3.11 setup |
| Default training attention | Actual launcher with its FlashAttention setting in the temporary training environment | Failed because `flash_attn` was absent; the successful smoke used an explicit SDPA override |
| Source-video filtering | Full actual SFT against the 902-question benchmark | Original code mixed legacy and canonical IDs. Corrected default retained 56,087 explicitly identified rows and excluded 116,031 unresolved rows; it does not reconstruct the paper mixture |
| Data authoring scripts | Actual released historical 686-row QA data and 902-row benchmark metadata | Track assignment ran on 686 rows; normalization, source construction, and temporal reconstruction could not run to completion because required source fields and timestamps are missing |

The fixed-letter CPU mock also completed on all 902 questions: both scoring
paths returned 129/902 for the fixed answer `A`, matching the gold-answer count.
That mock neither decodes videos nor performs model inference.

## Fixes required by the real attempts

- Evaluation aggregation now recognizes an upstream prediction file and its
  symlink as the same file. Distinct prediction tables remain ambiguous.
- Evaluation setup disables Python's user site during installation and saves
  that setting for conda activation. The original installation had silently
  borrowed four dependencies from `~/.local`.
- `training/prepare_sft.py` creates a model-input-only JSONL before Arrow loading.
  Keep the original metadata for source auditing and filter sources before
  stripping that metadata. The converter does not trim videos or reconstruct
  train/test separation.
- The source filter now requires explicit canonical IDs or an explicit mapping
  from old IDs; it no longer treats both namespaces as interchangeable.

## Environment and verification limits

The new evaluation environment uses Python 3.10.21, PyTorch 2.6.0+cu124,
torchvision 0.21.0, Transformers 5.6.2, qwen-vl-utils 0.0.14, Pandas 2.3.3,
NumPy 2.2.6, Pillow 12.3.0, PyAV 17.1.0, and Decord 0.6.0. After isolation,
`pip check` has no missing or conflicting requirements, but still reports
Decord's upstream wheel-platform metadata warning. Imports and actual video
decoding worked; the wheel metadata was not modified to hide that warning.

The inference weights were the existing local cache of the public base model,
not the final EgoTools paper checkpoint. The training smoke deliberately used
2B, one GPU, accumulation 1, `max_steps=1`, SDPA, and no checkpoint saving.
The default recipe is 8B, eight GPUs, accumulation 16, FlashAttention, and one
epoch. The small test does not verify that full recipe, checkpoint save/resume,
or the final trained model's reported accuracy.

Only two benchmark videos and one training example's video were decoded for
the real-model checks. Public inventories cover 902/902 `video` references,
899/902 optional `clip_video` references, and 405/405 distinct staging SFT video
references. Those counts do not prove all media can be decoded. Downloads
used an already approved account; benchmark base and the June increment are
manually gated.

The final 184,679-example mixture, 1,000-question benchmark, matching checkpoint,
complete source-ID crosswalk, and original curation inputs are still needed.
External EgoSchema/EgoThink runners and the final manually curated research
track labels are not reconstructed by this repository. Full multi-GPU training,
complete benchmark evaluation, external API evaluation, and comparison with all
paper tables remain unverified. See [DATA.md](DATA.md) and
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
