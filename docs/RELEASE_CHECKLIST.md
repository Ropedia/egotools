# Release Checklist

This checklist tracks the remaining author-supplied resources and release steps.
The code version is `1.0.0`; a version number alone does not establish a public
release or reproduction of the paper experiments.

## Repository Contents

- Installable data preparation, evaluation, download, validation, and scoring
  modules under `src/egotools/`.
- Environment recipes, a training launcher, synthetic offline examples, and
  [measured validation records](VALIDATION.md).
- External MS-Swift and VLMEvalKit dependencies, with adapted-code notices in
  [Third-party software](THIRD_PARTY.md).
- Separate resource mappings for the pending final release and earlier
  experimental assets.

Dataset media, model weights, experiment outputs, participant records, annotation
submissions, and internal review systems are outside the source release.

## Before Public Release

- [ ] Select and add the project `LICENSE` file.
- [ ] Publish or identify the final 184,679-example training resource.
- [ ] Publish or identify the final 1,000-question benchmark resource.
- [ ] Publish or identify the final paper model checkpoint.
- [ ] Fill `configs/resources.yaml` and the blank README resource links with the
      confirmed final locations, keeping historical mappings separate.
- [ ] Confirm dataset/model licenses and access terms.
- [ ] Add the archival paper link and final citation metadata to `CITATION.cff`.
- [ ] Verify the final manifests, media references, and source-video split
      membership using the released metadata.
- [ ] Run inference on a fresh download of the final benchmark and update
      `docs/VALIDATION.md` with its actual scope and results.
- [ ] Review package contents and run the documented offline and development
      checks from a clean checkout.
- [ ] After author approval, publish the repository, annotated `v1.0.0` tag,
      and matching GitHub Release.

Final licensing, resource publication, and release timing remain author decisions.
