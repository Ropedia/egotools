# Release checklist

This document records what the code candidate contains and what remains before
the authors' planned public paper release.

## Completed for private staging

- The `main` branch is a new root commit with no private development history.
- Dataset media, model weights, logs, cached results, identities, assignments,
  annotation submissions, and internal web applications are excluded.
- `ms-swift` and `VLMEvalKit` are external dependencies rather than vendored
  repository copies.
- Code version metadata is set to `1.0.0`.
- Existing Hugging Face repositories are linked and labeled as pre-release
  resources where their counts differ from the paper.

## Required before public `v1.0.0`

- [ ] Select and add the project `LICENSE` file.
- [ ] Publish or identify the final 184,679-example training repository.
- [ ] Publish or identify the final 1,000-question benchmark repository.
- [ ] Publish or identify the final paper model checkpoint.
- [ ] Update `configs/resources.yaml`, README resource links, and expected row
      counts to those final repositories.
- [ ] Confirm the final dataset/model licenses and access terms.
- [ ] Add the archival paper URL and final BibTeX entry to `CITATION.cff`.
- [ ] Run the smoke evaluation against a fresh download of the final benchmark.
- [ ] After the private review, change repository visibility and create annotated tag
      `v1.0.0` and the matching GitHub Release.

The repository stays private during this review, as requested. Selecting a
project license and publishing the formal release remain author decisions.
