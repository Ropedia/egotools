# EgoTools evaluation adapter

`vlmeval_ext/` integrates a local EgoTools manifest with an external VLMEvalKit
installation. `scripts/run_eval.py` supports CPU mock checks, a command dry run,
and real-model inference with explicit dataset, checkpoint, and sampling options.

```bash
python -m pip install -e '.[eval]'
python eval/scripts/run_eval.py --help
```

See [the evaluation guide](../docs/EVALUATION.md) for data preparation, dependency
installation, model runs, and scoring. VLMEvalKit, downloaded media, checkpoints,
and generated results are excluded from Git.
