# Contributing

This project combines an MSc dissertation archive with a new experiment pipeline.
Read the [methodology](docs/methodology.md), [verification record](docs/verification.md)
and [historical catalogue](docs/experiment_catalogue.md) before changing research
code.

## Development workflow

1. Create a focused branch from `main`.
2. Follow the [README setup](README.md#setup) and activate `.venv`.
3. Make a small change with a clear reason and update the relevant documentation.
4. Run the applicable checks below and inspect `git diff` and `git status`.
5. Open a pull request describing the change, checks run and research impact.

```sh
python -m unittest discover -s tests -v
python scripts/check_publication.py
```

Also run `python scripts/smoke_test.py` after changing training, evaluation or
checkpoint loading. This uses a small synthetic model/data workflow; it does not
establish pretrained-model performance. Include additional targeted checks when
new behavior is not covered. Full training needs an explicit experiment and
compute budget; continuous integration does not launch GPU experiments.

## Preserve the research record

- Keep historical reported results separate from new measured results.
- Do not silently correct old scores or claim historical runs were reproduced.
- Preserve train/validation/test assignments, label IDs and dataset provenance.
- Select settings and checkpoints on validation; reserve test for final evaluation.
- Record experiment configuration, model revision, seed and hardware when reporting new results.
- Describe unsuccessful experiments and missing evidence as carefully as successful ones.

Never commit credentials, case records, private audit files, environments, model
weights or run artifacts. Use the ignored local directories documented in the
README. Include source attribution for adapted code and respect upstream terms.
Original contributions are submitted under the repository's Apache-2.0 license;
third-party materials retain their own licenses. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
