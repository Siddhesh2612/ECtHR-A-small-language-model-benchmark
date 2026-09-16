# Project status

Updated: 16 September 2026.

## Completed

- Phase 1: curated 21 distinct historical notebook versions, preserved source-labelled results, documented discrepancies and kept the private archive outside Git.
- Phase 2: implemented and verified ECtHR-A preparation for JSONL and Parquet, official labels, split-size checks, provenance hashes and duplicate reporting.
- Phase 3: implemented configuration-driven full fine-tuning, validation-only checkpoint selection, separate held-out evaluation, LexGLUE-style metrics and an offline synthetic smoke workflow.
- Repository setup: Python packaging, pinned verified environment snapshot, VS Code tasks, learning documentation, unit tests, publication checks and a lightweight GitHub Actions workflow.
- Repository publication: public GitHub remote, Apache-2.0 licensing for original contributions, citation metadata and contribution guidance.
- Phase 4 pilot infrastructure: deterministic private train/validation subsets, immutable GPT-Neo 125M revision, hard workload ceilings, explicit wall-time budget, recorded status/logs and no pilot access to final test evaluation.

The Phase 1–3 repository has been published on `main` at [Siddhesh2612/ECtHR-A-small-language-model-benchmark](https://github.com/Siddhesh2612/ECtHR-A-small-language-model-benchmark). GitHub publication is a software milestone, not a reproduction of the dissertation results. The repository includes Apache-2.0 licensing for original contributions, citation metadata and contribution instructions; external datasets, model weights and third-party materials retain their own terms.

## Current evidence

- 29 unit tests pass, including bounded-pilot and deterministic-subset tests.
- The offline train/save/reload/evaluate smoke workflow passes.
- The pretrained pilot plan command passes without downloading a model, preparing data or training.
- The real local ECtHR-A splits prepare as 9,000/1,000/1,000 rows.
- Exact cross-split text overlap: zero.
- Within-split repeated text: one additional training row, retained and reported; labels match.
- Publication-boundary scan passes for the full Git candidate set.

See `verification.md` for what these checks establish and their limits.

## Next milestone

Execute and inspect the bounded pretrained pilot before any full GPU run:

1. Review the pinned model, 16/8 development subset, 128-token context, two-step limit and 30-minute maximum budget.
2. Run the pilot through `scripts/run_pretrained_pilot.py --budget-seconds 1800 --execute`.
3. Inspect the status, log, resolved configuration, subset manifest, loss, validation predictions and truncation behavior.
4. Record observed runtime and memory on the chosen hardware.
5. Adjust and rerun a new pilot without using the test split.
6. Write a separate full-experiment configuration only after the pilot path is verified.

LoRA and quantization follow only after the full-fine-tuning baseline path is verified with pretrained weights. This makes later comparisons interpretable.
