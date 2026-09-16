# Project status

Updated: 16 September 2026.

## Completed

- Phase 1: curated 21 distinct historical notebook versions, preserved source-labelled results, documented discrepancies and kept the private archive outside Git.
- Phase 2: implemented and verified ECtHR-A preparation for JSONL and Parquet, official labels, split-size checks, provenance hashes and duplicate reporting.
- Phase 3: implemented configuration-driven full fine-tuning, validation-only checkpoint selection, separate held-out evaluation, LexGLUE-style metrics and an offline synthetic smoke workflow.
- Repository setup: Python packaging, pinned verified environment snapshot, VS Code tasks, learning documentation, unit tests, publication checks and a lightweight GitHub Actions workflow.

The verified Phase 1–3 repository has an initial local commit on `main`. No GitHub remote has been created and nothing has been pushed.

## Current evidence

- 16 unit tests pass.
- The offline train/save/reload/evaluate smoke workflow passes.
- The real local ECtHR-A splits prepare as 9,000/1,000/1,000 rows.
- Exact cross-split text overlap: zero.
- Within-split repeated text: one additional training row, retained and reported; labels match.
- Publication-boundary scan passes for the full Git candidate set.

See `verification.md` for what these checks establish and their limits.

## Next milestone

Define a small pretrained experiment before any full GPU run:

1. Choose and pin one model revision.
2. Decide the development subset, maximum sequence length and training-step budget.
3. Estimate memory, runtime and cost for the intended hardware.
4. Run the subset experiment using train and validation only.
5. Inspect loss, predictions, truncation and run artifacts.
6. Adjust the configuration before authorizing a full experiment.

LoRA and quantization follow only after the full-fine-tuning baseline path is verified with pretrained weights. This makes later comparisons interpretable.
