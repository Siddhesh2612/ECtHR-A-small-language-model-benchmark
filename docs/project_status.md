# Project status

Updated: 20 September 2026.

## Completed

- Phase 1: curated 21 distinct historical notebook versions, preserved source-labelled results, documented discrepancies and kept the private archive outside Git.
- Phase 2: implemented and verified ECtHR-A preparation for JSONL and Parquet, official labels, split-size checks, provenance hashes and duplicate reporting.
- Phase 3: implemented configuration-driven full fine-tuning, validation-only checkpoint selection, separate held-out evaluation, LexGLUE-style metrics and an offline synthetic smoke workflow.
- Repository setup: Python packaging, pinned verified environment snapshot, VS Code tasks, learning documentation, unit tests, publication checks and a lightweight GitHub Actions workflow.
- Repository publication: public GitHub remote, Apache-2.0 licensing for original contributions, citation metadata and contribution guidance.
- Phase 4 pilot infrastructure: deterministic private train/validation subsets, immutable GPT-Neo 125M revision, hard workload ceilings, explicit wall-time budget, recorded status/logs and no pilot access to final test evaluation.
- Phase 4 execution: the bounded pretrained pilot, a 200-step GPU staging run and the full three-epoch GPT-Neo 125M baseline completed successfully on Kaggle.
- Baseline evidence: validation micro-F1 0.5684, test micro-F1 0.5361, 2,240-second training wall time and 3.52 GB peak allocated CUDA memory on a Tesla T4. The private run archive is identified by SHA-256 in `results/new/gpt_neo_125m_full_finetuning.json`.
- Phase 5 preparation: matched LoRA and NF4 QLoRA configurations, validation rules and adapter-aware evaluation support are implemented for GPU execution.

The Phase 1–3 repository has been published on `main` at [Siddhesh2612/ECtHR-A-small-language-model-benchmark](https://github.com/Siddhesh2612/ECtHR-A-small-language-model-benchmark). GitHub publication is a software milestone, not a reproduction of the dissertation results. The repository includes Apache-2.0 licensing for original contributions, citation metadata and contribution instructions; external datasets, model weights and third-party materials retain their own terms.

## Current evidence

- 33 unit tests pass, including bounded-pilot, deterministic-subset and adapter-configuration tests.
- The offline train/save/reload/evaluate smoke workflow passes.
- The pretrained pilot plan command passes without downloading a model, preparing data or training.
- The real local ECtHR-A splits prepare as 9,000/1,000/1,000 rows.
- Exact cross-split text overlap: zero.
- Within-split repeated text: one additional training row, retained and reported; labels match.
- Publication-boundary scan passes for the full Git candidate set.

See `verification.md` for what these checks establish and their limits.

## Next milestone

Execute the matched adapter experiments on the same Kaggle GPU class when
possible:

1. Run a short LoRA staging check, then the full three-epoch LoRA configuration.
2. Run a short QLoRA staging check, then the matched full QLoRA configuration.
3. Select checkpoints using validation micro-F1 and evaluate each frozen method
   once on the held-out test split.
4. Compare F1, trainable parameters, training time, peak CUDA memory, model
   footprint and inference throughput with the full-fine-tuning baseline.
5. Preserve private model artifacts and publish only configurations, aggregate
   metrics, hashes and hardware/runtime evidence.
