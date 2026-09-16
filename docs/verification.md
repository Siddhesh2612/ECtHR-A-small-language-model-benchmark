# Verification status

Last verified: 16 September 2026 on macOS arm64 with Python 3.12.14.

## Phase 1: historical archive

Verified:

- 21 distinct notebook code versions were curated from the private audit.
- Notebook outputs, execution counts, attachments and private runtime metadata were removed.
- Hard-coded Hugging Face credential literals and personal absolute paths were removed from publication copies.
- All curated notebooks parse as valid notebook JSON.
- The historical results CSV preserves 21 source-transcribed rows, including blanks and unresolved discrepancies.
- Publication checks found no checked credential patterns, court-data files, model artifacts or private audit files in Git candidate files.

The original dissertation folder was not modified. Historical notebooks were not rerun, and their reported scores are not treated as reproduced results.

## Phase 2: data preparation

The local MSc ECtHR-A integer-label files were copied into ignored `data/raw/` files with source hashes and prepared into ignored `data/processed/` JSONL files.

| Split | Rows | No-violation rows | Repeated text rows beyond first | Repeat rows with different labels |
| --- | ---: | ---: | ---: | ---: |
| Train | 9,000 | 914 | 1 | 0 |
| Validation | 1,000 | 175 | 0 | 0 |
| Test | 1,000 | 153 | 0 | 0 |

The preparation check found no exact text overlap across splits. It does not test near-duplicate, paraphrased or related cases. The one repeated training text was preserved, because removing it would alter the supplied benchmark split; its labels match the first occurrence.

Prepared-data validation confirms:

- every case has non-empty text and a label list;
- label IDs are unique integers from 0 through 9;
- the official ten-label mapping is recorded;
- paragraph order is retained and paragraphs are joined with newlines;
- expected split sizes are enforced in full mode;
- input and output SHA-256 hashes are recorded;
- existing prepared outputs are never overwritten.

The local acquisition record cannot recover the exact upstream dataset revision. File hashes identify the local input bytes but do not establish their upstream version or prove semantic correctness.

## Phase 3: training pipeline

The offline smoke workflow completed all of these stages:

1. Created synthetic train, validation and test cases.
2. Built a tiny randomly initialized GPT-2 sequence classifier locally.
3. Trained for two CPU steps.
4. Selected and saved a checkpoint using validation micro-F1.
5. Reloaded the saved model and tokenizer from disk.
6. Evaluated once on the separate synthetic test split.
7. Wrote resolved configuration, runtime, validation metrics, test metrics and numeric predictions to a unique ignored run directory.

The test metric values have no research meaning. The smoke run verifies pipeline plumbing, model serialization and separation of training from final evaluation.

Evaluation output publication is interruption-safe: predictions are staged first and test metrics are written last as the completion marker. A simulated interrupted write test verifies that the run can be retried.

## Automated checks

Command:

```sh
python -m unittest discover -s tests -v
```

Result: **16 tests passed**. Coverage includes data structure and split checks, official labels, multi-hot encoding, derived no-violation metrics, invalid metric inputs and recoverable evaluation writes.

Command:

```sh
python scripts/smoke_test.py
```

Result: passed end to end in offline mode. The run and model artifacts are ignored by Git.

Command:

```sh
python scripts/check_publication.py
```

Result: passed for the complete current Git candidate set, including 21 notebooks. This targeted scan checks known credential patterns, notebook outputs, personal home paths and excluded binary artifact types. It is not a comprehensive secret scanner or a legal review of publication rights.

`python -m pip check` reported no broken requirements in the verified local environment. The exact environment snapshot is stored in `requirements/verified-macos-py312.txt`.

## Not yet verified

- No pretrained model has been downloaded or fine-tuned with the new pipeline.
- `configs/gpt_neo_125m.json` is a proposed new experiment, not a recovered historical configuration.
- The optional LoRA path has code-level validation but no completed training run.
- CUDA behavior, GPU memory measurement and multi-GPU execution are untested.
- Quantization is outside Phase 3 and is not implemented.
- No controlled comparison with Legal-BERT or other models has been run.
- No remote GitHub repository, release or push has been created.

Before a full run, define the model revision, hardware, precision, training budget and estimated cost. Run a small pretrained subset experiment first, then pin the resolved model revision for the full experiment.
