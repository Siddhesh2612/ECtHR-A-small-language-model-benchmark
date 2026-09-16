# Small Language Model Benchmarking on ECtHR-A

MSc dissertation experiments in multi-label legal text classification, with a
new reproducible pipeline for data preparation and model fine-tuning.

**Two bodies of work are kept distinct:** `notebooks/historical/` and
`results/historical/` preserve the dissertation; `src/ecthr_benchmark/` implements
new experiments using a documented evaluation protocol. Historical results have
not been reproduced by this new code.

## Start here

1. Open `ecthr-benchmark.code-workspace` in VS Code.
2. Read [the learning guide](docs/learning_guide.md) while exploring the modules.
3. Run the lightweight tests and offline smoke workflow below.
4. Inspect [the experiment catalogue](docs/experiment_catalogue.md) for the MSc work.

Current progress and the next research milestone are recorded in
[project status](docs/project_status.md).

The workspace settings select `.venv/bin/python` on macOS/Linux. In VS Code,
**Terminal → Run Task** provides tests, data preparation, the offline smoke run
and publication checks. The recommended Python and Jupyter extensions are listed
in `.vscode/extensions.json`; installing them is optional and separate.

## Setup

Use Python 3.12 for the verified environment. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,train]'
```

The core preparation and metrics code needs only Python for JSONL inputs.
The `data` extra adds Parquet support. The `train` extra installs the model stack.
The versions in `pyproject.toml` belong to the new pipeline, not a reconstruction
of the original 2024 environment. The verified macOS/Python 3.12 environment is
recorded in `requirements/verified-macos-py312.txt`. On that platform, install it
first with `python -m pip install -r requirements/verified-macos-py312.txt`, then
install this project. Other platforms, especially CUDA machines, need their own
validated dependency snapshot. Optional notebook tooling can be installed
with `python -m pip install -e '.[notebooks]'`.

## Phase 1: historical archive

There are 21 distinct sanitized notebook code versions and 21 preserved historical
result rows. See [the catalogue](docs/experiment_catalogue.md) and
[historical results](results/historical/reported_results.csv).

The source archive had incorrect article descriptions, model-name discrepancies,
test-set use during checkpoint selection in some experiments and figure issues.
These are documented, not hidden by renaming scores. Historic notebooks are not
the supported runnable entry point and still contain historical dependencies.

## Phase 2: prepare data

Place the raw splits in `data/raw/` following [the data guide](data/README.md).
For the original dissertation folder, this helper copies only raw integer-label
files and records their hashes:

```sh
python scripts/import_local_data.py --source-dir '/path/to/Dissertation Project/ecthr_a'
python -m ecthr_benchmark.data --input-dir data/raw --output-dir data/processed
```

The pipeline validates all three splits before writing. Expected sizes are
9,000 training, 1,000 validation and 1,000 test rows. It preserves numeric label
IDs and paragraph order, rejects malformed input and cross-split text overlap, and
writes counts of within-split duplicate text without dropping cases. It
writes JSONL files plus a provenance/distribution manifest. It refuses to
overwrite existing prepared files. Choose a new output directory for a new
preparation version; do not delete a dataset used by a saved run.

For synthetic or intentionally small fixtures only, pass `--allow-small`.
Raw data, processed cases, checkpoints and the private audit are Git-ignored.

## Phase 3: verify training without downloading a model

```sh
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

The smoke workflow builds a tiny randomly initialized GPT-2 classifier and
synthetic data locally. It runs two CPU training steps, selects a checkpoint on
validation, saves it, reloads it and evaluates on a separate synthetic test split.
It runs with model-download access disabled. **Its metrics are infrastructure
checks, not ECtHR-A research findings.**

Each run gets a unique directory under `runs/smoke/`. To run the stages separately:

```sh
python -m ecthr_benchmark.train --config configs/smoke.json
python -m ecthr_benchmark.evaluate --run-dir runs/smoke/PRINTED_RUN_ID
```

Use the run directory printed by training. Evaluation refuses to overwrite
existing test metrics. To repeat the workflow, create a new run.

## Phase 4: bounded pretrained pilot

The first real-model pilot uses `EleutherAI/gpt-neo-125m` at the immutable model
revision recorded in `configs/pilot_gpt_neo_125m.json`. It deterministically
selects 16 training and eight validation rows, trains for at most two optimizer
steps with 128-token inputs, and defaults to CPU float32. It is a pipeline pilot,
not a benchmark result or a recovery of the missing historical configuration.

Inspect the complete plan without preparing data, downloading weights or training:

```sh
python scripts/run_pretrained_pilot.py --budget-seconds 120
```

After reviewing the printed plan, execute it with a hard whole-process budget:

```sh
python scripts/run_pretrained_pilot.py --budget-seconds 1800 --execute
```

The execution command requires the verified `data/processed` files from Phase 2.
It creates a private deterministic train/validation subset inside the ignored run
directory and records the selected source indices, hashes, configuration, log and
completion status. The runner terminates the training process when the supplied
30–1,800 second budget expires. Pilot test evaluation is deliberately disabled;
promoting a configuration to a full experiment is a separate research decision.

## Future full pretrained experiment

`configs/gpt_neo_125m.json` is a new full-fine-tuning configuration. The original
GPT-Neo 125M training notebook was not recovered, so this configuration is not
claimed as its reproduction. It uses prepared ECtHR-A data and downloads pretrained
weights when first run. Review context length, batch size and compute resources
before launching:

```sh
python -m ecthr_benchmark.train --config configs/gpt_neo_125m.json
python -m ecthr_benchmark.evaluate --run-dir runs/gpt_neo_125m/PRINTED_RUN_ID
```

Training reads only train and validation examples. The preparation manifest
registers the expected test hash. Final evaluation checks the data identity and
reloads the saved model. A model revision of `main` is resolved at download time;
the resolved commit is recorded. Pin that commit in follow-up configurations for
an exact model version.

An optional LoRA code path exists but has not been verified in the initial
full-fine-tuning smoke test. Quantization and multi-model automation are later
extensions, not completed features.

## Repository map

```text
configs/                    Experiment settings
src/ecthr_benchmark/         Preparation, labels, metrics, train and evaluate
scripts/                    Local import, archive curation and verification helpers
tests/                      Data and metric correctness checks
notebooks/historical/       Sanitized MSc notebook archive
results/historical/         Source-labelled dissertation results
docs/                       Methods, learning guide, catalogue and verification
data/                       Data instructions; actual records are ignored
.vscode/                    Interpreter, tasks and extension recommendations
```

`project-audit/` and the earlier private review documents stay local. The archive
curation helper needs that private audit and is not needed to run the new pipeline.
GitHub Actions runs the dependency-light unit tests and publication check on each
push and pull request; it does not download models or run training.

## Research protocol and attribution

The new pipeline trains ten article outputs with binary cross-entropy and derives
no violation during evaluation. See [methodology](docs/methodology.md),
[verification](docs/verification.md), and [third-party notices](THIRD_PARTY_NOTICES.md).
The [LexGLUE paper](https://aclanthology.org/2022.acl-long.297/) and
[benchmark implementation](https://github.com/coastalcph/lex-glue) are the primary
references for the dataset and evaluation convention.

## Repository, citation and license

The project is published at
[Siddhesh2612/ECtHR-A-small-language-model-benchmark](https://github.com/Siddhesh2612/ECtHR-A-small-language-model-benchmark).
This repository contains the MSc research archive and the new preparation/training
implementation; publication does not make historical scores reproduced results.
No new pretrained ECtHR-A benchmark result is claimed by the synthetic smoke run.

Research software metadata is provided in [CITATION.cff](CITATION.cff). Please cite
this repository when using its code and cite LexGLUE separately when using the
benchmark. For reproducibility, record the commit used in your experiment.

Original contributions to this repository are licensed under
[Apache-2.0](LICENSE), subject to the scope in
[third-party notices](THIRD_PARTY_NOTICES.md). The license does not relicense the
ECtHR-A dataset, pretrained model weights, software dependencies or third-party
figures; obtain those materials under their respective terms.

See [contributing](CONTRIBUTING.md) for the development workflow. Before each
commit or push, run `python scripts/check_publication.py` and inspect `git diff`
and `git status`. Token removal does not revoke old credentials; rotate any
still-active tokens from the original archive.
