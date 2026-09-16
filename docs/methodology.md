# Methodology for new experiments

## Scope

Predict violated ECtHR articles from factual paragraphs using supervised
multi-label classification. The supplied pretrained configuration is a new
configuration for future work, not recovery of a missing dissertation run.

## Data protocol

Preserve ECtHR-A's 9,000/1,000/1,000 train/validation/test split. Validate integer
label IDs and preserve paragraph order. Join paragraphs with newlines. Reject
exact prepared-text overlap across splits. Record within-split duplicates for review;
do not silently remove cases and change the benchmark. Exact overlap detection
does not detect all near-duplicate or related cases.

Prepared data remain model-independent. Model-specific tokenization truncates to
the configured length and pads for batching. The ten official article targets
are multi-hot encoded during training.

## Training protocol

Initialize the random seed before constructing a classification head. Use
multi-label binary cross-entropy and Hugging Face Trainer's optimizer and
scheduler. Train only on training cases. Choose checkpoints using validation
micro-F1. Preserve the resolved configuration, actual input hashes and runtime
metadata with each uniquely identified run.

## Pretrained pilot protocol

Before a full experiment, use a deterministic subset of the existing training
and validation splits. Sampling operates on source row indices without balancing,
relabeling or deduplication; selected rows keep their original order. The pilot
manifest records the parent manifest hash, original input hashes, selected
indices, label counts and duplicate counts. Test examples are not read or copied.

The supported first pilot fixes the model family and requires an immutable
40-character model revision. Hard ceilings limit rows, optimizer steps, context
length, batch sizes and wall time. These controls make a configuration error fail
before it can silently start a full run. The pilot is used to inspect execution,
loss, truncation and artifacts. It does not produce a held-out test result and is
not directly comparable with a full ECtHR-A benchmark experiment.

## Evaluation protocol

Predict a violation for logits strictly greater than zero, equivalent to sigmoid
probability greater than 0.5. Add a derived no-violation category to both predictions
and targets. Calculate micro-F1, macro-F1 and individual label F1 scores.

Test evaluation reloads the saved checkpoint in a separate command. Do not use
test scores to select hyperparameters or checkpoints. Historical scores using
different protocols remain in `results/historical` and must not be pooled with
new results.

## Limits of current verification

An offline synthetic smoke run verifies execution and model reload, not predictive
validity. Pretrained GPU runs, LoRA and quantization require separate verification.
Runtime from one machine is not a cross-model efficiency benchmark. The initial
implementation does not make a claim of superiority over Legal-BERT.

## Source attribution

Evaluation convention follows [LexGLUE](https://github.com/coastalcph/lex-glue/blob/main/experiments/ecthr.py).
The new implementation uses Hugging Face Transformers and PyTorch. See
`THIRD_PARTY_NOTICES.md` and the historical catalogue for the original code's provenance.
