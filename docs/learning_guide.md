# Learning guide: Phases 1–3

## 1. Why keep an archive and a package?

A notebook records exploration: trying a learning rate, inspecting outputs,
changing an idea. A package stores the reusable steps needed by many experiments.
Your archived notebooks retain the MSc work. `src/ecthr_benchmark/` is the new
implementation. New code does not retroactively validate an old reported score.

**Try it:** open one archived Phi-2 notebook and find `eval_dataset`. Compare it
with the new training module. The new module uses validation for model selection.

## 2. What is the classification target?

Each case may have several violated articles. A multi-hot target has ten positions,
one per article. For IDs `[0, 2]`, the target is `[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]`.
The official ID order maps to Articles 2, 3, 5, 6, 8, 9, 10, 11, 14 and Protocol 1
Article 1. IDs are vocabulary positions, not article numbers.

An empty label list produces ten zeros. At evaluation, a no-violation category is
derived when none of the article predictions is positive. This makes “no
violation” mutually exclusive with predicted violations. The historical explicit
eleventh output could predict both simultaneously, so that is a different method.

**Try it:** inspect `labels.py` and its tests. Work out which actual articles `[0, 2]`
represent before running the example above.

## 3. Why validate data before training?

A training loop can run successfully with incorrect labels or overlapping splits.
Preparation checks structure and exact text overlap before writing outputs.
The manifest records file hashes: a hash changes when file contents change, so it
identifies the exact input bytes. It does not prove the source data is correct.

We preserve paragraph order and join paragraphs with newlines. We do not remove
stop words: they may carry useful meaning, and pretrained tokenizers expect text.

**Try it:** run a data test that inserts the same case into train and validation.
Notice that a validation failure is preferable to an apparently successful score.

## 4. Preparation versus tokenization

Preparation produces text plus label IDs, reusable across models. Tokenization
turns text into each model's own vocabulary IDs. An ID of 100 means whatever that
particular tokenizer assigns to position 100; it is not universal.

`max_length` limits model input. Truncation discards later tokens; padding fills
short inputs so they can be batched. The attention mask marks real tokens versus
padding. Longer context can retain more evidence but uses more compute and memory.

## 5. What does the classification head learn?

The transformer produces representations. The classification head converts a
case representation into ten raw scores, called logits. A sigmoid maps each score
independently to a value between zero and one. With a fixed 0.5 threshold, a logit
greater than zero predicts an article violation.

Binary cross-entropy compares each article prediction with its binary target.
We use the model's multi-label loss with floating-point targets. A single-label
softmax loss would incorrectly force every case into one category.

## 6. Configuration, seeds and checkpoints

A configuration records what varies between experiments: model, learning rate,
context length, batch size and training duration. Shared code describes how those
settings are executed. A random seed is set before model construction because a
new classification head starts with random weights. A seed improves repeatability;
it does not guarantee identical results across hardware and software versions.

A checkpoint stores model state. The validation micro-F1 selects the checkpoint
to retain. Test evaluation is a separate command after training. Repeatedly using
test results to decide the next configuration would still leak test information.

## 7. Micro-F1 and macro-F1

F1 combines precision and recall. Micro-F1 aggregates true/false positives and
false negatives across the decisions. Macro-F1 computes F1 for each label and
averages them equally, making rare labels count as much as common ones. The
benchmark convention here includes the derived no-violation category; labels
with a zero F1 denominator contribute zero, as in the source implementation.

**Try it:** read the no-violation metric test. A perfect prediction on a tiny
sample need not have macro-F1 of 1 if some labels never occur in that sample.

## 8. What a smoke test proves

The smoke configuration creates a tiny random GPT-2 architecture and synthetic
text locally. It trains for a few steps, selects on synthetic validation data,
saves a model and allows separate reload/evaluation. No pretrained weights or
court data are downloaded for this test.

It verifies the plumbing. Its scores say nothing about ECtHR-A accuracy, pretrained
model quality, or the dissertation findings. A full pretrained run is the next
research step and requires an appropriate compute environment.

## 9. Why start with a bounded pretrained pilot?

The smoke run tests code using a randomly initialized synthetic model. A
pretrained pilot crosses the next boundary: it downloads a real checkpoint,
tokenizes real prepared cases and performs a few optimization steps. Keeping the
subset and step count small makes failures cheap to diagnose before a full run.

The pilot subset is deterministic: the same seed and parent files select the same
source row indices. Deterministic selection makes runs traceable, although it does
not guarantee identical floating-point results on different hardware.

**Try it:** run the pilot command without `--execute`. Identify the data, model,
step and wall-time limits in the printed plan. No model or court data is touched.

## 10. Revision pins and compute budgets

A model name such as `main` can point to different files later. A full commit hash
is an immutable revision: it identifies one exact state of the model repository.
The pilot additionally requires a wall-time budget. Row, context and step limits
bound the planned workload; the timer bounds the process when downloads or
training take unexpectedly long.

The runner is the supported pilot entry point because it enforces that timer.
Calling the lower-level training module still checks the configuration ceilings,
but it does not own the outer process lifetime.

## Reading

- [LexGLUE experiment code](https://github.com/coastalcph/lex-glue/blob/main/experiments/ecthr.py)
- [LexGLUE dataset card](https://huggingface.co/datasets/coastalcph/lex_glue/blob/main/README.md)
- [Transformers GPT-2 classification](https://huggingface.co/docs/transformers/v4.56.2/model_doc/gpt2)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
