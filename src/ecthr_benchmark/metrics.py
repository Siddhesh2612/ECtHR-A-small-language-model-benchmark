"""LexGLUE-style F1: ten article decisions plus derived no-violation.

Protocol reference: https://github.com/coastalcph/lex-glue/blob/main/experiments/ecthr.py
This implementation uses only the Python standard library.
"""
import math

from .labels import LABEL_NAMES as ARTICLE_NAMES, NUM_LABELS, multi_hot


def classification_metrics(logits, targets):
    """Targets are ten-element binary vectors; strictly positive logits predict 1."""
    logits, targets = list(logits), list(targets)
    if not logits or len(logits) != len(targets):
        raise ValueError('Predictions and targets must have the same nonzero row count.')
    counts = [[0, 0, 0] for _ in range(NUM_LABELS + 1)]
    for scores, truth in zip(logits, targets):
        if len(scores) != NUM_LABELS or len(truth) != NUM_LABELS:
            raise ValueError('Each row must contain exactly ten labels.')
        if any(not math.isfinite(float(score)) for score in scores):
            raise ValueError('Logits must be finite.')
        if any(value not in (0, 1) for value in truth):
            raise ValueError('Targets must be binary.')
        predicted = [int(score > 0) for score in scores]
        actual = [int(value) for value in truth]
        predicted.append(int(not any(predicted)))
        actual.append(int(not any(actual)))
        for i, (p, t) in enumerate(zip(predicted, actual)):
            counts[i][0] += p * t
            counts[i][1] += p * (1 - t)
            counts[i][2] += (1 - p) * t
    def f1(tp, fp, fn):
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0
    per_label = [f1(*values) for values in counts]
    totals = [sum(values[i] for values in counts) for i in range(3)]
    return {'micro_f1': f1(*totals), 'macro_f1': sum(per_label) / len(per_label),
            **{f'f1_article_{name}': value for name, value in zip(ARTICLE_NAMES, per_label)},
            'f1_no_violation': per_label[-1]}


def trainer_metrics(prediction):
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return classification_metrics(logits, prediction.label_ids)
