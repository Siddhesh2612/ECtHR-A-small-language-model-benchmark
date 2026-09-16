"""ECtHR-A label IDs from the LexGLUE dataset metadata.

An empty label list means no violation; it is not an eleventh training label.
"""

LABEL_NAMES = ("2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1")
NUM_LABELS = len(LABEL_NAMES)


def validate_labels(labels):
    """Return a copy without silently coercing, sorting, or dropping labels."""
    if not isinstance(labels, list):
        raise ValueError("labels must be a list of integer IDs")
    if any(type(label) is not int or not 0 <= label < NUM_LABELS for label in labels):
        raise ValueError("labels must contain integer IDs 0..9 (not strings or booleans)")
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate label IDs in one example")
    return labels.copy()


def multi_hot(labels):
    """Convert IDs to ten binary targets, including all zeros for no violation."""
    targets = [0.0] * NUM_LABELS
    for label in validate_labels(labels):
        targets[label] = 1.0
    return targets
