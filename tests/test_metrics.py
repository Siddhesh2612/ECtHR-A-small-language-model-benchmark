import unittest
from ecthr_benchmark.metrics import classification_metrics, multi_hot


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions_including_no_violation(self):
        labels = [multi_hot([i]) for i in range(10)] + [multi_hot([])]
        logits = [[1 if v else -1 for v in row] for row in labels]
        result = classification_metrics(logits, labels)
        self.assertEqual(result['micro_f1'], 1)
        self.assertEqual(result['macro_f1'], 1)

    def test_no_violation_is_derived_at_zero_threshold(self):
        result = classification_metrics([[0] * 10], [multi_hot([])])
        self.assertEqual(result['micro_f1'], 1)
        self.assertEqual(result['macro_f1'], 1 / 11)
        self.assertEqual(result['f1_no_violation'], 1)

    def test_wrong_violation_loses_no_violation(self):
        result = classification_metrics([[1] + [-1] * 9], [multi_hot([])])
        self.assertEqual(result['micro_f1'], 0)
        self.assertEqual(result['macro_f1'], 0)

    def test_multi_label_partial_match(self):
        result = classification_metrics([[1, -1] + [-1] * 8], [multi_hot([0, 1])])
        self.assertAlmostEqual(result['micro_f1'], 2 / 3)

    def test_rejects_wrong_shapes_nan_and_invalid_ids(self):
        with self.assertRaises(ValueError):
            classification_metrics([[1] * 11], [[0] * 11])
        with self.assertRaises(ValueError):
            classification_metrics([[float('nan')] * 10], [[0] * 10])
        with self.assertRaises(ValueError):
            multi_hot([10])
        with self.assertRaises(ValueError):
            multi_hot([0, 0])


if __name__ == '__main__':
    unittest.main()
