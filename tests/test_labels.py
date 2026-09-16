import unittest

from ecthr_benchmark.labels import LABEL_NAMES, multi_hot, validate_labels


class LabelTests(unittest.TestCase):
    def test_official_mapping_and_empty_target(self):
        self.assertEqual(LABEL_NAMES, ("2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1"))
        self.assertEqual(multi_hot([]), [0.0] * 10)
        self.assertEqual(multi_hot([0, 9]), [1.0] + [0.0] * 8 + [1.0])

    def test_invalid_labels_fail_instead_of_coercing(self):
        for labels in ([10], [-1], [True], [0.0], ["0"], [2, 2], None, "2"):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                validate_labels(labels)

    def test_label_order_is_preserved(self):
        labels = [9, 0, 3]
        self.assertEqual(validate_labels(labels), labels)
        self.assertIsNot(validate_labels(labels), labels)
