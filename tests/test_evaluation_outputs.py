import json
from pathlib import Path
import tempfile
import unittest

from ecthr_benchmark.evaluate import publish_evaluation


class EvaluationOutputTests(unittest.TestCase):
    def test_failed_prediction_stream_does_not_mark_complete_and_can_retry(self):
        def failing_rows():
            yield {'row_index': 0}
            raise OSError('simulated interrupted prediction write')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(OSError):
                publish_evaluation(root, {'synthetic': True}, failing_rows())
            self.assertFalse((root / 'test_metrics.json').exists())
            self.assertEqual(list(root.iterdir()), [])
            publish_evaluation(root, {'synthetic': True}, [{'row_index': 0}])
            self.assertTrue(json.loads((root / 'test_metrics.json').read_text())['synthetic'])
            self.assertEqual(len((root / 'test_predictions.jsonl').read_text().splitlines()), 1)
            with self.assertRaises(FileExistsError):
                publish_evaluation(root, {}, [])


if __name__ == '__main__':
    unittest.main()
