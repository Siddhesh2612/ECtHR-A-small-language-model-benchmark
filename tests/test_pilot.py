import json
from pathlib import Path
import tempfile
import unittest

from ecthr_benchmark.evaluate import evaluate
from ecthr_benchmark.pilot import immutable_revision, validate_pilot


class PilotBudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("configs/pilot_gpt_neo_125m.json").read_text())
        self.config["compute_budget_seconds"] = 120

    def test_bounded_config_is_pinned(self):
        validate_pilot(self.config)
        self.assertTrue(immutable_revision(self.config["model_revision"]))
        self.assertFalse(immutable_revision("main"))

    def test_budget_required_and_bounded(self):
        for value in (None, 0, 1801, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_pilot({**self.config, "compute_budget_seconds": value})

    def test_full_run_expansion_rejected(self):
        for key, value in [
            ("max_steps", -1),
            ("max_steps", 21),
            ("train_rows", 10000),
            ("max_length", 2048),
            ("model_name", "EleutherAI/gpt-neo-1.3B"),
        ]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_pilot({**self.config, key: value})

    def test_pilot_test_evaluation_rejected_before_model_or_data_access(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "resolved_config.json").write_text(json.dumps(self.config))
            with self.assertRaisesRegex(ValueError, "disabled for pilot"):
                evaluate(directory)


if __name__ == "__main__":
    unittest.main()
