import json
from pathlib import Path
import tempfile
import unittest

from ecthr_benchmark.data import file_sha256, prepare_dataset, prepare_row


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name) / "raw"
        self.raw.mkdir()
        self.output = Path(self.temp.name) / "prepared"
        self.write("train", [{"text": ["First fact.", "Second fact."], "labels": [9, 0]},
                             {"text": "No violation training case.", "labels": []}])
        self.write("validation", [{"text": "Validation case.", "labels": [3]}])
        self.write("test", [{"text": "Test case.", "labels": []}])

    def write(self, split, rows):
        (self.raw / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_preparation_preserves_text_labels_and_manifest(self):
        manifest = prepare_dataset(self.raw, self.output, allow_small=True)
        rows = [json.loads(line) for line in (self.output / "train.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0], {"text": "First fact.\nSecond fact.", "labels": [9, 0]})
        train = manifest["splits"]["train"]
        self.assertEqual(train["rows"], 2)
        self.assertEqual(train["label_counts"]["9"], 1)
        self.assertEqual(train["no_violation_rows"], 1)
        self.assertEqual(train["prepared_sha256"], file_sha256(self.output / "train.jsonl"))
        self.assertEqual(manifest["mode"], "small_subset_or_fixture")

    def test_full_mode_rejects_small_data_without_writing(self):
        with self.assertRaisesRegex(ValueError, "expected 9000"):
            prepare_dataset(self.raw, self.output)
        self.assertFalse(self.output.exists())

    def test_cross_split_overlap_rejected_before_writing(self):
        self.write("test", [{"text": "Validation case.", "labels": []}])
        with self.assertRaisesRegex(ValueError, "cross-split overlap"):
            prepare_dataset(self.raw, self.output, allow_small=True)
        self.assertFalse(self.output.exists())

    def test_duplicates_within_split_preserved_and_conflicts_counted(self):
        rows = [{"text": "same", "labels": [0, 3]},
                {"text": "same", "labels": [3, 0]},
                {"text": "same", "labels": []}]
        self.write("train", rows)
        manifest = prepare_dataset(self.raw, self.output, allow_small=True)
        train = manifest["splits"]["train"]
        self.assertEqual(train["rows"], 3)
        self.assertEqual(train["duplicate_text_rows"], 2)
        self.assertEqual(train["duplicate_text_rows_with_labels_differing_from_first"], 1)
        self.assertEqual(manifest["checks"]["exact_duplicate_texts"], 2)
        prepared = [json.loads(line) for line in (self.output / "train.jsonl").read_text().splitlines()]
        self.assertEqual(prepared, rows)

    def test_existing_outputs_not_overwritten(self):
        prepare_dataset(self.raw, self.output, allow_small=True)
        before = file_sha256(self.output / "train.jsonl")
        with self.assertRaisesRegex(ValueError, "already contains"):
            prepare_dataset(self.raw, self.output, allow_small=True)
        self.assertEqual(before, file_sha256(self.output / "train.jsonl"))

    def test_bad_rows_fail(self):
        for row in ({}, {"text": " ", "labels": []}, {"text": [], "labels": []},
                    {"text": ["fact", 1], "labels": []}, {"text": "fact", "labels": [10]}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                prepare_row(row)

    def test_invalid_json_and_missing_split_fail(self):
        (self.raw / "test.jsonl").write_text("{bad}\n")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            prepare_dataset(self.raw, self.output, allow_small=True)
        (self.raw / "test.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "missing test"):
            prepare_dataset(self.raw, self.output, allow_small=True)
