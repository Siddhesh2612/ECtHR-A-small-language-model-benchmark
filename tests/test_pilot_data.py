"""Pilot selection tests use fictional strings; no court examples are included."""

import json
from pathlib import Path
import tempfile
import unittest

from ecthr_benchmark.data import file_sha256, prepare_dataset
from ecthr_benchmark.pilot_data import create_pilot_subset


class PilotDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = self.root / "raw"
        raw.mkdir()
        self.rows = {
            split: [{"text": f"Fictional {split} case {i}", "labels": [] if i % 3 == 0 else [i % 10]}
                    for i in range(count)]
            for split, count in (("train", 40), ("validation", 16), ("test", 4))
        }
        # Preserve an exact duplicate with different labels and one with matching
        # label membership in a different order.
        self.rows["train"][0] = {"text": "Fictional repeated case", "labels": [3, 0]}
        self.rows["train"][1] = {"text": "Fictional repeated case", "labels": [0, 3]}
        self.rows["train"][2] = {"text": "Fictional repeated case", "labels": []}
        for split, rows in self.rows.items():
            self.write_rows(raw / f"{split}.jsonl", rows)
        self.source = self.root / "prepared"
        self.source_manifest = prepare_dataset(raw, self.source, allow_small=True)

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def manifest_change(self, change):
        change(self.source_manifest)
        (self.source / "manifest.json").write_text(json.dumps(self.source_manifest), encoding="utf-8")

    def test_deterministic_selection_and_provenance(self):
        first = create_pilot_subset(self.source, self.root / "a", 8, 4, 42)
        second = create_pilot_subset(self.source, self.root / "b", 8, 4, 42)
        self.assertEqual(first, second)
        self.assertEqual(first["mode"], "pretrained_pilot_subset")
        self.assertFalse(first["benchmark_comparable"])
        self.assertEqual(first["source_prepared_manifest_sha256"], file_sha256(self.source / "manifest.json"))
        for split in ("train", "validation"):
            info = first["splits"][split]
            indices = info["selected_indices_zero_based"]
            self.assertEqual(indices, sorted(set(indices)))
            actual = [json.loads(line) for line in (self.root / "a" / f"{split}.jsonl").read_text().splitlines()]
            self.assertEqual(actual, [self.rows[split][index] for index in indices])
            self.assertEqual(info["source_split"], split)
            self.assertEqual(info["source_prepared_sha256"], file_sha256(self.source / f"{split}.jsonl"))
            self.assertEqual(info["source_duplicate_text_rows"],
                             self.source_manifest["splits"][split]["duplicate_text_rows"])
            self.assertEqual(info["prepared_sha256"], file_sha256(self.root / "a" / f"{split}.jsonl"))
            self.assertEqual(info["original_input_sha256"],
                             [item["sha256"] for item in self.source_manifest["splits"][split]["sources"]])

    def test_seed_changes_selection(self):
        a = create_pilot_subset(self.source, self.root / "a", 8, 4, 42)
        b = create_pilot_subset(self.source, self.root / "b", 8, 4, 43)
        self.assertNotEqual(a["splits"]["train"]["selected_indices_zero_based"],
                            b["splits"]["train"]["selected_indices_zero_based"])

    def test_duplicates_and_conflicting_labels_preserved(self):
        manifest = create_pilot_subset(self.source, self.root / "a", 40, 16)
        info = manifest["splits"]["train"]
        self.assertEqual(info["duplicate_text_rows"], 2)
        self.assertEqual(info["duplicate_text_rows_with_labels_differing_from_first"], 1)
        output = [json.loads(line) for line in (self.root / "a" / "train.jsonl").read_text().splitlines()]
        self.assertEqual(output, self.rows["train"])

    def test_test_content_never_read_or_written_and_manifest_has_no_case_text(self):
        (self.source / "test.jsonl").unlink()  # Deliberately unavailable to pilot code.
        manifest = create_pilot_subset(self.source, self.root / "a", 8, 4)
        self.assertFalse((self.root / "a" / "test.jsonl").exists())
        self.assertTrue(manifest["splits"]["test"]["excluded_from_pilot"])
        self.assertEqual(manifest["splits"]["test"]["prepared_sha256"],
                         self.source_manifest["splits"]["test"]["prepared_sha256"])
        encoded = json.dumps(manifest)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("Fictional", encoded)

    def test_hash_mismatch_and_invalid_sizes_fail_before_writing(self):
        for size in (0, -1, 41, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                create_pilot_subset(self.source, self.root / "a", size, 4)
            self.assertFalse((self.root / "a").exists())
        with (self.source / "train.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            create_pilot_subset(self.source, self.root / "a", 8, 4)
        self.assertFalse((self.root / "a").exists())

    def test_cross_split_overlap_rechecked_across_full_sources(self):
        # Update the source hash too: a stale/misleading source assertion must
        # not allow overlapping cases even if a small subset omits that row.
        changed = self.rows["validation"][:]
        changed[-1] = self.rows["train"][-1]
        self.write_rows(self.source / "validation.jsonl", changed)
        self.manifest_change(lambda m: m["splits"]["validation"].update(
            prepared_sha256=file_sha256(self.source / "validation.jsonl")))
        with self.assertRaisesRegex(ValueError, "cross-split overlap"):
            create_pilot_subset(self.source, self.root / "a", 1, 1)
        self.assertFalse((self.root / "a").exists())

    def test_requires_original_overlap_check(self):
        self.manifest_change(lambda m: m["checks"].pop("exact_cross_split_overlap"))
        with self.assertRaisesRegex(ValueError, "record zero"):
            create_pilot_subset(self.source, self.root / "a", 8, 4)

    def test_wrong_mapping_and_row_count_rejected(self):
        self.manifest_change(lambda m: m["label_mapping"].update({"0": "1"}))
        with self.assertRaisesRegex(ValueError, "label mapping"):
            create_pilot_subset(self.source, self.root / "a", 8, 4)
        self.manifest_change(lambda m: m["label_mapping"].update({"0": "2"}))
        self.manifest_change(lambda m: m["splits"]["train"].update(rows=999))
        with self.assertRaisesRegex(ValueError, "row count"):
            create_pilot_subset(self.source, self.root / "a", 8, 4)

    def test_existing_directory_cannot_be_overwritten(self):
        create_pilot_subset(self.source, self.root / "a", 8, 4)
        original_hash = file_sha256(self.root / "a" / "manifest.json")
        with self.assertRaisesRegex(ValueError, "already exists"):
            create_pilot_subset(self.source, self.root / "a", 8, 4)
        self.assertEqual(file_sha256(self.root / "a" / "manifest.json"), original_hash)
