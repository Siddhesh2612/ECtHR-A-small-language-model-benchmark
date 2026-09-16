"""Deterministic private train/validation subsets for a bounded pretrained pilot.

This module never reads test examples. Test-overlap verification is inherited
from the original preparation manifest; train/validation overlap is rechecked.
Generated JSONL contains court data and belongs only in Git-ignored directories.
"""

import hashlib
import json
from pathlib import Path

from .data import file_sha256, prepare_row
from .labels import LABEL_NAMES


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _selected_indices(count, size, seed, split):
    # A documented hash ordering is stable without depending on random.sample's
    # implementation. Row identity, not unique text, is the sampling unit.
    ranked = sorted(range(count), key=lambda i: (
        _sha256(f"ecthr-pilot-v1:{seed}:{split}:{i}".encode("ascii")), i))
    return sorted(ranked[:size])


def _duplicate_counts(rows):
    seen = {}
    duplicates = conflicts = 0
    for row in rows:
        digest = _sha256(row["text"].encode("utf-8"))
        labels = set(row["labels"])
        if digest in seen:
            duplicates += 1
            conflicts += labels != seen[digest]
        else:
            seen[digest] = labels
    return duplicates, conflicts


def create_pilot_subset(source_dir, output_dir, train_rows, validation_rows, seed=42):
    """Write train/validation JSONL and return a content-free provenance manifest.

    Samples row indices without replacement from their existing split; selected
    rows retain source order and within-split repeats. No balancing, relabeling,
    or deduplication is performed. The output directory must not already exist.
    All inputs and overlap checks are validated before creating output files.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    sizes = {"train": train_rows, "validation": validation_rows}
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if any(type(size) is not int or size < 1 for size in sizes.values()):
        raise ValueError("pilot split sizes must be positive integers")
    if output_dir.exists():
        raise ValueError("pilot output directory already exists; choose a new directory")
    manifest_bytes = (source_dir / "manifest.json").read_bytes()
    source_manifest = json.loads(manifest_bytes)
    expected_mapping = {str(i): name for i, name in enumerate(LABEL_NAMES)}
    if source_manifest.get("label_mapping") != expected_mapping:
        raise ValueError("prepared label mapping does not match ECtHR-A")
    if source_manifest.get("mode") not in ("full_expected_sizes", "small_subset_or_fixture"):
        raise ValueError("pilot requires an original prepared dataset, not another pilot subset")
    overlap = source_manifest.get("checks", {}).get("exact_cross_split_overlap")
    if type(overlap) is not int or overlap != 0:
        raise ValueError("original preparation must record zero exact cross-split overlap")
    test_reference = source_manifest.get("splits", {}).get("test", {})
    if not _valid_hash(test_reference.get("prepared_sha256")):
        raise ValueError("original preparation must include the reference test hash")

    selected = {}
    manifest = {
        "schema_version": 1,
        "dataset": "LexGLUE ECtHR-A",
        "mode": "pretrained_pilot_subset",
        "pilot": True,
        "benchmark_comparable": False,
        "label_mapping": expected_mapping,
        "source_prepared_manifest_sha256": _sha256(manifest_bytes),
        "source_prepared_mode": source_manifest["mode"],
        "selection": {
            "seed": seed,
            "algorithm": "sha256(ecthr-pilot-v1:{seed}:{split}:{zero_based_index}); select lowest hashes",
            "sampling_unit": "source row index",
            "replacement": False,
            "output_order": "original source row order",
            "stratified": False,
            "deduplicated": False,
        },
        "checks": {
            "exact_train_validation_overlap": 0,
            "source_exact_cross_split_overlap": 0,
            "test_overlap_check": "inherited from source preparation; test examples not read",
            "near_duplicates_checked": False,
        },
        "splits": {
            "test": {
                "prepared_sha256": test_reference["prepared_sha256"],
                "rows": test_reference.get("rows"),
                "excluded_from_pilot": True,
                "reference_only": True,
            }
        },
    }
    train_text_hashes = set()
    for split, size in sizes.items():
        source_info = source_manifest.get("splits", {}).get(split, {})
        original_hashes = [item.get("sha256") for item in source_info.get("sources", [])]
        if not original_hashes or not all(_valid_hash(value) for value in original_hashes):
            raise ValueError(f"{split} original input hashes are missing or invalid")
        content = (source_dir / f"{split}.jsonl").read_bytes()
        source_hash = _sha256(content)
        if source_hash != source_info.get("prepared_sha256"):
            raise ValueError(f"{split} hash does not match prepared manifest")
        rows = []
        for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            try:
                rows.append(prepare_row(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{split} row {number} is invalid") from exc
        if len(rows) != source_info.get("rows"):
            raise ValueError(f"{split} row count does not match prepared manifest")
        if size > len(rows):
            raise ValueError(f"requested {split} size exceeds available source rows")
        text_hashes = {_sha256(row["text"].encode("utf-8")) for row in rows}
        if split == "train":
            train_text_hashes = text_hashes
        elif train_text_hashes & text_hashes:
            raise ValueError("exact cross-split overlap between source train and validation")
        indices = _selected_indices(len(rows), size, seed, split)
        selected[split] = [rows[i] for i in indices]
        duplicate_rows, conflicts = _duplicate_counts(selected[split])
        label_counts = {str(i): 0 for i in range(len(LABEL_NAMES))}
        for row in selected[split]:
            for label in row["labels"]:
                label_counts[str(label)] += 1
        manifest["splits"][split] = {
            "rows": size,
            "source_rows": len(rows),
            "source_split": split,
            "source_prepared_sha256": source_hash,
            "source_duplicate_text_rows": source_info.get("duplicate_text_rows"),
            "source_duplicate_text_rows_with_labels_differing_from_first":
                source_info.get("duplicate_text_rows_with_labels_differing_from_first"),
            "original_input_sha256": original_hashes,
            "selected_indices_zero_based": indices,
            "prepared_file": f"{split}.jsonl",
            "label_counts": label_counts,
            "no_violation_rows": sum(not row["labels"] for row in selected[split]),
            "duplicate_text_rows": duplicate_rows,
            "duplicate_text_rows_with_labels_differing_from_first": conflicts,
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in selected.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split]["prepared_sha256"] = file_sha256(path)
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest
