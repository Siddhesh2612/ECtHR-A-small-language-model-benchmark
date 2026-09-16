"""Prepare local raw LexGLUE ECtHR-A JSONL or Parquet with provenance checks.

No text is downloaded or tokenized here. Each model uses its own tokenizer later.
"""

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .labels import LABEL_NAMES, validate_labels

SPLITS = ("train", "validation", "test")
EXPECTED_SIZES = {"train": 9000, "validation": 1000, "test": 1000}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_row(row):
    """Join ordered fact paragraphs using a newline; preserve numeric label IDs."""
    if not isinstance(row, dict) or "text" not in row or "labels" not in row:
        raise ValueError("each example must have text and labels fields")
    text = row["text"]
    if isinstance(text, list):
        if not text or any(not isinstance(part, str) for part in text):
            raise ValueError("text paragraphs must be a nonempty list of strings")
        text = "\n".join(text)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string or list of paragraphs")
    return {"text": text, "labels": validate_labels(row["labels"])}


def read_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{line_number}: invalid JSON") from exc
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet input requires the optional pyarrow package") from exc
        parquet = pq.ParquetFile(path)
        if not {"text", "labels"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"{path.name}: missing text or labels column")
        for batch in parquet.iter_batches(columns=["text", "labels"]):
            yield from batch.to_pylist()
    else:
        raise ValueError("input must be JSONL or Parquet")


def find_split_files(input_dir, split):
    direct = [input_dir / f"{split}{suffix}" for suffix in (".jsonl", ".parquet")]
    direct = [path for path in direct if path.is_file()]
    shards = sorted(input_dir.glob(f"{split}-*-of-*.parquet"))
    if len(direct) > 1 or (direct and shards):
        raise ValueError(f"ambiguous input files for {split}; retain one format")
    files = direct or shards
    if not files:
        raise ValueError(f"missing {split}.jsonl, {split}.parquet or Parquet shards")
    return files


def length_summary(values):
    ordered = sorted(values)
    return {
        "min": ordered[0], "max": ordered[-1],
        "mean": statistics.mean(ordered), "median": statistics.median(ordered),
    }


def prepare_dataset(input_dir, output_dir, *, allow_small=False, source_note=None):
    """Validate all splits before writing; refuse existing generated files.

    Full mode enforces official split sizes. --allow-small is for explicitly
    non-benchmark fixtures/subsets. Within-split duplicate texts are counted and
    preserved; cross-split overlap fails. Cases are never silently removed.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    targets = [output_dir / f"{split}.jsonl" for split in SPLITS]
    targets.append(output_dir / "manifest.json")
    if any(path.exists() for path in targets):
        raise ValueError("output already contains prepared files; choose a new directory")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "LexGLUE ECtHR-A",
        "source_note": source_note,
        "source_revision_status": "not independently verified; source file hashes recorded",
        "mode": "small_subset_or_fixture" if allow_small else "full_expected_sizes",
        "label_mapping": {str(i): name for i, name in enumerate(LABEL_NAMES)},
        "preparation": {"paragraph_separator": "\n", "label_ids_preserved": True,
                        "text_normalization": "none", "tokenized": False},
        "checks": {"exact_duplicate_texts": 0, "exact_cross_split_overlap": 0,
                   "near_duplicates_checked": False},
        "splits": {},
    }
    prepared = {}
    seen = {}
    for split in SPLITS:
        files = find_split_files(input_dir, split)
        rows = []
        duplicate_rows = 0
        conflicting_duplicate_rows = 0
        frequencies = {str(i): 0 for i in range(len(LABEL_NAMES))}
        for path in files:
            for index, raw in enumerate(read_rows(path), 1):
                try:
                    row = prepare_row(raw)
                except ValueError as exc:
                    raise ValueError(f"{path.name}:row {index}: {exc}") from exc
                digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                if digest in seen:
                    previous = seen[digest]
                    if previous[0] != split:
                        raise ValueError(f"exact cross-split overlap: {split}/{path.name}:{index} "
                                         f"and {previous[:3]}")
                    duplicate_rows += 1
                    if set(row["labels"]) != previous[3]:
                        conflicting_duplicate_rows += 1
                else:
                    seen[digest] = (split, path.name, index, set(row["labels"]))
                rows.append(row)
                for label in row["labels"]:
                    frequencies[str(label)] += 1
        if not rows:
            raise ValueError(f"{split} is empty")
        if not allow_small and len(rows) != EXPECTED_SIZES[split]:
            raise ValueError(f"{split}: expected {EXPECTED_SIZES[split]} rows, found {len(rows)}; "
                             "use --allow-small only for a subset or synthetic fixture")
        prepared[split] = rows
        manifest["checks"]["exact_duplicate_texts"] += duplicate_rows
        manifest["splits"][split] = {
            "rows": len(rows),
            "duplicate_text_rows": duplicate_rows,
            "duplicate_text_rows_with_labels_differing_from_first": conflicting_duplicate_rows,
            "sources": [{"file": path.name, "sha256": file_sha256(path)} for path in files],
            "label_counts": frequencies,
            "no_violation_rows": sum(not row["labels"] for row in rows),
            "character_lengths": length_summary([len(row["text"]) for row in rows]),
            "whitespace_word_lengths": length_summary([len(row["text"].split()) for row in rows]),
            "newline_segment_counts": length_summary([row["text"].count("\n") + 1 for row in rows]),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in prepared.items():
        target = output_dir / f"{split}.jsonl"
        with target.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split]["prepared_file"] = target.name
        manifest["splits"][split]["prepared_sha256"] = file_sha256(target)
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-small", action="store_true", help="Explicitly allow non-benchmark split sizes")
    parser.add_argument("--source-note", help="Dataset origin/revision note supplied by the researcher")
    args = parser.parse_args()
    try:
        manifest = prepare_dataset(args.input_dir, args.output_dir,
                                   allow_small=args.allow_small, source_note=args.source_note)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(json.dumps({"mode": manifest["mode"],
                      "rows": {split: details["rows"] for split, details in manifest["splits"].items()},
                      "manifest": str(args.output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
