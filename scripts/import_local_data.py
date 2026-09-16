"""Copy the dissertation's original integer-label Parquet splits into data/raw.

This deliberately selects ecthr_a_{split}.parquet, never derived description or
tokenized variants. The originals are not modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    sources = {s: args.source_dir / f"ecthr_a_{s}.parquet" for s in ("train", "validation", "test")}
    for source in sources.values():
        if not source.is_file():
            parser.error(f"Missing original split: {source.name}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; existing input data will not be overwritten.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"source": "local dissertation ECtHR-A raw split files", "upstream_revision": None, "files": {}}
    for split, source in sources.items():
        dest = args.output_dir / f"{split}.parquet"
        shutil.copyfile(source, dest)
        provenance["files"][dest.name] = {"original_filename": source.name, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    (args.output_dir / "acquisition.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Copied three raw splits to {args.output_dir}. Upstream revision is unknown; file hashes recorded.")


if __name__ == "__main__":
    main()
