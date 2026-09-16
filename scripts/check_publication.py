"""Check files Git would include for obvious accidental private material.

This is a targeted repository check, not a guarantee that every possible secret
format or every publication-rights concern has been detected.
"""
from pathlib import Path
import json
import re
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root, check=True, capture_output=True)
    paths = sorted(set(p for p in result.stdout.decode().split("\0") if p))
    patterns = [re.compile(r"hf_[A-Za-z0-9]{20,}"), re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), re.compile(r"AKIA[A-Z0-9]{16}")]
    failures = []
    notebooks = 0
    for name in paths:
        p = root / name
        if not p.is_file():
            continue
        if name.startswith(("project-audit/", "data/raw/", "data/processed/", ".venv/", "runs/")):
            failures.append(f"Private/generated path tracked: {name}")
        if p.suffix in {".parquet", ".safetensors", ".zip", ".mov"}:
            failures.append(f"Unexpected binary artifact: {name}")
        if p.stat().st_size > 10_000_000:
            failures.append(f"Large publication file: {name}")
        text = p.read_text(errors="replace")
        if any(pattern.search(text) for pattern in patterns):
            failures.append(f"Credential pattern found: {name}")
        if p.suffix == ".ipynb":
            notebooks += 1
            nb = json.loads(text)
            if any(c.get("outputs") or c.get("execution_count") is not None for c in nb.get("cells", []) if c.get("cell_type") == "code"):
                failures.append(f"Notebook has execution outputs: {name}")
            if re.search(r"/(?:Users|home)/[A-Za-z0-9_.-]+/", text):
                failures.append(f"Notebook has machine-specific home path: {name}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Publication checks passed: {len(paths)} candidate files; {notebooks} notebooks; no checked credential patterns or private artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
