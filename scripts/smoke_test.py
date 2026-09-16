"""Run the complete offline synthetic train/save/reload/evaluate check."""
import os
from pathlib import Path

# Set these before importing model libraries: the smoke path needs no network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))

from ecthr_benchmark.train import train
from ecthr_benchmark.evaluate import evaluate


if __name__ == "__main__":
    run_dir = train("configs/smoke.json")
    result = evaluate(run_dir)
    assert result["synthetic"] and result["test_rows"] == 4
    assert (run_dir / "model" / "config.json").is_file()
    assert (run_dir / "test_predictions.jsonl").is_file()
    print(f"Offline pipeline verification passed. Synthetic artifacts: {run_dir}")
