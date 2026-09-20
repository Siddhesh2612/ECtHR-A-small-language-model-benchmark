# New experiment results

This directory contains publication-safe aggregate results produced by the current
benchmark pipeline. These records are separate from the historical notebook scores
under `results/historical/`.

Model weights, checkpoints, per-case predictions and ECtHR-A text remain outside
Git. Each result record identifies its retained private artifact by SHA-256 so the
published metrics can be traced back to the corresponding run bundle.

Staging runs are pipeline checks and must not be added to the benchmark comparison.
