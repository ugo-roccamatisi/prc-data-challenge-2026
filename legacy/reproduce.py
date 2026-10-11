"""Single entry point for the first-campaign pipeline.

Prerequisites:
  1. pip install -r requirements.txt
  2. Download the challenge data into ./data/ (see README, Data section)

This runs legacy/pipeline.py: training, temporal validation on January + July
2025, retraining on full 2025, and writing submission files into
./submissions/. For the best second-campaign submission, use the production
commands documented in the README.

First run: ~20-40 min (feature computation, then cached); ~10-15 min
per subsequent run.
"""
import runpy
from pathlib import Path

if not any(Path("data").glob("training_*.parquet")):
    raise SystemExit("Data missing: see README, Data section.")

runpy.run_path(Path("legacy") / "pipeline.py", run_name="__main__")
