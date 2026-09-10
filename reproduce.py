"""Single entry point: reproduces the final submission.

Prerequisites:
  1. pip install -r requirements.txt
  2. Download the challenge data into ./data/ (see README, Data section)

The pipeline (src/pipeline.py): training, temporal validation on
January + July 2025, retraining on full 2025, and writing of the
submission files into ./submissions/.

First run: ~20-40 min (feature computation, then cached); ~10-15 min
per subsequent run.
"""
import runpy
from pathlib import Path

if not any(Path("data").glob("training_*.parquet")):
    raise SystemExit("Data missing: see README, Data section.")

runpy.run_path(Path("src") / "pipeline.py", run_name="__main__")
