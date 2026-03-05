import shutil
from datetime import datetime
from pathlib import Path

import polars as pl

raw_dir = Path("logs/data/raw")
interim_dir = Path("logs/data/interim")
archive_dir = Path("logs/data/archive")
# Current date and time is an easy way to give a unique name to the files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = interim_dir / f"concat_{timestamp}.parquet"

raw_files = list(raw_dir.glob("*.parquet"))

if not raw_files:
    print("No parquet files in raw dir")
else:
    lfs = pl.scan_parquet("logs/data/raw/*.parquet")
    lf = lfs.unique()

    lf.sink_parquet(output_path)

    # Move raw files to archive
    for file in raw_files:
        dest = archive_dir / f"{file.stem}_{timestamp}{file.suffix}"
        shutil.move(str(file), dest)
