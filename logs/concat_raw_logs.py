from datetime import datetime
from pathlib import Path

import polars as pl

interim_dir = Path("logs/data/interim")

# Current date and time is an easy way to give a unique name to the files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = interim_dir / f"concat_{timestamp}.parquet"

lfs = pl.scan_parquet("logs/data/raw/*.parquet")
lf = lfs.unique()

lf.sink_parquet(output_path)
