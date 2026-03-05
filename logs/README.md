# LOGS Readme

This folder is all about the logs fo the agentic workflow.

### Folder structure

```sh
├── logs                     # <- cluedo-ai/logs
│   ├── get_logfire_logs.py  # <- step1
│   ├── concat_raw_logs.py   # <- step2
│   ├── process_logs.py      # <- step3
│   ├── data
│   │   ├── archive          # <- raw files move to archive in step2
│   │   ├── interim          # <- concat files arrives here from step2
│   │   ├── processed
│   │   └── raw              # <- data arrives here
│   ├── notebooks            # <- scratchpad notebook to explore logs data and make process script
│   └── README.md
```

### Data life cycle

1. Logs are pulled from Logfire with get_logfire_logs.py and go to /data/raw
2. All parquet files in data/raw are concatenanted. Output goes to data/interim. Raw files are moved to archive
3. Process concat log files that have not been processed yet. Output goes to data/processed

### Important notes on the scripts

Data is processed with Polars and it's lazy API.


get_logfire_logs.py:
- Logfire has a limit of 10000 rows per request and a limit at the number of rows pulled in the same minute.
