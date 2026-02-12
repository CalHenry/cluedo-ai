import os

import logfire.db_api
import polars as pl
from dotenv import load_dotenv

load_dotenv()
logfire_read_token = os.getenv("LOGFIRE_READ_TOKEN")


conn = logfire.db_api.connect(read_token=logfire_read_token)
query = """
SELECT *
FROM records
WHERE otel_scope_name = 'pydantic-ai'
"""
df = pl.read_database(query, conn)
df.write_csv()
print(df)
conn.close()
