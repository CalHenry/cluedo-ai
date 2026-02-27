import os
from datetime import datetime, timedelta

import logfire.db_api
import polars as pl
from dotenv import load_dotenv

load_dotenv()
logfire_read_token = os.getenv("LOGFIRE_READ_TOKEN")


conn = logfire.db_api.connect(
    read_token=logfire_read_token, min_timestamp=timedelta(days=7)
)
query = """
SELECT process_pid,
    start_timestamp,
    end_timestamp,
    duration,
    trace_id,
    span_id,
    kind,
    span_name,
    otel_status_code,
    message,
    attributes->>'gen_ai.usage.input_tokens' as input_tokens,
    attributes->>'gen_ai.usage.output_tokens' as output_tokens,
    attributes->>'gen_ai.response.finish_reasons' as finish_reason,
    attributes->>'gen_ai.input.messages' as attribute_messages
FROM records
ORDER BY start_timestamp DESC
"""
df = pl.read_database(query, conn)

datetime_today = datetime.now().strftime("%Y-%m-%d")

df.write_csv(f"data/df_{datetime_today}.csv")

conn.close()
