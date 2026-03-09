import os
import time
from datetime import datetime

import logfire.db_api
import polars as pl
from dotenv import load_dotenv

load_dotenv()
logfire_read_token = os.getenv("LOGFIRE_READ_TOKEN")


conn = logfire.db_api.connect(read_token=logfire_read_token)

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

# Paginate through all results
pages = []
limit = 9000
offset = 0
fetch_count = 0

while True:
    paginated_query = f"{query} LIMIT {limit} OFFSET {offset}"
    page = pl.read_database(paginated_query, conn)

    if page.is_empty():
        break

    pages.append(page)
    fetch_count += 1
    print(f"Fetched {offset + len(page)} rows...")

    if len(page) < limit:
        break

    offset += limit

    # Sleep every 4 fetches to respect the rate limit
    if fetch_count % 4 == 0:
        print("Rate limit pause: 70 seconds...")
        time.sleep(70)

df = pl.concat(pages)
print(f"Total rows fetched: {len(df)}")

datetime_today = datetime.now().strftime("%Y-%m-%d")
df.write_parquet(f"logs/data/raw/log_{datetime_today}.parquet")
conn.close()

# I don't expect to pull millions of rows with this script therefore we don't need Lazy API since the DB pull has to be eager anyway
