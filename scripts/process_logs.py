import argparse
from pathlib import Path

import polars as pl

"""
This script processes log data for the investigations of 'ai_cluedo'.
Logs are pulled from Logfire in the script 'get_logfire_logs.py' and placed in the data/ folder.
- 1 investigation is 1 run of main.py.
- 1 run is n turns to solve the case. A turn is defined by 'main.py::run_investigation'. Max number of turns is 15.
- 1 turn is the supervisor agent deciding to either gather data, process data or submit a solution. This creates a back and forth between the supervisor, the researcher agent and the processor agent. The number of steps in a turn is not set.

The input data has 3 levels:
   - process_pid level (1 investigation)
   - trace_id level (1 turn)
   - span_id level (each step during 1 turn)

Input data is at the 'span_id' level and we want to aggregate it to the 'process_id' level.
Final dataset will have a single row per process_id (investigation)
"""

# 1. Load data from cli argument
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input CSV")
args = parser.parse_args()

input_path = Path(args.input)
df = pl.scan_csv('data/raw/*.csv')


# —— 2. concat into a single lf

lf =
# ── Intermediate lazy frames ───────────────────────────────────────────────────

# Last span per turn (trace_id): used for token accounting.
last_span_per_turn = (
    df.sort("end_timestamp", descending=False)
    .group_by(["process_pid", "trace_id"])
    .last()
)

# All tool-call spans, with the tool name extracted.
tool_spans = df.filter(pl.col("span_name") == "running tool").with_columns(
    pl.col("message").str.split(": ").list.last().alias("tool_name")
)

# ── Expressions evaluated inside group_by("process_pid").agg() ────────────────

# total_turns: number of distinct trace_ids that contain a supervisor chat span
total_turns_expr = (
    pl.when(pl.col("span_name") == "chat ministral-3-3b-instruct-2512")
    .then(pl.col("trace_id"))
    .otherwise(None)
    .drop_nulls()
    .n_unique()
    .alias("total_turns")
)

# total_tool_calls
total_tool_calls_expr = (
    (pl.col("span_name") == "running tool").sum().alias("total_tool_calls")
)

# parallel_tool_calls: total tool calls that happen in turns where >1 tool ran
# (computed per-turn then summed – we use a sub-aggregation trick via struct)
# We calculate this from tool_spans separately (see below).

# tool_error_rate (fraction of non-error tool spans, as a percentage)
tool_error_rate_expr = (
    (
        (
            (
                (pl.col("span_name") == "running tool")
                & (pl.col("otel_status_code") == "ERROR")
            ).sum()
            / (pl.col("span_name") == "running tool").sum()
        )
        * 100
    )
    .round(2)
    .alias("tool_error_rate")
)

# Regex pattern for a solved case
SOLVED_PATTERN = r'"case_solved":\s*(true)'

# run_success flag
run_success_expr = (
    pl.col("attribute_messages")
    .str.extract(SOLVED_PATTERN)
    .is_not_null()
    .any()
    .alias("final_answer_correct")
)

# ── Main aggregation on the full span-level frame ─────────────────────────────

base_agg = df.group_by("process_pid").agg(
    total_turns_expr,
    total_tool_calls_expr,
    tool_error_rate_expr,
    run_success_expr,
)

# ── Token aggregation (from last_span_per_turn) ───────────────────────────────

token_agg = last_span_per_turn.group_by("process_pid").agg(
    pl.col("input_tokens").sum().alias("total_input_tokens"),
    pl.col("output_tokens").sum().alias("total_output_tokens"),
    (pl.col("input_tokens").sum() + pl.col("output_tokens").sum()).alias(
        "total_tokens"
    ),
)

# ── Tool-level aggregations (from tool_spans) ─────────────────────────────────

# Per-turn tool counts (needed for parallel / avg metrics)
tool_calls_per_turn = tool_spans.group_by(["process_pid", "trace_id"]).agg(
    pl.len().alias("tools_in_turn")
)

parallel_and_avg_agg = tool_calls_per_turn.group_by("process_pid").agg(
    # parallel_tool_calls: sum of tool calls in turns that had >1 tool call
    pl.when(pl.col("tools_in_turn") > 1)
    .then(pl.col("tools_in_turn"))
    .otherwise(0)
    .sum()
    .alias("parallel_tool_calls_per_run"),
    # avg_tools_per_turn across ALL turns (including single-tool turns)
    pl.col("tools_in_turn").mean().round(2).alias("avg_tools_per_turn"),
)

"""
# unique tool count and max consecutive same tool
tool_order_agg = (
    tool_spans.sort("start_timestamp")
    .group_by("process_pid")
    .agg(
        pl.col("tool_name").n_unique().alias("unique_tool_count"),
        # max_consecutive_same_tool: longest run of the same tool back-to-back.
        # We detect a "new group" whenever the tool name changes from the previous
        # row (within the process), assign a group id, then find the largest group.
        pl.col("tool_name")
        .map_elements(
            lambda s: (
                # s is a Polars Series; build group ids and return max group size
                s.to_frame("tool")
                .with_columns(
                    (pl.col("tool") != pl.col("tool").shift(1))
                    .fill_null(True)
                    .cum_sum()
                    .alias("group")
                )
                .group_by("group")
                .agg(pl.len().alias("n"))["n"]
                .max()
            ),
            return_dtype=pl.Int32,
        )
        .alias("max_consecutive_same_tool"),
    )
)
"""
# Details on the expression below:
# Goal: find the maximum number of consecutive tool call (same tool called multiple times without other tools called in between, ex: verify_alibi, verify_alibi)
# Read it is 3 parts:
# - Part1 inside 'with_columns()':
# Compute group ids for each consecutive tool call within process_pid. Each consecutive run of the same tool within a pid now has a unique group id. (2 consecutive tool -> group1, next tool -> group2)
# - Part2 '.group_by("process_pid", "group").agg(...)':
#  count how many time a group from part1 is present for each 'process_pid'. By 'process_pid', we have how many consecutive tool call for each unique tool_call
# - Part3 '.group_by("process_pid").agg(...)':
# By 'process_pid', get the bigger 'consecutive_run', which is for this 'process_pid', the most consecutive tool called.
tool_order_agg = (
    tool_spans.sort("start_timestamp")
    .with_row_index()
    .with_columns(
        (
            (
                pl.col("tool_name") != pl.col("tool_name").shift(1).over("process_pid")
            )  # detect run boundaries. over() is like group_by().agg() -> prevent shift() to be computed gobally, making sure we compute whithin each 'process_pid'
            | (
                pl.col("process_pid") != pl.col("process_pid").shift(1)
            )  # cumsum() is a global function, this condition force the groups to be clearly seperated
        )
        .cum_sum()
        .alias("group")
    )
    .group_by("process_pid", "group")
    .agg(pl.col("tool_name").first(), pl.len().alias("consecutive_count"))
    .group_by("process_pid")
    .agg(pl.col("consecutive_count").max())
)

# null tool responses
null_tool_responses_agg = tool_spans.group_by("process_pid").agg(
    pl.col("message").is_null().sum().alias("null_tool_responses")
)

# ── Join to final lazyframe ───────────────────────────────────────────────────

null_tool_responses_agg.explain()


run_logs_agg = (
    base_agg.join(token_agg, on="process_pid", how="left")
    .join(parallel_and_avg_agg, on="process_pid", how="left")
    .join(tool_order_agg, on="process_pid", how="left")
    .join(null_tool_responses_agg, on="process_pid", how="left")
    .collect()
)

# ── Save to disk --------------------------------------------------------------
output_path = Path("/data") / 'processed'
run_logs_agg.sink_csv(output_path)
