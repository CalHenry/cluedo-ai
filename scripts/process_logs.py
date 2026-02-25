import polars as pl

"""
This script processes log data for the investigations of 'ai_cluedo'.
Logs are pulled from Logfire in the script 'get_logfire_logs.py' and placed in the data/ folder.
- 1 investigation is 1 run of main.py.
- 1 run is n turns to solve the case. A turn is defined in 'main.py::run_investigation'. Max number of turns is 15.
- 1 turn is the supervisor agent deciding to either gather data, process data or submit a solution. This creates a back and forth between the supervisor, the researcher agent and the processor agent. The number of steps in a run is not set.

The input data has 3 levels:
   - process_pid level (1 investigation)
   - trace_id level (1 turn)
   - span_id level (each step during 1 turn)

Input data is at the 'span_id' level and we want to aggregate it to 'process_id' level.
Final dataset will have a single row per process_id (investigation)
"""

# 1. Load data
df = pl.read_csv("data/*.csv")

# 2. Define informations to extract
"""
- total_turns,
Tool usage:
  - total_tool_calls,
  - parallel_tool_calls
  - repeated_tool_calls
  - unique_tool_count,
  - max_consecutive_same_tool,
  - avg_tools_per_turn,
  - tool_success_rate,
- null_tool_responses,
- investigated_verified_suspect_flag (when the tool responded 'alibi_verified: true' but the model keeps searching)
- total_tokens,
- final_answer_correct
"""

total_turns = (
    df.group_by("span_name")
    .agg(pl.col("trace_id").n_unique().alias("run_count"))
    .filter(pl.col("span_name") == "chat ministral-3-3b-instruct-2512")["run_count"]
)

total_tool_call = (
    df.filter(pl.col("process_pid") == 28062)
    .group_by("span_name")
    .len()
    .filter(pl.col("span_name") == "running tool")["len"][0]
)

parallel_tool_calls = (
    df.filter(pl.col("process_pid") == 28062)
    .filter(pl.col("span_name") == "running tool")
    .group_by("trace_id")
    .len()
    .filter(pl.col("len") > 1)
    .get_column("len")
    .sum()
)

average_tool_call = (
    df.filter(pl.col("process_pid") == 28062)
    .filter(pl.col("span_name") == "running tool")
    .group_by("trace_id")
    .len()
    .filter(pl.col("len") > 1)
    .get_column("len")
    .mean()
)

# tool percentage success rate
error_count = df.filter(pl.col("otel_status_code") == "ERROR").height
total_count = df.filter(pl.col("process_pid") == 28062).height
tools_success_rate = (error_count / total_count) * 100

# ---tokens--- #
total_tokens_turn = (
    df.sort("end_timestamp", descending=False)
    .group_by("trace_id")
    .last()
    .select(["input_tokens", "output_tokens"])
    .sum()
    .sum_horizontal()
    .alias("total_tokens")
)


total_input_tokens = (
    df.sort("end_timestamp", descending=False)
    .group_by("trace_id")
    .last()
    .select("input_tokens")
    .sum()
    .get_column("intput_tokens")
    .alias("total_output_tokens")
)

total_output_token = (
    df.sort("end_timestamp", descending=False)
    .group_by("trace_id")
    .last()
    .select("output_tokens")
    .sum()
    .get_column("output_tokens")
    .alias("total_output_tokens")
)

# --- Tool calls ---#
# Series of all the tool calls in order of start_timestamp
ordered_tool_calls = (
    df.sort("start_timestamp")
    .filter(pl.col("message").str.starts_with("running tool:"))["message"]
    .str.split(": ")
    .list.last()
)

unique_tool_calls = ordered_tool_calls.n_unique()

max_consecutive_tool = (
    ordered_tool_calls.to_frame("tool")
    .with_row_index()
    .with_columns(
        # Group ID: increments each time the value changes
        (pl.col("tool") != pl.col("tool").shift(1))
        .cum_sum()
        .alias("group")  # this contains tool name or null if no tool name
    )
    .group_by("group")
    .agg(pl.col("tool").first(), pl.len().alias("max_consecutive_tool"))
    .select(pl.col("max_consecutive_tool").max())
)

avg_tool_calls_per_turn = (
    df.filter(pl.col("span_name") == "running tool")
    .group_by("trace_id")
    .len()
    .mean()
    .select(pl.col("len").round(2).alias("mean"))
)

# ---run info--- #
pattern = r'"case_solved":\s*(true)'
run_success = df["attribute_messages"].str.extract(pattern).is_not_null().any()


# hypothesis_verified = df["attribute_messages"].str.extract(pattern).is_not_null().any()


# 3. Create a new dataframe
run_logs_agg = df.group_by("process_pid").agg(
    [
        total_turns,
        total_tool_call,
        parallel_tool_calls,
        average_tool_call,
        tools_success_rate,
        total_tokens_turn,
        total_input_tokens,
        total_output_token,
        unique_tool_calls,
        max_consecutive_tool,
        avg_tool_calls_per_turn,
        run_success,
    ]
)
