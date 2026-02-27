# /// script
# dependencies = [
#     "marimo",
#     "polars==1.38.1",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook is a scratchpad for the script 'process_logs.py'.
    Here we create the functions and expressions to create the variables we want from the raw logs file.
    Also used to explore and understand the logs.
    This notebook is not well organized nor documented.
    This work leads to the script 'logs/process_logs.py' that is orgnized and documented.

    Variables to obtain:

    - total_turns,
    Tool usage:
      - total_tool_calls, #int
      - parallel_tool_calls
      - repeated_tool_calls
      - unique_tool_count,
      - max_consecutive_same_tool,
      - avg_tools_per_turn, # int
      - tool_success_rate, # %age
    - null_tool_responses, # bool
    - investigated_verified_suspect_flag (when the tool responded 'alibi_verified: true' but the model keeps searching)
    - total_tokens,  # int
    - final_answer_correct (label)
    """)
    return


@app.cell
def _(pl):
    df = pl.read_csv("logs/data/raw/df_2026-02-25.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, pl):
    # get number of runs
    (
        df.group_by("span_name")
        .agg(pl.col("trace_id").n_unique().alias("run_count"))
        .filter(pl.col("span_name") == "chat ministral-3-3b-instruct-2512")[
            "run_count"
        ]
    )
    return


@app.cell
def _(df, pl):
    # get total count of tool calls
    (
        df.group_by("span_name")
        .len()
        .filter(pl.col("span_name") == "running tool")["len"][0]
    )
    return


@app.cell
def _(df, pl):
    # count parallel tool calls
    (
        df.filter(pl.col("span_name") == "running tool")
        .group_by("trace_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("len")
        .sum()
    )
    return


@app.cell
def _(df, pl):
    (
        df.filter(pl.col("span_name") == "running tool")
        .group_by("trace_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("len")
        .mean()
    )
    return


@app.cell
def _(df, pl):
    # average tool call count per run
    (
        df.filter(pl.col("span_name") == "running tool")
        .group_by("trace_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("len")
        .mean()
    )
    return


@app.cell
def _(df, pl):
    (df.filter(pl.col("otel_status_code") == "ERROR").height)
    return


@app.cell
def _(df, pl):
    # tool % success rate
    error_count = df.filter(pl.col("otel_status_code") == "ERROR").height

    total_count = df.height

    (error_count / total_count)
    return


@app.cell
def _(df):
    # total tokens for a run
    (
        df.sort("end_timestamp", descending=False)
        .group_by("trace_id")
        .last()
        .select(["input_tokens", "output_tokens"])
        .sum()
        .sum_horizontal()
        .alias("total_tokens")
    )

    # total_input
    (
        df.sort("end_timestamp", descending=False)
        .group_by("trace_id")
        .last()
        .select("input_tokens")
        .sum()
        .get_column("input_tokens")
        .alias("total_output_tokens")
    )

    # total_output
    (
        df.sort("end_timestamp", descending=False)
        .group_by("trace_id")
        .last()
        .select("output_tokens")
        .sum()
        .get_column("output_tokens")
        .alias("total_output_tokens")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the same **'span_id'**, when 'span_name' = 'chat ministral-3-3b-instruct-2512', the previous row is = 'agent run' and looks like a dup of the above with no finish reason and same token usage.
    Not true when 'span_name' = 'chat lfm2.5-1.2b-instruct-mlx'
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Aggregation to do:
    - goal: 1 row per run

    3 levels:
    - span (span_id)
    - trace
    - run

    Raw data is at the span level


    - We can count the iterations with by grouping by 'trac_id' and 'span_name'
    """)
    return


@app.cell
def _(df, pl):
    (
        df.filter(pl.col("message").str.starts_with("running tool:"))["message"]
        .str.split(": ")
        .list.last()
    )
    return


@app.cell
def _(df, pl):
    # unique tool count 2
    (
        df.sort("start_timestamp")
        .filter(pl.col("message").str.starts_with("running tool:"))["message"]
        .str.split(": ")
        .list.last()
    )
    return


@app.cell
def _(df, pl):
    (
        (
            df.sort("start_timestamp")
            .filter(pl.col("message").str.starts_with("running tool:"))["message"]
            .str.split(": ")
            .list.last()
        )
        .to_frame("tool")
        .with_row_index()
        .with_columns(
            # Group ID: increments each time the value changes
            (pl.col("tool") != pl.col("tool").shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("group")
        )
        .group_by("group")
        .agg(pl.col("tool").first(), pl.len().alias("count"))
        .select(pl.col("count").max())
        .item()
    )
    return


@app.cell
def _(df, pl):
    # Focus on the struct variable "attribute_messages" that contains all the tool call content and response from the tools.
    # This variable contains info about max consecutive tool call, unique tool calls and repeated tool calls

    # Unique tool count
    (
        df.filter(pl.col("message").str.starts_with("running tool:"))["message"]
        .str.split(": ")
        .list.last()
        .n_unique()
    )

    # max consecutive same tool
    series = (
        df.sort("start_timestamp")
        .filter(pl.col("message").str.starts_with("running tool:"))["message"]
        .str.split(": ")
        .list.last()
    )
    (
        series.to_frame("tool")
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

    # avg_tools_per_turn
    (
        df.filter(pl.col("span_name") == "running tool")
        .group_by("trace_id")
        .len()
        .mean()
        .select(pl.col("len").round(2).alias("mean"))
    )
    return


@app.cell
def _(df, pl):
    df.group_by("span_name").agg(pl.col("trace_id").n_unique().alias("run_count"))
    return


@app.cell
def _(df, pl):
    # if the run succeded or not
    df.filter((pl.col("finish_reason") == '["tool_call"]')).select(
        pl.col("attribute_messages").str.contains(r'"case_solved"\s*:\s*true')
    )["attribute_messages"].any()
    return


@app.cell
def _(df):
    # investigated_verified_suspect_flag
    pattern = r'"case_solved":\s*(true)'

    df["attribute_messages"].str.extract(pattern).is_not_null().any()
    return


@app.cell
def _(df):
    # investigated_verified_suspect_flag
    pattern = r'"case_solved":\s*(true)'

    df["attribute_messages"].str.extract(pattern).is_not_null().any()
    return


@app.cell
def _(df, pl):
    # Unique tools per run
    unique_tool_per_run = (
        pl.col("message")
        .filter(pl.col("message").str.starts_with("running tool:"))
        .str.split(": ")
        .list.last()
        .n_unique()
        .alias("unique_tools_per_run")
    )

    tool_call = (
        pl.col("span_name")
        .filter(pl.col("span_name") == "running tool")
        .count()
        .alias("tool_calls_per_run")
    )

    (
        df.group_by("process_pid").agg(
            [
                unique_tool_per_run,
                tool_call,
            ]
        )
    )
    return


@app.cell
def _(lf, pl):
    tool_spans = lf.filter(pl.col("span_name") == "running tool").with_columns(
        pl.col("message").str.split(": ").list.last().alias("tool_name")
    )
    return (tool_spans,)


@app.cell
def _(pl, tool_spans):
    (
        tool_spans.sort("start_timestamp")
        .with_row_index()
        .with_columns(
            (
                (
                    pl.col("tool_name")
                    != pl.col("tool_name").shift(1).over("process_pid")
                )
                | (pl.col("process_pid") != pl.col("process_pid").shift(1))
            )
            .cum_sum()
            .alias("group")
        )
        .group_by("process_pid", "group")
        .agg(pl.col("tool_name").first(), pl.len().alias("consecutive_count"))
        .group_by("process_pid")
        .agg(pl.col("consecutive_count").max())
    )
    return


@app.cell
def _(pl, tool_spans):
    tool_calls_per_turn = tool_spans.group_by(["process_pid", "trace_id"]).agg(
        pl.len().alias("tools_in_turn")
    )
    return


@app.cell
def _(ordered_tool_calls, pl):
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
    return


if __name__ == "__main__":
    app.run()
