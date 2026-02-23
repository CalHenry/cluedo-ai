# /// script
# dependencies = [
#     "marimo",
#     "polars==1.38.1",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Variables to obtain:

    - total_turns, #int ✅
    Tool usage:
      - total_tool_calls, #int
      - parallel_tool_calls
      - repeated_tool_calls
      - unique_tool_count,  # int
      - max_consecutive_same_tool, # int
      - avg_tools_per_turn, # int
      - tool_success_rate, # %age
    - avg_tool_response_length, # int
    - null_tool_responses, # bool
    - investigated_verified_suspect_flag (when the tool responded 'alibi_verified: true' but the model keeps searching)
    - total_tokens,  # int
    - final_answer_correct (label)
    """)
    return


@app.cell
def _(pl):
    df = pl.read_csv("data/*csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("process_pid") == 28062).group_by(["span_name"]).n_unique()
    return


@app.cell
def _():
    return


@app.cell
def _(df, pl):
    # get number of runs
    df_agg = (
        df.filter(pl.col("process_pid") == 28062)
        .group_by("span_name")
        .agg(pl.col("trace_id").n_unique().alias("run_count"))
        .filter(pl.col("span_name") == "chat ministral-3-3b-instruct-2512")[
            "run_count"
        ]
    )
    return (df_agg,)


@app.cell
def _(df, pl):
    # get count of tool calls
    df.filter(pl.col("process_pid") == 28062).group_by("span_name").len().filter(
        pl.col("span_name") == "running tool"
    )["len"][0]
    return


@app.cell
def _(df_agg):
    df_agg
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


if __name__ == "__main__":
    app.run()
