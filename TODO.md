# Next step: Create a Workflow monitor using XGBoost

Inpired by this paper: [DETECTING SILENT FAILURES IN MULTI-AGENTIC AI TRAJECTORIES](https://arxiv.org/pdf/2511.04032)


The workflow works but sometimes fails or act in unexpected ways.  
I want to try to use XGBoost as an anomaly detector. 
I want to use the model to detect:
- Drifts (The agent diverges from the intended path, selecting tools or subsequent
agents that are irrelevant for an input query.)
- Cycles (The agent repeatedly invokes itself or other agents/tools by re-planning
resulting in redundant loops, wasted computation.)
- Missing details in the final output (The agent returns a response without errors, but misses crucial information requested in the input query.)
- Tool failure (External tools(APIs) may fail silently, return unexpected results, hit
rate limits—issues that the agent may not detect or handle gracefully.)
- Context propagation failures (Failure in propagating the correct context to dependent agents/tools.)
I have seen those errors in the many test runs I did.


### What is needed: 
- Trained XGBoost
  - Data from each run to train the model
  - Shoudl start with a small sample of ~500 runs 

### Steps: 
- 1: Extract the data

Collect data from the runs using logfires logs. Export them using a SQL query to the logfire API

- 2: Feature engineering

I have all the informations about the run, all the messages, context, tool called, arguments...
This is informations that needs to be transformed into features for XGBoost.
Draft of variables to create:

    - attempts,
    - total_turns,
    Tool usage:
      - total_tool_calls,
      - parallel_tool_calls
      - repeated_toolc_calls
      - unique_tool_count,
      - max_consecutive_same_tool,
      - avg_tools_per_turn,
      - tool_success_rate,
    - avg_tool_response_length,
    - null_tool_responses,
    - investigated_verified_suspect_flag (when the tool responded 'alibi_verified: true' but the model keeps searching)
    - total_tokens,
    - final_answer_correct (label)

- 3: Train
- 4: Test
