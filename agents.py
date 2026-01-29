from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from tools import (
    SupervisorContext,
    check_fingerprints,
    get_crime_scene_details,
    get_forensic_evidence,
    get_room_names,
    get_suspect_background,
    get_suspect_names,
    get_timeline_entry,
    get_tool_list,
    get_weapons_names,
    get_witness_statement,
    process_info,
    validate_solution,
    verify_alibi,
)

"""
3 agents:
- supervisor: he decides, can validate a guess and return the fibnal answer
- researcher: he can use the tools to query informations
- processor: he processes informations passed by the supervisor

I use LM studio to be able to use a MLX model with pydantic-ai. LM studio provide a OpenAI compatible API
2 actions to start lm studio (in the terminal):
- 'lms server start'
To see if the server is running you can do 'curl http://localhost:1234/v1/models'. It should return the list of the downloaded models - can help find the exact model name
- 'lms load <model_name>'
(optional) - 'lms ps' to see the loaded model
Use the GUI if you're not comfortable with the cli, it can all be done there + has server logs
"""


# Supervisor Agent - orchestrates workflow
supervisor_model = OpenAIChatModel(
    model_name="lfm2.5-1.2b-instruct-mlx",  # model name has to be filled but the actual name do not matter, only the correct url is required)
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:1234/v1",
    ),
)


class SupervisorDecision(BaseModel):
    action: Literal["delegate_to_researcher", "submit_answer"]
    instruction: str


supervisor_agent = Agent(
    supervisor_model,
    system_prompt="""You are a supervisor coordinating research and processing tasks.
    You have researcher at your service. You have to ask him to use tools to gether informations
    Your information can be processed using 'process_info'

    YOUR ROLE:
    - Break down the investigation into small, single-step tasks
    - move step at a time, start by gathering the suspect, weapon and rooms lists
    - Build understanding incrementally by asking simple requests to your agents
    - Give one clear instruction at a time to your agent

    Use the validation tool to test you hypothesis.
    Only submit the answer when you are sure about your hypothesis
 """,
    deps_type=SupervisorContext,
    output_type=SupervisorDecision,
    tools=[validate_solution, get_tool_list, process_info],
)

research_model = OpenAIChatModel(
    model_name="lfm2.5-1.2b-instruct-mlx",
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:1234/v1",
    ),
)

research_agent = Agent(
    research_model,
    system_prompt="""You are a research agent that executes exact instructions.
    STRICT RULES:
    1. Do ONLY what the supervisor explicitly requests
    2. Make EXACTLY ONE tool call per task
    3. Report only the direct result - no interpretation
    4. Keep responses under 3 sentences
    5. Do NOT chain multiple investigations
    6. Do NOT make assumptions about what else to check
""",
    model_settings={"temperature": 0.0},
    tools=[
        get_room_names,
        get_suspect_names,
        get_weapons_names,
        get_crime_scene_details,
        get_witness_statement,
        get_forensic_evidence,
        get_suspect_background,
        get_timeline_entry,
        check_fingerprints,
        verify_alibi,
    ],
)
# Processing Agent - transforms and processes data
process_model = OpenAIChatModel(
    model_name="lfm2.5-1.2b-instruct-mlx",
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:1234/v1",
    ),
)

process_agent = Agent(
    process_model,
    system_prompt="""Process the information passed to you.
    Synthetize it, highlight the most important point.
    Keep it concise
    DO NOT give instructions or recommendations, you only process""",
    model_settings={"temperature": 0.0},
)
