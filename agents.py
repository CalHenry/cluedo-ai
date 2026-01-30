from typing import Literal

from pydantic import BaseModel, Field
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
The processor is accessible as a tool by the supervisor that do not know that the processor is in fact an ai.

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
    model_name="ministral-3-3b-instruct-2512",
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:1234/v1",
    ),
)


class SupervisorDecision(BaseModel):
    action: Literal["delegate_to_researcher", "submit_answer"] = Field(
        description="Choose 'delegate_to_researcher' to ask your researcher or 'submit_answer' hen you are sure about your hypothesis"
    )
    instruction: str = Field(description="Simple instruction string")


supervisor_agent = Agent(
    supervisor_model,
    system_prompt="""SUPERVISOR - Cluedo Investigation

    You must respond in this format:
    {
      "action": "[choose one: delegate_to_researcher OR submit_answer]",
      "instruction": "[your message]"
    }

    TWO WAYS TO RESPOND:

    OPTION 1 - Ask your researcher to do something:
    {
      "action": "delegate_to_researcher",
      "instruction": "Use list_suspects to get all suspect names"
    }

    OPTION 2 - Provide final answer:
    {
      "action": "submit_answer",
      "instruction": "Suspect: Scarlet, Weapon: Rope, Room: Kitchen"
    }

    TOOLS YOU CAN USE DIRECTLY:
    - process_info(data) - Process information
    - validate_solution(suspect, weapon, room) - Check if hypothesis is correct

    YOUR WORKFLOW:
    □ Ask researcher for suspect list (use OPTION 1)
    □ Ask researcher for weapon list (use OPTION 1)
    □ Ask researcher for room list (use OPTION 1)
    □ Ask researcher to gather clues (use OPTION 1, can repeat)
    □ Call process_info yourself to analyze
    □ Call validate_solution yourself to test
    □ Only use OPTION 2 when validation passes

    FIRST RESPONSE: Use OPTION 1 to ask researcher for suspect list.
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
