from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from tools2 import (
    check_fingerprints,
    get_crime_scene_details,
    get_forensic_evidence,
    get_room_names,
    get_suspect_background,
    get_suspet_names,
    get_timeline_entry,
    get_tool_list,
    get_weapons_names,
    get_witness_statement,
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
        api_key="lm_studio",  # also useless but emphasize we use LM studio
    ),
)


class SupervisorDecision(BaseModel):
    action: Literal["delegate_to_researcher", "delegate_to_processor", "submit_answer"]
    instruction: str


supervisor_agent = Agent(
    supervisor_model,
    system_prompt="""You are a supervisor agent that coordinates research and processing tasks.
You have 2 agents at your service that you have ot delegate tasks to.
- Researcher agent: Use tools to gather informations.
- Processer agent: Analyse and synthetize informations. You have to give him the findings reported by the resercher agent.

 Your job is to:
 1. Understand the user's request
 2. Delegate to the Research Agent for data gathering and analysis
 3. Then delegate to the Processing Agent for transformation and formatting
 4. Finally, compile the results into a comprehensive answer

 Be clear and concise in your delegation. The agents are not tools.
 """,
    output_type=SupervisorDecision,
    tools=[validate_solution, get_tool_list],
)

research_model = OpenAIChatModel(
    model_name="lfm2.5-1.2b-instruct-mlx",
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm_studio",
    ),
)

research_agent = Agent(
    research_model,
    system_prompt="""You are a research agent specialized in data gathering.
You are under a supervisor that will tell you what to do. You don't analyse data you just gather it for your supervisor
 You have access to tools to gather data.
""",
    tools=[
        get_room_names,
        get_suspet_names,
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
        api_key="lm_studio",
    ),
)

process_agent = Agent(
    process_model,
    system_prompt="""You are a processing agent specialized in data transformation and formatting.

 Your job is to take research findings and process them into useful insights and well-formatted output.""",
)
