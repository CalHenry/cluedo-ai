from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 3 agents:
# - supervisor - he decides, can validate a guess and return the fibnal answer
# - researcher - he can use the tools to query informations
# - processor - he processes informations passed by the supervisor

# I use LM studio to be able to use a MLX model with pydantic-ai. LM studio provide a OpenAI compatible API
# 2 actions to start lm studio (in the terminal):
# - 'lms server start'
# To see if the server is running you can do 'curl http://localhost:1234/v1/models'. It should return the list of the downloaded models - can help find the exact name of the model
# - 'lms load <model_name>'
# (optional) - 'lms ps' to see the loaded model

# Supervisor Agent - orchestrates workflow
supervisor_model = OpenAIChatModel(
    model_name="lfm2.5-1.2b-instruct-mlx",  # model name has to be filled but a the actual string do not matter, only the correct url is required)
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm_studio",  # also useless but emphasize we use LM studio
    ),
)

supervisor_agent = Agent(
    supervisor_model,
    system_prompt="""You are a supervisor agent that coordinates research and processing tasks.

 Your job is to:
 1. Understand the user's request
 2. Delegate to the Research Agent for data gathering and analysis
 3. Then delegate to the Processing Agent for transformation and formatting
 4. Finally, compile the results into a comprehensive answer

 Be clear and concise in your delegation.""",
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
    system_prompt="""You are a research agent specialized in data gathering and analysis.

 You have access to tools to gather data.

 Your job is to thoroughly investigate the subject using these tools and return detailed findings.
 Use multiple tools to build a comprehensive analysis.""",
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
