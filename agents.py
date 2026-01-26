from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# 3 agents:
# - supervisor - he decides, can validate a guess and return the fibnal answer
# - researcher - he can use the tools to query informations
# - processor - he processes informations passed by the supervisor


# Supervisor Agent - orchestrates workflow
supervisor_model = OpenAIChatModel(
    model_name="qwen2.5:3b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
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
    model_name="qwen2.5:3b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

research_agent = Agent(
    research_model,
    system_prompt="""You are a research agent specialized in data gathering and analysis.

 You have access to tools to gather data.

 Your job is to thoroughly investigate the subject using these tools and return detailed findings.
 Use multiple tools to build a comprehensive analysis.""",
)

# Processing Agent - transforms and processes data
processing_model = OpenAIChatModel(
    model_name="qwen2.5:3b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

processing_agent = Agent(
    processing_model,
    system_prompt="""You are a processing agent specialized in data transformation and formatting.

 Your job is to take research findings and process them into useful insights and well-formatted output.""",
)
