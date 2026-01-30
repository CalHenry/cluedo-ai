# Cluedo for AI

This project is a simple version of a Cluedo like game for AI agents to play.  
The goal is to create a playground for ai agents to interact with each other so I can inspect each interaction from the first user prompt to the final answer of the models.   
I want to understand better how context is created, managed and relevant in an agentic workflow, so I build this playground where I control each element.  

The design is simple:
- 100% local
- no MCP server
- 3 agents: 1 supervisor, 1 researcher, 1 processor
- a limited set of tools

### Stack
- I use a MLX model for better performance on my mac when running the agents.
- [pydantic-ai](https://ai.pydantic.dev/) for the agentic framework and [Pydantic's Logfire](https://docs.pydantic.dev/latest/integrations/logfire/) to inspect every interaction between the agents.
- [LM studio](https://lmstudio.ai/) to be able to use a MLX model locally with pydantic-ai. LM studio emits a OpenAI compatible API for the model that is supported by pydantic-ai. LM studio makes it easy to download and use models from HuggingFace. It also comes as a CLI, once the model is downloaded, the setup takes 2 simple commands (start the server and load the LLM in memory).

Folder structure:
```sh
.
├── agents.py        # <- model and agents set up, system prompts
├── game_engine.py   # <- game related objects and pre-made reports
├── main.py          # <- logfire setup and execution function and logic, orchestration and user prompts
└── tools.py         # <- just the tools
```

### The game:  
My interest is not how well the ai can solve the case but how much I can make them interact with each other.  
Therefore, the game is very simple, the tools return small text with obvious information. I introduced enough randomness and variations, so the AI have to use the tools a few times before discovering the key elements.   
A Cluedo game has 6 suspects, 6 rooms and 6 weapons. Here instead of players we have an AI team investigating the murder of Dr.Black.  
They can gather information about the suspects, the rooms and the weapons using the tools. Finally, the supervisor can submit and answer to the case by providing a value for room, suspect and weapon. This ends the game regardless of the hypothesis being correct or not.

### Tools:
Tools are exclusive to the researcher agent or the supervisor.  
The supervisor can use 3 tools:
- get the list of tools available in the game (goal is to have the supervisor ask the researcher to use a specific tool)
- validate a hypothesis by passing a value for room, weapon and suspect. Returns true or false. I expected this tool to be very informative for the supervisor, but I never saw the supervisor analyzing this output or note that he has found the right weapon for example
- process_information: use the processor agent under the hood but the supervisor doesn't know that

All the other tools are for the researcher to use and are simple functions that return text about the suspects, the crime scene, the weapons...  

<details>
<summary>List of researcher's tools:</summary>
- get_room_names() <br>
- get_suspect_names() <br>
- get_weapons_names() <br>
- get_crime_scene_details(room_name) <br>
- get_witness_statement(witness_name) <br>
- get_forensic_evidence(evidence_id) <br>
- get_suspect_background(suspect_name) <br>
- get_timeline_entry(time_slot) <br>
- check_fingerprints(object_name) <br>
- verify_alibi(suspect_name, time_slot) <br>
- process_information() (processor agent) <br> 
</details>

### Agents and model:
I use 2 different LLMs: 
- [Ministral-3-3B-Instruct-2512-4bit](https://huggingface.co/mlx-community/Ministral-3-3B-Instruct-2512-4bit) for the supervisor
- [LFM2.5-1.2B-Instruct-4bit](https://huggingface.co/mlx-community/LFM2.5-1.2B-Instruct-4bit) for the researcher and the processor

The models are small and this impacts directly how the agents plays the game, understand their role and follow the instructions.   
Pros:   
- Very fast inference, no pressure on the RAM and the hardware handles this very well (M1pro).
- Makes me work a lot on the prompts (and understand a lot about prompt engineering)  

Cons:  
- Models are stupid without prompt engineering. They pass next to very obvious information, they don't weight the different information enough (for example, at the end of the murder room description, there is "⚠️  THIS IS THE MURDER SCENE", but it doesn't strike the ai whatsoever)

Some of the issues related to the small sizes of the models should be manageable with better prompts.

The processor was previously a standalone agent like the researcher is, but I had a major issue: the supervisor was not using the processor and was asking the researcher to process info.  
To solve this i included the processor agent inside a tool available to the supervisor only.

### Current state of the project:
The game works, the agents interact with each other but don't solve the case yet. I consider to have reached my goal:
- I got a playground for ai to interact with each others.  
- I don't really manage and built the context, currently it's just under a variable 'supervisor_memory' and I never interact with it
- a run takes ~2 minutes
- the agents are sometimes successful. They are often close to the right answer (2/3 correct)

### Personal observations:

- I imagined that the cooperation and interactions between the agents would work out of the box (it doesn't)
- I expected the agents to understand who they were in the context of the investigation.
- I didn't expect the prompts to have such and influence on the behaviors of the agents (for example, the supervisor output is structured and validated. With a prompt with an action verb and one of the possible values (like 'delegate_researcher') for his answer, the supervisor would try to use 'delegate_researcher' as a tool, despite this tool not existing)
- Using AI to create the fake reports for the game was very handy

### Explore recorded runs 
In the folder 'run_examples', you can see the logfire logs of a few runs (configured by 'logfire.configure()' line 14 in main.py).  
I capture this content using the following command:
```sh
uv run main.py | tee output.txt # time uv run main.py to see the exec time
```
I added circles emojis to easily see when the researcher (🔵) or the processor (🟣) are used by the supervisor.

## Try it out yourself

### Requirements
- Powerful enough hardware
- LM studio installed
- LLM downloaded on disk with LM studio
- Python >= 3.12
- logfire>=4.19.0
- pydantic-ai-slim[openai]>=1.46.0

>[!IMPORTANT] 
> To use Logfire, you need to create a free account. You can follow the [Getting Started instructions](https://logfire.pydantic.dev/docs/).

1. clone the repo:
```sh
git clone https://github.com/CalHenry/cluedo-ai.git
```

2. Set up LM studio  
You can do this in the GUI, or in the terminal:  
- Start the LM studio server. 
```sh
lms server start
```
Default port is '1234'. (optional) Test it with 'curl http://localhost:1234/v1/models'

- Load the llm(s) in memory
```sh
lms load
```
You will be prompted with the list of available models and asked to choose the one you want to load. You can also pass the model name as an argument directly

4. From there you can run the main script with python or uv:  
```sh
uv run main.py
```
