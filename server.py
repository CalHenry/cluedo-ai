from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from agents import research_agent, supervisor_agent
from game_engine import CluedoGameEngine
from tools import create_game_tools

# set up
engine = CluedoGameEngine(seed=42)
scenario = engine.generate_scenario()
tool_impls = create_game_tools(engine)


# ---Tools---#
# Tools are for the researcher to use except for the validation tool that is only for the supervisor
@research_agent.tool
def get_crime_scene_details(room_name: str) -> str:
    """Examine a specific room for evidence..."""
    return tool_impls["get_crime_scene_details"](room_name)


@research_agent.tool
def get_witness_statement(witness_name: str) -> str:
    """Retrieve witness statement..."""
    return tool_impls["get_witness_statement"](witness_name)


@research_agent.tool
def get_forensic_evidence(evidence_id: str) -> str:
    """Retrieve detailed forensic analysis of a specific piece of evidence"""
    return tool_impls["get_forensic_evidence"](evidence_id)


@research_agent.tool
def get_suspect_background(suspect_name: str) -> dict:
    """
    Get background information about a suspect including their relationship to the victim,
    possible motive, and opportunity to commit the crime.
    """
    return tool_impls["get_suspect_background"](suspect_name)


@research_agent.tool
def get_timeline_entry(time_slot: str) -> str:
    """Get events that occurred during a specific time window on the night of the murder."""
    return tool_impls["get_timeline_entry"](time_slot)


@research_agent.tool
def check_fingerprints(object_name: str) -> dict:
    """Check fingerprint analysis for a specific object or evidence item."""
    return tool_impls["check_fingerprints"](object_name)


@research_agent.tool
def verify_alibi(suspect_name: str, time_slot: str) -> dict:
    """Cross-reference a suspect's alibi against timeline and evidence."""
    return tool_impls["verify_alibi"](suspect_name, time_slot)


# Supervisor only has the validation tool
@supervisor_agent.tool
def validate_solution(suspect: str, weapon: str, location: str) -> str:
    """Submit a solution to the murder mystery for validation"""
    return tool_impls["validate_solution"](suspect, weapon, location)
