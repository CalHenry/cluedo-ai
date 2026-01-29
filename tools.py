import inspect
import random

from pydantic import BaseModel
from pydantic_ai.agent import RunContext

from game_engine import CluedoGameEngine

game_engine = CluedoGameEngine(seed=42)
scenario = game_engine.generate_scenario()


class SupervisorContext(BaseModel):
    gathered_info: str


async def process_info(ctx: RunContext[SupervisorContext]) -> str:
    """Process the last response of the researcher. Return information passed processed and synthetized"""
    from agents import process_agent

    print("🟣")
    r = await process_agent.run(
        f"Information to process: {ctx.deps.gathered_info}",
        usage=ctx.usage,
    )
    return r.output


def get_room_names() -> str:
    """Return the rooms names in a list"""
    rooms_names = ", ".join(game_engine.ROOMS)
    return rooms_names


def get_suspect_names() -> str:
    """Return the suspect names in a list"""
    suspects_names = ", ".join(game_engine.SUSPECTS)
    return suspects_names


def get_weapons_names() -> str:
    """Return the weapons names in a list"""
    weapons_names = ", ".join(game_engine.WEAPONS)
    return weapons_names


def get_crime_scene_details(room_name: str) -> str:
    """
    Examine a specific room for evidence and details about the crime scene.

    Args:
        room_name: The name of the room to investigate (Study, Library, Kitchen,
                  Conservatory, Billiard Room, or Lounge)

    Returns:
        Detailed description of the room and any visible evidence
    """
    if not game_engine.scenario:
        return "ERROR: No active investigation. Game scenario not initialized."

    # Normalize room name
    room_name = room_name.strip()

    if room_name not in game_engine.ROOMS:
        available = ", ".join(game_engine.ROOMS)
        return f"ERROR: Unknown room '{room_name}'. Available rooms: {available}"

    # Check if there's evidence in this room
    if room_name in game_engine.scenario.crime_scene_evidence:
        evidence = game_engine.scenario.crime_scene_evidence[room_name]

        response = f"""CRIME SCENE REPORT - {room_name.upper()}
{"=" * 60}

ROOM DESCRIPTION:
The {room_name} is {"the primary crime scene" if room_name == game_engine.scenario.murder_location else "being investigated as part of the broader inquiry"}.

VISIBLE EVIDENCE:
- Evidence ID: {evidence.evidence_id}
- Item Found: {evidence.item_name}
- Details: {evidence.description}

TIME OF DISCOVERY: 10:15 PM (approximately 15-45 minutes after estimated time of death)

FORENSIC TEAM STATUS: Evidence has been collected and tagged for analysis.
For detailed forensic results, use get_forensic_evidence() with the evidence ID.

{"⚠️  THIS IS THE MURDER SCENE" if room_name == game_engine.scenario.murder_location else "No signs of struggle detected in this room."}
"""
    else:
        response = f"""CRIME SCENE REPORT - {room_name.upper()}
{"=" * 60}

ROOM DESCRIPTION:
The {room_name} has been checked during the initial sweep.

VISIBLE EVIDENCE:
No significant evidence found in this location.

NOTES:
Room appears undisturbed. No signs of recent activity related to the crime.
"""

    return response


def get_witness_statement(witness_name: str) -> str:
    """
    Retrieve the statement from a witness/suspect.

    Args:
        witness_name: Name of the person to interview (Miss Scarlet, Colonel Mustard,
                     Mrs White, Mr Green, Mrs Peacock, or Professor Plum)

    Returns:
        The witness's statement including their alibi and testimony
    """
    if not game_engine.scenario:
        return "ERROR: No active investigation. Game scenario not initialized."

    # Normalize witness name
    witness_name = witness_name.strip()

    if witness_name not in game_engine.SUSPECTS:
        available = ", ".join(game_engine.SUSPECTS)
        return (
            f"ERROR: Unknown person '{witness_name}'. Available witnesses: {available}"
        )

    statement = game_engine.scenario.witness_statements[witness_name]
    details = game_engine.SUSPECT_DETAILS[witness_name]

    response = f"""WITNESS STATEMENT - {witness_name.upper()}
{"=" * 60}

BACKGROUND:
- Occupation: {details["occupation"]}
- Relationship to Victim: {details["relationship"]}

STATEMENT TAKEN: {statement.time_of_statement}

ALIBI:
{statement.alibi}

TESTIMONY:
{statement.testimony}

REPORTED LOCATION DURING INCIDENT: {statement.location_during_murder}

INTERVIEWER NOTES:
{"⚠️  Alibi appears inconsistent with physical evidence" if witness_name == game_engine.scenario.murderer else "Statement appears consistent. No obvious deception detected."}
"""

    return response


def get_forensic_evidence(evidence_id: str) -> str:
    """
    Retrieve detailed forensic analysis of a specific piece of evidence.

    Args:
        evidence_id: The unique identifier for the evidence (e.g., FOR_WEAPON_001)

    Returns:
        Detailed forensic analysis report
    """
    if not game_engine.scenario:
        return "ERROR: No active investigation. Game scenario not initialized."

    evidence_id = evidence_id.strip().upper()

    if evidence_id not in game_engine.scenario.forensic_evidence:
        available = ", ".join(game_engine.scenario.forensic_evidence.keys())
        return f"ERROR: Unknown evidence ID '{evidence_id}'. Available evidence: {available}"

    evidence = game_engine.scenario.forensic_evidence[evidence_id]

    related = ""
    if evidence.related_evidence_ids:
        related = f"\nRELATED EVIDENCE: {', '.join(evidence.related_evidence_ids)}"

    response = f"""FORENSIC ANALYSIS REPORT
{"=" * 60}

EVIDENCE ID: {evidence.evidence_id}
ITEM: {evidence.item_name}
ANALYSIS TYPE: {evidence.analysis_type}

FINDINGS:
{evidence.findings}

SIGNIFICANCE:
{evidence.significance}{related}

LABORATORY: Metropolitan Police Forensic Laboratory
ANALYSIS COMPLETED: 11:45 PM (same night)
CHAIN OF CUSTODY: Verified
"""

    return response


def get_suspect_background(suspect_name: str) -> dict:
    """
    Get background information about a suspect including their relationship to the victim,
    possible motive, and opportunity to commit the crime.

    Args:
        suspect_name: Name of the suspect to investigate

    Returns:
        Dictionary containing background, motive, and opportunity information
    """
    if not game_engine.scenario:
        return {"error": "No active investigation. Game scenario not initialized."}

    suspect_name = suspect_name.strip()

    if suspect_name not in game_engine.SUSPECTS:
        available = ", ".join(game_engine.SUSPECTS)
        return {
            "error": f"Unknown suspect '{suspect_name}'. Available suspects: {available}"
        }

    details = game_engine.SUSPECT_DETAILS[suspect_name]
    statement = game_engine.scenario.witness_statements[suspect_name]
    is_murderer = suspect_name == game_engine.scenario.murderer

    return {
        "name": suspect_name,
        "occupation": details["occupation"],
        "relationship": details["relationship"],
        "opportunity": (
            f"Reported location: {statement.location_during_murder}. "
            f"{'Evidence suggests presence at crime scene.' if is_murderer else 'Alibi partially verified.'}"
        ),
        "notes": (
            "High suspicion - inconsistencies detected"
            if is_murderer
            else "Standard investigation subject"
        ),
    }


def get_timeline_entry(time_slot: str) -> str:
    """
    Get events that occurred during a specific time window on the night of the murder.

    Args:
        time_slot: Time window to investigate (format: "HH:MM" for start of 15-min window,
                e.g., "21:00", "21:15", "21:30")

    Returns:
        Description of known events during that time period
    """
    if not game_engine.scenario:
        return "ERROR: No active investigation. Game scenario not initialized."

    time_slot = time_slot.strip()

    # Define the timeline based on the scenario
    # Murder occurs between 9:30 PM and 10:00 PM
    timeline = {
        "21:00": "Dinner concludes. Guests begin dispersing to various rooms.",
        "21:15": "Most guests settled in different areas. Casual conversations ongoing.",
        "21:30": "CRITICAL WINDOW: Murder estimated to occur between 9:30-10:00 PM. "
        f"{game_engine.scenario.murderer} last seen near the {game_engine.scenario.murder_location}.",
        "21:45": "CRITICAL WINDOW CONTINUES: Victim not responding to calls. Growing concern among guests.",
        "22:00": "Body discovered. Initial shock and confusion. Rooms being secured.",
        "22:15": "Police called. Guests instructed to remain in their locations. Initial statements taken.",
        "22:30": "Forensic team arrives. Crime scene cordoned off. Formal interviews begin.",
        "22:45": "Evidence collection in progress. Witnesses being separated for detailed questioning.",
    }

    if time_slot not in timeline:
        available = ", ".join(sorted(timeline.keys()))
        return (
            f"ERROR: Invalid time slot '{time_slot}'. Available time slots: {available}"
        )

    response = f"""TIMELINE ENTRY - {time_slot}
{"=" * 60}

{timeline[time_slot]}

STATUS: {"⚠️  CRITICAL TIME PERIOD" if time_slot in ["21:30", "21:45"] else "Documented"}
"""

    return response


def check_fingerprints(object_name: str) -> dict:
    """
    Check fingerprint analysis for a specific object or evidence item.

    Args:
        object_name: Name of the object to check (weapon name or evidence item)

    Returns:
        Dictionary containing fingerprint analysis results
    """
    if not game_engine.scenario:
        return {"error": "No active investigation. Game scenario not initialized."}

    object_name = object_name.strip()

    # Check if it's the murder weapon
    if object_name.lower() == game_engine.scenario.murder_weapon.lower():
        murderer = game_engine.scenario.murderer
        return {
            "object": game_engine.scenario.murder_weapon,
            "fingerprints_found": True,
            "matches": [murderer],
            "quality": "Partial prints recovered",
            "analysis": f"Fingerprints match records for {murderer}. "
            f"DNA traces also present. High confidence match.",
            "notes": "⚠️  Direct physical evidence linking suspect to weapon",
        }

    # Check if it's another weapon (red herring)
    elif object_name in game_engine.WEAPONS:
        # Random innocent person for red herring
        innocent_suspects = [
            s for s in game_engine.SUSPECTS if s != game_engine.scenario.murderer
        ]
        red_herring_suspect = random.choice(innocent_suspects)

        return {
            "object": object_name,
            "fingerprints_found": True,
            "matches": [red_herring_suspect, "Dr. Black (victim)"],
            "quality": "Clear prints recovered",
            "analysis": f"Multiple prints identified: {red_herring_suspect} and victim. "
            f"Prints appear several days old based on degradation analysis.",
            "notes": "Item likely handled during normal household activities",
        }

    # Check for evidence items from forensic evidence
    elif "fiber" in object_name.lower() or "fabric" in object_name.lower():
        return {
            "object": "Fabric fibers",
            "fingerprints_found": False,
            "matches": [],
            "quality": "N/A",
            "analysis": "Fingerprints cannot be recovered from fabric fibers. "
            "See forensic evidence FOR_FIBER_001 for textile analysis.",
            "notes": "Wrong evidence type for fingerprint analysis",
        }

    # Unknown object
    else:
        available_weapons = ", ".join(game_engine.WEAPONS)
        return {
            "error": f"Unknown object '{object_name}'. Available weapons: {available_weapons}. "
            f"For other evidence, use get_forensic_evidence() with evidence IDs."
        }


def verify_alibi(suspect_name: str, time_slot: str) -> dict:
    """
    Cross-reference a suspect's alibi against timeline and evidence.

    Args:
        suspect_name: Name of the suspect whose alibi to verify
        time_slot: Time to check (format: "HH:MM", e.g., "21:30")

    Returns:
        Dictionary containing alibi verification results
    """
    if not game_engine.scenario:
        return {"error": "No active investigation. Game scenario not initialized."}

    suspect_name = suspect_name.strip()
    time_slot = time_slot.strip()

    if suspect_name not in game_engine.SUSPECTS:
        available = ", ".join(game_engine.SUSPECTS)
        return {
            "error": f"Unknown suspect '{suspect_name}'. Available suspects: {available}"
        }

    # Valid time slots for the critical period
    valid_times = ["21:00", "21:15", "21:30", "21:45", "22:00"]
    if time_slot not in valid_times:
        return {
            "error": f"Invalid time slot '{time_slot}'. Use: {', '.join(valid_times)}"
        }

    statement = game_engine.scenario.witness_statements[suspect_name]
    is_murderer = suspect_name == game_engine.scenario.murderer

    # Critical time window for the murder (9:30-10:00 PM)
    critical_window = time_slot in ["21:30", "21:45"]

    if is_murderer and critical_window:
        # Murderer's alibi doesn't check out during critical time
        return {
            "suspect": suspect_name,
            "claimed_location": statement.location_during_murder,
            "time_checked": time_slot,
            "alibi_verified": False,
            "discrepancies": [
                f"No witnesses can confirm presence in {statement.location_during_murder}",
                f"Physical evidence places suspect near {game_engine.scenario.murder_location}",
                "Timeline inconsistent with claimed activities",
            ],
            "confidence": "HIGH - Alibi does not hold up to scrutiny",
            "recommendation": "⚠️  Priority suspect - significant inconsistencies detected",
        }

    elif is_murderer and not critical_window:
        # Murderer's alibi before/after might be partially true
        return {
            "suspect": suspect_name,
            "claimed_location": statement.location_during_murder,
            "time_checked": time_slot,
            "alibi_verified": "Partial",
            "discrepancies": ["Some minor timeline gaps noted"],
            "confidence": "MEDIUM - Cannot fully confirm movements",
            "recommendation": "Continue investigation - some inconsistencies present",
        }

    else:
        # Innocent suspects have verifiable alibis
        if critical_window:
            verification_note = f"Witness corroboration available for {statement.location_during_murder}"
        else:
            verification_note = "No contradictory evidence found"

        return {
            "suspect": suspect_name,
            "claimed_location": statement.location_during_murder,
            "time_checked": time_slot,
            "alibi_verified": True,
            "discrepancies": [],
            "confidence": "MEDIUM-HIGH - Alibi appears consistent",
            "recommendation": f"Alibi holds. {verification_note}.",
        }


def validate_solution(suspect: str, weapon: str, location: str) -> dict:
    """Tool for supervisor to check if the case is solved"""
    if not game_engine.scenario:
        return {"error": "No active investigation. Game scenario not initialized."}

    correct = (
        suspect == game_engine.scenario.murderer
        and weapon == game_engine.scenario.murder_weapon
        and location == game_engine.scenario.murder_location
    )

    return {
        "case_solved": correct,
        "correct_suspect": suspect == game_engine.scenario.murderer,
        "correct_weapon": weapon == game_engine.scenario.murder_weapon,
        "correct_location": location == game_engine.scenario.murder_location,
        "feedback": "Case solved! Excellent detective work."
        if correct
        else "Not quite right. Keep investigating with the help of your agents",
    }


def get_tool_list() -> str:
    """Return a list of all tools and their purposes in the current file."""
    # Get all global objects in the current file
    current_globals = globals()

    # Filter for functions defined in this file (not imported)
    functions = [
        (name, obj)
        for name, obj in current_globals.items()
        if inspect.isfunction(obj) and obj.__module__ == __name__
    ]

    tool_list = []
    for name, func in functions:
        if name == "get_tool_list":
            continue

        # Extract the docstring - docstrings are built such that the first line explains the purpose of the function, or is enough to get the purpose of the function.
        # Therefore we can ignore the rest of the docstring to keep it simple for the supervisor
        docstring = inspect.getdoc(func) or ""
        # Extract the first line (purpose)
        purpose = docstring.split("\n")[0].strip() if docstring else "No description"
        signature = str(inspect.signature(func)).replace(" -> str", "")
        tool_list.append(f"{name}{signature} - {purpose}")

    return "\n    ".join(tool_list)
