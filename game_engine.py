import random
from typing import Literal

from pydantic import BaseModel, Field

RoomType = Literal[
    "Study", "Library", "Kitchen", "Conservatory", "Billiard Room", "Lounge"
]
WeaponType = Literal["Candlestick", "Knife", "Revolver", "Rope", "Lead Pipe", "Wrench"]
SuspectType = Literal[
    "Miss Scarlet",
    "Colonel Mustard",
    "Mrs White",
    "Mr Green",
    "Mrs Peacock",
    "Professor Plum",
]


class CrimeSceneEvidence(BaseModel):
    evidence_id: str
    item_name: str
    location: str
    description: str


class WitnessStatement(BaseModel):
    witness_name: str
    alibi: str
    testimony: str
    location_during_murder: str
    time_of_statement: str


class ForensicEvidence(BaseModel):
    evidence_id: str
    item_name: str
    analysis_type: str
    findings: str
    significance: str
    related_evidence_ids: list[str] = Field(default_factory=list)


class GameScenario(BaseModel):
    """The solution to the murder mystery"""

    murderer: SuspectType
    murder_weapon: WeaponType
    murder_location: RoomType
    murder_time: str  # e.g., "9:30 PM"

    # Supporting details
    crime_scene_evidence: dict[str, CrimeSceneEvidence]  # room_name -> evidence
    witness_statements: dict[str, WitnessStatement]  # witness_name -> statement
    forensic_evidence: dict[str, ForensicEvidence]  # evidence_id -> analysis
    red_herrings: list[str]  # misleading clues


class CluedoGameEngine:
    """Generates and manages the murder mystery scenario"""

    ROOMS: list[RoomType] = [
        "Study",
        "Library",
        "Kitchen",
        "Conservatory",
        "Billiard Room",
        "Lounge",
    ]
    WEAPONS: list[WeaponType] = [
        "Candlestick",
        "Knife",
        "Revolver",
        "Rope",
        "Lead Pipe",
        "Wrench",
    ]
    SUSPECTS: list[SuspectType] = [
        "Miss Scarlet",
        "Colonel Mustard",
        "Mrs White",
        "Mr Green",
        "Mrs Peacock",
        "Professor Plum",
    ]

    # Character backgrounds for witness generation
    SUSPECT_DETAILS = {
        "Miss Scarlet": {
            "occupation": "Actress",
            "relationship": "Former business partner",
        },
        "Colonel Mustard": {
            "occupation": "Retired Military",
            "relationship": "Old war friend",
        },
        "Mrs White": {"occupation": "Housekeeper", "relationship": "Employee"},
        "Mr Green": {"occupation": "Businessman", "relationship": "Rival investor"},
        "Mrs Peacock": {"occupation": "Socialite", "relationship": "Neighbor"},
        "Professor Plum": {
            "occupation": "Archaeologist",
            "relationship": "University colleague",
        },
    }

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.scenario: GameScenario | None = None

    def generate_scenario(self) -> GameScenario:
        """Generate a complete murder mystery scenario"""
        murderer = random.choice(self.SUSPECTS)
        weapon = random.choice(self.WEAPONS)
        location = random.choice(self.ROOMS)
        murder_time = random.choice(
            ["9:00 PM", "9:15 PM", "9:30 PM", "9:45 PM", "10:00 PM"]
        )

        # Generate crime scene evidence for each room
        crime_scene_evidence = self._generate_crime_scene_evidence(
            location, weapon, murderer
        )

        # Generate witness statements for each suspect
        witness_statements = self._generate_witness_statements(
            murderer, location, murder_time
        )

        # Generate forensic evidence
        forensic_evidence = self._generate_forensic_evidence(weapon, location, murderer)

        # Generate red herrings
        red_herrings = self._generate_red_herrings(murderer, weapon, location)

        self.scenario = GameScenario(
            murderer=murderer,
            murder_weapon=weapon,
            murder_location=location,
            murder_time=murder_time,
            crime_scene_evidence=crime_scene_evidence,
            witness_statements=witness_statements,
            forensic_evidence=forensic_evidence,
            red_herrings=red_herrings,
        )

        return self.scenario

    def _generate_crime_scene_evidence(
        self, murder_location: RoomType, weapon: WeaponType, murderer: SuspectType
    ) -> dict[str, CrimeSceneEvidence]:
        """Generate evidence found at each crime scene"""
        evidence = {}

        # Evidence in the actual murder room
        evidence[murder_location] = CrimeSceneEvidence(
            evidence_id=f"CSE_{murder_location.replace(' ', '_').upper()}_001",
            item_name=weapon,
            location=murder_location,
            description=f"The {weapon.lower()} was found near the victim's body. "
            f"Shows signs of recent use. Forensic team has collected it for analysis. "
            f"Blood spatters suggest the attack occurred here. "
            f"Partial footprint visible near the doorway.",
        )

        # Evidence in other rooms (potential red herrings or supporting clues)
        for room in self.ROOMS:
            if room != murder_location:
                # Some rooms have evidence, some don't
                if random.random() < 0.6:
                    item = random.choice(
                        [
                            "broken glass",
                            "cigarette butt",
                            "torn fabric",
                            "coffee cup",
                            "dropped glove",
                        ]
                    )
                    evidence[room] = CrimeSceneEvidence(
                        evidence_id=f"CSE_{room.replace(' ', '_').upper()}_001",
                        item_name=item,
                        location=room,
                        description=f"A {item} was found in the {room}. "
                        f"May or may not be related to the crime. "
                        f"Room shows signs of recent activity. "
                        f"No obvious signs of struggle.",
                    )

        return evidence

    def _generate_witness_statements(
        self, murderer: SuspectType, murder_location: RoomType, murder_time: str
    ) -> dict[str, WitnessStatement]:
        """Generate witness statements for all suspects"""
        statements = {}

        # Possible locations during the murder
        alibi_locations = ["Dining Room", "Garden", "Hallway", "Bedroom", "Bathroom"]

        for suspect in self.SUSPECTS:
            if suspect == murderer:
                # Murderer has a false alibi
                false_location = random.choice([loc for loc in alibi_locations])
                statements[suspect] = WitnessStatement(
                    witness_name=suspect,
                    alibi=f"I was in the {false_location} reading a book at {murder_time}.",
                    testimony="I heard nothing unusual. I was completely absorbed in my reading."
                    "I didn't see Dr. Black at all that evening.",
                    location_during_murder=false_location,
                    time_of_statement="10:30 PM",
                )
            else:
                # Innocent suspects have various alibis
                alibi_location = random.choice(alibi_locations)

                # Some witnesses saw/heard something useful
                if random.random() < 0.5:
                    testimony = random.choice(
                        [
                            f"I heard raised voices coming from the {murder_location} around {murder_time}.",
                            f"I saw someone leaving the {murder_location} in a hurry around {murder_time}.",
                            f"I noticed the {murder_location} door was closed, which was unusual.",
                            f"I heard a loud noise from the direction of the {murder_location}.",
                        ]
                    )
                else:
                    testimony = "I didn't notice anything unusual that evening."

                statements[suspect] = WitnessStatement(
                    witness_name=suspect,
                    alibi=f"I was in the {alibi_location} at {murder_time}.",
                    testimony=testimony,
                    location_during_murder=alibi_location,
                    time_of_statement="10:30 PM",
                )

        return statements

    def _generate_forensic_evidence(
        self, weapon: WeaponType, location: RoomType, murderer: SuspectType
    ) -> dict[str, ForensicEvidence]:
        """Generate detailed forensic analysis"""
        evidence = {}

        # Weapon analysis
        evidence["FOR_WEAPON_001"] = ForensicEvidence(
            evidence_id="FOR_WEAPON_001",
            item_name=weapon,
            analysis_type="Fingerprint and DNA Analysis",
            findings=f"Partial fingerprints recovered from the {weapon.lower()}. "
            f"DNA traces found on the handle. "
            f"Blood type matches victim (Type O+). "
            f"Fingerprint pattern suggests adult grip.",
            significance="Critical evidence linking the murderer to the weapon.",
            related_evidence_ids=["FOR_FIBER_001"],
        )

        # Fiber analysis
        evidence["FOR_FIBER_001"] = ForensicEvidence(
            evidence_id="FOR_FIBER_001",
            item_name="Fabric fibers",
            analysis_type="Textile Analysis",
            findings="Cotton fibers found on victim's clothing. "
            "Color analysis suggests origin from evening wear. "
            "Microscopic examination shows recent transfer.",
            significance="May help identify clothing worn by perpetrator.",
            related_evidence_ids=["FOR_WEAPON_001"],
        )

        # Blood spatter analysis
        evidence["FOR_BLOOD_001"] = ForensicEvidence(
            evidence_id="FOR_BLOOD_001",
            item_name="Blood spatter pattern",
            analysis_type="Blood Spatter Analysis",
            findings=f"Cast-off pattern indicates violent struggle. "
            f"Spatter distribution confirms attack occurred in {location}. "
            f"Trajectory analysis suggests attacker was right-handed.",
            significance="Confirms location and nature of attack.",
            related_evidence_ids=["FOR_WEAPON_001"],
        )

        # Footprint analysis
        evidence["FOR_PRINT_001"] = ForensicEvidence(
            evidence_id="FOR_PRINT_001",
            item_name="Footprint",
            analysis_type="Shoe Print Analysis",
            findings=f"Partial shoe print found near exit of {location}. "
            f"Size suggests adult footwear. "
            f"Tread pattern consistent with formal dress shoes.",
            significance="May help identify suspect based on shoe size and type.",
            related_evidence_ids=[],
        )

        return evidence

    def _generate_red_herrings(
        self, murderer: SuspectType, weapon: WeaponType, location: RoomType
    ) -> list[str]:
        """Generate misleading clues"""
        herrings = [
            f"A {random.choice([w for w in self.WEAPONS if w != weapon])} was found in the hallway.",
            f"{random.choice([s for s in self.SUSPECTS if s != murderer])} was seen arguing with Dr. Black earlier.",
            f"Strange noises were reported from the {random.choice([r for r in self.ROOMS if r != location])} that evening.",
            "An unidentified person was seen leaving the mansion around midnight.",
            "Dr. Black had recently changed his will, leaving everything to charity.",
        ]

        return random.sample(herrings, 3)
