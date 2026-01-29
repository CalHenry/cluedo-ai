import asyncio
from typing import cast

import logfire

from agents import (
    SupervisorContext,
    SupervisorDecision,
    research_agent,
    supervisor_agent,
)

logfire.configure()
logfire.instrument_pydantic_ai()


async def run_investigation(user_query: str):
    attempts = 0
    max_attempts = 15
    supervisor_memory = []
    research_findings_text = ""

    while attempts < max_attempts:
        attempts += 1
        print(f"\n--- Attempt {attempts}/{max_attempts} ---")
        # supervisor
        supervisor_response = await supervisor_agent.run(
            f"""Current evidence collected: {supervisor_memory}

            What is the next single step ?
            - request researcher to use a specific tool
            - ask processor to analyse current evidence
            - use validation tool to test a theory
            - submit final answer (only submit if you validated that your answser is correct)
            """,
            deps=SupervisorContext(gathered_info=research_findings_text),
        )

        decision = cast(
            SupervisorDecision, supervisor_response.output
        )  # Should be SupervisorDecision object. 'cast' is for the type checker

        print(f"Supervisor decision: {decision.action}")
        print(f"Instruction: {decision.instruction}")

        if decision.action == "delegate_to_researcher":
            print("🔵")
            # Pass specific instruction to researcher
            research_findings = await research_agent.run(
                f"""TASK: {decision.instruction}

                Use the appropriate tool once, report the result, then stop.
                Do not investigate further."""
            )
            research_findings_text = str(research_findings.output)
            supervisor_memory.append(
                f"[RESEARCH] {decision.instruction}\nFindings: {research_findings_text}"
            )

        elif decision.action == "submit_answer":
            # Supervisor is ready to submit solution
            print("\n" + "=" * 80)
            print("SUPERVISOR IS SUBMITTING SOLUTION")
            print("=" * 80)

            # You can validate here if you have a validation tool
            final_answer = decision.instruction

            return {
                "solution": final_answer,
                "evidence": supervisor_memory,
                "attempts_used": attempts,
            }

    # Max attempts reached
    return {
        "solution": "Investigation incomplete - max attempts reached",
        "evidence": supervisor_memory,
        "attempts_used": attempts,
    }


async def main():
    # Test the multi-agent workflow
    result = await run_investigation(
        "Investigate the crime of Dr.Black. Ask your agents do perform research and processing tasks. You should validate your hypothesis using the tool validate_solution() before writing the final report."
    )

    print("\n" + "=" * 80)
    print("FINAL INVESTIGATION REPORT")
    print("=" * 80)
    print(f"\nSolution:\n{result['solution']}")
    print(f"\nAttempts used: {result['attempts_used']}")
    print("\nEvidence trail:")
    for evidence in result["evidence"]:
        print(f"  {evidence}\n")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
