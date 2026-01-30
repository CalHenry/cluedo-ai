import asyncio
from typing import cast

import logfire
from pydantic_ai import usage

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

    # Create a UsageTracker to accumulate token usage across all runs
    usage_tracker = usage.RunUsage()

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
            - submit final answer (only submit if you validated that your answer is correct)
            """,
            deps=SupervisorContext(gathered_info=research_findings_text),
        )

        # Add this run's usage to the tracker
        usage_tracker += supervisor_response.usage()
        print(f"Supervisor tokens - {supervisor_response.usage()}")

        decision = cast(SupervisorDecision, supervisor_response.output)
        print(f"Supervisor decision: {decision.action}")
        print(f"Instruction: {decision.instruction}")

        if decision.action == "delegate_to_researcher":
            print("🔵")
            research_findings = await research_agent.run(
                f"""TASK: {decision.instruction}
                Use the appropriate tool once, report the result, then stop.
                Do not investigate further."""
            )

            # Add researcher's usage to the tracker
            usage_tracker += research_findings.usage()
            print(f"Researcher tokens - {research_findings.usage()}")

            research_findings_text = str(research_findings.output)
            supervisor_memory.append(
                f"[RESEARCH] {decision.instruction}\nFindings: {research_findings_text}"
            )

        elif decision.action == "submit_answer":
            print("\n" + "=" * 80)
            print("SUPERVISOR IS SUBMITTING SOLUTION")
            print("=" * 80)
            print("\n" + "-" * 80)
            print("Tokens metadata")
            print(f"\nTotal Token Usage: {usage_tracker}")
            print(f"  Request tokens: {usage_tracker.input_tokens}")
            print(f"  Response tokens: {usage_tracker.output_tokens}")
            print(f"  Total tokens: {usage_tracker.total_tokens}")
            print("=" * 80)

            final_answer = decision.instruction
            return {
                "solution": final_answer,
                "evidence": supervisor_memory,
                "attempts_used": attempts,
                "token_usage": usage_tracker,
            }

    # Max attempts reached
    print("\n" + "-" * 80)
    print("Tokens metadata")
    print(f"\nTotal Token Usage (incomplete): {usage_tracker}")
    print(f"  Request tokens: {usage_tracker.input_tokens}")
    print(f"  Response tokens: {usage_tracker.output_tokens}")
    print(f"  Total tokens: {usage_tracker.total_tokens}")

    return {
        "solution": "Investigation incomplete - max attempts reached",
        "evidence": supervisor_memory,
        "attempts_used": attempts,
        "token_usage": usage_tracker,
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
