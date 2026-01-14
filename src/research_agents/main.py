"""Main entry point for the research agents system."""

import asyncio

from .agents.main_agent import MainAgent
from .agents.report_agent import ReportAgent
from .agents.research_agent import ResearchAgent
from .agents.validation_agent import ValidationAgent
from .core.database import get_database
from .core.orchestrator import Orchestrator


async def chat_loop(
    main_agent: MainAgent,
    research_agent: ResearchAgent,
    validation_agent: ValidationAgent,
    orchestrator: Orchestrator,
) -> None:
    """Run an interactive chat loop."""
    print("Research Agents - Interactive Chat")
    print("=" * 40)
    print(f"Session ID: {main_agent.session_id}")
    print("Commands:")
    print("  quit/exit  - End the session")
    print("  clear      - Clear conversation history")
    print("  cache      - Show cached research count")
    print("  clearcache - Clear research cache")
    print("  status     - Show validation status")
    print("  agents     - List available agents")
    print("  history    - Show execution history")
    print("  dbstats    - Show database statistics")
    print("  dbreports  - List reports from database")
    print()

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                main_agent.conversation_history.clear()
                print("Conversation history cleared.\n")
                continue

            if user_input.lower() == "cache":
                count = len(main_agent.research_cache)
                print(f"Research cache contains {count} item(s).\n")
                continue

            if user_input.lower() == "clearcache":
                main_agent.clear_research_cache()
                print("Research cache cleared.\n")
                continue

            if user_input.lower() == "status":
                status = main_agent.get_validation_status()
                print("Validation Status:")
                print(f"  Pending validation: {status['pending_validation']} findings")
                print(f"  Validated findings: {status['validated_findings']} findings")
                print(f"  Research cache:     {status['research_cache_items']} items")
                print()
                continue

            if user_input.lower() == "agents":
                print("Available agents:")
                for name, desc in orchestrator.get_agent_descriptions().items():
                    print(f"  - {name}: {desc}")
                print()
                continue

            if user_input.lower() == "history":
                history = orchestrator.get_execution_history()
                if history:
                    print("Execution history:")
                    for i, entry in enumerate(history, 1):
                        auto = " (auto-selected)" if entry.get("auto_selected") else ""
                        print(f"  {i}. [{entry['agent']}{auto}] {entry['task'][:50]}...")
                else:
                    print("No execution history yet.")
                print()
                continue

            if user_input.lower() == "dbstats":
                try:
                    db = get_database()
                    await db.connect()
                    stats = await db.get_session_stats(main_agent.session_id)
                    print("Database Statistics (this session):")
                    print(f"  Findings:  {stats['findings_count']}")
                    print(f"  Validated: {stats['validated_count']}")
                    print(f"  Reports:   {stats['reports_count']}")
                except Exception as e:
                    print(f"Database unavailable: {e}")
                print()
                continue

            if user_input.lower() == "dbreports":
                try:
                    db = get_database()
                    await db.connect()
                    reports = await db.get_all_reports(limit=10)
                    if reports:
                        print("Recent reports in database:")
                        for r in reports:
                            print(f"  [{r['id']}] {r['title'][:50]}... ({r['created_at']})")
                    else:
                        print("No reports in database yet.")
                except Exception as e:
                    print(f"Database unavailable: {e}")
                print()
                continue

            print("Thinking...", end="", flush=True)
            response = await main_agent.chat(user_input)
            print("\r" + " " * 20 + "\r", end="")  # Clear "Thinking..."
            print(f"Assistant: {response}\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        await research_agent.close()
        await validation_agent.close()
        # Close database connection
        try:
            db = get_database()
            await db.end_session(main_agent.session_id)
            await db.close()
        except Exception:
            pass


async def main() -> None:
    """Run the research agents system."""
    orchestrator = Orchestrator(use_llm_selection=True)

    # Register agents
    research_agent = ResearchAgent()
    orchestrator.register_agent(research_agent)

    validation_agent = ValidationAgent()
    orchestrator.register_agent(validation_agent)

    report_agent = ReportAgent()
    orchestrator.register_agent(report_agent)

    main_agent = MainAgent(orchestrator)
    orchestrator.register_agent(main_agent)

    print(f"Registered agents: {orchestrator.list_agents()}\n")

    await chat_loop(main_agent, research_agent, validation_agent, orchestrator)


def run() -> None:
    """Entry point for the CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
