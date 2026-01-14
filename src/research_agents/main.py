"""Main entry point for the research agents system."""

import asyncio
import sys
import warnings

# Suppress RuntimeError from asyncio subprocess cleanup at exit
# This is a known issue with Playwright's subprocess cleanup
warnings.filterwarnings(
    "ignore",
    message="Event loop is closed",
    category=RuntimeWarning,
)

# Store original hooks for non-suppressed exceptions
_original_excepthook = sys.excepthook
_original_unraisablehook = getattr(sys, "unraisablehook", None)


def _suppress_event_loop_errors(exc_type, exc_value, exc_tb):
    """Suppress 'Event loop is closed' errors during shutdown."""
    if exc_type is RuntimeError and "Event loop is closed" in str(exc_value):
        return  # Suppress
    _original_excepthook(exc_type, exc_value, exc_tb)


def _suppress_unraisable_errors(unraisable):
    """Suppress 'Event loop is closed' errors from __del__ methods."""
    if unraisable.exc_type is RuntimeError and "Event loop is closed" in str(unraisable.exc_value):
        return  # Suppress
    if _original_unraisablehook:
        _original_unraisablehook(unraisable)
    else:
        # Default behavior: print to stderr
        print(f"Exception ignored in: {unraisable.object}", file=sys.stderr)
        sys.__excepthook__(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)


sys.excepthook = _suppress_event_loop_errors
sys.unraisablehook = _suppress_unraisable_errors

from .agents.main_agent import MainAgent
from .agents.report_agent import ReportAgent
from .agents.research_agent import ResearchAgent
from .agents.validation_agent import ValidationAgent
from .core.config import get_settings
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
    print("  research <query> - Search, validate, and generate report (resets context)")
    print("  dig <query>      - Follow-up research within current context")
    print("  context          - Show current research context")
    print("  quit/exit        - End the session")
    print("  clear            - Clear conversation history")
    print("  cache            - Show cached research count")
    print("  clearcache       - Clear research cache and context")
    print("  status           - Show validation status")
    print("  agents           - List available agents")
    print("  history          - Show execution history")
    print("  dbstats          - Show database statistics")
    print("  dbreports        - List reports from database")
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

            if user_input.lower() == "context":
                ctx = main_agent.research_context
                if ctx.is_active():
                    print("Current Research Context:")
                    print(f"  Topic: {ctx.topic}")
                    print(f"  Original query: {ctx.original_query}")
                    print(f"  Started: {ctx.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Queries made: {len(ctx.queries_made)}")
                    if ctx.queries_made:
                        for i, q in enumerate(ctx.queries_made, 1):
                            print(f"    {i}. {q}")
                    print(f"  Sources explored: {len(ctx.sources_explored)}")
                    print(f"  Raw findings: {len(ctx.raw_findings)}")
                    print(f"  Validated findings: {len(ctx.validated_findings)}")
                    print(f"  Removed findings: {len(ctx.removed_findings)}")
                    if ctx.key_facts:
                        print(f"  Key facts ({len(ctx.key_facts)}):")
                        for fact in ctx.key_facts[:3]:
                            print(f"    - {fact[:80]}...")
                else:
                    print("No active research context. Use 'research <query>' to start.")
                print()
                continue

            if user_input.lower() == "research" or user_input.lower().startswith("research "):
                query = user_input[8:].strip() if len(user_input) > 8 else ""
                if query:
                    # Reset context for new research
                    main_agent.research_context.reset(new_topic=query, new_query=query)
                    main_agent.clear_research_cache()
                    print(f"Starting new research context: {query}")

                    settings = get_settings()
                    max_repeats = settings.max_repeats
                    all_validated = []
                    attempt = 0

                    while attempt <= max_repeats:
                        attempt += 1
                        attempt_label = f" (attempt {attempt}/{max_repeats + 1})" if attempt > 1 else ""

                        # Step 1: Research
                        print(f"Researching: {query}...{attempt_label}")
                        result = await research_agent.execute(query)
                        if result.get("status") != "completed":
                            print(f"Research failed: {result.get('error', 'Unknown error')}")
                            if attempt <= max_repeats:
                                print("Retrying...")
                                continue
                            print()
                            break

                        findings = result.get("findings", [])
                        main_agent.current_query = query
                        print(f"Found {len(findings)} results.")

                        # Track findings in context
                        main_agent.research_context.add_findings(findings)

                        if not findings:
                            if attempt <= max_repeats:
                                print("No results found. Retrying...")
                                continue
                            print("No results found.")
                            print()
                            break

                        # Step 2: Validate
                        print(f"Validating {len(findings)} findings...")
                        validation_result = await validation_agent.validate_findings(findings)
                        validated = validation_result.get("validated", [])
                        removed = validation_result.get("removed", [])
                        stats = validation_result.get("stats", {})

                        # Track validation in context
                        main_agent.research_context.add_validated(validated, removed)

                        print(f"Validation: {len(validated)} verified, {len(removed)} removed ({stats.get('validation_rate', 0):.0%} rate)")

                        if validated:
                            all_validated.extend(validated)
                            # We have validated results, exit retry loop
                            break
                        elif attempt <= max_repeats:
                            print(f"No findings could be verified. Retrying ({attempt}/{max_repeats + 1})...")
                        else:
                            print("No findings could be verified after all attempts.")

                    # Update main agent state with all validated findings
                    if all_validated:
                        main_agent.validated_findings.extend(all_validated)
                        main_agent.research_cache.append({
                            "source": "Validated Research",
                            "content": "\n".join(
                                f"- {f.get('title', 'Untitled')}: {f.get('snippet', '')}"
                                for f in all_validated
                            ),
                        })

                    if not all_validated:
                        print()
                        continue

                    # Step 3: Generate Report
                    print("Generating report...")
                    report_agent = orchestrator.get_agent("report")
                    if report_agent:
                        report_title = f"Research Report: {query}"
                        report_context = {
                            "title": report_title,
                            "information": main_agent.research_cache,
                            "instructions": "Focus on the validated findings and their confidence levels.",
                        }
                        report_result = await report_agent.execute(report_title, report_context)

                        if report_result.get("status") == "completed":
                            report_content = report_result.get("report", "")
                            print("\n" + "=" * 50)
                            print(report_content)
                            print("=" * 50)
                        else:
                            print("Report generation failed.")
                    else:
                        # No report agent, just display validated findings
                        print("\n" + "=" * 50)
                        print(f"Research Results: {query}")
                        print("=" * 50)
                        for f in all_validated:
                            conf = f.get("validation", {}).get("confidence", 0)
                            print(f"\n[{conf:.0%}] {f.get('title', 'Untitled')}")
                            print(f"    {f.get('snippet', '')[:200]}")
                        print("=" * 50)
                else:
                    print("Please provide a search query. Usage: research <query>")
                print()
                continue

            # Dig command for follow-up research within context
            if user_input.lower() == "dig" or user_input.lower().startswith("dig "):
                follow_up_query = user_input[3:].strip() if len(user_input) > 3 else ""
                if not follow_up_query:
                    print("Please provide a follow-up query. Usage: dig <query>")
                    print()
                    continue

                if not main_agent.research_context.is_active():
                    print("No active research context. Use 'research <query>' first to establish context.")
                    print()
                    continue

                # Record follow-up query
                main_agent.research_context.add_follow_up(follow_up_query)

                # Get context for agents
                ctx = main_agent.research_context.get_context_for_agent()
                print(f"Follow-up research in context of: {ctx['topic']}")
                print(f"  Previous queries: {len(ctx['previous_queries'])}")
                print(f"  Sources already explored: {len(ctx['sources_already_explored'])}")

                # Research with context
                print(f"Researching: {follow_up_query}...")
                research_context = {
                    "depth": "normal",
                    "research_context": ctx,
                    "exclude_urls": ctx["sources_already_explored"],
                }
                result = await research_agent.execute(follow_up_query, research_context)

                if result.get("status") != "completed":
                    print(f"Research failed: {result.get('error', 'Unknown error')}")
                    print()
                    continue

                findings = result.get("findings", [])
                print(f"Found {len(findings)} new results.")

                # Track findings in context
                main_agent.research_context.add_findings(findings)

                if not findings:
                    print("No new results found.")
                    print()
                    continue

                # Validate with context
                print(f"Validating {len(findings)} findings...")
                validation_context = {
                    "research_context": ctx,
                    "min_confidence": 0.4,
                }
                validation_result = await validation_agent.validate_findings(findings, **validation_context)
                validated = validation_result.get("validated", [])
                removed = validation_result.get("removed", [])
                stats = validation_result.get("stats", {})

                # Track validation in context
                main_agent.research_context.add_validated(validated, removed)

                print(f"Validation: {len(validated)} verified, {len(removed)} removed ({stats.get('validation_rate', 0):.0%} rate)")

                if validated:
                    # Add to main agent state
                    main_agent.validated_findings.extend(validated)
                    main_agent.research_cache.append({
                        "source": f"Follow-up: {follow_up_query}",
                        "content": "\n".join(
                            f"- {f.get('title', 'Untitled')}: {f.get('snippet', '')}"
                            for f in validated
                        ),
                    })

                    # Show results
                    print("\nNew validated findings:")
                    for f in validated[:5]:
                        conf = f.get("validation", {}).get("confidence", 0)
                        print(f"  [{conf:.0%}] {f.get('title', 'Untitled')[:60]}")
                    if len(validated) > 5:
                        print(f"  ... and {len(validated) - 5} more")

                    # Update context summary
                    ctx_summary = main_agent.research_context
                    print(f"\nContext totals: {len(ctx_summary.validated_findings)} validated, {len(ctx_summary.sources_explored)} sources")
                else:
                    print("No findings could be verified.")

                print()
                continue

            if user_input.lower() == "validate":
                if not main_agent.pending_validation:
                    print("No pending findings to validate. Use 'research <query>' first.")
                else:
                    print(f"Validating {len(main_agent.pending_validation)} findings...")
                    result = await validation_agent.validate_findings(main_agent.pending_validation)
                    validated = result.get("validated", [])
                    removed = result.get("removed", [])
                    stats = result.get("stats", {})

                    # Update main agent state
                    main_agent.validated_findings.extend(validated)
                    if validated:
                        main_agent.research_cache.append({
                            "source": "Validated Research",
                            "content": "\n".join(
                                f"- {f.get('title', 'Untitled')}: {f.get('snippet', '')}"
                                for f in validated
                            ),
                        })
                    main_agent.pending_validation.clear()

                    print(f"Validation complete:")
                    print(f"  Verified: {len(validated)} findings")
                    print(f"  Removed:  {len(removed)} findings")
                    print(f"  Rate:     {stats.get('validation_rate', 0):.0%}")

                    if validated:
                        print("\nVerified findings:")
                        for f in validated[:3]:
                            conf = f.get("validation", {}).get("confidence", 0)
                            print(f"  [{conf:.0%}] {f.get('title', 'Untitled')[:50]}")
                        if len(validated) > 3:
                            print(f"  ... and {len(validated) - 3} more")
                print()
                continue

            print("Thinking...", end="", flush=True)
            response = await main_agent.chat(user_input)
            print("\r" + " " * 20 + "\r", end="")  # Clear "Thinking..."
            print(f"Assistant: {response}\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        # Close agents gracefully
        try:
            await research_agent.close()
        except Exception:
            pass

        try:
            await validation_agent.close()
        except Exception:
            pass

        # Close database connection
        try:
            db = get_database()
            await db.end_session(main_agent.session_id)
            await db.close()
        except Exception:
            pass

        # Give async tasks time to complete cleanup
        try:
            await asyncio.sleep(0.1)
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
    # Use a custom exception handler to suppress event loop closed errors
    # that occur during subprocess cleanup
    def exception_handler(loop, context):
        exception = context.get("exception")
        if isinstance(exception, RuntimeError) and "Event loop is closed" in str(exception):
            return  # Suppress this specific error
        # For other exceptions, use default handling
        loop.default_exception_handler(context)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(exception_handler)
        loop.run_until_complete(main())
    finally:
        try:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Give tasks a chance to clean up
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            try:
                loop.close()
            except Exception:
                pass


if __name__ == "__main__":
    run()
