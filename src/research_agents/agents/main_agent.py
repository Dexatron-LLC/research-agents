"""Main coordinating agent that manages other agents."""

import json
import uuid
from typing import Any

import httpx

from ..core.base_agent import BaseAgent
from ..core.config import Settings, get_settings
from ..core.database import DatabaseService, get_database
from ..core.orchestrator import Orchestrator


class MainAgent(BaseAgent):
    """Main agent that coordinates other agents to accomplish tasks."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        settings: Settings | None = None,
        database: DatabaseService | None = None,
    ):
        super().__init__(name="main")
        self.orchestrator = orchestrator
        self.settings = settings or get_settings()
        self.db = database or get_database()
        self.session_id = str(uuid.uuid4())
        self.conversation_history: list[dict[str, str]] = []
        self.research_cache: list[dict[str, Any]] = []
        self.validated_findings: list[dict[str, Any]] = []
        self.pending_validation: list[dict[str, Any]] = []
        self.current_query: str = ""
        self.current_report: str | None = None
        self.current_report_title: str | None = None
        self._db_initialized = False

    async def _ensure_db_initialized(self) -> None:
        """Ensure database is connected and initialized."""
        if self._db_initialized:
            return
        try:
            await self.db.connect()
            await self.db.initialize_schema()
            await self.db.create_session(self.session_id)
            self._db_initialized = True
        except Exception as e:
            # Log but continue without DB if unavailable
            print(f"Warning: Database unavailable ({e}). Running without persistence.")
            self._db_initialized = True  # Don't retry

    def _get_current_state_info(self) -> str:
        """Get current state information for the system prompt."""
        parts = []

        if self.current_report:
            parts.append(f"- REPORT AVAILABLE: '{self.current_report_title}' (ready to save)")
        else:
            parts.append("- No report currently available")

        if self.pending_validation:
            parts.append(f"- Pending validation: {len(self.pending_validation)} findings")

        if self.validated_findings:
            parts.append(f"- Validated findings: {len(self.validated_findings)}")

        if self.research_cache:
            parts.append(f"- Research cache: {len(self.research_cache)} items")

        return "\n".join(parts) if parts else "No active research session"

    def _get_coordinator_system_prompt(self) -> str:
        """Build system prompt combining YAML definition with available agents and tools."""
        # Get base prompt from YAML definition
        base_prompt = super().get_system_prompt()

        # Build list of available agents
        agents_info = []
        for name in self.orchestrator.list_agents():
            agent = self.orchestrator.get_agent(name)
            if agent and agent.name != "main":
                agents_info.append(f"- {agent.name}: {agent.description}")

        agents_list = "\n".join(agents_info) if agents_info else "No sub-agents available"

        # Build current state info
        state_info = self._get_current_state_info()

        tool_instructions = f"""
## Available Agents
{agents_list}

## Current State
{state_info}

## Tool Usage
To use an agent, respond with a JSON tool call. Available tool calls:

1. Research (search the web):
{{"tool": "research", "query": "the search query"}}

2. Validate research findings (fact-check against trusted sources):
{{"tool": "validation", "action": "validate"}}

3. Create a report from validated research:
{{"tool": "report", "action": "create", "title": "Report Title", "instructions": "optional formatting instructions"}}

4. Save the current report (ONLY use when a report exists - check Current State above):
{{"tool": "report", "action": "save", "filename": "optional_filename"}}

5. List saved reports:
{{"tool": "report", "action": "list"}}

6. Load a saved report:
{{"tool": "report", "action": "load", "filename": "report_filename"}}

## Workflow
1. When a user asks for research, use the research tool. Results are cached for validation.
2. After research, validate the findings using the validation tool to fact-check against trusted sources.
3. The validation agent will remove unverifiable claims and may request replacement research.
4. A report is AUTOMATICALLY created after successful validation - no need to call the report tool manually.
5. When user asks to save the report and a report exists (see Current State), use the save tool immediately.
6. For general conversation, respond normally without tool calls.

IMPORTANT: When the user asks to "save this report" or "save the report" and Current State shows a report exists,
you MUST respond with the save tool call: {{"tool": "report", "action": "save"}}

Always validate research before creating reports to ensure accuracy. All data is stored in the database for persistence."""

        return f"{base_prompt}\n{tool_instructions}"

    async def _call_llm(self, user_message: str) -> str:
        """Call the LLM API."""
        if not self.settings.api_key:
            return "Error: ANTHROPIC_API_KEY not configured. Please set it in .env or environment."

        await self._ensure_db_initialized()
        self.conversation_history.append({"role": "user", "content": user_message})

        # Save to database
        try:
            await self.db.save_message(self.session_id, "user", user_message)
        except Exception:
            pass  # Continue without DB

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.settings.get_api_url("/v1/messages"),
                headers={
                    "x-api-key": self.settings.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.settings.default_model,
                    "max_tokens": self.settings.max_tokens,
                    "system": self._get_coordinator_system_prompt(),
                    "messages": self.conversation_history,
                },
                timeout=self.settings.request_timeout,
            )

            if response.status_code != 200:
                return f"Error: API returned status {response.status_code}: {response.text}"

            data = response.json()
            assistant_message = data["content"][0]["text"]
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            # Save to database
            try:
                await self.db.save_message(self.session_id, "assistant", assistant_message)
            except Exception:
                pass  # Continue without DB

            return assistant_message

    async def _handle_validation(self) -> dict[str, Any]:
        """Handle validation of pending research findings."""
        validation_agent = self.orchestrator.get_agent("validation")
        if not validation_agent:
            return {"error": "Validation agent not available"}

        if not self.pending_validation:
            return {"error": "No research findings to validate. Please perform research first."}

        # Validate all pending findings
        context = {
            "findings": self.pending_validation,
            "min_confidence": 0.5,
        }
        result = await validation_agent.execute("validate findings", context)

        if result.get("status") != "completed":
            return result

        # Process validation results
        validated = result.get("validated", [])
        removed = result.get("removed", [])
        replacement_needed = result.get("replacement_needed", [])

        # Update caches
        self.validated_findings.extend(validated)
        self.pending_validation.clear()

        # Update research cache with validated findings
        if validated:
            self.research_cache.append({
                "source": "Validated Research",
                "content": self._format_validated_findings(validated),
                "metadata": {
                    "validated_count": len(validated),
                    "removed_count": len(removed),
                }
            })

        # Save validation results to database
        try:
            # Get finding IDs from database
            db_findings = await self.db.get_findings_by_session(self.session_id)
            title_to_id = {f["title"]: f["id"] for f in db_findings}

            for finding in validated:
                validation_data = finding.get("validation", {})
                finding_id = title_to_id.get(finding.get("title"))
                if finding_id:
                    await self.db.save_validation(
                        finding_id=finding_id,
                        session_id=self.session_id,
                        status=validation_data.get("status", "VERIFIED"),
                        confidence=validation_data.get("confidence", 0.5),
                        reason=validation_data.get("reason", ""),
                        sources=validation_data.get("sources", []),
                    )

            for finding in removed:
                validation_data = finding.get("validation", {})
                finding_id = title_to_id.get(finding.get("title"))
                if finding_id:
                    await self.db.save_validation(
                        finding_id=finding_id,
                        session_id=self.session_id,
                        status=validation_data.get("status", "UNVERIFIED"),
                        confidence=validation_data.get("confidence", 0.0),
                        reason=validation_data.get("reason", ""),
                        sources=validation_data.get("sources", []),
                    )
        except Exception:
            pass  # Continue without DB

        # Auto-create report after validation
        report_result = None
        if validated:
            report_result = await self._auto_create_report(validated)

        return {
            "status": "completed",
            "validated": validated,
            "removed": removed,
            "replacement_needed": replacement_needed,
            "stats": result.get("stats", {}),
            "report": report_result,
        }

    async def _auto_create_report(self, validated_findings: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Automatically create a report from validated findings."""
        report_agent = self.orchestrator.get_agent("report")
        if not report_agent:
            return None

        # Build title from query
        title = f"Research Report: {self.current_query}" if self.current_query else "Research Report"

        # Prepare information for report
        context = {
            "title": title,
            "information": self.research_cache,
            "instructions": "Focus on the validated findings and their confidence levels.",
        }

        result = await report_agent.execute(title, context)

        if result.get("status") == "completed":
            # Store report in main agent's state for later saving
            report_content = result.get("report", "")
            self.current_report = report_content
            self.current_report_title = title

            # Save report to database
            try:
                await self.db.save_report(
                    session_id=self.session_id,
                    title=title,
                    content=report_content,
                    findings_count=len(validated_findings),
                )
            except Exception:
                pass  # Continue without DB

        return result

    async def _handle_replacement_research(
        self,
        replacement_needed: list[dict[str, Any]],
        original_query: str,
    ) -> dict[str, Any]:
        """Request replacement research for removed findings."""
        research_agent = self.orchestrator.get_agent("research")
        if not research_agent:
            return {"error": "Research agent not available"}

        # Build a query that excludes the removed claims
        removed_topics = [r.get("original_claim", "")[:50] for r in replacement_needed]
        exclusion_note = f"Exclude information about: {', '.join(removed_topics)}"

        new_query = f"{original_query} (Need alternative sources. {exclusion_note})"
        result = await research_agent.execute(new_query)

        if result.get("status") == "completed":
            # Add new findings to pending validation
            findings = result.get("findings", [])
            self.pending_validation.extend(findings)

        return result

    async def _handle_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call from the LLM."""
        tool_name = tool_call.get("tool")

        if tool_name == "research":
            query = tool_call.get("query", "")
            self.current_query = query
            agent = self.orchestrator.get_agent("research")
            if agent:
                result = await agent.execute(query)
                if result.get("status") == "completed":
                    findings = result.get("findings", [])
                    # Store findings for validation instead of directly in cache
                    self.pending_validation.extend(findings)
                    result["pending_validation"] = True
                    result["original_query"] = query

                    # Save findings to database
                    try:
                        for finding in findings:
                            await self.db.save_finding(
                                session_id=self.session_id,
                                query=query,
                                title=finding.get("title", ""),
                                snippet=finding.get("snippet", ""),
                                url=finding.get("url", ""),
                                source=finding.get("source", "web_search"),
                            )
                    except Exception:
                        pass  # Continue without DB

                return result
            return {"error": "Research agent not available"}

        elif tool_name == "validation":
            action = tool_call.get("action", "validate")
            if action == "validate":
                return await self._handle_validation()
            return {"error": f"Unknown validation action: {action}"}

        elif tool_name == "report":
            action = tool_call.get("action", "create")
            agent = self.orchestrator.get_agent("report")
            if not agent:
                return {"error": "Report agent not available"}

            if action == "create":
                if not self.research_cache:
                    if self.pending_validation:
                        return {"error": "Research findings need validation before creating a report. Use the validation tool first."}
                    return {"error": "No research data available. Please perform some research first."}
                context = {
                    "title": tool_call.get("title", "Research Report"),
                    "information": self.research_cache,
                    "instructions": tool_call.get("instructions"),
                }
                return await agent.execute(tool_call.get("title", "Research Report"), context)

            elif action == "save":
                # Ensure report agent has the current report
                if self.current_report and not agent.current_report:
                    agent.current_report = self.current_report
                    agent.current_title = self.current_report_title

                if not agent.current_report and not self.current_report:
                    return {"error": "No report available to save. Please generate a report first."}

                context = {"save": True, "filename": tool_call.get("filename")}
                result = await agent.execute("save", context)

                # Clear the current report after saving
                if result.get("status") == "saved":
                    self.current_report = None
                    self.current_report_title = None

                return result

            elif action == "list":
                context = {"list": True}
                return await agent.execute("list", context)

            elif action == "load":
                context = {"load": tool_call.get("filename", "")}
                return await agent.execute("load", context)

            return {"error": f"Unknown report action: {action}"}

        agent = self.orchestrator.get_agent(tool_name)
        if agent:
            return await agent.execute(tool_call.get("query", ""))
        return {"error": f"Unknown agent: {tool_name}"}

    def _parse_tool_call(self, response: str) -> dict[str, Any] | None:
        """Try to parse a tool call from the response."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                potential_json = response[start:end]
                parsed = json.loads(potential_json)
                if "tool" in parsed:
                    return parsed
        except json.JSONDecodeError:
            pass
        return None

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Process a user message and return a response.

        Args:
            task: The user's message
            context: Optional context

        Returns:
            Dictionary containing the response
        """
        response = await self._call_llm(task)

        tool_call = self._parse_tool_call(response)
        if tool_call:
            tool_result = await self._handle_tool_call(tool_call)
            tool_name = tool_call.get("tool")

            if tool_name == "research" and tool_result.get("status") == "completed":
                findings_summary = self._format_findings(tool_result.get("findings", []))
                followup = await self._call_llm(
                    f"Research results for '{tool_call.get('query', '')}': {findings_summary}\n\n"
                    f"Found {len(tool_result.get('findings', []))} results pending validation. "
                    "Let the user know we should validate these findings against trusted sources before creating a report."
                )
                return {
                    "response": followup,
                    "tool_used": tool_name,
                    "findings": tool_result.get("findings", []),
                    "pending_validation": True,
                }

            elif tool_name == "validation" and tool_result.get("status") == "completed":
                stats = tool_result.get("stats", {})
                removed = tool_result.get("removed", [])
                validated = tool_result.get("validated", [])
                replacement_needed = tool_result.get("replacement_needed", [])
                auto_report = tool_result.get("report")

                summary = (
                    f"Validation complete:\n"
                    f"- Validated: {stats.get('validated_count', 0)} findings\n"
                    f"- Removed: {stats.get('removed_count', 0)} findings\n"
                    f"- Validation rate: {stats.get('validation_rate', 0):.0%}"
                )

                if removed:
                    removed_items = "\n".join(
                        f"  - {r.get('validation', {}).get('claim', 'Unknown')[:60]}... "
                        f"(Reason: {r.get('validation', {}).get('reason', 'Unknown')})"
                        for r in removed[:3]
                    )
                    summary += f"\n\nRemoved findings:\n{removed_items}"

                if replacement_needed:
                    summary += f"\n\n{len(replacement_needed)} items need replacement research."

                # Include auto-generated report info
                if auto_report and auto_report.get("status") == "completed":
                    report_preview = auto_report.get("report", "")[:300]
                    summary += f"\n\nA report has been automatically generated from the validated findings.\nPreview:\n{report_preview}..."

                followup = await self._call_llm(
                    f"{summary}\n\nInform the user of the validation results. "
                    "A report was automatically created from the validated findings and saved to the database. "
                    "If items were removed, ask if they want replacement research."
                )

                result = {
                    "response": followup,
                    "tool_used": "validation",
                    "validated": validated,
                    "removed": removed,
                    "replacement_needed": replacement_needed,
                    "stats": stats,
                }
                if auto_report:
                    result["report"] = auto_report.get("report")
                return result

            elif tool_name == "report":
                action = tool_call.get("action", "create")

                if action == "create" and tool_result.get("status") == "completed":
                    report_preview = tool_result.get("report", "")[:500]
                    followup = await self._call_llm(
                        f"Report created successfully from validated research. Preview:\n\n{report_preview}...\n\n"
                        "Let the user know the report is ready and ask if they'd like to save it."
                    )
                    return {
                        "response": followup,
                        "tool_used": "report",
                        "report": tool_result.get("report"),
                    }

                elif action == "save" and tool_result.get("status") == "saved":
                    filepath = tool_result.get("filepath", "unknown")
                    followup = await self._call_llm(
                        f"Report saved successfully to: {filepath}\n\nConfirm this to the user."
                    )
                    return {
                        "response": followup,
                        "tool_used": "report",
                        "filepath": filepath,
                    }

                elif action == "list":
                    reports = tool_result.get("reports", [])
                    if reports:
                        reports_list = "\n".join(f"- {r}" for r in reports)
                        followup = await self._call_llm(
                            f"Available reports:\n{reports_list}\n\nPresent this list to the user."
                        )
                    else:
                        followup = await self._call_llm(
                            "No saved reports found. Let the user know."
                        )
                    return {"response": followup, "tool_used": "report"}

                elif action == "load":
                    if tool_result.get("status") == "loaded":
                        report_preview = tool_result.get("report", "")[:500]
                        followup = await self._call_llm(
                            f"Report loaded. Preview:\n\n{report_preview}...\n\nLet the user know the report is loaded."
                        )
                        return {
                            "response": followup,
                            "tool_used": "report",
                            "report": tool_result.get("report"),
                        }

            if "error" in tool_result:
                return {"response": f"Error: {tool_result['error']}", "tool_result": tool_result}

            return {"response": response, "tool_result": tool_result}

        return {"response": response}

    def _format_findings(self, findings: list[dict[str, Any]]) -> str:
        """Format findings for the LLM."""
        parts = []
        for f in findings:
            parts.append(f"- {f.get('title', 'Untitled')}: {f.get('snippet', '')}")
        return "\n".join(parts)

    def _format_validated_findings(self, validated: list[dict[str, Any]]) -> str:
        """Format validated findings for the research cache."""
        parts = []
        for f in validated:
            validation = f.get("validation", {})
            confidence = validation.get("confidence", 0)
            sources = validation.get("sources", [])
            source_note = f" (Verified by: {', '.join(sources[:2])})" if sources else ""
            parts.append(
                f"- [{confidence:.0%} confidence] {f.get('title', 'Untitled')}: "
                f"{f.get('snippet', '')}{source_note}"
            )
        return "\n".join(parts)

    def clear_research_cache(self) -> None:
        """Clear the research cache and validation state."""
        self.research_cache.clear()
        self.validated_findings.clear()
        self.pending_validation.clear()
        self.current_report = None
        self.current_report_title = None

    def get_validation_status(self) -> dict[str, Any]:
        """Get current validation status."""
        return {
            "pending_validation": len(self.pending_validation),
            "validated_findings": len(self.validated_findings),
            "research_cache_items": len(self.research_cache),
            "has_report": self.current_report is not None,
            "report_title": self.current_report_title,
        }

    async def chat(self, message: str) -> str:
        """Simple chat interface that returns just the response text."""
        result = await self.execute(message)
        return result.get("response", "No response generated.")
