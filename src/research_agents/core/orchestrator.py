"""Orchestrator for coordinating multiple agents."""

import json
import re
from typing import Any

import httpx

from .base_agent import BaseAgent
from .config import Settings, get_settings


class Orchestrator:
    """Coordinates multiple agents to accomplish complex tasks."""

    def __init__(self, use_llm_selection: bool = True, settings: Settings | None = None):
        """Initialize the orchestrator.

        Args:
            use_llm_selection: Whether to use LLM for agent selection (falls back to keyword matching if False or no API key)
            settings: Optional settings instance (uses global settings if not provided)
        """
        self.agents: dict[str, BaseAgent] = {}
        self.use_llm_selection = use_llm_selection
        self.settings = settings or get_settings()
        self.execution_history: list[dict[str, Any]] = []

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent

    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent by name.

        Returns:
            True if agent was removed, False if not found
        """
        if name in self.agents:
            del self.agents[name]
            return True
        return False

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get an agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self.agents.keys())

    def get_agent_descriptions(self) -> dict[str, str]:
        """Get all agent names and descriptions."""
        return {name: agent.description for name, agent in self.agents.items()}

    def _get_selectable_agents(self) -> dict[str, BaseAgent]:
        """Get agents that can be auto-selected (excludes 'main' agent)."""
        return {name: agent for name, agent in self.agents.items() if name != "main"}

    def _keyword_select(self, task: str) -> str | None:
        """Select an agent based on keyword matching.

        Args:
            task: The task description

        Returns:
            Agent name or None if no match
        """
        task_lower = task.lower()

        # Define keyword patterns for each agent type
        patterns: dict[str, list[str]] = {
            "research": [
                r"\b(search|find|look up|research|what is|who is|when did|where is)\b",
                r"\b(latest|recent|current|news|information about)\b",
                r"\b(learn about|tell me about|explain)\b",
            ],
            "report": [
                r"\b(report|summarize|compile|document|write up)\b",
                r"\b(create a report|generate report|make a report)\b",
                r"\b(save|store|export)\b.*\b(report|document)\b",
                r"\b(list|show|load)\b.*\b(reports?)\b",
            ],
            "validation": [
                r"\b(validate|verify|fact.?check|confirm|check)\b",
                r"\b(is.*(true|accurate|correct|reliable))\b",
                r"\b(credible|trustworthy|legitimate)\b",
                r"\b(source|citation|evidence|proof)\b",
            ],
        }

        # Score each agent based on pattern matches
        scores: dict[str, int] = {}
        selectable = self._get_selectable_agents()

        for agent_name in selectable:
            if agent_name in patterns:
                score = 0
                for pattern in patterns[agent_name]:
                    if re.search(pattern, task_lower):
                        score += 1
                if score > 0:
                    scores[agent_name] = score

        if scores:
            return max(scores, key=scores.get)
        return None

    async def _llm_select(self, task: str) -> str | None:
        """Use LLM to select the best agent for a task.

        Args:
            task: The task description

        Returns:
            Agent name or None if selection failed
        """
        if not self.settings.api_key:
            return None

        selectable = self._get_selectable_agents()
        if not selectable:
            return None

        agents_desc = "\n".join(
            f"- {name}: {agent.description}"
            for name, agent in selectable.items()
        )

        prompt = f"""Given the following task, select the most appropriate agent to handle it.

Available agents:
{agents_desc}

Task: {task}

Respond with ONLY a JSON object in this format:
{{"agent": "agent_name", "reason": "brief reason for selection"}}

If no agent is suitable, respond with:
{{"agent": null, "reason": "why no agent fits"}}"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.settings.get_api_url("/v1/messages"),
                    headers={
                        "x-api-key": self.settings.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.settings.fast_model,
                        "max_tokens": 256,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=self.settings.request_timeout,
                )

                if response.status_code != 200:
                    return None

                data = response.json()
                content = data["content"][0]["text"]

                # Parse the JSON response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    result = json.loads(content[start:end])
                    agent_name = result.get("agent")
                    if agent_name and agent_name in selectable:
                        return agent_name

        except (httpx.RequestError, json.JSONDecodeError, KeyError):
            pass

        return None

    async def select_agent(self, task: str) -> str | None:
        """Select the best agent for a task.

        Uses LLM selection if enabled and available, falls back to keyword matching.

        Args:
            task: The task description

        Returns:
            Agent name or None if no suitable agent found
        """
        # Try LLM selection first if enabled
        if self.use_llm_selection and self.settings.api_key:
            agent_name = await self._llm_select(task)
            if agent_name:
                return agent_name

        # Fall back to keyword matching
        return self._keyword_select(task)

    async def run(
        self,
        task: str,
        agent_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a task using the specified agent or auto-select.

        Args:
            task: The task to execute
            agent_name: Optional specific agent to use
            context: Optional context to pass to the agent

        Returns:
            Results from the agent execution
        """
        selected_agent: str | None = agent_name

        # Auto-select if no agent specified
        if not selected_agent:
            selected_agent = await self.select_agent(task)
            if not selected_agent:
                return {
                    "status": "error",
                    "error": "No suitable agent found for this task",
                    "task": task,
                }

        agent = self.get_agent(selected_agent)
        if not agent:
            return {
                "status": "error",
                "error": f"Agent '{selected_agent}' not found",
                "task": task,
            }

        # Execute the task
        result = await agent.execute(task, context)

        # Record execution history
        self.execution_history.append({
            "task": task,
            "agent": selected_agent,
            "auto_selected": agent_name is None,
            "status": result.get("status", "unknown"),
        })

        return result

    async def run_pipeline(
        self,
        tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run multiple tasks in sequence, passing context between them.

        Args:
            tasks: List of task dicts with 'task' and optional 'agent', 'context'

        Returns:
            List of results from each task
        """
        results: list[dict[str, Any]] = []
        accumulated_context: dict[str, Any] = {}

        for task_spec in tasks:
            task = task_spec.get("task", "")
            agent_name = task_spec.get("agent")
            context = task_spec.get("context", {})

            # Merge accumulated context
            merged_context = {**accumulated_context, **context}

            result = await self.run(task, agent_name, merged_context)
            results.append(result)

            # Accumulate results for next task
            if result.get("status") == "completed":
                accumulated_context["previous_result"] = result

        return results

    def get_execution_history(self) -> list[dict[str, Any]]:
        """Get the execution history."""
        return self.execution_history.copy()

    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
