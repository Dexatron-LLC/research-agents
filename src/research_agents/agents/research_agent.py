"""Research agent that uses web search to gather information."""

from typing import Any

from ..core.base_agent import BaseAgent
from ..tools.web_search import WebSearchTool


class ResearchAgent(BaseAgent):
    """Agent specialized in researching topics using web search."""

    def __init__(self):
        super().__init__(name="research")
        self.search_tool = WebSearchTool()

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a research task.

        Args:
            task: The research question or topic
            context: Optional context from other agents

        Returns:
            Dictionary containing research findings
        """
        results = self.search_tool.search(task, num_results=5)

        findings: list[dict[str, Any]] = []
        sources: list[str] = []

        for result in results:
            findings.append({
                "title": result.title,
                "snippet": result.snippet,
                "url": result.url,
            })
            sources.append(result.url)

        return {
            "task": task,
            "status": "completed",
            "findings": findings,
            "sources": sources,
        }

    async def close(self) -> None:
        """Clean up resources."""
        await self.search_tool.close()
