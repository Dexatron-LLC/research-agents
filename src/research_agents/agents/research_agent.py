"""Research agent that performs thorough web research using search, page fetching, and LLM analysis."""

import asyncio
from typing import Any

import httpx

from ..core.base_agent import BaseAgent
from ..core.config import Settings, get_settings
from ..tools.web_search import WebSearchTool


class ResearchAgent(BaseAgent):
    """Agent specialized in thorough research using web search and content analysis."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="research")
        self.settings = settings or get_settings()
        self.search_tool = WebSearchTool()

    def _get_research_system_prompt(self) -> str:
        """Get system prompt for research analysis."""
        base_prompt = super().get_system_prompt()

        guidelines = """
## Research Analysis Guidelines
- Extract key facts, statistics, and insights from the provided content
- Identify the main claims and supporting evidence
- Note any conflicting information or different perspectives
- Evaluate source credibility based on the content
- Summarize findings concisely but comprehensively
- Always cite which source each finding comes from
"""
        return f"{base_prompt}\n{guidelines}"

    async def _call_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call the LLM API for analysis."""
        if not self.settings.api_key:
            return "Error: ANTHROPIC_API_KEY not configured."

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.settings.get_api_url("/v1/messages"),
                headers={
                    "x-api-key": self.settings.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.settings.fast_model,  # Use fast model for research tasks
                    "max_tokens": 2048,
                    "system": system_prompt or self._get_research_system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.settings.request_timeout,
            )

            if response.status_code != 200:
                return f"Error: API returned status {response.status_code}"

            data = response.json()
            return data["content"][0]["text"]

    async def _generate_search_queries(self, topic: str) -> list[str]:
        """Generate multiple search queries to explore a topic thoroughly."""
        prompt = f"""Generate 3 different search queries to thoroughly research this topic: "{topic}"

The queries should:
1. The main topic query
2. A query that explores a specific aspect or subtopic
3. A query that looks for recent news or developments

Return only the queries, one per line, no numbering or explanations."""

        result = await self._call_llm(prompt, system_prompt="You are a search query generator. Output only search queries, one per line.")

        if result.startswith("Error:"):
            return [topic]  # Fallback to original topic

        queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
        return queries[:3] if queries else [topic]

    async def _fetch_page_content(self, url: str) -> dict[str, Any] | None:
        """Fetch and return page content, handling errors gracefully."""
        try:
            content = await self.search_tool.fetch_page(url, max_content_length=15000)
            return content
        except Exception:
            return None

    async def _analyze_content(self, url: str, title: str, content: str, topic: str) -> dict[str, Any]:
        """Use LLM to analyze page content and extract relevant findings."""
        prompt = f"""Analyze this web page content for information about: "{topic}"

Source: {title}
URL: {url}

Content:
{content[:10000]}

Extract:
1. Key facts and findings relevant to the topic (3-5 bullet points)
2. Any statistics or data mentioned
3. The credibility assessment (high/medium/low) based on the content quality
4. A brief summary (2-3 sentences)

Format your response as:
FINDINGS:
- [finding 1]
- [finding 2]
...

STATISTICS:
- [stat 1]
...

CREDIBILITY: [high/medium/low]

SUMMARY:
[summary text]"""

        result = await self._call_llm(prompt)

        if result.startswith("Error:"):
            return {
                "url": url,
                "title": title,
                "findings": [],
                "summary": "Could not analyze content",
                "credibility": "unknown",
            }

        # Parse the LLM response
        findings = []
        statistics = []
        credibility = "medium"
        summary = ""

        lines = result.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("FINDINGS:"):
                current_section = "findings"
            elif line.startswith("STATISTICS:"):
                current_section = "statistics"
            elif line.startswith("CREDIBILITY:"):
                current_section = None
                credibility = line.replace("CREDIBILITY:", "").strip().lower()
            elif line.startswith("SUMMARY:"):
                current_section = "summary"
            elif line.startswith("- ") and current_section == "findings":
                findings.append(line[2:])
            elif line.startswith("- ") and current_section == "statistics":
                statistics.append(line[2:])
            elif current_section == "summary" and line:
                summary += line + " "

        return {
            "url": url,
            "title": title,
            "findings": findings,
            "statistics": statistics,
            "credibility": credibility,
            "summary": summary.strip(),
        }

    async def _synthesize_research(self, topic: str, analyzed_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Synthesize findings from multiple sources into coherent research results."""
        # Compile all findings
        all_findings = []
        for source in analyzed_sources:
            for finding in source.get("findings", []):
                all_findings.append({
                    "content": finding,
                    "source": source["title"],
                    "url": source["url"],
                    "credibility": source.get("credibility", "medium"),
                })

        # Create structured findings
        findings = []
        for i, f in enumerate(all_findings[:15]):  # Limit to top 15 findings
            findings.append({
                "title": f"{topic} - Finding {i + 1}",
                "snippet": f["content"],
                "url": f["url"],
                "source": f["source"],
                "credibility": f["credibility"],
            })

        return findings

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a thorough research task.

        Args:
            task: The research question or topic
            context: Optional context with settings like 'depth' (quick/normal/thorough)

        Returns:
            Dictionary containing comprehensive research findings
        """
        context = context or {}
        depth = context.get("depth", "normal")

        # Determine search parameters based on depth
        if depth == "quick":
            num_queries = 1
            pages_to_fetch = 2
        elif depth == "thorough":
            num_queries = 3
            pages_to_fetch = 5
        else:  # normal
            num_queries = 2
            pages_to_fetch = 3

        all_search_results = []
        sources: set[str] = set()

        # Step 1: Generate search queries
        queries = await self._generate_search_queries(task)
        queries = queries[:num_queries]

        # Step 2: Perform searches
        for query in queries:
            results = self.search_tool.search(query, num_results=5)
            for result in results:
                if result.url and result.url not in sources:
                    all_search_results.append(result)
                    sources.add(result.url)

        if not all_search_results:
            return {
                "task": task,
                "status": "completed",
                "findings": [],
                "sources": [],
                "message": "No search results found",
            }

        # Step 3: Fetch and analyze top pages
        pages_to_analyze = all_search_results[:pages_to_fetch]
        analyzed_sources = []

        fetch_tasks = [
            self._fetch_page_content(result.url)
            for result in pages_to_analyze
        ]
        fetched_pages = await asyncio.gather(*fetch_tasks)

        # Step 4: Analyze each fetched page
        analysis_tasks = []
        for result, page_content in zip(pages_to_analyze, fetched_pages):
            if page_content and page_content.get("content"):
                analysis_tasks.append(
                    self._analyze_content(
                        url=result.url,
                        title=result.title,
                        content=page_content["content"],
                        topic=task,
                    )
                )

        if analysis_tasks:
            analyzed_sources = await asyncio.gather(*analysis_tasks)

        # Step 5: Synthesize findings
        findings = await self._synthesize_research(task, analyzed_sources)

        # Also include search snippets for sources we couldn't fetch
        for result in all_search_results[pages_to_fetch:pages_to_fetch + 5]:
            findings.append({
                "title": result.title,
                "snippet": result.snippet,
                "url": result.url,
                "source": "Search Result",
                "credibility": "unverified",
            })

        return {
            "task": task,
            "status": "completed",
            "findings": findings,
            "sources": list(sources),
            "queries_used": queries,
            "pages_analyzed": len(analyzed_sources),
            "depth": depth,
        }

    async def close(self) -> None:
        """Clean up resources."""
        await self.search_tool.close()
