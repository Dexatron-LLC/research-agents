"""Research agent that performs thorough web research using search, page fetching, and LLM analysis."""

import asyncio
from typing import Any

import httpx

from ..core.base_agent import BaseAgent
from ..core.config import Settings, get_settings
from ..tools.web_search import WebSearchTool


class ResearchAgent(BaseAgent):
    """Agent specialized in thorough research using web search and content analysis."""

    def __init__(self, settings: Settings | None = None, verbose: bool = True):
        super().__init__(name="research")
        self.settings = settings or get_settings()
        self.search_tool = WebSearchTool()
        self.verbose = verbose

    def _log(self, message: str, indent: int = 0) -> None:
        """Print a verbose log message."""
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}[Research] {message}")

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

    async def _generate_search_queries(self, topic: str, research_context: dict[str, Any] | None = None) -> list[str]:
        """Generate multiple search queries to explore a topic thoroughly."""
        self._log("Generating search queries...")

        # Build context-aware prompt for follow-up queries
        context_info = ""
        if research_context and research_context.get("is_follow_up"):
            original_topic = research_context.get("topic", "")
            previous_queries = research_context.get("previous_queries", [])
            key_facts = research_context.get("key_facts_found", [])

            context_info = f"""
This is a FOLLOW-UP query in the context of ongoing research about: "{original_topic}"
Previous queries made: {previous_queries}
Key facts already discovered:
{chr(10).join(f'- {fact[:100]}' for fact in key_facts[:5])}

Generate queries that dig DEEPER into this specific aspect, avoiding repetition of previous queries.
"""

        prompt = f"""Generate 3 different search queries to thoroughly research this topic: "{topic}"
{context_info}
The queries should:
1. The main topic query
2. A query that explores a specific aspect or subtopic
3. A query that looks for recent news or developments

Return only the queries, one per line, no numbering or explanations."""

        result = await self._call_llm(prompt, system_prompt="You are a search query generator. Output only search queries, one per line.")

        if result.startswith("Error:"):
            self._log(f"Query generation failed, using original topic", indent=1)
            return [topic]  # Fallback to original topic

        queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
        queries = queries[:3] if queries else [topic]

        for i, q in enumerate(queries, 1):
            self._log(f"Query {i}: {q}", indent=1)

        return queries

    async def _fetch_page_content(self, url: str) -> dict[str, Any] | None:
        """Fetch and return page content, handling errors gracefully."""
        try:
            content = await self.search_tool.fetch_page(url, max_content_length=15000)
            return content
        except Exception as e:
            self._log(f"FAILED to fetch: {url[:60]}... ({type(e).__name__})", indent=2)
            return None

    async def _analyze_content(self, url: str, title: str, content: str, topic: str) -> dict[str, Any]:
        """Use LLM to analyze page content and extract relevant findings."""
        self._log(f"Analyzing: {title[:50]}...", indent=2)

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
            self._log(f"Analysis FAILED: {result}", indent=3)
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

        self._log(f"Found {len(findings)} findings, credibility: {credibility}", indent=3)

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
        self._log("Synthesizing findings from all sources...")

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

        self._log(f"Synthesized {len(findings)} total findings", indent=1)
        return findings

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a thorough research task.

        Args:
            task: The research question or topic
            context: Optional context with settings like 'depth' (quick/normal/thorough),
                     'research_context' for follow-up queries, 'exclude_urls' to skip

        Returns:
            Dictionary containing comprehensive research findings
        """
        context = context or {}
        depth = context.get("depth", "normal")
        self.verbose = context.get("verbose", self.verbose)
        exclude_urls = set(context.get("exclude_urls", []))
        research_context = context.get("research_context", {})

        self._log(f"Starting research on: {task}")
        self._log(f"Depth: {depth}", indent=1)

        # Log if this is a follow-up query
        if research_context.get("is_follow_up"):
            self._log(f"Follow-up query in context of: {research_context.get('topic', 'unknown')}", indent=1)
            self._log(f"Excluding {len(exclude_urls)} already-explored URLs", indent=1)

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

        # Step 1: Generate search queries (with context for follow-ups)
        queries = await self._generate_search_queries(task, research_context)
        queries = queries[:num_queries]

        # Step 2: Perform searches
        self._log("Searching the web...")
        for query in queries:
            self._log(f"Searching: {query[:50]}...", indent=1)
            results = self.search_tool.search(query, num_results=5)
            self._log(f"Found {len(results)} results", indent=2)
            for result in results:
                if result.url and result.url not in sources:
                    # Skip URLs that have already been explored in context
                    if result.url in exclude_urls:
                        self._log(f"Skipping (already explored): {result.url[:50]}...", indent=3)
                        continue
                    all_search_results.append(result)
                    sources.add(result.url)

        self._log(f"Total unique results: {len(all_search_results)}")

        if not all_search_results:
            self._log("No search results found!")
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

        self._log(f"Fetching {len(pages_to_analyze)} pages...")
        for result in pages_to_analyze:
            self._log(f"Fetching: {result.url[:60]}...", indent=1)

        fetch_tasks = [
            self._fetch_page_content(result.url)
            for result in pages_to_analyze
        ]
        fetched_pages = await asyncio.gather(*fetch_tasks)

        # Count successful fetches
        successful_fetches = sum(1 for p in fetched_pages if p and p.get("content"))
        self._log(f"Successfully fetched {successful_fetches}/{len(pages_to_analyze)} pages")

        # Step 4: Analyze each fetched page
        self._log("Analyzing page content...")
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

        self._log(f"Analyzed {len(analyzed_sources)} pages")

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

        self._log(f"Research complete: {len(findings)} findings from {len(sources)} sources")

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
