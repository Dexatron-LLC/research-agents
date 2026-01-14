"""Validation agent that performs thorough fact-checking against authoritative sources."""

import asyncio
import json
from typing import Any

import httpx

from ..core.base_agent import BaseAgent
from ..core.config import Settings, get_settings
from ..tools.web_search import WebSearchTool


class ValidationAgent(BaseAgent):
    """Agent specialized in thorough validation of research findings against trusted sources."""

    # Trusted domains for fact-checking
    TRUSTED_DOMAINS = [
        "wikipedia.org",
        "scholar.google.com",
        "pubmed.ncbi.nlm.nih.gov",
        "arxiv.org",
        "nature.com",
        "science.org",
        "sciencedirect.com",
        ".edu",
        ".gov",
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "nytimes.com",
        "theguardian.com",
    ]

    def __init__(self, settings: Settings | None = None, verbose: bool = True):
        super().__init__(name="validation")
        self.settings = settings or get_settings()
        self.search_tool = WebSearchTool()
        self.verbose = verbose

    def _log(self, message: str, indent: int = 0) -> None:
        """Print a verbose log message."""
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}[Validation] {message}")

    def _is_trusted_source(self, url: str) -> bool:
        """Check if a URL is from a trusted domain."""
        url_lower = url.lower()
        for domain in self.TRUSTED_DOMAINS:
            if domain in url_lower:
                return True
        return False

    def _get_trust_level(self, url: str) -> str:
        """Get the trust level of a source based on its domain."""
        url_lower = url.lower()

        # Highest trust: academic and scientific
        high_trust = ["scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "arxiv.org",
                      "nature.com", "science.org", "sciencedirect.com", ".edu", ".gov"]
        for domain in high_trust:
            if domain in url_lower:
                return "high"

        # Medium trust: reputable encyclopedias and news
        medium_trust = ["wikipedia.org", "reuters.com", "apnews.com", "bbc.com"]
        for domain in medium_trust:
            if domain in url_lower:
                return "medium"

        return "low"

    def _get_validation_system_prompt(self) -> str:
        """Get system prompt for validation tasks."""
        base_prompt = super().get_system_prompt()

        validation_instructions = """
## Validation Task
You are validating a research claim against source content. Analyze thoroughly and determine:
- VERIFIED: Multiple trusted sources confirm the claim with specific evidence
- PARTIALLY_VERIFIED: Some evidence supports it, but details differ or are incomplete
- UNVERIFIED: No credible evidence found to support the claim
- CONTRADICTED: Trusted sources provide evidence contradicting the claim

When analyzing:
1. Look for specific facts, statistics, or statements that support or contradict the claim
2. Note if the sources are discussing the same topic/context as the claim
3. Consider the recency and authority of the sources
4. Extract direct quotes or evidence from the content

Respond with a JSON object:
{
    "status": "VERIFIED|PARTIALLY_VERIFIED|UNVERIFIED|CONTRADICTED",
    "confidence": 0.0-1.0,
    "reason": "Detailed explanation with specific evidence",
    "evidence": ["Direct quotes or specific facts from sources"],
    "sources": ["list of supporting source URLs"]
}"""

        return f"{base_prompt}\n{validation_instructions}"

    def _get_content_analysis_prompt(self) -> str:
        """Get system prompt for analyzing source content against a claim."""
        return """You are analyzing source content to find evidence for or against a claim.
Extract:
1. Any statements that directly support or contradict the claim
2. Relevant statistics or data
3. The context and authority of the information

Be precise and quote specific passages when possible."""

    async def _call_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call the LLM API for validation analysis."""
        if not self.settings.api_key:
            return json.dumps({
                "status": "UNVERIFIED",
                "confidence": 0.0,
                "reason": "API key not configured",
                "evidence": [],
                "sources": []
            })

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
                    "max_tokens": 2048,
                    "system": system_prompt or self._get_validation_system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.settings.request_timeout,
            )

            if response.status_code != 200:
                return json.dumps({
                    "status": "UNVERIFIED",
                    "confidence": 0.0,
                    "reason": f"API error: {response.status_code}",
                    "evidence": [],
                    "sources": []
                })

            data = response.json()
            return data["content"][0]["text"]

    async def _fetch_page_content(self, url: str) -> dict[str, Any] | None:
        """Fetch page content, handling errors gracefully."""
        try:
            content = await self.search_tool.fetch_page(url, max_content_length=12000)
            return content
        except Exception as e:
            self._log(f"FAILED to fetch: {url[:50]}... ({type(e).__name__})", indent=3)
            return None

    async def _analyze_source_content(
        self,
        claim: str,
        url: str,
        title: str,
        content: str,
    ) -> dict[str, Any]:
        """Analyze source content for evidence supporting or contradicting a claim."""
        self._log(f"Analyzing: {title[:40]}...", indent=3)

        prompt = f"""Analyze this source content for evidence about the following claim:

CLAIM: "{claim}"

SOURCE: {title}
URL: {url}

CONTENT:
{content[:8000]}

Find and extract:
1. Any statements that directly relate to the claim
2. Supporting or contradicting evidence
3. Relevant statistics or facts
4. The credibility and context of the information

Format your response as:
RELEVANCE: [high/medium/low/none]
SUPPORTS_CLAIM: [yes/partially/no/contradicts]
EVIDENCE:
- [specific quote or fact 1]
- [specific quote or fact 2]
ANALYSIS: [Brief analysis of how this source relates to the claim]"""

        result = await self._call_llm(prompt, system_prompt=self._get_content_analysis_prompt())

        # Parse the response
        relevance = "low"
        supports = "no"
        evidence = []
        analysis = ""

        lines = result.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("RELEVANCE:"):
                relevance = line.replace("RELEVANCE:", "").strip().lower()
            elif line.startswith("SUPPORTS_CLAIM:"):
                supports = line.replace("SUPPORTS_CLAIM:", "").strip().lower()
            elif line.startswith("EVIDENCE:"):
                current_section = "evidence"
            elif line.startswith("ANALYSIS:"):
                current_section = "analysis"
                analysis = line.replace("ANALYSIS:", "").strip()
            elif line.startswith("- ") and current_section == "evidence":
                evidence.append(line[2:])
            elif current_section == "analysis" and line:
                analysis += " " + line

        trust_level = self._get_trust_level(url)
        self._log(f"Result: relevance={relevance}, supports={supports}, trust={trust_level}", indent=4)

        return {
            "url": url,
            "title": title,
            "relevance": relevance,
            "supports_claim": supports,
            "evidence": evidence,
            "analysis": analysis.strip(),
            "trust_level": trust_level,
        }

    def _parse_validation_result(self, llm_response: str) -> dict[str, Any]:
        """Parse the LLM's validation response."""
        try:
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(llm_response[start:end])
                # Ensure evidence field exists
                if "evidence" not in result:
                    result["evidence"] = []
                return result
        except json.JSONDecodeError:
            pass

        return {
            "status": "UNVERIFIED",
            "confidence": 0.0,
            "reason": "Failed to parse validation result",
            "evidence": [],
            "sources": []
        }

    async def validate_claim(self, claim: str, original_source: str = "") -> dict[str, Any]:
        """Validate a single claim with thorough source analysis.

        Args:
            claim: The claim to validate
            original_source: The original source URL of the claim

        Returns:
            Validation result with status, confidence, reason, evidence, and sources
        """
        self._log(f"Validating claim: {claim[:60]}...")

        # Step 1: Search for validation using trusted source keywords
        search_queries = [
            f"{claim} site:wikipedia.org OR site:edu OR site:gov",
            claim,  # Broader search as backup
        ]

        all_trusted_results = []
        seen_urls: set[str] = set()

        self._log("Searching trusted sources...", indent=1)
        for query in search_queries:
            self._log(f"Query: {query[:50]}...", indent=2)
            search_results = self.search_tool.search(query, num_results=8)
            trusted_count = 0
            for result in search_results:
                if self._is_trusted_source(result.url) and result.url not in seen_urls:
                    all_trusted_results.append({
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                    })
                    seen_urls.add(result.url)
                    trusted_count += 1
            self._log(f"Found {trusted_count} trusted sources", indent=2)

            if len(all_trusted_results) >= 5:
                break

        self._log(f"Total trusted sources found: {len(all_trusted_results)}", indent=1)

        # Step 2: Fetch and analyze content from top trusted sources
        sources_to_analyze = all_trusted_results[:3]
        analyzed_sources = []

        if sources_to_analyze:
            self._log(f"Fetching {len(sources_to_analyze)} pages for analysis...", indent=1)

            # Fetch pages in parallel
            for source in sources_to_analyze:
                self._log(f"Fetching: {source['url'][:50]}...", indent=2)

            fetch_tasks = [
                self._fetch_page_content(source["url"])
                for source in sources_to_analyze
            ]
            fetched_pages = await asyncio.gather(*fetch_tasks)

            successful_fetches = sum(1 for p in fetched_pages if p and p.get("content"))
            self._log(f"Successfully fetched {successful_fetches}/{len(sources_to_analyze)} pages", indent=1)

            # Analyze each page
            self._log("Analyzing source content...", indent=1)
            analysis_tasks = []
            for source, page_content in zip(sources_to_analyze, fetched_pages):
                if page_content and page_content.get("content"):
                    analysis_tasks.append(
                        self._analyze_source_content(
                            claim=claim,
                            url=source["url"],
                            title=source["title"],
                            content=page_content["content"],
                        )
                    )

            if analysis_tasks:
                analyzed_sources = await asyncio.gather(*analysis_tasks)

            self._log(f"Analyzed {len(analyzed_sources)} sources", indent=1)

        # Step 3: Synthesize validation result using LLM
        self._log("Synthesizing validation result...", indent=1)

        if analyzed_sources:
            # Build detailed evidence summary
            evidence_summary = []
            for source in analyzed_sources:
                source_text = f"\n### Source: {source['title']} ({source['trust_level']} trust)\n"
                source_text += f"URL: {source['url']}\n"
                source_text += f"Relevance: {source['relevance']}\n"
                source_text += f"Supports Claim: {source['supports_claim']}\n"
                if source['evidence']:
                    source_text += "Evidence:\n"
                    for e in source['evidence']:
                        source_text += f"  - {e}\n"
                source_text += f"Analysis: {source['analysis']}\n"
                evidence_summary.append(source_text)

            prompt = f"""Based on thorough analysis of trusted sources, validate this claim:

CLAIM: "{claim}"
ORIGINAL SOURCE: {original_source or 'Unknown'}

ANALYZED SOURCES AND EVIDENCE:
{''.join(evidence_summary)}

Additional search results (snippets only):
{chr(10).join(f"- [{r['title']}]({r['url']}): {r['snippet']}" for r in all_trusted_results[3:6])}

Provide your final validation assessment considering:
1. How many sources support vs contradict the claim
2. The trust level of supporting sources
3. The strength and specificity of the evidence
4. Any caveats or nuances in the claim"""
        else:
            # No detailed analysis available, use snippets only
            if all_trusted_results:
                sources_text = "\n".join(
                    f"- [{r['title']}]({r['url']}): {r['snippet']}"
                    for r in all_trusted_results[:5]
                )
                prompt = f"""Validate this claim based on search snippets:

CLAIM: "{claim}"
ORIGINAL SOURCE: {original_source or 'Unknown'}

Search results from trusted sources:
{sources_text}

Note: Full page content could not be fetched. Base your assessment on available snippets."""
            else:
                prompt = f"""Validate this claim:

CLAIM: "{claim}"
ORIGINAL SOURCE: {original_source or 'Unknown'}

No results found from trusted sources (Wikipedia, .edu, .gov, scientific journals).
Provide your assessment based on the lack of corroborating evidence."""

        llm_response = await self._call_llm(prompt)
        result = self._parse_validation_result(llm_response)

        # Add metadata
        result["claim"] = claim
        result["original_source"] = original_source
        result["trusted_results_found"] = len(all_trusted_results)
        result["sources_analyzed"] = len(analyzed_sources)

        # Log final result
        status_symbol = "PASS" if result["status"] in ("VERIFIED", "PARTIALLY_VERIFIED") else "FAIL"
        self._log(f"Result: {status_symbol} - {result['status']} (confidence: {result['confidence']:.0%})", indent=1)

        return result

    async def validate_findings(
        self,
        findings: list[dict[str, Any]],
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Validate a list of research findings with thorough fact-checking.

        Args:
            findings: List of findings with 'title', 'snippet', 'url'
            min_confidence: Minimum confidence threshold to keep a finding

        Returns:
            Dictionary with validated, removed, and stats
        """
        self._log(f"Validating {len(findings)} findings (min confidence: {min_confidence:.0%})...")

        validated = []
        removed = []

        for i, finding in enumerate(findings, 1):
            self._log(f"--- Finding {i}/{len(findings)} ---")
            claim = f"{finding.get('title', '')}: {finding.get('snippet', '')}"
            original_url = finding.get("url", "")

            result = await self.validate_claim(claim, original_url)

            finding_with_validation = {
                **finding,
                "validation": result,
            }

            if result["status"] in ("VERIFIED", "PARTIALLY_VERIFIED") and result["confidence"] >= min_confidence:
                validated.append(finding_with_validation)
                self._log(f"Finding {i}: ACCEPTED", indent=1)
            else:
                removed.append(finding_with_validation)
                self._log(f"Finding {i}: REJECTED ({result['status']}, {result['confidence']:.0%})", indent=1)

        self._log(f"Validation complete: {len(validated)} accepted, {len(removed)} rejected")

        return {
            "validated": validated,
            "removed": removed,
            "stats": {
                "total": len(findings),
                "validated_count": len(validated),
                "removed_count": len(removed),
                "validation_rate": len(validated) / len(findings) if findings else 0,
            }
        }

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a validation task.

        Args:
            task: Task description or claim to validate
            context: Should contain 'findings' list to validate, or 'claim' for single validation

        Returns:
            Dictionary containing validation results
        """
        context = context or {}
        self.verbose = context.get("verbose", self.verbose)

        # Single claim validation
        if context.get("claim"):
            result = await self.validate_claim(
                context["claim"],
                context.get("original_source", "")
            )
            return {
                "status": "completed",
                "validation": result,
            }

        # Batch validation of findings
        findings = context.get("findings", [])
        if findings:
            min_confidence = context.get("min_confidence", 0.5)
            results = await self.validate_findings(findings, min_confidence)
            return {
                "status": "completed",
                **results,
            }

        # If no findings provided, treat task as a single claim
        result = await self.validate_claim(task)
        return {
            "status": "completed",
            "validation": result,
        }

    async def close(self) -> None:
        """Clean up resources."""
        await self.search_tool.close()
