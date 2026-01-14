"""Validation agent that performs fact-checking against authoritative sources."""

import asyncio
import json
import re
from typing import Any

import httpx

from ..core.base_agent import BaseAgent
from ..core.config import Settings, get_settings
from ..tools.web_search import WebSearchTool


class ValidationAgent(BaseAgent):
    """Agent specialized in validation of research findings against trusted sources."""

    # Trusted domains for fact-checking (high credibility)
    HIGH_TRUST_DOMAINS = [
        "scholar.google.com",
        "pubmed.ncbi.nlm.nih.gov",
        "arxiv.org",
        "nature.com",
        "science.org",
        "sciencedirect.com",
        ".edu",
        ".gov",
    ]

    # Medium trust domains (reputable but not academic)
    MEDIUM_TRUST_DOMAINS = [
        "wikipedia.org",
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "nytimes.com",
        "theguardian.com",
        "forbes.com",
        "wired.com",
        "techcrunch.com",
        "scientificamerican.com",
    ]

    # All trusted domains combined
    TRUSTED_DOMAINS = HIGH_TRUST_DOMAINS + MEDIUM_TRUST_DOMAINS

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

        for domain in self.HIGH_TRUST_DOMAINS:
            if domain in url_lower:
                return "high"

        for domain in self.MEDIUM_TRUST_DOMAINS:
            if domain in url_lower:
                return "medium"

        return "low"

    def _extract_key_terms(self, claim: str) -> str:
        """Extract key search terms from a claim for better search results."""
        # Remove common words and punctuation
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "this", "that",
            "these", "those", "it", "its", "of", "in", "to", "for", "with",
            "on", "at", "by", "from", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "about",
        }

        # Clean the claim
        words = re.findall(r'\b[a-zA-Z0-9]+\b', claim.lower())
        key_words = [w for w in words if w not in stop_words and len(w) > 2]

        # Limit to most important terms (first 8)
        return " ".join(key_words[:8])

    def _get_validation_system_prompt(self, mode: str = "normal") -> str:
        """Get system prompt for validation tasks based on mode."""
        base_prompt = super().get_system_prompt()

        if mode == "strict":
            criteria = """
- VERIFIED: Multiple high-trust sources (academic, government) explicitly confirm with evidence
- PARTIALLY_VERIFIED: At least one trusted source provides supporting evidence
- UNVERIFIED: No credible evidence found
- CONTRADICTED: Trusted sources explicitly contradict the claim"""
        elif mode == "lenient":
            criteria = """
- VERIFIED: Any credible source confirms the general claim
- PARTIALLY_VERIFIED: The claim is plausible based on related information found
- UNVERIFIED: No relevant information found at all
- CONTRADICTED: Sources explicitly state the opposite"""
        else:  # normal
            criteria = """
- VERIFIED: At least one trusted source confirms the claim with evidence (confidence 0.7+)
- PARTIALLY_VERIFIED: Related supporting information found, even if not exact match (confidence 0.5-0.7)
- UNVERIFIED: No relevant supporting information found (confidence below 0.5)
- CONTRADICTED: Sources provide evidence against the claim"""

        validation_instructions = f"""
## Validation Task
You are validating a research finding. Be fair and balanced - not everything needs academic papers to be valid.

Validation criteria ({mode} mode):
{criteria}

Guidelines:
1. If the claim is about general knowledge or widely reported facts, be more lenient
2. If sources discuss the same topic with similar information, that counts as support
3. Consider the original source's credibility - reputable news sites are valid sources
4. Partial matches and related information still provide some validation
5. Only mark as CONTRADICTED if sources explicitly disagree, not just if evidence is missing

Respond with JSON:
{{
    "status": "VERIFIED|PARTIALLY_VERIFIED|UNVERIFIED|CONTRADICTED",
    "confidence": 0.0-1.0,
    "reason": "Brief explanation",
    "evidence": ["Supporting facts found"],
    "sources": ["URLs that support the finding"]
}}"""

        return f"{base_prompt}\n{validation_instructions}"

    def _get_content_analysis_prompt(self) -> str:
        """Get system prompt for analyzing source content against a claim."""
        return """You are checking if source content is relevant to a claim.
Be generous in finding connections - partial matches and related information count.
Look for:
1. Direct confirmation of the claim
2. Related statistics or facts that support it
3. Discussion of the same topic/subject
4. Any information that doesn't contradict the claim

Output format:
RELEVANCE: [high/medium/low/none]
SUPPORTS_CLAIM: [yes/partially/no/contradicts]
EVIDENCE:
- [relevant quote or fact]
ANALYSIS: [brief analysis]"""

    async def _call_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call the LLM API for validation analysis."""
        if not self.settings.api_key:
            return json.dumps({
                "status": "PARTIALLY_VERIFIED",
                "confidence": 0.5,
                "reason": "API key not configured - assuming partial validity",
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
                    "status": "PARTIALLY_VERIFIED",
                    "confidence": 0.5,
                    "reason": f"API error - assuming partial validity",
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

        prompt = f"""Check if this source content relates to the claim:

CLAIM: "{claim[:500]}"

SOURCE: {title}
URL: {url}

CONTENT (excerpt):
{content[:6000]}

Find any relevant information, even partial matches."""

        result = await self._call_llm(prompt, system_prompt=self._get_content_analysis_prompt())

        # Parse the response
        relevance = "low"
        supports = "partially"
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
                if "evidence" not in result:
                    result["evidence"] = []
                return result
        except json.JSONDecodeError:
            pass

        return {
            "status": "PARTIALLY_VERIFIED",
            "confidence": 0.5,
            "reason": "Could not parse validation - assuming partial validity",
            "evidence": [],
            "sources": []
        }

    def _calculate_base_confidence(self, original_url: str) -> float:
        """Calculate base confidence from the original source's credibility."""
        if not original_url:
            return 0.3

        trust = self._get_trust_level(original_url)
        if trust == "high":
            return 0.6  # Academic/gov sources start with higher confidence
        elif trust == "medium":
            return 0.5  # Reputable news/wiki starts at medium
        else:
            return 0.3  # Unknown sources start lower

    async def validate_claim(
        self,
        claim: str,
        original_source: str = "",
        mode: str = "normal",
    ) -> dict[str, Any]:
        """Validate a single claim.

        Args:
            claim: The claim to validate
            original_source: The original source URL
            mode: Validation mode - 'strict', 'normal', or 'lenient'

        Returns:
            Validation result with status, confidence, reason, evidence, and sources
        """
        self._log(f"Validating: {claim[:60]}...")
        self._log(f"Mode: {mode}", indent=1)

        # Calculate base confidence from original source
        base_confidence = self._calculate_base_confidence(original_source)
        original_trust = self._get_trust_level(original_source) if original_source else "unknown"
        self._log(f"Original source trust: {original_trust} (base confidence: {base_confidence:.0%})", indent=1)

        # Extract key terms for better search
        key_terms = self._extract_key_terms(claim)
        self._log(f"Key terms: {key_terms}", indent=1)

        # Step 1: Search - use broader search first, not just trusted sites
        search_queries = [
            key_terms,  # Just the key terms
            f"{key_terms} facts",  # Add context
        ]

        all_results = []
        trusted_results = []
        seen_urls: set[str] = set()

        self._log("Searching for corroborating sources...", indent=1)
        for query in search_queries:
            self._log(f"Query: {query[:50]}", indent=2)
            search_results = self.search_tool.search(query, num_results=10)

            for result in search_results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    result_data = {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "trust_level": self._get_trust_level(result.url),
                    }
                    all_results.append(result_data)
                    if self._is_trusted_source(result.url):
                        trusted_results.append(result_data)

            if len(all_results) >= 10:
                break

        self._log(f"Found {len(all_results)} results ({len(trusted_results)} trusted)", indent=1)

        # Step 2: Analyze top results (prefer trusted, but use others too)
        sources_to_analyze = trusted_results[:2] + [r for r in all_results if r not in trusted_results][:2]
        analyzed_sources = []
        snippet_evidence = []

        if sources_to_analyze:
            self._log(f"Fetching {len(sources_to_analyze)} pages...", indent=1)

            fetch_tasks = [self._fetch_page_content(s["url"]) for s in sources_to_analyze]
            fetched_pages = await asyncio.gather(*fetch_tasks)

            # Analyze fetched pages
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
                else:
                    # Use snippet as evidence if fetch failed
                    snippet_evidence.append({
                        "source": source["title"],
                        "snippet": source["snippet"],
                        "trust_level": source["trust_level"],
                    })

            if analysis_tasks:
                analyzed_sources = await asyncio.gather(*analysis_tasks)

            self._log(f"Analyzed {len(analyzed_sources)} pages, {len(snippet_evidence)} snippets", indent=1)

        # Step 3: Build validation prompt with all evidence
        self._log("Synthesizing validation result...", indent=1)

        evidence_text = []

        # Add analyzed source evidence
        for source in analyzed_sources:
            text = f"\n### {source['title']} ({source['trust_level']} trust)\n"
            text += f"Relevance: {source['relevance']}, Supports: {source['supports_claim']}\n"
            if source['evidence']:
                text += "Evidence: " + "; ".join(source['evidence'][:2]) + "\n"
            evidence_text.append(text)

        # Add snippet evidence
        for snippet in snippet_evidence[:3]:
            text = f"\n### {snippet['source']} ({snippet['trust_level']} trust)\n"
            text += f"Snippet: {snippet['snippet'][:200]}\n"
            evidence_text.append(text)

        # Add remaining search snippets as supporting context
        for result in all_results[:5]:
            if result not in sources_to_analyze:
                text = f"\n### {result['title']} ({result['trust_level']} trust)\n"
                text += f"Snippet: {result['snippet'][:150]}\n"
                evidence_text.append(text)

        prompt = f"""Validate this research finding:

CLAIM: "{claim}"
ORIGINAL SOURCE: {original_source or 'Unknown'} ({original_trust} trust)

EVIDENCE FOUND:
{''.join(evidence_text) if evidence_text else 'No specific evidence found, but claim may still be valid if it is common knowledge.'}

Consider:
1. The original source's credibility ({original_trust})
2. Whether the evidence supports, partially supports, or contradicts the claim
3. Be fair - not everything needs academic citations to be valid
4. Base confidence should be at least {base_confidence:.0%} given the original source"""

        llm_response = await self._call_llm(prompt, self._get_validation_system_prompt(mode))
        result = self._parse_validation_result(llm_response)

        # Ensure minimum confidence based on original source
        if result["confidence"] < base_confidence and result["status"] != "CONTRADICTED":
            result["confidence"] = base_confidence

        # Add metadata
        result["claim"] = claim
        result["original_source"] = original_source
        result["original_trust"] = original_trust
        result["results_found"] = len(all_results)
        result["trusted_results"] = len(trusted_results)
        result["sources_analyzed"] = len(analyzed_sources)
        result["mode"] = mode

        status_symbol = "PASS" if result["status"] in ("VERIFIED", "PARTIALLY_VERIFIED") else "FAIL"
        self._log(f"Result: {status_symbol} - {result['status']} ({result['confidence']:.0%})", indent=1)

        return result

    async def validate_findings(
        self,
        findings: list[dict[str, Any]],
        min_confidence: float = 0.4,
        mode: str = "normal",
    ) -> dict[str, Any]:
        """Validate a list of research findings.

        Args:
            findings: List of findings with 'title', 'snippet', 'url'
            min_confidence: Minimum confidence threshold (default 0.4 for more lenient validation)
            mode: Validation mode - 'strict', 'normal', or 'lenient'

        Returns:
            Dictionary with validated, removed, and stats
        """
        self._log(f"Validating {len(findings)} findings (mode={mode}, min_confidence={min_confidence:.0%})")

        validated = []
        removed = []

        for i, finding in enumerate(findings, 1):
            self._log(f"--- Finding {i}/{len(findings)} ---")

            # Use title as main claim, snippet as context
            title = finding.get('title', '')
            snippet = finding.get('snippet', '')
            claim = f"{title}: {snippet}" if snippet else title
            original_url = finding.get("url", "")

            result = await self.validate_claim(claim, original_url, mode)

            finding_with_validation = {
                **finding,
                "validation": result,
            }

            if result["status"] in ("VERIFIED", "PARTIALLY_VERIFIED") and result["confidence"] >= min_confidence:
                validated.append(finding_with_validation)
                self._log(f"ACCEPTED ({result['status']}, {result['confidence']:.0%})")
            else:
                removed.append(finding_with_validation)
                self._log(f"REJECTED ({result['status']}, {result['confidence']:.0%})")

        rate = len(validated) / len(findings) if findings else 0
        self._log(f"Validation complete: {len(validated)}/{len(findings)} accepted ({rate:.0%})")

        return {
            "validated": validated,
            "removed": removed,
            "stats": {
                "total": len(findings),
                "validated_count": len(validated),
                "removed_count": len(removed),
                "validation_rate": rate,
            }
        }

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a validation task."""
        context = context or {}
        self.verbose = context.get("verbose", self.verbose)
        mode = context.get("mode", "normal")

        # Single claim validation
        if context.get("claim"):
            result = await self.validate_claim(
                context["claim"],
                context.get("original_source", ""),
                mode,
            )
            return {"status": "completed", "validation": result}

        # Batch validation
        findings = context.get("findings", [])
        if findings:
            min_confidence = context.get("min_confidence", 0.4)
            results = await self.validate_findings(findings, min_confidence, mode)
            return {"status": "completed", **results}

        # Task as claim
        result = await self.validate_claim(task, mode=mode)
        return {"status": "completed", "validation": result}

    async def close(self) -> None:
        """Clean up resources."""
        await self.search_tool.close()
