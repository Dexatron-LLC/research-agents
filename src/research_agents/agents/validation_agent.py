"""Validation agent that fact-checks research findings against authoritative sources."""

import json
from typing import Any

import httpx

from ..core.base_agent import BaseAgent
from ..core.config import Settings, get_settings
from ..tools.web_search import WebSearchTool


class ValidationAgent(BaseAgent):
    """Agent specialized in validating research findings against trusted sources."""

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

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="validation")
        self.settings = settings or get_settings()
        self.search_tool = WebSearchTool()

    def _is_trusted_source(self, url: str) -> bool:
        """Check if a URL is from a trusted domain.

        Args:
            url: The URL to check

        Returns:
            True if the URL is from a trusted domain
        """
        url_lower = url.lower()
        for domain in self.TRUSTED_DOMAINS:
            if domain in url_lower:
                return True
        return False

    def _get_validation_system_prompt(self) -> str:
        """Get system prompt for validation tasks."""
        base_prompt = super().get_system_prompt()

        validation_instructions = """
## Validation Task
You are validating a research claim. Analyze the search results and determine if the claim is:
- VERIFIED: Multiple trusted sources confirm the claim
- PARTIALLY_VERIFIED: Some evidence supports it, but incomplete
- UNVERIFIED: No credible evidence found
- CONTRADICTED: Trusted sources contradict the claim

Respond with a JSON object:
{
    "status": "VERIFIED|PARTIALLY_VERIFIED|UNVERIFIED|CONTRADICTED",
    "confidence": 0.0-1.0,
    "reason": "Brief explanation",
    "sources": ["list of supporting source URLs"]
}"""

        return f"{base_prompt}\n{validation_instructions}"

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API for validation analysis."""
        if not self.settings.api_key:
            return json.dumps({
                "status": "UNVERIFIED",
                "confidence": 0.0,
                "reason": "API key not configured",
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
                    "model": self.settings.fast_model,  # Use fast model for validation
                    "max_tokens": 1024,
                    "system": self._get_validation_system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.settings.request_timeout,
            )

            if response.status_code != 200:
                return json.dumps({
                    "status": "UNVERIFIED",
                    "confidence": 0.0,
                    "reason": f"API error: {response.status_code}",
                    "sources": []
                })

            data = response.json()
            return data["content"][0]["text"]

    def _parse_validation_result(self, llm_response: str) -> dict[str, Any]:
        """Parse the LLM's validation response."""
        try:
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(llm_response[start:end])
        except json.JSONDecodeError:
            pass

        return {
            "status": "UNVERIFIED",
            "confidence": 0.0,
            "reason": "Failed to parse validation result",
            "sources": []
        }

    async def validate_claim(self, claim: str, original_source: str = "") -> dict[str, Any]:
        """Validate a single claim against trusted sources.

        Args:
            claim: The claim to validate
            original_source: The original source URL of the claim

        Returns:
            Validation result with status, confidence, reason, and sources
        """
        # Search for validation using trusted source keywords
        search_query = f"{claim} site:wikipedia.org OR site:edu OR site:gov OR site:nature.com"
        search_results = self.search_tool.search(search_query, num_results=8)

        # Filter to only trusted sources
        trusted_results = []
        for result in search_results:
            if self._is_trusted_source(result.url):
                trusted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                })

        if not trusted_results:
            # Try a broader search
            search_results = self.search_tool.search(claim, num_results=10)
            for result in search_results:
                if self._is_trusted_source(result.url):
                    trusted_results.append({
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                    })

        # Build prompt for LLM analysis
        if trusted_results:
            sources_text = "\n".join(
                f"- [{r['title']}]({r['url']}): {r['snippet']}"
                for r in trusted_results
            )
            prompt = f"""Validate this claim:
"{claim}"

Original source: {original_source or 'Unknown'}

Search results from trusted sources:
{sources_text}

Analyze whether these trusted sources support, contradict, or are silent on this claim."""
        else:
            prompt = f"""Validate this claim:
"{claim}"

Original source: {original_source or 'Unknown'}

No results found from trusted sources (Wikipedia, .edu, .gov, scientific journals).

Based on the lack of corroborating evidence from authoritative sources, provide your assessment."""

        llm_response = await self._call_llm(prompt)
        result = self._parse_validation_result(llm_response)
        result["claim"] = claim
        result["original_source"] = original_source
        result["trusted_results_found"] = len(trusted_results)

        return result

    async def validate_findings(
        self,
        findings: list[dict[str, Any]],
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Validate a list of research findings.

        Args:
            findings: List of findings with 'title', 'snippet', 'url'
            min_confidence: Minimum confidence threshold to keep a finding

        Returns:
            Dictionary with validated, removed, and replacement_needed lists
        """
        validated = []
        removed = []
        replacement_needed = []

        for finding in findings:
            claim = f"{finding.get('title', '')}: {finding.get('snippet', '')}"
            original_url = finding.get("url", "")

            result = await self.validate_claim(claim, original_url)

            finding_with_validation = {
                **finding,
                "validation": result,
            }

            if result["status"] in ("VERIFIED", "PARTIALLY_VERIFIED") and result["confidence"] >= min_confidence:
                validated.append(finding_with_validation)
            else:
                removed.append(finding_with_validation)
                replacement_needed.append({
                    "original_claim": claim,
                    "removal_reason": result["reason"],
                    "status": result["status"],
                })

        return {
            "validated": validated,
            "removed": removed,
            "replacement_needed": replacement_needed,
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
