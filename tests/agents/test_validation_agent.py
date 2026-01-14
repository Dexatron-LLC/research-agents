"""Tests for the validation agent."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agents.agents.validation_agent import ValidationAgent
from research_agents.core.config import Settings
from research_agents.tools.web_search import SearchResult


class TestValidationAgent:
    """Tests for the ValidationAgent class."""

    def test_init(self):
        """Test validation agent initialization."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        assert agent.name == "validation"
        assert agent.search_tool is not None

    def test_trusted_domains(self):
        """Test that trusted domains are defined."""
        assert len(ValidationAgent.TRUSTED_DOMAINS) > 0
        assert "wikipedia.org" in ValidationAgent.TRUSTED_DOMAINS
        assert ".edu" in ValidationAgent.TRUSTED_DOMAINS
        assert ".gov" in ValidationAgent.TRUSTED_DOMAINS

    def test_loads_yaml_definition(self):
        """Test that the agent loads its YAML definition."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        assert agent.role == "Senior Fact-Checker and Research Validator"
        assert agent.definition is not None

    def test_is_trusted_source(self):
        """Test trusted source detection."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        assert agent._is_trusted_source("https://en.wikipedia.org/wiki/Test") is True
        assert agent._is_trusted_source("https://harvard.edu/research") is True
        assert agent._is_trusted_source("https://cdc.gov/health") is True
        assert agent._is_trusted_source("https://nature.com/articles") is True
        assert agent._is_trusted_source("https://random-blog.com") is False
        assert agent._is_trusted_source("https://example.com") is False

    def test_get_trust_level(self):
        """Test trust level classification."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        # High trust sources
        assert agent._get_trust_level("https://harvard.edu/research") == "high"
        assert agent._get_trust_level("https://cdc.gov/health") == "high"
        assert agent._get_trust_level("https://nature.com/articles") == "high"
        assert agent._get_trust_level("https://pubmed.ncbi.nlm.nih.gov/123") == "high"

        # Medium trust sources
        assert agent._get_trust_level("https://en.wikipedia.org/wiki/Test") == "medium"
        assert agent._get_trust_level("https://reuters.com/article") == "medium"
        assert agent._get_trust_level("https://bbc.com/news") == "medium"

        # Low trust sources
        assert agent._get_trust_level("https://random-blog.com") == "low"
        assert agent._get_trust_level("https://example.com") == "low"

    def test_get_validation_system_prompt(self):
        """Test validation system prompt generation."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        prompt = agent._get_validation_system_prompt()

        assert "Validation Task" in prompt
        assert "VERIFIED" in prompt
        assert "UNVERIFIED" in prompt
        assert "CONTRADICTED" in prompt
        assert "confidence" in prompt
        assert "evidence" in prompt

    def test_get_content_analysis_prompt(self):
        """Test content analysis prompt."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        prompt = agent._get_content_analysis_prompt()
        assert "source content" in prompt.lower()
        assert "evidence" in prompt.lower()


class TestValidationAgentLLM:
    """Tests for LLM-related functionality."""

    async def test_call_llm_no_api_key(self):
        """Test _call_llm returns error when no API key."""
        mock_settings = MagicMock()
        mock_settings.api_key = ""

        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            agent = ValidationAgent(settings=mock_settings)
            result = await agent._call_llm("test prompt")

        import json
        parsed = json.loads(result)
        # On errors, we now return PARTIALLY_VERIFIED to be less strict
        assert parsed["status"] == "PARTIALLY_VERIFIED"
        assert "API key" in parsed["reason"]

    async def test_call_llm_success(self):
        """Test _call_llm with successful API response."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"
        mock_settings.fast_model = "test-model"
        mock_settings.request_timeout = 60
        mock_settings.get_api_url.return_value = "https://api.test.com/v1/messages"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "LLM response text"}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
                agent = ValidationAgent(settings=mock_settings)
                result = await agent._call_llm("test prompt")

        assert result == "LLM response text"


class TestValidationAgentContentAnalysis:
    """Tests for content fetching and analysis."""

    async def test_fetch_page_content_success(self):
        """Test fetching page content."""
        mock_settings = MagicMock()
        mock_tool = MagicMock()
        mock_tool.fetch_page = AsyncMock(return_value={
            "url": "https://test.com",
            "title": "Test Page",
            "content": "Page content here",
        })

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            result = await agent._fetch_page_content("https://test.com")

        assert result is not None
        assert result["content"] == "Page content here"

    async def test_fetch_page_content_error(self):
        """Test fetching page content handles errors."""
        mock_settings = MagicMock()
        mock_tool = MagicMock()
        mock_tool.fetch_page = AsyncMock(side_effect=Exception("Network error"))

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            result = await agent._fetch_page_content("https://test.com")

        assert result is None

    async def test_analyze_source_content(self):
        """Test analyzing source content for evidence."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        llm_response = """RELEVANCE: high
SUPPORTS_CLAIM: yes
EVIDENCE:
- The source confirms this fact with data
- Statistics show 90% accuracy
ANALYSIS: This source strongly supports the claim with multiple data points."""

        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            agent = ValidationAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value=llm_response)

            result = await agent._analyze_source_content(
                claim="Test claim",
                url="https://harvard.edu/article",
                title="Harvard Research",
                content="Test content",
            )

        assert result["url"] == "https://harvard.edu/article"
        assert result["relevance"] == "high"
        assert result["supports_claim"] == "yes"
        assert len(result["evidence"]) == 2
        assert result["trust_level"] == "high"


class TestValidationAgentValidateClaim:
    """Tests for the validate_claim method."""

    async def test_validate_claim_with_page_analysis(self):
        """Test validating a claim with full page analysis."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = [
            SearchResult("Wikipedia", "https://en.wikipedia.org/wiki/Test", "Confirms the claim"),
            SearchResult("Harvard", "https://harvard.edu/test", "Academic source"),
        ]
        mock_tool.fetch_page = AsyncMock(return_value={
            "url": "https://test.com",
            "content": "Page content with evidence",
        })

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)

            # Mock the analysis methods
            agent._analyze_source_content = AsyncMock(return_value={
                "url": "https://test.com",
                "title": "Test",
                "relevance": "high",
                "supports_claim": "yes",
                "evidence": ["Supporting evidence"],
                "analysis": "Analysis text",
                "trust_level": "high",
            })
            agent._call_llm = AsyncMock(return_value='{"status": "VERIFIED", "confidence": 0.9, "reason": "Confirmed", "evidence": ["proof"], "sources": ["https://wikipedia.org"]}')

            result = await agent.validate_claim("The sky is blue")

        assert result["status"] == "VERIFIED"
        assert result["confidence"] == 0.9
        assert result["claim"] == "The sky is blue"
        assert result["sources_analyzed"] > 0

    async def test_validate_claim_no_trusted_results(self):
        """Test validating when no trusted sources found."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value='{"status": "UNVERIFIED", "confidence": 0.1, "reason": "No sources", "evidence": [], "sources": []}')

            result = await agent.validate_claim("Unverifiable claim")

        assert result["status"] == "UNVERIFIED"
        assert result["sources_analyzed"] == 0

    async def test_validate_claim_with_original_source(self):
        """Test validating a claim with original source."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value='{"status": "UNVERIFIED", "confidence": 0.0, "reason": "No data", "evidence": [], "sources": []}')

            result = await agent.validate_claim(
                "Test claim",
                original_source="https://example.com"
            )

        assert result["original_source"] == "https://example.com"


class TestValidationAgentValidateFindings:
    """Tests for the validate_findings method."""

    async def test_validate_findings_all_verified(
        self,
        mock_research_findings: list[dict[str, Any]],
    ):
        """Test validating findings where all pass."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = [
            SearchResult("Wikipedia", "https://wikipedia.org/test", "Confirmed")
        ]
        mock_tool.fetch_page = AsyncMock(return_value={"content": "test content"})

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            agent._analyze_source_content = AsyncMock(return_value={
                "url": "url", "title": "t", "relevance": "high",
                "supports_claim": "yes", "evidence": [], "analysis": "",
                "trust_level": "medium"
            })
            agent._call_llm = AsyncMock(return_value='{"status": "VERIFIED", "confidence": 0.85, "reason": "Confirmed", "evidence": [], "sources": []}')

            result = await agent.validate_findings(mock_research_findings)

        assert len(result["validated"]) == 2
        assert len(result["removed"]) == 0
        assert result["stats"]["validation_rate"] == 1.0

    async def test_validate_findings_some_removed(
        self,
        mock_research_findings: list[dict[str, Any]],
    ):
        """Test validating findings where some are removed."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)

            # First call returns verified, second returns unverified
            agent._call_llm = AsyncMock(side_effect=[
                '{"status": "VERIFIED", "confidence": 0.8, "reason": "OK", "evidence": [], "sources": []}',
                '{"status": "UNVERIFIED", "confidence": 0.2, "reason": "No sources", "evidence": [], "sources": []}',
            ])

            result = await agent.validate_findings(mock_research_findings)

        assert len(result["validated"]) == 1
        assert len(result["removed"]) == 1

    async def test_validate_findings_custom_confidence(
        self,
        mock_research_findings: list[dict[str, Any]],
    ):
        """Test validating with custom confidence threshold."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value='{"status": "VERIFIED", "confidence": 0.6, "reason": "Partial", "evidence": [], "sources": []}')

            # With 0.5 threshold, both should pass
            result = await agent.validate_findings(mock_research_findings, min_confidence=0.5)
            assert len(result["validated"]) == 2

    async def test_validate_findings_empty(self):
        """Test validating empty findings list."""
        mock_settings = MagicMock()

        mock_tool = MagicMock()

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            result = await agent.validate_findings([])

        assert result["validated"] == []
        assert result["removed"] == []
        assert result["stats"]["total"] == 0

    async def test_validate_findings_stats(
        self,
        mock_research_findings: list[dict[str, Any]],
    ):
        """Test validation statistics."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value='{"status": "VERIFIED", "confidence": 0.8, "reason": "OK", "evidence": [], "sources": []}')

            result = await agent.validate_findings(mock_research_findings)

        stats = result["stats"]
        assert stats["total"] == 2
        assert stats["validated_count"] == 2
        assert stats["removed_count"] == 0
        assert stats["validation_rate"] == 1.0


class TestValidationAgentExecute:
    """Tests for the execute method."""

    async def test_execute_with_single_claim(self):
        """Test execute with a single claim in context."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value='{"status": "VERIFIED", "confidence": 0.9, "reason": "OK", "evidence": [], "sources": []}')

            context = {"claim": "Test claim", "original_source": "https://example.com"}
            result = await agent.execute("validate", context)

        assert result["status"] == "completed"
        assert "validation" in result

    async def test_execute_with_findings_list(
        self,
        mock_research_findings: list[dict[str, Any]],
    ):
        """Test execute with findings list in context."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value='{"status": "VERIFIED", "confidence": 0.8, "reason": "OK", "evidence": [], "sources": []}')

            context = {"findings": mock_research_findings}
            result = await agent.execute("validate findings", context)

        assert result["status"] == "completed"
        assert "validated" in result
        assert "removed" in result

    async def test_execute_task_as_claim(self):
        """Test execute with task used as claim."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value='{"status": "UNVERIFIED", "confidence": 0.1, "reason": "No data", "evidence": [], "sources": []}')

            result = await agent.execute("The earth is flat")

        assert result["status"] == "completed"
        assert "validation" in result

    async def test_close(self):
        """Test closing the agent."""
        mock_settings = MagicMock()
        mock_tool = MagicMock()
        mock_tool.close = AsyncMock()

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_tool):
            agent = ValidationAgent(settings=mock_settings)
            await agent.close()

        mock_tool.close.assert_called_once()

    def test_repr(self):
        """Test string representation."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        repr_str = repr(agent)
        assert "ValidationAgent" in repr_str
        assert "validation" in repr_str


class TestValidationAgentParseResult:
    """Tests for parsing validation results."""

    def test_parse_validation_result_valid_json(self):
        """Test parsing valid JSON response."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        response = '{"status": "VERIFIED", "confidence": 0.9, "reason": "OK", "evidence": ["fact1"], "sources": ["url1"]}'
        result = agent._parse_validation_result(response)

        assert result["status"] == "VERIFIED"
        assert result["confidence"] == 0.9
        assert result["evidence"] == ["fact1"]

    def test_parse_validation_result_with_text_around_json(self):
        """Test parsing JSON embedded in text."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        response = 'Here is my analysis:\n{"status": "VERIFIED", "confidence": 0.8, "reason": "OK", "sources": []}\nEnd of response.'
        result = agent._parse_validation_result(response)

        assert result["status"] == "VERIFIED"
        assert "evidence" in result  # Should add empty evidence if missing

    def test_parse_validation_result_invalid_json(self):
        """Test parsing invalid JSON response."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            with patch("research_agents.agents.validation_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ValidationAgent()

        response = 'This is not valid JSON at all'
        result = agent._parse_validation_result(response)

        # On parse errors, we now return PARTIALLY_VERIFIED to be less strict
        assert result["status"] == "PARTIALLY_VERIFIED"
        assert result["confidence"] == 0.5
        assert "parse" in result["reason"].lower()
