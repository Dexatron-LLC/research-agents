"""Tests for the validation agent."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from research_agents.agents.validation_agent import ValidationAgent
from research_agents.core.config import Settings
from research_agents.tools.web_search import SearchResult


class TestValidationAgent:
    """Tests for the ValidationAgent class."""

    def test_init(self):
        """Test validation agent initialization."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
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
            agent = ValidationAgent()

        assert agent.role == "Senior Fact-Checker and Research Validator"
        assert agent.definition is not None

    def test_is_trusted_source(self):
        """Test trusted source detection."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            agent = ValidationAgent()

        assert agent._is_trusted_source("https://en.wikipedia.org/wiki/Test") is True
        assert agent._is_trusted_source("https://harvard.edu/research") is True
        assert agent._is_trusted_source("https://cdc.gov/health") is True
        assert agent._is_trusted_source("https://nature.com/articles") is True
        assert agent._is_trusted_source("https://random-blog.com") is False
        assert agent._is_trusted_source("https://example.com") is False

    def test_get_validation_system_prompt(self):
        """Test validation system prompt generation."""
        with patch("research_agents.agents.validation_agent.WebSearchTool"):
            agent = ValidationAgent()

        prompt = agent._get_validation_system_prompt()

        assert "Validation Task" in prompt
        assert "VERIFIED" in prompt
        assert "UNVERIFIED" in prompt
        assert "CONTRADICTED" in prompt
        assert "confidence" in prompt


class TestValidationAgentValidateClaim:
    """Tests for the validate_claim method."""

    async def test_validate_claim_verified(self, mock_httpx_client):
        """Test validating a claim that gets verified."""
        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                title="Wikipedia Article",
                url="https://en.wikipedia.org/wiki/Test",
                snippet="This confirms the claim.",
            )
        ]

        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "VERIFIED", "confidence": 0.9, "reason": "Confirmed", "sources": ["https://wikipedia.org"]}'}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)
            result = await agent.validate_claim("The sky is blue")

        assert result["status"] == "VERIFIED"
        assert result["confidence"] == 0.9
        assert result["claim"] == "The sky is blue"

    async def test_validate_claim_unverified(self, mock_httpx_client):
        """Test validating a claim that cannot be verified."""
        mock_search = MagicMock()
        mock_search.search.return_value = []  # No trusted results

        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "UNVERIFIED", "confidence": 0.1, "reason": "No sources found", "sources": []}'}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)
            result = await agent.validate_claim("Unverifiable claim")

        assert result["status"] == "UNVERIFIED"
        assert result["trusted_results_found"] == 0

    async def test_validate_claim_with_original_source(self, mock_httpx_client):
        """Test validating a claim with original source."""
        mock_search = MagicMock()
        mock_search.search.return_value = []

        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "UNVERIFIED", "confidence": 0.0, "reason": "No data", "sources": []}'}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)
            result = await agent.validate_claim(
                "Test claim",
                original_source="https://example.com"
            )

        assert result["original_source"] == "https://example.com"

    async def test_validate_claim_no_api_key(self):
        """Test validation without API key."""
        mock_search = MagicMock()
        mock_search.search.return_value = []

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="")
            agent = ValidationAgent(settings)
            result = await agent.validate_claim("Test claim")

        assert result["status"] == "UNVERIFIED"
        assert "API key" in result["reason"]


class TestValidationAgentValidateFindings:
    """Tests for the validate_findings method."""

    async def test_validate_findings_all_verified(
        self,
        mock_research_findings: list[dict[str, Any]],
        mock_httpx_client,
    ):
        """Test validating findings where all pass."""
        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult("Wikipedia", "https://wikipedia.org/test", "Confirmed")
        ]

        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "VERIFIED", "confidence": 0.85, "reason": "Confirmed", "sources": ["https://wikipedia.org"]}'}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)
            result = await agent.validate_findings(mock_research_findings)

        assert len(result["validated"]) == 2
        assert len(result["removed"]) == 0
        assert result["stats"]["validation_rate"] == 1.0

    async def test_validate_findings_some_removed(
        self,
        mock_research_findings: list[dict[str, Any]],
        mock_httpx_client,
    ):
        """Test validating findings where some are removed."""
        mock_search = MagicMock()
        mock_search.search.return_value = []

        # First call returns verified, second returns unverified
        mock_httpx_client.post.return_value.json.side_effect = [
            {"content": [{"text": '{"status": "VERIFIED", "confidence": 0.8, "reason": "OK", "sources": []}'}]},
            {"content": [{"text": '{"status": "UNVERIFIED", "confidence": 0.2, "reason": "No sources", "sources": []}'}]},
        ]

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)
            result = await agent.validate_findings(mock_research_findings)

        assert len(result["validated"]) == 1
        assert len(result["removed"]) == 1
        assert len(result["replacement_needed"]) == 1

    async def test_validate_findings_custom_confidence(
        self,
        mock_research_findings: list[dict[str, Any]],
        mock_httpx_client,
    ):
        """Test validating with custom confidence threshold."""
        mock_search = MagicMock()
        mock_search.search.return_value = []

        # Both return 0.6 confidence
        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "VERIFIED", "confidence": 0.6, "reason": "Partial", "sources": []}'}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)

            # With 0.5 threshold, both should pass
            result1 = await agent.validate_findings(mock_research_findings, min_confidence=0.5)
            assert len(result1["validated"]) == 2

    async def test_validate_findings_empty(self, mock_httpx_client):
        """Test validating empty findings list."""
        mock_search = MagicMock()

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)
            result = await agent.validate_findings([])

        assert result["validated"] == []
        assert result["removed"] == []
        assert result["stats"]["total"] == 0

    async def test_validate_findings_stats(
        self,
        mock_research_findings: list[dict[str, Any]],
        mock_httpx_client,
    ):
        """Test validation statistics."""
        mock_search = MagicMock()
        mock_search.search.return_value = []

        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "VERIFIED", "confidence": 0.8, "reason": "OK", "sources": []}'}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)
            result = await agent.validate_findings(mock_research_findings)

        stats = result["stats"]
        assert stats["total"] == 2
        assert stats["validated_count"] == 2
        assert stats["removed_count"] == 0
        assert stats["validation_rate"] == 1.0


class TestValidationAgentExecute:
    """Tests for the execute method."""

    async def test_execute_with_single_claim(self, mock_httpx_client):
        """Test execute with a single claim in context."""
        mock_search = MagicMock()
        mock_search.search.return_value = []

        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "VERIFIED", "confidence": 0.9, "reason": "OK", "sources": []}'}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)

            context = {"claim": "Test claim", "original_source": "https://example.com"}
            result = await agent.execute("validate", context)

        assert result["status"] == "completed"
        assert "validation" in result

    async def test_execute_with_findings_list(
        self,
        mock_research_findings: list[dict[str, Any]],
        mock_httpx_client,
    ):
        """Test execute with findings list in context."""
        mock_search = MagicMock()
        mock_search.search.return_value = []

        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "VERIFIED", "confidence": 0.8, "reason": "OK", "sources": []}'}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)

            context = {"findings": mock_research_findings}
            result = await agent.execute("validate findings", context)

        assert result["status"] == "completed"
        assert "validated" in result
        assert "removed" in result

    async def test_execute_task_as_claim(self, mock_httpx_client):
        """Test execute with task used as claim."""
        mock_search = MagicMock()
        mock_search.search.return_value = []

        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "UNVERIFIED", "confidence": 0.1, "reason": "No data", "sources": []}'}]
        }

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            settings = Settings(anthropic_api_key="test-key")
            agent = ValidationAgent(settings)

            result = await agent.execute("The earth is flat")

        assert result["status"] == "completed"
        assert "validation" in result

    async def test_close(self):
        """Test closing the agent."""
        mock_search = MagicMock()

        async def mock_close():
            pass

        mock_search.close = mock_close

        with patch("research_agents.agents.validation_agent.WebSearchTool", return_value=mock_search):
            agent = ValidationAgent()
            await agent.close()

        # Should not raise any exceptions
