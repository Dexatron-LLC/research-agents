"""Tests for the research agent."""

from unittest.mock import MagicMock, patch

import pytest

from research_agents.agents.research_agent import ResearchAgent
from research_agents.tools.web_search import SearchResult


class TestResearchAgent:
    """Tests for the ResearchAgent class."""

    def test_init(self):
        """Test research agent initialization."""
        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent()

        assert agent.name == "research"
        assert agent.search_tool is not None

    def test_inherits_from_base_agent(self):
        """Test that ResearchAgent inherits from BaseAgent."""
        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent()

        # Should have base agent properties
        assert hasattr(agent, "role")
        assert hasattr(agent, "goal")
        assert hasattr(agent, "backstory")
        assert hasattr(agent, "get_system_prompt")

    def test_loads_yaml_definition(self):
        """Test that the agent loads its YAML definition."""
        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent()

        # Should have loaded from research.yaml
        assert agent.role == "Senior Research Analyst"
        assert agent.definition is not None

    async def test_execute_returns_findings(self, mock_search_results: list[SearchResult]):
        """Test execute returns structured findings."""
        mock_tool = MagicMock()
        mock_tool.search.return_value = mock_search_results

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent()
            result = await agent.execute("test query")

        assert result["status"] == "completed"
        assert result["task"] == "test query"
        assert len(result["findings"]) == 3
        assert len(result["sources"]) == 3

    async def test_execute_finding_structure(self, mock_search_results: list[SearchResult]):
        """Test that findings have correct structure."""
        mock_tool = MagicMock()
        mock_tool.search.return_value = mock_search_results

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent()
            result = await agent.execute("test query")

        finding = result["findings"][0]
        assert "title" in finding
        assert "snippet" in finding
        assert "url" in finding
        assert finding["title"] == "Test Result 1"
        assert finding["url"] == "https://example.com/1"

    async def test_execute_with_context(self, mock_search_results: list[SearchResult]):
        """Test execute with context parameter."""
        mock_tool = MagicMock()
        mock_tool.search.return_value = mock_search_results

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent()
            result = await agent.execute("test query", context={"key": "value"})

        # Context shouldn't affect results for research agent
        assert result["status"] == "completed"

    async def test_execute_with_no_results(self):
        """Test execute when search returns no results."""
        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent()
            result = await agent.execute("obscure query")

        assert result["status"] == "completed"
        assert result["findings"] == []
        assert result["sources"] == []

    async def test_execute_calls_search_tool(self, mock_search_results: list[SearchResult]):
        """Test that execute calls the search tool correctly."""
        mock_tool = MagicMock()
        mock_tool.search.return_value = mock_search_results

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent()
            await agent.execute("my search query")

        mock_tool.search.assert_called_once_with("my search query", num_results=5)

    async def test_close(self):
        """Test closing the agent cleans up resources."""
        mock_tool = MagicMock()
        mock_tool.close = MagicMock(return_value=None)
        # Make close a coroutine
        async def mock_close():
            pass
        mock_tool.close = mock_close

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent()
            await agent.close()

        # Just verify no exception was raised

    def test_repr(self):
        """Test string representation."""
        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent()

        repr_str = repr(agent)
        assert "ResearchAgent" in repr_str
        assert "research" in repr_str
