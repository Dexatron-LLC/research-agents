"""Tests for the research agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agents.agents.research_agent import ResearchAgent
from research_agents.tools.web_search import SearchResult


class TestResearchAgent:
    """Tests for the ResearchAgent class."""

    def test_init(self):
        """Test research agent initialization."""
        with patch("research_agents.agents.research_agent.WebSearchTool"):
            with patch("research_agents.agents.research_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ResearchAgent()

        assert agent.name == "research"
        assert agent.search_tool is not None

    def test_inherits_from_base_agent(self):
        """Test that ResearchAgent inherits from BaseAgent."""
        with patch("research_agents.agents.research_agent.WebSearchTool"):
            with patch("research_agents.agents.research_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ResearchAgent()

        # Should have base agent properties
        assert hasattr(agent, "role")
        assert hasattr(agent, "goal")
        assert hasattr(agent, "backstory")
        assert hasattr(agent, "get_system_prompt")

    def test_loads_yaml_definition(self):
        """Test that the agent loads its YAML definition."""
        with patch("research_agents.agents.research_agent.WebSearchTool"):
            with patch("research_agents.agents.research_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ResearchAgent()

        # Should have loaded from research.yaml
        assert agent.role == "Senior Research Analyst"
        assert agent.definition is not None

    def test_get_research_system_prompt(self):
        """Test getting the research system prompt."""
        with patch("research_agents.agents.research_agent.WebSearchTool"):
            with patch("research_agents.agents.research_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ResearchAgent()

        prompt = agent._get_research_system_prompt()
        assert "Research Analysis Guidelines" in prompt
        assert "Extract key facts" in prompt


class TestResearchAgentLLM:
    """Tests for LLM-related functionality."""

    async def test_call_llm_no_api_key(self):
        """Test _call_llm returns error when no API key."""
        mock_settings = MagicMock()
        mock_settings.api_key = ""

        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent(settings=mock_settings)
            result = await agent._call_llm("test prompt")

        assert "Error" in result
        assert "ANTHROPIC_API_KEY" in result

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

        with patch("research_agents.agents.research_agent.WebSearchTool"):
            with patch("research_agents.agents.research_agent.httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
                agent = ResearchAgent(settings=mock_settings)
                result = await agent._call_llm("test prompt")

        assert result == "LLM response text"

    async def test_call_llm_api_error(self):
        """Test _call_llm handles API errors."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"
        mock_settings.fast_model = "test-model"
        mock_settings.request_timeout = 60
        mock_settings.get_api_url.return_value = "https://api.test.com/v1/messages"

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("research_agents.agents.research_agent.WebSearchTool"):
            with patch("research_agents.agents.research_agent.httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
                agent = ResearchAgent(settings=mock_settings)
                result = await agent._call_llm("test prompt")

        assert "Error" in result
        assert "500" in result


class TestResearchAgentSearchQueries:
    """Tests for search query generation."""

    async def test_generate_search_queries_success(self):
        """Test generating multiple search queries."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value="query 1\nquery 2\nquery 3")
            queries = await agent._generate_search_queries("test topic")

        assert len(queries) == 3
        assert "query 1" in queries

    async def test_generate_search_queries_fallback(self):
        """Test fallback when query generation fails."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value="Error: API failed")
            queries = await agent._generate_search_queries("fallback topic")

        assert queries == ["fallback topic"]


class TestResearchAgentContentAnalysis:
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

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent(settings=mock_settings)
            result = await agent._fetch_page_content("https://test.com")

        assert result is not None
        assert result["content"] == "Page content here"

    async def test_fetch_page_content_error(self):
        """Test fetching page content handles errors."""
        mock_settings = MagicMock()
        mock_tool = MagicMock()
        mock_tool.fetch_page = AsyncMock(side_effect=Exception("Network error"))

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent(settings=mock_settings)
            result = await agent._fetch_page_content("https://test.com")

        assert result is None

    async def test_analyze_content_success(self):
        """Test analyzing page content."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        llm_response = """FINDINGS:
- Finding one
- Finding two

STATISTICS:
- 50% of users

CREDIBILITY: high

SUMMARY:
This is a summary of the content."""

        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value=llm_response)

            result = await agent._analyze_content(
                url="https://test.com",
                title="Test Page",
                content="Test content",
                topic="test topic",
            )

        assert result["url"] == "https://test.com"
        assert result["title"] == "Test Page"
        assert len(result["findings"]) == 2
        assert "Finding one" in result["findings"]
        assert result["credibility"] == "high"
        assert "summary" in result["summary"].lower() or len(result["summary"]) > 0

    async def test_analyze_content_llm_error(self):
        """Test analyze_content handles LLM errors."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent(settings=mock_settings)
            agent._call_llm = AsyncMock(return_value="Error: API failed")

            result = await agent._analyze_content(
                url="https://test.com",
                title="Test Page",
                content="Test content",
                topic="test topic",
            )

        assert result["findings"] == []
        assert result["credibility"] == "unknown"


class TestResearchAgentSynthesis:
    """Tests for research synthesis."""

    async def test_synthesize_research(self):
        """Test synthesizing research from multiple sources."""
        mock_settings = MagicMock()

        analyzed_sources = [
            {
                "url": "https://source1.com",
                "title": "Source 1",
                "findings": ["Finding A", "Finding B"],
                "credibility": "high",
            },
            {
                "url": "https://source2.com",
                "title": "Source 2",
                "findings": ["Finding C"],
                "credibility": "medium",
            },
        ]

        with patch("research_agents.agents.research_agent.WebSearchTool"):
            agent = ResearchAgent(settings=mock_settings)
            findings = await agent._synthesize_research("test topic", analyzed_sources)

        assert len(findings) == 3
        assert findings[0]["snippet"] == "Finding A"
        assert findings[0]["source"] == "Source 1"
        assert findings[0]["credibility"] == "high"


class TestResearchAgentExecute:
    """Tests for the main execute method."""

    async def test_execute_full_workflow(self, mock_search_results: list[SearchResult]):
        """Test complete execute workflow."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = mock_search_results
        mock_tool.fetch_page = AsyncMock(return_value={
            "url": "https://test.com",
            "title": "Test",
            "content": "Content here",
        })

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent(settings=mock_settings)
            # Mock the LLM calls
            agent._generate_search_queries = AsyncMock(return_value=["query 1", "query 2"])
            agent._analyze_content = AsyncMock(return_value={
                "url": "https://test.com",
                "title": "Test",
                "findings": ["Finding 1"],
                "credibility": "high",
            })

            result = await agent.execute("test query")

        assert result["status"] == "completed"
        assert result["task"] == "test query"
        assert "findings" in result
        assert "sources" in result
        assert "queries_used" in result

    async def test_execute_with_depth_quick(self, mock_search_results: list[SearchResult]):
        """Test execute with quick depth setting."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = mock_search_results
        mock_tool.fetch_page = AsyncMock(return_value={"content": "test"})

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent(settings=mock_settings)
            agent._generate_search_queries = AsyncMock(return_value=["query"])
            agent._analyze_content = AsyncMock(return_value={
                "url": "url", "title": "t", "findings": [], "credibility": "medium"
            })

            result = await agent.execute("test", context={"depth": "quick"})

        assert result["depth"] == "quick"

    async def test_execute_with_depth_thorough(self, mock_search_results: list[SearchResult]):
        """Test execute with thorough depth setting."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = mock_search_results
        mock_tool.fetch_page = AsyncMock(return_value={"content": "test"})

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent(settings=mock_settings)
            agent._generate_search_queries = AsyncMock(return_value=["q1", "q2", "q3"])
            agent._analyze_content = AsyncMock(return_value={
                "url": "url", "title": "t", "findings": [], "credibility": "medium"
            })

            result = await agent.execute("test", context={"depth": "thorough"})

        assert result["depth"] == "thorough"

    async def test_execute_no_search_results(self):
        """Test execute when search returns no results."""
        mock_settings = MagicMock()
        mock_settings.api_key = "test-key"

        mock_tool = MagicMock()
        mock_tool.search.return_value = []

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent(settings=mock_settings)
            agent._generate_search_queries = AsyncMock(return_value=["query"])

            result = await agent.execute("obscure query")

        assert result["status"] == "completed"
        assert result["findings"] == []
        assert "No search results found" in result.get("message", "")

    async def test_close(self):
        """Test closing the agent cleans up resources."""
        mock_settings = MagicMock()
        mock_tool = MagicMock()
        mock_tool.close = AsyncMock()

        with patch("research_agents.agents.research_agent.WebSearchTool", return_value=mock_tool):
            agent = ResearchAgent(settings=mock_settings)
            await agent.close()

        mock_tool.close.assert_called_once()

    def test_repr(self):
        """Test string representation."""
        with patch("research_agents.agents.research_agent.WebSearchTool"):
            with patch("research_agents.agents.research_agent.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                agent = ResearchAgent()

        repr_str = repr(agent)
        assert "ResearchAgent" in repr_str
        assert "research" in repr_str
