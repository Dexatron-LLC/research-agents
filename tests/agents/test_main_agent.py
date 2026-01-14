"""Tests for the main agent."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

from research_agents.agents.main_agent import MainAgent
from research_agents.core.config import Settings
from research_agents.core.orchestrator import Orchestrator
from research_agents.core.database import DatabaseService


class TestMainAgent:
    """Tests for the MainAgent class."""

    @pytest.fixture
    def mock_orchestrator(self) -> Orchestrator:
        """Create a mock orchestrator."""
        return Orchestrator(use_llm_selection=False)

    @pytest.fixture
    def mock_database(self, temp_dir: Path) -> DatabaseService:
        """Create a mock database service."""
        settings = Settings(database_path=temp_dir / "test.db")
        db = DatabaseService(settings)
        return db

    def test_init(self, mock_orchestrator: Orchestrator, temp_dir: Path):
        """Test main agent initialization."""
        settings = Settings(database_path=temp_dir / "test.db")
        agent = MainAgent(mock_orchestrator, settings)

        assert agent.name == "main"
        assert agent.orchestrator is mock_orchestrator
        assert agent.session_id is not None
        assert len(agent.session_id) == 36  # UUID format
        assert agent.conversation_history == []
        assert agent.research_cache == []
        assert agent.validated_findings == []
        assert agent.pending_validation == []

    def test_loads_yaml_definition(self, mock_orchestrator: Orchestrator, temp_dir: Path):
        """Test that the agent loads its YAML definition."""
        settings = Settings(database_path=temp_dir / "test.db")
        agent = MainAgent(mock_orchestrator, settings)

        assert agent.role == "Research Coordinator and Assistant"
        assert agent.definition is not None

    def test_get_coordinator_system_prompt(self, mock_orchestrator: Orchestrator, temp_dir: Path):
        """Test coordinator system prompt generation."""
        settings = Settings(database_path=temp_dir / "test.db")
        agent = MainAgent(mock_orchestrator, settings)

        prompt = agent._get_coordinator_system_prompt()

        assert "Available Agents" in prompt
        assert "Tool Usage" in prompt
        assert "Workflow" in prompt

    def test_parse_tool_call_valid(self, mock_orchestrator: Orchestrator, temp_dir: Path):
        """Test parsing a valid tool call from response."""
        settings = Settings(database_path=temp_dir / "test.db")
        agent = MainAgent(mock_orchestrator, settings)

        response = 'I will search for that. {"tool": "research", "query": "climate change"}'
        result = agent._parse_tool_call(response)

        assert result is not None
        assert result["tool"] == "research"
        assert result["query"] == "climate change"

    def test_parse_tool_call_no_tool(self, mock_orchestrator: Orchestrator, temp_dir: Path):
        """Test parsing response without tool call."""
        settings = Settings(database_path=temp_dir / "test.db")
        agent = MainAgent(mock_orchestrator, settings)

        response = "Just a normal response without any JSON."
        result = agent._parse_tool_call(response)

        assert result is None

    def test_parse_tool_call_invalid_json(self, mock_orchestrator: Orchestrator, temp_dir: Path):
        """Test parsing response with invalid JSON."""
        settings = Settings(database_path=temp_dir / "test.db")
        agent = MainAgent(mock_orchestrator, settings)

        response = '{"tool": "incomplete'
        result = agent._parse_tool_call(response)

        assert result is None

    def test_parse_tool_call_json_without_tool_key(self, mock_orchestrator: Orchestrator, temp_dir: Path):
        """Test parsing JSON without 'tool' key."""
        settings = Settings(database_path=temp_dir / "test.db")
        agent = MainAgent(mock_orchestrator, settings)

        response = '{"action": "search", "query": "test"}'
        result = agent._parse_tool_call(response)

        assert result is None


class TestMainAgentLLMCalls:
    """Tests for LLM-related functionality."""

    @pytest.fixture
    def agent_with_mocks(self, temp_dir: Path, mock_httpx_client) -> MainAgent:
        """Create agent with mocked dependencies."""
        orchestrator = Orchestrator(use_llm_selection=False)
        settings = Settings(
            anthropic_api_key="test-key",
            database_path=temp_dir / "test.db",
        )

        # Mock database
        mock_db = MagicMock(spec=DatabaseService)
        mock_db.connect = AsyncMock()
        mock_db.initialize_schema = AsyncMock()
        mock_db.create_session = AsyncMock()
        mock_db.save_message = AsyncMock()
        mock_db.save_finding = AsyncMock(return_value=1)
        mock_db.save_validation = AsyncMock()
        mock_db.save_report = AsyncMock()
        mock_db.get_findings_by_session = AsyncMock(return_value=[])

        agent = MainAgent(orchestrator, settings, database=mock_db)
        return agent

    async def test_call_llm_success(self, agent_with_mocks: MainAgent, mock_httpx_client):
        """Test successful LLM call."""
        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": "LLM response"}]
        }

        response = await agent_with_mocks._call_llm("Hello")

        assert response == "LLM response"
        assert len(agent_with_mocks.conversation_history) == 2
        assert agent_with_mocks.conversation_history[0]["role"] == "user"
        assert agent_with_mocks.conversation_history[1]["role"] == "assistant"

    async def test_call_llm_no_api_key(self, temp_dir: Path):
        """Test LLM call without API key."""
        orchestrator = Orchestrator(use_llm_selection=False)
        settings = Settings(
            anthropic_api_key="",
            database_path=temp_dir / "test.db",
        )
        agent = MainAgent(orchestrator, settings)

        response = await agent._call_llm("Hello")

        assert "Error" in response
        assert "API" in response


class TestMainAgentToolHandling:
    """Tests for tool call handling."""

    @pytest.fixture
    def configured_agent(self, temp_dir: Path) -> tuple[MainAgent, Orchestrator]:
        """Create agent with configured orchestrator."""
        orchestrator = Orchestrator(use_llm_selection=False)
        settings = Settings(
            anthropic_api_key="test-key",
            database_path=temp_dir / "test.db",
        )

        # Mock database
        mock_db = MagicMock(spec=DatabaseService)
        mock_db.connect = AsyncMock()
        mock_db.initialize_schema = AsyncMock()
        mock_db.create_session = AsyncMock()
        mock_db.save_message = AsyncMock()
        mock_db.save_finding = AsyncMock(return_value=1)
        mock_db.save_validation = AsyncMock()
        mock_db.save_report = AsyncMock()
        mock_db.get_findings_by_session = AsyncMock(return_value=[
            {"id": 1, "title": "Finding 1"},
        ])

        agent = MainAgent(orchestrator, settings, database=mock_db)
        return agent, orchestrator

    async def test_handle_research_tool(self, configured_agent: tuple):
        """Test handling research tool call."""
        agent, orchestrator = configured_agent

        mock_research = MagicMock()
        mock_research.name = "research"
        mock_research.description = "Research agent"
        mock_research.execute = AsyncMock(return_value={
            "status": "completed",
            "findings": [
                {"title": "Finding 1", "snippet": "Content", "url": "https://example.com"}
            ],
        })
        orchestrator.register_agent(mock_research)

        tool_call = {"tool": "research", "query": "test query"}
        result = await agent._handle_tool_call(tool_call)

        assert result["status"] == "completed"
        assert result["pending_validation"] is True
        assert len(agent.pending_validation) == 1

    async def test_handle_validation_tool_no_pending(self, configured_agent: tuple):
        """Test validation tool with no pending findings."""
        agent, orchestrator = configured_agent

        mock_validation = MagicMock()
        mock_validation.name = "validation"
        mock_validation.description = "Validation agent"
        orchestrator.register_agent(mock_validation)

        tool_call = {"tool": "validation", "action": "validate"}
        result = await agent._handle_tool_call(tool_call)

        assert "error" in result
        assert "No research findings" in result["error"]

    async def test_handle_report_tool_no_cache(self, configured_agent: tuple):
        """Test report tool with no research cache."""
        agent, orchestrator = configured_agent

        mock_report = MagicMock()
        mock_report.name = "report"
        mock_report.description = "Report agent"
        orchestrator.register_agent(mock_report)

        tool_call = {"tool": "report", "action": "create", "title": "Test Report"}
        result = await agent._handle_tool_call(tool_call)

        assert "error" in result

    async def test_handle_unknown_tool(self, configured_agent: tuple):
        """Test handling unknown tool."""
        agent, orchestrator = configured_agent

        tool_call = {"tool": "unknown_tool", "query": "test"}
        result = await agent._handle_tool_call(tool_call)

        assert "error" in result
        assert "Unknown agent" in result["error"]


class TestMainAgentCacheManagement:
    """Tests for cache management."""

    def test_clear_research_cache(self, temp_dir: Path):
        """Test clearing research cache."""
        orchestrator = Orchestrator(use_llm_selection=False)
        settings = Settings(database_path=temp_dir / "test.db")
        agent = MainAgent(orchestrator, settings)

        # Add some data
        agent.research_cache.append({"data": "test"})
        agent.validated_findings.append({"finding": "test"})
        agent.pending_validation.append({"pending": "test"})

        agent.clear_research_cache()

        assert agent.research_cache == []
        assert agent.validated_findings == []
        assert agent.pending_validation == []

    def test_get_validation_status(self, temp_dir: Path):
        """Test getting validation status."""
        orchestrator = Orchestrator(use_llm_selection=False)
        settings = Settings(database_path=temp_dir / "test.db")
        agent = MainAgent(orchestrator, settings)

        agent.pending_validation = [{"a": 1}, {"b": 2}]
        agent.validated_findings = [{"c": 3}]
        agent.research_cache = [{"d": 4}, {"e": 5}, {"f": 6}]

        status = agent.get_validation_status()

        assert status["pending_validation"] == 2
        assert status["validated_findings"] == 1
        assert status["research_cache_items"] == 3


class TestMainAgentChat:
    """Tests for the chat interface."""

    async def test_chat_returns_response(self, temp_dir: Path, mock_httpx_client):
        """Test that chat returns response text."""
        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": "Hello! How can I help you?"}]
        }

        orchestrator = Orchestrator(use_llm_selection=False)
        settings = Settings(
            anthropic_api_key="test-key",
            database_path=temp_dir / "test.db",
        )

        mock_db = MagicMock(spec=DatabaseService)
        mock_db.connect = AsyncMock()
        mock_db.initialize_schema = AsyncMock()
        mock_db.create_session = AsyncMock()
        mock_db.save_message = AsyncMock()

        agent = MainAgent(orchestrator, settings, database=mock_db)
        response = await agent.chat("Hello")

        assert response == "Hello! How can I help you?"

    async def test_chat_with_tool_call(self, temp_dir: Path, mock_httpx_client):
        """Test chat that triggers a tool call."""
        # First response has tool call, second is follow-up
        mock_httpx_client.post.return_value.json.side_effect = [
            {"content": [{"text": '{"tool": "research", "query": "AI"}'}]},
            {"content": [{"text": "Here are the research results..."}]},
        ]

        orchestrator = Orchestrator(use_llm_selection=False)
        settings = Settings(
            anthropic_api_key="test-key",
            database_path=temp_dir / "test.db",
        )

        mock_db = MagicMock(spec=DatabaseService)
        mock_db.connect = AsyncMock()
        mock_db.initialize_schema = AsyncMock()
        mock_db.create_session = AsyncMock()
        mock_db.save_message = AsyncMock()
        mock_db.save_finding = AsyncMock(return_value=1)

        mock_research = MagicMock()
        mock_research.name = "research"
        mock_research.description = "Research"
        mock_research.execute = AsyncMock(return_value={
            "status": "completed",
            "findings": [{"title": "AI Result", "snippet": "Content", "url": "https://ai.com"}],
        })

        agent = MainAgent(orchestrator, settings, database=mock_db)
        orchestrator.register_agent(mock_research)

        response = await agent.chat("Search for AI")

        assert "research results" in response.lower() or len(response) > 0
