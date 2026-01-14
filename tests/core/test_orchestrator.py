"""Tests for the orchestrator module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_agents.core.base_agent import BaseAgent
from research_agents.core.orchestrator import Orchestrator


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, name: str, description: str = "Mock agent"):
        super().__init__(name=name, description=description, load_definition=False)
        self.execute_mock = AsyncMock(return_value={"status": "completed"})

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.execute_mock(task, context)


class TestOrchestrator:
    """Tests for the Orchestrator class."""

    def test_init_default(self):
        """Test default orchestrator initialization."""
        orchestrator = Orchestrator()
        assert orchestrator.use_llm_selection is True
        assert orchestrator.agents == {}
        assert orchestrator.execution_history == []

    def test_init_without_llm(self):
        """Test orchestrator without LLM selection."""
        orchestrator = Orchestrator(use_llm_selection=False)
        assert orchestrator.use_llm_selection is False

    def test_register_agent(self, mock_orchestrator: Orchestrator):
        """Test registering an agent."""
        agent = MockAgent("test")
        mock_orchestrator.register_agent(agent)

        assert "test" in mock_orchestrator.agents
        assert mock_orchestrator.agents["test"] is agent

    def test_unregister_agent(self, mock_orchestrator: Orchestrator):
        """Test unregistering an agent."""
        agent = MockAgent("test")
        mock_orchestrator.register_agent(agent)

        result = mock_orchestrator.unregister_agent("test")

        assert result is True
        assert "test" not in mock_orchestrator.agents

    def test_unregister_nonexistent_agent(self, mock_orchestrator: Orchestrator):
        """Test unregistering a nonexistent agent."""
        result = mock_orchestrator.unregister_agent("nonexistent")
        assert result is False

    def test_get_agent(self, mock_orchestrator: Orchestrator):
        """Test getting an agent by name."""
        agent = MockAgent("test")
        mock_orchestrator.register_agent(agent)

        retrieved = mock_orchestrator.get_agent("test")

        assert retrieved is agent

    def test_get_nonexistent_agent(self, mock_orchestrator: Orchestrator):
        """Test getting a nonexistent agent."""
        result = mock_orchestrator.get_agent("nonexistent")
        assert result is None

    def test_list_agents(self, mock_orchestrator: Orchestrator):
        """Test listing all agents."""
        mock_orchestrator.register_agent(MockAgent("agent1"))
        mock_orchestrator.register_agent(MockAgent("agent2"))
        mock_orchestrator.register_agent(MockAgent("agent3"))

        agents = mock_orchestrator.list_agents()

        assert len(agents) == 3
        assert "agent1" in agents
        assert "agent2" in agents
        assert "agent3" in agents

    def test_get_agent_descriptions(self, mock_orchestrator: Orchestrator):
        """Test getting agent descriptions."""
        mock_orchestrator.register_agent(MockAgent("agent1", "Description 1"))
        mock_orchestrator.register_agent(MockAgent("agent2", "Description 2"))

        descriptions = mock_orchestrator.get_agent_descriptions()

        assert descriptions["agent1"] == "Description 1"
        assert descriptions["agent2"] == "Description 2"


class TestOrchestratorKeywordSelection:
    """Tests for keyword-based agent selection."""

    @pytest.fixture
    def orchestrator_with_agents(self, mock_orchestrator: Orchestrator):
        """Create orchestrator with standard agents."""
        mock_orchestrator.register_agent(MockAgent("research", "Research agent"))
        mock_orchestrator.register_agent(MockAgent("report", "Report agent"))
        mock_orchestrator.register_agent(MockAgent("validation", "Validation agent"))
        return mock_orchestrator

    def test_keyword_select_research(self, orchestrator_with_agents: Orchestrator):
        """Test keyword selection for research tasks."""
        result = orchestrator_with_agents._keyword_select("Search for information about AI")
        assert result == "research"

    def test_keyword_select_research_phrases(self, orchestrator_with_agents: Orchestrator):
        """Test various research-related phrases."""
        phrases = [
            "Find information about climate change",
            "Look up the latest news",
            "What is machine learning?",
            "Research quantum computing",
            "Tell me about renewable energy",
        ]
        for phrase in phrases:
            result = orchestrator_with_agents._keyword_select(phrase)
            assert result == "research", f"Failed for phrase: {phrase}"

    def test_keyword_select_report(self, orchestrator_with_agents: Orchestrator):
        """Test keyword selection for report tasks."""
        result = orchestrator_with_agents._keyword_select("Create a report from the research")
        assert result == "report"

    def test_keyword_select_report_phrases(self, orchestrator_with_agents: Orchestrator):
        """Test various report-related phrases."""
        phrases = [
            "Summarize the findings",
            "Generate a report",
            "Compile the data into a document",
        ]
        for phrase in phrases:
            result = orchestrator_with_agents._keyword_select(phrase)
            assert result == "report", f"Failed for phrase: {phrase}"

    def test_keyword_select_validation(self, orchestrator_with_agents: Orchestrator):
        """Test keyword selection for validation tasks."""
        result = orchestrator_with_agents._keyword_select("Validate this claim")
        assert result == "validation"

    def test_keyword_select_validation_phrases(self, orchestrator_with_agents: Orchestrator):
        """Test various validation-related phrases."""
        phrases = [
            "Verify this information",
            "Fact-check the article",
            "Is this claim credible?",
            "Check if this is true",
        ]
        for phrase in phrases:
            result = orchestrator_with_agents._keyword_select(phrase)
            assert result == "validation", f"Failed for phrase: {phrase}"

    def test_keyword_select_no_match(self, orchestrator_with_agents: Orchestrator):
        """Test when no keywords match."""
        result = orchestrator_with_agents._keyword_select("Hello, how are you?")
        assert result is None

    def test_keyword_select_excludes_main(self, mock_orchestrator: Orchestrator):
        """Test that main agent is excluded from selection."""
        mock_orchestrator.register_agent(MockAgent("main", "Main agent"))
        mock_orchestrator.register_agent(MockAgent("research", "Research agent"))

        result = mock_orchestrator._keyword_select("Search for something")
        assert result == "research"
        assert result != "main"


class TestOrchestratorRun:
    """Tests for running tasks through the orchestrator."""

    async def test_run_with_specified_agent(self, mock_orchestrator: Orchestrator):
        """Test running a task with a specified agent."""
        agent = MockAgent("test")
        mock_orchestrator.register_agent(agent)

        result = await mock_orchestrator.run("test task", agent_name="test")

        assert result["status"] == "completed"
        agent.execute_mock.assert_called_once()

    async def test_run_with_auto_select(self, mock_orchestrator: Orchestrator):
        """Test running a task with auto-selection."""
        agent = MockAgent("research", "Research agent")
        mock_orchestrator.register_agent(agent)

        result = await mock_orchestrator.run("Search for AI information")

        assert result["status"] == "completed"
        agent.execute_mock.assert_called_once()

    async def test_run_no_suitable_agent(self, mock_orchestrator: Orchestrator):
        """Test when no suitable agent is found."""
        result = await mock_orchestrator.run("Hello world")

        assert result["status"] == "error"
        assert "No suitable agent" in result["error"]

    async def test_run_agent_not_found(self, mock_orchestrator: Orchestrator):
        """Test when specified agent doesn't exist."""
        result = await mock_orchestrator.run("task", agent_name="nonexistent")

        assert result["status"] == "error"
        assert "not found" in result["error"]

    async def test_run_records_history(self, mock_orchestrator: Orchestrator):
        """Test that execution is recorded in history."""
        agent = MockAgent("test")
        mock_orchestrator.register_agent(agent)

        await mock_orchestrator.run("test task", agent_name="test")

        history = mock_orchestrator.get_execution_history()
        assert len(history) == 1
        assert history[0]["task"] == "test task"
        assert history[0]["agent"] == "test"
        assert history[0]["auto_selected"] is False

    async def test_run_with_context(self, mock_orchestrator: Orchestrator):
        """Test running with context passed to agent."""
        agent = MockAgent("test")
        mock_orchestrator.register_agent(agent)

        context = {"key": "value"}
        await mock_orchestrator.run("task", agent_name="test", context=context)

        agent.execute_mock.assert_called_with("task", context)


class TestOrchestratorPipeline:
    """Tests for pipeline execution."""

    async def test_run_pipeline_single_task(self, mock_orchestrator: Orchestrator):
        """Test pipeline with a single task."""
        agent = MockAgent("test")
        mock_orchestrator.register_agent(agent)

        tasks = [{"task": "test task", "agent": "test"}]
        results = await mock_orchestrator.run_pipeline(tasks)

        assert len(results) == 1
        assert results[0]["status"] == "completed"

    async def test_run_pipeline_multiple_tasks(self, mock_orchestrator: Orchestrator):
        """Test pipeline with multiple tasks."""
        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2")
        mock_orchestrator.register_agent(agent1)
        mock_orchestrator.register_agent(agent2)

        tasks = [
            {"task": "task 1", "agent": "agent1"},
            {"task": "task 2", "agent": "agent2"},
        ]
        results = await mock_orchestrator.run_pipeline(tasks)

        assert len(results) == 2
        assert all(r["status"] == "completed" for r in results)

    async def test_run_pipeline_accumulates_context(self, mock_orchestrator: Orchestrator):
        """Test that pipeline accumulates context from results."""
        agent = MockAgent("test")
        agent.execute_mock.return_value = {
            "status": "completed",
            "data": "test data",
        }
        mock_orchestrator.register_agent(agent)

        tasks = [
            {"task": "task 1", "agent": "test"},
            {"task": "task 2", "agent": "test"},
        ]
        await mock_orchestrator.run_pipeline(tasks)

        # Second call should have previous_result in context
        calls = agent.execute_mock.call_args_list
        assert len(calls) == 2
        second_call_context = calls[1][0][1]
        assert "previous_result" in second_call_context


class TestOrchestratorLLMSelection:
    """Tests for LLM-based agent selection."""

    @pytest.fixture
    def orchestrator_with_api_key(self, temp_dir):
        """Create orchestrator with API key configured."""
        from research_agents.core.config import Settings
        settings = Settings(
            anthropic_api_key="test-api-key",
            database_path=temp_dir / "test.db",
        )
        orchestrator = Orchestrator(use_llm_selection=True, settings=settings)
        orchestrator.register_agent(MockAgent("research", "Research agent"))
        orchestrator.register_agent(MockAgent("validation", "Validation agent"))
        return orchestrator

    async def test_llm_select_no_api_key(self, temp_dir):
        """Test LLM selection returns None without API key."""
        from research_agents.core.config import Settings
        settings = Settings(
            anthropic_api_key="",
            database_path=temp_dir / "test.db",
        )
        orchestrator = Orchestrator(settings=settings)
        result = await orchestrator._llm_select("Search for AI")
        assert result is None

    async def test_llm_select_no_selectable_agents(self, temp_dir):
        """Test LLM selection with no selectable agents."""
        from research_agents.core.config import Settings
        settings = Settings(
            anthropic_api_key="test-key",
            database_path=temp_dir / "test.db",
        )
        orchestrator = Orchestrator(settings=settings)
        # Only register main agent (which is excluded)
        orchestrator.register_agent(MockAgent("main", "Main agent"))

        result = await orchestrator._llm_select("Search for AI")
        assert result is None

    async def test_llm_select_successful(self, orchestrator_with_api_key, mocker):
        """Test successful LLM agent selection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": '{"agent": "research", "reason": "Search task"}'}]
        }

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await orchestrator_with_api_key._llm_select("Search for AI information")
        assert result == "research"

    async def test_llm_select_invalid_agent(self, orchestrator_with_api_key, mocker):
        """Test LLM selection with invalid agent name."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": '{"agent": "nonexistent", "reason": "No reason"}'}]
        }

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await orchestrator_with_api_key._llm_select("Do something")
        assert result is None

    async def test_llm_select_null_agent(self, orchestrator_with_api_key, mocker):
        """Test LLM selection returning null agent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": '{"agent": null, "reason": "No suitable agent"}'}]
        }

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await orchestrator_with_api_key._llm_select("Random task")
        assert result is None

    async def test_llm_select_api_error(self, orchestrator_with_api_key, mocker):
        """Test LLM selection handles API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await orchestrator_with_api_key._llm_select("Search for AI")
        assert result is None

    async def test_llm_select_invalid_json(self, orchestrator_with_api_key, mocker):
        """Test LLM selection handles invalid JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "Not valid JSON at all"}]
        }

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await orchestrator_with_api_key._llm_select("Search for AI")
        assert result is None

    async def test_llm_select_request_error(self, orchestrator_with_api_key, mocker):
        """Test LLM selection handles request errors."""
        import httpx

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=httpx.RequestError("Connection failed"))

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await orchestrator_with_api_key._llm_select("Search for AI")
        assert result is None

    async def test_select_agent_uses_llm_when_enabled(self, orchestrator_with_api_key, mocker):
        """Test select_agent uses LLM selection when enabled."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": '{"agent": "validation", "reason": "Validation needed"}'}]
        }

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await orchestrator_with_api_key.select_agent("Check if claim is true")
        assert result == "validation"

    async def test_select_agent_falls_back_to_keyword(self, orchestrator_with_api_key, mocker):
        """Test select_agent falls back to keyword matching when LLM fails."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await orchestrator_with_api_key.select_agent("Search for AI information")
        assert result == "research"  # Falls back to keyword match


class TestOrchestratorHistory:
    """Tests for execution history management."""

    async def test_get_execution_history(self, mock_orchestrator: Orchestrator):
        """Test getting execution history."""
        agent = MockAgent("test")
        mock_orchestrator.register_agent(agent)

        await mock_orchestrator.run("task 1", agent_name="test")
        await mock_orchestrator.run("task 2", agent_name="test")

        history = mock_orchestrator.get_execution_history()

        assert len(history) == 2
        assert history[0]["task"] == "task 1"
        assert history[1]["task"] == "task 2"

    async def test_execution_history_is_copy(self, mock_orchestrator: Orchestrator):
        """Test that get_execution_history returns a copy."""
        agent = MockAgent("test")
        mock_orchestrator.register_agent(agent)
        await mock_orchestrator.run("task", agent_name="test")

        history = mock_orchestrator.get_execution_history()
        history.clear()

        # Original should be unchanged
        assert len(mock_orchestrator.get_execution_history()) == 1

    def test_clear_execution_history(self, mock_orchestrator: Orchestrator):
        """Test clearing execution history."""
        mock_orchestrator.execution_history = [{"task": "old task"}]
        mock_orchestrator.clear_execution_history()

        assert mock_orchestrator.execution_history == []
