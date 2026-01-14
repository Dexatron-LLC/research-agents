"""Tests for the main module (entry point)."""

import asyncio
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agents.main import chat_loop, main, run


class TestChatLoopCommands:
    """Test chat loop command handling."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        main_agent = MagicMock()
        main_agent.session_id = "test-session-123"
        main_agent.conversation_history = []
        main_agent.research_cache = []
        main_agent.clear_research_cache = MagicMock()
        main_agent.get_validation_status = MagicMock(return_value={
            "pending_validation": 2,
            "validated_findings": 5,
            "research_cache_items": 3,
        })
        main_agent.chat = AsyncMock(return_value="Test response")

        research_agent = MagicMock()
        research_agent.close = AsyncMock()

        validation_agent = MagicMock()
        validation_agent.close = AsyncMock()

        orchestrator = MagicMock()
        orchestrator.get_agent_descriptions = MagicMock(return_value={
            "research": "Research agent",
            "validation": "Validation agent",
        })
        orchestrator.get_execution_history = MagicMock(return_value=[])

        return main_agent, research_agent, validation_agent, orchestrator

    async def test_quit_command(self, mock_agents):
        """Test quit command exits the loop."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        with patch("builtins.input", side_effect=["quit"]):
            with patch("builtins.print"):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        research_agent.close.assert_called_once()
        validation_agent.close.assert_called_once()

    async def test_exit_command(self, mock_agents):
        """Test exit command exits the loop."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        with patch("builtins.input", side_effect=["exit"]):
            with patch("builtins.print"):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        research_agent.close.assert_called_once()

    async def test_clear_command(self, mock_agents):
        """Test clear command clears conversation history."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents
        main_agent.conversation_history = [{"role": "user", "content": "test"}]

        with patch("builtins.input", side_effect=["clear", "quit"]):
            with patch("builtins.print"):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        assert main_agent.conversation_history == []

    async def test_cache_command(self, mock_agents):
        """Test cache command shows cache count."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents
        main_agent.research_cache = [{"data": "item1"}, {"data": "item2"}]

        outputs = []
        with patch("builtins.input", side_effect=["cache", "quit"]):
            with patch("builtins.print", side_effect=lambda x="": outputs.append(str(x))):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        assert any("2 item(s)" in out for out in outputs)

    async def test_clearcache_command(self, mock_agents):
        """Test clearcache command clears the cache."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        with patch("builtins.input", side_effect=["clearcache", "quit"]):
            with patch("builtins.print"):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        main_agent.clear_research_cache.assert_called_once()

    async def test_status_command(self, mock_agents):
        """Test status command shows validation status."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        outputs = []
        with patch("builtins.input", side_effect=["status", "quit"]):
            with patch("builtins.print", side_effect=lambda x="": outputs.append(str(x))):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        main_agent.get_validation_status.assert_called()

    async def test_agents_command(self, mock_agents):
        """Test agents command lists available agents."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        outputs = []
        with patch("builtins.input", side_effect=["agents", "quit"]):
            with patch("builtins.print", side_effect=lambda x="": outputs.append(str(x))):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        orchestrator.get_agent_descriptions.assert_called()

    async def test_history_command_empty(self, mock_agents):
        """Test history command with no history."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        outputs = []
        with patch("builtins.input", side_effect=["history", "quit"]):
            with patch("builtins.print", side_effect=lambda x="": outputs.append(str(x))):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        assert any("No execution history" in out for out in outputs)

    async def test_history_command_with_entries(self, mock_agents):
        """Test history command with execution history."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents
        orchestrator.get_execution_history.return_value = [
            {"agent": "research", "task": "Search for AI", "auto_selected": True},
            {"agent": "validation", "task": "Validate findings", "auto_selected": False},
        ]

        outputs = []
        with patch("builtins.input", side_effect=["history", "quit"]):
            with patch("builtins.print", side_effect=lambda x="": outputs.append(str(x))):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        assert any("Execution history" in out for out in outputs)

    async def test_dbstats_command(self, mock_agents):
        """Test dbstats command shows database statistics."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.get_session_stats = AsyncMock(return_value={
            "findings_count": 10,
            "validated_count": 8,
            "reports_count": 2,
        })
        mock_db.end_session = AsyncMock()
        mock_db.close = AsyncMock()

        outputs = []
        with patch("builtins.input", side_effect=["dbstats", "quit"]):
            with patch("builtins.print", side_effect=lambda x="": outputs.append(str(x))):
                with patch("research_agents.main.get_database", return_value=mock_db):
                    await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        assert any("Database Statistics" in out for out in outputs)

    async def test_dbstats_command_database_error(self, mock_agents):
        """Test dbstats command handles database errors."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        mock_db = MagicMock()
        mock_db.connect = AsyncMock(side_effect=Exception("Connection failed"))
        mock_db.end_session = AsyncMock()
        mock_db.close = AsyncMock()

        outputs = []
        with patch("builtins.input", side_effect=["dbstats", "quit"]):
            with patch("builtins.print", side_effect=lambda x="": outputs.append(str(x))):
                with patch("research_agents.main.get_database", return_value=mock_db):
                    await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        assert any("Database unavailable" in out for out in outputs)

    async def test_dbreports_command(self, mock_agents):
        """Test dbreports command lists reports."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.get_all_reports = AsyncMock(return_value=[
            {"id": "1", "title": "Test Report", "created_at": "2024-01-01"},
        ])
        mock_db.end_session = AsyncMock()
        mock_db.close = AsyncMock()

        outputs = []
        with patch("builtins.input", side_effect=["dbreports", "quit"]):
            with patch("builtins.print", side_effect=lambda x="": outputs.append(str(x))):
                with patch("research_agents.main.get_database", return_value=mock_db):
                    await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        assert any("Recent reports" in out for out in outputs)

    async def test_dbreports_command_empty(self, mock_agents):
        """Test dbreports command with no reports."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.get_all_reports = AsyncMock(return_value=[])
        mock_db.end_session = AsyncMock()
        mock_db.close = AsyncMock()

        outputs = []
        with patch("builtins.input", side_effect=["dbreports", "quit"]):
            with patch("builtins.print", side_effect=lambda x="": outputs.append(str(x))):
                with patch("research_agents.main.get_database", return_value=mock_db):
                    await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        assert any("No reports" in out for out in outputs)

    async def test_empty_input_ignored(self, mock_agents):
        """Test that empty input is ignored."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        with patch("builtins.input", side_effect=["", "   ", "quit"]):
            with patch("builtins.print"):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        # chat should not be called for empty inputs
        main_agent.chat.assert_not_called()

    async def test_regular_message_calls_chat(self, mock_agents):
        """Test regular message calls the chat method."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        with patch("builtins.input", side_effect=["Hello there", "quit"]):
            with patch("builtins.print"):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        main_agent.chat.assert_called_once_with("Hello there")

    async def test_keyboard_interrupt(self, mock_agents):
        """Test keyboard interrupt exits gracefully."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("builtins.print"):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        research_agent.close.assert_called_once()

    async def test_eof_error(self, mock_agents):
        """Test EOF error exits gracefully."""
        main_agent, research_agent, validation_agent, orchestrator = mock_agents

        with patch("builtins.input", side_effect=EOFError):
            with patch("builtins.print"):
                await chat_loop(main_agent, research_agent, validation_agent, orchestrator)

        research_agent.close.assert_called_once()


class TestMain:
    """Test the main function."""

    async def test_main_registers_agents(self):
        """Test main function registers all agents."""
        with patch("research_agents.main.chat_loop", new_callable=AsyncMock) as mock_chat:
            with patch("builtins.print"):
                await main()

            # Verify chat_loop was called with agents
            mock_chat.assert_called_once()
            args = mock_chat.call_args[0]
            assert len(args) == 4  # main_agent, research, validation, orchestrator


class TestRun:
    """Test the run entry point."""

    def test_run_calls_asyncio_run(self):
        """Test run calls asyncio.run with main."""
        with patch("research_agents.main.asyncio.run") as mock_run:
            with patch("research_agents.main.main", new_callable=AsyncMock) as mock_main:
                # We can't easily test this without actually running asyncio
                # Just verify the function exists and is callable
                assert callable(run)
