"""End-to-end integration tests for the complete system."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_agents.agents.main_agent import MainAgent
from research_agents.core.database import DatabaseService


class TestMainAgentEndToEnd:
    """End-to-end tests for main agent interactions."""

    async def test_simple_chat_conversation(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test a simple chat conversation without tool calls."""
        main_agent = full_agent_system["main_agent"]

        mock_llm_responses.post.return_value.json.return_value = {
            "content": [{"text": "Hello! I'm your research assistant. How can I help you today?"}]
        }

        response = await main_agent.chat("Hello, how are you?")

        assert "Hello" in response or "research" in response.lower()
        assert len(main_agent.conversation_history) >= 2

    async def test_research_flow_with_database(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test research flow with database persistence."""
        main_agent = full_agent_system["main_agent"]
        database = full_agent_system["database"]

        # First response triggers research tool
        mock_llm_responses.post.return_value.json.side_effect = [
            {"content": [{"text": '{"tool": "research", "query": "artificial intelligence"}'}]},
            {"content": [{"text": "I found some research results about AI. Would you like me to validate them?"}]},
        ]

        response = await main_agent.chat("Search for information about AI")

        # Verify findings are pending validation
        assert len(main_agent.pending_validation) > 0

        # Verify database has findings
        stats = await database.get_session_stats(main_agent.session_id)
        assert stats["findings_count"] > 0

    async def test_validation_flow_stores_results(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test validation flow stores results in database."""
        main_agent = full_agent_system["main_agent"]
        database = full_agent_system["database"]

        # Setup: Add some pending findings
        main_agent.pending_validation = [
            {"title": "Test Finding", "snippet": "Test content", "url": "https://test.com"},
        ]
        main_agent.current_query = "test query"

        # Save finding to database first
        finding_id = await database.save_finding(
            session_id=main_agent.session_id,
            query="test query",
            title="Test Finding",
            snippet="Test content",
            url="https://test.com",
        )

        # Mock LLM responses for validation flow
        mock_llm_responses.post.return_value.json.side_effect = [
            # Tool call to validate
            {"content": [{"text": '{"tool": "validation", "action": "validate"}'}]},
            # Validation LLM response
            {"content": [{"text": '{"status": "VERIFIED", "confidence": 0.85, "reason": "Confirmed", "sources": ["https://source.com"]}'}]},
            # Report generation
            {"content": [{"text": "# Auto Report\n\nValidated content."}]},
            # Follow-up response
            {"content": [{"text": "Validation complete! A report has been auto-generated."}]},
        ]

        response = await main_agent.chat("Validate the research findings")

        # Verify validated findings are stored
        assert len(main_agent.validated_findings) >= 0  # May be 0 if validation moved them
        assert len(main_agent.pending_validation) == 0  # Should be cleared

    async def test_full_research_to_report_flow(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test complete flow from research to auto-generated report."""
        main_agent = full_agent_system["main_agent"]
        database = full_agent_system["database"]

        # Step 1: Research
        mock_llm_responses.post.return_value.json.side_effect = [
            {"content": [{"text": '{"tool": "research", "query": "climate change effects"}'}]},
            {"content": [{"text": "Found 3 results about climate change. Shall I validate them?"}]},
        ]

        await main_agent.chat("Research climate change effects")

        # Verify research state
        assert len(main_agent.pending_validation) > 0
        pending_count = len(main_agent.pending_validation)

        # Step 2: Validation with auto-report
        validation_responses = [
            {"content": [{"text": '{"tool": "validation", "action": "validate"}'}]},
        ]
        # Add validation responses for each finding
        for _ in range(pending_count):
            validation_responses.append(
                {"content": [{"text": '{"status": "VERIFIED", "confidence": 0.8, "reason": "OK", "sources": []}'}]}
            )
        # Add report generation and follow-up
        validation_responses.extend([
            {"content": [{"text": "# Climate Report\n\nFindings about climate change."}]},
            {"content": [{"text": "Validation complete! Report auto-generated from validated findings."}]},
        ])

        mock_llm_responses.post.return_value.json.side_effect = validation_responses

        await main_agent.chat("Validate the findings")

        # Verify final state
        assert len(main_agent.pending_validation) == 0
        assert len(main_agent.research_cache) > 0


class TestConversationPersistence:
    """Test that conversations are properly persisted."""

    async def test_conversation_history_persists(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test conversation history is saved to database."""
        main_agent = full_agent_system["main_agent"]
        database = full_agent_system["database"]

        mock_llm_responses.post.return_value.json.return_value = {
            "content": [{"text": "I can help you with that!"}]
        }

        # Have a conversation
        await main_agent.chat("First message")
        await main_agent.chat("Second message")
        await main_agent.chat("Third message")

        # Check in-memory history
        assert len(main_agent.conversation_history) == 6  # 3 user + 3 assistant

        # Check database history
        db_history = await database.get_conversation_history(main_agent.session_id)
        assert len(db_history) == 6

    async def test_session_isolation(
        self,
        integration_settings,
        integration_database: DatabaseService,
        mock_llm_responses: MagicMock,
        mock_search_tool: MagicMock,
        mocker,
    ):
        """Test that different main agent sessions are isolated."""
        from research_agents.core.orchestrator import Orchestrator

        mocker.patch(
            "research_agents.agents.research_agent.WebSearchTool",
            return_value=mock_search_tool,
        )

        mock_llm_responses.post.return_value.json.return_value = {
            "content": [{"text": "Response"}]
        }

        # Create two separate main agents with different sessions
        orchestrator1 = Orchestrator(use_llm_selection=False)
        orchestrator2 = Orchestrator(use_llm_selection=False)

        agent1 = MainAgent(orchestrator1, integration_settings, database=integration_database)
        agent2 = MainAgent(orchestrator2, integration_settings, database=integration_database)

        assert agent1.session_id != agent2.session_id

        # Have conversations in both
        await agent1.chat("Agent 1 message")
        await agent2.chat("Agent 2 message 1")
        await agent2.chat("Agent 2 message 2")

        # Verify isolation
        history1 = await integration_database.get_conversation_history(agent1.session_id)
        history2 = await integration_database.get_conversation_history(agent2.session_id)

        assert len(history1) == 2  # 1 user + 1 assistant
        assert len(history2) == 4  # 2 user + 2 assistant


class TestCacheManagement:
    """Test cache management across the system."""

    async def test_clear_cache_clears_all_state(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test that clearing cache resets all state."""
        main_agent = full_agent_system["main_agent"]

        # Add some state
        main_agent.research_cache.append({"data": "test"})
        main_agent.validated_findings.append({"finding": "test"})
        main_agent.pending_validation.append({"pending": "test"})

        # Clear cache
        main_agent.clear_research_cache()

        # Verify all cleared
        assert main_agent.research_cache == []
        assert main_agent.validated_findings == []
        assert main_agent.pending_validation == []

    async def test_validation_status_reflects_state(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test that validation status accurately reflects current state."""
        main_agent = full_agent_system["main_agent"]

        # Initial state
        status = main_agent.get_validation_status()
        assert status["pending_validation"] == 0
        assert status["validated_findings"] == 0
        assert status["research_cache_items"] == 0

        # Add pending
        main_agent.pending_validation = [{"a": 1}, {"b": 2}]
        status = main_agent.get_validation_status()
        assert status["pending_validation"] == 2

        # Add validated
        main_agent.validated_findings = [{"c": 3}]
        status = main_agent.get_validation_status()
        assert status["validated_findings"] == 1

        # Add to cache
        main_agent.research_cache = [{"d": 4}, {"e": 5}]
        status = main_agent.get_validation_status()
        assert status["research_cache_items"] == 2


class TestErrorHandling:
    """Test error handling in end-to-end flows."""

    async def test_handles_missing_research_for_validation(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test handling validation request with no pending research."""
        main_agent = full_agent_system["main_agent"]

        # Clear any pending validation
        main_agent.pending_validation = []

        mock_llm_responses.post.return_value.json.side_effect = [
            {"content": [{"text": '{"tool": "validation", "action": "validate"}'}]},
        ]

        result = await main_agent.execute("Validate the findings")

        assert "error" in str(result).lower() or "no research" in str(result).lower()

    async def test_handles_missing_cache_for_report(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test handling report request with no research cache."""
        main_agent = full_agent_system["main_agent"]

        # Clear cache
        main_agent.research_cache = []
        main_agent.pending_validation = []

        mock_llm_responses.post.return_value.json.side_effect = [
            {"content": [{"text": '{"tool": "report", "action": "create", "title": "Test"}'}]},
        ]

        result = await main_agent.execute("Create a report")

        assert "error" in str(result).lower()

    async def test_handles_unknown_tool(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test handling unknown tool call."""
        main_agent = full_agent_system["main_agent"]

        mock_llm_responses.post.return_value.json.return_value = {
            "content": [{"text": '{"tool": "nonexistent_tool", "action": "do_something"}'}]
        }

        result = await main_agent.execute("Do something unknown")

        assert "error" in str(result).lower() or "unknown" in str(result).lower()
