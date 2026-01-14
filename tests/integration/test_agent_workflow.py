"""Integration tests for agent workflows."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_agents.core.orchestrator import Orchestrator
from research_agents.tools.web_search import SearchResult


class TestResearchAgentWorkflow:
    """Test research agent integration with other components."""

    async def test_research_returns_structured_findings(self, full_agent_system: dict):
        """Test that research agent returns properly structured findings."""
        research_agent = full_agent_system["research_agent"]

        result = await research_agent.execute("test query about climate change")

        assert result["status"] == "completed"
        assert "findings" in result
        assert "sources" in result
        assert len(result["findings"]) > 0

        # Verify finding structure
        finding = result["findings"][0]
        assert "title" in finding
        assert "snippet" in finding
        assert "url" in finding

    async def test_research_findings_include_sources(self, full_agent_system: dict):
        """Test that research findings include source URLs."""
        research_agent = full_agent_system["research_agent"]

        result = await research_agent.execute("artificial intelligence research")

        sources = result["sources"]
        assert len(sources) > 0
        assert all(s.startswith("http") for s in sources)


class TestValidationAgentWorkflow:
    """Test validation agent integration."""

    async def test_validation_with_trusted_sources(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test validation identifies trusted vs untrusted sources."""
        validation_agent = full_agent_system["validation_agent"]

        # Configure mock to return verified status
        mock_llm_responses.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "VERIFIED", "confidence": 0.9, "reason": "Confirmed", "sources": ["https://wikipedia.org"]}'}]
        }

        findings = [
            {
                "title": "Trusted Finding",
                "snippet": "Content from Wikipedia",
                "url": "https://wikipedia.org/test",
            },
        ]

        result = await validation_agent.validate_findings(findings)

        assert "validated" in result
        assert "removed" in result
        assert "stats" in result

    async def test_validation_filters_unverified(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test that validation correctly filters unverified findings."""
        validation_agent = full_agent_system["validation_agent"]

        # First call returns verified, second returns unverified
        mock_llm_responses.post.return_value.json.side_effect = [
            {"content": [{"text": '{"status": "VERIFIED", "confidence": 0.85, "reason": "OK", "sources": []}'}]},
            {"content": [{"text": '{"status": "UNVERIFIED", "confidence": 0.1, "reason": "No sources", "sources": []}'}]},
        ]

        findings = [
            {"title": "Good Finding", "snippet": "Valid content", "url": "https://edu.example.com"},
            {"title": "Bad Finding", "snippet": "Unverifiable", "url": "https://random.com"},
        ]

        result = await validation_agent.validate_findings(findings)

        assert len(result["validated"]) == 1
        assert len(result["removed"]) == 1
        assert result["stats"]["validation_rate"] == 0.5


class TestReportAgentWorkflow:
    """Test report agent integration."""

    async def test_report_compilation(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test compiling a report from research data."""
        report_agent = full_agent_system["report_agent"]

        mock_llm_responses.post.return_value.json.return_value = {
            "content": [{"text": "# Research Report\n\n## Summary\n\nTest findings compiled."}]
        }

        information = [
            {"source": "Research 1", "content": "Finding 1 details"},
            {"source": "Research 2", "content": "Finding 2 details"},
        ]

        report = await report_agent.compile_report("Test Report", information)

        assert "Research Report" in report or "Report" in report
        assert report_agent.current_report is not None
        assert report_agent.current_title == "Test Report"

    async def test_report_save_and_load(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test saving and loading reports."""
        report_agent = full_agent_system["report_agent"]

        mock_llm_responses.post.return_value.json.return_value = {
            "content": [{"text": "# Test Report\n\nReport content here."}]
        }

        # Compile report
        await report_agent.compile_report(
            "Integration Test Report",
            [{"source": "Test", "content": "Data"}],
        )

        # Save report
        filepath = await report_agent.save_report("integration_test")

        assert filepath.exists()
        assert filepath.name == "integration_test.md"

        # Clear and reload
        report_agent.current_report = None
        loaded = report_agent.load_report("integration_test")

        assert "Test Report" in loaded


class TestOrchestratorAgentSelection:
    """Test orchestrator's agent selection with real agents."""

    async def test_orchestrator_selects_research_agent(self, full_agent_system: dict):
        """Test orchestrator selects research agent for search queries."""
        orchestrator = full_agent_system["orchestrator"]

        selected = await orchestrator.select_agent("Search for information about AI")

        assert selected == "research"

    async def test_orchestrator_selects_validation_agent(self, full_agent_system: dict):
        """Test orchestrator selects validation agent for validation queries."""
        orchestrator = full_agent_system["orchestrator"]

        selected = await orchestrator.select_agent("Validate this claim")

        assert selected == "validation"

    async def test_orchestrator_selects_report_agent(self, full_agent_system: dict):
        """Test orchestrator selects report agent for report queries."""
        orchestrator = full_agent_system["orchestrator"]

        selected = await orchestrator.select_agent("Create a report from findings")

        assert selected == "report"

    async def test_orchestrator_runs_selected_agent(self, full_agent_system: dict):
        """Test orchestrator executes the selected agent."""
        orchestrator = full_agent_system["orchestrator"]

        result = await orchestrator.run("Find information about machine learning")

        assert result["status"] == "completed"
        assert "findings" in result

        # Check history was recorded
        history = orchestrator.get_execution_history()
        assert len(history) == 1
        assert history[0]["agent"] == "research"


class TestAgentPipeline:
    """Test running agents in a pipeline."""

    async def test_research_to_validation_pipeline(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test pipeline from research to validation."""
        orchestrator = full_agent_system["orchestrator"]

        # Configure validation response
        mock_llm_responses.post.return_value.json.return_value = {
            "content": [{"text": '{"status": "VERIFIED", "confidence": 0.8, "reason": "OK", "sources": []}'}]
        }

        tasks = [
            {"task": "Research quantum computing", "agent": "research"},
            {
                "task": "Validate findings",
                "agent": "validation",
                "context": {"findings": []},  # Will be populated from previous
            },
        ]

        # First run research
        research_result = await orchestrator.run(tasks[0]["task"], tasks[0]["agent"])
        assert research_result["status"] == "completed"

        # Then validate with findings from research
        validation_context = {"findings": research_result["findings"]}
        validation_result = await orchestrator.run(
            tasks[1]["task"],
            tasks[1]["agent"],
            validation_context,
        )
        assert validation_result["status"] == "completed"
        assert "validated" in validation_result

    async def test_full_research_validation_report_pipeline(
        self,
        full_agent_system: dict,
        mock_llm_responses: MagicMock,
    ):
        """Test complete pipeline: research → validation → report."""
        orchestrator = full_agent_system["orchestrator"]

        # Mock responses for the pipeline
        responses = [
            # Validation responses (one per finding)
            {"content": [{"text": '{"status": "VERIFIED", "confidence": 0.85, "reason": "OK", "sources": []}'}]},
            {"content": [{"text": '{"status": "VERIFIED", "confidence": 0.75, "reason": "OK", "sources": []}'}]},
            {"content": [{"text": '{"status": "UNVERIFIED", "confidence": 0.2, "reason": "No sources", "sources": []}'}]},
            # Report response
            {"content": [{"text": "# Research Report\n\nValidated findings compiled."}]},
        ]
        mock_llm_responses.post.return_value.json.side_effect = responses

        # Step 1: Research
        research_result = await orchestrator.run(
            "Research renewable energy",
            "research",
        )
        assert research_result["status"] == "completed"
        findings = research_result["findings"]

        # Step 2: Validation
        validation_result = await orchestrator.run(
            "Validate findings",
            "validation",
            {"findings": findings},
        )
        assert validation_result["status"] == "completed"
        validated = validation_result["validated"]

        # Step 3: Report
        report_context = {
            "title": "Renewable Energy Report",
            "information": [
                {"source": f["title"], "content": f["snippet"]}
                for f in validated
            ],
        }
        report_result = await orchestrator.run(
            "Create report",
            "report",
            report_context,
        )
        assert report_result["status"] == "completed"
        assert "report" in report_result

        # Verify execution history
        history = orchestrator.get_execution_history()
        assert len(history) == 3
        assert [h["agent"] for h in history] == ["research", "validation", "report"]
