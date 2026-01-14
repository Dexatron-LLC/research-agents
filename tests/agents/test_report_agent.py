"""Tests for the report agent."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from research_agents.agents.report_agent import ReportAgent
from research_agents.core.config import Settings


class TestReportAgent:
    """Tests for the ReportAgent class."""

    def test_init(self, temp_dir: Path):
        """Test report agent initialization."""
        settings = Settings(reports_dir=temp_dir / "reports")
        agent = ReportAgent(settings)

        assert agent.name == "report"
        assert agent.settings is not None
        assert agent.reports_dir == temp_dir / "reports"
        assert agent.current_report is None
        assert agent.current_title is None

    def test_loads_yaml_definition(self, temp_dir: Path):
        """Test that the agent loads its YAML definition."""
        settings = Settings(reports_dir=temp_dir / "reports")
        agent = ReportAgent(settings)

        assert agent.role == "Technical Report Writer"
        assert agent.definition is not None

    def test_get_report_system_prompt(self, temp_dir: Path):
        """Test system prompt generation."""
        settings = Settings(reports_dir=temp_dir / "reports")
        agent = ReportAgent(settings)

        prompt = agent._get_report_system_prompt()

        assert "Technical Report Writer" in prompt
        assert "## Report Writing Guidelines" in prompt
        assert "markdown" in prompt.lower()

    async def test_compile_report(self, temp_dir: Path, mock_httpx_client):
        """Test compiling a report."""
        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": "# Test Report\n\nReport content here."}]
        }

        settings = Settings(
            reports_dir=temp_dir / "reports",
            anthropic_api_key="test-key",
        )
        agent = ReportAgent(settings)

        information = [
            {"source": "Source 1", "content": "Content 1"},
            {"source": "Source 2", "content": "Content 2"},
        ]
        report = await agent.compile_report("Test Report", information)

        assert "# Test Report" in report
        assert agent.current_report == report
        assert agent.current_title == "Test Report"

    async def test_compile_report_with_instructions(self, temp_dir: Path, mock_httpx_client):
        """Test compiling a report with additional instructions."""
        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": "Report content"}]
        }

        settings = Settings(
            reports_dir=temp_dir / "reports",
            anthropic_api_key="test-key",
        )
        agent = ReportAgent(settings)

        information = [{"source": "Test", "content": "Data"}]
        await agent.compile_report(
            "Report",
            information,
            additional_instructions="Use bullet points",
        )

        # Verify the API was called
        mock_httpx_client.post.assert_called_once()

    async def test_compile_report_no_api_key(self, temp_dir: Path):
        """Test compiling report without API key."""
        settings = Settings(
            reports_dir=temp_dir / "reports",
            anthropic_api_key="",
        )
        agent = ReportAgent(settings)

        information = [{"source": "Test", "content": "Data"}]
        report = await agent.compile_report("Test", information)

        assert "Error" in report
        assert "API" in report

    async def test_save_report(self, temp_dir: Path):
        """Test saving a report."""
        settings = Settings(reports_dir=temp_dir / "reports")
        agent = ReportAgent(settings)
        agent.current_report = "# Test Report\n\nContent here."
        agent.current_title = "Test Report"

        filepath = await agent.save_report()

        assert filepath.exists()
        assert filepath.suffix == ".md"
        assert filepath.read_text() == "# Test Report\n\nContent here."

    async def test_save_report_custom_filename(self, temp_dir: Path):
        """Test saving a report with custom filename."""
        settings = Settings(reports_dir=temp_dir / "reports")
        agent = ReportAgent(settings)
        agent.current_report = "Content"
        agent.current_title = "Title"

        filepath = await agent.save_report(filename="custom_name")

        assert filepath.name == "custom_name.md"

    async def test_save_report_no_current_report(self, temp_dir: Path):
        """Test saving when no report has been generated."""
        settings = Settings(reports_dir=temp_dir / "reports")
        agent = ReportAgent(settings)

        with pytest.raises(ValueError, match="No report to save"):
            await agent.save_report()

    def test_list_reports(self, temp_dir: Path):
        """Test listing saved reports."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir(parents=True)

        # Create test reports
        (reports_dir / "report1.md").write_text("Report 1")
        (reports_dir / "report2.md").write_text("Report 2")
        (reports_dir / "not_a_report.txt").write_text("Not MD")

        settings = Settings(reports_dir=reports_dir)
        agent = ReportAgent(settings)

        reports = agent.list_reports()

        assert len(reports) == 2
        assert all(r.suffix == ".md" for r in reports)

    def test_list_reports_empty(self, temp_dir: Path):
        """Test listing reports when none exist."""
        settings = Settings(reports_dir=temp_dir / "empty_reports")
        agent = ReportAgent(settings)

        reports = agent.list_reports()

        assert reports == []

    def test_load_report(self, temp_dir: Path):
        """Test loading a saved report."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "existing.md").write_text("# Existing Report\n\nContent.")

        settings = Settings(reports_dir=reports_dir)
        agent = ReportAgent(settings)

        content = agent.load_report("existing.md")

        assert content == "# Existing Report\n\nContent."
        assert agent.current_report == content
        assert agent.current_title == "existing"

    def test_load_report_without_extension(self, temp_dir: Path):
        """Test loading report without .md extension."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "test.md").write_text("Content")

        settings = Settings(reports_dir=reports_dir)
        agent = ReportAgent(settings)

        content = agent.load_report("test")

        assert content == "Content"

    def test_load_report_not_found(self, temp_dir: Path):
        """Test loading nonexistent report."""
        settings = Settings(reports_dir=temp_dir / "reports")
        agent = ReportAgent(settings)

        with pytest.raises(FileNotFoundError):
            agent.load_report("nonexistent.md")


class TestReportAgentExecute:
    """Tests for the execute method."""

    async def test_execute_create_report(self, temp_dir: Path, mock_httpx_client):
        """Test execute to create a report."""
        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": "Generated report content"}]
        }

        settings = Settings(
            reports_dir=temp_dir / "reports",
            anthropic_api_key="test-key",
        )
        agent = ReportAgent(settings)

        context = {
            "title": "Test Report",
            "information": [{"source": "Test", "content": "Data"}],
        }
        result = await agent.execute("Test Report", context)

        assert result["status"] == "completed"
        assert "report" in result

    async def test_execute_save_report(self, temp_dir: Path):
        """Test execute to save current report."""
        settings = Settings(reports_dir=temp_dir / "reports")
        agent = ReportAgent(settings)
        agent.current_report = "Report content"

        context = {"save": True}
        result = await agent.execute("save", context)

        assert result["status"] == "saved"
        assert "filepath" in result

    async def test_execute_save_with_filename(self, temp_dir: Path):
        """Test execute to save with custom filename."""
        settings = Settings(reports_dir=temp_dir / "reports")
        agent = ReportAgent(settings)
        agent.current_report = "Content"

        context = {"save": True, "filename": "custom"}
        result = await agent.execute("save", context)

        assert "custom.md" in result["filepath"]

    async def test_execute_list_reports(self, temp_dir: Path):
        """Test execute to list reports."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "report1.md").write_text("R1")
        (reports_dir / "report2.md").write_text("R2")

        settings = Settings(reports_dir=reports_dir)
        agent = ReportAgent(settings)

        context = {"list": True}
        result = await agent.execute("list", context)

        assert result["status"] == "completed"
        assert len(result["reports"]) == 2

    async def test_execute_load_report(self, temp_dir: Path):
        """Test execute to load a report."""
        reports_dir = temp_dir / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "existing.md").write_text("Loaded content")

        settings = Settings(reports_dir=reports_dir)
        agent = ReportAgent(settings)

        context = {"load": "existing.md"}
        result = await agent.execute("load", context)

        assert result["status"] == "loaded"
        assert result["report"] == "Loaded content"

    async def test_execute_without_information_uses_task(self, temp_dir: Path, mock_httpx_client):
        """Test execute without information list uses task as content."""
        mock_httpx_client.post.return_value.json.return_value = {
            "content": [{"text": "Report"}]
        }

        settings = Settings(
            reports_dir=temp_dir / "reports",
            anthropic_api_key="test-key",
        )
        agent = ReportAgent(settings)

        result = await agent.execute("Write about AI")

        assert result["status"] == "completed"
