"""Fixtures specifically for integration tests."""

import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_agents.agents.main_agent import MainAgent
from research_agents.agents.report_agent import ReportAgent
from research_agents.agents.research_agent import ResearchAgent
from research_agents.agents.validation_agent import ValidationAgent
from research_agents.core.config import Settings
from research_agents.core.database import DatabaseService
from research_agents.core.orchestrator import Orchestrator
from research_agents.tools.web_search import SearchResult


@pytest.fixture
def integration_temp_dir() -> Path:
    """Create a temporary directory for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def integration_settings(integration_temp_dir: Path) -> Settings:
    """Create settings for integration tests."""
    return Settings(
        anthropic_api_key="test-integration-key",
        anthropic_base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-20250514",
        fast_model="claude-haiku-4-20250514",
        max_tokens=2048,
        request_timeout=30.0,
        reports_dir=integration_temp_dir / "reports",
        database_path=integration_temp_dir / "integration_test.db",
    )


@pytest.fixture
async def integration_database(integration_settings: Settings) -> AsyncGenerator[DatabaseService, None]:
    """Create and initialize a real database for integration tests."""
    db = DatabaseService(integration_settings)
    await db.connect()
    await db.initialize_schema()
    yield db
    await db.close()


@pytest.fixture
def mock_search_tool(mocker) -> MagicMock:
    """Mock the web search tool for integration tests."""
    mock_tool = MagicMock()
    mock_tool.search.return_value = [
        SearchResult(
            title="Integration Test Result 1",
            url="https://wikipedia.org/integration1",
            snippet="This is verified content from a trusted source.",
        ),
        SearchResult(
            title="Integration Test Result 2",
            url="https://harvard.edu/research",
            snippet="Academic research supporting the claim.",
        ),
        SearchResult(
            title="Integration Test Result 3",
            url="https://example.com/random",
            snippet="Unverified content from random source.",
        ),
    ]

    async def mock_close():
        pass

    mock_tool.close = mock_close
    return mock_tool


@pytest.fixture
def mock_llm_responses(mocker) -> MagicMock:
    """Mock LLM API responses for integration tests."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    # Default response
    mock_response.json.return_value = {
        "content": [{"text": "This is a mock LLM response."}]
    }

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    mocker.patch("httpx.AsyncClient", return_value=mock_client)
    return mock_client


@pytest.fixture
def integration_orchestrator(integration_settings: Settings) -> Orchestrator:
    """Create an orchestrator for integration tests."""
    return Orchestrator(use_llm_selection=False, settings=integration_settings)


@pytest.fixture
async def full_agent_system(
    integration_settings: Settings,
    integration_database: DatabaseService,
    integration_orchestrator: Orchestrator,
    mock_search_tool: MagicMock,
    mocker,
) -> dict:
    """Create a complete agent system for integration tests.

    Returns a dict with all agents and supporting components.
    """
    # Mock the WebSearchTool for all agents
    mocker.patch(
        "research_agents.agents.research_agent.WebSearchTool",
        return_value=mock_search_tool,
    )
    mocker.patch(
        "research_agents.agents.validation_agent.WebSearchTool",
        return_value=mock_search_tool,
    )

    # Create agents
    research_agent = ResearchAgent()
    research_agent.search_tool = mock_search_tool

    validation_agent = ValidationAgent(integration_settings)
    validation_agent.search_tool = mock_search_tool

    report_agent = ReportAgent(integration_settings)

    main_agent = MainAgent(
        integration_orchestrator,
        integration_settings,
        database=integration_database,
    )

    # Register agents
    integration_orchestrator.register_agent(research_agent)
    integration_orchestrator.register_agent(validation_agent)
    integration_orchestrator.register_agent(report_agent)
    integration_orchestrator.register_agent(main_agent)

    return {
        "orchestrator": integration_orchestrator,
        "main_agent": main_agent,
        "research_agent": research_agent,
        "validation_agent": validation_agent,
        "report_agent": report_agent,
        "database": integration_database,
        "settings": integration_settings,
        "search_tool": mock_search_tool,
    }
