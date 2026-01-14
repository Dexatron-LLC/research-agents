"""Shared pytest fixtures for research agents tests."""

import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_agents.core.config import Settings
from research_agents.core.database import DatabaseService
from research_agents.core.orchestrator import Orchestrator
from research_agents.core.agent_definition import AgentDefinition
from research_agents.tools.web_search import SearchResult


@pytest.fixture
def mock_settings() -> Settings:
    """Create a Settings instance with test values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Settings(
            anthropic_api_key="test-api-key",
            anthropic_base_url="https://api.anthropic.com",
            default_model="claude-sonnet-4-20250514",
            fast_model="claude-haiku-4-20250514",
            max_tokens=1024,
            request_timeout=30.0,
            reports_dir=Path(tmpdir) / "reports",
            database_path=Path(tmpdir) / "test.db",
        )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def test_database(temp_dir: Path) -> AsyncGenerator[DatabaseService, None]:
    """Create a test database service."""
    settings = Settings(database_path=temp_dir / "test.db")
    db = DatabaseService(settings)
    await db.connect()
    await db.initialize_schema()
    yield db
    await db.close()


@pytest.fixture
def mock_orchestrator() -> Orchestrator:
    """Create a mock orchestrator."""
    return Orchestrator(use_llm_selection=False)


@pytest.fixture
def sample_agent_definition() -> AgentDefinition:
    """Create a sample agent definition for testing."""
    return AgentDefinition(
        name="test_agent",
        role="Test Agent Role",
        goal="Accomplish test tasks effectively",
        backstory="You are a test agent created for unit testing purposes.",
        description="A test agent",
        tools=["web_search"],
        allow_delegation=False,
        max_iterations=5,
        metadata={"test_key": "test_value"},
    )


@pytest.fixture
def sample_yaml_content() -> str:
    """Sample YAML content for agent definition tests."""
    return """
name: yaml_test_agent
role: YAML Test Agent
goal: Test YAML loading functionality
backstory: Created from YAML for testing
description: Test agent from YAML
tools:
  - web_search
  - report
allow_delegation: true
max_iterations: 15
metadata:
  category: testing
  priority: high
"""


@pytest.fixture
def mock_search_results() -> list[SearchResult]:
    """Create mock search results."""
    return [
        SearchResult(
            title="Test Result 1",
            url="https://example.com/1",
            snippet="This is the first test result snippet.",
        ),
        SearchResult(
            title="Test Result 2",
            url="https://example.com/2",
            snippet="This is the second test result snippet.",
        ),
        SearchResult(
            title="Test Result 3",
            url="https://example.com/3",
            snippet="This is the third test result snippet.",
        ),
    ]


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Create a mock LLM API response."""
    return {
        "content": [{"text": "This is a test response from the LLM."}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }


@pytest.fixture
def mock_validation_response() -> dict[str, Any]:
    """Create a mock validation LLM response."""
    return {
        "content": [
            {
                "text": '{"status": "VERIFIED", "confidence": 0.85, "reason": "Confirmed by trusted sources", "sources": ["https://wikipedia.org/test"]}'
            }
        ],
    }


@pytest.fixture
def mock_research_findings() -> list[dict[str, Any]]:
    """Create mock research findings for validation tests."""
    return [
        {
            "title": "Climate Change Facts",
            "snippet": "Global temperatures have risen by 1.1°C since pre-industrial times.",
            "url": "https://example.com/climate",
        },
        {
            "title": "Renewable Energy Growth",
            "snippet": "Solar energy capacity has grown 20x in the last decade.",
            "url": "https://example.com/energy",
        },
    ]


@pytest.fixture
def mock_httpx_client(mocker) -> MagicMock:
    """Create a mock httpx client for API calls."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "content": [{"text": "Mock response"}],
    }

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    mocker.patch("httpx.AsyncClient", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_ddgs(mocker) -> MagicMock:
    """Create a mock DDGS client for web search."""
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {"title": "Mock Result 1", "href": "https://mock1.com", "body": "Mock snippet 1"},
        {"title": "Mock Result 2", "href": "https://mock2.com", "body": "Mock snippet 2"},
    ]
    mocker.patch("research_agents.tools.web_search.DDGS", return_value=mock_ddgs)
    return mock_ddgs
