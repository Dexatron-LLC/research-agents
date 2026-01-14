"""Tests for the base agent module."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from research_agents.core.agent_definition import AgentDefinition
from research_agents.core.base_agent import BaseAgent


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing."""

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"status": "completed", "task": task}


class TestBaseAgent:
    """Tests for the BaseAgent class."""

    def test_init_with_name_only(self):
        """Test initialization with just a name."""
        agent = ConcreteAgent(name="test", load_definition=False)
        assert agent.name == "test"
        assert agent.description == "test agent"  # Default when no definition
        assert agent._definition is None

    def test_init_with_description(self):
        """Test initialization with custom description."""
        agent = ConcreteAgent(name="test", description="Custom description", load_definition=False)
        assert agent.description == "Custom description"

    def test_init_with_definition(self):
        """Test initialization with a pre-loaded definition."""
        definition = AgentDefinition(
            name="test",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory",
            description="Definition description",
        )
        agent = ConcreteAgent(name="test", definition=definition, load_definition=False)
        assert agent._definition is definition
        assert agent.description == "Definition description"

    def test_init_description_overrides_definition(self):
        """Test that explicit description overrides definition."""
        definition = AgentDefinition(
            name="test",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory",
            description="Definition description",
        )
        agent = ConcreteAgent(
            name="test",
            description="Override description",
            definition=definition,
            load_definition=False,
        )
        assert agent.description == "Override description"

    def test_init_uses_role_as_description_fallback(self):
        """Test that role is used as description fallback when no description."""
        definition = AgentDefinition(
            name="test",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory",
            description=None,
        )
        agent = ConcreteAgent(name="test", definition=definition, load_definition=False)
        assert agent.description == "Test Role"

    def test_try_load_definition_success(self, mocker):
        """Test successful YAML definition loading."""
        mock_definition = AgentDefinition(
            name="test",
            role="Loaded Role",
            goal="Loaded Goal",
            backstory="Loaded backstory",
        )
        mock_loader = MagicMock()
        mock_loader.exists.return_value = True
        mock_loader.load.return_value = mock_definition

        mocker.patch(
            "research_agents.core.base_agent.AgentDefinitionLoader",
            return_value=mock_loader,
        )

        agent = ConcreteAgent(name="test", load_definition=True)
        assert agent._definition is mock_definition

    def test_try_load_definition_not_found(self, mocker):
        """Test YAML definition not found."""
        mock_loader = MagicMock()
        mock_loader.exists.return_value = False

        mocker.patch(
            "research_agents.core.base_agent.AgentDefinitionLoader",
            return_value=mock_loader,
        )

        agent = ConcreteAgent(name="nonexistent", load_definition=True)
        assert agent._definition is None

    def test_try_load_definition_exception(self, mocker):
        """Test handling of exception during definition loading."""
        mock_loader = MagicMock()
        mock_loader.exists.side_effect = Exception("Load error")

        mocker.patch(
            "research_agents.core.base_agent.AgentDefinitionLoader",
            return_value=mock_loader,
        )

        agent = ConcreteAgent(name="test", load_definition=True)
        assert agent._definition is None

    def test_definition_property(self):
        """Test the definition property."""
        definition = AgentDefinition(
            name="test",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory",
        )
        agent = ConcreteAgent(name="test", definition=definition, load_definition=False)
        assert agent.definition is definition

    def test_definition_property_none(self):
        """Test definition property when None."""
        agent = ConcreteAgent(name="test", load_definition=False)
        assert agent.definition is None

    def test_role_property_with_definition(self):
        """Test role property with definition."""
        definition = AgentDefinition(
            name="test",
            role="Custom Role",
            goal="Test Goal",
            backstory="Test backstory",
        )
        agent = ConcreteAgent(name="test", definition=definition, load_definition=False)
        assert agent.role == "Custom Role"

    def test_role_property_without_definition(self):
        """Test role property without definition returns name."""
        agent = ConcreteAgent(name="test_agent", load_definition=False)
        assert agent.role == "test_agent"

    def test_goal_property_with_definition(self):
        """Test goal property with definition."""
        definition = AgentDefinition(
            name="test",
            role="Test Role",
            goal="Custom Goal",
            backstory="Test backstory",
        )
        agent = ConcreteAgent(name="test", definition=definition, load_definition=False)
        assert agent.goal == "Custom Goal"

    def test_goal_property_without_definition(self):
        """Test goal property without definition returns default."""
        agent = ConcreteAgent(name="myagent", load_definition=False)
        assert agent.goal == "Accomplish tasks as the myagent agent"

    def test_backstory_property_with_definition(self):
        """Test backstory property with definition."""
        definition = AgentDefinition(
            name="test",
            role="Test Role",
            goal="Test Goal",
            backstory="Custom Backstory",
        )
        agent = ConcreteAgent(name="test", definition=definition, load_definition=False)
        assert agent.backstory == "Custom Backstory"

    def test_backstory_property_without_definition(self):
        """Test backstory property without definition returns empty."""
        agent = ConcreteAgent(name="test", load_definition=False)
        assert agent.backstory == ""

    def test_get_system_prompt_with_definition(self):
        """Test system prompt with definition."""
        definition = AgentDefinition(
            name="test",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory",
        )
        agent = ConcreteAgent(name="test", definition=definition, load_definition=False)
        prompt = agent.get_system_prompt()
        assert "Test Role" in prompt
        assert "Test Goal" in prompt

    def test_get_system_prompt_without_definition(self):
        """Test system prompt without definition."""
        agent = ConcreteAgent(name="myagent", description="My custom agent", load_definition=False)
        prompt = agent.get_system_prompt()
        assert "myagent" in prompt
        assert "My custom agent" in prompt

    def test_repr(self):
        """Test string representation."""
        agent = ConcreteAgent(name="test", load_definition=False)
        repr_str = repr(agent)
        assert "ConcreteAgent" in repr_str
        assert "test" in repr_str

    async def test_execute_abstract(self):
        """Test that execute is implemented in concrete class."""
        agent = ConcreteAgent(name="test", load_definition=False)
        result = await agent.execute("test task")
        assert result["status"] == "completed"
        assert result["task"] == "test task"

    async def test_execute_with_context(self):
        """Test execute with context."""
        agent = ConcreteAgent(name="test", load_definition=False)
        result = await agent.execute("task", {"key": "value"})
        assert result["status"] == "completed"
