"""Tests for the agent definition module."""

from pathlib import Path

import pytest

from research_agents.core.agent_definition import AgentDefinition, AgentDefinitionLoader


class TestAgentDefinition:
    """Tests for the AgentDefinition class."""

    def test_create_agent_definition(self, sample_agent_definition: AgentDefinition):
        """Test creating an AgentDefinition instance."""
        assert sample_agent_definition.name == "test_agent"
        assert sample_agent_definition.role == "Test Agent Role"
        assert sample_agent_definition.goal == "Accomplish test tasks effectively"
        assert sample_agent_definition.backstory == "You are a test agent created for unit testing purposes."
        assert sample_agent_definition.description == "A test agent"
        assert sample_agent_definition.tools == ["web_search"]
        assert sample_agent_definition.allow_delegation is False
        assert sample_agent_definition.max_iterations == 5
        assert sample_agent_definition.metadata == {"test_key": "test_value"}

    def test_default_values(self):
        """Test AgentDefinition default values."""
        definition = AgentDefinition(
            name="minimal",
            role="Minimal Agent",
            goal="Test defaults",
            backstory="Minimal backstory",
        )
        assert definition.description is None
        assert definition.tools == []
        assert definition.allow_delegation is False
        assert definition.verbose is False
        assert definition.max_iterations == 10
        assert definition.metadata == {}

    def test_get_system_prompt(self, sample_agent_definition: AgentDefinition):
        """Test system prompt generation."""
        prompt = sample_agent_definition.get_system_prompt()

        assert "Test Agent Role" in prompt
        assert "Accomplish test tasks effectively" in prompt
        assert "You are a test agent created for unit testing purposes." in prompt
        assert "## Your Goal" in prompt
        assert "## Your Background" in prompt
        assert "## Guidelines" in prompt

    def test_from_yaml_string(self, sample_yaml_content: str):
        """Test loading from YAML string."""
        definition = AgentDefinition.from_yaml_string(sample_yaml_content)

        assert definition.name == "yaml_test_agent"
        assert definition.role == "YAML Test Agent"
        assert definition.goal == "Test YAML loading functionality"
        assert definition.tools == ["web_search", "report"]
        assert definition.allow_delegation is True
        assert definition.max_iterations == 15
        assert definition.metadata["category"] == "testing"
        assert definition.metadata["priority"] == "high"

    def test_from_yaml_file(self, temp_dir: Path, sample_yaml_content: str):
        """Test loading from YAML file."""
        yaml_path = temp_dir / "test_agent.yaml"
        yaml_path.write_text(sample_yaml_content)

        definition = AgentDefinition.from_yaml(yaml_path)

        assert definition.name == "yaml_test_agent"
        assert definition.role == "YAML Test Agent"

    def test_from_yaml_file_not_found(self, temp_dir: Path):
        """Test error when YAML file doesn't exist."""
        yaml_path = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            AgentDefinition.from_yaml(yaml_path)

    def test_to_yaml(self, temp_dir: Path, sample_agent_definition: AgentDefinition):
        """Test saving to YAML file."""
        yaml_path = temp_dir / "output" / "saved_agent.yaml"
        sample_agent_definition.to_yaml(yaml_path)

        assert yaml_path.exists()

        # Verify we can load it back
        loaded = AgentDefinition.from_yaml(yaml_path)
        assert loaded.name == sample_agent_definition.name
        assert loaded.role == sample_agent_definition.role
        assert loaded.goal == sample_agent_definition.goal


class TestAgentDefinitionLoader:
    """Tests for the AgentDefinitionLoader class."""

    def test_init_with_custom_dir(self, temp_dir: Path):
        """Test initializing loader with custom directory."""
        loader = AgentDefinitionLoader(temp_dir)
        assert loader.definitions_dir == temp_dir

    def test_init_default_dir(self):
        """Test initializing loader with default directory."""
        loader = AgentDefinitionLoader()
        assert "agents" in str(loader.definitions_dir)
        assert "definitions" in str(loader.definitions_dir)

    def test_exists(self, temp_dir: Path, sample_yaml_content: str):
        """Test checking if agent definition exists."""
        loader = AgentDefinitionLoader(temp_dir)

        # Create a test file
        yaml_path = temp_dir / "existing_agent.yaml"
        yaml_path.write_text(sample_yaml_content)

        assert loader.exists("existing_agent") is True
        assert loader.exists("nonexistent_agent") is False

    def test_load(self, temp_dir: Path, sample_yaml_content: str):
        """Test loading an agent definition by name."""
        loader = AgentDefinitionLoader(temp_dir)

        # Create a test file
        yaml_path = temp_dir / "test_agent.yaml"
        yaml_path.write_text(sample_yaml_content)

        definition = loader.load("test_agent")
        assert definition.name == "yaml_test_agent"
        assert definition.role == "YAML Test Agent"

    def test_load_all(self, temp_dir: Path):
        """Test loading all agent definitions."""
        loader = AgentDefinitionLoader(temp_dir)

        # Create multiple test files
        for i in range(3):
            yaml_content = f"""
name: agent_{i}
role: Agent {i}
goal: Goal {i}
backstory: Backstory {i}
"""
            yaml_path = temp_dir / f"agent_{i}.yaml"
            yaml_path.write_text(yaml_content)

        definitions = loader.load_all()

        assert len(definitions) == 3
        assert "agent_0" in definitions
        assert "agent_1" in definitions
        assert "agent_2" in definitions

    def test_load_all_empty_dir(self, temp_dir: Path):
        """Test loading from empty directory."""
        loader = AgentDefinitionLoader(temp_dir)
        definitions = loader.load_all()
        assert definitions == {}

    def test_load_all_nonexistent_dir(self, temp_dir: Path):
        """Test loading from nonexistent directory."""
        loader = AgentDefinitionLoader(temp_dir / "nonexistent")
        definitions = loader.load_all()
        assert definitions == {}
