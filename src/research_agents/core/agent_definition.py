"""Agent definition schema and loader for YAML-based agent configuration."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AgentDefinition(BaseModel):
    """Definition of an agent loaded from YAML configuration."""

    name: str = Field(..., description="Unique identifier for the agent")
    role: str = Field(..., description="The agent's role or job title")
    goal: str = Field(..., description="What the agent is trying to achieve")
    backstory: str = Field(..., description="Background context that shapes agent behavior")
    description: str | None = Field(None, description="Short description for agent selection")
    tools: list[str] = Field(default_factory=list, description="List of tools the agent can use")
    allow_delegation: bool = Field(default=False, description="Whether agent can delegate to others")
    verbose: bool = Field(default=False, description="Whether to output detailed logs")
    max_iterations: int = Field(default=10, description="Maximum iterations for task completion")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")

    def get_system_prompt(self) -> str:
        """Generate a system prompt from the agent definition.

        Returns:
            Formatted system prompt incorporating role, goal, and backstory
        """
        return f"""You are {self.role}.

## Your Goal
{self.goal}

## Your Background
{self.backstory}

## Guidelines
- Stay focused on your goal
- Use your expertise and background to inform your responses
- Be thorough but concise
- If you cannot complete a task, explain why clearly"""

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "AgentDefinition":
        """Load an agent definition from a YAML file.

        Args:
            yaml_path: Path to the YAML file

        Returns:
            AgentDefinition instance
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Agent definition file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_yaml_string(cls, yaml_content: str) -> "AgentDefinition":
        """Load an agent definition from a YAML string.

        Args:
            yaml_content: YAML content as a string

        Returns:
            AgentDefinition instance
        """
        data = yaml.safe_load(yaml_content)
        return cls(**data)

    def to_yaml(self, yaml_path: Path | str) -> None:
        """Save the agent definition to a YAML file.

        Args:
            yaml_path: Path to save the YAML file
        """
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


class AgentDefinitionLoader:
    """Loader for managing multiple agent definitions."""

    def __init__(self, definitions_dir: Path | str | None = None):
        """Initialize the loader.

        Args:
            definitions_dir: Directory containing agent YAML files
        """
        if definitions_dir:
            self.definitions_dir = Path(definitions_dir)
        else:
            # Default to agents directory in the package
            self.definitions_dir = Path(__file__).parent.parent / "agents" / "definitions"

    def load(self, agent_name: str) -> AgentDefinition:
        """Load an agent definition by name.

        Args:
            agent_name: Name of the agent (filename without .yaml extension)

        Returns:
            AgentDefinition instance
        """
        yaml_path = self.definitions_dir / f"{agent_name}.yaml"
        return AgentDefinition.from_yaml(yaml_path)

    def load_all(self) -> dict[str, AgentDefinition]:
        """Load all agent definitions from the directory.

        Returns:
            Dictionary mapping agent names to their definitions
        """
        definitions = {}
        if self.definitions_dir.exists():
            for yaml_file in self.definitions_dir.glob("*.yaml"):
                agent_name = yaml_file.stem
                definitions[agent_name] = AgentDefinition.from_yaml(yaml_file)
        return definitions

    def exists(self, agent_name: str) -> bool:
        """Check if an agent definition exists.

        Args:
            agent_name: Name of the agent

        Returns:
            True if definition file exists
        """
        yaml_path = self.definitions_dir / f"{agent_name}.yaml"
        return yaml_path.exists()
