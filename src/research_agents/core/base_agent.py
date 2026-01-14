"""Base agent class that all agents inherit from."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .agent_definition import AgentDefinition, AgentDefinitionLoader


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
        description: str | None = None,
        definition: AgentDefinition | None = None,
        load_definition: bool = True,
    ):
        """Initialize the agent.

        Args:
            name: Unique identifier for the agent
            description: Short description (overrides definition if provided)
            definition: Pre-loaded agent definition
            load_definition: Whether to auto-load definition from YAML
        """
        self.name = name
        self._definition = definition

        # Try to load definition from YAML if not provided
        if self._definition is None and load_definition:
            self._definition = self._try_load_definition()

        # Set description from definition or parameter
        if description:
            self.description = description
        elif self._definition:
            self.description = self._definition.description or self._definition.role
        else:
            self.description = f"{name} agent"

    def _try_load_definition(self) -> AgentDefinition | None:
        """Try to load the agent definition from YAML.

        Returns:
            AgentDefinition if found, None otherwise
        """
        try:
            loader = AgentDefinitionLoader()
            if loader.exists(self.name):
                return loader.load(self.name)
        except Exception:
            pass
        return None

    @property
    def definition(self) -> AgentDefinition | None:
        """Get the agent's definition."""
        return self._definition

    @property
    def role(self) -> str:
        """Get the agent's role."""
        if self._definition:
            return self._definition.role
        return self.name

    @property
    def goal(self) -> str:
        """Get the agent's goal."""
        if self._definition:
            return self._definition.goal
        return f"Accomplish tasks as the {self.name} agent"

    @property
    def backstory(self) -> str:
        """Get the agent's backstory."""
        if self._definition:
            return self._definition.backstory
        return ""

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.

        If a definition is loaded, uses the definition's prompt generator.
        Otherwise returns a basic prompt.

        Returns:
            System prompt string
        """
        if self._definition:
            return self._definition.get_system_prompt()
        return f"You are the {self.name} agent. {self.description}"

    @abstractmethod
    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a task and return results.

        Args:
            task: The task description to execute
            context: Optional context from other agents

        Returns:
            Dictionary containing the execution results
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, role={self.role!r})"
