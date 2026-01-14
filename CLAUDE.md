# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies and sync
uv sync

# Install Playwright browsers (required for web fetching)
uv run playwright install chromium

# Run the interactive chat application
uv run research-agents

# Run a single module/test
uv run python -c "from research_agents.main import main; import asyncio; asyncio.run(main())"
```

## Configuration

Configuration is managed via `pydantic-settings` in `src/research_agents/core/config.py`. Settings are loaded from:
1. `.env` file (if present)
2. Environment variables (fallback)

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `ANTHROPIC_API_KEY` - Required for LLM functionality
- `ANTHROPIC_BASE_URL` - API endpoint (default: https://api.anthropic.com)
- `DEFAULT_MODEL` - Main model for agents (default: claude-sonnet-4-20250514)
- `FAST_MODEL` - Model for agent selection (default: claude-haiku-4-20250514)

Access settings in code:
```python
from research_agents.core.config import get_settings
settings = get_settings()  # Cached singleton
```

## Agent Definitions (CrewAI-style YAML)

Each agent has a YAML definition file in `src/research_agents/agents/definitions/` that defines:
- **name**: Unique identifier
- **role**: The agent's job title/function
- **goal**: What the agent is trying to achieve
- **backstory**: Background context that shapes behavior
- **tools**: List of available tools
- **allow_delegation**: Whether agent can delegate tasks
- **metadata**: Custom key-value pairs

Example (`research.yaml`):
```yaml
name: research
role: Senior Research Analyst
goal: >
  Find accurate, relevant, and up-to-date information from the web...
backstory: >
  You are a seasoned research analyst with expertise in information retrieval...
tools:
  - web_search
  - page_fetch
```

Access in code:
```python
from research_agents.core import AgentDefinitionLoader

loader = AgentDefinitionLoader()
definition = loader.load("research")
print(definition.role, definition.goal)
```

Agents auto-load their definitions from YAML on initialization. The definition generates the system prompt via `agent.get_system_prompt()`.

## Architecture

This is a multi-agent system where specialized agents are coordinated by an orchestrator to perform research tasks.

### Core Components (`src/research_agents/core/`)

- **BaseAgent**: Abstract base class. Auto-loads YAML definition on init. Provides `role`, `goal`, `backstory` properties.
- **AgentDefinition**: Pydantic model for YAML schema. Generates system prompts from role/goal/backstory.
- **Orchestrator**: Registers agents and routes tasks. Agent selection uses LLM-based selection (Claude Haiku) with keyword-matching fallback.
- **Settings**: Pydantic settings class for configuration management

### Agents (`src/research_agents/agents/`)

- **MainAgent**: Conversation coordinator that interprets user requests and delegates to sub-agents via JSON tool calls. Maintains conversation history and research cache.
- **ResearchAgent**: Performs web searches using the `ddgs` library (DuckDuckGo)
- **ReportAgent**: Synthesizes cached research into markdown reports, can save/load/list reports from disk

### Tools (`src/research_agents/tools/`)

- **WebSearchTool**: Combines `ddgs` for search and Playwright for page fetching/screenshots

### Data Flow

1. User input → MainAgent (interprets via LLM)
2. MainAgent emits JSON tool calls → Orchestrator routes to appropriate agent
3. Research results cached in MainAgent.research_cache
4. Report agent uses cached research to generate markdown reports

### Adding New Agents

1. Create YAML definition in `agents/definitions/{name}.yaml` with role, goal, backstory
2. Create class inheriting from `BaseAgent` in `agents/`
3. Call `super().__init__(name="{name}")` - definition auto-loads
4. Implement `async execute(task, context)` returning `{"status": "completed", ...}`
5. Register in `main.py`: `orchestrator.register_agent(YourAgent())`
6. Add keyword patterns in `Orchestrator._keyword_select()` for fallback routing
7. Update MainAgent's system prompt to describe the new agent's tool call format
