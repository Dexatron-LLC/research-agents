# Research Agents

A Python-based multi-agent system for automated research, validation, and report generation.

## Features

- **Research Agent**: Web searching and information gathering using DuckDuckGo
- **Validation Agent**: Fact-checking findings against trusted sources
- **Report Agent**: Generating markdown reports from validated research
- **Main Agent**: Coordinating the workflow with an interactive chat interface
- **SQLite Database**: Session persistence for findings, validations, and reports
- **Orchestrator**: Keyword and LLM-based agent selection

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

### Linux/Mac

```bash
# Make the script executable (first time only)
chmod +x research.sh

# Run the application
./research.sh
```

### Windows

```cmd
research.bat
```

The launcher scripts will automatically:
1. Check if `uv` is installed
2. Set up the virtual environment on first run
3. Start the interactive research assistant

## Manual Installation

If you prefer to run manually without the launcher scripts:

```bash
# Clone the repository
git clone https://github.com/Dexatron-LLC/research-agents.git
cd research-agents

# Install dependencies
uv sync

# Run the application
uv run research-agents
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

## Usage

Once running, you can interact with the research assistant through the chat interface:

```
You: Search for information about renewable energy
Assistant: [Performs research and returns findings]

You: Validate the findings
Assistant: [Validates findings against trusted sources and auto-generates a report]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `quit` / `exit` | Exit the application |
| `clear` | Clear the conversation history |
| `status` | Show current validation status |
| `cache` | Display research cache contents |
| `dbstats` | Show database statistics for current session |
| `dbreports` | List all reports in the database |

## Project Structure

```
research-agents/
├── src/research_agents/
│   ├── agents/           # Agent implementations
│   │   ├── definitions/  # YAML agent definitions
│   │   ├── main_agent.py
│   │   ├── report_agent.py
│   │   ├── research_agent.py
│   │   └── validation_agent.py
│   ├── core/             # Core components
│   │   ├── base_agent.py
│   │   ├── config.py
│   │   ├── database.py
│   │   └── orchestrator.py
│   ├── tools/            # Tools for agents
│   │   └── web_search.py
│   └── main.py           # Entry point
├── tests/                # Unit and integration tests
├── data/                 # SQLite database storage
├── research.sh           # Linux/Mac launcher
├── research.bat          # Windows launcher
└── pyproject.toml
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run only unit tests
uv run pytest tests/core tests/agents tests/tools

# Run only integration tests
uv run pytest tests/integration
```

## License

MIT
