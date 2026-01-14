#!/bin/bash
# Research Agents - Multi-agent research system
# Run the interactive research assistant

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    uv sync
fi

# Run the application
uv run research-agents "$@"
