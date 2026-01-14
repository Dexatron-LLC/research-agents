@echo off
REM Research Agents - Multi-agent research system
REM Run the interactive research assistant

pushd "%~dp0"

REM Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: uv is not installed.
    echo Install it with: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    popd
    exit /b 1
)

REM Sync dependencies if needed
if not exist ".venv" (
    echo Setting up virtual environment...
    uv sync
)

REM Run the application
uv run research-agents %*

popd
