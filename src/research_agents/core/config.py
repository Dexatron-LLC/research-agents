"""Configuration management for the research agents system."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Configuration
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        description="Anthropic API base URL",
    )

    # Model Configuration
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default model for main agent and report generation",
    )
    fast_model: str = Field(
        default="claude-haiku-4-20250514",
        description="Fast model for agent selection and simple tasks",
    )

    # Agent Configuration
    max_tokens: int = Field(
        default=2048,
        description="Default max tokens for LLM responses",
    )
    request_timeout: float = Field(
        default=60.0,
        description="Default timeout for API requests in seconds",
    )

    # Report Configuration
    reports_dir: Path = Field(
        default=Path("reports"),
        description="Directory to store generated reports",
    )

    # Database Configuration
    database_path: Path = Field(
        default=Path("data/research_agents.db"),
        description="Path to SQLite database file",
    )

    @property
    def api_key(self) -> str:
        """Alias for anthropic_api_key for convenience."""
        return self.anthropic_api_key

    @property
    def base_url(self) -> str:
        """Alias for anthropic_base_url for convenience."""
        return self.anthropic_base_url

    def get_api_url(self, endpoint: str) -> str:
        """Construct full API URL for an endpoint.

        Args:
            endpoint: API endpoint path (e.g., "/v1/messages")

        Returns:
            Full URL
        """
        base = self.anthropic_base_url.rstrip("/")
        endpoint = endpoint.lstrip("/")
        return f"{base}/{endpoint}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance (cached after first call)
    """
    return Settings()


def reload_settings() -> Settings:
    """Reload settings (clears cache).

    Returns:
        Fresh Settings instance
    """
    get_settings.cache_clear()
    return get_settings()
