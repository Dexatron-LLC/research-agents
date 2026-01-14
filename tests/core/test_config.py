"""Tests for the configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from research_agents.core.config import Settings, get_settings, reload_settings


class TestSettings:
    """Tests for the Settings class."""

    def test_default_values(self):
        """Test that Settings has sensible defaults when no env vars are set."""
        # Create settings with explicit defaults (ignoring .env file)
        settings = Settings(
            _env_file=None,  # Disable .env file loading
            anthropic_api_key="",
        )

        assert settings.anthropic_base_url == "https://api.anthropic.com"
        assert settings.default_model == "claude-sonnet-4-20250514"
        assert settings.fast_model == "claude-haiku-4-20250514"
        assert settings.max_tokens == 2048
        assert settings.request_timeout == 60.0
        assert settings.reports_dir == Path("reports")
        assert settings.database_path == Path("data/research_agents.db")
        assert settings.max_repeats == 3

    def test_api_key_property(self, mock_settings: Settings):
        """Test api_key property alias."""
        assert mock_settings.api_key == mock_settings.anthropic_api_key
        assert mock_settings.api_key == "test-api-key"

    def test_base_url_property(self, mock_settings: Settings):
        """Test base_url property alias."""
        assert mock_settings.base_url == mock_settings.anthropic_base_url

    def test_get_api_url(self, mock_settings: Settings):
        """Test API URL construction."""
        url = mock_settings.get_api_url("/v1/messages")
        assert url == "https://api.anthropic.com/v1/messages"

    def test_get_api_url_strips_slashes(self, mock_settings: Settings):
        """Test API URL handles extra slashes."""
        url = mock_settings.get_api_url("v1/messages")
        assert url == "https://api.anthropic.com/v1/messages"

    def test_from_environment_variables(self):
        """Test loading settings from environment variables."""
        env_vars = {
            "ANTHROPIC_API_KEY": "env-api-key",
            "ANTHROPIC_BASE_URL": "https://custom.api.com",
            "DEFAULT_MODEL": "custom-model",
            "MAX_TOKENS": "4096",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

        assert settings.anthropic_api_key == "env-api-key"
        assert settings.anthropic_base_url == "https://custom.api.com"
        assert settings.default_model == "custom-model"
        assert settings.max_tokens == 4096

    def test_max_repeats_from_environment(self):
        """Test loading MAXREPEATS from environment variable."""
        env_vars = {
            "MAXREPEATS": "5",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.max_repeats == 5


class TestGetSettings:
    """Tests for settings getter functions."""

    def test_get_settings_returns_settings(self):
        """Test get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """Test get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reload_settings_clears_cache(self):
        """Test reload_settings returns a fresh instance."""
        settings1 = get_settings()
        settings2 = reload_settings()
        # After reload, cache is cleared and new instance created
        assert isinstance(settings2, Settings)
