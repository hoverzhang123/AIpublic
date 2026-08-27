"""
Unit tests for src/config.py - Configuration management.

Tests the Config dataclass initialization, environment variable reading,
validation logic, and immutability.
"""

import os
from pathlib import Path

import pytest

from src.config import Config


class TestConfigInitialization:
    """Tests for Config class initialization and default values."""

    def test_config_has_default_values(self, tmp_path, monkeypatch):
        """Config initialization provides sensible defaults."""
        # Setup: create a valid FAQ directory
        faq_dir = tmp_path / "faq"
        faq_dir.mkdir()
        (faq_dir / "test.md").write_text("# Test")

        # Clear env vars that would override defaults
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LOCAL_API_KEY", raising=False)

        # Set minimal required env vars
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("MILVUS_DOCS_PATH", str(faq_dir))

        # Create config
        config = Config()

        # Verify defaults
        assert config.milvus_uri == "http://localhost:19530"
        assert config.collection_name == "my_rag_collection"
        assert config.max_retrievals == 3
        assert config.metric_type == "IP"
        assert config.consistency_level == "Strong"
        assert config.log_level == "INFO"
        assert config.llm_base_url == "https://openrouter.ai/api/v1"

    def test_config_reads_env_variables(self, tmp_path, monkeypatch):
        """Config reads values from environment variables."""
        # Setup
        faq_dir = tmp_path / "faq"
        faq_dir.mkdir()
        (faq_dir / "test.md").write_text("# Test")

        # Clear LOCAL_API_KEY since it takes priority over OPENAI_API_KEY
        monkeypatch.delenv("LOCAL_API_KEY", raising=False)

        # Set custom env vars
        monkeypatch.setenv("MILVUS_URI", "http://custom-milvus:19530")
        monkeypatch.setenv("MILVUS_COLLECTION", "custom_collection")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-custom-key")
        monkeypatch.setenv("MAX_RETRIEVALS", "10")
        monkeypatch.setenv("MILVUS_METRIC", "COSINE")
        monkeypatch.setenv("MILVUS_CONSISTENCY", "Eventually")
        monkeypatch.setenv("MILVUS_DOCS_PATH", str(faq_dir))

        # Create config
        config = Config()

        # Verify custom values
        assert config.milvus_uri == "http://custom-milvus:19530"
        assert config.collection_name == "custom_collection"
        assert config.openai_api_key == "sk-custom-key"
        assert config.max_retrievals == 10
        assert config.metric_type == "COSINE"
        assert config.consistency_level == "Eventually"

    def test_config_prefers_local_api_key_over_openai_key(self, tmp_path, monkeypatch):
        """Config prefers LOCAL_API_KEY if both LOCAL_API_KEY and OPENAI_API_KEY are set."""
        faq_dir = tmp_path / "faq"
        faq_dir.mkdir()
        (faq_dir / "test.md").write_text("# Test")

        monkeypatch.setenv("LOCAL_API_KEY", "sk-local-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
        monkeypatch.setenv("MILVUS_DOCS_PATH", str(faq_dir))

        config = Config()

        assert config.openai_api_key == "sk-local-key"

    def test_config_faq_path_is_pathlib_path(self, tmp_path, monkeypatch):
        """Config.faq_path is a Path object."""
        faq_dir = tmp_path / "faq"
        faq_dir.mkdir()
        (faq_dir / "test.md").write_text("# Test")

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("MILVUS_DOCS_PATH", str(faq_dir))

        config = Config()

        assert isinstance(config.faq_path, Path)
        assert config.faq_path == faq_dir


class TestConfigValidation:
    """Tests for Config.validate() method."""

    def test_validate_succeeds_with_valid_config(self, sample_config):
        """validate() succeeds when all required fields are set."""
        # sample_config fixture provides a valid config
        config = sample_config
        # Should not raise
        config.validate()

    def test_validate_raises_on_missing_api_key(self, tmp_path, monkeypatch):
        """validate() raises ValueError when API key is not set."""
        faq_dir = tmp_path / "faq"
        faq_dir.mkdir()
        (faq_dir / "test.md").write_text("# Test")

        # Clear both API key env vars
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LOCAL_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("MILVUS_DOCS_PATH", str(faq_dir))

        config = Config()

        with pytest.raises(ValueError, match="DeepSeek.*OpenAI.*API key"):
            config.validate()

    def test_validate_raises_on_missing_faq_directory(self, monkeypatch):
        """validate() raises FileNotFoundError when FAQ directory doesn't exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("MILVUS_DOCS_PATH", "/nonexistent/path/to/faq")

        config = Config()

        with pytest.raises(FileNotFoundError, match="FAQ source directory does not exist"):
            config.validate()

    def test_validate_raises_when_faq_path_is_not_directory(self, tmp_path, monkeypatch):
        """validate() raises FileNotFoundError when FAQ path is a file, not a directory."""
        # Create a file instead of directory
        faq_file = tmp_path / "faq.txt"
        faq_file.write_text("Not a directory")

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("MILVUS_DOCS_PATH", str(faq_file))

        config = Config()

        with pytest.raises(FileNotFoundError, match="FAQ source directory does not exist"):
            config.validate()


class TestConfigImmutability:
    """Tests for Config immutability (frozen dataclass)."""

    def test_config_is_immutable(self, sample_config):
        """Config is frozen and cannot be modified after creation."""
        config = sample_config

        with pytest.raises(Exception):  # frozen dataclass raises dataclass.FrozenInstanceError
            config.collection_name = "different_name"

    def test_config_frozen_prevents_new_attributes(self, sample_config):
        """Config frozen status prevents adding new attributes."""
        config = sample_config

        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            config.new_attribute = "value"


class TestConfigTypes:
    """Tests for Config field types."""

    def test_config_max_retrievals_is_int(self, tmp_path, monkeypatch):
        """max_retrievals is converted to int from env var string."""
        faq_dir = tmp_path / "faq"
        faq_dir.mkdir()
        (faq_dir / "test.md").write_text("# Test")

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("MAX_RETRIEVALS", "5")
        monkeypatch.setenv("MILVUS_DOCS_PATH", str(faq_dir))

        config = Config()

        assert isinstance(config.max_retrievals, int)
        assert config.max_retrievals == 5

    def test_config_string_fields_are_strings(self, sample_config):
        """String config fields are strings."""
        config = sample_config

        assert isinstance(config.milvus_uri, str)
        assert isinstance(config.collection_name, str)
        assert isinstance(config.openai_api_key, str)
        assert isinstance(config.openrouter_model, str)
        assert isinstance(config.metric_type, str)
        assert isinstance(config.consistency_level, str)
        assert isinstance(config.log_level, str)
        assert isinstance(config.llm_base_url, str)
