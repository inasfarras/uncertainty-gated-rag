"""Tests for configuration settings."""

import pytest
from agentic_rag.config import Settings


def test_default_settings() -> None:
    """Test that default settings are correctly set."""
    settings = Settings()

    # LLM Configuration
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.embed_model == "sentence-transformers/all-MiniLM-L6-v2"

    # Token and Round Limits
    assert settings.max_tokens_total == 3500
    assert settings.max_rounds == 2
    assert settings.low_budget_tokens == 500

    # Retrieval Configuration
    assert settings.retrieval_k == 8
    assert settings.graph_k == 20
    assert settings.use_rerank is True

    # Quality Thresholds
    assert settings.faithfulness_tau == 0.75
    assert settings.overlap_tau == 0.50

    # Directory Paths
    assert settings.data_dir == "./data/corpus"
    assert settings.index_dir == "./artifacts/faiss"
    assert settings.log_dir == "./artifacts/logs"
    assert settings.report_dir == "./artifacts/reports"

    # System Configuration
    assert settings.log_level == "INFO"
    assert settings.seed == 42


def test_llm_provider_validation() -> None:
    """Test that LLM provider validation works correctly."""
    # Valid providers
    settings = Settings(llm_provider="openai")
    assert settings.llm_provider == "openai"

    settings = Settings(llm_provider="ollama")
    assert settings.llm_provider == "ollama"

    settings = Settings(llm_provider="local")
    assert settings.llm_provider == "local"

    # Invalid provider should raise validation error
    with pytest.raises(ValueError):
        Settings(llm_provider="invalid_provider")  # type: ignore[arg-type]


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that settings can be loaded from environment variables."""
    # Set environment variables
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama2")
    monkeypatch.setenv("MAX_TOKENS_TOTAL", "5000")
    monkeypatch.setenv("RETRIEVAL_K", "10")
    monkeypatch.setenv("USE_RERANK", "false")
    monkeypatch.setenv("FAITHFULNESS_TAU", "0.8")
    monkeypatch.setenv("SEED", "123")

    settings = Settings()

    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama2"
    assert settings.max_tokens_total == 5000
    assert settings.retrieval_k == 10
    assert settings.use_rerank is False
    assert settings.faithfulness_tau == 0.8
    assert settings.seed == 123


def test_case_insensitive_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables are case insensitive."""
    monkeypatch.setenv("llm_provider", "local")
    monkeypatch.setenv("MAX_ROUNDS", "3")

    settings = Settings()

    assert settings.llm_provider == "local"
    assert settings.max_rounds == 3


def test_type_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables are properly converted to correct types."""
    monkeypatch.setenv("MAX_TOKENS_TOTAL", "4000")
    monkeypatch.setenv("FAITHFULNESS_TAU", "0.85")
    monkeypatch.setenv("USE_RERANK", "true")
    monkeypatch.setenv("SEED", "999")

    settings = Settings()

    assert isinstance(settings.max_tokens_total, int)
    assert settings.max_tokens_total == 4000

    assert isinstance(settings.faithfulness_tau, float)
    assert settings.faithfulness_tau == 0.85

    assert isinstance(settings.use_rerank, bool)
    assert settings.use_rerank is True

    assert isinstance(settings.seed, int)
    assert settings.seed == 999


def test_invalid_type_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that invalid type conversions raise appropriate errors."""
    monkeypatch.setenv("MAX_TOKENS_TOTAL", "not_a_number")

    with pytest.raises(ValueError):
        Settings()


def test_settings_singleton_import() -> None:
    """Test that the singleton settings instance can be imported."""
    from agentic_rag.config import settings

    assert isinstance(settings, Settings)
    assert settings.llm_provider == "openai"  # Default value
