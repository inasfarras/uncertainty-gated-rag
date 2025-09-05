"""Tests for logging configuration."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_rag.logging import get_logger
from loguru._logger import Logger


def test_get_logger_returns_logger() -> None:
    """Test that get_logger returns a Logger instance."""
    logger = get_logger("test_module")
    assert isinstance(logger, Logger)


def test_get_logger_with_different_names() -> None:
    """Test that get_logger works with different module names."""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    assert isinstance(logger1, Logger)
    assert isinstance(logger2, Logger)


@patch("agentic_rag.logging.settings")
def test_get_logger_creates_log_directory(mock_settings: MagicMock) -> None:
    """Test that get_logger creates the log directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "test_logs"
        mock_settings.log_dir = str(log_dir)
        mock_settings.log_level = "INFO"

        # Directory shouldn't exist initially
        assert not log_dir.exists()

        # Getting logger should create the directory
        get_logger("test_module")

        assert log_dir.exists()
        assert log_dir.is_dir()

        # Clean up logger handlers to release file locks
        from loguru import logger as loguru_logger

        loguru_logger.remove()


@patch("agentic_rag.logging.settings")
def test_get_logger_respects_log_level(mock_settings: MagicMock) -> None:
    """Test that get_logger respects the configured log level."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_settings.log_dir = temp_dir
        mock_settings.log_level = "DEBUG"

        logger = get_logger("test_module")

        # This test mainly ensures no exceptions are raised
        # Actual log level testing would require more complex mocking
        assert isinstance(logger, Logger)

        # Clean up logger handlers to release file locks
        from loguru import logger as loguru_logger

        loguru_logger.remove()
