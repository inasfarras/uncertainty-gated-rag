"""Pytest configuration file."""

import pytest
from fastapi.testclient import TestClient

from agentic_rag.api.app import create_app


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)
