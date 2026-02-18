"""Tests for Server module - FastAPI agent server."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentbuilder.Server.base import create_agent_app


class TestCreateAgentApp:
    """Tests for create_agent_app()."""

    def _make_mock_agent(self, run_return="Agent response"):
        agent = MagicMock()
        agent.run.return_value = run_return
        return agent

    def test_returns_fastapi_app(self):
        """Test create_agent_app() returns a FastAPI app."""
        agent = self._make_mock_agent()
        app = create_agent_app(agent, name="test_agent", description="A test agent")

        assert isinstance(app, FastAPI)

    def test_info_endpoint(self):
        """Test GET /info returns agent name and description."""
        agent = self._make_mock_agent()
        app = create_agent_app(agent, name="math_agent", description="Solves math")
        client = TestClient(app)

        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "math_agent"
        assert data["description"] == "Solves math"

    def test_run_endpoint(self):
        """Test POST /run calls agent.run() with correct message."""
        agent = self._make_mock_agent(run_return="42")
        app = create_agent_app(agent, name="math_agent", description="Solves math")
        client = TestClient(app)

        response = client.post("/run", json={"message": "What is 6 * 7?"})

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "42"
        agent.run.assert_called_once_with("What is 6 * 7?")

    def test_reset_endpoint(self):
        """Test POST /reset calls agent.reset()."""
        agent = self._make_mock_agent()
        app = create_agent_app(agent, name="test", description="Test")
        client = TestClient(app)

        response = client.post("/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        agent.reset.assert_called_once()

    def test_health_endpoint(self):
        """Test GET /health returns healthy status."""
        agent = self._make_mock_agent()
        app = create_agent_app(agent, name="test", description="Test")
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_run_with_empty_message(self):
        """Test POST /run with empty message."""
        agent = self._make_mock_agent(run_return="")
        app = create_agent_app(agent, name="test", description="Test")
        client = TestClient(app)

        response = client.post("/run", json={"message": ""})

        assert response.status_code == 200
        agent.run.assert_called_once_with("")

    def test_run_error_propagates(self):
        """Test POST /run propagates agent errors as 500."""
        agent = self._make_mock_agent()
        agent.run.side_effect = RuntimeError("Agent crashed")
        app = create_agent_app(agent, name="test", description="Test")
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/run", json={"message": "crash"})

        assert response.status_code == 500
