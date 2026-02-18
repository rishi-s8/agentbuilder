"""Tests for Server module - FastAPI agent server with session management."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentbuilder.Server.base import create_agent_app


class TestCreateAgentApp:
    """Tests for create_agent_app() with session-based endpoints."""

    def _make_factory(self, run_return="Agent response"):
        """Return (factory, agent) where factory always returns the same mock agent."""
        agent = MagicMock()
        agent.run.return_value = run_return
        return lambda: agent, agent

    def _make_multi_factory(self):
        """Return a factory that creates a new mock agent on each call."""
        agents = []

        def factory():
            agent = MagicMock()
            agent.run.return_value = f"agent-{len(agents)}"
            agents.append(agent)
            return agent

        return factory, agents

    def test_returns_fastapi_app(self):
        """Test create_agent_app() returns a FastAPI app."""
        factory, _ = self._make_factory()
        app = create_agent_app(factory, name="test_agent", description="A test agent")

        assert isinstance(app, FastAPI)

    def test_info_endpoint(self):
        """Test GET /info returns agent name and description."""
        factory, _ = self._make_factory()
        app = create_agent_app(factory, name="math_agent", description="Solves math")
        client = TestClient(app)

        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "math_agent"
        assert data["description"] == "Solves math"

    def test_health_endpoint(self):
        """Test GET /health returns healthy status."""
        factory, _ = self._make_factory()
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_create_session_returns_201(self):
        """Test POST /sessions returns 201 with a session_id."""
        factory, _ = self._make_factory()
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app)

        response = client.post("/sessions")

        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0

    def test_create_session_factory_error_returns_500(self):
        """Test POST /sessions returns 500 when factory raises."""

        def bad_factory():
            raise RuntimeError("Factory exploded")

        app = create_agent_app(bad_factory, name="test", description="Test")
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/sessions")

        assert response.status_code == 500

    def test_session_run_calls_agent_run(self):
        """Test POST /sessions/{id}/run calls agent.run() with correct message."""
        factory, agent = self._make_factory(run_return="42")
        app = create_agent_app(factory, name="math", description="Math")
        client = TestClient(app)

        # Create session
        session_id = client.post("/sessions").json()["session_id"]

        # Run
        response = client.post(
            f"/sessions/{session_id}/run", json={"message": "What is 6 * 7?"}
        )

        assert response.status_code == 200
        assert response.json()["response"] == "42"
        agent.run.assert_called_once_with("What is 6 * 7?")

    def test_session_run_invalid_id_returns_404(self):
        """Test POST /sessions/{id}/run returns 404 for unknown session."""
        factory, _ = self._make_factory()
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app)

        response = client.post("/sessions/nonexistent/run", json={"message": "hello"})

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    def test_session_reset_calls_agent_reset(self):
        """Test POST /sessions/{id}/reset calls agent.reset()."""
        factory, agent = self._make_factory()
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app)

        session_id = client.post("/sessions").json()["session_id"]
        response = client.post(f"/sessions/{session_id}/reset")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        agent.reset.assert_called_once()

    def test_session_reset_invalid_id_returns_404(self):
        """Test POST /sessions/{id}/reset returns 404 for unknown session."""
        factory, _ = self._make_factory()
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app)

        response = client.post("/sessions/nonexistent/reset")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    def test_delete_session(self):
        """Test DELETE /sessions/{id} removes session; subsequent run returns 404."""
        factory, _ = self._make_factory()
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app)

        session_id = client.post("/sessions").json()["session_id"]

        # Delete
        response = client.delete(f"/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Subsequent run should 404
        response = client.post(f"/sessions/{session_id}/run", json={"message": "hello"})
        assert response.status_code == 404

    def test_delete_nonexistent_session_returns_404(self):
        """Test DELETE /sessions/{id} returns 404 for unknown session."""
        factory, _ = self._make_factory()
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app)

        response = client.delete("/sessions/nonexistent")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    def test_two_sessions_are_isolated(self):
        """Test that two sessions get their own agent instances."""
        factory, agents = self._make_multi_factory()
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app)

        sid1 = client.post("/sessions").json()["session_id"]
        sid2 = client.post("/sessions").json()["session_id"]

        assert sid1 != sid2
        assert len(agents) == 2

        # Run on session 1
        client.post(f"/sessions/{sid1}/run", json={"message": "msg1"})
        agents[0].run.assert_called_once_with("msg1")
        agents[1].run.assert_not_called()

        # Run on session 2
        client.post(f"/sessions/{sid2}/run", json={"message": "msg2"})
        agents[1].run.assert_called_once_with("msg2")

    def test_run_with_empty_message(self):
        """Test POST /sessions/{id}/run with empty message."""
        factory, agent = self._make_factory(run_return="")
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app)

        session_id = client.post("/sessions").json()["session_id"]
        response = client.post(f"/sessions/{session_id}/run", json={"message": ""})

        assert response.status_code == 200
        agent.run.assert_called_once_with("")

    def test_run_error_propagates(self):
        """Test POST /sessions/{id}/run propagates agent errors as 500."""
        factory, agent = self._make_factory()
        agent.run.side_effect = RuntimeError("Agent crashed")
        app = create_agent_app(factory, name="test", description="Test")
        client = TestClient(app, raise_server_exceptions=False)

        session_id = client.post("/sessions").json()["session_id"]
        response = client.post(f"/sessions/{session_id}/run", json={"message": "crash"})

        assert response.status_code == 500

    def test_neither_factory_raises(self):
        """Test create_agent_app() requires agent_factory argument."""
        with pytest.raises(TypeError):
            create_agent_app()
