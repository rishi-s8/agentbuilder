"""Tests for RemoteAgentTool - remote sub-agent delegation via HTTP with sessions."""

from unittest.mock import MagicMock, call, patch

import pytest

from agentbuilder.Tools.base import Response
from agentbuilder.Tools.remote_agent_tool import RemoteAgentTool


def _mock_info_response(name="remote_agent", description="A remote agent"):
    """Create a mock response for GET /info."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"name": name, "description": description}
    resp.raise_for_status.return_value = None
    return resp


def _mock_session_response(session_id="abc123"):
    """Create a mock response for POST /sessions."""
    resp = MagicMock()
    resp.status_code = 201
    resp.json.return_value = {"session_id": session_id}
    resp.raise_for_status.return_value = None
    return resp


def _mock_run_response(response_text="Remote result"):
    """Create a mock response for POST /sessions/{id}/run."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": response_text}
    resp.raise_for_status.return_value = None
    return resp


def _mock_reset_response():
    """Create a mock response for POST /sessions/{id}/reset."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok"}
    resp.raise_for_status.return_value = None
    return resp


def _mock_delete_response():
    """Create a mock response for DELETE /sessions/{id}."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok"}
    resp.raise_for_status.return_value = None
    return resp


class TestRemoteAgentTool:
    """Tests for RemoteAgentTool class."""

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_fetches_info_on_init(self, mock_get, mock_post):
        """Test RemoteAgentTool fetches /info on init and populates name/description."""
        mock_get.return_value = _mock_info_response(
            name="math_agent", description="Solves math"
        )
        mock_post.return_value = _mock_session_response()

        tool = RemoteAgentTool(base_url="http://localhost:8100")

        mock_get.assert_called_once_with("http://localhost:8100/info")
        assert tool.name == "math_agent"
        assert tool.description == "Solves math"

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_creates_session_on_init(self, mock_get, mock_post):
        """Test RemoteAgentTool creates a session via POST /sessions on init."""
        mock_get.return_value = _mock_info_response()
        mock_post.return_value = _mock_session_response(session_id="sess42")

        tool = RemoteAgentTool(base_url="http://localhost:8100")

        mock_post.assert_called_once_with("http://localhost:8100/sessions")
        assert tool._session_id == "sess42"

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_strips_trailing_slash(self, mock_get, mock_post):
        """Test that trailing slashes in base_url are stripped."""
        mock_get.return_value = _mock_info_response()
        mock_post.return_value = _mock_session_response()

        tool = RemoteAgentTool(base_url="http://localhost:8100/")

        mock_get.assert_called_once_with("http://localhost:8100/info")
        assert tool.base_url == "http://localhost:8100"

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_delegate_resets_then_runs(self, mock_get, mock_post):
        """Test _delegate() resets session then runs task."""
        mock_get.return_value = _mock_info_response()
        # First post = create session, second = reset, third = run
        mock_post.side_effect = [
            _mock_session_response(session_id="s1"),
            _mock_reset_response(),
            _mock_run_response("42"),
        ]

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        result = tool._delegate(task="What is 6 * 7?")

        assert result == "42"
        # Verify call order: create session, reset, run
        assert mock_post.call_args_list == [
            call("http://localhost:8100/sessions"),
            call("http://localhost:8100/sessions/s1/reset"),
            call(
                "http://localhost:8100/sessions/s1/run",
                json={"message": "What is 6 * 7?"},
            ),
        ]

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_execute_returns_response(self, mock_get, mock_post):
        """Test execute() works through the Tool.execute() path."""
        mock_get.return_value = _mock_info_response()
        mock_post.side_effect = [
            _mock_session_response(),
            _mock_reset_response(),
            _mock_run_response("Done"),
        ]

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        response = tool.execute(task="Do something")

        assert isinstance(response, Response)
        assert response.success is True
        assert response.data == "Done"

    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_info_unavailable_raises(self, mock_get):
        """Test error handling when /info is unavailable."""
        from requests.exceptions import ConnectionError

        mock_get.side_effect = ConnectionError("Connection refused")

        with pytest.raises(ConnectionError):
            RemoteAgentTool(base_url="http://localhost:9999")

    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_info_non_200_raises(self, mock_get):
        """Test error handling when /info returns non-200."""
        from requests.exceptions import HTTPError

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_resp

        with pytest.raises(HTTPError):
            RemoteAgentTool(base_url="http://localhost:8100")

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_connection_error_on_run(self, mock_get, mock_post):
        """Test error handling when session run connection fails."""
        from requests.exceptions import ConnectionError

        mock_get.return_value = _mock_info_response()
        # Session creation succeeds, but reset/run fails
        mock_post.side_effect = [
            _mock_session_response(),
            ConnectionError("Connection refused"),
        ]

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        response = tool.execute(task="Do something")

        assert response.success is False
        assert "Connection refused" in response.error

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_to_openai_format(self, mock_get, mock_post):
        """Test OpenAI format output."""
        mock_get.return_value = _mock_info_response(
            name="helper", description="Helps"
        )
        mock_post.return_value = _mock_session_response()

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        fmt = tool.to_openai_format()

        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "helper"
        assert fmt["function"]["description"] == "Helps"
        assert "task" in fmt["function"]["parameters"]["properties"]

    @patch("agentbuilder.Tools.remote_agent_tool.requests.delete")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_close_deletes_session(self, mock_get, mock_post, mock_delete):
        """Test close() sends DELETE /sessions/{id}."""
        mock_get.return_value = _mock_info_response()
        mock_post.return_value = _mock_session_response(session_id="s1")
        mock_delete.return_value = _mock_delete_response()

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        mock_delete.reset_mock()  # clear any __del__ calls from GC'd tools
        tool.close()

        mock_delete.assert_called_once_with("http://localhost:8100/sessions/s1")
        assert tool._session_id is None

    @patch("agentbuilder.Tools.remote_agent_tool.requests.delete")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_close_is_idempotent(self, mock_get, mock_post, mock_delete):
        """Test close() can be called multiple times safely."""
        mock_get.return_value = _mock_info_response()
        mock_post.return_value = _mock_session_response(session_id="s1")
        mock_delete.return_value = _mock_delete_response()

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        tool.close()
        tool.close()  # second call should be a no-op

        mock_delete.assert_called_once()

    @patch("agentbuilder.Tools.remote_agent_tool.requests.delete")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_close_swallows_connection_error(self, mock_get, mock_post, mock_delete):
        """Test close() swallows exceptions from the DELETE call."""
        from requests.exceptions import ConnectionError

        mock_get.return_value = _mock_info_response()
        mock_post.return_value = _mock_session_response(session_id="s1")
        mock_delete.side_effect = ConnectionError("Connection refused")

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        tool.close()  # should not raise

        assert tool._session_id is None


class TestCreateRemoteAgentTool:
    """Tests for create_remote_agent_tool() factory."""

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_create_remote_agent_tool(self, mock_get, mock_post):
        """Test create_remote_agent_tool() creates a RemoteAgentTool with a session."""
        mock_get.return_value = _mock_info_response(
            name="remote_math", description="Remote math agent"
        )
        mock_post.return_value = _mock_session_response()

        from agentbuilder.utils import create_remote_agent_tool

        tool = create_remote_agent_tool(base_url="http://localhost:8100")

        assert isinstance(tool, RemoteAgentTool)
        assert tool.name == "remote_math"
        assert tool.description == "Remote math agent"
        assert tool._session_id is not None
