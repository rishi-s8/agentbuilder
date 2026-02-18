"""Tests for RemoteAgentTool - remote sub-agent delegation via HTTP."""

from unittest.mock import MagicMock, patch

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


def _mock_run_response(response_text="Remote result"):
    """Create a mock response for POST /run."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": response_text}
    resp.raise_for_status.return_value = None
    return resp


class TestRemoteAgentTool:
    """Tests for RemoteAgentTool class."""

    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_fetches_info_on_init(self, mock_get):
        """Test RemoteAgentTool fetches /info on init and populates name/description."""
        mock_get.return_value = _mock_info_response(
            name="math_agent", description="Solves math"
        )

        tool = RemoteAgentTool(base_url="http://localhost:8100")

        mock_get.assert_called_once_with("http://localhost:8100/info")
        assert tool.name == "math_agent"
        assert tool.description == "Solves math"

    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_strips_trailing_slash(self, mock_get):
        """Test that trailing slashes in base_url are stripped."""
        mock_get.return_value = _mock_info_response()

        tool = RemoteAgentTool(base_url="http://localhost:8100/")

        mock_get.assert_called_once_with("http://localhost:8100/info")
        assert tool.base_url == "http://localhost:8100"

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_delegate_sends_post_to_run(self, mock_get, mock_post):
        """Test _delegate() sends correct POST to /run and returns response."""
        mock_get.return_value = _mock_info_response()
        mock_post.return_value = _mock_run_response("42")

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        result = tool._delegate(task="What is 6 * 7?")

        mock_post.assert_called_once_with(
            "http://localhost:8100/run",
            json={"message": "What is 6 * 7?"},
        )
        assert result == "42"

    @patch("agentbuilder.Tools.remote_agent_tool.requests.post")
    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_execute_returns_response(self, mock_get, mock_post):
        """Test execute() works through the Tool.execute() path."""
        mock_get.return_value = _mock_info_response()
        mock_post.return_value = _mock_run_response("Done")

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
        """Test error handling when /run connection fails."""
        from requests.exceptions import ConnectionError

        mock_get.return_value = _mock_info_response()
        mock_post.side_effect = ConnectionError("Connection refused")

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        response = tool.execute(task="Do something")

        assert response.success is False
        assert "Connection refused" in response.error

    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_to_openai_format(self, mock_get):
        """Test OpenAI format output."""
        mock_get.return_value = _mock_info_response(
            name="helper", description="Helps"
        )

        tool = RemoteAgentTool(base_url="http://localhost:8100")
        fmt = tool.to_openai_format()

        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "helper"
        assert fmt["function"]["description"] == "Helps"
        assert "task" in fmt["function"]["parameters"]["properties"]


class TestCreateRemoteAgentTool:
    """Tests for create_remote_agent_tool() factory."""

    @patch("agentbuilder.Tools.remote_agent_tool.requests.get")
    def test_create_remote_agent_tool(self, mock_get):
        """Test create_remote_agent_tool() creates a RemoteAgentTool."""
        mock_get.return_value = _mock_info_response(
            name="remote_math", description="Remote math agent"
        )

        from agentbuilder.utils import create_remote_agent_tool

        tool = create_remote_agent_tool(base_url="http://localhost:8100")

        assert isinstance(tool, RemoteAgentTool)
        assert tool.name == "remote_math"
        assert tool.description == "Remote math agent"
