"""Tests for error scenarios and failure handling."""

import json
from unittest.mock import Mock, patch

import pytest
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

from agentbuilder.Action.base import ExecuteToolsAction
from agentbuilder.Client.openai_client import ConversationWrapper
from agentbuilder.Loop.base import AgenticLoop
from agentbuilder.Planner.base import AgenticPlanner
from agentbuilder.Tools.base import Tool


class TestOpenAIAPIErrors:
    """Tests for OpenAI API error handling."""

    @patch("agentbuilder.Client.openai_client.OpenAI")
    def test_send_message_rate_limit_error(self, mock_openai):
        """Test handling of rate limit errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=Mock(status_code=429), body=None
        )
        mock_openai.return_value = mock_client

        wrapper = ConversationWrapper(api_key="test_key", model="gpt-4")

        with pytest.raises(RateLimitError):
            wrapper.send_message("Hello")

    @patch("agentbuilder.Client.openai_client.OpenAI")
    def test_send_message_authentication_error(self, mock_openai):
        """Test handling of authentication errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key", response=Mock(status_code=401), body=None
        )
        mock_openai.return_value = mock_client

        wrapper = ConversationWrapper(api_key="invalid_key", model="gpt-4")

        with pytest.raises(AuthenticationError):
            wrapper.send_message("Hello")

    @patch("agentbuilder.Client.openai_client.OpenAI")
    def test_send_message_api_error(self, mock_openai):
        """Test handling of general API errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = APIError(
            "Internal server error", request=Mock(), body=None
        )
        mock_openai.return_value = mock_client

        wrapper = ConversationWrapper(api_key="test_key", model="gpt-4")

        with pytest.raises(APIError):
            wrapper.send_message("Hello")

    @patch("agentbuilder.Client.openai_client.OpenAI")
    def test_send_message_connection_error(self, mock_openai):
        """Test handling of connection errors."""
        mock_client = Mock()
        # Use a generic exception to simulate connection errors
        # (APIConnectionError has a complex signature)
        mock_client.chat.completions.create.side_effect = ConnectionError(
            "Connection failed"
        )
        mock_openai.return_value = mock_client

        wrapper = ConversationWrapper(api_key="test_key", model="gpt-4")

        with pytest.raises(ConnectionError):
            wrapper.send_message("Hello")


class TestMalformedResponses:
    """Tests for handling malformed API responses."""

    @patch("agentbuilder.Client.openai_client.OpenAI")
    def test_send_message_empty_response(self, mock_openai):
        """Test handling of empty response from API."""
        mock_response = Mock()
        mock_response.choices = []

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        wrapper = ConversationWrapper(api_key="test_key", model="gpt-4")

        # Should raise IndexError when trying to access choices[0]
        with pytest.raises(IndexError):
            wrapper.send_message("Hello")

    @patch("agentbuilder.Client.openai_client.OpenAI")
    def test_send_message_none_content(self, mock_openai):
        """Test handling of None content in response."""
        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = None

        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        wrapper = ConversationWrapper(api_key="test_key", model="gpt-4")
        result = wrapper.send_message("Hello")

        # Should handle None content gracefully
        assert result is None
        assert len(wrapper.conversation_history) == 2


class TestToolExecutionErrors:
    """Tests for tool execution error scenarios."""

    def test_tool_execution_raises_exception(self, mock_conversation_wrapper):
        """Test tool execution when function raises an exception."""

        def failing_function(x: int) -> int:
            raise ValueError("Something went wrong")

        tool = Tool(
            name="failing_tool",
            description="A tool that fails",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            function=failing_function,
        )

        tool_calls = [
            {
                "id": "call_123",
                "function": {"name": "failing_tool", "arguments": json.dumps({"x": 5})},
            }
        ]

        action = ExecuteToolsAction(
            tool_calls=tool_calls,
            tool_map={"failing_tool": tool},
            conversation_wrapper=mock_conversation_wrapper,
            verbose=False,
        )

        # Should not raise - should catch exception and add error to conversation
        action.run()

        # Verify error was added to conversation
        mock_conversation_wrapper.add_tool_message.assert_called_once()
        args = mock_conversation_wrapper.add_tool_message.call_args[0]
        result = json.loads(args[2])
        assert result["success"] is False
        assert "Something went wrong" in result["error"]

    def test_tool_execution_with_type_error(self, mock_conversation_wrapper):
        """Test tool execution with wrong argument types."""

        def strict_function(x: int) -> int:
            return x + 1

        tool = Tool(
            name="strict_tool",
            description="A tool with strict types",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            function=strict_function,
        )

        # Pass string instead of integer
        tool_calls = [
            {
                "id": "call_456",
                "function": {
                    "name": "strict_tool",
                    "arguments": json.dumps({"x": "not_a_number"}),
                },
            }
        ]

        action = ExecuteToolsAction(
            tool_calls=tool_calls,
            tool_map={"strict_tool": tool},
            conversation_wrapper=mock_conversation_wrapper,
            verbose=False,
        )

        # Should catch TypeError and return error response
        action.run()

        mock_conversation_wrapper.add_tool_message.assert_called_once()
        args = mock_conversation_wrapper.add_tool_message.call_args[0]
        result = json.loads(args[2])
        assert result["success"] is False
        assert result["error"] is not None

    def test_tool_execution_with_invalid_json_arguments(
        self, mock_conversation_wrapper
    ):
        """Test tool execution with invalid JSON in arguments."""
        tool = Tool(
            name="test_tool",
            description="Test tool",
            parameters={"type": "object"},
            function=lambda: "test",
        )

        tool_calls = [
            {
                "id": "call_789",
                "function": {"name": "test_tool", "arguments": "invalid json {{"},
            }
        ]

        action = ExecuteToolsAction(
            tool_calls=tool_calls,
            tool_map={"test_tool": tool},
            conversation_wrapper=mock_conversation_wrapper,
            verbose=False,
        )

        # Should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            action.run()


class TestLoopErrorRecovery:
    """Tests for error recovery in agentic loop."""

    @patch("agentbuilder.Client.openai_client.OpenAI")
    def test_loop_continues_after_tool_failure(self, mock_openai):
        """Test that loop can continue after a tool execution fails."""
        # This would require a more complex setup with real planner
        # Just documenting the scenario for now
        pass

    def test_loop_max_iterations_with_failing_tool(
        self, mock_conversation_wrapper, sample_tool_map
    ):
        """Test that loop respects max iterations even with failing tools."""
        planner = Mock()

        # Mock planner to always return an action that doesn't complete
        mock_action = Mock()
        mock_action.run = Mock(return_value=None)
        planner.step.return_value = mock_action

        loop = AgenticLoop(
            mock_conversation_wrapper,
            planner,
            sample_tool_map,
            verbose=False,
            max_iterations=3,
        )

        result = loop.run("Test")

        # Should stop after 3 iterations
        assert planner.step.call_count == 3
        assert result == "Max iterations reached"


class TestConversationLoadErrors:
    """Tests for conversation load/save error scenarios."""

    def test_load_conversation_file_not_found(self, base_conversation):
        """Test loading conversation from non-existent file."""
        with pytest.raises(FileNotFoundError):
            base_conversation.load_conversation("/nonexistent/path.json")

    def test_load_conversation_invalid_json(self, base_conversation, tmp_path):
        """Test loading conversation from file with invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json {{")

        with pytest.raises(json.JSONDecodeError):
            base_conversation.load_conversation(str(invalid_file))

    def test_load_conversation_missing_role(self, base_conversation, tmp_path):
        """Test loading conversation with messages missing 'role' field."""
        invalid_messages = [{"content": "test", "no_role_field": "user"}]
        invalid_file = tmp_path / "invalid_format.json"
        invalid_file.write_text(json.dumps(invalid_messages))

        # Should raise KeyError when role field is missing
        with pytest.raises(KeyError):
            base_conversation.load_conversation(str(invalid_file))


# Import fixtures from conftest
@pytest.fixture
def base_conversation():
    """Base conversation wrapper for testing."""
    from agentbuilder.Client.base import BaseConversationWrapper

    return BaseConversationWrapper()


@pytest.fixture
def mock_conversation_wrapper():
    """Mock conversation wrapper."""
    from agentbuilder.Client.base import BaseConversationWrapper

    wrapper = Mock(spec=BaseConversationWrapper)
    wrapper.conversation_history = []
    wrapper.add_tool_message = Mock()
    return wrapper


@pytest.fixture
def sample_tool_map():
    """Sample tool map."""
    from agentbuilder.Tools.base import Tool

    tool = Tool(
        name="test", description="Test tool", parameters={}, function=lambda: "test"
    )
    return {"test": tool}
