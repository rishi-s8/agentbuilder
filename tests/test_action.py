"""Tests for Action classes."""

import json
from unittest.mock import MagicMock, Mock

import pytest

from agentbuilder.Action.base import (Action, AssistantMessageAction,
                                      CompleteAction, EmptyAction,
                                      ExecuteToolsAction, MakeLLMRequestAction,
                                      SystemMessageAction, ToolMessageAction,
                                      UserMessageAction)
from agentbuilder.Tools.base import Response


class TestBaseAction:
    """Tests for base Action class."""

    def test_action_run_not_implemented(self):
        """Test that base Action.run() raises NotImplementedError."""
        action = Action()
        with pytest.raises(NotImplementedError):
            action.run()


class TestExecuteToolsAction:
    """Tests for ExecuteToolsAction."""

    def test_execute_single_tool_dict_format(
        self, sample_tool_map, mock_conversation_wrapper
    ):
        """Test executing a single tool call in dict format."""
        tool_calls = [
            {
                "id": "call_123",
                "function": {"name": "add", "arguments": json.dumps({"x": 5, "y": 3})},
            }
        ]

        action = ExecuteToolsAction(
            tool_calls=tool_calls,
            tool_map=sample_tool_map,
            conversation_wrapper=mock_conversation_wrapper,
            verbose=True,
        )

        action.run()

        # Verify tool message was added
        mock_conversation_wrapper.add_tool_message.assert_called_once()
        args = mock_conversation_wrapper.add_tool_message.call_args[0]
        assert args[0] == "call_123"
        assert args[1] == "add"
        result = json.loads(args[2])
        assert result["success"] is True
        assert result["data"] == 8

    def test_execute_single_tool_object_format(
        self, sample_tool_map, mock_conversation_wrapper
    ):
        """Test executing a single tool call in object format."""
        tool_call = Mock()
        tool_call.id = "call_456"
        tool_call.function.name = "add"
        tool_call.function.arguments = json.dumps({"x": 10, "y": 20})

        action = ExecuteToolsAction(
            tool_calls=[tool_call],
            tool_map=sample_tool_map,
            conversation_wrapper=mock_conversation_wrapper,
            verbose=True,
        )

        action.run()

        # Verify tool message was added
        mock_conversation_wrapper.add_tool_message.assert_called_once()
        args = mock_conversation_wrapper.add_tool_message.call_args[0]
        assert args[0] == "call_456"
        assert args[1] == "add"
        result = json.loads(args[2])
        assert result["success"] is True
        assert result["data"] == 30

    def test_execute_tool_not_found(self, mock_conversation_wrapper):
        """Test executing a tool that doesn't exist."""
        tool_calls = [
            {
                "id": "call_789",
                "function": {"name": "nonexistent_tool", "arguments": json.dumps({})},
            }
        ]

        action = ExecuteToolsAction(
            tool_calls=tool_calls,
            tool_map={},
            conversation_wrapper=mock_conversation_wrapper,
            verbose=True,
        )

        action.run()

        # Verify error message was added
        mock_conversation_wrapper.add_tool_message.assert_called_once()
        args = mock_conversation_wrapper.add_tool_message.call_args[0]
        result = json.loads(args[2])
        assert "error" in result
        assert "not found" in result["error"]

    def test_execute_multiple_tools(self, sample_tool_map, mock_conversation_wrapper):
        """Test executing multiple tool calls."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "add", "arguments": json.dumps({"x": 1, "y": 2})},
            },
            {
                "id": "call_2",
                "function": {"name": "add", "arguments": json.dumps({"x": 3, "y": 4})},
            },
        ]

        action = ExecuteToolsAction(
            tool_calls=tool_calls,
            tool_map=sample_tool_map,
            conversation_wrapper=mock_conversation_wrapper,
            verbose=True,
        )

        action.run()

        # Verify both tool messages were added
        assert mock_conversation_wrapper.add_tool_message.call_count == 2

    def test_execute_tools_verbose(
        self, sample_tool_map, mock_conversation_wrapper, capsys
    ):
        """Test verbose output during tool execution."""
        tool_calls = [
            {
                "id": "call_999",
                "function": {"name": "add", "arguments": json.dumps({"x": 7, "y": 8})},
            }
        ]

        action = ExecuteToolsAction(
            tool_calls=tool_calls,
            tool_map=sample_tool_map,
            conversation_wrapper=mock_conversation_wrapper,
            verbose=True,
        )

        action.run()

        captured = capsys.readouterr()
        assert "Executing" in captured.out
        assert "add" in captured.out


class TestMakeLLMRequestAction:
    """Tests for MakeLLMRequestAction."""

    def test_make_llm_request_without_tools(self, mock_conversation_wrapper):
        """Test making an LLM request without tools."""
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_conversation_wrapper.client.chat.completions.create.return_value = (
            mock_response
        )

        action = MakeLLMRequestAction(
            conversation_wrapper=mock_conversation_wrapper, tool_map={}, verbose=True
        )

        action.run()

        # Verify LLM request was made
        mock_conversation_wrapper.client.chat.completions.create.assert_called_once()
        # Verify assistant message was added
        mock_conversation_wrapper.add_assistant_message.assert_called_once()

    def test_make_llm_request_with_tools(
        self, mock_conversation_wrapper, sample_tool_map
    ):
        """Test making an LLM request with tools available."""
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_conversation_wrapper.client.chat.completions.create.return_value = (
            mock_response
        )

        action = MakeLLMRequestAction(
            conversation_wrapper=mock_conversation_wrapper,
            tool_map=sample_tool_map,
            verbose=True,
        )

        action.run()

        # Verify tools were passed in the request
        call_kwargs = (
            mock_conversation_wrapper.client.chat.completions.create.call_args[1]
        )
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert call_kwargs["tool_choice"] == "auto"

    def test_make_llm_request_with_tool_calls_response(
        self, mock_conversation_wrapper, sample_tool_map
    ):
        """Test handling LLM response with tool calls."""
        mock_tool_call = Mock()
        mock_tool_call.model_dump.return_value = {
            "id": "call_1",
            "function": {"name": "add"},
        }

        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_response.choices = [Mock(message=mock_message)]
        mock_conversation_wrapper.client.chat.completions.create.return_value = (
            mock_response
        )

        action = MakeLLMRequestAction(
            conversation_wrapper=mock_conversation_wrapper,
            tool_map=sample_tool_map,
            verbose=True,
        )

        action.run()

        # Verify assistant message with tool calls was added
        mock_conversation_wrapper.add_assistant_message.assert_called_once()
        call_args = mock_conversation_wrapper.add_assistant_message.call_args
        assert call_args[0][0] is None  # content
        assert call_args[1]["tool_calls"] is not None


class TestCompleteAction:
    """Tests for CompleteAction."""

    def test_complete_action_returns_content(self):
        """Test that CompleteAction returns the content."""
        action = CompleteAction(content="Final answer", iterations=5, verbose=True)
        result = action.run()
        assert result == "Final answer"

    def test_complete_action_verbose(self, capsys):
        """Test verbose output for CompleteAction."""
        action = CompleteAction(content="Done", iterations=3, verbose=True)
        action.run()

        captured = capsys.readouterr()
        assert "Completed" in captured.out
        assert "3" in captured.out


class TestEmptyAction:
    """Tests for EmptyAction."""

    def test_empty_action_returns_empty_string(self):
        """Test that EmptyAction returns empty string."""
        action = EmptyAction(iterations=0, verbose=True)
        result = action.run()
        assert result == ""

    def test_empty_action_verbose(self, capsys):
        """Test verbose output for EmptyAction."""
        action = EmptyAction(iterations=1, verbose=True)
        action.run()

        captured = capsys.readouterr()
        assert "Completed" in captured.out
        assert "empty" in captured.out


class TestSystemMessageAction:
    """Tests for SystemMessageAction."""

    def test_to_message(self):
        """Test converting SystemMessageAction to message format."""
        action = SystemMessageAction(content="You are a helpful assistant")
        message = action.to_message()

        assert message["role"] == "system"
        assert message["content"] == "You are a helpful assistant"

    def test_from_message(self):
        """Test creating SystemMessageAction from message dict."""
        msg = {"role": "system", "content": "Test system message"}
        action = SystemMessageAction.from_message(msg)

        assert action.content == "Test system message"


class TestUserMessageAction:
    """Tests for UserMessageAction."""

    def test_to_message(self):
        """Test converting UserMessageAction to message format."""
        action = UserMessageAction(content="Hello!")
        message = action.to_message()

        assert message["role"] == "user"
        assert message["content"] == "Hello!"

    def test_from_message(self):
        """Test creating UserMessageAction from message dict."""
        msg = {"role": "user", "content": "Test user message"}
        action = UserMessageAction.from_message(msg)

        assert action.content == "Test user message"


class TestAssistantMessageAction:
    """Tests for AssistantMessageAction."""

    def test_to_message_without_tool_calls(self):
        """Test converting AssistantMessageAction to message format without tool calls."""
        action = AssistantMessageAction(content="Hello there!")
        message = action.to_message()

        assert message["role"] == "assistant"
        assert message["content"] == "Hello there!"
        assert "tool_calls" not in message

    def test_to_message_with_tool_calls(self):
        """Test converting AssistantMessageAction to message format with tool calls."""
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        action = AssistantMessageAction(content=None, tool_calls=tool_calls)
        message = action.to_message()

        assert message["role"] == "assistant"
        assert message["content"] is None
        assert message["tool_calls"] == tool_calls

    def test_from_message_without_tool_calls(self):
        """Test creating AssistantMessageAction from message dict without tool calls."""
        msg = {"role": "assistant", "content": "Test response"}
        action = AssistantMessageAction.from_message(msg)

        assert action.content == "Test response"
        assert action.tool_calls is None

    def test_from_message_with_tool_calls(self):
        """Test creating AssistantMessageAction from message dict with tool calls."""
        tool_calls = [{"id": "call_1"}]
        msg = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        action = AssistantMessageAction.from_message(msg)

        assert action.content is None
        assert action.tool_calls == tool_calls


class TestToolMessageAction:
    """Tests for ToolMessageAction."""

    def test_to_message(self):
        """Test converting ToolMessageAction to message format."""
        action = ToolMessageAction(
            tool_call_id="call_123", name="add", content='{"success": true, "data": 5}'
        )
        message = action.to_message()

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_123"
        assert message["name"] == "add"
        assert message["content"] == '{"success": true, "data": 5}'

    def test_from_message(self):
        """Test creating ToolMessageAction from message dict."""
        msg = {
            "role": "tool",
            "tool_call_id": "call_456",
            "name": "multiply",
            "content": '{"success": false}',
        }
        action = ToolMessageAction.from_message(msg)

        assert action.tool_call_id == "call_456"
        assert action.name == "multiply"
        assert action.content == '{"success": false}'
