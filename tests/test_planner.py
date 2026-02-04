"""Tests for AgenticPlanner."""

from unittest.mock import Mock

import pytest

from agentbuilder.Action.base import (AssistantMessageAction, CompleteAction,
                                      EmptyAction, ExecuteToolsAction,
                                      MakeLLMRequestAction,
                                      SystemMessageAction, ToolMessageAction,
                                      UserMessageAction)
from agentbuilder.Planner.base import AgenticPlanner


class TestAgenticPlanner:
    """Tests for AgenticPlanner."""

    @pytest.fixture
    def mock_conversation_wrapper(self):
        """Mock conversation wrapper."""
        wrapper = Mock()
        wrapper.conversation_history = []
        return wrapper

    @pytest.fixture
    def sample_tool_map(self, sample_tool):
        """Sample tool map."""
        return {sample_tool.name: sample_tool}

    @pytest.fixture
    def planner(self, mock_conversation_wrapper, sample_tool_map):
        """Create a planner instance."""
        return AgenticPlanner(mock_conversation_wrapper, sample_tool_map, verbose=False)

    def test_init(self, mock_conversation_wrapper, sample_tool_map):
        """Test planner initialization."""
        planner = AgenticPlanner(
            mock_conversation_wrapper, sample_tool_map, verbose=True
        )

        assert planner.conversation_wrapper == mock_conversation_wrapper
        assert planner.tool_map == sample_tool_map
        assert planner.verbose is True

    def test_reset(self, planner):
        """Test reset method (currently no-op)."""
        planner.reset()  # Should not raise any errors

    def test_step_empty_history(self, planner):
        """Test step with empty conversation history."""
        action = planner.step([], iterations=0)

        assert isinstance(action, EmptyAction)
        assert action.iterations == 0

    def test_step_user_message(self, planner, mock_conversation_wrapper):
        """Test step when last message is from user."""
        history = [UserMessageAction(content="Hello")]

        action = planner.step(history, iterations=1)

        assert isinstance(action, MakeLLMRequestAction)
        assert action.conversation_wrapper == mock_conversation_wrapper
        assert action.verbose is False

    def test_step_tool_message(self, planner, mock_conversation_wrapper):
        """Test step when last message is a tool response."""
        history = [
            UserMessageAction(content="Hello"),
            AssistantMessageAction(content=None, tool_calls=[{"id": "call_1"}]),
            ToolMessageAction(tool_call_id="call_1", name="test", content="{}"),
        ]

        action = planner.step(history, iterations=2)

        assert isinstance(action, MakeLLMRequestAction)
        assert action.conversation_wrapper == mock_conversation_wrapper

    def test_step_assistant_message_with_tool_calls(self, planner):
        """Test step when assistant requests tool calls."""
        tool_calls = [
            {
                "id": "call_123",
                "function": {"name": "add", "arguments": '{"x": 1, "y": 2}'},
            }
        ]
        history = [
            UserMessageAction(content="Add 1 and 2"),
            AssistantMessageAction(content=None, tool_calls=tool_calls),
        ]

        action = planner.step(history, iterations=1)

        assert isinstance(action, ExecuteToolsAction)
        assert action.tool_calls == tool_calls
        assert action.tool_map == planner.tool_map

    def test_step_assistant_message_with_content(self, planner):
        """Test step when assistant provides final response."""
        history = [
            UserMessageAction(content="Hello"),
            AssistantMessageAction(content="Hi there!", tool_calls=None),
        ]

        action = planner.step(history, iterations=3)

        assert isinstance(action, CompleteAction)
        assert action.content == "Hi there!"
        assert action.iterations == 3

    def test_step_system_message_only(self, planner):
        """Test step with only system message (unhandled state)."""
        history = [SystemMessageAction(content="System prompt")]

        with pytest.raises(NotImplementedError, match="Unhandled conversation state"):
            planner.step(history, iterations=1)

    def test_step_assistant_message_no_content_no_tools(self, planner):
        """Test step with assistant message that has neither content nor tool calls."""
        history = [
            UserMessageAction(content="Test"),
            AssistantMessageAction(content=None, tool_calls=None),
        ]

        with pytest.raises(NotImplementedError, match="Unhandled conversation state"):
            planner.step(history, iterations=1)

    def test_step_verbose_mode(
        self, mock_conversation_wrapper, sample_tool_map, capsys
    ):
        """Test verbose output during step."""
        planner = AgenticPlanner(
            mock_conversation_wrapper, sample_tool_map, verbose=True
        )

        history = [UserMessageAction(content="Test")]
        planner.step(history, iterations=1)

        captured = capsys.readouterr()
        assert "Analyzing conversation" in captured.out
        assert "1 messages" in captured.out

    def test_step_multiple_tool_calls(self, planner):
        """Test step with multiple tool calls."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "add", "arguments": '{"x": 1, "y": 2}'},
            },
            {
                "id": "call_2",
                "function": {"name": "add", "arguments": '{"x": 3, "y": 4}'},
            },
        ]
        history = [
            UserMessageAction(content="Do some math"),
            AssistantMessageAction(content=None, tool_calls=tool_calls),
        ]

        action = planner.step(history, iterations=1)

        assert isinstance(action, ExecuteToolsAction)
        assert len(action.tool_calls) == 2

    def test_step_complex_conversation_flow(
        self, planner, mock_conversation_wrapper, sample_tool_map
    ):
        """Test step with a complex multi-turn conversation."""
        # Turn 1: User asks
        history = [UserMessageAction(content="What is 5 + 3?")]
        action = planner.step(history, iterations=1)
        assert isinstance(action, MakeLLMRequestAction)

        # Turn 2: Assistant requests tool
        tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "add", "arguments": '{"x": 5, "y": 3}'},
            }
        ]
        history.append(AssistantMessageAction(content=None, tool_calls=tool_calls))
        action = planner.step(history, iterations=2)
        assert isinstance(action, ExecuteToolsAction)

        # Turn 3: Tool responds
        history.append(
            ToolMessageAction(
                tool_call_id="call_1",
                name="add",
                content='{"success": true, "data": 8}',
            )
        )
        action = planner.step(history, iterations=3)
        assert isinstance(action, MakeLLMRequestAction)

        # Turn 4: Assistant provides final answer
        history.append(AssistantMessageAction(content="The answer is 8"))
        action = planner.step(history, iterations=4)
        assert isinstance(action, CompleteAction)
        assert action.content == "The answer is 8"
