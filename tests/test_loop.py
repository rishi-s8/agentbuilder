"""Tests for AgenticLoop."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from agentbuilder.Loop.base import AgenticLoop
from agentbuilder.Action.base import (
    UserMessageAction,
    AssistantMessageAction,
    CompleteAction,
    EmptyAction,
    MakeLLMRequestAction,
    ExecuteToolsAction
)


class TestAgenticLoop:
    """Tests for AgenticLoop."""
    
    @pytest.fixture
    def mock_conversation_wrapper(self):
        """Mock conversation wrapper."""
        wrapper = Mock()
        wrapper.conversation_history = []
        wrapper.add_user_message = Mock()
        wrapper.reset = Mock()
        return wrapper
    
    @pytest.fixture
    def mock_planner(self):
        """Mock planner."""
        planner = Mock()
        planner.reset = Mock()
        return planner
    
    @pytest.fixture
    def sample_tool_map(self, sample_tool):
        """Sample tool map."""
        return {sample_tool.name: sample_tool}
    
    @pytest.fixture
    def agentic_loop(self, mock_conversation_wrapper, mock_planner, sample_tool_map):
        """Create an agentic loop instance."""
        return AgenticLoop(
            mock_conversation_wrapper,
            mock_planner,
            sample_tool_map,
            verbose=False,
            max_iterations=10
        )
    
    def test_init(self, mock_conversation_wrapper, mock_planner, sample_tool_map):
        """Test agentic loop initialization."""
        loop = AgenticLoop(
            mock_conversation_wrapper,
            mock_planner,
            sample_tool_map,
            verbose=True,
            max_iterations=20
        )
        
        assert loop.conversation == mock_conversation_wrapper
        assert loop.planner == mock_planner
        assert loop.tool_map == sample_tool_map
        assert loop.verbose is True
        assert loop.max_iterations == 20
    
    def test_reset(self, agentic_loop, mock_conversation_wrapper, mock_planner):
        """Test reset method."""
        agentic_loop.reset()
        
        mock_conversation_wrapper.reset.assert_called_once()
        mock_planner.reset.assert_called_once()
    
    def test_run_simple_completion(self, agentic_loop, mock_conversation_wrapper, mock_planner):
        """Test run with immediate completion."""
        # Mock planner to return CompleteAction
        complete_action = CompleteAction(content="Done!", iterations=1, verbose=False)
        complete_action.run = Mock(return_value="Done!")
        mock_planner.step.return_value = complete_action
        
        result = agentic_loop.run("Hello")
        
        # Verify user message was added
        mock_conversation_wrapper.add_user_message.assert_called_once_with("Hello")
        
        # Verify planner was called
        mock_planner.step.assert_called_once()
        
        # Verify result
        assert result == "Done!"
    
    def test_run_empty_action(self, agentic_loop, mock_conversation_wrapper, mock_planner):
        """Test run with empty action."""
        empty_action = EmptyAction(iterations=1, verbose=False)
        empty_action.run = Mock(return_value="")
        mock_planner.step.return_value = empty_action
        
        result = agentic_loop.run("Test")
        
        assert result == ""
    
    def test_run_max_iterations(self, agentic_loop, mock_conversation_wrapper, mock_planner, capsys):
        """Test run reaching max iterations."""
        # Mock planner to return non-completing actions
        mock_action = Mock()
        mock_action.run = Mock(return_value=None)
        mock_planner.step.return_value = mock_action
        
        result = agentic_loop.run("Test")
        
        # Verify max iterations were reached
        assert mock_planner.step.call_count == 10
        assert result == "Max iterations reached"
    
    def test_run_max_iterations_verbose(self, mock_conversation_wrapper, mock_planner, sample_tool_map, capsys):
        """Test verbose output when max iterations reached."""
        loop = AgenticLoop(
            mock_conversation_wrapper,
            mock_planner,
            sample_tool_map,
            verbose=True,
            max_iterations=5
        )
        
        mock_action = Mock()
        mock_action.run = Mock(return_value=None)
        mock_planner.step.return_value = mock_action
        
        loop.run("Test")
        
        captured = capsys.readouterr()
        assert "Max iterations" in captured.out
        assert "5" in captured.out
    
    def test_run_multi_step_conversation(self, mock_conversation_wrapper, mock_planner, sample_tool_map):
        """Test run with multiple steps before completion."""
        loop = AgenticLoop(
            mock_conversation_wrapper,
            mock_planner,
            sample_tool_map,
            verbose=False,
            max_iterations=10
        )
        
        # Create sequence of actions
        action1 = Mock()
        action1.run = Mock(return_value=None)
        
        action2 = Mock()
        action2.run = Mock(return_value=None)
        
        complete_action = CompleteAction(content="Final answer", iterations=3, verbose=False)
        complete_action.run = Mock(return_value="Final answer")
        
        mock_planner.step.side_effect = [action1, action2, complete_action]
        
        result = loop.run("Test")
        
        # Verify all actions were executed
        assert mock_planner.step.call_count == 3
        action1.run.assert_called_once()
        action2.run.assert_called_once()
        
        # Verify final result
        assert result == "Final answer"
    
    def test_run_with_tool_execution_flow(self, mock_conversation_wrapper, sample_tool_map):
        """Test run with realistic tool execution flow."""
        # Create a real planner
        from agentbuilder.Planner.base import AgenticPlanner
        planner = AgenticPlanner(mock_conversation_wrapper, sample_tool_map, verbose=False)
        
        loop = AgenticLoop(
            mock_conversation_wrapper,
            planner,
            sample_tool_map,
            verbose=False,
            max_iterations=10
        )
        
        # Mock conversation history to simulate a tool call flow
        mock_conversation_wrapper.conversation_history = []
        
        # Step 1: User message -> should trigger LLM request
        # Step 2: Assistant with tool calls -> should trigger tool execution
        # Step 3: Tool response -> should trigger LLM request
        # Step 4: Assistant with content -> should complete
        
        # We'll manually step through to test the flow
        mock_conversation_wrapper.conversation_history = [
            UserMessageAction(content="What is 5 + 3?")
        ]
        action1 = planner.step(mock_conversation_wrapper.conversation_history, 1)
        assert isinstance(action1, MakeLLMRequestAction)
        
        # Add assistant message with tool call
        tool_calls = [{"id": "call_1", "function": {"name": "add", "arguments": '{"x": 5, "y": 3}'}}]
        mock_conversation_wrapper.conversation_history.append(
            AssistantMessageAction(content=None, tool_calls=tool_calls)
        )
        action2 = planner.step(mock_conversation_wrapper.conversation_history, 2)
        assert isinstance(action2, ExecuteToolsAction)
    
    def test_run_handles_none_return(self, agentic_loop, mock_conversation_wrapper, mock_planner):
        """Test that run handles None returns from actions properly."""
        complete_action = CompleteAction(content="Result", iterations=1, verbose=False)
        complete_action.run = Mock(return_value=None)
        mock_planner.step.return_value = complete_action
        
        result = agentic_loop.run("Test")
        
        # Should return empty string when result is None
        assert result == ""
    
    def test_run_iteration_count(self, mock_conversation_wrapper, mock_planner, sample_tool_map):
        """Test that iteration count is properly passed to planner."""
        loop = AgenticLoop(
            mock_conversation_wrapper,
            mock_planner,
            sample_tool_map,
            verbose=False,
            max_iterations=5
        )
        
        action1 = Mock()
        action1.run = Mock(return_value=None)
        
        action2 = Mock()
        action2.run = Mock(return_value=None)
        
        complete_action = CompleteAction(content="Done", iterations=3, verbose=False)
        complete_action.run = Mock(return_value="Done")
        
        mock_planner.step.side_effect = [action1, action2, complete_action]
        
        loop.run("Test")
        
        # Check that iterations were passed correctly
        calls = mock_planner.step.call_args_list
        assert calls[0][0][1] == 1  # First iteration
        assert calls[1][0][1] == 2  # Second iteration
        assert calls[2][0][1] == 3  # Third iteration
    
    def test_run_preserves_conversation_history(self, mock_conversation_wrapper, mock_planner, sample_tool_map):
        """Test that conversation history is properly passed to planner."""
        loop = AgenticLoop(
            mock_conversation_wrapper,
            mock_planner,
            sample_tool_map,
            verbose=False,
            max_iterations=10
        )
        
        complete_action = CompleteAction(content="Done", iterations=1, verbose=False)
        complete_action.run = Mock(return_value="Done")
        mock_planner.step.return_value = complete_action
        
        loop.run("Test message")
        
        # Verify planner received conversation history
        call_args = mock_planner.step.call_args[0]
        assert call_args[0] == mock_conversation_wrapper.conversation_history
