"""Tests for utils module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from agentbuilder.utils import create_agent
from agentbuilder.Loop.base import AgenticLoop


class TestCreateAgent:
    """Tests for create_agent utility function."""
    
    @pytest.fixture
    def sample_tools(self, sample_tool):
        """Sample tools list."""
        return [sample_tool]
    
    @patch('agentbuilder.utils.ConversationWrapper')
    @patch('agentbuilder.utils.AgenticPlanner')
    @patch('agentbuilder.utils.AgenticLoop')
    def test_create_agent_basic(self, mock_loop_class, mock_planner_class, mock_conversation_class, sample_tools):
        """Test basic agent creation."""
        mock_conversation = Mock()
        mock_conversation_class.return_value = mock_conversation
        
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop
        
        result = create_agent(
            model_name="gpt-4",
            tools=sample_tools,
            api_key="test_key"
        )
        
        # Verify ConversationWrapper was created
        mock_conversation_class.assert_called_once_with(
            api_key="test_key",
            model="gpt-4",
            base_url=None,
            verbose=True,
            system_prompt=None
        )
        
        # Verify AgenticPlanner was created
        mock_planner_class.assert_called_once_with(
            mock_conversation,
            {'add': sample_tools[0]},
            verbose=True
        )
        
        # Verify AgenticLoop was created
        mock_loop_class.assert_called_once_with(
            mock_conversation,
            mock_planner,
            {'add': sample_tools[0]},
            verbose=True,
            max_iterations=80
        )
        
        # Verify the loop is returned
        assert result == mock_loop
    
    @patch('agentbuilder.utils.ConversationWrapper')
    @patch('agentbuilder.utils.AgenticPlanner')
    @patch('agentbuilder.utils.AgenticLoop')
    def test_create_agent_with_base_url(self, mock_loop_class, mock_planner_class, mock_conversation_class, sample_tools):
        """Test agent creation with custom base URL."""
        mock_conversation = Mock()
        mock_conversation_class.return_value = mock_conversation
        
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop
        
        create_agent(
            model_name="gpt-4",
            tools=sample_tools,
            api_key="test_key",
            base_url="https://custom.api.com/v1"
        )
        
        # Verify base_url was passed
        call_kwargs = mock_conversation_class.call_args[1]
        assert call_kwargs["base_url"] == "https://custom.api.com/v1"
    
    @patch('agentbuilder.utils.ConversationWrapper')
    @patch('agentbuilder.utils.AgenticPlanner')
    @patch('agentbuilder.utils.AgenticLoop')
    def test_create_agent_with_system_prompt(self, mock_loop_class, mock_planner_class, mock_conversation_class, sample_tools):
        """Test agent creation with system prompt."""
        mock_conversation = Mock()
        mock_conversation_class.return_value = mock_conversation
        
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop
        
        create_agent(
            model_name="gpt-4",
            tools=sample_tools,
            api_key="test_key",
            system_prompt="You are a helpful assistant"
        )
        
        # Verify system_prompt was passed
        call_kwargs = mock_conversation_class.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a helpful assistant"
    
    @patch('agentbuilder.utils.ConversationWrapper')
    @patch('agentbuilder.utils.AgenticPlanner')
    @patch('agentbuilder.utils.AgenticLoop')
    def test_create_agent_verbose_false(self, mock_loop_class, mock_planner_class, mock_conversation_class, sample_tools):
        """Test agent creation with verbose=False."""
        mock_conversation = Mock()
        mock_conversation_class.return_value = mock_conversation
        
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop
        
        create_agent(
            model_name="gpt-4",
            tools=sample_tools,
            api_key="test_key",
            verbose=False
        )
        
        # Verify verbose was passed to all components
        assert mock_conversation_class.call_args[1]["verbose"] is False
        assert mock_planner_class.call_args[1]["verbose"] is False
        assert mock_loop_class.call_args[1]["verbose"] is False
    
    @patch('agentbuilder.utils.ConversationWrapper')
    @patch('agentbuilder.utils.AgenticPlanner')
    @patch('agentbuilder.utils.AgenticLoop')
    def test_create_agent_custom_max_iterations(self, mock_loop_class, mock_planner_class, mock_conversation_class, sample_tools):
        """Test agent creation with custom max iterations."""
        mock_conversation = Mock()
        mock_conversation_class.return_value = mock_conversation
        
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop
        
        create_agent(
            model_name="gpt-4",
            tools=sample_tools,
            api_key="test_key",
            max_iterations=50
        )
        
        # Verify max_iterations was passed
        call_kwargs = mock_loop_class.call_args[1]
        assert call_kwargs["max_iterations"] == 50
    
    @patch('agentbuilder.utils.ConversationWrapper')
    @patch('agentbuilder.utils.AgenticPlanner')
    @patch('agentbuilder.utils.AgenticLoop')
    def test_create_agent_multiple_tools(self, mock_loop_class, mock_planner_class, mock_conversation_class, sample_tool):
        """Test agent creation with multiple tools."""
        from agentbuilder.Tools.base import Tool
        
        tool1 = sample_tool
        tool2 = Tool(
            name="multiply",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"]
            },
            function=lambda x, y: x * y
        )
        
        tools = [tool1, tool2]
        
        mock_conversation = Mock()
        mock_conversation_class.return_value = mock_conversation
        
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop
        
        create_agent(
            model_name="gpt-4",
            tools=tools,
            api_key="test_key"
        )
        
        # Verify tool map was created correctly
        expected_tool_map = {'add': tool1, 'multiply': tool2}
        
        planner_tool_map = mock_planner_class.call_args[0][1]
        assert planner_tool_map == expected_tool_map
        
        loop_tool_map = mock_loop_class.call_args[0][2]
        assert loop_tool_map == expected_tool_map
    
    @patch('agentbuilder.utils.ConversationWrapper')
    @patch('agentbuilder.utils.AgenticPlanner')
    @patch('agentbuilder.utils.AgenticLoop')
    def test_create_agent_no_api_key(self, mock_loop_class, mock_planner_class, mock_conversation_class, sample_tools):
        """Test agent creation without API key (should use environment variable)."""
        mock_conversation = Mock()
        mock_conversation_class.return_value = mock_conversation
        
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop
        
        create_agent(
            model_name="gpt-4",
            tools=sample_tools
        )
        
        # Verify api_key=None was passed (will use env var)
        call_kwargs = mock_conversation_class.call_args[1]
        assert call_kwargs["api_key"] is None
    
    @patch('agentbuilder.utils.ConversationWrapper')
    @patch('agentbuilder.utils.AgenticPlanner')
    @patch('agentbuilder.utils.AgenticLoop')
    def test_create_agent_returns_agentic_loop(self, mock_loop_class, mock_planner_class, mock_conversation_class, sample_tools):
        """Test that create_agent returns an AgenticLoop instance."""
        mock_conversation = Mock()
        mock_conversation_class.return_value = mock_conversation
        
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        
        mock_loop = Mock(spec=AgenticLoop)
        mock_loop_class.return_value = mock_loop
        
        result = create_agent(
            model_name="gpt-4",
            tools=sample_tools,
            api_key="test_key"
        )
        
        # Verify the result is the mock loop (simulating AgenticLoop)
        assert result == mock_loop
