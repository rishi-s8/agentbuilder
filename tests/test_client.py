"""Tests for Client classes."""
import json
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from agentbuilder.Client.base import BaseConversationWrapper
from agentbuilder.Client.openai_client import ConversationWrapper
from agentbuilder.Action.base import (
    SystemMessageAction,
    UserMessageAction,
    AssistantMessageAction,
    ToolMessageAction
)


class TestBaseConversationWrapper:
    """Tests for BaseConversationWrapper."""
    
    def test_init(self):
        """Test initialization."""
        wrapper = BaseConversationWrapper()
        assert wrapper.client is None
        assert wrapper.model is None
        assert wrapper.conversation_history == []
    
    def test_add_system_message(self):
        """Test adding a system message."""
        wrapper = BaseConversationWrapper()
        wrapper.add_system_message("You are helpful")
        
        assert len(wrapper.conversation_history) == 1
        assert isinstance(wrapper.conversation_history[0], SystemMessageAction)
        assert wrapper.conversation_history[0].content == "You are helpful"
    
    def test_add_user_message(self):
        """Test adding a user message."""
        wrapper = BaseConversationWrapper()
        wrapper.add_user_message("Hello!")
        
        assert len(wrapper.conversation_history) == 1
        assert isinstance(wrapper.conversation_history[0], UserMessageAction)
        assert wrapper.conversation_history[0].content == "Hello!"
    
    def test_add_assistant_message_without_tool_calls(self):
        """Test adding an assistant message without tool calls."""
        wrapper = BaseConversationWrapper()
        wrapper.add_assistant_message("Hi there!")
        
        assert len(wrapper.conversation_history) == 1
        assert isinstance(wrapper.conversation_history[0], AssistantMessageAction)
        assert wrapper.conversation_history[0].content == "Hi there!"
        assert wrapper.conversation_history[0].tool_calls is None
    
    def test_add_assistant_message_with_tool_calls(self):
        """Test adding an assistant message with tool calls."""
        wrapper = BaseConversationWrapper()
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        wrapper.add_assistant_message(None, tool_calls=tool_calls)
        
        assert len(wrapper.conversation_history) == 1
        assert isinstance(wrapper.conversation_history[0], AssistantMessageAction)
        assert wrapper.conversation_history[0].content is None
        assert wrapper.conversation_history[0].tool_calls == tool_calls
    
    def test_add_tool_message(self):
        """Test adding a tool message."""
        wrapper = BaseConversationWrapper()
        wrapper.add_tool_message("call_123", "add", '{"result": 5}')
        
        assert len(wrapper.conversation_history) == 1
        assert isinstance(wrapper.conversation_history[0], ToolMessageAction)
        assert wrapper.conversation_history[0].tool_call_id == "call_123"
        assert wrapper.conversation_history[0].name == "add"
        assert wrapper.conversation_history[0].content == '{"result": 5}'
    
    def test_to_messages(self):
        """Test converting conversation history to message format."""
        wrapper = BaseConversationWrapper()
        wrapper.add_system_message("System prompt")
        wrapper.add_user_message("Hello")
        wrapper.add_assistant_message("Hi there!")
        
        messages = wrapper.to_messages()
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
    
    def test_reset_conversation(self):
        """Test resetting conversation history."""
        wrapper = BaseConversationWrapper()
        wrapper.add_user_message("Test")
        wrapper.add_assistant_message("Response")
        
        assert len(wrapper.conversation_history) == 2
        
        wrapper.reset_conversation()
        
        assert len(wrapper.conversation_history) == 0
    
    def test_reset_alias(self):
        """Test that reset() is an alias for reset_conversation()."""
        wrapper = BaseConversationWrapper()
        wrapper.add_user_message("Test")
        
        wrapper.reset()
        
        assert len(wrapper.conversation_history) == 0
    
    def test_get_history(self):
        """Test getting conversation history."""
        wrapper = BaseConversationWrapper()
        wrapper.add_user_message("Test")
        
        history = wrapper.get_history()
        
        assert len(history) == 1
        assert isinstance(history[0], UserMessageAction)
    
    def test_get_last_message_empty(self):
        """Test getting last message from empty conversation."""
        wrapper = BaseConversationWrapper()
        
        last = wrapper.get_last_message()
        
        assert last is None
    
    def test_get_last_message(self):
        """Test getting last message."""
        wrapper = BaseConversationWrapper()
        wrapper.add_user_message("First")
        wrapper.add_assistant_message("Second")
        
        last = wrapper.get_last_message()
        
        assert isinstance(last, AssistantMessageAction)
        assert last.content == "Second"
    
    def test_save_conversation(self):
        """Test saving conversation to file."""
        wrapper = BaseConversationWrapper()
        wrapper.add_user_message("Test message")
        wrapper.add_assistant_message("Test response")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            wrapper.save_conversation(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 2
            assert data[0]["role"] == "user"
            assert data[1]["role"] == "assistant"
        finally:
            os.unlink(filepath)
    
    def test_load_conversation(self):
        """Test loading conversation from file."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Assistant"},
            {"role": "tool", "tool_call_id": "call_1", "name": "test", "content": "{}"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(messages, f)
            filepath = f.name
        
        try:
            wrapper = BaseConversationWrapper()
            wrapper.load_conversation(filepath)
            
            assert len(wrapper.conversation_history) == 4
            assert isinstance(wrapper.conversation_history[0], SystemMessageAction)
            assert isinstance(wrapper.conversation_history[1], UserMessageAction)
            assert isinstance(wrapper.conversation_history[2], AssistantMessageAction)
            assert isinstance(wrapper.conversation_history[3], ToolMessageAction)
        finally:
            os.unlink(filepath)


class TestConversationWrapper:
    """Tests for ConversationWrapper."""
    
    @patch('agentbuilder.Client.openai_client.OpenAI')
    def test_init_with_api_key(self, mock_openai):
        """Test initialization with explicit API key."""
        wrapper = ConversationWrapper(
            api_key="test_key",
            model="gpt-4",
            base_url="https://api.openai.com/v1"
        )
        
        mock_openai.assert_called_once_with(api_key="test_key", base_url="https://api.openai.com/v1")
        assert wrapper.model == "gpt-4"
        assert wrapper.verbose is False
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'env_key', 'MODEL': 'gpt-3.5-turbo', 'BASE_URL': 'https://test.com'})
    @patch('agentbuilder.Client.openai_client.OpenAI')
    @patch('dotenv.load_dotenv')
    def test_init_from_env(self, mock_load_dotenv, mock_openai):
        """Test initialization from environment variables."""
        wrapper = ConversationWrapper(api_key=None, model=None, base_url=None)
        
        mock_load_dotenv.assert_called_once()
        mock_openai.assert_called_once_with(api_key='env_key', base_url='https://test.com')
        assert wrapper.model == 'gpt-3.5-turbo'
    
    @patch('agentbuilder.Client.openai_client.OpenAI')
    def test_init_with_system_prompt(self, mock_openai):
        """Test initialization with system prompt."""
        wrapper = ConversationWrapper(
            api_key="test_key",
            model="gpt-4",
            system_prompt="You are a helpful assistant"
        )
        
        assert len(wrapper.conversation_history) == 1
        assert isinstance(wrapper.conversation_history[0], SystemMessageAction)
        assert wrapper.conversation_history[0].content == "You are a helpful assistant"
    
    @patch('agentbuilder.Client.openai_client.OpenAI')
    def test_send_message(self, mock_openai):
        """Test sending a message and getting response."""
        # Mock the response
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_response.choices = [Mock(message=mock_message)]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        wrapper = ConversationWrapper(api_key="test_key", model="gpt-4")
        response = wrapper.send_message("Hello")
        
        # Verify message was added and API was called
        assert len(wrapper.conversation_history) == 2  # user + assistant
        assert isinstance(wrapper.conversation_history[0], UserMessageAction)
        assert isinstance(wrapper.conversation_history[1], AssistantMessageAction)
        assert response == "Test response"
        
        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4"
    
    @patch('agentbuilder.Client.openai_client.OpenAI')
    def test_send_message_with_kwargs(self, mock_openai):
        """Test sending a message with additional kwargs."""
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Response"
        mock_response.choices = [Mock(message=mock_message)]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        wrapper = ConversationWrapper(api_key="test_key", model="gpt-4")
        wrapper.send_message("Hello", temperature=0.7, max_tokens=100)
        
        # Verify kwargs were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100
    
    @patch('agentbuilder.Client.openai_client.OpenAI')
    def test_verbose_mode(self, mock_openai):
        """Test verbose mode."""
        wrapper = ConversationWrapper(
            api_key="test_key",
            model="gpt-4",
            verbose=True
        )
        
        assert wrapper.verbose is True
