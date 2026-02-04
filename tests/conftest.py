"""Shared fixtures for agentbuilder tests."""
import pytest
from unittest.mock import Mock, MagicMock
from agentbuilder.Client.base import BaseConversationWrapper
from agentbuilder.Tools.base import Tool, Response


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = Mock()
    response = Mock()
    message = Mock()
    message.content = "Test response"
    message.tool_calls = None
    response.choices = [Mock(message=message)]
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def sample_tool():
    """Sample tool for testing."""
    def sample_function(x: int, y: int) -> int:
        return x + y
    
    return Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"}
            },
            "required": ["x", "y"]
        },
        function=sample_function
    )


@pytest.fixture
def sample_tool_map(sample_tool):
    """Sample tool map for testing."""
    return {sample_tool.name: sample_tool}


@pytest.fixture
def base_conversation():
    """Base conversation wrapper for testing."""
    return BaseConversationWrapper()


@pytest.fixture
def mock_conversation_wrapper():
    """Mock conversation wrapper with history."""
    wrapper = Mock(spec=BaseConversationWrapper)
    wrapper.conversation_history = []
    wrapper.model = "gpt-4"
    wrapper.client = Mock()
    wrapper.to_messages = Mock(return_value=[])
    wrapper.add_tool_message = Mock()
    wrapper.add_assistant_message = Mock()
    return wrapper
