"""
Base conversation wrapper for managing conversation history.
"""
import json
from typing import List, Dict, Optional, Any

from agentbuilder.Action.base import (
    Action,
    SystemMessageAction,
    UserMessageAction,
    AssistantMessageAction,
    ToolMessageAction
)


class BaseConversationWrapper:
    """Base conversation wrapper with conversation history management"""
    
    def __init__(self):
        """Initialize the base conversation wrapper."""
        self.client = None
        self.model = None
        self.conversation_history: List[Action] = []
    
    def to_messages(self) -> List[Dict]:
        """Convert action history to message format"""
        messages = []
        for action in self.conversation_history:
            message = action.to_message()
            if message is not None:
                messages.append(message)
        return messages
    
    def add_system_message(self, content: str):
        """Add a system message to the conversation history"""
        self.conversation_history.append(SystemMessageAction(content=content))
    
    def add_user_message(self, content: str):
        """Add a user message to the conversation history"""
        self.conversation_history.append(UserMessageAction(content=content))
    
    def add_assistant_message(self, content: Optional[str], tool_calls: Optional[List] = None):
        """Add an assistant message to the conversation history"""
        self.conversation_history.append(
            AssistantMessageAction(content=content, tool_calls=tool_calls)
        )
    
    def add_tool_message(self, tool_call_id: str, name: str, content: str):
        """Add a tool response to the conversation history"""
        self.conversation_history.append(
            ToolMessageAction(tool_call_id=tool_call_id, name=name, content=content)
        )
    
    def reset_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def reset(self):
        """Reset all conversation state (alias for reset_conversation)"""
        self.reset_conversation()
    
    def get_history(self) -> List[Action]:
        """Get the full conversation history"""
        return self.conversation_history
    
    def get_last_message(self) -> Optional[Action]:
        """Get the last message in the conversation"""
        return self.conversation_history[-1] if self.conversation_history else None
    
    def save_conversation(self, filepath: str):
        """Save conversation history to a JSON file"""
        with open(filepath, 'w') as f:
            # Convert actions to messages for serialization
            json.dump(self.to_messages(), f, indent=2)
    
    def load_conversation(self, filepath: str):
        """Load conversation history from a JSON file"""
        role_mapping = {
            "system": SystemMessageAction,
            "user": UserMessageAction,
            "assistant": AssistantMessageAction,
            "tool": ToolMessageAction
        }
        
        with open(filepath, 'r') as f:
            messages = json.load(f)
            self.conversation_history = []
            for msg in messages:
                action_class = role_mapping.get(msg["role"])
                if action_class:
                    self.conversation_history.append(action_class.from_message(msg))
