"""
Base conversation wrapper for managing conversation history.

Provides the foundation for all conversation wrappers with methods to add,
retrieve, persist, and reset conversation messages.
"""

import json
from typing import Any, Dict, List, Optional

from agentbuilder.Action.base import (Action, AssistantMessageAction,
                                      SystemMessageAction, ToolMessageAction,
                                      UserMessageAction)


class BaseConversationWrapper:
    """Base conversation wrapper with conversation history management.

    Stores conversation turns as :class:`~agentbuilder.Action.base.Action`
    objects and provides serialization to/from OpenAI message dicts.

    Attributes:
        client: The underlying LLM client (set by subclasses).
        model: The model identifier string (set by subclasses).
        conversation_history: Ordered list of message actions.

    Example::

        from agentbuilder.Client.openai_client import ConversationWrapper

        conv = ConversationWrapper(model="gpt-4o-mini")
        conv.add_user_message("Hello!")
        messages = conv.to_messages()
        # [{"role": "user", "content": "Hello!"}]
    """

    def __init__(self):
        """Initialize the base conversation wrapper."""
        self.client = None
        self.model = None
        self.conversation_history: List[Action] = []

    def to_messages(self) -> List[Dict]:
        """Convert action history to OpenAI message format.

        Returns:
            List of message dicts suitable for the ``messages`` parameter
            of an OpenAI API call.
        """
        messages = []
        for action in self.conversation_history:
            message = action.to_message()
            if message is not None:
                messages.append(message)
        return messages

    def add_system_message(self, content: str):
        """Add a system message to the conversation history.

        Args:
            content: The system prompt text.
        """
        self.conversation_history.append(SystemMessageAction(content=content))

    def add_user_message(self, content: str):
        """Add a user message to the conversation history.

        Args:
            content: The user's message text.
        """
        self.conversation_history.append(UserMessageAction(content=content))

    def add_assistant_message(
        self, content: Optional[str], tool_calls: Optional[List] = None
    ):
        """Add an assistant message to the conversation history.

        Args:
            content: The assistant's text response.
            tool_calls: Optional list of tool-call dicts.
        """
        self.conversation_history.append(
            AssistantMessageAction(content=content, tool_calls=tool_calls)
        )

    def add_tool_message(self, tool_call_id: str, name: str, content: str):
        """Add a tool response to the conversation history.

        Args:
            tool_call_id: Unique ID of the tool call being responded to.
            name: Name of the tool that was executed.
            content: JSON-serialized tool result.
        """
        self.conversation_history.append(
            ToolMessageAction(tool_call_id=tool_call_id, name=name, content=content)
        )

    def reset_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def reset(self):
        """Reset all conversation state (alias for :meth:`reset_conversation`)."""
        self.reset_conversation()

    def get_history(self) -> List[Action]:
        """Get the full conversation history.

        Returns:
            List of :class:`~agentbuilder.Action.base.Action` objects.
        """
        return self.conversation_history

    def get_last_message(self) -> Optional[Action]:
        """Get the last message in the conversation.

        Returns:
            The most recent :class:`~agentbuilder.Action.base.Action`, or
            ``None`` if the history is empty.
        """
        return self.conversation_history[-1] if self.conversation_history else None

    def save_conversation(self, filepath: str):
        """Save conversation history to a JSON file.

        Args:
            filepath: Path to the output JSON file.

        Example::

            conv.save_conversation("chat_history.json")
        """
        with open(filepath, "w") as f:
            # Convert actions to messages for serialization
            json.dump(self.to_messages(), f, indent=2)

    def load_conversation(self, filepath: str):
        """Load conversation history from a JSON file.

        Replaces the current history with the contents of the file.

        Args:
            filepath: Path to a JSON file previously written by
                :meth:`save_conversation`.

        Example::

            conv.load_conversation("chat_history.json")
        """
        role_mapping = {
            "system": SystemMessageAction,
            "user": UserMessageAction,
            "assistant": AssistantMessageAction,
            "tool": ToolMessageAction,
        }

        with open(filepath, "r") as f:
            messages = json.load(f)
            self.conversation_history = []
            for msg in messages:
                action_class = role_mapping.get(msg["role"])
                if action_class:
                    self.conversation_history.append(action_class.from_message(msg))
