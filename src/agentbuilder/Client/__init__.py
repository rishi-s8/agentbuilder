"""
Client module for LLM conversation management.

Provides conversation wrappers that manage chat history and interact with
OpenAI-compatible APIs.

Classes:
  :class:`BaseConversationWrapper` -- abstract base with history management.
  :class:`ConversationWrapper` -- OpenAI-specific implementation with
  automatic ``.env`` loading and ``send_message()`` for simple calls.
"""

from agentbuilder.Client.base import BaseConversationWrapper
from agentbuilder.Client.openai_client import ConversationWrapper

__all__ = ["BaseConversationWrapper", "ConversationWrapper"]
