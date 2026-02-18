"""
Conversation wrapper for OpenAI API interactions.

Extends :class:`~agentbuilder.Client.base.BaseConversationWrapper` with an
OpenAI client and automatic ``.env`` loading for API credentials.

Note:
    When ``api_key`` is not provided explicitly, the wrapper loads
    environment variables from a ``.env`` file using ``python-dotenv``.
    The following variables are read: ``OPENAI_API_KEY``, ``MODEL``,
    ``BASE_URL``.
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agentbuilder.Client.base import BaseConversationWrapper


class ConversationWrapper(BaseConversationWrapper):
    """OpenAI conversation wrapper for managing conversations and making LLM calls.

    This is the default conversation backend used by
    :func:`~agentbuilder.utils.create_agent`.

    Example::

        from agentbuilder.Client.openai_client import ConversationWrapper

        conv = ConversationWrapper(
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )
        reply = conv.send_message("Hello!")
        print(reply)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = "https://inference.rcp.epfl.ch/v1",
        verbose: bool = False,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the conversation wrapper.

        Args:
            api_key: OpenAI API key. If ``None``, loads from
                ``OPENAI_API_KEY`` env var or ``.env`` file.
            model: Model identifier (e.g. ``"gpt-4o-mini"``). Falls back to
                the ``MODEL`` env var.
            base_url: Custom API endpoint URL. Falls back to ``BASE_URL``
                env var if ``api_key`` is ``None``.
            verbose: Whether to print execution details.
            system_prompt: System prompt to set conversation context.

        Note:
            When *api_key* is ``None``, ``python-dotenv`` loads a ``.env``
            file from the working directory.  Environment variables
            ``OPENAI_API_KEY``, ``MODEL``, and ``BASE_URL`` are then
            read as fallbacks.
        """
        super().__init__()

        # Load environment variables from .env file if it exists
        if api_key is None:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if model is None:  # Use default model if not specified
                model = os.getenv("MODEL")
            if base_url is None:
                base_url = os.getenv("BASE_URL")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.verbose = verbose

        # Add system prompt to conversation history if provided
        if system_prompt:
            self.add_system_message(system_prompt)

    def send_message(self, message: str, **kwargs) -> str:
        """
        Send a message and get a response (simple LLM call without tool orchestration).

        For agentic tool orchestration, use
        :meth:`~agentbuilder.Loop.base.AgenticLoop.run` instead.

        Args:
            message: User message to send.
            **kwargs: Additional parameters passed to
                ``client.chat.completions.create()``.

        Returns:
            The assistant's response as a string.

        Example::

            conv = ConversationWrapper(model="gpt-4o-mini")
            answer = conv.send_message("What is 2+2?")
            print(answer)  # "4"
        """
        # Add user message
        self.add_user_message(message)

        # Get response
        response = self.client.chat.completions.create(
            model=self.model, messages=self.to_messages(), **kwargs
        )

        # Extract assistant's reply
        assistant_message = response.choices[0].message

        # Add to history
        self.add_assistant_message(assistant_message.content)

        return assistant_message.content
