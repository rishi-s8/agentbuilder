"""
Conversation wrapper for OpenAI API interactions.
"""
import os
import json
from typing import List, Dict, Optional, Any
from openai import OpenAI

from agentbuilder.Client.base import BaseConversationWrapper


class ConversationWrapper(BaseConversationWrapper):
    """OpenAI conversation wrapper for managing conversations and making LLM calls"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: Optional[str] = None, 
                 base_url: Optional[str] = "https://inference.rcp.epfl.ch/v1",
                 verbose: bool = False):
        """
        Initialize the conversation wrapper.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or .env file)
            model: Model to use
            base_url: Custom API endpoint URL
            verbose: Whether to print execution details
        """
        super().__init__()
        
        # Load environment variables from .env file if it exists
        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if model is None:  # Use default model if not specified
                model = os.getenv('MODEL')
            if base_url is None:
                base_url = os.getenv('BASE_URL')
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.verbose = verbose
    
    def send_message(self, message: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Send a message and get a response (simple LLM call without tool orchestration).
        For agentic tool orchestration, use AgenticLoop.run() instead.
        
        Args:
            message: User message to send
            system_prompt: Optional system prompt to set context
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            Assistant's response as a string
        """
        # Add system message if provided and conversation is empty
        if system_prompt and not self.conversation_history:
            self.add_system_message(system_prompt)
        
        # Add user message
        self.add_user_message(message)
        
        # Get response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.to_messages(),
            **kwargs
        )
        
        # Extract assistant's reply
        assistant_message = response.choices[0].message
        
        # Add to history
        self.add_assistant_message(assistant_message.content)
        
        return assistant_message.content
