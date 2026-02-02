"""
Conversation wrapper for OpenAI API interactions.
"""
import os
import json
from typing import List, Dict, Optional, Any
from openai import OpenAI

from agentbuilder.Client.base import BaseConversationWrapper


class ConversationWrapper(BaseConversationWrapper):
    """OpenAI conversation wrapper with optional tool orchestration"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: Optional[str] = None, 
                 base_url: Optional[str] = "https://inference.rcp.epfl.ch/v1",
                 tools: Optional[List] = None,
                 planner: Optional[Any] = None,
                 agentic_loop: Optional[Any] = None,
                 verbose: bool = False):
        """
        Initialize the conversation wrapper.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or .env file)
            model: Model to use
            base_url: Custom API endpoint URL
            tools: List of Tool objects to use (optional, for tool orchestration)
            planner: AgenticPlanner instance (optional, for tool orchestration)
            agentic_loop: AgenticLoop instance (optional, for tool orchestration)
            verbose: Whether to print execution details (for tool orchestration)
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
        
        # Tool orchestration attributes (optional)
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools} if tools else {}
        self.verbose = verbose
        self.planner = planner
        self.agentic_loop = agentic_loop
    
    def send_message(self, message: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Send a message and get a response.
        If agentic_loop is provided, uses it for tool orchestration.
        Otherwise, makes a simple LLM call.
        
        Args:
            message: User message to send
            system_prompt: Optional system prompt to set context
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            Assistant's response as a string
        """
        # Use agentic loop if available
        if self.agentic_loop is not None:
            return self.agentic_loop.run(message, system_prompt)
        
        # Otherwise, simple LLM call
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
    
    def add_tool(self, tool):
        """Add a new tool to the orchestrator"""
        self.tools.append(tool)
        self.tool_map[tool.name] = tool
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the orchestrator"""
        self.tools = [t for t in self.tools if t.name != tool_name]
        if tool_name in self.tool_map:
            del self.tool_map[tool_name]
    
    def list_tools(self) -> List[str]:
        """Get a list of available tool names"""
        return [tool.name for tool in self.tools]
