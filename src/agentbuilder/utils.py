"""
Utility functions for agentbuilder package.
"""
from typing import List, Optional

from agentbuilder.Client.openai_client import ConversationWrapper
from agentbuilder.Planner.base import AgenticPlanner
from agentbuilder.Loop.base import AgenticLoop


def create_agent(
    model_name: str,
    tools: List,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    verbose: bool = True,
    max_iterations: int = 80,
    system_prompt: Optional[str] = None
) -> AgenticLoop:
    """
    Create an agentic loop with specified model and tools.
    
    Args:
        model_name: Name of the model to use
        tools: List of Tool objects available to the agent
        api_key: OpenAI API key (if None, uses environment variable)
        base_url: Custom API endpoint URL
        verbose: Whether to print execution details
        max_iterations: Maximum number of iterations for the agentic loop
        system_prompt: System prompt to set conversation context
    
    Returns:
        AgenticLoop: Configured agentic loop ready to run
    """
    # Create tool map
    tool_map = {tool.name: tool for tool in tools}
    
    # Create conversation wrapper with system prompt
    conversation = ConversationWrapper(
        api_key=api_key,
        model=model_name,
        base_url=base_url,
        verbose=verbose,
        system_prompt=system_prompt
    )
    
    # Create planner
    planner = AgenticPlanner(conversation, tool_map, verbose=verbose)
    
    # Create agentic loop
    agentic_loop = AgenticLoop(
        conversation,
        planner,
        tool_map,
        verbose=verbose,
        max_iterations=max_iterations
    )
    
    return agentic_loop
