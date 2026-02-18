"""
Utility functions for agentbuilder package.
"""

from typing import List, Optional

from agentbuilder.Client.openai_client import ConversationWrapper
from agentbuilder.Loop.base import AgenticLoop
from agentbuilder.Planner.base import AgenticPlanner
from agentbuilder.Tools.agent_tool import AgentTool
from agentbuilder.Tools.remote_agent_tool import RemoteAgentTool


def create_agent(
    model_name: str,
    tools: List,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    verbose: bool = True,
    max_iterations: int = 80,
    system_prompt: Optional[str] = None,
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
        system_prompt=system_prompt,
    )

    # Create planner
    planner = AgenticPlanner(conversation, tool_map, verbose=verbose)

    # Create agentic loop
    agentic_loop = AgenticLoop(
        conversation, planner, tool_map, verbose=verbose, max_iterations=max_iterations
    )

    return agentic_loop


def create_agent_tool(
    name: str,
    description: str,
    model_name: str,
    tools: List,
    system_prompt: Optional[str] = None,
    max_iterations: int = 80,
    **kwargs,
) -> AgentTool:
    """
    Create a local sub-agent tool that delegates tasks in-process.

    Args:
        name: Name for the sub-agent tool
        description: Description of what the sub-agent does
        model_name: Name of the model to use for the sub-agent
        tools: List of Tool objects available to the sub-agent
        system_prompt: System prompt for the sub-agent
        max_iterations: Maximum iterations for the sub-agent loop
        **kwargs: Additional keyword arguments passed to create_agent()

    Returns:
        AgentTool wrapping the sub-agent
    """
    agent = create_agent(
        model_name=model_name,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        **kwargs,
    )
    return AgentTool(agent=agent, name=name, description=description)


def create_remote_agent_tool(base_url: str) -> RemoteAgentTool:
    """
    Create a remote sub-agent tool that connects to an already-running agent server.

    Auto-discovers the agent's name and description from GET {base_url}/info.

    Args:
        base_url: Base URL of the remote agent server (e.g., "http://localhost:8100")

    Returns:
        RemoteAgentTool connected to the remote agent
    """
    return RemoteAgentTool(base_url=base_url)


def create_code_agent(
    model_name: str,
    sandbox,
    additional_tools: Optional[List] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = 80,
    **kwargs,
) -> AgenticLoop:
    """
    Create an agent equipped with code execution tools.

    Args:
        model_name: Name of the model to use
        sandbox: A Sandbox instance for code execution
        additional_tools: Extra tools to include alongside code execution tools
        system_prompt: System prompt (defaults to a code-focused prompt)
        max_iterations: Maximum iterations for the agentic loop
        **kwargs: Additional keyword arguments passed to create_agent()

    Returns:
        AgenticLoop configured with code execution capabilities
    """
    from agentbuilder.Tools.code_execution import (
        CodeExecutionTool,
        create_sandbox_tools,
    )

    code_tool = CodeExecutionTool(sandbox)
    sandbox_tools = create_sandbox_tools(sandbox)
    all_tools = [code_tool] + sandbox_tools + (additional_tools or [])

    if system_prompt is None:
        system_prompt = (
            "You are a helpful coding assistant. You can write and execute Python code "
            "to solve problems. Use the execute_code tool to run code. Variables and "
            "imports persist between calls. You can also read/write files and install "
            "packages in the sandbox."
        )

    return create_agent(
        model_name=model_name,
        tools=all_tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        **kwargs,
    )
