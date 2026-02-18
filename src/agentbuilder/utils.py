"""
Utility functions for agentbuilder package.

High-level factory functions that assemble the framework's components into
ready-to-use agents. Most users should start here rather than constructing
:class:`~agentbuilder.Loop.base.AgenticLoop` manually.
"""

from typing import Callable, List, Optional

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

    This is the primary entry point for building a tool-calling agent.
    It wires together a :class:`~agentbuilder.Client.openai_client.ConversationWrapper`,
    an :class:`~agentbuilder.Planner.base.AgenticPlanner`, and an
    :class:`~agentbuilder.Loop.base.AgenticLoop`.

    Args:
        model_name: Name of the model to use (e.g. ``"gpt-4o-mini"``).
        tools: List of :class:`~agentbuilder.Tools.base.Tool` objects
            available to the agent.
        api_key: OpenAI API key. If ``None``, loaded from the
            ``OPENAI_API_KEY`` environment variable or ``.env`` file.
        base_url: Custom API endpoint URL for OpenAI-compatible providers.
        verbose: Whether to print execution details to stdout.
        max_iterations: Maximum plan-execute cycles before the loop stops.
        system_prompt: System prompt to set conversation context.

    Returns:
        A configured :class:`~agentbuilder.Loop.base.AgenticLoop` ready to
        run via :meth:`~agentbuilder.Loop.base.AgenticLoop.run`.

    Example::

        from agentbuilder.utils import create_agent
        from agentbuilder.Tools.base import tool_from_function
        from pydantic import BaseModel, Field

        class AddParams(BaseModel):
            a: int = Field(description="First number")
            b: int = Field(description="Second number")

        def add(params: AddParams) -> int:
            \"\"\"Add two numbers.\"\"\"
            return params.a + params.b

        agent = create_agent(
            model_name="gpt-4o-mini",
            tools=[tool_from_function(add)],
            system_prompt="You are a calculator.",
        )
        result = agent.run("What is 2 + 3?")
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

    The returned :class:`~agentbuilder.Tools.agent_tool.AgentTool` can be
    added to a parent agent's tool list, enabling multi-agent delegation.

    Args:
        name: Name for the sub-agent tool (exposed to the LLM).
        description: Description of what the sub-agent does.
        model_name: Name of the model to use for the sub-agent.
        tools: List of :class:`~agentbuilder.Tools.base.Tool` objects
            available to the sub-agent.
        system_prompt: System prompt for the sub-agent.
        max_iterations: Maximum iterations for the sub-agent loop.
        **kwargs: Additional keyword arguments passed to
            :func:`create_agent`.

    Returns:
        An :class:`~agentbuilder.Tools.agent_tool.AgentTool` wrapping the
        sub-agent.

    Example::

        math_tool = create_agent_tool(
            name="math_expert",
            description="Solves math problems",
            model_name="gpt-4o-mini",
            tools=[add_tool, multiply_tool],
            system_prompt="You are a math expert.",
        )
        # Now use math_tool in a parent agent's tool list
        parent = create_agent(
            model_name="gpt-4o-mini",
            tools=[math_tool],
        )
    """
    agent = create_agent(
        model_name=model_name,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        **kwargs,
    )
    return AgentTool(agent=agent, name=name, description=description)


def create_agent_factory(
    model_name: str,
    tools: List,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    verbose: bool = True,
    max_iterations: int = 80,
    system_prompt: Optional[str] = None,
) -> Callable[[], AgenticLoop]:
    """
    Create a factory callable that produces fresh AgenticLoop instances.

    Each invocation of the returned callable creates a new
    :class:`~agentbuilder.Loop.base.AgenticLoop` with its own
    :class:`~agentbuilder.Client.openai_client.ConversationWrapper`.
    The ``tool_map`` objects are read-only and safe to share across
    instances.

    This is primarily used with :func:`~agentbuilder.Server.base.serve_agent`
    so that each HTTP session gets its own conversation state.

    Args:
        model_name: Name of the model to use.
        tools: List of :class:`~agentbuilder.Tools.base.Tool` objects.
        api_key: OpenAI API key (if ``None``, uses environment variable).
        base_url: Custom API endpoint URL.
        verbose: Whether to print execution details.
        max_iterations: Maximum number of iterations for the agentic loop.
        system_prompt: System prompt to set conversation context.

    Returns:
        A zero-argument callable that returns a fresh
        :class:`~agentbuilder.Loop.base.AgenticLoop` each time.

    Example::

        from agentbuilder.utils import create_agent_factory
        from agentbuilder.Server import serve_agent

        factory = create_agent_factory(
            model_name="gpt-4o-mini",
            tools=[add_tool],
            system_prompt="You are a calculator.",
        )
        serve_agent(factory, name="calculator", description="A calculator agent")
    """
    # Build tool_map once -- tools are read-only, safe to share
    tool_map = {tool.name: tool for tool in tools}

    def factory() -> AgenticLoop:
        conversation = ConversationWrapper(
            api_key=api_key,
            model=model_name,
            base_url=base_url,
            verbose=verbose,
            system_prompt=system_prompt,
        )
        planner = AgenticPlanner(conversation, tool_map, verbose=verbose)
        return AgenticLoop(
            conversation,
            planner,
            tool_map,
            verbose=verbose,
            max_iterations=max_iterations,
        )

    return factory


def create_remote_agent_tool(base_url: str) -> RemoteAgentTool:
    """
    Create a remote sub-agent tool that connects to an already-running agent server.

    Auto-discovers the agent's name and description from
    ``GET {base_url}/info``, then creates a session via
    ``POST {base_url}/sessions``. Call :meth:`~agentbuilder.Tools.remote_agent_tool.RemoteAgentTool.close`
    on the returned tool when done to delete the remote session.

    Args:
        base_url: Base URL of the remote agent server
            (e.g. ``"http://localhost:8100"``).

    Returns:
        A :class:`~agentbuilder.Tools.remote_agent_tool.RemoteAgentTool`
        connected to the remote agent with an active session.

    Raises:
        requests.HTTPError: If the ``/info`` or ``/sessions`` endpoints
            return a non-2xx status code.
        requests.ConnectionError: If the remote server is not reachable.

    Example::

        from agentbuilder.utils import create_remote_agent_tool

        remote = create_remote_agent_tool("http://localhost:8100")
        try:
            parent = create_agent(
                model_name="gpt-4o-mini",
                tools=[remote],
            )
            result = parent.run("Delegate this task")
        finally:
            remote.close()
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

    Automatically creates a :class:`~agentbuilder.Tools.code_execution.CodeExecutionTool`
    and sandbox management tools (``read_file``, ``write_file``, ``install_package``)
    from the provided :class:`~agentbuilder.Sandbox.base.Sandbox`.

    Args:
        model_name: Name of the model to use.
        sandbox: A :class:`~agentbuilder.Sandbox.base.Sandbox` instance
            for code execution (e.g. :class:`~agentbuilder.Sandbox.docker_sandbox.DockerSandbox`).
        additional_tools: Extra :class:`~agentbuilder.Tools.base.Tool` objects
            to include alongside the code execution tools.
        system_prompt: System prompt (defaults to a code-focused prompt if
            ``None``).
        max_iterations: Maximum iterations for the agentic loop.
        **kwargs: Additional keyword arguments passed to
            :func:`create_agent`.

    Returns:
        An :class:`~agentbuilder.Loop.base.AgenticLoop` configured with
        code execution capabilities.

    Note:
        Requires Docker to be installed and running on the host machine
        when using :class:`~agentbuilder.Sandbox.docker_sandbox.DockerSandbox`.
        Install the ``code`` extra: ``pip install agentbuilder[code]``.

    Example::

        from agentbuilder.Sandbox.docker_sandbox import DockerSandbox
        from agentbuilder.utils import create_code_agent

        with DockerSandbox() as sandbox:
            agent = create_code_agent(
                model_name="gpt-4o-mini",
                sandbox=sandbox,
            )
            result = agent.run("Write a script that prints the first 10 primes")
            print(result)
    """
    from agentbuilder.Tools.code_execution import (CodeExecutionTool,
                                                   create_sandbox_tools)

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
