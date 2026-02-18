"""
AgentBuilder -- A flexible Python framework for building agentic AI systems.

AgentBuilder provides modular components for creating AI agents that can plan,
use tools, execute code, and delegate to sub-agents. The framework is built
around an OpenAI-compatible conversation loop with a pluggable planner.

Submodules:
    Action: Executable action data types for the agentic loop.
    Client: Conversation wrappers for LLM API interactions.
    Loop: The core agentic execution loop.
    Planner: Decision-making planner that drives the loop.
    Tools: Tool definitions, including sub-agent and code-execution tools.
    Sandbox: Isolated code-execution environments (Docker).
    Server: FastAPI server for exposing agents over HTTP.

Quick-start example::

    from agentbuilder.Tools.base import tool_from_function
    from agentbuilder.utils import create_agent
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
        system_prompt="You are a calculator assistant.",
    )
    response = agent.run("What is 2 + 3?")
    print(response)

See Also:
    :mod:`agentbuilder.utils` for high-level factory functions.
"""
