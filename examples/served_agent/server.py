"""
Served Agent -- Server

Serves a calculator agent over HTTP on port 8100.
Requires the 'server' extra: pip install agentbuilder[server]

Run this script first, then use client.py to connect.
"""

from pydantic import BaseModel, Field

from agentbuilder.Server import serve_agent
from agentbuilder.Tools.base import tool_from_function
from agentbuilder.utils import create_agent_factory


class AddParams(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


class MultiplyParams(BaseModel):
    x: int = Field(description="First factor")
    y: int = Field(description="Second factor")


def add(params: AddParams) -> int:
    """Add two numbers together."""
    return params.a + params.b


def multiply(params: MultiplyParams) -> int:
    """Multiply two numbers together."""
    return params.x * params.y


add_tool = tool_from_function(add)
multiply_tool = tool_from_function(multiply)


def main():
    factory = create_agent_factory(
        model_name=None,  # Uses MODEL from .env
        tools=[add_tool, multiply_tool],
        system_prompt="You are a calculator assistant.",
    )

    print("Starting calculator agent server on http://localhost:8100")
    serve_agent(
        factory,
        name="calculator",
        description="A calculator agent that can add and multiply numbers",
        port=8100,
    )


if __name__ == "__main__":
    main()
