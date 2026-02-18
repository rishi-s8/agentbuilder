"""
Multi-Agent Delegation Example

Demonstrates a parent agent that delegates tasks to a specialist sub-agent
running in the same process.
"""

from pydantic import BaseModel, Field

from agentbuilder.Tools.base import tool_from_function
from agentbuilder.utils import create_agent, create_agent_tool


# Define arithmetic tools for the sub-agent
class AddParams(BaseModel):
    """Parameters for adding two numbers."""

    a: int = Field(description="First number")
    b: int = Field(description="Second number")


class MultiplyParams(BaseModel):
    """Parameters for multiplying two numbers."""

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
    """Run the multi-agent delegation example."""
    print("=" * 60)
    print("Multi-Agent Delegation Example")
    print("=" * 60)
    print()

    # Create a specialist sub-agent tool
    math_agent = create_agent_tool(
        name="math_expert",
        description="A specialist that solves arithmetic problems step by step. "
        "Delegate any math questions to this tool.",
        model_name=None,  # Uses MODEL from .env
        tools=[add_tool, multiply_tool],
        system_prompt="You are a math expert. Solve problems step by step using the available tools.",
        max_iterations=20,
    )

    # Create the parent agent with the sub-agent as a tool
    parent = create_agent(
        model_name=None,  # Uses MODEL from .env
        tools=[math_agent],
        system_prompt=(
            "You are a helpful assistant. When the user asks math questions, "
            "delegate them to the math_expert tool."
        ),
        verbose=True,
        max_iterations=20,
    )

    # Example 1: Simple delegation
    print("\n--- Example 1: Simple Delegation ---")
    response = parent.run("What is 15 times 23?")
    print(f"\nFinal Answer: {response}")

    parent.reset()

    # Example 2: Multi-step delegation
    print("\n\n--- Example 2: Multi-Step Delegation ---")
    response = parent.run("Add 100 and 250, then multiply the result by 4.")
    print(f"\nFinal Answer: {response}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
