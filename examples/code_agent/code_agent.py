"""
Code Execution Agent Example

Demonstrates an agent that writes and executes Python code in a Docker sandbox,
including reusing tools defined for tool-calling agents.

Requires Docker to be running and the 'code' extra installed:
    pip install agentbuilder[code]
"""

from pydantic import BaseModel, Field

from agentbuilder.Sandbox.docker_sandbox import DockerSandbox
from agentbuilder.Tools.base import tool_from_function
from agentbuilder.utils import create_agent, create_code_agent


# --- Define a reusable tool ---

class AddParams(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


def add(params: AddParams) -> int:
    """Add two numbers together."""
    return params.a + params.b


add_tool = tool_from_function(add)


def main():
    """Run the code execution agent with example tasks."""
    print("=" * 60)
    print("Code Execution Agent Example")
    print("=" * 60)
    print()

    # Example 1: Tool-calling agent uses add_tool via function calling
    print("--- Example 1: Tool-calling agent ---")
    agent1 = create_agent(
        model_name=None,  # Uses MODEL from .env
        tools=[add_tool],
        system_prompt="You are a calculator. Use the add tool.",
    )
    response = agent1.run("What is 5 + 3?")
    print(f"\nFinal Answer: {response}")

    # Example 2: Code agent — same tool injected into the sandbox
    print("\n\n--- Example 2: Code agent with injected tool ---")
    with DockerSandbox(image="python:3.11-slim") as sandbox:
        agent2 = create_code_agent(
            model_name=None,  # Uses MODEL from .env
            sandbox=sandbox,
            tools=[add_tool],
            verbose=True,
        )

        # The agent can call add(a=5, b=3) directly in its generated code
        response = agent2.run(
            "Use the add function to compute 5 + 3, then print the result."
        )
        print(f"\nFinal Answer: {response}")

        agent2.reset()

        # Example 3: Code agent without injected tools
        print("\n\n--- Example 3: Code agent (plain) ---")
        agent3 = create_code_agent(
            model_name=None,
            sandbox=sandbox,
            verbose=True,
        )

        response = agent3.run(
            "Write a Python script that generates the first 20 Fibonacci numbers "
            "and prints them as a formatted list."
        )
        print(f"\nFinal Answer: {response}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
