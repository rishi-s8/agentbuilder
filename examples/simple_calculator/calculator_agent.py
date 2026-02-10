"""
Simple Calculator Agent Example

This example demonstrates how to create a basic agent with simple tools
using the agentbuilder framework and OpenAI client.
"""

import os
from datetime import datetime

from pydantic import BaseModel, Field

from agentbuilder.Tools.base import tool_from_function
from agentbuilder.utils import create_agent


# Define parameter models for each tool
class AddParams(BaseModel):
    """Parameters for adding two numbers"""

    a: int = Field(description="First number to add")
    b: int = Field(description="Second number to add")


class AddReturn(BaseModel):
    """Return model for adding two numbers"""

    result: int = Field(description="Result of the addition")


class MultiplyParams(BaseModel):
    """Parameters for multiplying two numbers"""

    x: int = Field(description="First number to multiply")
    y: int = Field(description="Second number to multiply")


class MultiplyReturn(BaseModel):
    """Return model for multiplying two numbers"""

    result: int = Field(description="Result of the multiplication")


class TimeParams(BaseModel):
    """No parameters needed for getting time"""

    pass


class TimeReturn(BaseModel):
    """Return model for getting current time"""

    datetime: str = Field(description="Current date and time")
    date: str = Field(description="Current date")
    time: str = Field(description="Current time")


# Define tool functions
def add(params: AddParams) -> AddReturn:
    """Add two numbers together and return the result"""
    result = params.a + params.b
    return AddReturn(result=result)


def multiply(params: MultiplyParams) -> MultiplyReturn:
    """Multiply two numbers together and return the result"""
    result = params.x * params.y
    return MultiplyReturn(result=result)


def get_current_time(params: TimeParams) -> TimeReturn:
    """Get the current date and time"""
    now = datetime.now()
    return TimeReturn(
        datetime=now.strftime("%Y-%m-%d %H:%M:%S"),
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H:%M:%S"),
    )


# Create tools from functions
add_tool = tool_from_function(add)
multiply_tool = tool_from_function(multiply)
time_tool = tool_from_function(get_current_time)


def main():
    """Run the calculator agent with example tasks"""

    # Create the agent with our tools
    model_name = os.environ.get("MODEL")
    base_url = os.environ.get("BASE_URL")

    agent = create_agent(
        model_name=model_name,
        base_url=base_url,
        tools=[add_tool, multiply_tool, time_tool],
        system_prompt="You are a helpful calculator assistant. Use the available tools to help users with calculations and time queries.",
        verbose=True,
        max_iterations=10,
    )

    print("=" * 60)
    print("Calculator Agent Example")
    print("=" * 60)
    print()

    # Example 1: Simple addition
    print("\n--- Example 1: Simple Addition ---")
    response = agent.run("What is 15 plus 27?")
    print(f"\nFinal Answer: {response}")

    # Reset the agent for next task
    agent.reset()

    # Example 2: Multiplication
    print("\n\n--- Example 2: Multiplication ---")
    response = agent.run("Calculate 8 times 9")
    print(f"\nFinal Answer: {response}")

    # Reset the agent for next task
    agent.reset()

    # Example 3: Get current time
    print("\n\n--- Example 3: Current Time ---")
    response = agent.run("What time is it now?")
    print(f"\nFinal Answer: {response}")

    # Reset the agent for next task
    agent.reset()

    # Example 4: Multi-step problem
    print("\n\n--- Example 4: Multi-step Problem ---")
    response = agent.run("First add 5 and 10, then multiply the result by 3")
    print(f"\nFinal Answer: {response}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
