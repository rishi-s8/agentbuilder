"""
Code Execution Agent Example

Demonstrates an agent that writes and executes Python code in a Docker sandbox.
Requires Docker to be running and the 'code' extra installed:
    pip install agentbuilder[code]
"""

from agentbuilder.Sandbox.docker_sandbox import DockerSandbox
from agentbuilder.utils import create_code_agent


def main():
    """Run the code execution agent with example tasks."""
    print("=" * 60)
    print("Code Execution Agent Example")
    print("=" * 60)
    print()

    with DockerSandbox(image="python:3.11-slim") as sandbox:
        agent = create_code_agent(
            model_name=None,  # Uses MODEL from .env
            sandbox=sandbox,
            verbose=True,
        )

        # Example 1: Generate Fibonacci numbers
        print("\n--- Example 1: Fibonacci Numbers ---")
        response = agent.run(
            "Write a Python script that generates the first 20 Fibonacci numbers "
            "and prints them as a formatted list."
        )
        print(f"\nFinal Answer: {response}")

        agent.reset()

        # Example 2: Data analysis
        print("\n\n--- Example 2: Simple Data Analysis ---")
        response = agent.run(
            "Create a list of 10 random numbers between 1 and 100, "
            "then calculate and print the mean, median, min, and max."
        )
        print(f"\nFinal Answer: {response}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
