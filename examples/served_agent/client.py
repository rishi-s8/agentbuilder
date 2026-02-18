"""
Served Agent -- Client

Connects to the served calculator agent and delegates tasks to it.
Requires the server to be running (see server.py).
"""

from agentbuilder.utils import create_agent, create_remote_agent_tool


def main():
    print("Connecting to served agent at http://localhost:8100...")

    remote = create_remote_agent_tool("http://localhost:8100")
    print(f"Connected to: {remote.name} - {remote.description}")

    try:
        parent = create_agent(
            model_name=None,  # Uses MODEL from .env
            tools=[remote],
            system_prompt="You are a helpful assistant. Use the calculator tool for math.",
            verbose=True,
        )

        print("\n--- Sending task ---")
        response = parent.run("What is 42 + 58?")
        print(f"\nFinal Answer: {response}")

        parent.reset()

        print("\n--- Sending another task ---")
        response = parent.run("Multiply 7 by 8")
        print(f"\nFinal Answer: {response}")

    finally:
        remote.close()
        print("\nSession closed.")


if __name__ == "__main__":
    main()
