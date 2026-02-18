# AgentBuilder

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://agentbuilder.readthedocs.io)

A flexible Python framework for building agentic AI systems with tool orchestration, planning, and conversation management.

> **Warning**: This project is under active development with regular breaking changes in the API.

## Key Features

- **Tool-calling agents** -- define tools as Python functions with Pydantic models and let the LLM orchestrate them
- **Code execution** -- run Python in isolated Docker sandboxes with persistent state
- **Multi-agent delegation** -- compose agents locally or across HTTP boundaries
- **Server mode** -- expose any agent as a session-isolated FastAPI service
- **Conversation management** -- save, load, and reset chat history
- **OpenAI-compatible** -- works with any OpenAI-compatible API endpoint

## Quick Install

```bash
pip install agentbuilder
```

## Example

```python
from pydantic import BaseModel, Field
from agentbuilder.Tools.base import tool_from_function
from agentbuilder.utils import create_agent

class AddParams(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")

def add(params: AddParams) -> int:
    """Add two numbers."""
    return params.a + params.b

agent = create_agent(
    model_name="gpt-4o-mini",
    tools=[tool_from_function(add)],
    system_prompt="You are a calculator assistant.",
)

response = agent.run("What is 15 + 27?")
print(response)

agent.reset()  # Clear state for next conversation
```

## Architecture

```
AgenticLoop  ←→  Planner  →  Actions
     ↕                          ↕
ConversationWrapper          Tools
                               ↕
                    Sandbox / SubAgents / Server
```

- **AgenticLoop** orchestrates the plan-execute cycle
- **Planner** decides the next action based on conversation state
- **Actions** execute tool calls, LLM requests, or signal completion
- **Tools** wrap Python functions for LLM function-calling
- **Sandbox** provides isolated Docker environments for code execution
- **Server** exposes agents over HTTP with session isolation

## Optional Dependencies

| Extra | What it adds | Install |
|-------|-------------|---------|
| `server` | FastAPI + Uvicorn for HTTP serving | `pip install agentbuilder[server]` |
| `code` | Docker SDK for sandboxed code execution | `pip install agentbuilder[code]` |
| `dev` | pytest, black, isort for development | `pip install agentbuilder[dev]` |
| `docs` | Sphinx + Furo for documentation | `pip install agentbuilder[docs]` |

## Documentation

Full documentation is available at [agentbuilder.readthedocs.io](https://agentbuilder.readthedocs.io), including:

- [Quickstart](https://agentbuilder.readthedocs.io/quickstart.html)
- [Concept Guides](https://agentbuilder.readthedocs.io/concepts/index.html)
- [User Guides](https://agentbuilder.readthedocs.io/guides/index.html)
- [API Reference](https://agentbuilder.readthedocs.io/api/index.html)
- [Examples](https://agentbuilder.readthedocs.io/examples/index.html)

## Examples

See the [`examples/`](examples/) directory for runnable projects:

- **[Simple Calculator](examples/simple_calculator/)** -- basic tool orchestration
- **[Code Agent](examples/code_agent/)** -- Docker sandbox code execution
- **[Multi-Agent](examples/multi_agent/)** -- parent-child agent delegation
- **[Served Agent](examples/served_agent/)** -- HTTP agent serving and remote access

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bash
git clone https://github.com/rishi-s8/agentbuilder.git
cd agentbuilder
pip install -e ".[dev]"
pytest
```

## License

CC BY-SA 4.0 (See LICENSE file for details.)
