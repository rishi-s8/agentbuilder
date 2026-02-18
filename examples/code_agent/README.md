# Code Execution Agent Example

An agent that writes and executes Python code in an isolated Docker sandbox.

## Prerequisites

- Docker installed and running
- `pip install agentbuilder[code]`

## Setup

```bash
cp .env.example .env
# Edit .env with your API key
```

## Run

```bash
python code_agent.py
```

## What It Demonstrates

- Creating a `DockerSandbox` with security restrictions
- Using `create_code_agent()` to build a code-capable agent
- Executing Python code with persistent state between calls
- Automatic cleanup via context manager
