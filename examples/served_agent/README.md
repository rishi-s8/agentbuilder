# Served Agent Example

Demonstrates serving an agent over HTTP and connecting to it from a client.

## Prerequisites

- `pip install agentbuilder[server]`

## Setup

```bash
cp .env.example .env
# Edit .env with your API key
```

## Run

```bash
# Terminal 1: Start the server
python server.py

# Terminal 2: Run the client
python client.py
```

## What It Demonstrates

- Creating an agent factory with `create_agent_factory()`
- Serving an agent with `serve_agent()`
- Connecting to a served agent with `create_remote_agent_tool()`
- Session isolation and automatic cleanup
