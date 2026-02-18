# Multi-Agent Delegation Example

A parent agent that delegates math questions to a specialist sub-agent.

## Setup

```bash
cp .env.example .env
# Edit .env with your API key
```

## Run

```bash
python multi_agent.py
```

## What It Demonstrates

- Creating a sub-agent tool with `create_agent_tool()`
- Composing agents in a parent-child relationship
- Automatic sub-agent reset between delegations
- Multi-step reasoning with delegation
