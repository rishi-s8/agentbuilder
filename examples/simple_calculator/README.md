# Simple Calculator Agent Example

This is a minimal example showing how to build an agent using the agentbuilder framework with OpenAI client.

## Overview

This example demonstrates:
- Creating simple tools with Pydantic models
- Setting up an OpenAI-compatible agent
- Running an agentic loop with tool orchestration

## Features

The calculator agent has three simple tools:
- **add**: Add two numbers
- **multiply**: Multiply two numbers  
- **get_current_time**: Get the current date and time

## Running the Example

```bash
python calculator_agent.py
```

The agent will:
1. Demonstrate basic arithmetic operations
2. Show how to get the current time
3. Solve a multi-step problem using multiple tools

## Code Structure

- `calculator_agent.py`: Main agent implementation with tools and example usage
- `.env`: Configuration file for API credentials (you need to create this)

## How It Works

1. **Define Tools**: Each tool is defined as a simple function with a Pydantic model for parameters
2. **Create Agent**: Use `create_agent()` utility to set up the agent with tools
3. **Run Tasks**: Call `agent.run()` with natural language requests
4. **Tool Orchestration**: The agent automatically plans and executes the right tools
