# AgentBuilder Examples

This directory contains examples demonstrating how to use the agentbuilder framework.

## Available Examples

### 1. Simple Calculator (`simple_calculator/`)

**Best for**: Learning tool orchestration

A practical calculator agent demonstrating:
- Multiple arithmetic tools
- Time/date utilities
- Multi-step problem solving

**Tools included**: add, multiply, get_current_time

[Go to Simple Calculator](simple_calculator/)

### 2. Code Execution Agent (`code_agent/`)

**Best for**: Running Python in Docker sandboxes

An agent that writes and executes Python code:
- Docker-based sandbox with persistent state
- Code execution, file I/O, package installation
- Automatic container cleanup

**Requires**: Docker, `pip install agentbuilder[code]`

[Go to Code Agent](code_agent/)

### 3. Multi-Agent Delegation (`multi_agent/`)

**Best for**: Composing specialist agents

A parent agent that delegates to a math sub-agent:
- Local sub-agent creation with `create_agent_tool()`
- Parent-child agent composition
- Automatic sub-agent reset between delegations

[Go to Multi-Agent](multi_agent/)

### 4. Served Agent (`served_agent/`)

**Best for**: Agents over HTTP

Serve an agent as a FastAPI service and connect from a client:
- Agent factory and server setup
- Session-isolated HTTP API
- Remote agent tool for client-side delegation

**Requires**: `pip install agentbuilder[server]`

[Go to Served Agent](served_agent/)

## Quick Start

1. **Install agentbuilder**:

   - **From GitHub (recommended for most users):**
     ```bash
     pip install git+https://github.com/rishi-s8/agentbuilder
     ```

   - **From local source (for development):**
     ```bash
     cd /path/to/agentbuilder
     pip install -e .
     ```

2. **Set up environment**:
   ```bash
   cd examples/<example_name>
   cp .env.example .env
   # Edit .env with your API key
   ```

3. **Run an example**:
   ```bash
   python <script_name>.py
   ```

## Example Structure

Each example follows this structure:
```
example_name/
├── README.md          # Detailed explanation
├── .env.example       # Environment template
└── main_script.py     # Runnable example
```

## Common Setup

All examples require:
- Python 3.9+
- An OpenAI API key (or compatible API)
- agentbuilder installed

Set your API credentials in a `.env` file:
```bash
OPENAI_API_KEY=your_key_here
MODEL=your_model_here
BASE_URL=your_base_url_here
```

## Learning Path

We recommend starting with:

1. **simple_calculator/** -- Learn tool orchestration and multi-step tasks
2. **multi_agent/** -- Learn agent composition and delegation
3. **code_agent/** -- Learn sandboxed code execution
4. **served_agent/** -- Learn HTTP agent serving

## Need Help?

- Check the main [README](../README.md)
- Read the [documentation](https://agentbuilder.readthedocs.io)
- Review the [source code](../src/agentbuilder/)

## Contributing

Have a cool example? Contributions are welcome! Make sure your example:
- Has clear documentation
- Demonstrates a specific use case or pattern
- Is well-commented for learning
