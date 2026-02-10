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

[Go to Simple Calculator →](simple_calculator/)

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
   cd examples/simple_calculator
   cp .env.example .env
   # Edit .env with your API key
   ```

3. **Run an example**:
   ```bash
   python simple_agent.py
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

- **simple_calculator/** - Learn tool orchestration and multi-step tasks

## Creating Your Own Examples

To create a new example:

1. Create a directory: `examples/my_example/`
2. Add a `README.md` explaining what it does
3. Create your main script with:
   - Tool definitions (Pydantic models + functions)
   - Agent creation with `create_agent()`
   - Example usage demonstrating your tools
4. Add `.env.example` for configuration

## Need Help?

- Check the main [README](../README.md)
- Review the [source code](../src/agentbuilder/)
- Look at existing examples for patterns

## Contributing

Have a cool example? Contributions are welcome! Make sure your example:
- Has clear documentation
- Demonstrates a specific use case or pattern
- Is well-commented for learning
