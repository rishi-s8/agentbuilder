AgentBuilder
============

A flexible Python framework for building agentic AI systems with tool
orchestration, planning, and conversation management.

.. warning::

   This project is under active development with regular breaking changes
   in the API.

Key Features
------------

- **Tool-calling agents** -- define tools as Python functions with Pydantic
  models and let the LLM orchestrate them.
- **Code execution** -- run Python in isolated Docker sandboxes with
  persistent state.
- **Multi-agent delegation** -- compose agents locally or across HTTP
  boundaries.
- **Server mode** -- expose any agent as a session-isolated FastAPI service.
- **Conversation management** -- save, load, and reset chat history.

Getting Started
---------------

.. code-block:: bash

   pip install agentbuilder

.. code-block:: python

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
   print(agent.run("What is 2 + 3?"))

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   concepts/index
   guides/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Examples & Extras

   examples/index
   faq
   contributing
   changelog
