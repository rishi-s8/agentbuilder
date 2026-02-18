Simple Calculator
=================

**Location:** ``examples/simple_calculator/``

This example demonstrates basic tool-calling with arithmetic tools.

What It Does
------------

- Defines ``add``, ``multiply``, and ``get_current_time`` tools using
  Pydantic models and :func:`~agentbuilder.Tools.base.tool_from_function`.
- Creates an agent with :func:`~agentbuilder.utils.create_agent`.
- Runs four example tasks including multi-step reasoning.

Key Code
--------

.. code-block:: python

   from pydantic import BaseModel, Field
   from agentbuilder.Tools.base import tool_from_function
   from agentbuilder.utils import create_agent

   class AddParams(BaseModel):
       a: int = Field(description="First number to add")
       b: int = Field(description="Second number to add")

   def add(params: AddParams) -> int:
       """Add two numbers together and return the result."""
       return params.a + params.b

   add_tool = tool_from_function(add)

   agent = create_agent(
       model_name="gpt-4o-mini",
       tools=[add_tool],
       system_prompt="You are a calculator assistant.",
       verbose=True,
   )

   response = agent.run("What is 15 plus 27?")
   print(response)

Running It
----------

.. code-block:: bash

   cd examples/simple_calculator
   cp .env.example .env
   # Edit .env with your API key
   python calculator_agent.py
