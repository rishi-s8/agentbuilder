Building Tool-Calling Agents
============================

This guide walks through creating an agent that uses tools to answer
questions.

Defining Tools
--------------

Every tool needs a Pydantic model for its parameters and a function:

.. code-block:: python

   from pydantic import BaseModel, Field

   class MultiplyParams(BaseModel):
       x: int = Field(description="First factor")
       y: int = Field(description="Second factor")

   def multiply(params: MultiplyParams) -> int:
       """Multiply two numbers together."""
       return params.x * params.y

Convert to a :class:`~agentbuilder.Tools.base.Tool`:

.. code-block:: python

   from agentbuilder.Tools.base import tool_from_function
   tool = tool_from_function(multiply)

Creating the Agent
------------------

Use :func:`~agentbuilder.utils.create_agent` to wire everything together:

.. code-block:: python

   from agentbuilder.utils import create_agent

   agent = create_agent(
       model_name="gpt-4o-mini",
       tools=[tool],
       system_prompt="You are a math tutor. Use tools when needed.",
       verbose=True,
       max_iterations=20,
   )

Configuration Options
---------------------

- **model_name** -- any OpenAI-compatible model identifier.
- **verbose** -- set to ``True`` to see each planning step and tool call.
- **max_iterations** -- safety limit; increase for complex multi-step tasks.
- **system_prompt** -- guides the LLM's behaviour and tool-use strategy.
- **base_url** -- point to a custom API endpoint (e.g. local inference).

Running the Agent
-----------------

.. code-block:: python

   response = agent.run("What is 6 times 7?")
   print(response)

Multi-Step Reasoning
--------------------

The agent can chain multiple tool calls in a single conversation:

.. code-block:: python

   response = agent.run("Add 5 and 10, then multiply the result by 3.")
   print(response)

The LLM will request the ``add`` tool first, receive the result, then
request the ``multiply`` tool with the intermediate value.

Resetting Between Conversations
--------------------------------

Always call :meth:`~agentbuilder.Loop.base.AgenticLoop.reset` between
independent conversations to clear history:

.. code-block:: python

   agent.reset()
   response = agent.run("New question here")
