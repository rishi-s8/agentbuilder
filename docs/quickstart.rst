Quickstart
==========

This 5-minute tutorial walks you through building your first tool-calling
agent.

1. Install
----------

.. code-block:: bash

   pip install agentbuilder

2. Set Up Credentials
---------------------

Create a ``.env`` file:

.. code-block:: bash

   OPENAI_API_KEY=your_key_here
   MODEL=gpt-4o-mini

3. Define a Tool
----------------

Tools are plain Python functions annotated with a Pydantic model:

.. code-block:: python

   from pydantic import BaseModel, Field

   class AddParams(BaseModel):
       a: int = Field(description="First number")
       b: int = Field(description="Second number")

   def add(params: AddParams) -> int:
       """Add two numbers together."""
       return params.a + params.b

4. Create the Agent
-------------------

.. code-block:: python

   from agentbuilder.Tools.base import tool_from_function
   from agentbuilder.utils import create_agent

   tool = tool_from_function(add)

   agent = create_agent(
       model_name="gpt-4o-mini",
       tools=[tool],
       system_prompt="You are a helpful calculator.",
   )

5. Run It
---------

.. code-block:: python

   response = agent.run("What is 15 + 27?")
   print(response)
   # The assistant will use the add tool and respond with "42"

6. Reset for Next Conversation
------------------------------

.. code-block:: python

   agent.reset()
   response = agent.run("Now add 100 and 200")
   print(response)

Next Steps
----------

- :doc:`concepts/architecture` -- understand the component model.
- :doc:`guides/custom-tools` -- build more sophisticated tools.
- :doc:`guides/multi-agent` -- delegate tasks to sub-agents.
- :doc:`examples/index` -- runnable example projects.
