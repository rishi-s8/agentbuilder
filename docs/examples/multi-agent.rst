Multi-Agent Delegation
======================

**Location:** ``examples/multi_agent/``

This example demonstrates a parent agent that delegates tasks to a
specialist sub-agent.

What It Does
------------

- Creates a math sub-agent with arithmetic tools.
- Wraps it as an :class:`~agentbuilder.Tools.agent_tool.AgentTool`.
- Creates a parent agent that can delegate math questions.

Key Code
--------

.. code-block:: python

   from agentbuilder.utils import create_agent, create_agent_tool
   from agentbuilder.Tools.base import tool_from_function

   # Create sub-agent tool
   math_agent = create_agent_tool(
       name="math_expert",
       description="Solves arithmetic problems",
       model_name="gpt-4o-mini",
       tools=[add_tool, multiply_tool],
       system_prompt="You are a math expert. Show your work.",
   )

   # Create parent agent
   parent = create_agent(
       model_name="gpt-4o-mini",
       tools=[math_agent],
       system_prompt="Delegate math questions to the math_expert tool.",
   )

   result = parent.run("What is 15 * 23 + 7?")
   print(result)

Running It
----------

.. code-block:: bash

   cd examples/multi_agent
   cp .env.example .env
   python multi_agent.py
