Multi-Agent Systems
===================

AgentBuilder supports composing multiple agents through delegation tools.

Local Sub-Agents
----------------

Use :class:`~agentbuilder.Tools.agent_tool.AgentTool` (or the
:func:`~agentbuilder.utils.create_agent_tool` helper) to delegate tasks
to an agent running in the same process:

.. code-block:: python

   from agentbuilder.utils import create_agent, create_agent_tool

   # Create a specialist sub-agent tool
   math_tool = create_agent_tool(
       name="math_expert",
       description="Solves arithmetic problems step by step",
       model_name="gpt-4o-mini",
       tools=[add_tool, multiply_tool],
       system_prompt="You solve math problems.",
   )

   # Create the parent agent with the sub-agent as a tool
   parent = create_agent(
       model_name="gpt-4o-mini",
       tools=[math_tool],
       system_prompt="You are a helpful assistant. Delegate math questions to the math_expert tool.",
   )

   result = parent.run("What is 15 * 23?")

The sub-agent is **reset before each delegation**, ensuring a fresh
context every time.

Remote Sub-Agents
-----------------

Use :class:`~agentbuilder.Tools.remote_agent_tool.RemoteAgentTool` to
delegate to an agent running as an HTTP server:

.. code-block:: python

   from agentbuilder.utils import create_remote_agent_tool, create_agent

   remote = create_remote_agent_tool("http://localhost:8100")
   try:
       parent = create_agent(
           model_name="gpt-4o-mini",
           tools=[remote],
       )
       result = parent.run("Delegate this task")
   finally:
       remote.close()

The remote tool auto-discovers the agent's name and description from the
``/info`` endpoint and manages a session lifecycle automatically.

Combining Patterns
------------------

You can mix local and remote sub-agents freely:

.. code-block:: python

   parent = create_agent(
       model_name="gpt-4o-mini",
       tools=[local_math_agent, remote_search_agent, my_custom_tool],
   )

The parent agent's LLM decides which tool to use based on the task.
