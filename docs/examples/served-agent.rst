Served Agent
============

**Location:** ``examples/served_agent/``

This example demonstrates serving an agent over HTTP and connecting to it
from a parent agent via :class:`~agentbuilder.Tools.remote_agent_tool.RemoteAgentTool`.

Prerequisites
-------------

- ``pip install agentbuilder[server]``

What It Does
------------

1. ``server.py`` -- serves a calculator agent on port 8100.
2. ``client.py`` -- creates a parent agent that delegates to the served
   agent using :func:`~agentbuilder.utils.create_remote_agent_tool`.

Server Code
-----------

.. code-block:: python

   from agentbuilder.utils import create_agent_factory
   from agentbuilder.Server import serve_agent

   factory = create_agent_factory(
       model_name="gpt-4o-mini",
       tools=[add_tool, multiply_tool],
       system_prompt="You are a calculator.",
   )

   serve_agent(factory, name="calculator", description="A calculator agent", port=8100)

Client Code
-----------

.. code-block:: python

   from agentbuilder.utils import create_remote_agent_tool, create_agent

   remote = create_remote_agent_tool("http://localhost:8100")
   try:
       parent = create_agent(
           model_name="gpt-4o-mini",
           tools=[remote],
       )
       result = parent.run("What is 42 + 58?")
       print(result)
   finally:
       remote.close()

Running It
----------

.. code-block:: bash

   # Terminal 1: Start the server
   cd examples/served_agent
   cp .env.example .env
   python server.py

   # Terminal 2: Run the client
   cd examples/served_agent
   python client.py
