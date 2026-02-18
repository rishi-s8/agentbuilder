Code Execution Agents
=====================

AgentBuilder can create agents that write and execute Python code in
isolated Docker containers.

Prerequisites
-------------

- Docker installed and running.
- The ``code`` extra: ``pip install agentbuilder[code]``

Setting Up a Sandbox
--------------------

.. code-block:: python

   from agentbuilder.Sandbox.docker_sandbox import DockerSandbox

   sandbox = DockerSandbox(
       image="python:3.11-slim",
       mem_limit="512m",
       network_disabled=True,
   )

The sandbox starts a container with a persistent REPL -- variables and
imports survive between calls.

Creating the Agent
------------------

:func:`~agentbuilder.utils.create_code_agent` assembles the agent with
code-execution tools automatically:

.. code-block:: python

   from agentbuilder.utils import create_code_agent

   with DockerSandbox() as sandbox:
       agent = create_code_agent(
           model_name="gpt-4o-mini",
           sandbox=sandbox,
       )
       result = agent.run("Write a function that checks if a number is prime, then test it on 17.")
       print(result)

Available Tools
---------------

The code agent gets four tools:

- **execute_code** -- run Python code in the sandbox.
- **read_file** -- read a file from the sandbox filesystem.
- **write_file** -- write a file to the sandbox filesystem.
- **install_package** -- ``pip install`` a package in the sandbox.

Security Notes
--------------

- Containers run with ``no-new-privileges`` and all capabilities dropped.
- Networking is disabled by default.
- Memory and CPU are limited.
- Always use the context manager (``with``) to ensure cleanup.

Mixing Tools
------------

Pass additional tools alongside the code-execution tools:

.. code-block:: python

   agent = create_code_agent(
       model_name="gpt-4o-mini",
       sandbox=sandbox,
       additional_tools=[my_custom_tool],
   )
