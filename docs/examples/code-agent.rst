Code Execution Agent
====================

**Location:** ``examples/code_agent/``

This example demonstrates an agent that writes and executes Python code in
a Docker sandbox.

Prerequisites
-------------

- Docker installed and running
- ``pip install agentbuilder[code]``

What It Does
------------

- Creates a :class:`~agentbuilder.Sandbox.docker_sandbox.DockerSandbox`.
- Uses :func:`~agentbuilder.utils.create_code_agent` to build an agent
  with code execution, file I/O, and package installation tools.
- Asks the agent to solve a coding problem.

Key Code
--------

.. code-block:: python

   from agentbuilder.Sandbox.docker_sandbox import DockerSandbox
   from agentbuilder.utils import create_code_agent

   with DockerSandbox() as sandbox:
       agent = create_code_agent(
           model_name="gpt-4o-mini",
           sandbox=sandbox,
       )
       result = agent.run(
           "Write a Python script that generates the first 20 Fibonacci numbers "
           "and prints them as a formatted list."
       )
       print(result)

Running It
----------

.. code-block:: bash

   cd examples/code_agent
   cp .env.example .env
   python code_agent.py
