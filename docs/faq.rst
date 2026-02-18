FAQ
===

API Key Issues
--------------

**I get an authentication error.**

Make sure your ``.env`` file is in the working directory and contains:

.. code-block:: bash

   OPENAI_API_KEY=sk-...

Or pass ``api_key`` explicitly:

.. code-block:: python

   agent = create_agent(
       model_name="gpt-4o-mini",
       tools=[...],
       api_key="sk-...",
   )

Docker Issues
-------------

**Docker is not available / ``docker.errors.DockerException``.**

- Ensure Docker Desktop (or dockerd) is running.
- Check with ``docker ps`` from the terminal.
- On Linux, make sure your user is in the ``docker`` group.

**Container runs out of memory.**

Increase the ``mem_limit`` parameter:

.. code-block:: python

   sandbox = DockerSandbox(mem_limit="1g")

Debugging
---------

**How do I see what the agent is doing?**

Set ``verbose=True`` (the default for :func:`~agentbuilder.utils.create_agent`):

.. code-block:: python

   agent = create_agent(
       model_name="gpt-4o-mini",
       tools=[...],
       verbose=True,
   )

This prints each planning step, tool call, and LLM request.

**The agent returns "Max iterations reached".**

Increase ``max_iterations``:

.. code-block:: python

   agent = create_agent(
       model_name="gpt-4o-mini",
       tools=[...],
       max_iterations=100,
   )

Custom API Endpoints
--------------------

**How do I use a different LLM provider?**

Pass ``base_url`` to point to any OpenAI-compatible endpoint:

.. code-block:: python

   agent = create_agent(
       model_name="my-model",
       tools=[...],
       base_url="http://localhost:11434/v1",  # e.g. Ollama
   )

Or set ``BASE_URL`` in your ``.env`` file.
