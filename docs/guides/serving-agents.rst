Serving Agents over HTTP
========================

AgentBuilder can expose any agent as a FastAPI service with session
isolation.

Prerequisites
-------------

Install the server extra:

.. code-block:: bash

   pip install agentbuilder[server]

Creating an Agent Factory
-------------------------

The server needs a factory callable that produces fresh agent instances
(one per session):

.. code-block:: python

   from agentbuilder.utils import create_agent_factory

   factory = create_agent_factory(
       model_name="gpt-4o-mini",
       tools=[my_tool],
       system_prompt="You are a helpful assistant.",
   )

Starting the Server
-------------------

.. code-block:: python

   from agentbuilder.Server import serve_agent

   serve_agent(
       factory,
       name="my_agent",
       description="A helpful assistant",
       port=8100,
   )

API Endpoints
-------------

Once running, the server provides these endpoints:

.. list-table::
   :header-rows: 1

   * - Method
     - Path
     - Description
   * - GET
     - ``/info``
     - Agent name and description
   * - GET
     - ``/health``
     - Health check
   * - POST
     - ``/sessions``
     - Create a new session (returns ``session_id``)
   * - POST
     - ``/sessions/{id}/run``
     - Send a message (body: ``{"message": "..."}``); returns ``{"response": "..."}``
   * - POST
     - ``/sessions/{id}/reset``
     - Reset a session's conversation history
   * - DELETE
     - ``/sessions/{id}``
     - Delete a session

Usage Example
-------------

.. code-block:: bash

   # Health check
   $ curl http://localhost:8100/health
   {"status": "healthy"}

   # Create a session
   $ curl -X POST http://localhost:8100/sessions
   {"session_id": "abc123"}

   # Send a message
   $ curl -X POST http://localhost:8100/sessions/abc123/run \
       -H "Content-Type: application/json" \
       -d '{"message": "Hello!"}'
   {"response": "Hi there! How can I help?"}

Session Isolation
-----------------

Each session gets its own :class:`~agentbuilder.Loop.base.AgenticLoop`
instance with independent conversation history. Sessions do not share
state.

Using ``create_agent_app`` Directly
------------------------------------

For more control (e.g., mounting under a larger FastAPI app):

.. code-block:: python

   from agentbuilder.Server.base import create_agent_app

   app = create_agent_app(factory, "my_agent", "Description")
   # Use `app` with uvicorn or mount it as a sub-application
