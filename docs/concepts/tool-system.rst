Tool System
===========

Tools are the mechanism by which agents interact with the outside world.

Tool Class
----------

:class:`~agentbuilder.Tools.base.Tool` wraps a Python callable with:

- **name** -- identifier used in LLM function-calling.
- **description** -- human-readable text shown to the LLM.
- **parameters** -- JSON Schema dict describing accepted arguments.
- **function** -- the actual callable to execute.

The tool converts itself to OpenAI format via
:meth:`~agentbuilder.Tools.base.Tool.to_openai_format` and executes via
:meth:`~agentbuilder.Tools.base.Tool.execute`, which returns a
:class:`~agentbuilder.Tools.base.Response`.

Response
--------

:class:`~agentbuilder.Tools.base.Response` is a simple dataclass:

- ``success`` (bool) -- whether execution succeeded.
- ``data`` (Any) -- the return value.
- ``error`` (Optional[str]) -- error message on failure.

Creating Tools with ``tool_from_function``
------------------------------------------

The easiest way to create a tool is with
:func:`~agentbuilder.Tools.base.tool_from_function`:

.. code-block:: python

   from pydantic import BaseModel, Field
   from agentbuilder.Tools.base import tool_from_function

   class GreetParams(BaseModel):
       name: str = Field(description="Person to greet")

   def greet(params: GreetParams) -> str:
       """Greet someone by name."""
       return f"Hello, {params.name}!"

   tool = tool_from_function(greet)

Conventions:

- The function must accept **exactly one** parameter.
- That parameter must be annotated as a ``pydantic.BaseModel`` subclass.
- The function's docstring becomes the tool description.
- Return-type annotations are appended to the description automatically.

Manual Tool Construction
------------------------

For full control, construct a :class:`~agentbuilder.Tools.base.Tool`
directly:

.. code-block:: python

   from agentbuilder.Tools.base import Tool

   tool = Tool(
       name="greet",
       description="Greet someone",
       parameters={
           "type": "object",
           "properties": {
               "name": {"type": "string", "description": "Person to greet"},
           },
           "required": ["name"],
       },
       function=lambda name: f"Hello, {name}!",
   )

OpenAI Format
-------------

:meth:`Tool.to_openai_format` returns a dict matching the OpenAI tools
specification:

.. code-block:: python

   {
       "type": "function",
       "function": {
           "name": "greet",
           "description": "Greet someone",
           "parameters": { ... }
       }
   }

This is handled internally by
:class:`~agentbuilder.Action.base.MakeLLMRequestAction`.

Specialised Tools
-----------------

- :class:`~agentbuilder.Tools.agent_tool.AgentTool` -- delegate to a local
  sub-agent.
- :class:`~agentbuilder.Tools.remote_agent_tool.RemoteAgentTool` -- delegate
  to a remote agent over HTTP.
- :class:`~agentbuilder.Tools.code_execution.CodeExecutionTool` -- execute
  Python code in a sandbox.

See :doc:`/guides/custom-tools` for a detailed guide.
