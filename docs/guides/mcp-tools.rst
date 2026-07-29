Using MCP Tools
===============

AgentBuilder can discover tools from a remote Streamable HTTP MCP server and
adapt their JSON schemas to its normal tool interface.

Install the optional dependency (Python 3.10 or later):

.. code-block:: bash

   pip install agentbuilder[mcp]

Loading a Copilot-Compatible Config
-----------------------------------

For an HTTP server already authenticated by GitHub Copilot CLI:

.. code-block:: python

   from agentbuilder.Tools.mcp import MCPToolSet
   from agentbuilder.utils import create_agent

   with MCPToolSet.from_config(".mcp.json", "workiq_mail") as mail:
       agent = create_agent(
           model_name="gpt-chat-latest",
           tools=mail.tools,
           system_prompt="You are a careful mail assistant.",
       )
       print(agent.run("Find unread mail from today"))

``MCPToolSet`` reuses the matching registration and token in
``~/.copilot/mcp-oauth-config`` and silently refreshes the token when possible.
Credentials are not copied into the project.

Using an Explicit Bearer Token
------------------------------

Set ``MCP_ACCESS_TOKEN`` before calling ``from_config`` to override Copilot
OAuth cache discovery. You can also construct
:class:`~agentbuilder.Tools.mcp.StaticBearerTokenProvider` directly.

Confirming Consequential Tools
------------------------------

Pass ``confirmation_tools`` and ``confirmation_callback`` to require local
approval before selected MCP tools execute. The WorkIQ Mail example uses this
for send, reply, forward, delete, and update operations.
