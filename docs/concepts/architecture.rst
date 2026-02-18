Architecture
============

AgentBuilder is built around a small set of composable components. Each
component has a single responsibility and can be replaced or extended
independently.

Component Overview
------------------

::

   ┌─────────────────────────────────────────────────┐
   │                  AgenticLoop                     │
   │  ┌───────────┐   ┌──────────┐   ┌───────────┐  │
   │  │  Planner   │──▶│  Action   │──▶│  Tools    │  │
   │  └───────────┘   └──────────┘   └───────────┘  │
   │        │                              │         │
   │        ▼                              ▼         │
   │  ┌──────────────────────────────────────────┐   │
   │  │        ConversationWrapper (Client)       │   │
   │  └──────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────┘

Components
----------

**AgenticLoop** (:class:`~agentbuilder.Loop.base.AgenticLoop`)
   The main orchestrator. Repeatedly asks the Planner for the next action
   and executes it. Stops when the agent completes or the iteration limit
   is reached.

**Planner** (:class:`~agentbuilder.Planner.base.AgenticPlanner`)
   A stateless decision engine. Inspects the last message in the
   conversation and returns the appropriate action. Does **not** make any
   LLM calls itself.

**Actions** (:mod:`agentbuilder.Action`)
   Atomic units of work. Control-flow actions
   (:class:`~agentbuilder.Action.base.ExecuteToolsAction`,
   :class:`~agentbuilder.Action.base.MakeLLMRequestAction`,
   :class:`~agentbuilder.Action.base.CompleteAction`) drive the loop.
   Message actions store conversation history.

**ConversationWrapper** (:class:`~agentbuilder.Client.openai_client.ConversationWrapper`)
   Manages the chat transcript as a list of Action objects and provides
   methods to add messages, serialize to OpenAI format, and save/load
   history.

**Tools** (:mod:`agentbuilder.Tools`)
   Callable functions exposed to the LLM. Built from Python functions
   with Pydantic parameter models. Specialised tools include
   :class:`~agentbuilder.Tools.agent_tool.AgentTool` (local sub-agent),
   :class:`~agentbuilder.Tools.remote_agent_tool.RemoteAgentTool`
   (HTTP sub-agent), and
   :class:`~agentbuilder.Tools.code_execution.CodeExecutionTool`
   (sandboxed code execution).

**Sandbox** (:mod:`agentbuilder.Sandbox`)
   Isolated execution environments for running untrusted code.
   :class:`~agentbuilder.Sandbox.docker_sandbox.DockerSandbox` runs code
   in Docker containers with security restrictions.

**Server** (:mod:`agentbuilder.Server`)
   Wraps an agent in a FastAPI application with session isolation,
   enabling agents to be accessed over HTTP.

Data Flow
---------

1. User provides a message via ``agent.run(message)``.
2. The message is appended to the conversation history.
3. The Planner inspects the last message and returns an Action.
4. The Loop executes the Action (LLM call, tool execution, or completion).
5. Steps 3-4 repeat until a ``CompleteAction`` or ``EmptyAction`` is
   returned.
