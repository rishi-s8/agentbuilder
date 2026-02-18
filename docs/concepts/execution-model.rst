Execution Model
===============

The Plan-Execute Loop
---------------------

AgentBuilder uses a plan-execute loop where the
:class:`~agentbuilder.Planner.base.AgenticPlanner` inspects the
conversation state and the
:class:`~agentbuilder.Loop.base.AgenticLoop` executes the resulting
action.

::

   User message
        │
        ▼
   ┌─── Loop iteration ────────────────────────────┐
   │  Planner.step()                                │
   │    │                                           │
   │    ├── Empty history?    → EmptyAction (done)  │
   │    ├── Last = Assistant  → CompleteAction       │
   │    │   (text, no tools)    (done)              │
   │    ├── Last = Assistant  → ExecuteToolsAction   │
   │    │   (with tool_calls)   (execute tools)     │
   │    └── Last = User /    → MakeLLMRequestAction  │
   │        Tool message        (call LLM)          │
   │                                                │
   │  action.run()                                  │
   │    │                                           │
   │    └── Updates conversation history            │
   └────────────────────────────────────────────────┘
        │
        ▼
   Repeat until CompleteAction or max_iterations

Planner Decision Tree
---------------------

The planner is a simple, deterministic state machine:

1. **Empty conversation** -- return ``EmptyAction``.
2. **Last message = AssistantMessage with tool_calls** -- return
   ``ExecuteToolsAction`` to process the pending tool calls.
3. **Last message = ToolMessage or UserMessage** -- return
   ``MakeLLMRequestAction`` to get the next LLM response.
4. **Last message = AssistantMessage with text, no tool_calls** -- return
   ``CompleteAction`` (the agent is done).

Termination Conditions
----------------------

The loop terminates when:

- A ``CompleteAction`` is returned (normal completion).
- An ``EmptyAction`` is returned (no messages to process).
- ``max_iterations`` is reached (safety limit). In this case the loop
  returns ``"Max iterations reached"``.

Iteration Counting
------------------

Each trip through the loop counts as one iteration. A typical
tool-calling interaction takes 2-3 iterations:

1. ``MakeLLMRequestAction`` -- sends the user message to the LLM.
2. ``ExecuteToolsAction`` -- executes the tool calls.
3. ``MakeLLMRequestAction`` -- sends tool results back to the LLM.
4. ``CompleteAction`` -- the LLM responds with final text.

Set ``max_iterations`` high enough for your use case (default is 80 for
:func:`~agentbuilder.utils.create_agent`).
