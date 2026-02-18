"""
Action data types for the agentic framework.

This module provides all action classes used by the planner and loop to
represent decisions and conversation messages.

Action classes fall into two categories:

**Control-flow actions** (returned by :class:`~agentbuilder.Planner.base.AgenticPlanner`):
    - :class:`ExecuteToolsAction` -- execute pending tool calls.
    - :class:`MakeLLMRequestAction` -- send the conversation to the LLM.
    - :class:`CompleteAction` -- the agent has finished (final text ready).
    - :class:`EmptyAction` -- the conversation is empty; nothing to do.

**Message actions** (stored in conversation history):
    - :class:`SystemMessageAction`
    - :class:`UserMessageAction`
    - :class:`AssistantMessageAction`
    - :class:`ToolMessageAction`
"""

from agentbuilder.Action.base import (
    Action,
    AssistantMessageAction,
    CompleteAction,
    EmptyAction,
    ExecuteToolsAction,
    MakeLLMRequestAction,
    SystemMessageAction,
    ToolMessageAction,
    UserMessageAction,
)

__all__ = [
    "Action",
    "ExecuteToolsAction",
    "MakeLLMRequestAction",
    "CompleteAction",
    "EmptyAction",
    "SystemMessageAction",
    "UserMessageAction",
    "AssistantMessageAction",
    "ToolMessageAction",
]
