"""
Tools module for agentbuilder.

Provides the building blocks for exposing callable functions to an LLM agent:

- :class:`Tool` -- base tool wrapping a callable with an OpenAI-compatible
  JSON schema.
- :func:`tool_from_function` -- convenience factory that derives the schema
  automatically from a Pydantic-annotated function.
- :class:`AgentTool` -- delegates a task to a local sub-agent (in-process).
- :class:`RemoteAgentTool` -- delegates a task to a remote agent over HTTP.

Code-execution tools live in :mod:`agentbuilder.Tools.code_execution`.
"""

from agentbuilder.Tools.agent_tool import AgentTool
from agentbuilder.Tools.base import Tool, tool_from_function
from agentbuilder.Tools.remote_agent_tool import RemoteAgentTool

__all__ = ["Tool", "tool_from_function", "AgentTool", "RemoteAgentTool"]
