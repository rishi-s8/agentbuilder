"""
Loop module containing the core agentic execution loop.

The :class:`AgenticLoop` orchestrates the plan-execute cycle: it repeatedly
asks the planner for the next action and executes it until the agent completes
or the iteration limit is reached.
"""

from agentbuilder.Loop.base import AgenticLoop

__all__ = ["AgenticLoop"]
