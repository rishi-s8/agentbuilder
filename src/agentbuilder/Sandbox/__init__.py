"""
Sandbox module for isolated code execution.

Provides abstract and concrete sandbox implementations for running untrusted
code safely:

- :class:`Sandbox` -- abstract base class defining the sandbox interface.
- :class:`ExecutionResult` -- dataclass holding execution output.
- :class:`DockerSandbox` -- production implementation using Docker containers
  with a persistent REPL server.
"""

from agentbuilder.Sandbox.base import ExecutionResult, Sandbox
from agentbuilder.Sandbox.docker_sandbox import DockerSandbox

__all__ = ["Sandbox", "ExecutionResult", "DockerSandbox"]
