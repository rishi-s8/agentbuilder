"""
Abstract sandbox interface for isolated code execution.

Defines the :class:`Sandbox` ABC that all sandbox implementations must
follow, and the :class:`ExecutionResult` dataclass returned by execution
methods.

Implementations:
    :class:`~agentbuilder.Sandbox.docker_sandbox.DockerSandbox` --
    Docker-based sandbox with a persistent REPL.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of code execution in a sandbox.

    Attributes:
        stdout: Standard output captured during execution.
        stderr: Standard error captured during execution.
        success: ``True`` if the code ran without errors.
        exit_code: Process exit code (``0`` on success).
    """

    stdout: str
    stderr: str
    success: bool
    exit_code: int


class Sandbox(ABC):
    """Abstract base class for sandboxed code execution environments.

    Subclasses must implement :meth:`execute`, :meth:`read_file`,
    :meth:`write_file`, :meth:`install_package`, and :meth:`close`.

    Supports the context-manager protocol for automatic cleanup::

        with DockerSandbox() as sandbox:
            result = sandbox.execute("print('hello')")
            print(result.stdout)
        # sandbox.close() called automatically
    """

    @abstractmethod
    def execute(self, code: str, timeout: int = 30) -> ExecutionResult:
        """
        Execute code in the sandbox.

        Args:
            code: The Python code to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            An :class:`ExecutionResult` with stdout, stderr, success
            flag, and exit code.
        """

    @abstractmethod
    def read_file(self, path: str) -> str:
        """
        Read a file from the sandbox filesystem.

        Args:
            path: Path to the file inside the sandbox.

        Returns:
            File contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """
        Write a file to the sandbox filesystem.

        Args:
            path: Path to the file inside the sandbox.
            content: Content to write.
        """

    @abstractmethod
    def install_package(self, package: str) -> ExecutionResult:
        """
        Install a Python package in the sandbox.

        Args:
            package: Package name to install (e.g. ``"numpy"``).

        Returns:
            An :class:`ExecutionResult` from the installation command.
        """

    @abstractmethod
    def close(self) -> None:
        """Clean up sandbox resources (stop containers, release locks, etc.)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
