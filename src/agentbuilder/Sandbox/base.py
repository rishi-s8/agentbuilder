"""
Abstract sandbox interface for isolated code execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of code execution in a sandbox."""

    stdout: str
    stderr: str
    success: bool
    exit_code: int


class Sandbox(ABC):
    """Abstract base class for sandboxed code execution environments."""

    @abstractmethod
    def execute(self, code: str, timeout: int = 30) -> ExecutionResult:
        """
        Execute code in the sandbox.

        Args:
            code: The code to execute
            timeout: Maximum execution time in seconds

        Returns:
            ExecutionResult with stdout, stderr, success flag, and exit code
        """

    @abstractmethod
    def read_file(self, path: str) -> str:
        """
        Read a file from the sandbox filesystem.

        Args:
            path: Path to the file inside the sandbox

        Returns:
            File contents as a string
        """

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """
        Write a file to the sandbox filesystem.

        Args:
            path: Path to the file inside the sandbox
            content: Content to write
        """

    @abstractmethod
    def install_package(self, package: str) -> ExecutionResult:
        """
        Install a Python package in the sandbox.

        Args:
            package: Package name to install

        Returns:
            ExecutionResult from the installation
        """

    @abstractmethod
    def close(self) -> None:
        """Clean up sandbox resources."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
