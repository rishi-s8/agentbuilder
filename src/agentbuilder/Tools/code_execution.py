"""
Code execution tools for sandboxed environments.
"""

from typing import List

from pydantic import BaseModel, Field

from agentbuilder.Sandbox.base import Sandbox
from agentbuilder.Tools.base import Tool, tool_from_function


class CodeExecutionTool(Tool):
    """Tool that executes code in a sandboxed environment."""

    def __init__(self, sandbox: Sandbox):
        """
        Initialize a CodeExecutionTool.

        Args:
            sandbox: The Sandbox instance to execute code in
        """
        self.sandbox = sandbox

        parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute",
                }
            },
            "required": ["code"],
        }

        super().__init__(
            name="execute_code",
            description="Execute Python code in a sandboxed environment. "
            "Variables and imports persist between calls.",
            parameters=parameters,
            function=self._execute,
        )

    def _execute(self, code: str) -> dict:
        """
        Execute code in the sandbox.

        Args:
            code: Python code to execute

        Returns:
            Dict with stdout, stderr, and success fields
        """
        result = self.sandbox.execute(code)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.success,
        }


def create_sandbox_tools(sandbox: Sandbox) -> List[Tool]:
    """
    Create file and package management tools for a sandbox.

    Args:
        sandbox: The Sandbox instance to create tools for

    Returns:
        List of Tool objects for read_file, write_file, and install_package
    """

    class ReadFileParams(BaseModel):
        path: str = Field(description="Path to the file inside the sandbox")

    def read_file(params: ReadFileParams) -> str:
        """Read a file from the sandbox filesystem."""
        return sandbox.read_file(params.path)

    class WriteFileParams(BaseModel):
        path: str = Field(description="Path to the file inside the sandbox")
        content: str = Field(description="Content to write to the file")

    def write_file(params: WriteFileParams) -> str:
        """Write a file to the sandbox filesystem."""
        sandbox.write_file(params.path, params.content)
        return f"File written to {params.path}"

    class InstallPackageParams(BaseModel):
        package: str = Field(description="Package name to install (e.g., 'numpy')")

    def install_package(params: InstallPackageParams) -> dict:
        """Install a Python package in the sandbox."""
        result = sandbox.install_package(params.package)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.success,
        }

    return [
        tool_from_function(read_file),
        tool_from_function(write_file),
        tool_from_function(install_package),
    ]
