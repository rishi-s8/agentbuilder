"""
Docker-based sandbox for isolated code execution with persistent state.
"""

import json
import os
import tarfile
import tempfile
from io import BytesIO

import docker

from agentbuilder.Sandbox.base import ExecutionResult, Sandbox

_REPL_SERVER_PATH = os.path.join(os.path.dirname(__file__), "_repl_server.py")


class DockerSandbox(Sandbox):
    """
    Sandbox that runs code in an isolated Docker container.

    Uses a persistent REPL server inside the container so variables,
    imports, and functions persist between execute() calls.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        mem_limit: str = "512m",
        cpu_quota: int = 50000,
        network_disabled: bool = True,
        working_dir: str = "/workspace",
    ):
        """
        Initialize DockerSandbox and start a container.

        Args:
            image: Docker image to use
            mem_limit: Memory limit for the container
            cpu_quota: CPU quota (microseconds per 100ms period)
            network_disabled: Whether to disable networking
            working_dir: Working directory inside the container
        """
        self.image = image
        self.working_dir = working_dir
        self.client = docker.from_env()

        # Start container with security restrictions
        self.container = self.client.containers.run(
            image,
            command=["sleep", "infinity"],
            detach=True,
            mem_limit=mem_limit,
            cpu_quota=cpu_quota,
            network_disabled=network_disabled,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            working_dir=working_dir,
        )

        # Create workspace directory
        self.container.exec_run(["mkdir", "-p", working_dir])

        # Copy the REPL server script into the container
        self._copy_file_to_container(
            _REPL_SERVER_PATH, f"{working_dir}/_repl_server.py"
        )

        # Start the REPL server process
        self._exec_id = self.client.api.exec_create(
            self.container.id,
            ["python", "-u", f"{working_dir}/_repl_server.py"],
            stdin=True,
            stdout=True,
            stderr=True,
            workdir=working_dir,
        )["Id"]
        self._socket = self.client.api.exec_start(
            self._exec_id, socket=True, demux=False
        )
        # Get the underlying socket for bidirectional communication
        self._sock = self._socket._sock

    def _copy_file_to_container(self, local_path: str, container_path: str):
        """Copy a file into the container using a tar archive."""
        with open(local_path, "rb") as f:
            file_data = f.read()

        tar_stream = BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            info = tarfile.TarInfo(name=os.path.basename(container_path))
            info.size = len(file_data)
            tar.addfile(info, BytesIO(file_data))
        tar_stream.seek(0)

        self.container.put_archive(os.path.dirname(container_path), tar_stream)

    def execute(self, code: str, timeout: int = 30) -> ExecutionResult:
        """
        Execute code in the persistent REPL.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            ExecutionResult with stdout, stderr, success, and exit_code
        """
        command = json.dumps({"code": code, "timeout": timeout}) + "\n"
        self._sock.sendall(command.encode())

        # Read response (may come in chunks, read until newline)
        data = b""
        while True:
            chunk = self._sock.recv(4096)
            if not chunk:
                return ExecutionResult(
                    stdout="",
                    stderr="REPL server connection closed",
                    success=False,
                    exit_code=1,
                )
            data += chunk
            if b"\n" in data:
                break

        # Parse the first complete line (docker may prepend stream header bytes)
        raw = data.split(b"\n")[0]
        # Docker stream protocol: first 8 bytes are header
        # Try to find JSON start
        json_start = raw.find(b"{")
        if json_start >= 0:
            raw = raw[json_start:]

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            return ExecutionResult(
                stdout="",
                stderr=f"Failed to parse REPL response: {raw!r}",
                success=False,
                exit_code=1,
            )

        return ExecutionResult(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            success=result.get("success", False),
            exit_code=result.get("exit_code", 1),
        )

    def read_file(self, path: str) -> str:
        """Read a file from the container."""
        exit_code, output = self.container.exec_run(["cat", path])
        if exit_code != 0:
            raise FileNotFoundError(
                f"File not found in container: {path} ({output.decode()})"
            )
        return output.decode()

    def write_file(self, path: str, content: str) -> None:
        """Write a file to the container."""
        tar_stream = BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = content.encode()
            info = tarfile.TarInfo(name=os.path.basename(path))
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
        tar_stream.seek(0)
        self.container.put_archive(os.path.dirname(path) or "/", tar_stream)

    def install_package(self, package: str) -> ExecutionResult:
        """Install a Python package in the container."""
        exit_code, output = self.container.exec_run(["pip", "install", package])
        stdout = output.decode()
        return ExecutionResult(
            stdout=stdout,
            stderr="" if exit_code == 0 else stdout,
            success=exit_code == 0,
            exit_code=exit_code,
        )

    def close(self) -> None:
        """Stop and remove the container."""
        try:
            self._sock.close()
        except Exception:
            pass
        try:
            self.container.stop(timeout=5)
            self.container.remove(force=True)
        except Exception:
            pass
