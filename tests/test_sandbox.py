"""Tests for Sandbox module - DockerSandbox with mocked Docker client."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from agentbuilder.Sandbox.base import ExecutionResult, Sandbox
from agentbuilder.Sandbox.docker_sandbox import DockerSandbox


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_creation(self):
        result = ExecutionResult(stdout="hello", stderr="", success=True, exit_code=0)
        assert result.stdout == "hello"
        assert result.stderr == ""
        assert result.success is True
        assert result.exit_code == 0

    def test_failed_result(self):
        result = ExecutionResult(
            stdout="", stderr="error occurred", success=False, exit_code=1
        )
        assert result.success is False
        assert result.exit_code == 1


class TestSandboxABC:
    """Tests for Sandbox abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Sandbox()

    def test_context_manager_protocol(self):
        """Test that a concrete sandbox supports context manager."""

        class FakeSandbox(Sandbox):
            closed = False

            def execute(self, code, timeout=30):
                return ExecutionResult("", "", True, 0)

            def read_file(self, path):
                return ""

            def write_file(self, path, content):
                pass

            def install_package(self, package):
                return ExecutionResult("", "", True, 0)

            def close(self):
                self.closed = True

        with FakeSandbox() as sb:
            assert isinstance(sb, Sandbox)
        assert sb.closed is True


class TestDockerSandbox:
    """Tests for DockerSandbox with mocked Docker client."""

    @patch("agentbuilder.Sandbox.docker_sandbox.docker")
    def _make_sandbox(self, mock_docker):
        """Create a DockerSandbox with mocked Docker."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        # Mock the exec for REPL server startup
        mock_sock = MagicMock()
        mock_socket_wrapper = MagicMock()
        mock_socket_wrapper._sock = mock_sock

        mock_client.api.exec_create.return_value = {"Id": "exec123"}
        mock_client.api.exec_start.return_value = mock_socket_wrapper

        sandbox = DockerSandbox()

        return sandbox, mock_client, mock_container, mock_sock

    def test_creation_starts_container(self):
        """Test DockerSandbox creates container with correct config."""
        sandbox, mock_client, mock_container, _ = self._make_sandbox()

        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args

        assert call_kwargs[0][0] == "python:3.11-slim"
        assert call_kwargs[1]["detach"] is True
        assert call_kwargs[1]["mem_limit"] == "512m"
        assert call_kwargs[1]["cpu_quota"] == 50000
        assert call_kwargs[1]["network_disabled"] is True
        assert call_kwargs[1]["security_opt"] == ["no-new-privileges"]
        assert call_kwargs[1]["cap_drop"] == ["ALL"]

    def test_execute_sends_json_command(self):
        """Test execute() sends correct JSON to the REPL socket."""
        import json

        sandbox, _, _, mock_sock = self._make_sandbox()

        response_json = json.dumps(
            {"stdout": "42\n", "stderr": "", "success": True, "exit_code": 0}
        )
        mock_sock.recv.return_value = response_json.encode() + b"\n"

        result = sandbox.execute("print(42)")

        # Verify command was sent
        mock_sock.sendall.assert_called_once()
        sent = mock_sock.sendall.call_args[0][0].decode()
        command = json.loads(sent.strip())
        assert command["code"] == "print(42)"
        assert command["timeout"] == 30

        assert isinstance(result, ExecutionResult)
        assert result.stdout == "42\n"
        assert result.success is True

    def test_execute_with_custom_timeout(self):
        """Test execute() passes custom timeout."""
        import json

        sandbox, _, _, mock_sock = self._make_sandbox()

        response_json = json.dumps(
            {"stdout": "", "stderr": "", "success": True, "exit_code": 0}
        )
        mock_sock.recv.return_value = response_json.encode() + b"\n"

        sandbox.execute("pass", timeout=60)

        sent = mock_sock.sendall.call_args[0][0].decode()
        command = json.loads(sent.strip())
        assert command["timeout"] == 60

    def test_execute_connection_closed(self):
        """Test execute() handles REPL connection closed."""
        sandbox, _, _, mock_sock = self._make_sandbox()
        mock_sock.recv.return_value = b""

        result = sandbox.execute("print('hello')")

        assert result.success is False
        assert "connection closed" in result.stderr.lower()

    def test_read_file(self):
        """Test read_file() calls exec_run with cat."""
        sandbox, _, mock_container, _ = self._make_sandbox()
        mock_container.exec_run.return_value = (0, b"file contents")

        content = sandbox.read_file("/workspace/test.py")

        mock_container.exec_run.assert_called_with(["cat", "/workspace/test.py"])
        assert content == "file contents"

    def test_read_file_not_found(self):
        """Test read_file() raises FileNotFoundError."""
        sandbox, _, mock_container, _ = self._make_sandbox()
        mock_container.exec_run.return_value = (1, b"No such file")

        with pytest.raises(FileNotFoundError):
            sandbox.read_file("/workspace/nonexistent.py")

    def test_write_file(self):
        """Test write_file() calls put_archive."""
        sandbox, _, mock_container, _ = self._make_sandbox()

        sandbox.write_file("/workspace/test.py", "print('hello')")

        mock_container.put_archive.assert_called()

    def test_install_package(self):
        """Test install_package() calls pip install."""
        sandbox, _, mock_container, _ = self._make_sandbox()
        mock_container.exec_run.return_value = (
            0,
            b"Successfully installed numpy",
        )

        result = sandbox.install_package("numpy")

        mock_container.exec_run.assert_called_with(["pip", "install", "numpy"])
        assert result.success is True
        assert "numpy" in result.stdout

    def test_install_package_failure(self):
        """Test install_package() with failed installation."""
        sandbox, _, mock_container, _ = self._make_sandbox()
        mock_container.exec_run.return_value = (1, b"ERROR: No matching distribution")

        result = sandbox.install_package("nonexistent-pkg")

        assert result.success is False
        assert result.exit_code == 1

    def test_close(self):
        """Test close() stops and removes container."""
        sandbox, _, mock_container, mock_sock = self._make_sandbox()

        sandbox.close()

        mock_sock.close.assert_called_once()
        mock_container.stop.assert_called_once_with(timeout=5)
        mock_container.remove.assert_called_once_with(force=True)

    def test_context_manager(self):
        """Test DockerSandbox works as context manager."""
        sandbox, _, mock_container, mock_sock = self._make_sandbox()

        sandbox.__exit__(None, None, None)

        mock_sock.close.assert_called_once()
        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
