"""Tests for code execution tools."""

from unittest.mock import MagicMock

import pytest

from agentbuilder.Sandbox.base import ExecutionResult, Sandbox
from agentbuilder.Tools.base import Response, Tool
from agentbuilder.Tools.code_execution import CodeExecutionTool, create_sandbox_tools


class FakeSandbox(Sandbox):
    """Minimal concrete sandbox for testing."""

    def __init__(self):
        self.executed = []
        self.files = {}
        self.packages = []

    def execute(self, code, timeout=30):
        self.executed.append(code)
        return ExecutionResult(stdout="output", stderr="", success=True, exit_code=0)

    def read_file(self, path):
        if path in self.files:
            return self.files[path]
        raise FileNotFoundError(path)

    def write_file(self, path, content):
        self.files[path] = content

    def install_package(self, package):
        self.packages.append(package)
        return ExecutionResult(
            stdout=f"Installed {package}", stderr="", success=True, exit_code=0
        )

    def close(self):
        pass


class TestCodeExecutionTool:
    """Tests for CodeExecutionTool."""

    def test_creation(self):
        """Test CodeExecutionTool creation."""
        sandbox = FakeSandbox()
        tool = CodeExecutionTool(sandbox)

        assert tool.name == "execute_code"
        assert "Python code" in tool.description
        assert "code" in tool.parameters["properties"]

    def test_to_openai_format(self):
        """Test OpenAI format output."""
        sandbox = FakeSandbox()
        tool = CodeExecutionTool(sandbox)
        fmt = tool.to_openai_format()

        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "execute_code"
        assert "code" in fmt["function"]["parameters"]["properties"]

    def test_execute_success(self):
        """Test successful code execution."""
        sandbox = FakeSandbox()
        tool = CodeExecutionTool(sandbox)

        response = tool.execute(code="print('hello')")

        assert isinstance(response, Response)
        assert response.success is True
        assert response.data["stdout"] == "output"
        assert response.data["success"] is True
        assert sandbox.executed == ["print('hello')"]

    def test_execute_failure(self):
        """Test code execution failure."""
        sandbox = MagicMock(spec=Sandbox)
        sandbox.execute.return_value = ExecutionResult(
            stdout="", stderr="NameError: name 'x' is not defined",
            success=False, exit_code=1,
        )
        tool = CodeExecutionTool(sandbox)

        response = tool.execute(code="print(x)")

        assert response.success is True  # Tool execution succeeded
        assert response.data["success"] is False  # Code execution failed
        assert "NameError" in response.data["stderr"]

    def test_sandbox_exception_propagated(self):
        """Test that sandbox exceptions are caught by Tool.execute()."""
        sandbox = MagicMock(spec=Sandbox)
        sandbox.execute.side_effect = RuntimeError("Docker crashed")
        tool = CodeExecutionTool(sandbox)

        response = tool.execute(code="print('hello')")

        assert response.success is False
        assert "Docker crashed" in response.error


class TestCreateSandboxTools:
    """Tests for create_sandbox_tools()."""

    def test_returns_three_tools(self):
        """Test create_sandbox_tools() returns 3 tools."""
        sandbox = FakeSandbox()
        tools = create_sandbox_tools(sandbox)

        assert len(tools) == 3
        assert all(isinstance(t, Tool) for t in tools)

    def test_tool_names(self):
        """Test the created tools have correct names."""
        sandbox = FakeSandbox()
        tools = create_sandbox_tools(sandbox)
        names = {t.name for t in tools}

        assert "read_file" in names
        assert "write_file" in names
        assert "install_package" in names

    def test_read_file_tool(self):
        """Test read_file tool works."""
        sandbox = FakeSandbox()
        sandbox.files["/workspace/test.py"] = "print('hello')"
        tools = create_sandbox_tools(sandbox)
        read_tool = next(t for t in tools if t.name == "read_file")

        response = read_tool.execute(path="/workspace/test.py")

        assert response.success is True
        assert response.data == "print('hello')"

    def test_write_file_tool(self):
        """Test write_file tool works."""
        sandbox = FakeSandbox()
        tools = create_sandbox_tools(sandbox)
        write_tool = next(t for t in tools if t.name == "write_file")

        response = write_tool.execute(
            path="/workspace/test.py", content="print('hello')"
        )

        assert response.success is True
        assert sandbox.files["/workspace/test.py"] == "print('hello')"

    def test_install_package_tool(self):
        """Test install_package tool works."""
        sandbox = FakeSandbox()
        tools = create_sandbox_tools(sandbox)
        install_tool = next(t for t in tools if t.name == "install_package")

        response = install_tool.execute(package="numpy")

        assert response.success is True
        assert "numpy" in sandbox.packages


class TestCreateCodeAgent:
    """Tests for create_code_agent() factory."""

    def test_create_code_agent(self, mocker):
        """Test create_code_agent() creates an agent with code tools."""
        mock_agent = MagicMock()
        mock_create_agent = mocker.patch(
            "agentbuilder.utils.create_agent", return_value=mock_agent
        )

        from agentbuilder.utils import create_code_agent

        sandbox = FakeSandbox()
        agent = create_code_agent(model_name="gpt-4", sandbox=sandbox)

        assert agent is mock_agent
        mock_create_agent.assert_called_once()

        call_kwargs = mock_create_agent.call_args
        tools = call_kwargs[1]["tools"] if "tools" in call_kwargs[1] else call_kwargs[0][1]
        # Should have execute_code + read_file + write_file + install_package = 4 tools
        assert len(tools) == 4

    def test_create_code_agent_with_additional_tools(self, mocker):
        """Test create_code_agent() includes additional tools."""
        mock_agent = MagicMock()
        mock_create = mocker.patch(
            "agentbuilder.utils.create_agent", return_value=mock_agent
        )

        from agentbuilder.utils import create_code_agent

        extra_tool = MagicMock()
        extra_tool.name = "extra"

        sandbox = FakeSandbox()
        create_code_agent(
            model_name="gpt-4",
            sandbox=sandbox,
            additional_tools=[extra_tool],
        )

        call_kwargs = mock_create.call_args[1]
        tools = call_kwargs["tools"]
        # 4 sandbox tools + 1 extra = 5
        assert len(tools) == 5

    def test_create_code_agent_custom_prompt(self, mocker):
        """Test create_code_agent() passes custom system_prompt."""
        mock_agent = MagicMock()
        mock_create = mocker.patch(
            "agentbuilder.utils.create_agent", return_value=mock_agent
        )

        from agentbuilder.utils import create_code_agent

        sandbox = FakeSandbox()
        create_code_agent(
            model_name="gpt-4",
            sandbox=sandbox,
            system_prompt="Custom prompt",
        )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["system_prompt"] == "Custom prompt"
