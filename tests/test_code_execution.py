"""Tests for code execution tools."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agentbuilder.Sandbox.base import ExecutionResult, Sandbox
from agentbuilder.Tools.base import Response, Tool
from agentbuilder.Tools.code_execution import (CodeExecutionTool,
                                               create_sandbox_tools)


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
            stdout="",
            stderr="NameError: name 'x' is not defined",
            success=False,
            exit_code=1,
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
        """Test create_code_agent() creates an agent with only execute_code tool."""
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
        tools = (
            call_kwargs[1]["tools"] if "tools" in call_kwargs[1] else call_kwargs[0][1]
        )
        # Should have only execute_code (1 tool)
        assert len(tools) == 1
        assert tools[0].name == "execute_code"

    def test_create_code_agent_with_tools(self, mocker):
        """Test create_code_agent() injects tools into sandbox and updates prompt."""
        mock_agent = MagicMock()
        mock_create = mocker.patch(
            "agentbuilder.utils.create_agent", return_value=mock_agent
        )

        from agentbuilder.utils import create_code_agent

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        add_tool = Tool(
            name="add",
            description="Add two numbers together.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
            function=add,
        )

        sandbox = FakeSandbox()
        create_code_agent(
            model_name="gpt-4",
            sandbox=sandbox,
            tools=[add_tool],
        )

        # Tool source should have been injected into sandbox
        assert len(sandbox.executed) > 0

        # Still only execute_code as function-calling tool
        call_kwargs = mock_create.call_args[1]
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0].name == "execute_code"

        # System prompt should mention the injected tool
        system_prompt = call_kwargs["system_prompt"]
        assert "add" in system_prompt

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


class TestInjectToolsIntoSandbox:
    """Tests for _inject_tools_into_sandbox."""

    def test_inject_pydantic_tool(self):
        """Test injecting a tool with _source_func and _param_type (from tool_from_function)."""
        from agentbuilder.utils import _inject_tools_into_sandbox

        class MultiplyParams(BaseModel):
            x: int
            y: int

        def multiply(params: MultiplyParams) -> int:
            """Multiply two numbers."""
            return params.x * params.y

        # Simulate what tool_from_function does: set _source_func and _param_type
        tool = Tool(
            name="multiply",
            description="Multiply two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
            function=lambda **kwargs: multiply(MultiplyParams(**kwargs)),
        )
        tool._source_func = multiply
        tool._param_type = MultiplyParams

        sandbox = FakeSandbox()
        _inject_tools_into_sandbox(sandbox, [tool])

        # Should inject 3 pieces: model source, function source, wrapper
        assert len(sandbox.executed) == 3

        # Model class should be in first injection
        assert "MultiplyParams" in sandbox.executed[0]

        # Original function should be in second injection
        assert "def multiply" in sandbox.executed[1]

        # Wrapper should be in third injection
        assert "def multiply(**kwargs)" in sandbox.executed[2]
        assert "MultiplyParams(**kwargs)" in sandbox.executed[2]

    def test_inject_direct_tool(self):
        """Test injecting a directly constructed Tool."""
        from agentbuilder.utils import _inject_tools_into_sandbox

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = Tool(
            name="greet",
            description="Greet someone",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            function=greet,
        )

        sandbox = FakeSandbox()
        _inject_tools_into_sandbox(sandbox, [tool])

        # Should inject function source directly
        assert len(sandbox.executed) == 1
        assert "def greet" in sandbox.executed[0]


class TestBuildToolsPrompt:
    """Tests for _build_tools_prompt."""

    def test_build_tools_prompt(self):
        """Test that _build_tools_prompt produces correct descriptions."""
        from agentbuilder.utils import _build_tools_prompt

        tool = Tool(
            name="add",
            description="Add two numbers together.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
            function=lambda a, b: a + b,
        )
        result = _build_tools_prompt([tool])

        assert "add" in result
        assert "a: integer" in result
        assert "b: integer" in result
        assert "Add two numbers together." in result
