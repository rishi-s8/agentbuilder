"""Tests for Tool classes."""

import pytest
from pydantic import BaseModel, Field

from agentbuilder.Tools.base import Response, Tool, tool_from_function


class TestResponse:
    """Tests for Response class."""

    def test_successful_response(self):
        """Test creating a successful response."""
        response = Response(success=True, data={"result": 42})

        assert response.success is True
        assert response.data == {"result": 42}
        assert response.error is None

    def test_failed_response(self):
        """Test creating a failed response."""
        response = Response(success=False, data=None, error="Something went wrong")

        assert response.success is False
        assert response.data is None
        assert response.error == "Something went wrong"

    def test_to_dict_success(self):
        """Test converting successful response to dict."""
        response = Response(success=True, data=100)
        result = response.to_dict()

        assert result["success"] is True
        assert result["data"] == 100
        assert "error" not in result

    def test_to_dict_with_error(self):
        """Test converting failed response to dict."""
        response = Response(success=False, data=None, error="Error message")
        result = response.to_dict()

        assert result["success"] is False
        assert result["data"] is None
        assert result["error"] == "Error message"


class TestTool:
    """Tests for Tool class."""

    def test_tool_initialization(self):
        """Test Tool initialization."""

        def test_func(x: int) -> int:
            return x * 2

        tool = Tool(
            name="double",
            description="Double a number",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            function=test_func,
        )

        assert tool.name == "double"
        assert tool.description == "Double a number"
        assert tool.parameters["type"] == "object"
        assert callable(tool.function)

    def test_to_openai_format(self):
        """Test converting tool to OpenAI format."""

        def test_func(x: int) -> int:
            return x * 2

        tool = Tool(
            name="multiply",
            description="Multiply by 2",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            function=test_func,
        )

        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "multiply"
        assert openai_format["function"]["description"] == "Multiply by 2"
        assert openai_format["function"]["parameters"]["type"] == "object"

    def test_execute_success(self):
        """Test successful tool execution."""

        def add_numbers(x: int, y: int) -> int:
            return x + y

        tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
            },
            function=add_numbers,
        )

        response = tool.execute(x=5, y=3)

        assert response.success is True
        assert response.data == 8
        assert response.error is None

    def test_execute_failure(self):
        """Test tool execution with exception."""

        def failing_func(x: int) -> int:
            raise ValueError("Test error")

        tool = Tool(
            name="fail",
            description="A failing function",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            function=failing_func,
        )

        response = tool.execute(x=1)

        assert response.success is False
        assert response.data is None
        assert "Test error" in response.error

    def test_execute_with_missing_arguments(self):
        """Test tool execution with missing arguments."""

        def func_with_args(x: int, y: int) -> int:
            return x + y

        tool = Tool(
            name="test", description="Test", parameters={}, function=func_with_args
        )

        response = tool.execute(x=5)  # Missing y

        assert response.success is False
        assert response.error is not None


class TestToolFromFunction:
    """Tests for tool_from_function decorator."""

    def test_simple_pydantic_model(self):
        """Test creating tool from function with simple Pydantic model."""

        class AddParams(BaseModel):
            x: int
            y: int

        def add(params: AddParams) -> int:
            """Add two numbers together."""
            return params.x + params.y

        tool = tool_from_function(add)

        assert tool.name == "add"
        assert "Add two numbers" in tool.description
        assert tool.parameters["type"] == "object"
        assert "x" in tool.parameters["properties"]
        assert "y" in tool.parameters["properties"]
        assert "x" in tool.parameters["required"]
        assert "y" in tool.parameters["required"]

    def test_pydantic_model_with_descriptions(self):
        """Test tool creation with field descriptions."""

        class ComplexParams(BaseModel):
            """Parameters for complex operation."""

            name: str = Field(description="The name to use")
            count: int = Field(description="How many times", default=1)

        def complex_func(params: ComplexParams) -> str:
            """Perform a complex operation."""
            return params.name * params.count

        tool = tool_from_function(complex_func)

        assert tool.name == "complex_func"
        assert "complex operation" in tool.description
        assert "name" in tool.parameters["properties"]
        assert "count" in tool.parameters["properties"]
        assert "name" in tool.parameters["required"]
        assert "count" not in tool.parameters["required"]  # Has default

    def test_tool_execution_from_function(self):
        """Test that tools created from functions can be executed."""

        class MultiplyParams(BaseModel):
            x: int
            y: int

        def multiply(params: MultiplyParams) -> int:
            """Multiply two numbers."""
            return params.x * params.y

        tool = tool_from_function(multiply)
        response = tool.execute(x=6, y=7)

        assert response.success is True
        assert response.data == 42

    def test_invalid_function_no_params(self):
        """Test that function without parameters raises error."""

        def no_params():
            return "test"

        with pytest.raises(ValueError, match="exactly one parameter"):
            tool_from_function(no_params)

    def test_invalid_function_multiple_params(self):
        """Test that function with multiple parameters raises error."""

        def multiple_params(x: int, y: int):
            return x + y

        with pytest.raises(ValueError, match="exactly one parameter"):
            tool_from_function(multiple_params)

    def test_invalid_function_non_pydantic(self):
        """Test that function with non-Pydantic parameter raises error."""

        def non_pydantic(x: int):
            return x

        with pytest.raises(ValueError, match="Pydantic BaseModel"):
            tool_from_function(non_pydantic)

    def test_optional_fields(self):
        """Test handling of optional fields."""

        class OptionalParams(BaseModel):
            required_field: str
            optional_field: int = 10

        def func_with_optional(params: OptionalParams) -> str:
            """Function with optional parameters."""
            return f"{params.required_field}: {params.optional_field}"

        tool = tool_from_function(func_with_optional)

        # Test with only required field
        response = tool.execute(required_field="test")
        assert response.success is True
        assert response.data == "test: 10"

        # Test with both fields
        response = tool.execute(required_field="test", optional_field=20)
        assert response.success is True
        assert response.data == "test: 20"

    def test_complex_types(self):
        """Test handling of complex types in Pydantic models."""
        from typing import Dict, List

        class ComplexTypes(BaseModel):
            names: List[str]
            scores: Dict[str, int]

        def process_data(params: ComplexTypes) -> int:
            """Process complex data."""
            return len(params.names) + len(params.scores)

        tool = tool_from_function(process_data)

        response = tool.execute(
            names=["Alice", "Bob"], scores={"Alice": 100, "Bob": 90, "Charlie": 85}
        )

        assert response.success is True
        assert response.data == 5  # 2 names + 3 scores

    def test_to_openai_format_from_function(self):
        """Test that tools from functions have correct OpenAI format."""

        class TestParams(BaseModel):
            value: int

        def test_func(params: TestParams) -> int:
            """A test function."""
            return params.value

        tool = tool_from_function(test_func)
        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "test_func"
        assert "test function" in openai_format["function"]["description"]
        assert "value" in openai_format["function"]["parameters"]["properties"]
