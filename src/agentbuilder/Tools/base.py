"""
Base classes for tool definition and execution.

Provides the :class:`Tool` base class that wraps a callable with an
OpenAI-compatible JSON schema, and :func:`tool_from_function` to derive
a Tool automatically from a Pydantic-annotated function.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel


@dataclass
class Response:
    """Standard response format for tool execution.

    Attributes:
        success: Whether the tool executed without errors.
        data: The return value of the tool function (may be any type).
        error: Error message if ``success`` is ``False``.

    Example::

        resp = Response(success=True, data={"result": 42})
        resp.to_dict()
        # {"success": True, "data": {"result": 42}}
    """

    success: bool
    data: Any
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format.

        Pydantic models in ``data`` are automatically serialized via
        ``model_dump()``.

        Returns:
            Dict with ``success``, ``data``, and optionally ``error`` keys.
        """
        data = self.data
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        result = {"success": self.success, "data": data}
        if self.error:
            result["error"] = self.error
        return result


class Tool:
    """Base tool class for function calling.

    Wraps a callable with metadata so it can be exposed to an LLM in
    OpenAI function-calling format.

    Attributes:
        name: Tool name (must match the function-call ``name``).
        description: Human-readable description shown to the LLM.
        parameters: JSON Schema dict describing the tool's parameters.
        function: The underlying callable to execute.

    Example::

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = Tool(
            name="greet",
            description="Greet a user by name.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            function=greet,
        )
        resp = tool.execute(name="Alice")
        # Response(success=True, data="Hello, Alice!")
    """

    def __init__(
        self, name: str, description: str, parameters: Dict, function: Callable
    ):
        """
        Initialize a tool.

        Args:
            name: Name of the tool (used in LLM function calling).
            description: Description of what the tool does.
            parameters: JSON Schema dict for the tool's parameters.
            function: The actual function to execute.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    def to_openai_format(self) -> Dict:
        """Convert tool to OpenAI function calling format.

        Returns:
            Dict with ``type`` and ``function`` keys matching the OpenAI
            tools specification.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> Response:
        """Execute the tool function with given arguments.

        Args:
            **kwargs: Keyword arguments matching the tool's parameter
                schema.

        Returns:
            A :class:`Response` with ``success=True`` and the return
            value, or ``success=False`` with an error message.
        """
        try:
            result = self.function(**kwargs)
            return Response(success=True, data=result)
        except Exception as e:
            return Response(success=False, data=None, error=str(e))


def tool_from_function(func: Callable) -> Tool:
    """
    Create a Tool from a function with a Pydantic parameter annotation.

    The function must accept exactly one parameter whose type annotation is
    a :class:`pydantic.BaseModel` subclass. The JSON schema is derived
    automatically from the model.

    Args:
        func: Function with a single Pydantic BaseModel parameter.

    Returns:
        A :class:`Tool` object ready for use in an agent.

    Raises:
        ValueError: If *func* does not have exactly one parameter, or if
            the parameter's annotation is not a Pydantic ``BaseModel``
            subclass.

    Example::

        from pydantic import BaseModel, Field

        class AddParams(BaseModel):
            a: int = Field(description="First number")
            b: int = Field(description="Second number")

        def add(params: AddParams) -> int:
            \"\"\"Add two numbers together.\"\"\"
            return params.a + params.b

        tool = tool_from_function(add)
        # tool.name == "add"
        # tool.execute(a=1, b=2) -> Response(success=True, data=3)
    """
    # Get function signature
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) != 1:
        raise ValueError(
            f"Function {func.__name__} must have exactly one parameter (Pydantic model)"
        )

    param = params[0]
    param_type = param.annotation

    if not (inspect.isclass(param_type) and issubclass(param_type, BaseModel)):
        raise ValueError(f"Parameter must be a Pydantic BaseModel, got {param_type}")

    # Get the Pydantic model schema
    schema = param_type.model_json_schema()

    # Create OpenAI-compatible parameters
    parameters = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }

    # Add descriptions from the schema
    if "description" in schema:
        parameters["description"] = schema["description"]

    # Build description with return type information
    description = func.__doc__ or f"Execute {func.__name__}"

    # Extract and append return type information
    if sig.return_annotation and sig.return_annotation != inspect.Signature.empty:
        return_type = sig.return_annotation
        if inspect.isclass(return_type) and issubclass(return_type, BaseModel):
            return_schema = return_type.model_json_schema()
            description += f"\n\nReturns: {return_schema}"
        else:
            description += f"\n\nReturns: {return_type}"

    # Wrapper function that creates Pydantic instance
    def wrapper(**kwargs):
        params_instance = param_type(**kwargs)
        return func(params_instance)

    # Create the tool
    return Tool(
        name=func.__name__,
        description=description,
        parameters=parameters,
        function=wrapper,
    )
