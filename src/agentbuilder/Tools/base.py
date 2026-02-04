"""
Base classes for tool definition and execution.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel


@dataclass
class Response:
    """Standard response format for tool execution"""

    success: bool
    data: Any
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        result = {"success": self.success, "data": self.data}
        if self.error:
            result["error"] = self.error
        return result


class Tool:
    """Base tool class for function calling"""

    def __init__(
        self, name: str, description: str, parameters: Dict, function: Callable
    ):
        """
        Initialize a tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            parameters: JSON schema for parameters
            function: The actual function to execute
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    def to_openai_format(self) -> Dict:
        """Convert tool to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> Response:
        """Execute the tool function with given arguments"""
        try:
            result = self.function(**kwargs)
            return Response(success=True, data=result)
        except Exception as e:
            return Response(success=False, data=None, error=str(e))


def tool_from_function(func: Callable) -> Tool:
    """
    Create a Tool from a function with Pydantic parameter annotation.

    Args:
        func: Function with a single Pydantic BaseModel parameter

    Returns:
        Tool object
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

    # Wrapper function that creates Pydantic instance
    def wrapper(**kwargs):
        params_instance = param_type(**kwargs)
        return func(params_instance)

    # Create the tool
    return Tool(
        name=func.__name__,
        description=func.__doc__ or f"Execute {func.__name__}",
        parameters=parameters,
        function=wrapper,
    )
