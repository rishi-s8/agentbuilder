"""
Action data types for the agentic framework.
"""

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Action:
    """Base class for all actions"""

    def run(self):
        """Execute the action"""
        raise NotImplementedError("Subclasses must implement run()")


# Control Flow Action Types (for planner decisions)
@dataclass
class ExecuteToolsAction(Action):
    """Action to execute tool calls"""

    tool_calls: List[Any] = field(default_factory=list)
    tool_map: dict = field(default_factory=dict)
    conversation_wrapper: Any = None
    verbose: bool = False

    def run(self):
        """Execute all tool calls and add results to conversation"""
        if self.verbose:
            print(f"🔧 Executing {len(self.tool_calls)} tool call(s)...")

        for tool_call in self.tool_calls:
            # Handle both dict and object formats
            if isinstance(tool_call, dict):
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]
            else:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id

            if self.verbose:
                print(f"   → {function_name}({function_args})")

            # Execute the tool
            if function_name in self.tool_map:
                tool_response = self.tool_map[function_name].execute(**function_args)
                result_content = json.dumps(tool_response.to_dict())

                if self.verbose:
                    status = "✓" if tool_response.success else "✗"
                    print(f"     {status} Result: {result_content[:100]}...")
            else:
                result_content = json.dumps(
                    {"error": f"Tool {function_name} not found"}
                )
                if self.verbose:
                    print(f"     ✗ Tool not found")

            # Add tool response to conversation
            self.conversation_wrapper.add_tool_message(
                tool_call_id, function_name, result_content
            )


@dataclass
class MakeLLMRequestAction(Action):
    """Action to make an LLM request"""

    conversation_wrapper: Any = None
    tool_map: dict = field(default_factory=dict)
    verbose: bool = False

    def run(self):
        """Make an LLM request and add response to conversation"""
        if self.verbose:
            print(f"🤖 Making LLM request...")

        # Make LLM request with tools
        tools = (
            [tool.to_openai_format() for tool in self.tool_map.values()]
            if self.tool_map
            else None
        )

        response = self.conversation_wrapper.client.chat.completions.create(
            model=self.conversation_wrapper.model,
            messages=self.conversation_wrapper.to_messages(),
            tools=tools,
            tool_choice="auto" if tools else None,
        )

        assistant_message = response.choices[0].message

        # Add response to history
        self.conversation_wrapper.add_assistant_message(
            assistant_message.content,
            tool_calls=(
                [tc.model_dump() for tc in assistant_message.tool_calls]
                if assistant_message.tool_calls
                else None
            ),
        )

        if self.verbose:
            if assistant_message.tool_calls:
                print(f"   ✓ Response: Tool calls requested")
            else:
                print(
                    f"   ✓ Response: {assistant_message.content[:100] if assistant_message.content else 'No content'}..."
                )


@dataclass
class CompleteAction(Action):
    """Action indicating completion"""

    content: str = ""
    iterations: int = 0
    verbose: bool = False

    def run(self):
        """Return the final content"""
        if self.verbose:
            print(f"✅ Completed in {self.iterations} iteration(s)\n")
        return self.content


@dataclass
class EmptyAction(Action):
    """Action indicating empty conversation"""

    iterations: int = 0
    verbose: bool = False

    def run(self):
        """Return empty string for empty conversation"""
        if self.verbose:
            print(
                f"✅ Completed (empty conversation) in {self.iterations} iteration(s)\n"
            )
        return ""


# Message Action Types (for conversation history)
@dataclass
class SystemMessageAction(Action):
    """System message in conversation"""

    content: str = ""

    def to_message(self):
        """Convert to OpenAI message format"""
        return {"role": "system", "content": self.content}

    @classmethod
    def from_message(cls, msg: dict):
        """Create SystemMessageAction from OpenAI message dict"""
        return cls(content=msg["content"])


@dataclass
class UserMessageAction(Action):
    """User message in conversation"""

    content: str = ""

    def to_message(self):
        """Convert to OpenAI message format"""
        return {"role": "user", "content": self.content}

    @classmethod
    def from_message(cls, msg: dict):
        """Create UserMessageAction from OpenAI message dict"""
        return cls(content=msg["content"])


@dataclass
class AssistantMessageAction(Action):
    """Assistant message in conversation"""

    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None

    def to_message(self):
        """Convert to OpenAI message format"""
        msg = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    @classmethod
    def from_message(cls, msg: dict):
        """Create AssistantMessageAction from OpenAI message dict"""
        return cls(content=msg.get("content"), tool_calls=msg.get("tool_calls"))


@dataclass
class ToolMessageAction(Action):
    """Tool response message in conversation"""

    tool_call_id: str = ""
    name: str = ""
    content: str = ""

    def to_message(self):
        """Convert to OpenAI message format"""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
        }

    @classmethod
    def from_message(cls, msg: dict):
        """Create ToolMessageAction from OpenAI message dict"""
        return cls(
            tool_call_id=msg["tool_call_id"], name=msg["name"], content=msg["content"]
        )
