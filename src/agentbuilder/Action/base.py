"""
Action data types for the agentic framework.

Actions are the atomic units of work in the plan-execute loop. They fall into
two categories:

**Control-flow actions** -- returned by the planner to drive the loop:
:class:`ExecuteToolsAction`, :class:`MakeLLMRequestAction`,
:class:`CompleteAction`, :class:`EmptyAction`.

**Message actions** -- stored in conversation history to represent the
chat transcript:
:class:`SystemMessageAction`, :class:`UserMessageAction`,
:class:`AssistantMessageAction`, :class:`ToolMessageAction`.

Every action exposes a :meth:`run` method so the agentic loop can execute it
polymorphically.
"""

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Action:
    """Base class for all actions in the agentic framework.

    Subclasses must override :meth:`run` to provide their execution logic.
    """

    def run(self):
        """Execute the action.

        Raises:
            NotImplementedError: Always -- subclasses must provide an
                implementation.
        """
        raise NotImplementedError("Subclasses must implement run()")


# Control Flow Action Types (for planner decisions)
@dataclass
class ExecuteToolsAction(Action):
    """Execute pending tool calls from the last assistant message.

    The planner returns this action when the most recent assistant message
    contains ``tool_calls``.  :meth:`run` iterates over each call, looks
    up the tool in ``tool_map``, executes it, and appends a
    :class:`ToolMessageAction` to the conversation.

    Attributes:
        tool_calls: List of tool-call dicts or objects from the LLM
            response.
        tool_map: Mapping of tool names to
            :class:`~agentbuilder.Tools.base.Tool` instances.
        conversation_wrapper: The active
            :class:`~agentbuilder.Client.base.BaseConversationWrapper`.
        verbose: Whether to print execution details.
    """

    tool_calls: List[Any] = field(default_factory=list)
    tool_map: dict = field(default_factory=dict)
    conversation_wrapper: Any = None
    verbose: bool = False

    def run(self):
        """Execute all tool calls and add results to conversation."""
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
    """Send the current conversation to the LLM and store its response.

    The planner returns this action when the conversation needs an LLM
    completion -- typically after a user message or after tool results have
    been appended.

    Attributes:
        conversation_wrapper: The active
            :class:`~agentbuilder.Client.base.BaseConversationWrapper`.
        tool_map: Mapping of tool names to
            :class:`~agentbuilder.Tools.base.Tool` instances (used to
            build the ``tools`` parameter for the API call).
        verbose: Whether to print execution details.
    """

    conversation_wrapper: Any = None
    tool_map: dict = field(default_factory=dict)
    verbose: bool = False

    def run(self):
        """Make an LLM request and add response to conversation."""
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
    """Signals that the agent has finished and holds the final text.

    Attributes:
        content: The agent's final response text.
        iterations: Number of loop iterations that were executed.
        verbose: Whether to print a completion message.
    """

    content: str = ""
    iterations: int = 0
    verbose: bool = False

    def run(self):
        """Return the final content."""
        if self.verbose:
            print(f"✅ Completed in {self.iterations} iteration(s)\n")
        return self.content


@dataclass
class EmptyAction(Action):
    """Signals that the conversation is empty -- nothing to do.

    Attributes:
        iterations: Number of loop iterations at the time of detection.
        verbose: Whether to print a completion message.
    """

    iterations: int = 0
    verbose: bool = False

    def run(self):
        """Return empty string for empty conversation."""
        if self.verbose:
            print(
                f"✅ Completed (empty conversation) in {self.iterations} iteration(s)\n"
            )
        return ""


# Message Action Types (for conversation history)
@dataclass
class SystemMessageAction(Action):
    """System message in conversation history.

    Attributes:
        content: The system prompt text.
    """

    content: str = ""

    def to_message(self):
        """Convert to OpenAI message format.

        Returns:
            dict: ``{"role": "system", "content": ...}``
        """
        return {"role": "system", "content": self.content}

    @classmethod
    def from_message(cls, msg: dict):
        """Create SystemMessageAction from OpenAI message dict.

        Args:
            msg: A dict with at least a ``"content"`` key.
        """
        return cls(content=msg["content"])


@dataclass
class UserMessageAction(Action):
    """User message in conversation history.

    Attributes:
        content: The user's message text.
    """

    content: str = ""

    def to_message(self):
        """Convert to OpenAI message format.

        Returns:
            dict: ``{"role": "user", "content": ...}``
        """
        return {"role": "user", "content": self.content}

    @classmethod
    def from_message(cls, msg: dict):
        """Create UserMessageAction from OpenAI message dict.

        Args:
            msg: A dict with at least a ``"content"`` key.
        """
        return cls(content=msg["content"])


@dataclass
class AssistantMessageAction(Action):
    """Assistant message in conversation history.

    May contain text content, tool calls, or both.

    Attributes:
        content: The assistant's text response (may be ``None`` when only
            tool calls are present).
        tool_calls: List of tool-call dicts requested by the assistant.
    """

    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None

    def to_message(self):
        """Convert to OpenAI message format.

        Returns:
            dict: ``{"role": "assistant", "content": ..., "tool_calls": ...}``
        """
        msg = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    @classmethod
    def from_message(cls, msg: dict):
        """Create AssistantMessageAction from OpenAI message dict.

        Args:
            msg: A dict with ``"content"`` and optionally ``"tool_calls"``
                keys.
        """
        return cls(content=msg.get("content"), tool_calls=msg.get("tool_calls"))


@dataclass
class ToolMessageAction(Action):
    """Tool response message in conversation history.

    Attributes:
        tool_call_id: The unique ID of the tool call this responds to.
        name: Name of the tool that was executed.
        content: JSON-serialized result from the tool.
    """

    tool_call_id: str = ""
    name: str = ""
    content: str = ""

    def to_message(self):
        """Convert to OpenAI message format.

        Returns:
            dict: ``{"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}``
        """
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
        }

    @classmethod
    def from_message(cls, msg: dict):
        """Create ToolMessageAction from OpenAI message dict.

        Args:
            msg: A dict with ``"tool_call_id"``, ``"name"``, and
                ``"content"`` keys.
        """
        return cls(
            tool_call_id=msg["tool_call_id"], name=msg["name"], content=msg["content"]
        )
