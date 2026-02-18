"""
Agentic planner for analyzing conversation state and creating executable actions.

The :class:`AgenticPlanner` examines the last message in the conversation
history and decides which action the loop should execute next -- without
making any LLM calls itself.
"""

from typing import Any, Dict, List

from agentbuilder.Action.base import (Action, AssistantMessageAction,
                                      CompleteAction, EmptyAction,
                                      ExecuteToolsAction, MakeLLMRequestAction,
                                      ToolMessageAction, UserMessageAction)


class AgenticPlanner:
    """Plans the next action based on conversation history without making LLM calls.

    The planner follows a simple decision tree:

    1. **Empty history** -> :class:`~agentbuilder.Action.base.EmptyAction`
    2. **Last message is AssistantMessage with tool_calls** ->
       :class:`~agentbuilder.Action.base.ExecuteToolsAction`
    3. **Last message is ToolMessage or UserMessage** ->
       :class:`~agentbuilder.Action.base.MakeLLMRequestAction`
    4. **Last message is AssistantMessage with text, no tool_calls** ->
       :class:`~agentbuilder.Action.base.CompleteAction`
    """

    def __init__(
        self, conversation_wrapper, tool_map: Dict[str, Any], verbose: bool = True
    ):
        """
        Initialize the agentic planner.

        Args:
            conversation_wrapper: A
                :class:`~agentbuilder.Client.base.BaseConversationWrapper`
                managing conversation history.
            tool_map: Mapping of tool names to
                :class:`~agentbuilder.Tools.base.Tool` objects.
            verbose: Whether to print planning details.
        """
        self.conversation_wrapper = conversation_wrapper
        self.tool_map = tool_map
        self.verbose = verbose

    def reset(self):
        """Reset planner state.

        Currently the planner is stateless, so this is a no-op. Provided
        for interface consistency with
        :meth:`~agentbuilder.Loop.base.AgenticLoop.reset`.
        """
        pass

    def step(self, conversation_history: List[Action], iterations: int = 0) -> Action:
        """
        Analyze conversation state and decide what action to take next.

        Decision tree:
            - Empty history -> ``EmptyAction``
            - Last = ``AssistantMessageAction`` with ``tool_calls`` -> ``ExecuteToolsAction``
            - Last = ``ToolMessageAction`` or ``UserMessageAction`` -> ``MakeLLMRequestAction``
            - Last = ``AssistantMessageAction`` with content, no tool_calls -> ``CompleteAction``

        Args:
            conversation_history: Current conversation history (list of
                :class:`~agentbuilder.Action.base.Action` objects).
            iterations: Current iteration count (passed through to
                terminal actions).

        Returns:
            An :class:`~agentbuilder.Action.base.Action` with a
            :meth:`run` method.

        Raises:
            NotImplementedError: If the conversation ends in an
                unrecognised action type.
        """
        if self.verbose:
            print(
                f"🤔 Analyzing conversation ({len(conversation_history)} messages)..."
            )

        if not conversation_history:
            return EmptyAction(iterations=iterations, verbose=self.verbose)

        last_action = conversation_history[-1]

        # If last message is AssistantMessage with tool calls, execute them
        if isinstance(last_action, AssistantMessageAction) and last_action.tool_calls:
            return ExecuteToolsAction(
                tool_calls=last_action.tool_calls,
                tool_map=self.tool_map,
                conversation_wrapper=self.conversation_wrapper,
                verbose=self.verbose,
            )

        # If last message is ToolMessage or UserMessage, make an LLM request
        elif isinstance(last_action, (ToolMessageAction, UserMessageAction)):
            return MakeLLMRequestAction(
                conversation_wrapper=self.conversation_wrapper,
                tool_map=self.tool_map,
                verbose=self.verbose,
            )

        # If last message is AssistantMessage with content and no tool calls, we're done
        elif (
            isinstance(last_action, AssistantMessageAction)
            and last_action.content
            and not last_action.tool_calls
        ):
            return CompleteAction(
                content=last_action.content, iterations=iterations, verbose=self.verbose
            )

        # Unhandled conversation state
        else:
            raise NotImplementedError(
                f"Unhandled conversation state: {type(last_action).__name__}"
            )
