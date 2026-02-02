"""
Agentic planner for analyzing conversation state and creating executable actions.
"""
from typing import List, Dict, Any

from agentbuilder.Action.base import (
    Action,
    UserMessageAction,
    AssistantMessageAction,
    ToolMessageAction,
    ExecuteToolsAction,
    MakeLLMRequestAction,
    CompleteAction,
    EmptyAction
)


class AgenticPlanner:
    """Plans the next action based on conversation history without making LLM calls"""
    
    def __init__(self, conversation_wrapper, tool_map: Dict[str, Any], verbose: bool = True):
        """
        Initialize the agentic planner.
        
        Args:
            conversation_wrapper: Conversation wrapper for history management
            tool_map: Mapping of tool names to Tool objects
            verbose: Whether to print planning details
        """
        self.conversation_wrapper = conversation_wrapper
        self.tool_map = tool_map
        self.verbose = verbose
    
    def step(self, conversation_history: List[Action], iterations: int = 0) -> Action:
        """
        Analyze conversation state and decide what action to take next.
        
        Args:
            conversation_history: Current conversation history (list of Action objects)
            iterations: Current iteration count
        
        Returns:
            Action object with run() method: ExecuteToolsAction, MakeLLMRequestAction, CompleteAction, or EmptyAction
        """
        if self.verbose:
            print(f"🤔 Analyzing conversation ({len(conversation_history)} messages)...")
        
        if not conversation_history:
            return EmptyAction(iterations=iterations, verbose=self.verbose)
        
        last_action = conversation_history[-1]
        
        # If last message is AssistantMessage with tool calls, execute them
        if isinstance(last_action, AssistantMessageAction) and last_action.tool_calls:
            return ExecuteToolsAction(
                tool_calls=last_action.tool_calls,
                tool_map=self.tool_map,
                conversation_wrapper=self.conversation_wrapper,
                verbose=self.verbose
            )
        
        # If last message is ToolMessage or UserMessage, make an LLM request
        elif isinstance(last_action, (ToolMessageAction, UserMessageAction)):
            return MakeLLMRequestAction(
                conversation_wrapper=self.conversation_wrapper,
                tool_map=self.tool_map,
                verbose=self.verbose
            )
        
        # If last message is AssistantMessage with content and no tool calls, we're done
        elif isinstance(last_action, AssistantMessageAction) and last_action.content and not last_action.tool_calls:
            return CompleteAction(
                content=last_action.content,
                iterations=iterations,
                verbose=self.verbose
            )
        
        # Unhandled conversation state
        else:
            raise NotImplementedError(f"Unhandled conversation state: {type(last_action).__name__}")
