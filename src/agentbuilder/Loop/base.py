"""
Agentic loop for executing actions with planning and tool execution.
"""
from typing import Dict, Optional

from agentbuilder.Action.base import (
    CompleteAction,
    EmptyAction
)


class AgenticLoop:
    """Executes agentic loop with planning and tool execution"""
    
    def __init__(self, conversation_wrapper, 
                 planner,
                 tool_map: Dict[str, any],
                 verbose: bool = True,
                 max_iterations: int = 10):
        """
        Initialize the agentic loop.
        
        Args:
            conversation_wrapper: Conversation wrapper for history management
            planner: Agentic planner for deciding next actions
            tool_map: Mapping of tool names to Tool objects
            verbose: Whether to print execution details
            max_iterations: Maximum number of iterations
        """
        self.conversation = conversation_wrapper
        self.planner = planner
        self.tool_map = tool_map
        self.verbose = verbose
        self.max_iterations = max_iterations
    
    def reset(self):
        """Reset all agentic loop state including conversation history and planner state"""
        self.conversation.reset()
        self.planner.reset()
    
    def run(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Run the agentic loop.
        
        Args:
            message: User message to process
            system_prompt: Optional system prompt
        
        Returns:
            Final response string
        """
        # Add system prompt if provided
        if system_prompt and not self.conversation.conversation_history:
            self.conversation.add_system_message(system_prompt)
        
        # Add user message
        self.conversation.add_user_message(message)
        
        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1
            
            # Use planner to decide what to do next and get executable action
            action = self.planner.step(self.conversation.conversation_history, iterations)
            
            # Execute the action
            result = action.run()
            
            # If action returns a result (Complete or Empty), we're done
            if isinstance(action, (CompleteAction, EmptyAction)):
                return result if result is not None else ""
        
        if self.verbose:
            print(f"⚠️  Max iterations ({self.max_iterations}) reached\n")
        
        return "Max iterations reached"
