"""
Agentic loop for executing actions with planning and tool execution.

The :class:`AgenticLoop` is the core orchestrator: it repeatedly asks the
:class:`~agentbuilder.Planner.base.AgenticPlanner` for the next action and
executes it until the agent produces a final response or the iteration limit
is reached.
"""

from typing import Dict, Optional

from agentbuilder.Action.base import CompleteAction, EmptyAction


class AgenticLoop:
    """Executes the plan-execute agentic loop.

    Example::

        from agentbuilder.utils import create_agent
        agent = create_agent(
            model_name="gpt-4o-mini",
            tools=[my_tool],
        )
        response = agent.run("Hello!")
        agent.reset()  # clear state for next conversation

    Note:
        If the agent reaches *max_iterations* without completing, the loop
        returns the string ``"Max iterations reached"``.  Increase
        ``max_iterations`` for complex multi-step tasks.
    """

    def __init__(
        self,
        conversation_wrapper,
        planner,
        tool_map: Dict[str, any],
        verbose: bool = True,
        max_iterations: int = 10,
    ):
        """
        Initialize the agentic loop.

        Args:
            conversation_wrapper: A
                :class:`~agentbuilder.Client.base.BaseConversationWrapper`
                managing conversation history.
            planner: An
                :class:`~agentbuilder.Planner.base.AgenticPlanner` that
                decides which action to take next.
            tool_map: Mapping of tool names to
                :class:`~agentbuilder.Tools.base.Tool` objects.
            verbose: Whether to print execution details to stdout.
            max_iterations: Maximum number of plan-execute cycles.
        """
        self.conversation = conversation_wrapper
        self.planner = planner
        self.tool_map = tool_map
        self.verbose = verbose
        self.max_iterations = max_iterations

    def reset(self):
        """Reset all agentic loop state including conversation history and planner state."""
        self.conversation.reset()
        self.planner.reset()

    def run(self, message: str) -> str:
        """
        Run the agentic loop with a user message.

        Appends the message to the conversation, then enters the
        plan-execute loop until completion or ``max_iterations`` is
        reached.

        Args:
            message: User message to process.

        Returns:
            The agent's final response string.

        Example::

            response = agent.run("What is 2 + 3?")
            print(response)
            agent.reset()  # reset before next conversation
        """
        # Add user message
        self.conversation.add_user_message(message)

        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1

            # Use planner to decide what to do next and get executable action
            action = self.planner.step(
                self.conversation.conversation_history, iterations
            )

            # Execute the action
            result = action.run()

            # If action returns a result (Complete or Empty), we're done
            if isinstance(action, (CompleteAction, EmptyAction)):
                return result if result is not None else ""

        if self.verbose:
            print(f"⚠️  Max iterations ({self.max_iterations}) reached\n")

        return "Max iterations reached"
