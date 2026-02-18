"""
Local sub-agent delegation tool.

Provides :class:`AgentTool`, which wraps an
:class:`~agentbuilder.Loop.base.AgenticLoop` so a parent agent can delegate
tasks to a child agent running in the same process.
"""

from agentbuilder.Loop.base import AgenticLoop
from agentbuilder.Tools.base import Tool


class AgentTool(Tool):
    """Tool that delegates tasks to a local sub-agent running in-process.

    When the parent agent invokes this tool, the sub-agent is **reset**
    (conversation cleared) and then run with the provided task message.
    This ensures each delegation starts with a clean context.

    Note:
        The sub-agent is reset before every delegation. If you need
        persistent conversation state across delegations, use
        :class:`~agentbuilder.Tools.remote_agent_tool.RemoteAgentTool`
        and manage session resets manually.

    Example::

        from agentbuilder.utils import create_agent, create_agent_tool

        math_agent = create_agent_tool(
            name="math_expert",
            description="Solves arithmetic problems",
            model_name="gpt-4o-mini",
            tools=[add_tool, multiply_tool],
        )
        parent = create_agent(
            model_name="gpt-4o-mini",
            tools=[math_agent],
        )
        result = parent.run("What is 6 * 7?")
    """

    def __init__(self, agent: AgenticLoop, name: str, description: str):
        """
        Initialize an AgentTool.

        Args:
            agent: The :class:`~agentbuilder.Loop.base.AgenticLoop`
                instance to delegate tasks to.
            name: Name of this tool (exposed to the LLM).
            description: Description of what this sub-agent does.
        """
        self.agent = agent

        parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to delegate to the sub-agent",
                }
            },
            "required": ["task"],
        }

        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            function=self._delegate,
        )

    def _delegate(self, task: str) -> str:
        """
        Reset the sub-agent and run it with the given task.

        Args:
            task: The task message to send to the sub-agent.

        Returns:
            The sub-agent's final response string.
        """
        self.agent.reset()
        return self.agent.run(task)
