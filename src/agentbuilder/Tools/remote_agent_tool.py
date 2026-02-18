"""
Remote sub-agent delegation tool via HTTP.
"""

import requests

from agentbuilder.Tools.base import Tool


class RemoteAgentTool(Tool):
    """Tool that delegates tasks to a remote sub-agent running as a FastAPI server."""

    def __init__(self, base_url: str):
        """
        Initialize a RemoteAgentTool.

        Fetches agent info from GET {base_url}/info to auto-discover
        the agent's name, description, and capabilities.

        Args:
            base_url: Base URL of the remote agent server (e.g., "http://localhost:8100")
        """
        self.base_url = base_url.rstrip("/")

        # Auto-discover agent info
        info_response = requests.get(f"{self.base_url}/info")
        info_response.raise_for_status()
        info = info_response.json()

        name = info["name"]
        description = info["description"]

        parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to delegate to the remote sub-agent",
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
        Send a task to the remote sub-agent via HTTP POST.

        Args:
            task: The task message to send to the remote sub-agent

        Returns:
            The remote sub-agent's response string
        """
        response = requests.post(
            f"{self.base_url}/run",
            json={"message": task},
        )
        response.raise_for_status()
        return response.json()["response"]
