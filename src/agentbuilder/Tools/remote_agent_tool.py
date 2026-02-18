"""
Remote sub-agent delegation tool via HTTP with session management.
"""

import requests

from agentbuilder.Tools.base import Tool


class RemoteAgentTool(Tool):
    """Tool that delegates tasks to a remote sub-agent running as a FastAPI server."""

    def __init__(self, base_url: str):
        """
        Initialize a RemoteAgentTool.

        Fetches agent info from GET {base_url}/info to auto-discover
        the agent's name, description, and capabilities, then creates
        a session via POST {base_url}/sessions.

        Args:
            base_url: Base URL of the remote agent server (e.g., "http://localhost:8100")
        """
        self.base_url = base_url.rstrip("/")
        self._session_id = None

        # Auto-discover agent info
        info_response = requests.get(f"{self.base_url}/info")
        info_response.raise_for_status()
        info = info_response.json()

        name = info["name"]
        description = info["description"]

        # Create a session
        session_response = requests.post(f"{self.base_url}/sessions")
        session_response.raise_for_status()
        self._session_id = session_response.json()["session_id"]

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
        Send a task to the remote sub-agent via its session endpoints.

        Resets the session before each delegation (fresh context),
        then runs the task.

        Args:
            task: The task message to send to the remote sub-agent

        Returns:
            The remote sub-agent's response string
        """
        # Reset before each delegation (matches AgentTool behavior)
        requests.post(f"{self.base_url}/sessions/{self._session_id}/reset")

        response = requests.post(
            f"{self.base_url}/sessions/{self._session_id}/run",
            json={"message": task},
        )
        response.raise_for_status()
        return response.json()["response"]

    def close(self):
        """
        Delete the remote session. Best-effort, swallows exceptions.
        Idempotent — safe to call multiple times.
        """
        if self._session_id is None:
            return
        try:
            requests.delete(f"{self.base_url}/sessions/{self._session_id}")
        except Exception:
            pass
        self._session_id = None

    def __del__(self):
        self.close()
