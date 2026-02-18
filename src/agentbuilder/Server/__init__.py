"""
Server module for exposing agents as FastAPI services.

Provides functions to wrap an agent in an HTTP API with session isolation:

- :func:`create_agent_app` -- create a FastAPI ``app`` with agent endpoints.
- :func:`serve_agent` -- one-liner to start a uvicorn server.
"""

from agentbuilder.Server.base import create_agent_app, serve_agent

__all__ = ["serve_agent", "create_agent_app"]
