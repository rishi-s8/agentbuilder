"""
FastAPI server for exposing agents as HTTP services with session isolation.

Each HTTP session gets its own
:class:`~agentbuilder.Loop.base.AgenticLoop` instance, created on demand
by an ``agent_factory`` callable (see :func:`~agentbuilder.utils.create_agent_factory`).

Endpoints:
    ``GET  /info``                      -- agent name and description.
    ``GET  /health``                    -- health check.
    ``POST /sessions``                  -- create a new session.
    ``POST /sessions/{id}/run``         -- run a message in a session.
    ``POST /sessions/{id}/reset``       -- reset a session's conversation.
    ``DELETE /sessions/{id}``           -- delete a session.
"""

import threading
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agentbuilder.Loop.base import AgenticLoop


class RunRequest(BaseModel):
    message: str


class RunResponse(BaseModel):
    response: str


class InfoResponse(BaseModel):
    name: str
    description: str


class StatusResponse(BaseModel):
    status: str


class SessionResponse(BaseModel):
    session_id: str


class _SessionStore:
    """Thread-safe store mapping session_id -> AgenticLoop."""

    def __init__(self):
        self._sessions: dict[str, AgenticLoop] = {}
        self._lock = threading.Lock()

    def create(self, agent_factory: Callable[[], AgenticLoop]) -> str:
        agent = agent_factory()
        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = agent
        return session_id

    def get(self, session_id: str) -> AgenticLoop:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            del self._sessions[session_id]


def create_agent_app(
    agent_factory: Callable[[], AgenticLoop],
    name: str,
    description: str,
) -> FastAPI:
    """
    Create a FastAPI app that exposes an agent as an HTTP service with session isolation.

    Args:
        agent_factory: A zero-argument callable that creates a fresh
            :class:`~agentbuilder.Loop.base.AgenticLoop` instance.
            Use :func:`~agentbuilder.utils.create_agent_factory` to build
            one.
        name: Name of the agent (returned by ``/info``).
        description: Description of what the agent does.

    Returns:
        A :class:`fastapi.FastAPI` app instance with session-isolated
        agent endpoints.

    Example::

        from agentbuilder.utils import create_agent_factory
        from agentbuilder.Server.base import create_agent_app

        factory = create_agent_factory(
            model_name="gpt-4o-mini",
            tools=[my_tool],
        )
        app = create_agent_app(factory, "my_agent", "A helpful agent")
        # Mount or run with uvicorn
    """
    app = FastAPI(title=name, description=description)
    store = _SessionStore()

    @app.get("/info", response_model=InfoResponse)
    def info():
        return InfoResponse(name=name, description=description)

    @app.get("/health", response_model=StatusResponse)
    def health():
        return StatusResponse(status="healthy")

    @app.post("/sessions", response_model=SessionResponse, status_code=201)
    def create_session():
        try:
            session_id = store.create(agent_factory)
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})
        return SessionResponse(session_id=session_id)

    @app.post("/sessions/{session_id}/run", response_model=RunResponse)
    def session_run(session_id: str, request: RunRequest):
        try:
            agent = store.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        result = agent.run(request.message)
        return RunResponse(response=result)

    @app.post("/sessions/{session_id}/reset", response_model=StatusResponse)
    def session_reset(session_id: str):
        try:
            agent = store.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        agent.reset()
        return StatusResponse(status="ok")

    @app.delete("/sessions/{session_id}", response_model=StatusResponse)
    def delete_session(session_id: str):
        try:
            store.delete(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return StatusResponse(status="ok")

    return app


def serve_agent(
    agent_factory: Callable[[], AgenticLoop],
    name: str,
    description: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    **uvicorn_kwargs,
):
    """
    Start a FastAPI server exposing the agent with session isolation.

    This is a convenience wrapper around :func:`create_agent_app` that
    starts a uvicorn server immediately.

    Args:
        agent_factory: A zero-argument callable that creates a fresh
            :class:`~agentbuilder.Loop.base.AgenticLoop` instance.
        name: Name of the agent.
        description: Description of what the agent does.
        host: Host to bind to (default ``"0.0.0.0"``).
        port: Port to listen on (default ``8000``).
        **uvicorn_kwargs: Additional keyword arguments passed to
            ``uvicorn.run()``.

    Example::

        from agentbuilder.utils import create_agent_factory
        from agentbuilder.Server import serve_agent

        factory = create_agent_factory(
            model_name="gpt-4o-mini",
            tools=[my_tool],
        )
        serve_agent(
            factory,
            name="calculator",
            description="A calculator agent",
            port=8100,
        )
    """
    import uvicorn

    app = create_agent_app(agent_factory, name, description)
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
