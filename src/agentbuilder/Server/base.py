"""
FastAPI server for exposing agents as HTTP services.
"""

from fastapi import FastAPI
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


def create_agent_app(
    agent: AgenticLoop, name: str, description: str
) -> FastAPI:
    """
    Create a FastAPI app that exposes an agent as an HTTP service.

    Args:
        agent: The AgenticLoop instance to serve
        name: Name of the agent
        description: Description of what the agent does

    Returns:
        FastAPI app instance
    """
    app = FastAPI(title=name, description=description)

    @app.get("/info", response_model=InfoResponse)
    def info():
        return InfoResponse(name=name, description=description)

    @app.post("/run", response_model=RunResponse)
    def run(request: RunRequest):
        result = agent.run(request.message)
        return RunResponse(response=result)

    @app.post("/reset", response_model=StatusResponse)
    def reset():
        agent.reset()
        return StatusResponse(status="ok")

    @app.get("/health", response_model=StatusResponse)
    def health():
        return StatusResponse(status="healthy")

    return app


def serve_agent(
    agent: AgenticLoop,
    name: str,
    description: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    **uvicorn_kwargs,
):
    """
    Start a FastAPI server exposing the agent.

    Args:
        agent: The AgenticLoop instance to serve
        name: Name of the agent
        description: Description of what the agent does
        host: Host to bind to
        port: Port to listen on
        **uvicorn_kwargs: Additional keyword arguments passed to uvicorn.run()
    """
    import uvicorn

    app = create_agent_app(agent, name, description)
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
