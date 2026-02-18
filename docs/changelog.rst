Changelog
=========

Unreleased
----------

- Added comprehensive docstrings to all modules.
- Added Sphinx documentation site with Furo theme.
- Added quickstart, concept guides, user guides, and API reference.
- Added new example projects: code_agent, multi_agent, served_agent.
- Added ``docs`` optional dependency group.
- Added ``project.urls`` to ``pyproject.toml``.
- Added ``.readthedocs.yaml`` for ReadTheDocs deployment.

0.0.0
-----

- Initial release with core framework.
- Tool-calling agents with Pydantic parameter models.
- OpenAI-compatible conversation wrapper with ``.env`` loading.
- Plan-execute agentic loop with configurable iteration limits.
- Local sub-agent delegation (AgentTool).
- Remote sub-agent delegation (RemoteAgentTool).
- Docker-based code execution sandbox.
- FastAPI server with session isolation.
- Simple calculator example.
