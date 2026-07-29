"""Tests for MCP tool adaptation and Copilot OAuth token loading."""

import asyncio
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentbuilder.Tools.mcp import (CopilotOAuthTokenProvider,
                                    MCPAuthenticationError,
                                    MCPConfigurationError, MCPToolSet)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_from_config_loads_http_server_and_copilot_token(tmp_path):
    """A Copilot-compatible MCP config creates a toolset for the named server."""
    server_url = "https://example.test/mcp"
    config_path = tmp_path / ".mcp.json"
    oauth_dir = tmp_path / "oauth"
    oauth_dir.mkdir()
    _write_json(
        config_path,
        {
            "mcpServers": {
                "mail": {
                    "type": "http",
                    "url": server_url,
                    "oauthPublicClient": True,
                }
            }
        },
    )
    _write_json(
        oauth_dir / "mail.json",
        {
            "serverUrl": server_url,
            "clientId": "client",
            "authorizationServerUrl": "https://login.example.test/v2.0",
        },
    )
    _write_json(
        oauth_dir / "mail.tokens.json",
        {
            "accessToken": "token",
            "expiresAt": int(time.time()) + 3600,
        },
    )

    toolset = MCPToolSet.from_config(
        config_path,
        "mail",
        copilot_oauth_cache=oauth_dir,
    )

    assert toolset.server_url == server_url
    assert isinstance(toolset.token_provider, CopilotOAuthTokenProvider)


def test_from_config_rejects_missing_server(tmp_path):
    """A clear configuration error is raised for an unknown server name."""
    config_path = tmp_path / ".mcp.json"
    _write_json(config_path, {"mcpServers": {}})

    with pytest.raises(MCPConfigurationError, match="not defined"):
        MCPToolSet.from_config(config_path, "missing")


def test_copilot_provider_requires_paired_token_file(tmp_path):
    """Registration files without a matching token file are not selected."""
    _write_json(
        tmp_path / "registration.json",
        {"serverUrl": "https://example.test/mcp"},
    )

    with pytest.raises(MCPAuthenticationError, match="No Copilot OAuth token"):
        CopilotOAuthTokenProvider.for_server(
            "https://example.test/mcp",
            cache_dir=tmp_path,
        )


def test_copilot_provider_returns_unexpired_token(tmp_path):
    """A valid cached access token is returned without a refresh request."""
    registration_path = tmp_path / "registration.json"
    token_path = tmp_path / "registration.tokens.json"
    _write_json(registration_path, {"serverUrl": "https://example.test/mcp"})
    _write_json(
        token_path,
        {
            "accessToken": "cached-token",
            "expiresAt": int(time.time()) + 3600,
        },
    )
    provider = CopilotOAuthTokenProvider(registration_path, token_path)

    assert asyncio.run(provider.get_access_token()) == "cached-token"


def test_mcp_definition_becomes_executable_agentbuilder_tool():
    """The MCP JSON schema and call result are preserved by the adapter."""
    toolset = MCPToolSet("https://example.test/mcp")
    result = {"content": [{"type": "text", "text": "ok"}]}

    def submit(coroutine, timeout):
        coroutine.close()
        return result

    toolset._submit = Mock(side_effect=submit)
    definition = SimpleNamespace(
        name="SearchMessages",
        description="Search mail",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    tool = toolset._create_tool(definition)
    response = tool.execute(query="from:alice")

    assert tool.name == "SearchMessages"
    assert tool.parameters["required"] == ["query"]
    assert response.success is True
    toolset._submit.assert_called_once()


def test_confirmation_callback_can_reject_mutating_tool():
    """Confirmation-gated tools do not reach the MCP server when declined."""
    confirmation = Mock(return_value=False)
    toolset = MCPToolSet(
        "https://example.test/mcp",
        confirmation_callback=confirmation,
        confirmation_tools=["DeleteMessage"],
    )
    toolset._submit = Mock()
    definition = SimpleNamespace(
        name="DeleteMessage",
        description="Delete mail",
        input_schema={"type": "object", "properties": {}},
    )

    response = toolset._create_tool(definition).execute(messageId="123")

    assert response.success is False
    assert "declined" in response.error
    toolset._submit.assert_not_called()


class _FakeResponse:
    """Minimal urlopen stand-in supporting the context-manager read pattern."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        return self._body


def _expired_provider(tmp_path: Path) -> CopilotOAuthTokenProvider:
    registration_path = tmp_path / "server.json"
    token_path = tmp_path / "server.tokens.json"
    _write_json(
        registration_path,
        {
            "serverUrl": "https://example.test/mcp",
            "clientId": "client",
            "authorizationServerUrl": "https://login.example.test/organizations/v2.0",
        },
    )
    _write_json(
        token_path,
        {
            "accessToken": "expired",
            "refreshToken": "refresh",
            "scope": "https://example.test/mcp/Mail.All",
            "expiresAt": int(time.time()) - 60,
        },
    )
    return CopilotOAuthTokenProvider(
        registration_path=registration_path,
        token_path=token_path,
    )


def test_refresh_uses_discovered_token_endpoint_and_omits_scope(tmp_path, monkeypatch):
    """Refresh posts to the advertised token endpoint without resending scope."""
    from agentbuilder.Tools import mcp as mcp_module

    monkeypatch.setattr(mcp_module, "_TOKEN_ENDPOINT_CACHE", {})
    discovered = "https://login.example.test/organizations/oauth2/v2.0/token"
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        if request.full_url.endswith("/token"):
            return _FakeResponse({"access_token": "fresh", "expires_in": 3600})
        return _FakeResponse({"token_endpoint": discovered})

    monkeypatch.setattr(mcp_module, "urlopen", fake_urlopen)

    provider = _expired_provider(tmp_path)
    assert provider._get_access_token_sync() == "fresh"

    token_request = calls[-1]
    assert token_request.full_url == discovered
    body = token_request.data.decode("utf-8")
    assert "grant_type=refresh_token" in body
    assert "scope=" not in body
    assert json.loads(provider.token_path.read_text())["accessToken"] == "fresh"


def test_refresh_reports_relogin_when_refresh_token_is_rejected(tmp_path, monkeypatch):
    """An invalid_grant response asks the user to sign in again."""
    from urllib.error import HTTPError

    from agentbuilder.Tools import mcp as mcp_module

    monkeypatch.setattr(mcp_module, "_TOKEN_ENDPOINT_CACHE", {})

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/token"):
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                io.BytesIO(
                    json.dumps(
                        {
                            "error": "invalid_grant",
                            "error_description": "token expired",
                        }
                    ).encode("utf-8")
                ),
            )
        return _FakeResponse(
            {"token_endpoint": "https://login.example.test/oauth2/v2.0/token"}
        )

    monkeypatch.setattr(mcp_module, "urlopen", fake_urlopen)

    provider = _expired_provider(tmp_path)
    with pytest.raises(MCPAuthenticationError, match="/mcp to sign in"):
        provider._get_access_token_sync()
