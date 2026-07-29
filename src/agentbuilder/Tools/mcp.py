"""Model Context Protocol tools for AgentBuilder.

This module adapts tools discovered from a Streamable HTTP MCP server into
AgentBuilder :class:`~agentbuilder.Tools.base.Tool` objects. MCP uses an async
client while AgentBuilder currently executes tools synchronously, so
:class:`MCPToolSet` owns a dedicated event-loop thread for the connection.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from agentbuilder.Tools.base import Tool


class MCPConfigurationError(ValueError):
    """Raised when an MCP server configuration is missing or unsupported."""


class MCPAuthenticationError(RuntimeError):
    """Raised when an MCP access token cannot be loaded or refreshed."""


class MCPToolExecutionError(RuntimeError):
    """Raised when an MCP server reports a tool execution failure."""


_RELOGIN_HINT = (
    "Run GitHub Copilot CLI in this workspace and use /mcp to sign in to the "
    "server again, then retry."
)

_REAUTHENTICATION_ERRORS = {
    "invalid_grant",
    "invalid_client",
    "unauthorized_client",
    "interaction_required",
    "consent_required",
    "login_required",
}

_TOKEN_ENDPOINT_CACHE: Dict[str, str] = {}


class StaticBearerTokenProvider:
    """Provide a fixed bearer token, typically loaded from an environment variable."""

    def __init__(self, token: str):
        if not token:
            raise MCPAuthenticationError("The bearer token is empty")
        self._token = token

    async def get_access_token(self) -> str:
        """Return the configured bearer token."""
        return self._token


class CopilotOAuthTokenProvider:
    """Reuse and refresh GitHub Copilot CLI's local MCP OAuth token.

    Copilot CLI stores OAuth client registrations and tokens in
    ``~/.copilot/mcp-oauth-config``. This provider selects a registration with
    an exact ``serverUrl`` match and a paired token file. Token values are never
    logged or copied into the project.
    """

    def __init__(
        self,
        registration_path: Path,
        token_path: Path,
        refresh_skew_seconds: int = 120,
    ):
        self.registration_path = registration_path
        self.token_path = token_path
        self.refresh_skew_seconds = refresh_skew_seconds
        self._refresh_lock = threading.Lock()

    @classmethod
    def for_server(
        cls,
        server_url: str,
        cache_dir: Optional[Path] = None,
    ) -> "CopilotOAuthTokenProvider":
        """Find a Copilot OAuth registration/token pair for an MCP server URL."""
        oauth_dir = (cache_dir or Path("~/.copilot/mcp-oauth-config")).expanduser()
        if not oauth_dir.is_dir():
            raise MCPAuthenticationError(
                f"Copilot MCP OAuth cache not found at {oauth_dir}. "
                "Authenticate the server in GitHub Copilot CLI first."
            )

        normalized_url = server_url.rstrip("/")
        matches = []
        for registration_path in sorted(oauth_dir.glob("*.json")):
            if registration_path.name.endswith(".tokens.json"):
                continue
            token_path = registration_path.with_name(
                f"{registration_path.stem}.tokens.json"
            )
            if not token_path.is_file():
                continue
            try:
                registration = _read_json(registration_path)
            except OSError, json.JSONDecodeError:
                continue
            if str(registration.get("serverUrl", "")).rstrip("/") == normalized_url:
                matches.append((registration_path, token_path))

        if not matches:
            raise MCPAuthenticationError(
                "No Copilot OAuth token is paired with MCP server "
                f"{server_url}. Run GitHub Copilot CLI in this folder and use "
                "/mcp to authenticate the server first."
            )

        registration_path, token_path = max(
            matches,
            key=lambda pair: pair[1].stat().st_mtime,
        )
        return cls(registration_path=registration_path, token_path=token_path)

    async def get_access_token(self) -> str:
        """Return a valid token, silently refreshing it when necessary."""
        return await asyncio.to_thread(self._get_access_token_sync)

    def _get_access_token_sync(self) -> str:
        with self._refresh_lock:
            tokens = _read_json(self.token_path)
            access_token = tokens.get("accessToken")
            expires_at = int(tokens.get("expiresAt") or 0)

            if access_token and expires_at > time.time() + self.refresh_skew_seconds:
                return str(access_token)

            return self._refresh_tokens(tokens)

    def _refresh_tokens(self, tokens: Dict[str, Any]) -> str:
        registration = _read_json(self.registration_path)
        refresh_token = tokens.get("refreshToken")
        client_id = registration.get("clientId")
        authorization_server = registration.get("authorizationServerUrl")

        if not refresh_token:
            raise MCPAuthenticationError(
                "The Copilot MCP access token has expired and no refresh token is "
                f"available. {_RELOGIN_HINT}"
            )
        if not client_id or not authorization_server:
            raise MCPAuthenticationError(
                "The Copilot MCP OAuth registration is incomplete."
            )

        form = {
            "client_id": str(client_id),
            "grant_type": "refresh_token",
            "refresh_token": str(refresh_token),
        }
        if registration.get("clientSecret"):
            form["client_secret"] = str(registration["clientSecret"])

        token_endpoint = _discover_token_endpoint(str(authorization_server))
        request = Request(
            token_endpoint,
            data=urlencode(form).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                refreshed = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read()
            details = _oauth_error_message(body)
            if _oauth_error_code(body) in _REAUTHENTICATION_ERRORS:
                raise MCPAuthenticationError(
                    f"The Copilot MCP refresh token is no longer valid ({details}). "
                    f"{_RELOGIN_HINT}"
                ) from error
            raise MCPAuthenticationError(
                f"Copilot MCP token refresh failed ({error.code}): {details}"
            ) from error
        except URLError as error:
            raise MCPAuthenticationError(
                f"Copilot MCP token refresh could not reach the authorization server: "
                f"{error.reason}"
            ) from error
        except json.JSONDecodeError as error:
            raise MCPAuthenticationError(
                "Copilot MCP token refresh returned invalid JSON."
            ) from error

        access_token = refreshed.get("access_token")
        if not access_token:
            raise MCPAuthenticationError(
                f"Copilot MCP token refresh failed: "
                f"{refreshed.get('error_description') or refreshed.get('error') or 'no access token returned'}"
            )

        updated = dict(tokens)
        updated["accessToken"] = access_token
        if refreshed.get("refresh_token"):
            updated["refreshToken"] = refreshed["refresh_token"]
        if refreshed.get("scope"):
            updated["scope"] = refreshed["scope"]
        updated["expiresAt"] = int(time.time()) + int(refreshed.get("expires_in", 3600))
        _write_json_atomically(self.token_path, updated)
        return str(access_token)


class MCPToolSet:
    """A live MCP connection whose discovered tools AgentBuilder can call.

    Use :meth:`from_config` to load a server from a Copilot-compatible
    ``.mcp.json`` file, then enter the toolset as a context manager before
    creating an agent.

    Example::

        from agentbuilder.Tools.mcp import MCPToolSet
        from agentbuilder.utils import create_agent

        with MCPToolSet.from_config("../.mcp.json", "workiq_mail") as mail:
            agent = create_agent(model_name="gpt-4o-mini", tools=mail.tools)
            print(agent.run("Find unread messages from today"))
    """

    def __init__(
        self,
        server_url: str,
        token_provider: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        confirmation_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        confirmation_tools: Optional[List[str]] = None,
        connect_timeout_seconds: float = 60,
        read_timeout_seconds: float = 300,
    ):
        self.server_url = server_url
        self.token_provider = token_provider
        self.headers = dict(headers or {})
        self.confirmation_callback = confirmation_callback
        self.confirmation_tools = set(confirmation_tools or [])
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.tools: List[Tool] = []

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()

    @classmethod
    def from_config(
        cls,
        config_path: Any,
        server_name: str,
        token_env_var: str = "MCP_ACCESS_TOKEN",
        copilot_oauth_cache: Optional[Any] = None,
        **kwargs,
    ) -> "MCPToolSet":
        """Create a toolset from a Copilot-compatible ``.mcp.json`` entry.

        A token in *token_env_var* takes precedence. For entries with
        ``oauthPublicClient: true``, the default is to reuse Copilot CLI's
        authenticated OAuth cache.
        """
        path = Path(config_path).expanduser().resolve()
        try:
            config = _read_json(path)
        except FileNotFoundError as error:
            raise MCPConfigurationError(f"MCP config not found: {path}") from error
        except json.JSONDecodeError as error:
            raise MCPConfigurationError(
                f"MCP config is invalid JSON: {path}"
            ) from error

        servers = config.get("mcpServers")
        if not isinstance(servers, dict) or server_name not in servers:
            raise MCPConfigurationError(
                f"MCP server {server_name!r} is not defined in {path}"
            )

        server = servers[server_name]
        if not isinstance(server, dict):
            raise MCPConfigurationError(
                f"MCP server {server_name!r} must be a JSON object"
            )

        server_type = server.get("type", "http")
        if server_type not in ("http", "streamable-http"):
            raise MCPConfigurationError(
                f"MCP server {server_name!r} uses unsupported type {server_type!r}; "
                "MCPToolSet currently supports Streamable HTTP servers"
            )

        server_url = server.get("url")
        if not server_url:
            raise MCPConfigurationError(
                f"MCP server {server_name!r} does not define a URL"
            )

        token_provider = kwargs.pop("token_provider", None)
        environment_token = os.getenv(token_env_var)
        if token_provider is None and environment_token:
            token_provider = StaticBearerTokenProvider(environment_token)
        elif token_provider is None and server.get("oauthPublicClient"):
            cache_dir = (
                Path(copilot_oauth_cache).expanduser() if copilot_oauth_cache else None
            )
            token_provider = CopilotOAuthTokenProvider.for_server(
                str(server_url),
                cache_dir=cache_dir,
            )

        configured_headers = server.get("headers", {})
        if configured_headers and not isinstance(configured_headers, dict):
            raise MCPConfigurationError(
                f"MCP server {server_name!r} headers must be a JSON object"
            )

        return cls(
            server_url=str(server_url),
            token_provider=token_provider,
            headers=configured_headers,
            **kwargs,
        )

    def __enter__(self) -> "MCPToolSet":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def start(self) -> None:
        """Connect to the MCP server and discover its tools."""
        if self._thread is not None:
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="agentbuilder-mcp",
            daemon=True,
        )
        self._thread.start()
        if not self._loop_ready.wait(timeout=self.connect_timeout_seconds):
            self.close()
            raise TimeoutError("Timed out while starting the MCP event loop")

        try:
            definitions = self._submit(
                self._discover_tools(),
                timeout=self.connect_timeout_seconds,
            )
            self.tools = [self._create_tool(definition) for definition in definitions]
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close the MCP connection and stop its event-loop thread."""
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return

        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=30)
        if not loop.is_closed():
            loop.close()
        self._loop = None
        self._thread = None
        self._loop_ready.clear()

    def _run_event_loop(self) -> None:
        if self._loop is None:
            return
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    def _submit(self, coroutine, timeout: float):
        if self._loop is None or not self._loop.is_running():
            coroutine.close()
            raise RuntimeError("MCPToolSet is not running")
        future: Future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=timeout)

    @asynccontextmanager
    async def _client_context(self):
        try:
            import httpx2
            from mcp import Client
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as error:
            raise ImportError(
                "MCP support requires Python 3.10+ and the MCP extra. "
                'Install it with: pip install -e ".[mcp]"'
            ) from error

        request_headers = dict(self.headers)
        if self.token_provider is not None:
            token = await self.token_provider.get_access_token()
            request_headers["Authorization"] = f"Bearer {token}"

        async with httpx2.AsyncClient(
            headers=request_headers,
            timeout=httpx2.Timeout(30.0, read=self.read_timeout_seconds),
            follow_redirects=True,
        ) as http_client:
            transport = streamable_http_client(
                self.server_url,
                http_client=http_client,
            )
            async with Client(
                transport,
                mode="legacy",
                read_timeout_seconds=self.read_timeout_seconds,
            ) as client:
                yield client

    async def _discover_tools(self):
        async with self._client_context() as client:
            return await self._list_all_tools(client)

    async def _list_all_tools(self, client):
        definitions = []
        cursor = None
        while True:
            page = await client.list_tools(cursor=cursor)
            definitions.extend(page.tools)
            cursor = getattr(page, "next_cursor", None)
            if cursor is None:
                return definitions

    def _create_tool(self, definition) -> Tool:
        name = definition.name
        description = definition.description or f"Call the MCP tool {name}"
        parameters = getattr(definition, "input_schema", None)
        if parameters is None:
            parameters = getattr(definition, "inputSchema", None)
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}

        def invoke(**arguments):
            if (
                name in self.confirmation_tools
                and self.confirmation_callback is not None
                and not self.confirmation_callback(name, arguments)
            ):
                raise PermissionError(f"User declined MCP tool {name}")
            return self._submit(
                self._call_tool(name, arguments),
                timeout=self.read_timeout_seconds,
            )

        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=invoke,
        )

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        async with self._client_context() as client:
            result = await client.call_tool(
                name,
                arguments,
                read_timeout_seconds=self.read_timeout_seconds,
            )
        payload = result.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        if getattr(result, "is_error", False):
            raise MCPToolExecutionError(_tool_error_message(payload))
        return payload


def _discover_token_endpoint(authorization_server: str) -> str:
    """Resolve an authorization server's token endpoint from its metadata.

    The OAuth 2.0 token endpoint is read from authorization-server metadata
    (RFC 8414) or the OpenID Connect discovery document. It is never guessed by
    appending a path to the issuer URL, because providers such as Microsoft
    Entra ID serve tokens from an unrelated path.
    """
    issuer = authorization_server.rstrip("/")
    cached = _TOKEN_ENDPOINT_CACHE.get(issuer)
    if cached:
        return cached

    parsed = urlsplit(issuer)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    candidates = [
        f"{origin}/.well-known/oauth-authorization-server{path}",
        f"{origin}/.well-known/openid-configuration{path}",
        f"{issuer}/.well-known/oauth-authorization-server",
        f"{issuer}/.well-known/openid-configuration",
    ]

    errors = []
    for candidate in candidates:
        try:
            with urlopen(Request(candidate, method="GET"), timeout=30) as response:
                metadata = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, json.JSONDecodeError, UnicodeDecodeError) as error:
            errors.append(f"{candidate}: {error}")
            continue
        token_endpoint = metadata.get("token_endpoint")
        if token_endpoint:
            _TOKEN_ENDPOINT_CACHE[issuer] = str(token_endpoint)
            return str(token_endpoint)
        errors.append(f"{candidate}: no token_endpoint")

    raise MCPAuthenticationError(
        "Could not discover the OAuth token endpoint for authorization server "
        f"{authorization_server}. Tried: {'; '.join(errors)}"
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise json.JSONDecodeError("Expected a JSON object", "", 0)
    return value


def _write_json_atomically(path: Path, value: Dict[str, Any]) -> None:
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.chmod(temporary_path, mode)
    os.replace(temporary_path, path)


def _oauth_error_message(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except UnicodeDecodeError, json.JSONDecodeError:
        return "authorization server returned an error"
    return str(
        payload.get("error_description")
        or payload.get("error")
        or "authorization server returned an error"
    )


def _oauth_error_code(body: bytes) -> str:
    """Return the machine-readable ``error`` code from an OAuth error body."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except UnicodeDecodeError, json.JSONDecodeError:
        return ""
    return str(payload.get("error") or "")


def _tool_error_message(payload: Dict[str, Any]) -> str:
    text_blocks = []
    for block in payload.get("content", []):
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and block.get("text")
        ):
            text_blocks.append(str(block["text"]))
    return "\n".join(text_blocks) or "MCP tool execution failed"
