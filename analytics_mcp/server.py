#!/usr/bin/env python

# Copyright 2025 Google LLC All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for the Google Analytics MCP server.

Supports two transports:
- stdio (default): For local use with Claude Code, Gemini CLI, etc.
- HTTP (streamable): For remote deployment (Railway, etc.) with bearer
  token auth. Activate by setting MCP_TRANSPORT=http.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import tempfile
import traceback
from collections.abc import AsyncIterator
from urllib.parse import parse_qs, urlencode

import analytics_mcp.coordinator as coordinator
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.server


# ---------------------------------------------------------------------------
# Stdio transport (local)
# ---------------------------------------------------------------------------


async def run_server_async():
    """Runs the MCP server over standard I/O."""
    print("Starting MCP Stdio Server:", coordinator.app.name)
    async with mcp.server.stdio.stdio_server() as (
        read_stream,
        write_stream,
    ):
        await coordinator.app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=coordinator.app.name,
                server_version="1.0.0",
                capabilities=coordinator.app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def run_server():
    """Synchronous wrapper to run the async MCP server."""
    asyncio.run(run_server_async())


# ---------------------------------------------------------------------------
# Streamable HTTP transport (remote / Railway)
# ---------------------------------------------------------------------------


def _setup_google_credentials():
    """Write Google credentials JSON from env var to a temp file.

    On Railway (and similar platforms) you can't mount files, so the
    service-account JSON is passed as the GOOGLE_CREDENTIALS_JSON env var.
    This writes it to a temp file and sets GOOGLE_APPLICATION_CREDENTIALS
    so that google-auth picks it up via Application Default Credentials.
    """
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if creds_json:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        tmp.write(creds_json)
        tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
        print(
            "Google credentials written to temp file from "
            "GOOGLE_CREDENTIALS_JSON"
        )


def _get_oauth_config() -> tuple[str | None, str | None]:
    """Return (client_id, client_secret) from env, or (None, None)."""
    return (
        os.environ.get("OAUTH_CLIENT_ID"),
        os.environ.get("OAUTH_CLIENT_SECRET"),
    )


def _make_access_token(client_secret: str) -> str:
    """Derive a deterministic access token from the client secret."""
    return hashlib.sha256(
        f"mcp-access-{client_secret}".encode()
    ).hexdigest()


def run_http_server():
    """Start the MCP server with streamable HTTP transport."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from mcp.server.streamable_http_manager import (
        StreamableHTTPSessionManager,
    )

    _setup_google_credentials()

    client_id, client_secret = _get_oauth_config()
    access_token = (
        _make_access_token(client_secret) if client_secret else None
    )

    # In-memory store for auth codes (code -> code_challenge)
    auth_codes: dict[str, str] = {}

    session_manager = StreamableHTTPSessionManager(
        app=coordinator.app,
        json_response=False,
        stateless=True,
    )

    # -- Helpers -----------------------------------------------------------

    async def _send_json(send, status, body):
        data = json.dumps(body).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    [b"content-type", b"application/json"],
                ],
            }
        )
        await send({"type": "http.response.body", "body": data})

    async def _send_text(send, status, text):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    [b"content-type", b"text/plain"],
                ],
            }
        )
        await send(
            {"type": "http.response.body", "body": text.encode()}
        )

    async def _send_redirect(send, url):
        await send(
            {
                "type": "http.response.start",
                "status": 302,
                "headers": [
                    [b"location", url.encode()],
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    async def _read_body(receive) -> bytes:
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body", False):
                break
        return body

    def _get_base_url(headers):
        host = headers.get(b"host", b"localhost").decode()
        scheme = "https" if "railway" in host else "http"
        return f"{scheme}://{host}"

    # -- ASGI app: OAuth + auth + MCP routing ------------------------------

    async def auth_app(scope, receive, send):
        """ASGI app handling OAuth endpoints, auth, and MCP."""
        if scope["type"] == "lifespan":
            await session_manager.handle_request(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")
        headers = dict(scope.get("headers", []))

        # -- OAuth metadata discovery -----------------------------------

        if path == "/.well-known/oauth-protected-resource":
            base = _get_base_url(headers)
            await _send_json(
                send,
                200,
                {
                    "resource": f"{base}/mcp",
                    "authorization_servers": [base],
                },
            )
            return

        if path == "/.well-known/oauth-authorization-server":
            base = _get_base_url(headers)
            await _send_json(
                send,
                200,
                {
                    "issuer": base,
                    "authorization_endpoint": f"{base}/authorize",
                    "token_endpoint": f"{base}/oauth/token",
                    "registration_endpoint": (
                        f"{base}/oauth/register"
                    ),
                    "grant_types_supported": [
                        "authorization_code",
                    ],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": [
                        "client_secret_post",
                    ],
                    "response_types_supported": ["code"],
                },
            )
            return

        # -- Dynamic client registration --------------------------------

        if path == "/oauth/register" and method == "POST":
            body = await _read_body(receive)
            reg = json.loads(body) if body else {}
            await _send_json(
                send,
                201,
                {
                    "client_id": client_id or "mcp-client",
                    "client_secret": client_secret or "",
                    "redirect_uris": reg.get(
                        "redirect_uris", []
                    ),
                },
            )
            return

        # -- Authorization endpoint -------------------------------------

        if path == "/authorize" and method == "GET":
            qs = scope.get("query_string", b"").decode()
            params = parse_qs(qs)
            req_client_id = params.get("client_id", [None])[0]
            redirect_uri = params.get("redirect_uri", [None])[0]
            state = params.get("state", [None])[0]
            code_challenge = params.get(
                "code_challenge", [None]
            )[0]

            if not redirect_uri:
                await _send_text(
                    send, 400, "Missing redirect_uri"
                )
                return

            if (
                client_id
                and req_client_id
                and req_client_id != client_id
            ):
                await _send_text(
                    send, 401, "Invalid client_id"
                )
                return

            # Generate auth code and store with its challenge
            code = secrets.token_urlsafe(32)
            auth_codes[code] = code_challenge or ""

            # Redirect back to claude.ai with the code
            redirect_params = {"code": code}
            if state:
                redirect_params["state"] = state
            redirect_url = (
                f"{redirect_uri}?{urlencode(redirect_params)}"
            )
            await _send_redirect(send, redirect_url)
            return

        # -- Token endpoint ---------------------------------------------

        if path == "/oauth/token" and method == "POST":
            if not client_id or not client_secret:
                await _send_json(
                    send,
                    500,
                    {
                        "error": "server_error",
                        "error_description": "OAuth not configured",
                    },
                )
                return

            body = await _read_body(receive)
            params = parse_qs(body.decode())
            req_grant = params.get("grant_type", [None])[0]
            req_id = params.get("client_id", [None])[0]
            req_secret = params.get("client_secret", [None])[0]

            if req_grant == "authorization_code":
                code = params.get("code", [None])[0]
                code_verifier = params.get(
                    "code_verifier", [None]
                )[0]

                if not code or code not in auth_codes:
                    await _send_json(
                        send, 400, {"error": "invalid_grant"}
                    )
                    return

                # Verify PKCE challenge
                stored_challenge = auth_codes.pop(code)
                if stored_challenge and code_verifier:
                    computed = (
                        hashlib.sha256(code_verifier.encode())
                        .digest()
                    )
                    import base64

                    expected = (
                        base64.urlsafe_b64encode(computed)
                        .rstrip(b"=")
                        .decode()
                    )
                    if expected != stored_challenge:
                        await _send_json(
                            send,
                            400,
                            {"error": "invalid_grant"},
                        )
                        return

                await _send_json(
                    send,
                    200,
                    {
                        "access_token": access_token,
                        "token_type": "Bearer",
                        "expires_in": 86400,
                    },
                )
                return

            await _send_json(
                send,
                400,
                {"error": "unsupported_grant_type"},
            )
            return

        # -- MCP endpoint (auth required) -------------------------------

        if path == "/mcp" or path == "/mcp/":
            if access_token is not None:
                auth_value = headers.get(
                    b"authorization", b""
                ).decode()
                if auth_value != f"Bearer {access_token}":
                    await _send_json(
                        send,
                        401,
                        {"error": "invalid_token"},
                    )
                    return

            await session_manager.handle_request(
                scope, receive, send
            )
            return

        await _send_text(send, 404, "Not Found")

    # -- Starlette wrapper for lifespan ---------------------------------

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    starlette_app = Starlette(
        routes=[
            Mount("/", app=auth_app),
        ],
        lifespan=lifespan,
    )

    port = int(os.environ.get("PORT", "8000"))
    print(
        f"Starting MCP HTTP Server: {coordinator.app.name} "
        f"on 0.0.0.0:{port}"
    )
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "http":
        run_http_server()
    else:
        try:
            asyncio.run(run_server_async())
        except KeyboardInterrupt:
            print("\nMCP Server (stdio) stopped by user.")
        except Exception:
            print("MCP Server (stdio) encountered an error:")
            traceback.print_exc()
        finally:
            print("MCP Server (stdio) process exiting.")
