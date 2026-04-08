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
import tempfile
import traceback
from collections.abc import AsyncIterator

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

    async def _read_body(receive) -> bytes:
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body", False):
                break
        return body

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
            host = headers.get(b"host", b"localhost").decode()
            scheme = "https" if "railway" in host else "http"
            base = f"{scheme}://{host}"
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
            host = headers.get(b"host", b"localhost").decode()
            scheme = "https" if "railway" in host else "http"
            base = f"{scheme}://{host}"
            await _send_json(
                send,
                200,
                {
                    "issuer": base,
                    "token_endpoint": f"{base}/oauth/token",
                    "grant_types_supported": [
                        "client_credentials",
                    ],
                    "token_endpoint_auth_methods_supported": [
                        "client_secret_post",
                    ],
                    "response_types_supported": ["token"],
                },
            )
            return

        # -- OAuth token endpoint ---------------------------------------

        if path == "/oauth/token" and method == "POST":
            if not client_id or not client_secret:
                await _send_json(
                    send,
                    500,
                    {"error": "server_error", "error_description": "OAuth not configured"},
                )
                return

            body = await _read_body(receive)
            from urllib.parse import parse_qs

            params = parse_qs(body.decode())
            req_grant = params.get("grant_type", [None])[0]
            req_id = params.get("client_id", [None])[0]
            req_secret = params.get("client_secret", [None])[0]

            if req_grant != "client_credentials":
                await _send_json(
                    send,
                    400,
                    {"error": "unsupported_grant_type"},
                )
                return

            if req_id != client_id or req_secret != client_secret:
                await _send_json(
                    send,
                    401,
                    {"error": "invalid_client"},
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
