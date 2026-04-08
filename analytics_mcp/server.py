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


def _get_auth_token() -> str | None:
    """Return the expected bearer token, or None if auth is disabled."""
    return os.environ.get("MCP_AUTH_TOKEN")


def run_http_server():
    """Start the MCP server with streamable HTTP transport."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from mcp.server.streamable_http_manager import (
        StreamableHTTPSessionManager,
    )

    _setup_google_credentials()

    auth_token = _get_auth_token()

    session_manager = StreamableHTTPSessionManager(
        app=coordinator.app,
        json_response=False,
        stateless=True,
    )

    # -- ASGI middleware: bearer token auth + path routing ---------------

    async def auth_app(scope, receive, send):
        """ASGI app that checks bearer token then delegates to MCP."""
        if scope["type"] == "lifespan":
            await session_manager.handle_request(scope, receive, send)
            return

        # Check auth
        if auth_token is not None:
            headers = dict(scope.get("headers", []))
            auth_value = headers.get(b"authorization", b"").decode()
            if auth_value != f"Bearer {auth_token}":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            [b"content-type", b"text/plain"],
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"Unauthorized",
                    }
                )
                return

        # Route /mcp to session manager
        path = scope.get("path", "")
        if path == "/mcp" or path == "/mcp/":
            await session_manager.handle_request(
                scope, receive, send
            )
        else:
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [
                        [b"content-type", b"text/plain"],
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"Not Found",
                }
            )

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
