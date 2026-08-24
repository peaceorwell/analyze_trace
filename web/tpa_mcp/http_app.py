"""HTTP (streamable / SSE) entrypoint for the TPA MCP server.

Use this to expose the MCP server over HTTP so other machines / Claude Desktop /
other MCP clients can connect via a URL instead of a local stdio process.

Run locally with uvicorn from the web directory. Point the MCP proxy at the
analyze API using a trusted internal URL:

    TPA_INTERNAL_BASE_URL=http://127.0.0.1:8181 \
        uvicorn tpa_mcp.http_app:app --host 0.0.0.0 --port 8080

The MCP endpoint is served at the URL path given by TPA_MCP_PATH (default "/mcp"):

    endpoint URL = http://<host>:8080/mcp
    e.g.          http://tpa.cambricon.com:8080/mcp   (if deployed with a public
                                                        host port mapping)

The HTTP endpoint uses AnalyzeTokenVerifier, so clients must send their own
analyze-server Bearer token and this process must share the analyze token DB.
TPA_API_KEY is only the fallback for the local stdio entrypoint.
"""
from __future__ import annotations

import os

from .server import mcp

# Path the MCP server is mounted at, configurable but default /mcp.
_MCP_PATH: str = os.environ.get("TPA_MCP_PATH", "/mcp")

app = mcp.http_app(path=_MCP_PATH)

if __name__ == "__main__":
    import uvicorn

    print(f"TPA MCP listening on 0.0.0.0:8080 at {_MCP_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("TPA_MCP_PORT", 8080)))
