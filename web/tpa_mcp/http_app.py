"""HTTP (streamable / SSE) entrypoint for the TPA MCP server.

Use this to expose the MCP server over HTTP so other machines / Claude Desktop /
other MCP clients can connect via a URL instead of a local stdio process.

Run locally with uvicorn (from the tpa_mcp directory, with TPA_API_KEY set):

    uvicorn http_app:app --host 0.0.0.0 --port 8080

The MCP endpoint is served at the URL path given by TPA_MCP_PATH (default "/mcp"):

    endpoint URL = http://<host>:8080/mcp
    e.g.          http://tpa.cambricon.com:8080/mcp   (if deployed with a public
                                                        host port mapping)

The underlying tools read TPA_API_KEY from the environment, as with the stdio
entrypoint; there is no extra auth on the MCP endpoint itself unless you add
middleware / a reverse proxy in front.
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
