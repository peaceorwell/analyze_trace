"""Analyze-server token authentication for the TPA MCP server.

The MCP server is mounted inside the analyze FastAPI app. Every MCP tool call
goes back into the analyze server's own /api/* endpoints, which are per-user
isolated by the analyze ``auth_middleware`` (it resolves ``Authorization:
Bearer <token>`` against the ``api_tokens`` table).

This verifier reuses that same token store so each MCP client authenticates as
its own analyze user: the client sends ``Authorization: Bearer <user_token>``,
we validate it here, and the tool then re-uses that same token when it calls
/back into /api/* (so the middleware attributes the work to the right user).
"""
from __future__ import annotations

import hashlib

from fastmcp.server.auth import AccessToken, TokenVerifier

# The analyze app and this MCP server share a process, so we import the same
# async SQLite helpers / DB path (web/db.py) that the analyze server uses.
from db import get_db  # type: ignore[import-not-found]  # web/ is on sys.path when served

# Same table/query used by analyze server auth (web/server.py auth_middleware).
_TOKEN_QUERY = (
    "SELECT user_token, name, scope, revoked FROM api_tokens WHERE token_hash=?"
)


def hash_api_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class AnalyzeTokenVerifier(TokenVerifier):
    """Validates an analyze-server Bearer access token and maps it to a user."""

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token:
            return None
        db = await get_db()
        try:
            row = await (
                await db.execute(_TOKEN_QUERY, (hash_api_token(token),))
            ).fetchone()
        finally:
            await db.close()

        if not row or row["revoked"]:
            return None

        scope = (row["scope"] or "readonly").lower()
        user = row["user_token"]
        return AccessToken(
            token=token,
            client_id=user,
            scopes=[scope],
            expires_at=None,
            subject=user,
            claims={"token": token, "username": user, "scope": scope, "name": row["name"]},
        )
