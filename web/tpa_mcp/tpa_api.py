"""TPA (trace performance analyzer) HTTP client.

Thin wrapper around the TPA REST API using only the Python standard library
(urllib) so the MCP server has a minimal dependency surface beyond fastmcp.

Authentication: every request carries ``Authorization: Bearer <token>`` where
<token> is a user-created access token (see the "Access Token" section of the
web UI). The token is passed in explicitly; callers normally pull it from the
``TPA_API_KEY`` environment variable.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

DEFAULT_BASE_URL = os.environ.get("TPA_BASE_URL", "http://tpa.cambricon.com")


class TpaApiError(Exception):
    """Raised when the TPA API returns a non-2xx response or a transport error."""

    def __init__(self, status: Optional[int], reason: str, body: str = ""):
        self.status = status
        self.reason = reason
        self.body = body
        super().__init__(f"TPA API error ({status} {reason}): {body[:500]}")


class TpaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("TPA_API_KEY", "")

    def _headers(self, is_json: bool = False) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if is_json:
            headers["Content-Type"] = "application/json"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
        data: Optional[bytes] = None,
        content_type: Optional[str] = None,
    ) -> tuple[int, Any]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        headers = self._headers(is_json=body is not None)
        if data is not None:
            headers = self._headers()
            if content_type:
                headers["Content-Type"] = content_type

        req = urllib.request.Request(url, method=method, headers=headers)
        payload: Optional[bytes] = data
        if body is not None:
            payload = json.dumps(body).encode("utf-8")

        try:
            with urllib.request.urlopen(req, data=payload, timeout=60) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as e:
            raw = e.read()
            raise TpaApiError(
                e.code, e.reason, raw.decode("utf-8", errors="replace")
            ) from None
        except urllib.error.URLError as e:
            raise TpaApiError(None, str(e.reason)) from None

        if status in (204, 205):
            return status, None
        text = raw.decode("utf-8", errors="replace")
        try:
            return status, json.loads(text)
        except json.JSONDecodeError:
            return status, text

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params)[1]

    def post_json(self, path: str, body: dict[str, Any]) -> Any:
        return self._request("POST", path, body=body)[1]

    def post_multipart(
        self,
        path: str,
        fields: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]],
    ) -> Any:
        """POST a multipart/form-data request (used for trace uploads)."""
        boundary = "----tpa-mcp-boundary"
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n".encode("utf-8")
            )
        for name, (filename, content, ctype) in files.items():
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                    f"Content-Type: {ctype}\r\n\r\n"
                ).encode("utf-8")
                + content
                + b"\r\n"
            )
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(parts)
        return self._request(
            "POST",
            path,
            data=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )[1]
