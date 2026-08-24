"""TPA (trace performance analyzer) HTTP client.

Thin wrapper around the TPA REST API using only the Python standard library
(urllib) so the MCP server has a minimal dependency surface beyond fastmcp.

Authentication: every request carries ``Authorization: Bearer <token>`` where
<token> is a user-created access token (see the "Access Token" section of the
web UI). The token is passed in explicitly; callers normally pull it from the
``TPA_API_KEY`` environment variable.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
import secrets
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
        data: Optional[Any] = None,
        content_type: Optional[str] = None,
        content_length: Optional[int] = None,
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
            if content_length is not None:
                headers["Content-Length"] = str(content_length)

        payload = data
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method=method, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
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
        files: dict[str, tuple[str, str, str]],
    ) -> Any:
        """Stream a multipart/form-data request from local file paths."""
        boundary = f"----tpa-mcp-{secrets.token_hex(16)}"
        field_parts: list[bytes] = []
        for name, value in fields.items():
            field_parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n".encode("utf-8")
            )
        closing = f"--{boundary}--\r\n".encode("ascii")

        with ExitStack() as stack:
            file_parts = []
            for name, (filename, file_path, ctype) in files.items():
                safe_filename = filename.replace("\\", "\\\\").replace('"', '\\"')
                header = (
                    f"--{boundary}\r\n"
                    f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{safe_filename}\"\r\n"
                    f"Content-Type: {ctype}\r\n\r\n"
                ).encode("utf-8")
                file_obj = stack.enter_context(open(file_path, "rb"))
                file_parts.append((header, file_obj, os.fstat(file_obj.fileno()).st_size))

            content_length = sum(len(part) for part in field_parts) + len(closing)
            content_length += sum(len(header) + size + 2 for header, _, size in file_parts)

            def body_chunks():
                yield from field_parts
                for header, file_obj, _ in file_parts:
                    yield header
                    while chunk := file_obj.read(1024 * 1024):
                        yield chunk
                    yield b"\r\n"
                yield closing

            return self._request(
                "POST",
                path,
                data=body_chunks(),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=content_length,
            )[1]
