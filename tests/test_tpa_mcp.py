import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WEB_DIR))

from fastmcp.server import dependencies as fastmcp_dependencies  # noqa: E402
from tpa_mcp import server as mcp_server  # noqa: E402
from tpa_mcp.tpa_api import TpaClient  # noqa: E402


def test_mcp_client_uses_server_socket_not_host_header(monkeypatch):
    request = SimpleNamespace(
        scope={"server": ("0.0.0.0", 8181)},
        headers={"host": "169.254.169.254"},
    )
    monkeypatch.delenv("TPA_INTERNAL_BASE_URL", raising=False)
    monkeypatch.setattr(mcp_server, "_http_request_or_none", lambda: request)
    monkeypatch.setattr(
        fastmcp_dependencies,
        "get_access_token",
        lambda: SimpleNamespace(token="user-token"),
    )

    client = mcp_server._client()

    assert client.base_url == "http://127.0.0.1:8181"
    assert client.token == "user-token"


def test_remote_http_upload_rejects_server_paths_before_reading(monkeypatch):
    monkeypatch.setattr(mcp_server, "_http_request_or_none", lambda: object())
    monkeypatch.setattr(
        mcp_server,
        "_validate_local_file",
        lambda path: (_ for _ in ()).throw(AssertionError(f"read attempted: {path}")),
    )

    result = mcp_server.tpa_upload_trace("/srv/private/config.py")

    assert "disabled over remote HTTP MCP" in result["error"]


def test_multipart_upload_streams_file_chunks(tmp_path, monkeypatch):
    trace = tmp_path / "large.json"
    content = b"x" * (1024 * 1024 + 17)
    trace.write_bytes(content)
    captured = {}
    client = TpaClient(base_url="http://127.0.0.1:8181", token="token")

    def fake_request(method, path, **kwargs):
        chunks = list(kwargs["data"])
        captured.update(kwargs, chunks=chunks, payload=b"".join(chunks))
        return 201, {"id": "job-1"}

    monkeypatch.setattr(client, "_request", fake_request)

    result = client.post_multipart(
        "/api/jobs",
        {"save_triton_csv": "true"},
        {"file_a": (trace.name, str(trace), "application/json")},
    )

    assert result == {"id": "job-1"}
    assert captured["content_length"] == len(captured["payload"])
    assert captured["payload"].count(content) == 1
    assert max(len(chunk) for chunk in captured["chunks"]) <= 1024 * 1024


def test_stdio_entrypoint_can_be_loaded_as_a_script():
    runpy.run_path(str(WEB_DIR / "tpa_mcp" / "server.py"), run_name="tpa_mcp_probe")
