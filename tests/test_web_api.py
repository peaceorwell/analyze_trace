import asyncio
import gzip
import json
import os
import shlex
import sys
import tarfile
import textwrap
import time
from pathlib import Path
import shutil

import aiosqlite
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WEB_DIR))

import db as web_db  # noqa: E402
import backup as web_backup  # noqa: E402
import server as web_server  # noqa: E402


@pytest.fixture
def isolated_server(tmp_path, monkeypatch):
    storage_dir = tmp_path / "storage"
    monkeypatch.delenv("TRACE_LOG_FILE", raising=False)
    monkeypatch.setattr(web_db, "DB_PATH", str(storage_dir / "jobs.db"))
    monkeypatch.setattr(web_server, "STORAGE_DIR", str(storage_dir))
    monkeypatch.setattr(web_server, "BACKUP_DIR", str(storage_dir / "backups"))
    monkeypatch.setattr(web_server, "ALLOW_FILE_DOWNLOAD", True)
    monkeypatch.setattr(web_server, "ALLOW_CODE_EXECUTION", False)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", False)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_COMMAND_TEMPLATE", "")
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_EXTRA_ARGS", "")
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(web_server, "AUTH_MODE", "none")
    monkeypatch.setattr(web_server, "AUTH_ENABLED", False)
    monkeypatch.setattr(web_server, "ADMIN_USERS", set())
    web_server.LOGIN_FAILURES.clear()
    web_server.LOGIN_CAPTCHA_CHALLENGES.clear()
    web_server.ai_analysis_tasks.clear()
    return web_server


@pytest.fixture
def client(isolated_server):
    with TestClient(isolated_server.app) as test_client:
        yield test_client


def test_config_reports_local_execution_flags(client):
    r = client.get("/api/config")

    assert r.status_code == 200
    assert r.json() == {
        "version": "0.2.28",
        "auth_mode": "none",
        "auth_required": False,
        "allow_file_download": True,
        "allow_code_execution": False,
        "claude_analysis_enabled": False,
    }


def test_ai_analysis_is_disabled_by_default(client):
    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status) VALUES(?,?,?,?)",
                ("ai-disabled-job", "AI disabled", "single", "done"),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    r = client.post("/api/jobs/ai-disabled-job/ai-analysis", json={})

    assert r.status_code == 403


def test_claude_command_normalizes_legacy_permission_args(isolated_server, monkeypatch):
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_COMMAND", "/usr/local/node20/bin/claude")
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_COMMAND_TEMPLATE", "")
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_EXTRA_ARGS", "--permission-mode bypassPermissions")

    command = web_server._build_claude_command("OK", {})

    assert command == ["/usr/local/node20/bin/claude", "--dangerously-skip-permissions", "-p", "OK"]


def test_claude_command_template_normalizes_legacy_permission_args(isolated_server, monkeypatch):
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        "/usr/local/node20/bin/claude --permission-mode=bypassPermissions -p {prompt}",
    )

    command = web_server._build_claude_command("OK", {})

    assert command == ["/usr/local/node20/bin/claude", "--dangerously-skip-permissions", "-p", "OK"]


def test_claude_skills_mount_is_writable_copy(tmp_path):
    skills_dir = tmp_path / "source-skills"
    skill_dir = skills_dir / "e2e-profiling-analyzer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: e2e-profiling-analyzer\n---\n", encoding="utf-8")
    analysis_dir = tmp_path / "analysis"

    web_server._mount_claude_skills_for_analysis(str(analysis_dir), str(skills_dir))

    target = analysis_dir / ".claude" / "skills"
    assert target.exists()
    assert not target.is_symlink()
    assert (target / "e2e-profiling-analyzer" / "SKILL.md").exists()
    (analysis_dir / ".claude" / "session-env").mkdir()
    env = web_server._build_claude_env({"HOME": str(tmp_path)}, str(analysis_dir), {"TRACE_AI_JOB_ID": "x"})
    assert env["CLAUDE_PROJECT_DIR"] == str(analysis_dir)
    assert env["CLAUDE_CODE_PROJECT_DIR"] == str(analysis_dir)


def _fake_claude_template(tmp_path: Path, analysis_body: str) -> str:
    script = tmp_path / "fake_claude.py"
    body = textwrap.indent(textwrap.dedent(analysis_body).strip() or "pass", "    ")
    script.write_text(
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "\n"
        "prompt = sys.argv[1] if len(sys.argv) > 1 else \"\"\n"
        "if \"claude_tool_probe.txt\" in prompt:\n"
        "    pathlib.Path(\"claude_tool_probe.txt\").write_text(\n"
        "        os.environ.get(\"TRACE_AI_TOOL_PROBE_TOKEN\", \"\"),\n"
        "        encoding=\"utf-8\",\n"
        "    )\n"
        "    print(\"OK\")\n"
        "elif \"这是环境诊断\" in prompt:\n"
        "    skill = os.environ.get(\"TRACE_AI_SKILL\", \"\")\n"
        "    print(\"OK\")\n"
        "    print(os.path.exists(pathlib.Path(\".claude\") / \"skills\" / skill / \"SKILL.md\"))\n"
        "else:\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} {{prompt}}"


def test_ai_analysis_runs_configured_command(client, isolated_server, tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.pt.trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as f:
        f.write("{}")

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(id, label, mode, status, file_a_name, file_a_gzip_path)
                VALUES(?,?,?,?,?,?)
                """,
                ("ai-job", "AI job", "single", "done", "trace.pt.trace.json.gz", str(trace_path)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.result_dir("ai-job")).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        _fake_claude_template(
            tmp_path,
            """
            pathlib.Path('details.txt').write_text('Detail Artifact\\n', encoding='utf-8')
            pathlib.Path('metrics.json').write_text('{\"ok\": true}\\n', encoding='utf-8')
            pathlib.Path('report.md').write_text('# Local Report\\n', encoding='utf-8')
            pathlib.Path(os.environ['TRACE_AI_REPORT_PATH']).write_text('# Final Report\\n', encoding='utf-8')
            print('# AI OK')
            print('prompt_len=' + str(len(prompt)))
            """,
        ),
    )

    started = client.post("/api/jobs/ai-job/ai-analysis", json={})

    assert started.status_code == 202

    payload = {}
    for _ in range(80):
        payload = client.get("/api/jobs/ai-job/ai-analysis").json()
        if payload["status"] != "running":
            break
        time.sleep(0.05)

    assert payload["status"] == "done"
    assert payload["diagnostics"]["ok"] is True
    assert payload["content"].strip() == "# Final Report"
    artifact_paths = {item["path"] for item in payload["artifacts"]}
    assert "ai_analysis.md" in artifact_paths
    assert "report.md" in artifact_paths
    assert "details.txt" in artifact_paths
    assert "metrics.json" in artifact_paths
    assert "stdout.txt" in artifact_paths
    assert "stderr.txt" in artifact_paths
    assert "command.json" not in artifact_paths
    assert "ai_analysis_status.json" not in artifact_paths
    details_artifact = next(item for item in payload["artifacts"] if item["path"] == "details.txt")
    assert "Detail Artifact" in details_artifact["content"]

    detail = client.get("/api/jobs/ai-job").json()
    assert detail["ai_analysis"]["status"] == "done"
    assert detail["ai_analysis"]["report_exists"] is True


def test_ai_analysis_mounts_configured_claude_skills(client, tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.pt.trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as f:
        f.write("{}")
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "e2e-profiling-analyzer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: e2e-profiling-analyzer\n---\n", encoding="utf-8")
    compare_skill_dir = skills_dir / "e2e-profiling-comparator"
    compare_skill_dir.mkdir(parents=True)
    (compare_skill_dir / "SKILL.md").write_text("---\nname: e2e-profiling-comparator\n---\n", encoding="utf-8")

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(id, label, mode, status, file_a_name, file_a_gzip_path)
                VALUES(?,?,?,?,?,?)
                """,
                ("ai-skills-job", "AI skills job", "single", "done", trace_path.name, str(trace_path)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.result_dir("ai-skills-job")).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_SKILLS_DIR", str(skills_dir))
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        _fake_claude_template(
            tmp_path,
            """
            print(os.path.exists('.claude/skills/e2e-profiling-analyzer/SKILL.md'))
            print(os.environ['TRACE_CLAUDE_SKILLS_DIR'])
            """,
        ),
    )

    started = client.post("/api/jobs/ai-skills-job/ai-analysis", json={})

    assert started.status_code == 202

    payload = {}
    for _ in range(80):
        payload = client.get("/api/jobs/ai-skills-job/ai-analysis").json()
        if payload["status"] != "running":
            break
        time.sleep(0.05)

    assert payload["status"] == "done"
    assert payload["diagnostics"]["ok"] is True
    assert "True" in payload["content"]
    assert str(skills_dir) in payload["content"]


def test_ai_analysis_marks_permission_failure_output_as_error(client, isolated_server, tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.pt.trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as f:
        f.write("{}")

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(id, label, mode, status, file_a_name, file_a_gzip_path)
                VALUES(?,?,?,?,?,?)
                """,
                ("ai-permission-job", "AI permission job", "single", "done", trace_path.name, str(trace_path)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.result_dir("ai-permission-job")).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        (
            f"{shlex.quote(sys.executable)} -c "
            "\"print('ERROR: tool denied'); "
            "print('All tool calls are being denied.')\""
        ),
    )

    started = client.post("/api/jobs/ai-permission-job/ai-analysis", json={})

    assert started.status_code == 202

    payload = {}
    for _ in range(80):
        payload = client.get("/api/jobs/ai-permission-job/ai-analysis").json()
        if payload["status"] != "running":
            break
        time.sleep(0.05)

    assert payload["status"] == "error"
    assert payload["phase"] == "diagnostics_failed"
    assert payload["diagnostics"]["ok"] is False
    assert "AI environment diagnostics failed" in payload["error"]
    assert "AI 环境诊断未通过" in payload["content"]
    assert "工具权限探针" in payload["content"]
    assert "ERROR: tool denied" in payload["content"]


def test_ai_analysis_report_markdown_download(client):
    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status) VALUES(?,?,?,?)",
                ("ai-download-job", "AI report", "single", "done"),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.ai_analysis_dir("ai-download-job")).mkdir(parents=True, exist_ok=True)
    Path(web_server.ai_analysis_report_path("ai-download-job")).write_text(
        "# AI 性能分析报告\n\n## 结论概览\n\n- ok\n",
        encoding="utf-8",
    )

    response = client.get("/api/jobs/ai-download-job/ai-analysis/report.md")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "AI report-ai-analysis.md" in response.headers["content-disposition"]
    assert "# AI 性能分析报告" in response.text


def test_ai_analysis_reports_missing_claude_command(client, tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.pt.trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as f:
        f.write("{}")

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(id, label, mode, status, file_a_name, file_a_gzip_path)
                VALUES(?,?,?,?,?,?)
                """,
                ("ai-missing-command", "AI missing command", "single", "done", trace_path.name, str(trace_path)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.result_dir("ai-missing-command")).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_COMMAND_TEMPLATE", "")
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_COMMAND", "__missing_claude_code_for_test__")

    started = client.post("/api/jobs/ai-missing-command/ai-analysis", json={})

    assert started.status_code == 202

    payload = {}
    for _ in range(80):
        payload = client.get("/api/jobs/ai-missing-command/ai-analysis").json()
        if payload["status"] != "running":
            break
        time.sleep(0.05)

    assert payload["status"] == "error"
    assert payload["phase"] == "diagnostics_failed"
    assert payload["diagnostics"]["ok"] is False
    assert "AI 环境诊断未通过" in payload["content"]
    assert "Claude Code command not found" in payload["content"]
    assert "__missing_claude_code_for_test__" in payload["content"]
    assert "TRACE_CLAUDE_COMMAND" in payload["content"]


def test_ai_diagnostics_runs_command_and_skill_smoke(client, tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    for skill_name in ("e2e-profiling-analyzer", "e2e-profiling-comparator"):
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\n", encoding="utf-8")

    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_SKILLS_DIR", str(skills_dir))
    monkeypatch.setattr(web_server, "CLAUDE_DIAGNOSTIC_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        (
            f"{shlex.quote(sys.executable)} -c "
            "\"import os, sys; "
            "prompt = sys.argv[1] if len(sys.argv) > 1 else ''; "
            "open('claude_tool_probe.txt', 'w').write(os.environ['TRACE_AI_TOOL_PROBE_TOKEN']) if 'claude_tool_probe.txt' in prompt else None; "
            "print('OK'); "
            "print(os.path.exists('.claude/skills/e2e-profiling-analyzer/SKILL.md'))\" "
            "{prompt}"
        ),
    )

    response = client.post("/api/ai/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["command"]["status"] == "ok"
    assert checks["skills_dir"]["status"] == "ok"
    assert checks["single_skill"]["status"] == "ok"
    assert checks["compare_skill"]["status"] == "ok"
    assert checks["skills_mount"]["status"] == "ok"
    assert checks["base_smoke"]["status"] == "ok"
    assert checks["tool_probe"]["status"] == "ok"
    assert checks["skill_smoke"]["status"] == "ok"
    assert "True" in checks["skill_smoke"]["stdout_tail"]


def test_ai_diagnostics_reports_missing_command(client, monkeypatch):
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_COMMAND_TEMPLATE", "")
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_COMMAND", "__missing_claude_diag_for_test__")

    response = client.post("/api/ai/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["command"]["status"] == "error"
    assert "__missing_claude_diag_for_test__" in checks["command"]["detail"]
    assert checks["base_smoke"]["status"] == "skipped"


def test_ai_analysis_supports_compare_jobs(client, tmp_path, monkeypatch):
    trace_a = tmp_path / "a.pt.trace.json.gz"
    trace_b = tmp_path / "b.pt.trace.json.gz"
    for path in (trace_a, trace_b):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("{}")

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status,
                    file_a_name, file_a_gzip_path,
                    file_b_name, file_b_gzip_path
                )
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    "ai-compare-job", "AI compare", "compare", "done",
                    "a.pt.trace.json.gz", str(trace_a),
                    "b.pt.trace.json.gz", str(trace_b),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.result_dir("ai-compare-job")).mkdir(parents=True, exist_ok=True)
    skills_dir = tmp_path / "skills"
    single_skill_dir = skills_dir / "e2e-profiling-analyzer"
    single_skill_dir.mkdir(parents=True)
    (single_skill_dir / "SKILL.md").write_text("---\nname: e2e-profiling-analyzer\n---\n", encoding="utf-8")
    skill_dir = skills_dir / "compare-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: compare-skill\n---\n", encoding="utf-8")

    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(web_server, "CLAUDE_COMPARE_TRACE_SKILL", "compare-skill")
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_SKILLS_DIR", str(skills_dir))
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        _fake_claude_template(
            tmp_path,
            """
            print('# Compare OK')
            print(os.environ['TRACE_AI_MODE'])
            print(os.environ['TRACE_AI_SKILL'])
            print(os.environ['TRACE_AI_TRACE_B'])
            """,
        ),
    )

    started = client.post("/api/jobs/ai-compare-job/ai-analysis", json={})

    assert started.status_code == 202

    payload = {}
    for _ in range(80):
        payload = client.get("/api/jobs/ai-compare-job/ai-analysis").json()
        if payload["status"] != "running":
            break
        time.sleep(0.05)

    assert payload["status"] == "done"
    assert payload["diagnostics"]["ok"] is True
    assert "# Compare OK" in payload["content"]
    assert "compare-skill" in payload["content"]
    assert str(trace_b) in payload["content"]


def test_feedback_board_supports_images_and_replies(client):
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24

    created = client.post(
        "/api/feedback",
        data={"body": "建议增加批量导出"},
        files=[("images", ("idea.png", png_bytes, "image/png"))],
        headers={"X-Remote-User": "alice"},
    )

    assert created.status_code == 201
    message = created.json()
    assert message["body"] == "建议增加批量导出"
    assert message["user_display"] == "alice"
    assert len(message["attachments"]) == 1

    attachment = client.get(message["attachments"][0]["url"], headers={"X-Remote-User": "alice"})
    assert attachment.status_code == 200
    assert attachment.headers["content-type"].startswith("image/png")
    assert attachment.headers["content-disposition"].startswith("inline")
    assert attachment.content == png_bytes

    reply = client.post(
        "/api/feedback",
        data={"body": "这个确实有用", "parent_id": message["id"]},
        headers={"X-Remote-User": "bob"},
    )

    assert reply.status_code == 201

    listed = client.get("/api/feedback")
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["total"] == 1
    assert payload["data"][0]["id"] == message["id"]
    assert payload["data"][0]["attachments"][0]["filename"] == "idea.png"
    assert payload["data"][0]["reply_count"] == 1
    assert payload["data"][0]["last_activity_at"]
    assert payload["data"][0]["replies"][0]["body"] == "这个确实有用"
    assert payload["data"][0]["replies"][0]["user_display"] == "bob"

    detail = client.get(f"/api/feedback/{message['id']}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["id"] == message["id"]
    assert detail_payload["reply_count"] == 1
    assert detail_payload["replies"][0]["body"] == "这个确实有用"

    detail_from_reply = client.get(f"/api/feedback/{reply.json()['id']}")
    assert detail_from_reply.status_code == 200
    assert detail_from_reply.json()["id"] == message["id"]


def test_feedback_delete_requires_admin_and_removes_files(client, isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "ADMIN_USERS", {"admin"})
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24

    created = client.post(
        "/api/feedback",
        data={"body": "需要管理员清理"},
        files=[("images", ("cleanup.png", png_bytes, "image/png"))],
        headers={"X-Remote-User": "alice"},
    )
    assert created.status_code == 201
    post = created.json()
    attachment_url = post["attachments"][0]["url"]

    reply = client.post(
        "/api/feedback",
        data={"body": "一条回复", "parent_id": post["id"]},
        headers={"X-Remote-User": "bob"},
    )
    assert reply.status_code == 201

    denied = client.delete(f"/api/feedback/{post['id']}", headers={"X-Remote-User": "alice"})
    assert denied.status_code == 403

    deleted_reply = client.delete(
        f"/api/feedback/{reply.json()['id']}",
        headers={"X-Remote-User": "admin"},
    )
    assert deleted_reply.status_code == 200
    assert deleted_reply.json()["deleted"] == 1
    detail = client.get(f"/api/feedback/{post['id']}")
    assert detail.status_code == 200
    assert detail.json()["reply_count"] == 0

    deleted_post = client.delete(f"/api/feedback/{post['id']}", headers={"X-Remote-User": "admin"})
    assert deleted_post.status_code == 200
    assert deleted_post.json()["deleted"] == 1
    assert client.get(attachment_url).status_code == 404
    listed = client.get("/api/feedback")
    assert listed.status_code == 200
    assert listed.json()["total"] == 0


def test_feedback_board_rejects_non_images(client):
    response = client.post(
        "/api/feedback",
        data={"body": "bad file"},
        files=[("images", ("note.txt", b"not an image", "text/plain"))],
    )

    assert response.status_code == 400


def test_ops_endpoints_and_audit_logs(client):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["checks"]["db"] == "ok"
    assert ready["checks"]["storage"] == "ok"
    assert ready["checks"]["backup"] == "ok"
    assert ready["checks"]["log_file"] == "disabled"
    assert ready["paths"]["storage"]

    created = client.post(
        "/api/projects",
        json={"name": "Audit Project"},
        headers={"X-Remote-User": "alice"},
    )
    assert created.status_code == 201
    project_id = created.json()["id"]

    logs = client.get("/api/audit-logs", params={"action": "project.create"})
    assert logs.status_code == 200
    payload = logs.json()
    assert payload["total"] == 1
    assert payload["data"][0]["user"] == "alice"
    assert payload["data"][0]["resource_id"] == project_id

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "analyze_trace_app_uptime_seconds" in metrics.text
    assert "analyze_trace_http_requests_total" in metrics.text


def test_readyz_reports_log_file_writeability(client, tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_LOG_FILE", str(tmp_path / "logs" / "app.jsonl"))

    ready = client.get("/readyz").json()

    assert ready["status"] == "ok"
    assert ready["checks"]["log_file"] == "ok"
    assert ready["paths"]["log_file"].endswith("app.jsonl")


def test_ldap_auth_requires_login_and_isolates_user_data(isolated_server, monkeypatch):
    def fake_authenticate(username, password):
        if password != "ok":
            raise web_server.ldap_auth.AuthError("bad credentials")
        return {
            "username": username,
            "display_name": f"{username} User",
            "email": f"{username}@example.com",
            "dn": f"CN={username},DC=example,DC=com",
        }

    monkeypatch.setattr(web_server, "AUTH_MODE", "ldap")
    monkeypatch.setattr(web_server, "AUTH_ENABLED", True)
    monkeypatch.setattr(web_server.ldap_auth, "authenticate", fake_authenticate)

    with TestClient(isolated_server.app) as test_client:
        assert test_client.get("/api/projects").status_code == 401

        bad = test_client.post("/api/login", json={"username": "alice", "password": "bad"})
        assert bad.status_code == 401

        login = test_client.post("/api/login", json={"username": "alice", "password": "ok"})
        assert login.status_code == 200
        assert login.json()["user"]["username"] == "alice"

        created = test_client.post("/api/projects", json={"name": "Alice Project"})
        assert created.status_code == 201
        alice_project = created.json()
        assert alice_project["user_token"] == "alice"
        assert alice_project["is_public"] == 0

        shared_created = test_client.post("/api/projects", json={"name": "Shared Project", "is_public": True})
        assert shared_created.status_code == 201
        shared_project = shared_created.json()
        assert shared_project["user_token"] == "alice"
        assert shared_project["is_public"] == 1
        assert shared_project["is_owner"] is True

        async def insert_rows():
            db = await web_db.get_db()
            try:
                await db.execute("INSERT OR IGNORE INTO users(user_token) VALUES(?)", ("bob",))
                await db.execute(
                    "INSERT INTO projects(id, user_token, name) VALUES(?,?,?)",
                    ("bob-project", "bob", "Bob Project"),
                )
                await db.executemany(
                    "INSERT INTO jobs(id, project_id, user_token, label, mode, status) VALUES(?,?,?,?,?,?)",
                    [
                        ("alice-job", alice_project["id"], "alice", "alice job", "single", "done"),
                        ("shared-job", shared_project["id"], "alice", "shared job", "single", "done"),
                        ("bob-job", "bob-project", "bob", "bob job", "single", "done"),
                    ],
                )
                await db.commit()
            finally:
                await db.close()

        asyncio.run(insert_rows())

        projects = test_client.get("/api/projects")
        assert projects.status_code == 200
        assert {item["id"] for item in projects.json()} == {alice_project["id"], shared_project["id"]}

        jobs = test_client.get("/api/jobs")
        assert jobs.status_code == 200
        assert {item["id"] for item in jobs.json()["data"]} == {"alice-job", "shared-job"}

        assert test_client.get("/api/jobs/bob-job").status_code == 404
        assert test_client.get("/api/jobs/alice-job").status_code == 200

        assert test_client.post("/api/logout").status_code == 200
        bob_login = test_client.post("/api/login", json={"username": "bob", "password": "ok"})
        assert bob_login.status_code == 200

        bob_projects = test_client.get("/api/projects")
        assert bob_projects.status_code == 200
        assert {item["id"] for item in bob_projects.json()} == {"bob-project", shared_project["id"]}

        assert test_client.get("/api/jobs/alice-job").status_code == 404
        shared_detail = test_client.get("/api/jobs/shared-job")
        assert shared_detail.status_code == 200
        assert shared_detail.json()["is_owner"] is False

        cannot_patch = test_client.patch("/api/jobs/shared-job", json={"label": "bob edit"})
        assert cannot_patch.status_code == 404

        shared_jobs = test_client.get(f"/api/jobs?project_id={shared_project['id']}")
        assert shared_jobs.status_code == 200
        assert [item["id"] for item in shared_jobs.json()["data"]] == ["shared-job"]

        candidates = test_client.get(f"/api/compare-candidates?project_id={shared_project['id']}")
        assert candidates.status_code == 200
        assert [item["id"] for item in candidates.json()["data"]] == ["shared-job"]


@pytest.mark.parametrize(
    ("admin_identity", "expected"),
    [
        ("alice", True),
        ("alice@example.com", True),
        ("Alice User", True),
        ("bob", False),
    ],
)
def test_ldap_me_reports_admin_for_configured_identity(isolated_server, monkeypatch, admin_identity, expected):
    def fake_authenticate(username, password):
        return {
            "username": username,
            "display_name": f"{username.title()} User",
            "email": f"{username}@example.com",
            "dn": f"CN={username},DC=example,DC=com",
        }

    monkeypatch.setattr(web_server, "AUTH_MODE", "ldap")
    monkeypatch.setattr(web_server, "AUTH_ENABLED", True)
    monkeypatch.setattr(web_server, "ADMIN_USERS", {admin_identity.lower()})
    monkeypatch.setattr(web_server.ldap_auth, "authenticate", fake_authenticate)

    with TestClient(isolated_server.app) as test_client:
        before_login = test_client.get("/api/me")
        assert before_login.status_code == 200
        assert before_login.json()["is_admin"] is False

        login = test_client.post("/api/login", json={"username": "alice", "password": "ok"})
        assert login.status_code == 200
        assert login.json()["is_admin"] is expected

        me = test_client.get("/api/me")
        assert me.status_code == 200
        assert me.json()["authenticated"] is True
        assert me.json()["is_admin"] is expected


def test_ldap_login_requires_captcha_after_repeated_failures(isolated_server, monkeypatch):
    auth_calls = []

    def fake_authenticate(username, password):
        auth_calls.append((username, password))
        if password != "ok":
            raise web_server.ldap_auth.AuthError("bad credentials")
        return {
            "username": username,
            "display_name": f"{username} User",
            "email": f"{username}@example.com",
            "dn": f"CN={username},DC=example,DC=com",
        }

    monkeypatch.setattr(web_server, "AUTH_MODE", "ldap")
    monkeypatch.setattr(web_server, "AUTH_ENABLED", True)
    monkeypatch.setattr(web_server, "LOGIN_CAPTCHA_THRESHOLD", 5)
    monkeypatch.setattr(web_server.ldap_auth, "authenticate", fake_authenticate)

    with TestClient(isolated_server.app) as test_client:
        for _ in range(4):
            bad = test_client.post("/api/login", json={"username": "alice", "password": "bad"})
            assert bad.status_code == 401
            assert bad.json()["captcha_required"] is False

        fifth_bad = test_client.post("/api/login", json={"username": "alice", "password": "bad"})
        assert fifth_bad.status_code == 401
        fifth_payload = fifth_bad.json()
        assert fifth_payload["captcha_required"] is True
        assert fifth_payload["captcha_image"].startswith("data:image/svg+xml;base64,")
        assert len(auth_calls) == 5

        missing_captcha = test_client.post("/api/login", json={"username": "alice", "password": "ok"})
        assert missing_captcha.status_code == 400
        assert missing_captcha.json()["captcha_required"] is True
        assert len(auth_calls) == 5

        wrong_captcha = test_client.post(
            "/api/login",
            json={"username": "alice", "password": "ok", "captcha": "WRONG"},
        )
        assert wrong_captcha.status_code == 400
        assert wrong_captcha.json()["captcha_required"] is True
        assert len(auth_calls) == 5

        captcha_answer = next(iter(web_server.LOGIN_CAPTCHA_CHALLENGES.values()))["answer"]
        login = test_client.post(
            "/api/login",
            json={"username": "alice", "password": "ok", "captcha": captcha_answer},
        )
        assert login.status_code == 200
        assert login.json()["user"]["username"] == "alice"
        assert len(auth_calls) == 6
        assert web_server.LOGIN_FAILURES == {}
        assert web_server.LOGIN_CAPTCHA_CHALLENGES == {}


def test_job_share_converts_private_project_and_allows_other_users(isolated_server, monkeypatch):
    def fake_authenticate(username, password):
        return {
            "username": username,
            "display_name": f"{username} User",
            "email": f"{username}@example.com",
            "dn": f"CN={username},DC=example,DC=com",
        }

    monkeypatch.setattr(web_server, "AUTH_MODE", "ldap")
    monkeypatch.setattr(web_server, "AUTH_ENABLED", True)
    monkeypatch.setattr(web_server.ldap_auth, "authenticate", fake_authenticate)

    with TestClient(isolated_server.app) as test_client:
        assert test_client.post("/api/login", json={"username": "alice", "password": "ok"}).status_code == 200
        project = test_client.post("/api/projects", json={"name": "Private Project"}).json()

        async def insert_job():
            db = await web_db.get_db()
            try:
                await db.execute(
                    "INSERT INTO jobs(id, project_id, user_token, label, mode, status) VALUES(?,?,?,?,?,?)",
                    ("share-job", project["id"], "alice", "share me", "single", "done"),
                )
                await db.commit()
            finally:
                await db.close()

        asyncio.run(insert_job())

        share = test_client.post("/api/jobs/share-job/share")
        assert share.status_code == 200
        payload = share.json()
        assert payload["project_is_public"] is True
        assert payload["changed"] is True
        assert payload["url"].endswith("/#/job/share-job")

        assert test_client.post("/api/logout").status_code == 200
        assert test_client.post("/api/login", json={"username": "bob", "password": "ok"}).status_code == 200
        assert test_client.get("/api/jobs/share-job").status_code == 200


def test_ldap_bind_error_is_wrapped_as_auth_error(monkeypatch):
    class FakeLDAPException(Exception):
        pass

    class FakeConnection:
        def __init__(self, *args, **kwargs):
            raise FakeLDAPException("invalidCredentials")

    monkeypatch.setenv("LDAP_USER_DN_TEMPLATE", "CN={username},DC=example,DC=com")
    monkeypatch.setattr(web_server.ldap_auth, "_ldap_server", lambda: object())
    monkeypatch.setattr(
        web_server.ldap_auth,
        "_ldap_imports",
        lambda: (None, FakeConnection, None, None, lambda value: value, FakeLDAPException),
    )

    with pytest.raises(web_server.ldap_auth.AuthError, match="Invalid username or password"):
        web_server.ldap_auth.authenticate("alice", "bad")


def test_backup_script_creates_archive_and_manifest(client, isolated_server, tmp_path):
    backup_dir = tmp_path / "backups"
    manifest = web_backup.create_backup(isolated_server.STORAGE_DIR, str(backup_dir), retention_days=14)

    archive = Path(manifest["archive"])
    assert archive.exists()
    assert (backup_dir / "latest.json").exists()
    assert manifest["size_bytes"] == archive.stat().st_size
    assert len(manifest["sha256"]) == 64

    with tarfile.open(archive, "r:gz") as tar:
        assert "storage/jobs.db" in tar.getnames()


def test_project_crud_does_not_require_auth(client):
    created = client.post(
        "/api/projects",
        json={"name": "Local Project", "description": "single user"},
    )
    assert created.status_code == 201
    project = created.json()
    assert project["name"] == "Local Project"
    assert project["description"] == "single user"

    listed = client.get("/api/projects")
    assert listed.status_code == 200
    assert [p["id"] for p in listed.json()] == [project["id"]]

    updated = client.put(
        f"/api/projects/{project['id']}",
        json={"name": "Renamed", "description": ""},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Renamed"


def test_job_patch_does_not_require_auth(client):
    job_id = "job-local"

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status) VALUES(?,?,?,?)",
                (job_id, "before", "single", "done"),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    patched = client.patch(f"/api/jobs/{job_id}", json={"label": "after"})

    assert patched.status_code == 200
    assert patched.json()["label"] == "after"


def test_job_patch_can_pin_and_lists_pinned_first(client):
    async def insert_rows():
        db = await web_db.get_db()
        try:
            await db.execute("INSERT INTO projects(id, name) VALUES(?,?)", ("project-pin", "Pinned"))
            await db.executemany(
                """
                INSERT INTO jobs(id, project_id, label, mode, status, created_at, is_pinned)
                VALUES(?,?,?,?,?,?,?)
                """,
                [
                    ("old-job", "project-pin", "old", "single", "done", "2026-05-18 10:00:00", 0),
                    ("new-job", "project-pin", "new", "single", "done", "2026-05-18 12:00:00", 0),
                    ("pin-job", "project-pin", "pin", "single", "done", "2026-05-18 09:00:00", 0),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_rows())

    patched = client.patch("/api/jobs/pin-job", json={"is_pinned": True})
    assert patched.status_code == 200
    assert patched.json()["is_pinned"] == 1

    response = client.get("/api/job-groups/project-pin/jobs")
    assert response.status_code == 200
    assert [job["id"] for job in response.json()["data"][:3]] == ["pin-job", "new-job", "old-job"]


def test_project_restore_preserves_pinned_jobs(client):
    async def insert_rows():
        db = await web_db.get_db()
        try:
            await db.execute("INSERT INTO projects(id, name) VALUES(?,?)", ("project-restore", "Restore"))
            await db.execute(
                """
                INSERT INTO jobs(id, project_id, label, mode, status, is_pinned)
                VALUES(?,?,?,?,?,?)
                """,
                ("restore-job", "project-restore", "pin", "single", "done", 1),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_rows())

    assert client.delete("/api/projects/project-restore").status_code == 204
    restored = client.post("/api/deleted-projects/project-restore/restore")
    assert restored.status_code == 200

    response = client.get("/api/job-groups/project-restore/jobs")
    assert response.status_code == 200
    assert response.json()["data"][0]["is_pinned"] == 1


def test_job_groups_paginate_by_visible_groups(client):
    async def insert_rows():
        db = await web_db.get_db()
        try:
            await db.executemany(
                "INSERT INTO projects(id, name) VALUES(?,?)",
                [("project-b", "Beta"), ("project-a", "Alpha")],
            )
            await db.executemany(
                "INSERT INTO jobs(id, project_id, label, mode, status) VALUES(?,?,?,?,?)",
                [
                    ("job-a1", "project-a", "a1", "single", "done"),
                    ("job-a2", "project-a", "a2", "single", "done"),
                    ("job-b1", "project-b", "b1", "single", "done"),
                    ("job-none", None, "none", "single", "done"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_rows())

    first_page = client.get("/api/job-groups?limit=2&offset=0")
    assert first_page.status_code == 200
    assert first_page.json()["total"] == 3
    assert [group["id"] for group in first_page.json()["data"]] == ["project-a", "project-b"]
    assert [group["job_count"] for group in first_page.json()["data"]] == [2, 1]

    second_page = client.get("/api/job-groups?limit=2&offset=2")
    assert second_page.status_code == 200
    assert [group["id"] for group in second_page.json()["data"]] == ["__none__"]


def test_job_groups_search_returns_matching_visible_groups(client):
    async def insert_rows():
        db = await web_db.get_db()
        try:
            await db.executemany(
                "INSERT INTO projects(id, name) VALUES(?,?)",
                [("project-a", "Alpha"), ("project-b", "Beta")],
            )
            await db.executemany(
                "INSERT INTO jobs(id, project_id, label, mode, status, file_a_name) VALUES(?,?,?,?,?,?)",
                [
                    ("job-a1", "project-a", "baseline", "single", "done", "trace-a.json"),
                    ("job-b1", "project-b", "target needle", "single", "done", "trace-b.json"),
                    ("job-none", None, "needle ungrouped", "single", "done", "trace-none.json"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_rows())

    response = client.get("/api/job-groups?q=needle")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert [group["id"] for group in payload["data"]] == ["project-b", "__none__"]
    assert [group["job_count"] for group in payload["data"]] == [1, 1]


def test_group_jobs_load_lazily_with_search_and_pagination(client):
    async def insert_rows():
        db = await web_db.get_db()
        try:
            await db.execute("INSERT INTO projects(id, name) VALUES(?,?)", ("project-a", "Alpha"))
            await db.executemany(
                """
                INSERT INTO jobs(id, project_id, label, mode, status, file_a_name, created_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                [
                    ("job-a1", "project-a", "baseline", "single", "done", "trace-a.json", "2026-05-18 10:00:00"),
                    ("job-a2", "project-a", "needle two", "single", "done", "trace-b.json", "2026-05-18 11:00:00"),
                    ("job-a3", "project-a", "needle three", "single", "done", "trace-c.json", "2026-05-18 12:00:00"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_rows())

    first_page = client.get("/api/job-groups/project-a/jobs?q=needle&limit=1&offset=0")
    assert first_page.status_code == 200
    assert first_page.json()["total"] == 2
    assert [job["id"] for job in first_page.json()["data"]] == ["job-a3"]

    second_page = client.get("/api/job-groups/project-a/jobs?q=needle&limit=1&offset=1")
    assert second_page.status_code == 200
    assert [job["id"] for job in second_page.json()["data"]] == ["job-a2"]


def test_compare_candidates_have_independent_search_and_pagination(client):
    async def insert_rows():
        db = await web_db.get_db()
        try:
            await db.executemany(
                "INSERT INTO projects(id, name) VALUES(?,?)",
                [("project-a", "Alpha"), ("project-b", "Beta")],
            )
            await db.executemany(
                """
                INSERT INTO jobs(id, project_id, label, mode, status, file_a_name, created_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                [
                    ("job-a1", "project-a", "alpha one", "single", "done", "alpha-1.json", "2026-05-18 10:00:00"),
                    ("job-a2", "project-a", "alpha two", "single", "done", "alpha-2.json", "2026-05-18 11:00:00"),
                    ("job-b1", "project-b", "beta needle", "single", "done", "beta.json", "2026-05-18 12:00:00"),
                    ("job-cmp", "project-b", "needle compare", "compare", "done", "cmp.json", "2026-05-18 13:00:00"),
                    ("job-run", "project-b", "needle running", "single", "running", "run.json", "2026-05-18 14:00:00"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_rows())

    first_page = client.get("/api/compare-candidates?limit=2&offset=0")
    assert first_page.status_code == 200
    assert first_page.json()["total"] == 3
    assert len(first_page.json()["data"]) == 2

    search = client.get("/api/compare-candidates?q=needle")
    assert search.status_code == 200
    assert search.json()["total"] == 1
    assert [job["id"] for job in search.json()["data"]] == ["job-b1"]

    filtered = client.get("/api/compare-candidates?project_id=project-a")
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 2
    assert [job["id"] for job in filtered.json()["data"]] == ["job-a2", "job-a1"]


def test_bulk_job_actions_move_delete_files_and_delete_jobs(client, tmp_path):
    trace_a = tmp_path / "a.json"
    trace_b = tmp_path / "b.json"
    trace_a.write_text("{}")
    trace_b.write_text("{}")

    async def insert_rows():
        db = await web_db.get_db()
        try:
            await db.executemany(
                "INSERT INTO projects(id, name) VALUES(?,?)",
                [("project-a", "Alpha"), ("project-b", "Beta")],
            )
            await db.executemany(
                """
                INSERT INTO jobs(id, project_id, label, mode, status, file_a_path, file_a_exists)
                VALUES(?,?,?,?,?,?,?)
                """,
                [
                    ("job-a", "project-a", "a", "single", "done", str(trace_a), 1),
                    ("job-b", "project-a", "b", "single", "done", str(trace_b), 1),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_rows())

    moved = client.patch(
        "/api/jobs/bulk/project",
        json={"job_ids": ["job-a", "job-b"], "project_id": "project-b"},
    )
    assert moved.status_code == 200
    assert moved.json() == {"updated": 2}

    deleted_files = client.post(
        "/api/jobs/bulk/delete-files",
        json={"job_ids": ["job-a", "job-b"]},
    )
    assert deleted_files.status_code == 200
    assert deleted_files.json() == {"updated": 2, "files_deleted": 2}
    assert not trace_a.exists()
    assert not trace_b.exists()

    deleted_jobs = client.post(
        "/api/jobs/bulk/delete",
        json={"job_ids": ["job-a", "job-b"]},
    )
    assert deleted_jobs.status_code == 200
    assert deleted_jobs.json() == {"deleted": 2}

    async def fetch_rows():
        db = await web_db.get_db()
        try:
            rows = await (await db.execute("SELECT id FROM jobs")).fetchall()
            return [row["id"] for row in rows]
        finally:
            await db.close()

    assert asyncio.run(fetch_rows()) == []


def test_file_download_can_be_disabled(isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "ALLOW_FILE_DOWNLOAD", False)

    with TestClient(isolated_server.app) as test_client:
        r = test_client.get("/api/jobs/missing/files/a")

    assert r.status_code == 403
    assert r.json()["detail"] == "File download is disabled"


def test_code_execution_is_disabled_by_default(client):
    r = client.post(
        "/api/jobs/missing/run-triton-single",
        json={"code_path": "step_0_triton_codes/kernel.py"},
    )

    assert r.status_code == 403
    assert r.json()["detail"] == "Code execution is disabled"


def test_run_triton_single_reports_no_output_diagnostics(isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "ALLOW_CODE_EXECUTION", True)
    storage_dir = Path(isolated_server.STORAGE_DIR)
    job_id = "empty-output-job"
    code_dir = storage_dir / job_id / "results" / "step_0_triton_codes"
    code_dir.mkdir(parents=True)
    (code_dir / "kernel.py").write_text("# exits successfully without output\n", encoding="utf-8")

    async def seed_job():
        await web_db.init_db()
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status) VALUES(?,?,?,?)",
                (job_id, "done", "single", "done"),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(seed_job())

    with TestClient(isolated_server.app) as test_client:
        r = test_client.post(
            f"/api/jobs/{job_id}/run-triton-single",
            json={"code_path": "step_0_triton_codes/kernel.py"},
        )

    assert r.status_code == 200
    payload = r.json()
    assert payload["success"] is False
    assert "脚本执行成功，但没有输出可解析的结果" in payload["message"]
    assert "Command:" in payload["message"]
    assert "Code path:" in payload["message"]
    assert "stdout:\n<empty>" in payload["message"]
    assert "stderr:\n<empty>" in payload["message"]


def test_triton_code_paths_reject_sibling_prefix_traversal(isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "ALLOW_CODE_EXECUTION", True)
    storage_dir = Path(isolated_server.STORAGE_DIR)
    job_id = "job"
    sibling_dir = storage_dir / f"{job_id}-sibling"
    sibling_dir.mkdir(parents=True)
    (sibling_dir / "kernel.py").write_text("print('outside')\n")

    async def seed_job():
        await web_db.init_db()
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status, result_dir) VALUES(?,?,?,?,?)",
                (job_id, "done", "single", "done", str(storage_dir / job_id / "results")),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(seed_job())

    with TestClient(isolated_server.app) as test_client:
        run_resp = test_client.post(
            f"/api/jobs/{job_id}/run-triton-single",
            json={"code_path": "../job-sibling/kernel.py"},
        )

    assert run_resp.status_code == 400
    assert run_resp.json()["detail"] == "Invalid code_path"
    with pytest.raises(HTTPException) as exc_info:
        web_server._safe_child_path(
            str(storage_dir / job_id / "results"),
            "../job-sibling/kernel.py",
            "Invalid path",
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid path"


def test_startup_marks_interrupted_jobs_error(isolated_server):
    job_id = "interrupted-job"
    os.makedirs(Path(web_db.DB_PATH).parent, exist_ok=True)

    async def seed_db():
        await web_db.init_db()
        db = await aiosqlite.connect(web_db.DB_PATH)
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status) VALUES(?,?,?,?)",
                (job_id, "running", "single", "running"),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(seed_db())

    with TestClient(isolated_server.app):
        pass

    async def fetch_job():
        db = await web_db.get_db()
        try:
            row = await (
                await db.execute("SELECT status, error_msg FROM jobs WHERE id=?", (job_id,))
            ).fetchone()
            return dict(row)
        finally:
            await db.close()

    row = asyncio.run(fetch_job())
    assert row["status"] == "error"
    assert "Server restarted" in row["error_msg"]


def test_startup_enqueues_pending_jobs(isolated_server, monkeypatch):
    seen = []

    async def fake_run_analysis(job_id):
        seen.append(job_id)

    async def seed_db():
        await web_db.init_db()
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status) VALUES(?,?,?,?)",
                ("pending-job", "pending", "single", "pending"),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(seed_db())
    monkeypatch.setattr(web_server, "run_analysis", fake_run_analysis)

    with TestClient(isolated_server.app):
        for _ in range(50):
            if seen:
                break
            time.sleep(0.01)

    assert seen == ["pending-job"]


def test_compare_from_history_accepts_tar_gzip_sources(
    client,
    sample_trace_file_tar_gz,
    tmp_path,
):
    stored_a = tmp_path / "a.tar.gz"
    stored_b = tmp_path / "b.tar.gz"
    shutil.copyfile(sample_trace_file_tar_gz, stored_a)
    shutil.copyfile(sample_trace_file_tar_gz, stored_b)

    async def insert_jobs():
        db = await web_db.get_db()
        try:
            await db.executemany(
                """
                INSERT INTO jobs(
                    id, label, mode, status, file_a_name, file_a_gzip_path, file_a_exists
                ) VALUES(?,?,?,?,?,?,?)
                """,
                [
                    ("source-a", "a", "single", "done", "a.tar.gz", str(stored_a), 1),
                    ("source-b", "b", "single", "done", "b.tar.gz", str(stored_b), 1),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_jobs())

    created = client.post(
        "/api/jobs/compare",
        json={"job_id_a": "source-a", "job_id_b": "source-b"},
    )

    assert created.status_code == 201
    compare_job = created.json()
    assert compare_job["mode"] == "compare"
    assert compare_job["status"] in {"pending", "running", "done"}


def test_batch_compare_creates_jobs_from_baseline(client, sample_trace_file, monkeypatch):
    queued = []

    async def fake_enqueue(job_id):
        queued.append(job_id)

    monkeypatch.setattr(web_server, "enqueue_analysis_job", fake_enqueue)

    async def insert_jobs():
        db = await web_db.get_db()
        try:
            await db.executemany(
                """
                INSERT INTO jobs(
                    id, label, mode, status, file_a_name, file_a_path, file_a_exists
                ) VALUES(?,?,?,?,?,?,?)
                """,
                [
                    ("baseline-job", "baseline", "single", "done", "base.json", sample_trace_file, 1),
                    ("candidate-a", "candidate-a", "single", "done", "a.json", sample_trace_file, 1),
                    ("candidate-b", "candidate-b", "single", "done", "b.json", sample_trace_file, 1),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_jobs())

    response = client.post(
        "/api/jobs/batch-compare",
        json={
            "baseline_job_id": "baseline-job",
            "candidate_job_ids": ["candidate-a", "candidate-b"],
            "label_prefix": "batch",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["count"] == 2
    assert len(payload["data"]) == 2
    assert {job["source_job_a"] for job in payload["data"]} == {"baseline-job"}
    assert {job["source_job_b"] for job in payload["data"]} == {"candidate-a", "candidate-b"}
    assert all(job["mode"] == "compare" for job in payload["data"])
    assert all(job["label"].startswith("batch - baseline vs ") for job in payload["data"])
    assert queued == [job["id"] for job in payload["data"]]


def test_perfetto_json_download_extracts_tar_gzip_source(
    client,
    sample_trace_file_tar_gz,
    tmp_path,
):
    stored = tmp_path / "trace.tar.gz"
    shutil.copyfile(sample_trace_file_tar_gz, stored)

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status, file_a_name, file_a_gzip_path, file_a_exists
                ) VALUES(?,?,?,?,?,?,?)
                """,
                ("source-tar", "tar", "single", "done", "trace.tar.gz", str(stored), 1),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    r = client.get("/api/jobs/source-tar/files/a?format=json")

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["traceEvents"]


def test_json_download_decompresses_gzip_with_json_name(
    client,
    sample_trace_file_gz,
    tmp_path,
):
    stored = tmp_path / "trace.json.gz"
    shutil.copyfile(sample_trace_file_gz, stored)

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status, file_a_name, file_a_gzip_path, file_a_exists
                ) VALUES(?,?,?,?,?,?,?)
                """,
                ("compressed-json", "compressed", "single", "done", "trace.json", str(stored), 1),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    json_resp = client.get("/api/jobs/compressed-json/files/a?format=json")
    assert json_resp.status_code == 200
    assert json_resp.headers["content-type"].startswith("application/json")
    assert 'filename="trace.json"' in json_resp.headers["content-disposition"]
    assert json_resp.json()["traceEvents"]

    stored_resp = client.get("/api/jobs/compressed-json/files/a")
    assert stored_resp.status_code == 200
    assert stored_resp.headers["content-type"].startswith("application/gzip")
    assert 'filename="trace.json.gz"' in stored_resp.headers["content-disposition"]
    assert stored_resp.content.startswith(b"\x1f\x8b")


def test_zip_upload_is_normalized_to_json_gzip(
    sample_trace_file_zip,
    tmp_path,
):
    dest_json = tmp_path / "trace.json"
    gzip_path = [None]

    async def extract_upload():
        with open(sample_trace_file_zip, "rb") as f:
            upload = UploadFile(file=f, filename="trace.json.zip")
            await web_server.save_and_extract(upload, str(dest_json), gzip_path)

    asyncio.run(extract_upload())

    assert dest_json.exists()
    assert gzip_path[0] == str(dest_json) + ".gz"
    with open(dest_json, encoding="utf-8") as f:
        assert json.load(f)["traceEvents"]
    with gzip.open(gzip_path[0], "rt", encoding="utf-8") as f:
        assert json.load(f)["traceEvents"]


def test_trace_file_downloads_default_to_json_gzip_for_supported_formats(
    client,
    sample_trace_file,
    sample_trace_file_gz,
    sample_trace_file_tar_gz,
    sample_trace_file_zip,
    tmp_path,
):
    tgz_path = tmp_path / "trace.tgz"
    shutil.copyfile(sample_trace_file_tar_gz, tgz_path)

    async def insert_jobs():
        db = await web_db.get_db()
        try:
            await db.executemany(
                """
                INSERT INTO jobs(
                    id, label, mode, status,
                    file_a_name, file_a_path, file_a_gzip_path, file_a_exists
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        "download-json",
                        "json",
                        "single",
                        "done",
                        "trace.json",
                        sample_trace_file,
                        None,
                        1,
                    ),
                    (
                        "download-gzip",
                        "gzip",
                        "single",
                        "done",
                        "trace.json.gz",
                        None,
                        sample_trace_file_gz,
                        1,
                    ),
                    (
                        "download-zip",
                        "zip",
                        "single",
                        "done",
                        "trace.json.zip",
                        sample_trace_file_zip,
                        None,
                        1,
                    ),
                    (
                        "download-tar-gzip",
                        "tar-gzip",
                        "single",
                        "done",
                        "trace.tar.gz",
                        None,
                        sample_trace_file_tar_gz,
                        1,
                    ),
                    (
                        "download-tgz",
                        "tgz",
                        "single",
                        "done",
                        "trace.tgz",
                        None,
                        str(tgz_path),
                        1,
                    ),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_jobs())

    for job_id in (
        "download-json",
        "download-gzip",
        "download-zip",
        "download-tar-gzip",
        "download-tgz",
    ):
        json_resp = client.get(f"/api/jobs/{job_id}/files/a?format=json")
        assert json_resp.status_code == 200
        assert json_resp.headers["content-type"].startswith("application/json")
        assert 'filename="trace.json"' in json_resp.headers["content-disposition"]
        assert json_resp.json()["traceEvents"]

        gzip_resp = client.get(f"/api/jobs/{job_id}/files/a")
        assert gzip_resp.status_code == 200
        assert gzip_resp.headers["content-type"].startswith("application/gzip")
        assert 'filename="trace.json.gz"' in gzip_resp.headers["content-disposition"]
        assert json.loads(gzip.decompress(gzip_resp.content))["traceEvents"]


def test_uploaded_trace_formats_download_as_json_gzip(
    client,
    sample_trace_file,
    sample_trace_file_gz,
    sample_trace_file_zip,
):
    uploads = [
        ("upload-json", sample_trace_file, "trace.json"),
        ("upload-gzip", sample_trace_file_gz, "trace.json.gz"),
        ("upload-zip", sample_trace_file_zip, "trace.json.zip"),
    ]

    for label, path, filename in uploads:
        with open(path, "rb") as f:
            created = client.post(
                "/api/jobs",
                data={"label": label},
                files={"file_a": (filename, f, "application/octet-stream")},
            )
        assert created.status_code == 201
        job_id = created.json()["id"]

        asyncio.run(web_server.run_analysis(job_id))
        job_resp = client.get(f"/api/jobs/{job_id}")
        assert job_resp.status_code == 200
        job = job_resp.json()
        assert job["status"] == "done", job.get("error_msg")

        json_resp = client.get(f"/api/jobs/{job_id}/files/a?format=json")
        assert json_resp.status_code == 200
        assert json_resp.headers["content-type"].startswith("application/json")
        assert 'filename="trace.json"' in json_resp.headers["content-disposition"]
        assert json_resp.json()["traceEvents"]

        gzip_resp = client.get(f"/api/jobs/{job_id}/files/a")
        assert gzip_resp.status_code == 200
        assert gzip_resp.headers["content-type"].startswith("application/gzip")
        assert 'filename="trace.json.gz"' in gzip_resp.headers["content-disposition"]
        assert json.loads(gzip.decompress(gzip_resp.content))["traceEvents"]


def test_direct_two_file_upload_creates_compare_job(client, sample_trace_file, sample_trace_file_gz):
    with open(sample_trace_file, "rb") as file_a, open(sample_trace_file_gz, "rb") as file_b:
        created = client.post(
            "/api/jobs",
            files={
                "file_a": ("base.json", file_a, "application/octet-stream"),
                "file_b": ("target.json.gz", file_b, "application/octet-stream"),
            },
        )

    assert created.status_code == 201
    job = created.json()
    assert job["mode"] == "compare"
    assert job["label"] == "base.json vs target.json.gz"
    assert job["file_a_name"] == "base.json"
    assert job["file_b_name"] == "target.json.gz"
    assert Path(web_server.job_dir(job["id"]), "trace_a.json").exists()
    assert Path(web_server.job_dir(job["id"]), "trace_b.json").exists()


def test_done_job_exposes_perfetto_context(client, sample_trace_file):
    result_dir = Path(web_server.result_dir("done-job"))
    result_dir.mkdir(parents=True)
    context = {
        "a": {
            "step": 0,
            "step_name": "ProfilerStep#0",
            "ts_ns": 1000000000,
            "dur_ns": 100000000,
            "vis_start_ns": 990000000,
            "vis_end_ns": 1110000000,
        }
    }
    (result_dir / "perfetto_context.json").write_text(json.dumps(context))

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status, file_a_name, file_a_path, result_dir
                ) VALUES(?,?,?,?,?,?,?)
                """,
                ("done-job", "done", "single", "done", "trace.json", sample_trace_file, str(result_dir)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    r = client.get("/api/jobs/done-job")

    assert r.status_code == 200
    assert r.json()["perfetto_context"] == context
    assert (result_dir / "perfetto_context.json").exists()


def test_done_job_lists_result_files_and_paginates_tables(client):
    result_dir = Path(web_server.result_dir("table-job"))
    result_dir.mkdir(parents=True)
    (result_dir / "all_kernels_avg.csv").write_text(
        "kernel_name,count_pct,avg_dur_ms,dur_pct,family\n"
        "slow_kernel,50.0%,30,60.0%,gemm\n"
        "medium_kernel,30.0%,20,30.0%,gemm\n"
        "fast_kernel,20.0%,10,10.0%,other\n"
    )

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status, result_dir) VALUES(?,?,?,?,?)",
                ("table-job", "table", "single", "done", str(result_dir)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    detail = client.get("/api/jobs/table-job")
    assert detail.status_code == 200
    assert "results" not in detail.json()
    assert detail.json()["result_files"]["all_kernels_avg.csv"]["fields"] == [
        "kernel_name", "avg_dur_ms", "family"
    ]

    page = client.get(
        "/api/jobs/table-job/results/all_kernels_avg.csv",
        params={"limit": 2, "offset": 0, "sort_col": "avg_dur_ms", "sort_dir": "desc"},
    )
    assert page.status_code == 200
    payload = page.json()
    assert payload["total"] == 3
    assert payload["filtered_total"] == 3
    assert [row["kernel_name"] for row in payload["rows"]] == ["slow_kernel", "medium_kernel"]
    assert "dur_pct" not in payload["fields"]
    assert "count_pct" not in payload["rows"][0]

    filtered = client.get(
        "/api/jobs/table-job/results/all_kernels_avg.csv",
        params={"q": "fast", "limit": 10},
    )
    assert filtered.status_code == 200
    assert filtered.json()["filtered_total"] == 1
    assert filtered.json()["rows"][0]["kernel_name"] == "fast_kernel"

    old_percent_filter = client.get(
        "/api/jobs/table-job/results/all_kernels_avg.csv",
        params={"filters": json.dumps({"dur_pct": "60"}), "limit": 10},
    )
    assert old_percent_filter.status_code == 200
    assert old_percent_filter.json()["filtered_total"] == 3


def test_job_report_download_includes_markdown_summary(client):
    result_dir = Path(web_server.result_dir("report-job"))
    result_dir.mkdir(parents=True)
    (result_dir / "kernel_types_avg.csv").write_text(
        "type,avg_dur_ms,dur_pct\n"
        "gemm,12.3,70%\n"
        "attention,4.5,30%\n"
    )

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(id, label, mode, status, file_a_name, console_out, result_dir)
                VALUES(?,?,?,?,?,?,?)
                """,
                ("report-job", "report", "single", "done", "trace.json", "Top kernels\nok", str(result_dir)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    response = client.get("/api/jobs/report-job/report.md")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    text = response.text
    assert "# report" in text
    assert "## 任务信息" in text
    assert "Top kernels" in text
    assert "| type | avg_dur_ms |" in text
    assert "dur_pct" not in text


def test_all_kernels_cmp_without_family_exposes_virtual_family(client):
    result_dir = Path(web_server.result_dir("cmp-table-job"))
    result_dir.mkdir(parents=True)
    (result_dir / "all_kernels_cmp.csv").write_text(
        "kernel_name,avg_dur_ms_A,avg_dur_ms_B,delta_dur_ms\n"
        "gemm_kernel,1,3,2\n"
        "triton_poi_kernel,4,1,-3\n"
    )

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status, result_dir) VALUES(?,?,?,?,?)",
                ("cmp-table-job", "cmp", "compare", "done", str(result_dir)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    detail = client.get("/api/jobs/cmp-table-job")
    assert detail.status_code == 200
    assert detail.json()["result_files"]["all_kernels_cmp.csv"]["fields"] == [
        "kernel_name", "family", "avg_dur_ms_A", "avg_dur_ms_B", "delta_dur_ms"
    ]

    filtered = client.get(
        "/api/jobs/cmp-table-job/results/all_kernels_cmp.csv",
        params={
            "filters": json.dumps({"family": "gemm"}),
            "filter_ops": json.dumps({"family": "~"}),
            "limit": 10,
        },
    )
    assert filtered.status_code == 200
    assert filtered.json()["filtered_total"] == 1
    assert filtered.json()["rows"][0]["kernel_name"] == "gemm_kernel"
    assert filtered.json()["rows"][0]["family"] == "gemm"


def test_result_table_can_return_more_than_default_page_cap(client):
    result_dir = Path(web_server.result_dir("large-table-job"))
    result_dir.mkdir(parents=True)
    rows = ["kernel_name,avg_dur_ms,family"]
    rows.extend(f"kernel_{i},{i},other" for i in range(1005))
    (result_dir / "all_kernels_avg.csv").write_text("\n".join(rows) + "\n")

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status, result_dir) VALUES(?,?,?,?,?)",
                ("large-table-job", "large", "single", "done", str(result_dir)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    page = client.get(
        "/api/jobs/large-table-job/results/all_kernels_avg.csv",
        params={"limit": 1005, "offset": 0},
    )

    assert page.status_code == 200
    payload = page.json()
    assert payload["limit"] == 1005
    assert payload["total"] == 1005
    assert len(payload["rows"]) == 1005


def test_done_job_without_perfetto_context_still_loads(client, sample_trace_file):
    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status, file_a_name, file_a_path
                ) VALUES(?,?,?,?,?,?)
                """,
                ("old-job", "old", "single", "done", "trace.json", sample_trace_file),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    r = client.get("/api/jobs/old-job")

    assert r.status_code == 200
    assert r.json()["id"] == "old-job"
    assert r.json()["perfetto_context"] == {}


def test_compare_job_exposes_source_summaries_and_delete_impact(client, sample_trace_file):
    async def insert_jobs():
        db = await web_db.get_db()
        try:
            await db.execute("INSERT INTO projects(id, name) VALUES(?,?)", ("project-a", "Alpha"))
            await db.executemany(
                """
                INSERT INTO jobs(
                    id, project_id, label, mode, status, file_a_name, file_a_path,
                    source_job_a, source_job_b
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                [
                    ("source-a", "project-a", "base", "single", "done", "a.json", sample_trace_file, None, None),
                    ("source-b", None, "target", "single", "done", "b.json", sample_trace_file, None, None),
                    ("compare-job", None, "cmp", "compare", "done", "a.json", None, "source-a", "source-b"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_jobs())

    detail = client.get("/api/jobs/compare-job")
    assert detail.status_code == 200
    assert detail.json()["compare_sources"]["a"]["label"] == "base"
    assert detail.json()["compare_sources"]["a"]["project_name"] == "Alpha"
    assert detail.json()["compare_sources"]["b"]["label"] == "target"

    impact = client.get("/api/jobs/source-a/files/a/delete-impact")
    assert impact.status_code == 200
    assert impact.json()["count"] == 1
    assert impact.json()["dependent_compare_jobs"][0]["id"] == "compare-job"


def test_rerun_swapped_direct_compare_copies_reversed_files(
    client,
    sample_trace_file,
    tmp_path,
    monkeypatch,
):
    enqueued = []

    async def fake_enqueue(job_id):
        enqueued.append(job_id)

    trace_b = tmp_path / "b.json"
    shutil.copyfile(sample_trace_file, trace_b)
    monkeypatch.setattr(web_server, "enqueue_analysis_job", fake_enqueue)

    async def insert_jobs():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status,
                    file_a_name, file_a_path,
                    file_b_name, file_b_path
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    "direct-compare", "a vs b", "compare", "done",
                    "a.json", sample_trace_file,
                    "b.json", str(trace_b),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_jobs())

    response = client.post("/api/jobs/direct-compare/rerun-swapped")

    assert response.status_code == 201
    job = response.json()
    assert job["mode"] == "compare"
    assert job["label"] == "b.json vs a.json"
    assert job["file_a_name"] == "b.json"
    assert job["file_b_name"] == "a.json"
    assert job["source_job_a"] is None
    assert job["source_job_b"] is None
    assert Path(job["file_a_path"]).exists()
    assert Path(job["file_b_path"]).exists()
    assert enqueued == [job["id"]]


def test_rerun_swapped_source_compare_reverses_sources(
    client,
    sample_trace_file,
    monkeypatch,
):
    enqueued = []

    async def fake_enqueue(job_id):
        enqueued.append(job_id)

    monkeypatch.setattr(web_server, "enqueue_analysis_job", fake_enqueue)

    async def insert_jobs():
        db = await web_db.get_db()
        try:
            await db.executemany(
                """
                INSERT INTO jobs(
                    id, label, mode, status,
                    file_a_name, file_a_path,
                    source_job_a, source_job_b
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                [
                    ("source-a", "base", "single", "done", "a.json", sample_trace_file, None, None),
                    ("source-b", "target", "single", "done", "b.json", sample_trace_file, None, None),
                    ("compare-job", "base vs target", "compare", "done", "a.json", None, "source-a", "source-b"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_jobs())

    response = client.post("/api/jobs/compare-job/rerun-swapped")

    assert response.status_code == 201
    job = response.json()
    assert job["label"] == "target vs base"
    assert job["file_a_name"] == "b.json"
    assert job["file_b_name"] == "a.json"
    assert job["source_job_a"] == "source-b"
    assert job["source_job_b"] == "source-a"
    assert enqueued == [job["id"]]


def test_storage_summary(client, tmp_path):
    trace_a = tmp_path / "a.json"
    trace_b = tmp_path / "b.json"
    trace_a.write_text("a" * 10)
    trace_b.write_text("b" * 20)
    result_dir = Path(web_server.result_dir("done-job"))
    result_dir.mkdir(parents=True)
    (result_dir / "out.csv").write_text("x" * 5)

    async def insert_jobs():
        db = await web_db.get_db()
        try:
            await db.execute("INSERT INTO projects(id, name) VALUES(?,?)", ("project-a", "Alpha"))
            await db.executemany(
                """
                INSERT INTO jobs(
                    id, project_id, label, mode, status, file_a_path, result_dir,
                    source_job_a, source_job_b
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                [
                    ("done-job", "project-a", "done", "single", "done", str(trace_a), str(result_dir), None, None),
                    ("running-job", "project-a", "running", "single", "running", str(trace_b), "", None, None),
                    ("error-job", None, "error", "single", "error", None, "", None, None),
                    ("compare-job", None, "cmp", "compare", "done", None, "", "done-job", "running-job"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_jobs())

    storage = client.get("/api/storage/summary")
    assert storage.status_code == 200
    payload = storage.json()
    assert payload["totals"]["original_trace_bytes"] == 30
    assert payload["projects"][0]["name"] == "Alpha"
    done_job = next(job for job in payload["jobs"] if job["id"] == "done-job")
    assert done_job["used_by_compare_count"] == 1


def test_storage_summary_uses_cached_sizes(client):
    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status,
                    owned_bytes, result_bytes, original_trace_bytes
                ) VALUES(?,?,?,?,?,?,?)
                """,
                ("cached-job", "cached", "single", "done", 123, 45, 78),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    storage = client.get("/api/storage/summary")
    assert storage.status_code == 200
    payload = storage.json()
    assert payload["totals"]["owned_bytes"] == 123
    assert payload["totals"]["result_bytes"] == 45
    assert payload["totals"]["original_trace_bytes"] == 78
    assert payload["jobs"][0]["id"] == "cached-job"
