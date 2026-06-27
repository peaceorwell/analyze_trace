import asyncio
import gzip
import json
import logging
import os
import shlex
import socket
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
    monkeypatch.setattr(web_server, "MAX_UPLOAD_BYTES", 0)
    monkeypatch.setattr(web_server, "MIN_STORAGE_FREE_BYTES", 0)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", False)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_COMMAND_TEMPLATE", "")
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_EXTRA_ARGS", "")
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_MODEL", "Claude Code default")
    monkeypatch.setattr(web_server, "AUTH_MODE", "none")
    monkeypatch.setattr(web_server, "AUTH_ENABLED", False)
    monkeypatch.setattr(web_server, "ADMIN_USERS", set())
    web_server.LOGIN_FAILURES.clear()
    web_server.LOGIN_CAPTCHA_CHALLENGES.clear()
    web_server.ai_analysis_workers.clear()
    web_server.ai_analysis_tasks.clear()
    web_server.ai_analysis_queued_jobs.clear()
    web_server.ai_analysis_locks.clear()
    while not web_server.ai_analysis_queue.empty():
        web_server.ai_analysis_queue.get_nowait()
        web_server.ai_analysis_queue.task_done()
    return web_server


@pytest.fixture
def client(isolated_server):
    with TestClient(isolated_server.app) as test_client:
        yield test_client


def test_config_reports_local_execution_flags(client):
    r = client.get("/api/config")

    assert r.status_code == 200
    assert r.json() == {
        "version": "0.3.24",
        "auth_mode": "none",
        "auth_required": False,
        "allow_file_download": True,
        "allow_code_execution": False,
        "claude_analysis_enabled": False,
    }


def test_collect_results_hides_empty_pytorch_csvs_for_tensorflow_trace(isolated_server):
    rdir = Path(web_server.result_dir("tf-result-job"))
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "all_kernels_avg.csv").write_text(
        "kernel_name,family,avg_count,avg_dur_ms,avg_us_per_call\n"
        "void MLUMatMulGemm,gemm,1,0.2,200\n"
    )
    (rdir / "kernel_types_avg.csv").write_text("type,avg_count,avg_dur_ms\ngemm,1,0.2\n")
    (rdir / "tf_ops_avg.csv").write_text("op_name,avg_count,avg_dur_ms\ndense/MatMul:MatMul,1,0.1\n")
    (rdir / "triton_kernels_avg.csv").write_text(
        "kernel_name,avg_count,avg_dur_ms,avg_io_gb,avg_io_efficiency_gbps\n"
    )
    (rdir / "aten_ops_avg.csv").write_text("op_name,avg_count,avg_dur_ms\n")
    (rdir / "cncl_ops_avg.csv").write_text("op_name,avg_count,avg_dur_ms\n")
    (rdir / "non_triton_kernel_efficiency_avg.csv").write_text(
        "kernel_name,family,operator,input_dims,input_types,input_strides,concrete_inputs,"
        "operator_details,avg_count,avg_dur_ms,avg_us_per_call,avg_compute_efficiency,"
        "avg_io_efficiency,avg_op_efficiency\n"
    )

    results = web_server.collect_results("tf-result-job")

    assert "all_kernels_avg.csv" in results
    assert "kernel_types_avg.csv" in results
    assert "tf_ops_avg.csv" in results
    assert "triton_kernels_avg.csv" not in results
    assert "aten_ops_avg.csv" not in results
    assert "cncl_ops_avg.csv" not in results
    assert "non_triton_kernel_efficiency_avg.csv" not in results


def test_json_log_formatter_uses_configured_timezone():
    formatter = web_server.JsonLogFormatter(web_server._resolve_log_timezone("Asia/Shanghai"))
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.created = 0

    payload = json.loads(formatter.format(record))

    assert payload["time"].startswith("1970-01-01T08:00:00")
    assert payload["time"].endswith("+08:00")


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


def test_ai_analysis_report_download_supports_unicode_filename(client, isolated_server):
    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status) VALUES(?,?,?,?)",
                ("ai-unicode-download-job", "中文报告", "single", "done"),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    analysis_dir = Path(web_server.ai_analysis_dir("ai-unicode-download-job"))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / web_server.AI_ANALYSIS_REPORT_FILE).write_text("# 中文报告\n", encoding="utf-8")

    r = client.get("/api/jobs/ai-unicode-download-job/ai-analysis/report.md")

    assert r.status_code == 200
    assert r.text == "# 中文报告\n"
    disposition = r.headers["content-disposition"]
    assert 'filename="' in disposition
    assert "filename*=UTF-8''" in disposition
    disposition.encode("latin-1")


def test_upload_limit_cleans_partial_job_directory(client, sample_trace_file, monkeypatch):
    monkeypatch.setattr(web_server, "MAX_UPLOAD_BYTES", 1)

    with open(sample_trace_file, "rb") as trace:
        response = client.post(
            "/api/jobs",
            files={"file_a": ("trace.json", trace, "application/json")},
        )

    assert response.status_code == 413
    storage_dir = Path(web_server.STORAGE_DIR)
    assert not list(storage_dir.glob("*/trace_a.json"))


def test_ai_analysis_rejects_duplicate_and_delete_while_active(
    client,
    tmp_path,
    monkeypatch,
):
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
                ("ai-active-job", "AI active", "single", "done", trace_path.name, str(trace_path)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.result_dir("ai-active-job")).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        _fake_claude_template(
            tmp_path,
            """
            import time
            time.sleep(0.5)
            pathlib.Path(os.environ['TRACE_AI_REPORT_PATH']).write_text('# Slow Report\\n', encoding='utf-8')
            print('# AI OK')
            """,
        ),
    )

    first = client.post("/api/jobs/ai-active-job/ai-analysis", json={})
    second = client.post("/api/jobs/ai-active-job/ai-analysis", json={"force": True})
    delete_response = client.delete("/api/jobs/ai-active-job")

    assert first.status_code == 202
    assert second.status_code == 409
    assert delete_response.status_code == 409

    payload = {}
    for _ in range(100):
        payload = client.get("/api/jobs/ai-active-job/ai-analysis").json()
        if payload["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)

    assert payload["status"] == "done"


def test_ai_report_normalizes_flat_conclusion_evidence_advice_list(isolated_server):
    raw = """
# AI 性能分析报告

## 结论概览

- 结论：设备利用率低，compute gap 是主要瓶颈。
- 证据：gap_summary.py 显示 gap 占比 38%。
- 建议：优先用 MLU Graph 捕获稳定序列。
- 结论：小 kernel 启动开销偏高。
- 证据：867 个 kernel 平均 0.016 ms。
- 建议：合并 LayerNorm 与激活相关小 kernel。
""".lstrip()

    normalized = isolated_server._normalize_ai_report_markdown(raw)

    assert "### 发现 1：设备利用率低，compute gap 是主要瓶颈" in normalized
    assert "**结论：** 设备利用率低，compute gap 是主要瓶颈。" in normalized
    assert "**证据：** gap_summary.py 显示 gap 占比 38%。" in normalized
    assert "**建议：** 优先用 MLU Graph 捕获稳定序列。" in normalized
    assert "### 发现 2：小 kernel 启动开销偏高" in normalized
    assert "- 证据：" not in normalized


def test_ai_report_injects_triton_code_optimization_section(isolated_server):
    jid = "ai-triton-section-job"
    analysis_dir = Path(isolated_server.ai_analysis_dir(jid))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "triton_code_optimization.json").write_text(
        json.dumps(
            {
                "has_findings": True,
                "final_report_guidance": {
                    "summary_cn": "扫描 2 个 Triton output_code，1 个 kernel 存在代码级候选。",
                    "candidates": [
                        {
                            "kernel_name": "triton_x",
                            "file": "triton_x.py",
                            "total_ms": 1.23,
                            "bandwidth_utilization": 0.4,
                            "estimated_profile": {
                                "summary": "IO 12.00 MB / 9.76 GB/s；计算 1.20 Gops / 975.61 GOPS"
                            },
                            "strategies": ["div-to-mul"],
                            "evidence": "发现张量除法；发现 dtype 往返转换。",
                            "recommendation": "验证 reciprocal + multiply; 消除重复 dtype 转换。",
                        },
                        {
                            "kernel_name": "triton_y",
                            "file": "triton_y.py",
                            "total_ms": 0.45,
                            "bandwidth_utilization": 0.25,
                            "estimated_profile": {
                                "summary": "IO 2.00 MB / 4.44 GB/s；计算 0.10 Gops / 222.22 GOPS"
                            },
                            "strategies": ["bulk-io-opt"],
                            "evidence": "离散 load 较多",
                            "recommendation": "验证连续 bulk IO",
                        },
                    ],
                    "required_table_md": "\n".join(
                        [
                            "### Triton Kernel 代码优化候选",
                            "",
                            "| Kernel | 代码文件 | 耗时 | BW 利用率 | 计算速率估算 | 优化方向与建议 |",
                            "|---|---|---:|---:|---|---|",
                            "| `triton_x` | `triton_x.py` | 1.23 ms | 40.0% | 计算 1.20 Gops / 975.61 GOPS | 方向：div-to-mul<br>验证 reciprocal + multiply |",
                        ]
                    ),
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raw = """
# AI 性能分析报告

## 优先行动

### Triton Kernel 代码优化候选

| old | old |
|---|---|
| stale | stale |

---

## Triton Kernel 代码优化

| Kernel | 代码文件 | 耗时 | BW 利用率 | 计算速率估算 | 优化方向与建议 |
|---|---|---:|---:|---|---|
| `old_top_level` | `old.py` | 0.01 ms | 1.0% | old | old |

## 不确定性与下一步

- next

## 产物

- artifact
""".lstrip()

    finalized = isolated_server._finalize_ai_report_markdown(jid, raw)

    assert "## Triton Kernel 代码优化\n" in finalized
    assert finalized.index("## Triton Kernel 代码优化") < finalized.index("## 不确定性与下一步")
    assert "### Triton Kernel 代码优化候选" not in finalized
    assert "`triton_x`" in finalized
    assert "`triton_y`" in finalized
    assert "计算速率估算" in finalized
    assert "计算 1.20 Gops / 975.61 GOPS" in finalized
    assert "IO 12.00 MB / 9.76 GB/s" not in finalized
    assert "方向：div-to-mul" in finalized
    assert "• 验证 reciprocal + multiply<br>• 消除重复 dtype 转换" in finalized
    assert "| Kernel | 代码文件 | 耗时 | BW 利用率 | 主要方向 | 证据 | 建议 |" not in finalized
    assert "old_top_level" not in finalized
    assert "stale" not in finalized


def test_ai_report_injects_triton_code_section_from_legacy_kernels(isolated_server):
    jid = "ai-triton-section-legacy-job"
    analysis_dir = Path(isolated_server.ai_analysis_dir(jid))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "triton_code_optimization.json").write_text(
        json.dumps(
            {
                "has_findings": True,
                "summary": {
                    "scanned_files": 2,
                    "kernels_with_findings": 1,
                    "finding_count": 2,
                    "top_strategies": [
                        {"strategy": "bulk-io-opt", "count": 1},
                        {"strategy": "canonicalize", "count": 1},
                    ],
                },
                "final_report_guidance": {},
                "kernels": [
                    {
                        "kernel_name": "triton_poi_fused_x",
                        "file": "/tmp/triton_output_code_00_triton_poi_fused_x.txt",
                        "total_ms": 3.2,
                        "bandwidth_utilization": 0.25,
                        "estimated_profile": {
                            "summary": "IO 8.00 MB / 2.50 GB/s；计算 0.80 Gops / 250.00 GOPS"
                        },
                        "priority": "high",
                        "findings": [
                            {
                                "strategy": "bulk-io-opt",
                                "evidence": "tl.load x6, tl.store x1，可能存在碎片化访存。",
                                "recommendation": "优先改成连续 bulk IO。",
                            },
                            {
                                "strategy": "canonicalize / libdevice-opt",
                                "evidence": "发现 .to(tl.*) 转换 x8。",
                                "recommendation": "消除重复 dtype 往返转换。",
                            },
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raw = """
# AI 性能分析报告

## 优先行动

- action

## Triton Kernel 代码优化

> stale empty section

## 不确定性与下一步

- next
""".lstrip()

    finalized = isolated_server._finalize_ai_report_markdown(jid, raw)

    assert "## Triton Kernel 代码优化\n" in finalized
    assert "扫描 2 个 Triton output_code，1 个 kernel 存在代码级候选，共 2 条优化信号" in finalized
    assert "`triton_poi_fused_x`" in finalized
    assert "`triton_output_code_00_triton_poi_fused_x.txt`" in finalized
    assert "3.20 ms" in finalized
    assert "25.0%" in finalized
    assert "计算 0.80 Gops / 250.00 GOPS" in finalized
    assert "IO 8.00 MB / 2.50 GB/s" not in finalized
    assert "方向：bulk-io-opt, canonicalize, libdevice-opt" in finalized
    assert "stale empty section" not in finalized


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
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "x-project: torch_mlu"
    settings = json.loads((analysis_dir / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert settings["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "x-project: torch_mlu"


def test_claude_env_preserves_existing_custom_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "TRACE_CLAUDE_CUSTOM_HEADERS", "")
    monkeypatch.setattr(web_server, "CLAUDE_CUSTOM_HEADERS", "x-project: torch_mlu")
    analysis_dir = tmp_path / "analysis"

    env = web_server._build_claude_env(
        {"HOME": str(tmp_path), "ANTHROPIC_CUSTOM_HEADERS": "x-project: existing"},
        str(analysis_dir),
    )

    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "x-project: existing"
    settings = json.loads((analysis_dir / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert settings["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "x-project: existing"


def test_claude_env_trace_custom_headers_override_base_env(tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "TRACE_CLAUDE_CUSTOM_HEADERS", "x-project: override")
    monkeypatch.setattr(web_server, "CLAUDE_CUSTOM_HEADERS", "x-project: override")
    analysis_dir = tmp_path / "analysis"

    env = web_server._build_claude_env(
        {"HOME": str(tmp_path), "ANTHROPIC_CUSTOM_HEADERS": "x-project: existing"},
        str(analysis_dir),
    )

    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "x-project: override"
    settings = json.loads((analysis_dir / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert settings["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "x-project: override"


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
        if payload["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)

    assert payload["status"] == "done"
    assert payload["progress"] == 100
    assert payload["duration_ms"] >= 0
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
    assert detail["ai_analysis"]["duration_ms"] >= 0


def test_ai_analysis_keeps_report_versions(client, isolated_server, tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.pt.trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as f:
        f.write("{}")

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(id, user_token, label, mode, status, file_a_name, file_a_gzip_path)
                VALUES(?,?,?,?,?,?,?)
                """,
                ("ai-version-job", "owner", "AI version job", "single", "done", trace_path.name, str(trace_path)),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.result_dir("ai-version-job")).mkdir(parents=True, exist_ok=True)
    counter_path = tmp_path / "ai_counter.txt"

    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_MODEL", "claude-test-model")
    user_prompt = "请重点关注 attention kernel"
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        _fake_claude_template(
            tmp_path,
            f"""
            if '只回复一行 OK' in prompt:
                print('OK')
            else:
                counter = pathlib.Path({str(counter_path)!r})
                value = int(counter.read_text(encoding='utf-8')) + 1 if counter.exists() else 1
                counter.write_text(str(value), encoding='utf-8')
                prompt_seen = {user_prompt!r} in prompt
                report = f'# Report {{value}}\\n\\nGenerated version {{value}}\\n\\nprompt_seen={{prompt_seen}}\\n'
                pathlib.Path(os.environ['TRACE_AI_REPORT_PATH']).write_text(report, encoding='utf-8')
                print(report)
            """,
        ),
    )

    first = client.post(
        "/api/jobs/ai-version-job/ai-analysis",
        json={},
        headers={"X-Remote-User": "runner"},
    )
    assert first.status_code == 202
    first_payload = {}
    for _ in range(80):
        first_payload = client.get("/api/jobs/ai-version-job/ai-analysis").json()
        if first_payload["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)
    assert first_payload["status"] == "done"
    assert "# Report 1" in first_payload["content"]

    second = client.post(
        "/api/jobs/ai-version-job/ai-analysis",
        json={"force": True, "prompt": user_prompt},
        headers={"X-Remote-User": "runner"},
    )
    assert second.status_code == 202
    latest = {}
    for _ in range(80):
        latest = client.get("/api/jobs/ai-version-job/ai-analysis").json()
        if latest["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)

    assert latest["status"] == "done"
    assert "# Report 2" in latest["content"]
    assert "prompt_seen=True" in latest["content"]
    assert len(latest["versions"]) == 2
    assert latest["selected_version_id"] == latest["versions"][0]["id"]
    assert latest["versions"][0]["model"] == "claude-test-model"
    assert latest["versions"][0]["user_prompt"] == user_prompt
    assert latest["versions"][0]["generated_at"]
    assert latest["versions"][0]["trigger_user_token"] == "runner"

    old_version = latest["versions"][1]
    old_response = client.get(f"/api/jobs/ai-version-job/ai-analysis?version_id={old_version['id']}")
    assert old_response.status_code == 200
    old_payload = old_response.json()
    assert old_payload["selected_version_id"] == old_version["id"]
    assert "# Report 1" in old_payload["content"]

    download = client.get(f"/api/jobs/ai-version-job/ai-analysis/report.md?version_id={old_version['id']}")
    assert download.status_code == 200
    assert "# Report 1" in download.text


def test_ai_analysis_completion_sends_email_with_result_link(client, isolated_server, tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.pt.trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as f:
        f.write("{}")

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(id, user_token, label, mode, status, file_a_name, file_a_gzip_path)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    "ai-mail-job",
                    "owner",
                    "AI mail job",
                    "single",
                    "done",
                    "trace.pt.trace.json.gz",
                    str(trace_path),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.result_dir("ai-mail-job")).mkdir(parents=True, exist_ok=True)

    sent = []
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(web_server, "PUBLIC_BASE_URL", "http://trace.example")
    monkeypatch.setattr(web_server, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(web_server, "SENDMAIL_COMMAND", "")
    monkeypatch.setattr(
        web_server,
        "_send_email_sync",
        lambda recipients, subject, body: sent.append({
            "recipients": recipients,
            "subject": subject,
            "body": body,
        }),
    )
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        _fake_claude_template(
            tmp_path,
            """
            pathlib.Path(os.environ['TRACE_AI_REPORT_PATH']).write_text('# Final Report\\n', encoding='utf-8')
            print('# AI OK')
            """,
        ),
    )

    started = client.post(
        "/api/jobs/ai-mail-job/ai-analysis",
        json={},
        headers={"X-Remote-User": "runner"},
    )

    assert started.status_code == 202

    payload = {}
    for _ in range(100):
        payload = client.get("/api/jobs/ai-mail-job/ai-analysis").json()
        if payload["status"] not in {"queued", "running"} and sent:
            break
        time.sleep(0.05)

    assert payload["status"] == "done"
    assert len(sent) == 1
    assert sent[0]["recipients"] == ["runner@cambricon.com", "owner@cambricon.com"]
    assert "AI分析完成" in sent[0]["subject"]
    assert "打开 AI 分析结果: http://trace.example/#/job/ai-mail-job/ai" in sent[0]["body"]
    assert "下载 Markdown 报告: http://trace.example/api/jobs/ai-mail-job/ai-analysis/report.md" in sent[0]["body"]


def test_compare_ai_analysis_completion_sends_email_with_result_link(client, isolated_server, tmp_path, monkeypatch):
    trace_a = tmp_path / "trace-a.pt.trace.json.gz"
    trace_b = tmp_path / "trace-b.pt.trace.json.gz"
    with gzip.open(trace_a, "wt", encoding="utf-8") as f:
        f.write("{}")
    with gzip.open(trace_b, "wt", encoding="utf-8") as f:
        f.write("{}")

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, user_token, label, mode, status,
                    file_a_name, file_a_gzip_path,
                    file_b_name, file_b_gzip_path
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    "ai-compare-mail-job",
                    "compare-owner",
                    "AI compare mail job",
                    "compare",
                    "done",
                    "trace-a.pt.trace.json.gz",
                    str(trace_a),
                    "trace-b.pt.trace.json.gz",
                    str(trace_b),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    Path(web_server.result_dir("ai-compare-mail-job")).mkdir(parents=True, exist_ok=True)

    sent = []
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(web_server, "PUBLIC_BASE_URL", "http://trace.example")
    monkeypatch.setattr(web_server, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(web_server, "SENDMAIL_COMMAND", "")
    monkeypatch.setattr(
        web_server,
        "_send_email_sync",
        lambda recipients, subject, body: sent.append({
            "recipients": recipients,
            "subject": subject,
            "body": body,
        }),
    )
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        _fake_claude_template(
            tmp_path,
            """
            assert os.environ['TRACE_AI_MODE'] == 'compare'
            pathlib.Path(os.environ['TRACE_AI_REPORT_PATH']).write_text('# Compare Final Report\\n', encoding='utf-8')
            print('# Compare AI OK')
            """,
        ),
    )

    started = client.post(
        "/api/jobs/ai-compare-mail-job/ai-analysis",
        json={},
        headers={"X-Remote-User": "runner"},
    )

    assert started.status_code == 202

    payload = {}
    for _ in range(100):
        payload = client.get("/api/jobs/ai-compare-mail-job/ai-analysis").json()
        if payload["status"] not in {"queued", "running"} and sent:
            break
        time.sleep(0.05)

    assert payload["status"] == "done"
    assert len(sent) == 1
    assert sent[0]["recipients"] == ["runner@cambricon.com", "compare-owner@cambricon.com"]
    assert "AI分析完成" in sent[0]["subject"]
    assert "类型: 对比" in sent[0]["body"]
    assert "打开 AI 分析结果: http://trace.example/#/job/ai-compare-mail-job/ai" in sent[0]["body"]
    assert "下载 Markdown 报告: http://trace.example/api/jobs/ai-compare-mail-job/ai-analysis/report.md" in sent[0]["body"]


def test_source_compare_ai_analysis_emails_trigger_compare_owner_and_source_owners(client, isolated_server, tmp_path, monkeypatch):
    trace_a = tmp_path / "source-a.pt.trace.json.gz"
    trace_b = tmp_path / "source-b.pt.trace.json.gz"
    with gzip.open(trace_a, "wt", encoding="utf-8") as f:
        f.write("{}")
    with gzip.open(trace_b, "wt", encoding="utf-8") as f:
        f.write("{}")

    async def insert_jobs():
        db = await web_db.get_db()
        try:
            await db.executemany(
                """
                INSERT INTO jobs(id, user_token, label, mode, status, file_a_name, file_a_gzip_path)
                VALUES(?,?,?,?,?,?,?)
                """,
                [
                    ("source-a", "owner-a", "Source A", "single", "done", "source-a.pt.trace.json.gz", str(trace_a)),
                    ("source-b", "owner-b", "Source B", "single", "done", "source-b.pt.trace.json.gz", str(trace_b)),
                ],
            )
            await db.execute(
                """
                INSERT INTO jobs(
                    id, user_token, label, mode, status,
                    file_a_name, file_b_name,
                    source_job_a, source_job_b
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    "ai-source-compare-mail-job",
                    "compare-owner",
                    "Source A vs Source B",
                    "compare",
                    "done",
                    "source-a.pt.trace.json.gz",
                    "source-b.pt.trace.json.gz",
                    "source-a",
                    "source-b",
                ),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_jobs())
    Path(web_server.result_dir("ai-source-compare-mail-job")).mkdir(parents=True, exist_ok=True)

    sent = []
    monkeypatch.setattr(web_server, "CLAUDE_ANALYSIS_ENABLED", True)
    monkeypatch.setattr(web_server, "PUBLIC_BASE_URL", "http://trace.example")
    monkeypatch.setattr(web_server, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(web_server, "SENDMAIL_COMMAND", "")
    monkeypatch.setattr(
        web_server,
        "_send_email_sync",
        lambda recipients, subject, body: sent.append({
            "recipients": recipients,
            "subject": subject,
            "body": body,
        }),
    )
    monkeypatch.setattr(
        web_server,
        "CLAUDE_ANALYSIS_COMMAND_TEMPLATE",
        _fake_claude_template(
            tmp_path,
            """
            assert os.environ['TRACE_AI_MODE'] == 'compare'
            pathlib.Path(os.environ['TRACE_AI_REPORT_PATH']).write_text('# Source Compare Final Report\\n', encoding='utf-8')
            print('# Source Compare AI OK')
            """,
        ),
    )

    started = client.post(
        "/api/jobs/ai-source-compare-mail-job/ai-analysis",
        json={},
        headers={"X-Remote-User": "runner"},
    )

    assert started.status_code == 202

    payload = {}
    for _ in range(100):
        payload = client.get("/api/jobs/ai-source-compare-mail-job/ai-analysis").json()
        if payload["status"] not in {"queued", "running"} and sent:
            break
        time.sleep(0.05)

    assert payload["status"] == "done"
    assert len(sent) == 1
    assert sent[0]["recipients"] == [
        "runner@cambricon.com",
        "compare-owner@cambricon.com",
        "owner-a@cambricon.com",
        "owner-b@cambricon.com",
    ]
    assert "类型: 对比" in sent[0]["body"]


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
        if payload["status"] not in {"queued", "running"}:
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
        if payload["status"] not in {"queued", "running"}:
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
    assert "AI report-ai-analysis-" in response.headers["content-disposition"]
    assert "# AI 性能分析报告" in response.text


def test_ai_analysis_artifact_download(client):
    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                "INSERT INTO jobs(id, label, mode, status) VALUES(?,?,?,?)",
                ("ai-artifact-job", "AI artifact", "single", "done"),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())
    analysis_dir = Path(web_server.ai_analysis_dir("ai-artifact-job"))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "details.txt").write_text("Detail Artifact\n", encoding="utf-8")
    (analysis_dir / "trace_a.db").write_bytes(b"sqlite-db")
    (analysis_dir / "ai_analysis_status.json").write_text("{}", encoding="utf-8")
    code_dir = analysis_dir / "triton_output_code"
    code_dir.mkdir()
    (code_dir / "kernel.py").write_text("def kernel():\n    return 1\n", encoding="utf-8")
    (analysis_dir / "triton_output_code_00_kernel.txt").write_text("@triton.jit\ndef kernel():\n    pass\n", encoding="utf-8")

    text_response = client.get("/api/jobs/ai-artifact-job/ai-analysis/artifacts/details.txt")
    db_response = client.get("/api/jobs/ai-artifact-job/ai-analysis/artifacts/trace_a.db")
    internal_response = client.get("/api/jobs/ai-artifact-job/ai-analysis/artifacts/ai_analysis_status.json")
    traversal_response = client.get("/api/jobs/ai-artifact-job/ai-analysis/artifacts/%2E%2E/details.txt")
    py_preview_response = client.get(
        "/api/jobs/ai-artifact-job/ai-analysis/artifact-content/triton_output_code/kernel.py"
    )
    txt_preview_response = client.get(
        "/api/jobs/ai-artifact-job/ai-analysis/artifact-content/triton_output_code_00_kernel.txt"
    )
    unsupported_preview_response = client.get(
        "/api/jobs/ai-artifact-job/ai-analysis/artifact-content/details.txt"
    )
    internal_preview_response = client.get(
        "/api/jobs/ai-artifact-job/ai-analysis/artifact-content/ai_analysis_status.json"
    )
    traversal_preview_response = client.get(
        "/api/jobs/ai-artifact-job/ai-analysis/artifact-content/%2E%2E/details.txt"
    )

    assert text_response.status_code == 200
    assert text_response.text == "Detail Artifact\n"
    assert "details.txt" in text_response.headers["content-disposition"]
    assert db_response.status_code == 200
    assert db_response.content == b"sqlite-db"
    assert internal_response.status_code == 400
    assert traversal_response.status_code == 400
    assert py_preview_response.status_code == 200
    assert py_preview_response.json()["content"] == "def kernel():\n    return 1\n"
    assert py_preview_response.json()["language"] == "python"
    assert txt_preview_response.status_code == 200
    assert "@triton.jit" in txt_preview_response.json()["content"]
    assert unsupported_preview_response.status_code == 400
    assert internal_preview_response.status_code == 400
    assert traversal_preview_response.status_code == 400


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
        if payload["status"] not in {"queued", "running"}:
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
        if payload["status"] not in {"queued", "running"}:
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


def test_feedback_email_recipients_include_admin_and_mentions(isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "FEEDBACK_NOTIFICATION_ADMIN_EMAILS", ["admin@cambricon.com"])
    monkeypatch.setattr(isolated_server, "FEEDBACK_MENTION_DOMAIN", "cambricon.com")

    recipients = isolated_server._feedback_notification_recipients(
        "请 @alice 和 @bob_1、@alice.z 看下，alice@example.com 不应被误识别，@alice 去重"
    )

    assert recipients == [
        "admin@cambricon.com",
        "alice@cambricon.com",
        "bob_1@cambricon.com",
        "alice.z@cambricon.com",
    ]


def test_feedback_create_sends_email_notification(client, isolated_server, monkeypatch):
    sent = []
    logged = []
    monkeypatch.setattr(isolated_server, "FEEDBACK_NOTIFICATION_ADMIN_EMAILS", ["admin@cambricon.com"])
    monkeypatch.setattr(isolated_server, "FEEDBACK_MENTION_DOMAIN", "cambricon.com")
    monkeypatch.setattr(isolated_server, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(isolated_server, "SENDMAIL_COMMAND", "")
    monkeypatch.setattr(
        isolated_server,
        "_send_email_sync",
        lambda recipients, subject, body: sent.append({
            "recipients": recipients,
            "subject": subject,
            "body": body,
        }),
    )
    monkeypatch.setattr(
        isolated_server.logger,
        "info",
        lambda msg, *args, **kwargs: logged.append((msg, kwargs.get("extra") or {})),
    )

    response = client.post(
        "/api/feedback",
        data={"body": "这个功能很有用，@alice 帮忙看一下"},
        headers={"X-Remote-User": "bob"},
    )

    assert response.status_code == 201
    notification = response.json()["notification"]
    assert notification["status"] == "sent"
    assert notification["transport"] == "smtp"
    assert notification["recipients"] == ["admin@cambricon.com", "alice@cambricon.com"]
    assert len(sent) == 1
    assert sent[0]["recipients"] == ["admin@cambricon.com", "alice@cambricon.com"]
    assert "留言板新帖子" in sent[0]["subject"]
    assert "这个功能很有用，@alice 帮忙看一下" in sent[0]["body"]
    feedback_logs = [extra for msg, extra in logged if msg == "feedback_created"]
    assert feedback_logs
    assert feedback_logs[0]["event"] == "feedback_created"
    assert feedback_logs[0]["feedback_kind"] == "post"
    assert feedback_logs[0]["user"] == "bob"
    assert feedback_logs[0]["mentioned_emails"] == ["alice@cambricon.com"]
    assert feedback_logs[0]["notification_status"] == "sent"
    assert "这个功能很有用" in feedback_logs[0]["body_preview"]


def test_feedback_notification_body_includes_deep_link(isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "PUBLIC_BASE_URL", "http://trace.example")

    post_body = isolated_server._feedback_notification_body({
        "post_id": "post-1",
        "message_id": "post-1",
        "author": "Alice",
        "body": "new post",
    })
    reply_body = isolated_server._feedback_notification_body({
        "post_id": "post-1",
        "message_id": "reply-2",
        "parent_id": "post-1",
        "author": "Bob",
        "body": "reply",
    })

    assert "打开留言: http://trace.example/#/feedback/post-1" in post_body
    assert "打开留言: http://trace.example/#/feedback/post-1?message=reply-2" in reply_body
    assert "打开应用:" not in reply_body
    assert "帖子 ID:" not in reply_body
    assert "消息 ID:" not in reply_body
    assert "图片:" not in reply_body


def test_feedback_create_reports_email_send_failure(client, isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "FEEDBACK_NOTIFICATION_ADMIN_EMAILS", ["admin@cambricon.com"])
    monkeypatch.setattr(isolated_server, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(isolated_server, "SENDMAIL_COMMAND", "")

    def fail_send(*args, **kwargs):
        raise RuntimeError("smtp rejected")

    monkeypatch.setattr(isolated_server, "_send_email_sync", fail_send)

    response = client.post(
        "/api/feedback",
        data={"body": "发送失败需要可见"},
        headers={"X-Remote-User": "bob"},
    )

    assert response.status_code == 201
    notification = response.json()["notification"]
    assert notification["status"] == "failed"
    assert notification["transport"] == "smtp"
    assert "smtp rejected" in notification["detail"]


def test_feedback_email_error_explains_dns_failure(isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "SMTP_HOST", "smtp.invalid.local")

    message = isolated_server._email_error_message(socket.gaierror(-2, "Name or service not known"))

    assert "SMTP 主机无法解析" in message
    assert "smtp.invalid.local" in message


def test_feedback_create_reports_missing_email_transport(client, isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "FEEDBACK_NOTIFICATION_ADMIN_EMAILS", ["admin@cambricon.com"])
    monkeypatch.setattr(isolated_server, "SMTP_HOST", "")
    monkeypatch.setattr(isolated_server, "SENDMAIL_COMMAND", "")

    response = client.post(
        "/api/feedback",
        data={"body": "邮件状态需要可见"},
        headers={"X-Remote-User": "bob"},
    )

    assert response.status_code == 201
    notification = response.json()["notification"]
    assert notification["status"] == "missing_transport"
    assert notification["recipients"] == ["admin@cambricon.com"]


def test_email_diagnostics_reports_smtp_dns_failure(client, isolated_server, monkeypatch):
    monkeypatch.setattr(isolated_server, "SMTP_HOST", "smtp.invalid.local")
    monkeypatch.setattr(isolated_server, "SMTP_PORT", 25)
    monkeypatch.setattr(isolated_server, "SENDMAIL_COMMAND", "")

    def fail_getaddrinfo(*args, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(isolated_server.socket, "getaddrinfo", fail_getaddrinfo)

    response = client.get("/api/email/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["transport"] == "smtp"
    dns_checks = [item for item in payload["checks"] if item["label"] == "SMTP DNS 解析"]
    assert dns_checks
    assert dns_checks[0]["status"] == "fail"
    assert "SMTP 主机无法解析" in dns_checks[0]["detail"]


def test_mention_candidates_use_local_feedback_authors(client):
    async def insert_feedback_authors():
        db = await web_db.get_db()
        try:
            await db.executemany(
                """
                INSERT INTO feedback_messages(id, parent_id, user_token, user_display, body)
                VALUES(?,?,?,?,?)
                """,
                [
                    ("mention-a", None, "alice", "Alice Zhou", "hello"),
                    ("mention-b", None, "bob", "Bob", "hello"),
                    ("mention-c", None, "alice.z", "Alice Z", "hello"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_feedback_authors())

    response = client.get("/api/mention-candidates?q=ali")

    assert response.status_code == 200
    payload = response.json()
    assert [item["username"] for item in payload["data"]][:2] == ["alice", "alice.z"]
    assert payload["data"][0]["email"] == "alice@cambricon.com"


def test_feedback_list_supports_sort_modes(client):
    async def insert_feedback_rows():
        db = await web_db.get_db()
        try:
            await db.executemany(
                """
                INSERT INTO feedback_messages(id, parent_id, user_token, user_display, body, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                [
                    ("post-active", None, "alice", "alice", "最近有回复", "2026-06-01 09:00:00", "2026-06-01 09:00:00"),
                    ("post-hot", None, "bob", "bob", "讨论很多", "2026-06-01 10:00:00", "2026-06-01 10:00:00"),
                    ("post-new", None, "carol", "carol", "发布时间最新", "2026-06-01 12:00:00", "2026-06-01 12:00:00"),
                    ("reply-active", "post-active", "dave", "dave", "最近回复", "2026-06-01 13:00:00", "2026-06-01 13:00:00"),
                    ("reply-hot-a", "post-hot", "erin", "erin", "热度一", "2026-06-01 10:30:00", "2026-06-01 10:30:00"),
                    ("reply-hot-b", "post-hot", "frank", "frank", "热度二", "2026-06-01 10:40:00", "2026-06-01 10:40:00"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_feedback_rows())

    updated = client.get("/api/feedback").json()
    created = client.get("/api/feedback?sort=created").json()
    hot = client.get("/api/feedback?sort=hot").json()
    invalid = client.get("/api/feedback?sort=unknown").json()

    assert updated["sort"] == "updated"
    assert [item["id"] for item in updated["data"]] == ["post-active", "post-new", "post-hot"]
    assert created["sort"] == "created"
    assert [item["id"] for item in created["data"]] == ["post-new", "post-hot", "post-active"]
    assert hot["sort"] == "hot"
    assert [item["id"] for item in hot["data"]] == ["post-hot", "post-active", "post-new"]
    assert invalid["sort"] == "updated"


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


def test_feedback_author_can_edit_posts_and_replies(client):
    created = client.post(
        "/api/feedback",
        data={"body": "原始帖子"},
        headers={"X-Remote-User": "alice"},
    )
    assert created.status_code == 201
    post = created.json()

    denied = client.patch(
        f"/api/feedback/{post['id']}",
        json={"body": "别人编辑"},
        headers={"X-Remote-User": "bob"},
    )
    assert denied.status_code == 403

    edited = client.patch(
        f"/api/feedback/{post['id']}",
        json={"body": "编辑后的帖子"},
        headers={"X-Remote-User": "alice"},
    )
    assert edited.status_code == 200
    edited_payload = edited.json()
    assert edited_payload["body"] == "编辑后的帖子"
    assert edited_payload["edited_at"]
    assert edited_payload["edit_count"] == 1
    assert edited_payload["user_token"] == "alice"

    reply = client.post(
        "/api/feedback",
        data={"body": "原始回复", "parent_id": post["id"]},
        headers={"X-Remote-User": "bob"},
    )
    assert reply.status_code == 201

    edited_reply = client.patch(
        f"/api/feedback/{reply.json()['id']}",
        json={"body": "编辑后的回复"},
        headers={"X-Remote-User": "bob"},
    )
    assert edited_reply.status_code == 200
    assert edited_reply.json()["body"] == "编辑后的回复"

    detail = client.get(f"/api/feedback/{post['id']}", headers={"X-Remote-User": "bob"})
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["body"] == "编辑后的帖子"
    assert detail_payload["edit_count"] == 1
    assert detail_payload["replies"][0]["body"] == "编辑后的回复"
    assert detail_payload["replies"][0]["edit_count"] == 1


def test_feedback_reactions_toggle_per_user(client):
    created = client.post(
        "/api/feedback",
        data={"body": "表情测试"},
        headers={"X-Remote-User": "alice"},
    )
    assert created.status_code == 201
    post_id = created.json()["id"]
    reply = client.post(
        "/api/feedback",
        data={"body": "可以点赞", "parent_id": post_id},
        headers={"X-Remote-User": "bob"},
    )
    assert reply.status_code == 201
    reply_id = reply.json()["id"]

    first = client.post(
        f"/api/feedback/{reply_id}/reactions",
        json={"emoji": "👍"},
        headers={"X-Remote-User": "alice"},
    )
    assert first.status_code == 200
    assert first.json()["active"] is True
    assert first.json()["reactions"] == [{"emoji": "👍", "count": 1, "reacted": True}]

    second_user = client.post(
        f"/api/feedback/{reply_id}/reactions",
        json={"emoji": "👍"},
        headers={"X-Remote-User": "carol"},
    )
    assert second_user.status_code == 200
    assert second_user.json()["reactions"][0]["count"] == 2

    detail_for_alice = client.get(f"/api/feedback/{post_id}", headers={"X-Remote-User": "alice"}).json()
    reaction = detail_for_alice["replies"][0]["reactions"][0]
    assert reaction["emoji"] == "👍"
    assert reaction["count"] == 2
    assert reaction["reacted"] is True

    toggled_off = client.post(
        f"/api/feedback/{reply_id}/reactions",
        json={"emoji": "👍"},
        headers={"X-Remote-User": "alice"},
    )
    assert toggled_off.status_code == 200
    assert toggled_off.json()["active"] is False
    assert toggled_off.json()["reactions"][0]["count"] == 1

    unsupported = client.post(
        f"/api/feedback/{reply_id}/reactions",
        json={"emoji": "🧪"},
        headers={"X-Remote-User": "alice"},
    )
    assert unsupported.status_code == 400


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


def test_admin_usage_stats_tracks_daily_activity(client):
    assert client.get("/api/config", headers={"X-Remote-User": "alice"}).status_code == 200
    assert client.get("/api/projects", headers={"X-Remote-User": "alice"}).status_code == 200
    assert client.get("/api/config", headers={"X-Remote-User": "bob"}).status_code == 200

    async def insert_audit_rows():
        db = await web_db.get_db()
        try:
            await db.executemany(
                """
                INSERT INTO audit_logs(id, user, action, resource_type, resource_id)
                VALUES(?,?,?,?,?)
                """,
                [
                    ("usage-audit-upload", "alice", "job.create", "job", "job-a"),
                    ("usage-audit-compare", "alice", "job.compare_create", "job", "job-b"),
                    ("usage-audit-batch", "alice", "job.batch_compare_create", "job", "job-c"),
                    ("usage-audit-ai", "alice", "job.ai_analysis_start", "job", "job-a"),
                    ("usage-audit-feedback", "bob", "feedback.create", "feedback", "feedback-a"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_audit_rows())

    response = client.get("/api/admin/usage?days=7")
    assert response.status_code == 200
    payload = response.json()
    today = payload["today"]
    assert today["dau"] == 2
    assert today["requests"] == 3
    assert today["upload_jobs"] == 1
    assert today["compare_jobs"] == 2
    assert today["ai_runs"] == 1
    assert today["feedback_messages"] == 1
    assert payload["seven_days"]["active_users"] == 2
    assert [item["user_token"] for item in payload["top_users_today"]] == ["alice", "bob"]


def test_admin_usage_requires_admin_when_ldap_enabled(isolated_server, monkeypatch):
    def fake_authenticate(username, password):
        return {
            "username": username,
            "display_name": username.title(),
            "email": f"{username}@example.com",
            "dn": f"CN={username},DC=example,DC=com",
        }

    monkeypatch.setattr(web_server, "AUTH_MODE", "ldap")
    monkeypatch.setattr(web_server, "AUTH_ENABLED", True)
    monkeypatch.setattr(web_server, "ADMIN_USERS", {"admin"})
    monkeypatch.setattr(web_server.ldap_auth, "authenticate", fake_authenticate)

    with TestClient(isolated_server.app) as test_client:
        assert test_client.get("/api/admin/usage").status_code == 401
        assert test_client.post("/api/login", json={"username": "bob", "password": "ok"}).status_code == 200
        assert test_client.get("/api/admin/usage").status_code == 403
        assert test_client.post("/api/logout").status_code == 200
        assert test_client.post("/api/login", json={"username": "admin", "password": "ok"}).status_code == 200
        assert test_client.get("/api/admin/usage").status_code == 200


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

        favorite = test_client.put(
            f"/api/projects/{shared_project['id']}/favorite",
            json={"is_favorite": True},
        )
        assert favorite.status_code == 200
        assert favorite.json()["is_favorite"] == 1
        assert favorite.json()["is_owner"] is False

        favorite_groups = test_client.get("/api/job-groups?project_view=favorite")
        assert favorite_groups.status_code == 200
        assert [item["id"] for item in favorite_groups.json()["data"]] == [shared_project["id"]]

        favorite_jobs = test_client.get("/api/jobs?project_view=favorite")
        assert favorite_jobs.status_code == 200
        assert [item["id"] for item in favorite_jobs.json()["data"]] == ["shared-job"]

        mine_groups = test_client.get("/api/job-groups?project_view=mine")
        assert mine_groups.status_code == 200
        assert [item["id"] for item in mine_groups.json()["data"]] == ["bob-project"]

        shared_groups = test_client.get("/api/job-groups?project_view=shared")
        assert shared_groups.status_code == 200
        assert [item["id"] for item in shared_groups.json()["data"]] == [shared_project["id"]]

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
    monkeypatch.setattr(isolated_server, "PUBLIC_BASE_URL", "http://tpa.cambricon.com:1818")

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
        assert payload["url"] == "http://tpa.cambricon.com:1818/#/job/share-job"

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


def test_job_groups_include_empty_projects(client):
    async def insert_rows():
        db = await web_db.get_db()
        try:
            await db.executemany(
                "INSERT INTO projects(id, name) VALUES(?,?)",
                [("project-empty", "Alpha Empty"), ("project-filled", "Beta Filled")],
            )
            await db.executemany(
                "INSERT INTO jobs(id, project_id, label, mode, status) VALUES(?,?,?,?,?)",
                [
                    ("job-filled", "project-filled", "filled", "single", "done"),
                    ("job-none", None, "none", "single", "done"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_rows())

    response = client.get("/api/job-groups")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert [group["id"] for group in payload["data"]] == ["project-empty", "project-filled", "__none__"]
    assert [group["job_count"] for group in payload["data"]] == [0, 1, 1]

    filtered = client.get("/api/job-groups?project_id=project-empty")
    assert filtered.status_code == 200
    filtered_data = filtered.json()["data"]
    assert len(filtered_data) == 1
    assert {
        "id": filtered_data[0]["id"],
        "label": filtered_data[0]["label"],
        "job_count": filtered_data[0]["job_count"],
    } == {"id": "project-empty", "label": "Alpha Empty", "job_count": 0}
    assert filtered_data[0]["visibility"] == "personal"
    assert filtered_data[0]["is_favorite"] == 0

    searched = client.get("/api/job-groups?q=alpha")
    assert searched.status_code == 200
    searched_data = searched.json()["data"]
    assert len(searched_data) == 1
    assert {
        "id": searched_data[0]["id"],
        "label": searched_data[0]["label"],
        "job_count": searched_data[0]["job_count"],
    } == {"id": "project-empty", "label": "Alpha Empty", "job_count": 0}


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


def test_jobs_search_includes_project_name_and_returns_project_label(client):
    async def insert_rows():
        db = await web_db.get_db()
        try:
            await db.executemany(
                "INSERT INTO projects(id, name) VALUES(?,?)",
                [("project-a", "Alpha Team"), ("project-b", "Beta Team")],
            )
            await db.executemany(
                "INSERT INTO jobs(id, project_id, label, mode, status, file_a_name, created_at) VALUES(?,?,?,?,?,?,?)",
                [
                    ("job-a", "project-a", "baseline", "single", "done", "trace-a.json", "2026-05-18 10:00:00"),
                    ("job-b", "project-b", "target", "single", "done", "trace-b.json", "2026-05-18 11:00:00"),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_rows())

    by_project = client.get("/api/jobs?q=alpha")
    assert by_project.status_code == 200
    payload = by_project.json()
    assert payload["total"] == 1
    assert payload["data"][0]["id"] == "job-a"
    assert payload["data"][0]["project_name"] == "Alpha Team"

    by_file = client.get("/api/jobs?project_id=project-b&q=trace-b")
    assert by_file.status_code == 200
    assert [item["id"] for item in by_file.json()["data"]] == ["job-b"]


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


def test_startup_requeues_interrupted_jobs(isolated_server, monkeypatch):
    seen = []
    running_job_id = "interrupted-running-job"
    legacy_error_job_id = "interrupted-error-job"
    os.makedirs(Path(web_db.DB_PATH).parent, exist_ok=True)

    async def fake_run_analysis(job_id):
        seen.append(job_id)

    async def seed_db():
        await web_db.init_db()
        db = await aiosqlite.connect(web_db.DB_PATH)
        try:
            await db.executemany(
                "INSERT INTO jobs(id, label, mode, status, error_msg) VALUES(?,?,?,?,?)",
                [
                    (running_job_id, "running", "single", "running", ""),
                    (
                        legacy_error_job_id,
                        "interrupted error",
                        "single",
                        "error",
                        web_server.INTERRUPTED_ANALYSIS_ERROR,
                    ),
                ],
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(seed_db())
    monkeypatch.setattr(web_server, "run_analysis", fake_run_analysis)

    with TestClient(isolated_server.app):
        for _ in range(50):
            if set(seen) == {running_job_id, legacy_error_job_id}:
                break
            time.sleep(0.01)

    async def fetch_jobs():
        db = await web_db.get_db()
        try:
            rows = await (
                await db.execute("SELECT id, status, error_msg FROM jobs ORDER BY id")
            ).fetchall()
            return {row["id"]: dict(row) for row in rows}
        finally:
            await db.close()

    rows = asyncio.run(fetch_jobs())
    assert set(seen) == {running_job_id, legacy_error_job_id}
    assert rows[running_job_id]["status"] == "pending"
    assert rows[running_job_id]["error_msg"] == ""
    assert rows[legacy_error_job_id]["status"] == "pending"
    assert rows[legacy_error_job_id]["error_msg"] == ""


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
            await db.execute(
                "INSERT INTO projects(id, name) VALUES(?,?)",
                ("target-project", "Target"),
            )
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
        json={"job_id_a": "source-a", "job_id_b": "source-b", "project_id": "target-project"},
    )

    assert created.status_code == 201
    compare_job = created.json()
    assert compare_job["mode"] == "compare"
    assert compare_job["project_id"] == "target-project"
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


def test_plain_gzip_upload_keeps_compressed_without_materializing_json(
    sample_trace_file_gz,
    tmp_path,
):
    dest_json = tmp_path / "trace.json"
    gzip_path = [None]

    async def extract_upload():
        with open(sample_trace_file_gz, "rb") as f:
            upload = UploadFile(file=f, filename="trace.json.gz")
            await web_server.save_and_extract(upload, str(dest_json), gzip_path)

    asyncio.run(extract_upload())

    assert not dest_json.exists()
    assert gzip_path[0] == str(dest_json) + ".gz"
    with gzip.open(gzip_path[0], "rt", encoding="utf-8") as f:
        assert json.load(f)["traceEvents"]


def test_trace_json_size_guard_skips_plain_gzip_without_strict_check(
    sample_trace_file_gz,
    isolated_server,
    monkeypatch,
):
    monkeypatch.setattr(web_server, "MAX_TRACE_JSON_BYTES", 64)
    monkeypatch.setattr(web_server, "STRICT_GZIP_SIZE_CHECK", False)

    def fail_iter(_path):
        raise AssertionError("plain gzip should not be pre-scanned")

    monkeypatch.setattr(web_server, "_iter_json_chunks", fail_iter)

    web_server._assert_trace_json_size_supported(sample_trace_file_gz, "a")


def test_trace_json_size_guard_rejects_oversized_plain_gzip_in_strict_mode(
    sample_trace_file_gz,
    isolated_server,
    monkeypatch,
):
    monkeypatch.setattr(web_server, "MAX_TRACE_JSON_BYTES", 64)
    monkeypatch.setattr(web_server, "STRICT_GZIP_SIZE_CHECK", True)

    with pytest.raises(ValueError, match="超过当前分析上限"):
        web_server._assert_trace_json_size_supported(sample_trace_file_gz, "a")


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
        job = None
        for _ in range(100):
            job_resp = client.get(f"/api/jobs/{job_id}")
            assert job_resp.status_code == 200
            job = job_resp.json()
            if job["status"] == "done":
                break
            time.sleep(0.05)
        assert job is not None
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


def test_run_analysis_does_not_block_on_slow_progress_writer(
    isolated_server,
    sample_trace_file,
    tmp_path,
    monkeypatch,
):
    trace_path = tmp_path / "trace.json"
    shutil.copyfile(sample_trace_file, trace_path)
    progress_calls = []

    async def slow_progress_writer(job_id, message):
        progress_calls.append((job_id, message))
        await asyncio.sleep(60)

    monkeypatch.setattr(web_server, "_write_analysis_progress", slow_progress_writer)

    async def run_job():
        await web_db.init_db()
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, user_token, label, mode, status, file_a_name, file_a_path,
                    save_triton_csv, save_triton_code
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    "slow-progress-job", "local", "slow progress", "single", "pending",
                    "trace.json", str(trace_path), 0, 0,
                ),
            )
            await db.commit()
        finally:
            await db.close()

        await asyncio.wait_for(web_server.run_analysis("slow-progress-job"), timeout=5)

        db = await web_db.get_db()
        try:
            cursor = await db.execute("SELECT status, error_msg FROM jobs WHERE id=?", ("slow-progress-job",))
            return await web_server.row_to_dict(await cursor.fetchone())
        finally:
            await db.close()

    row = asyncio.run(run_job())

    assert row["status"] == "done", row["error_msg"]
    assert progress_calls


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
    detail = job
    for _ in range(100):
        detail = client.get(f"/api/jobs/{job['id']}").json()
        if detail["status"] == "done":
            break
        time.sleep(0.05)

    assert detail["status"] == "done"
    jdir = Path(web_server.job_dir(job["id"]))
    assert not (jdir / "trace_a.json").exists()
    assert not (jdir / "trace_b.json").exists()
    assert (jdir / "trace_a.json.gz").exists()
    assert (jdir / "trace_b.json.gz").exists()

    download_a = client.get(f"/api/jobs/{job['id']}/files/a")
    download_b = client.get(f"/api/jobs/{job['id']}/files/b")
    assert download_a.status_code == 200
    assert download_b.status_code == 200
    assert json.loads(gzip.decompress(download_a.content))["traceEvents"]
    assert json.loads(gzip.decompress(download_b.content))["traceEvents"]


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


def test_step_reanalysis_creates_single_job(
    client,
    sample_trace_file,
    monkeypatch,
):
    enqueued = []

    async def fake_enqueue(job_id):
        enqueued.append(job_id)

    monkeypatch.setattr(web_server, "enqueue_analysis_job", fake_enqueue)

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status, file_a_name, file_a_path,
                    save_triton_csv, save_triton_code
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                ("single-source", "single", "single", "done", "trace.json", sample_trace_file, 1, 1),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    response = client.post(
        "/api/jobs/single-source/reanalyze-steps",
        json={"step_filter_a": "0,2", "label": "single steps"},
    )

    assert response.status_code == 201
    job = response.json()
    assert job["mode"] == "single"
    assert job["label"] == "single steps"
    assert job["step_filter_a"] == "0,2"
    assert job["step_filter_b"] == ""
    assert job["save_triton_csv"] == 1
    assert job["save_triton_code"] == 1
    assert Path(job["file_a_path"]).exists()
    assert enqueued == [job["id"]]


def test_compare_trace_slot_analysis_creates_single_job(
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

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status,
                    file_a_name, file_a_path,
                    file_b_name, file_b_path,
                    save_triton_csv, save_triton_code
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    "quick-compare", "a vs b", "compare", "done",
                    "a.json", sample_trace_file,
                    "b.json", str(trace_b),
                    1, 1,
                ),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    response = client.post(
        "/api/jobs/quick-compare/analyze-trace-slot",
        json={"slot": "b"},
    )

    assert response.status_code == 201
    job = response.json()
    assert job["mode"] == "single"
    assert job["label"] == "b.json · 单独分析"
    assert job["file_a_name"] == "b.json"
    assert job["file_a_path"]
    assert Path(job["file_a_path"]).exists()
    assert job["file_a_gzip_path"] is None
    assert job["source_job_a"] is None
    assert job["save_triton_csv"] == 1
    assert job["save_triton_code"] == 1
    assert enqueued == [job["id"]]


def test_compare_trace_slot_analysis_runs_gzip_trace(
    client,
    sample_trace_file_gz,
    tmp_path,
    monkeypatch,
):
    enqueued = []

    async def fake_enqueue(job_id):
        enqueued.append(job_id)

    trace_b = tmp_path / "b.json.gz"
    shutil.copyfile(sample_trace_file_gz, trace_b)
    monkeypatch.setattr(web_server, "enqueue_analysis_job", fake_enqueue)

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status,
                    file_a_name, file_a_gzip_path,
                    file_b_name, file_b_gzip_path
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    "quick-gzip-compare", "gzip a vs b", "compare", "done",
                    "a.json.gz", sample_trace_file_gz,
                    "b.json.gz", str(trace_b),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    response = client.post(
        "/api/jobs/quick-gzip-compare/analyze-trace-slot",
        json={"slot": "a"},
    )

    assert response.status_code == 201
    job = response.json()
    assert job["mode"] == "single"
    assert job["file_a_path"] is None
    assert job["file_a_gzip_path"].endswith(".json.gz")
    assert enqueued == [job["id"]]

    asyncio.run(web_server.run_analysis(job["id"]))

    async def fetch_job():
        db = await web_db.get_db()
        try:
            cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],))
            return await web_server.row_to_dict(await cursor.fetchone())
        finally:
            await db.close()

    analyzed = asyncio.run(fetch_job())
    assert analyzed["status"] == "done"


def test_step_reanalysis_creates_compare_job_with_independent_steps(
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

    async def insert_job():
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
                    "compare-source", "base vs target", "compare", "done",
                    "a.json", sample_trace_file,
                    "b.json", str(trace_b),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    response = client.post(
        "/api/jobs/compare-source/reanalyze-steps",
        json={"step_filter_a": "0", "step_filter_b": "2"},
    )

    assert response.status_code == 201
    job = response.json()
    assert job["mode"] == "compare"
    assert job["step_filter_a"] == "0"
    assert job["step_filter_b"] == "2"
    assert "A step 0 / B step 2" in job["label"]
    assert Path(job["file_a_path"]).exists()
    assert Path(job["file_b_path"]).exists()
    assert enqueued == [job["id"]]


def test_step_reanalysis_runs_compare_job_with_gzip_only_traces(
    client,
    sample_trace_file_gz,
    tmp_path,
    monkeypatch,
):
    enqueued = []

    async def fake_enqueue(job_id):
        enqueued.append(job_id)

    trace_b = tmp_path / "b.json.gz"
    shutil.copyfile(sample_trace_file_gz, trace_b)
    monkeypatch.setattr(web_server, "enqueue_analysis_job", fake_enqueue)

    async def insert_job():
        db = await web_db.get_db()
        try:
            await db.execute(
                """
                INSERT INTO jobs(
                    id, label, mode, status,
                    file_a_name, file_a_gzip_path,
                    file_b_name, file_b_gzip_path
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    "compare-gzip-source", "gzip base vs target", "compare", "done",
                    "a.json.gz", sample_trace_file_gz,
                    "b.json.gz", str(trace_b),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(insert_job())

    response = client.post(
        "/api/jobs/compare-gzip-source/reanalyze-steps",
        json={"step_filter_a": "0", "step_filter_b": "2"},
    )

    assert response.status_code == 201
    job = response.json()
    assert job["file_a_path"] is None
    assert job["file_a_gzip_path"].endswith(".json.gz")
    assert job["file_b_path"] is None
    assert job["file_b_gzip_path"].endswith(".json.gz")
    assert enqueued == [job["id"]]

    asyncio.run(web_server.run_analysis(job["id"]))

    async def fetch_job():
        db = await web_db.get_db()
        try:
            cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],))
            return await web_server.row_to_dict(await cursor.fetchone())
        finally:
            await db.close()

    analyzed = asyncio.run(fetch_job())
    assert analyzed["status"] == "done"
    assert "Step filter A: 0" in analyzed["console_out"]
    assert "Step filter B: 2" in analyzed["console_out"]


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
