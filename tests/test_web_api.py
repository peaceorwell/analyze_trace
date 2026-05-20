import asyncio
import json
import os
import sys
import time
from pathlib import Path
import shutil

import aiosqlite
import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WEB_DIR))

import db as web_db  # noqa: E402
import server as web_server  # noqa: E402


@pytest.fixture
def isolated_server(tmp_path, monkeypatch):
    storage_dir = tmp_path / "storage"
    monkeypatch.setattr(web_db, "DB_PATH", str(storage_dir / "jobs.db"))
    monkeypatch.setattr(web_server, "STORAGE_DIR", str(storage_dir))
    monkeypatch.setattr(web_server, "ALLOW_FILE_DOWNLOAD", True)
    monkeypatch.setattr(web_server, "ALLOW_CODE_EXECUTION", False)
    return web_server


@pytest.fixture
def client(isolated_server):
    with TestClient(isolated_server.app) as test_client:
        yield test_client


def test_config_reports_local_execution_flags(client):
    r = client.get("/api/config")

    assert r.status_code == 200
    assert r.json() == {
        "allow_file_download": True,
        "allow_code_execution": False,
    }


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
        "kernel_name,avg_dur_ms,family\n"
        "slow_kernel,30,gemm\n"
        "medium_kernel,20,gemm\n"
        "fast_kernel,10,other\n"
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

    filtered = client.get(
        "/api/jobs/table-job/results/all_kernels_avg.csv",
        params={"q": "fast", "limit": 10},
    )
    assert filtered.status_code == 200
    assert filtered.json()["filtered_total"] == 1
    assert filtered.json()["rows"][0]["kernel_name"] == "fast_kernel"


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
