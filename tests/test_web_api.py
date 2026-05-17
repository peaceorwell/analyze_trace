import asyncio
import json
import os
import sys
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
    assert [len(group["jobs"]) for group in first_page.json()["data"]] == [2, 1]

    second_page = client.get("/api/job-groups?limit=2&offset=2")
    assert second_page.status_code == 200
    assert [group["id"] for group in second_page.json()["data"]] == ["__none__"]


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
