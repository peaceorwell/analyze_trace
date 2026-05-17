import argparse
import asyncio
import contextlib
import csv
import gzip
import io
import json
import os
import shutil
import sys
import tarfile
import types
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trace_analyzer import compute_avgs, parse_trace, run_triton_code_and_get_efficiency  # noqa: E402

from db import get_db, init_db, row_to_dict  # noqa: E402

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# Configured at startup via CLI; read-only after that
ALLOW_FILE_DOWNLOAD = os.environ.get("TRACE_NO_DOWNLOAD", "") == ""
ALLOW_CODE_EXECUTION = os.environ.get("TRACE_ENABLE_CODE_EXEC", "") == "1"

# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await mark_interrupted_jobs()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


async def mark_interrupted_jobs():
    """Fail jobs that were left in-flight by a previous server process."""
    db = await get_db()
    try:
        await db.execute("""
            UPDATE jobs
            SET status='error',
                error_msg='Server restarted before this analysis completed'
            WHERE status IN ('pending', 'running')
        """)
        await db.commit()
    finally:
        await db.close()


def job_dir(job_id: str) -> str:
    return os.path.join(STORAGE_DIR, job_id)


def result_dir(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "results")


def require_code_execution_enabled():
    if not ALLOW_CODE_EXECUTION:
        raise HTTPException(403, "Code execution is disabled")


async def save_upload(upload: UploadFile, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(1 << 20):  # 1 MB chunks
            await f.write(chunk)


def _extract_gz_to_json(gz_path: str, dest_path: str):
    """Extract a .gz or .tar.gz file and write the contained JSON to dest_path.

    Supports two layouts:
    - tar.gz / tgz: archive containing a folder with a .json file inside
    - plain gzip:   single gzip-compressed .json file
    """
    is_tar = False
    try:
        is_tar = tarfile.is_tarfile(gz_path)
    except Exception:
        pass

    if is_tar:
        with tarfile.open(gz_path, "r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json")]
            if not members:
                raise ValueError("压缩包中未找到 .json 文件")
            member = max(members, key=lambda m: m.size)
            f = tar.extractfile(member)
            if f is None:
                raise ValueError("无法读取压缩包内的 JSON 文件")
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(f, out)
        return

    # Plain gzip
    with gzip.open(gz_path, "rb") as gz:
        with open(dest_path, "wb") as out:
            shutil.copyfileobj(gz, out)


async def save_and_extract(upload: UploadFile, dest_json: str, gzip_path: list):
    """Save upload; if it's a .gz file, extract the JSON and keep the original compressed file."""
    if upload.filename and upload.filename.lower().endswith(".gz"):
        # Save original compressed file (keep it for download/perfetto)
        gzip_path[0] = dest_json + ".gz"
        await save_upload(upload, gzip_path[0])
        try:
            await asyncio.to_thread(_extract_gz_to_json, gzip_path[0], dest_json)
        except Exception as e:
            raise HTTPException(400, f"解压失败: {e}")
    else:
        gzip_path[0] = None
        await save_upload(upload, dest_json)


def _compress_json_to_gz(json_path: str, gz_path: str):
    """Compress a JSON file to .json.gz format."""
    with open(json_path, "rb") as src:
        with gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def csv_to_rows(path: str) -> dict:
    """Read a CSV file and return {fields, rows}."""
    if not os.path.exists(path):
        return {"fields": [], "rows": []}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return {"fields": reader.fieldnames or [], "rows": rows}


def collect_results(jid: str) -> dict:
    rdir = result_dir(jid)
    files = {}
    for name in ["all_kernels_avg.csv", "all_kernels_cmp.csv",
                 "triton_kernels_avg.csv", "triton_kernels_cmp.csv",
                 "aten_ops_avg.csv", "aten_ops_cmp.csv",
                 "kernel_types_avg.csv", "kernel_types_cmp.csv",
                 "cncl_ops_avg.csv", "cncl_ops_cmp.csv"]:
        full = os.path.join(rdir, name)
        if os.path.exists(full):
            files[name] = csv_to_rows(full)
    # Collect per-step triton kernel CSVs
    if os.path.isdir(rdir):
        for fname in sorted(os.listdir(rdir)):
            if fname.startswith("step_") and fname.endswith("_triton_kernels.csv"):
                full = os.path.join(rdir, fname)
                files[fname] = csv_to_rows(full)
    return files


def collect_perfetto_context(jid: str) -> dict:
    path = os.path.join(result_dir(jid), "perfetto_context.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _perfetto_context(data):
    if not data["step_stats"] or not data["step_ranges"]:
        return None
    step = max(data["step_stats"], key=lambda s: data["step_stats"][s][0])
    start_us, end_us = data["step_ranges"][step]
    dur_us = max(0, end_us - start_us)
    padding_us = max(int(dur_us * 0.1), 1)
    return {
        "step": step,
        "step_name": f"ProfilerStep#{step}",
        "ts_ns": start_us * 1000,
        "dur_ns": dur_us * 1000,
        "vis_start_ns": max(0, start_us - padding_us) * 1000,
        "vis_end_ns": (end_us + padding_us) * 1000,
    }


def _perfetto_context_from_trace(path):
    from trace_analyzer import compute_avgs, parse_trace

    return _perfetto_context(compute_avgs(parse_trace(path, []), []))


async def _resolve_job_trace_path(job: dict, slot: str):
    path = job.get(f"file_{slot}_gzip_path") or job.get(f"file_{slot}_path")
    if path:
        return path

    src_jid = job.get(f"source_job_{slot}")
    if not src_jid:
        return None

    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT file_a_path, file_a_gzip_path FROM jobs WHERE id=?",
            (src_jid,),
        )
        src = await row_to_dict(await cur.fetchone())
    finally:
        await db.close()

    if not src:
        return None
    return src.get("file_a_gzip_path") or src.get("file_a_path")


async def ensure_perfetto_context(job: dict) -> dict:
    context = collect_perfetto_context(job["id"])
    if context:
        return context

    rebuilt = {}
    for slot in ("a", "b"):
        path = await _resolve_job_trace_path(job, slot)
        if path and os.path.exists(path):
            try:
                value = await asyncio.to_thread(_perfetto_context_from_trace, path)
            except Exception:
                # Perfetto focus is optional; older traces should still load even
                # when they cannot be reparsed for context backfill.
                continue
            if value:
                rebuilt[slot] = value

    if rebuilt:
        os.makedirs(result_dir(job["id"]), exist_ok=True)
        with open(os.path.join(result_dir(job["id"]), "perfetto_context.json"), "w") as f:
            json.dump(rebuilt, f)
    return rebuilt


# ── Synchronous analysis (runs in thread pool, must not await) ────────────────

def _run_sync_analysis(job, kernel_types, rdir, path_a, path_b, name_a, name_b):
    """All blocking I/O lives here so the event loop stays free."""
    from trace_analyzer import (compute_avgs, parse_trace,
                                print_step_summary, print_kernel_type_breakdown, print_top_kernels,
                                write_single, print_comparison, write_comparison)

    buf = io.StringIO()
    perfetto_context = {}
    with contextlib.redirect_stdout(buf):
        if job["mode"] == "single":
            data = compute_avgs(parse_trace(path_a, kernel_types), kernel_types)
            fake_args = types.SimpleNamespace(
                output_dir=rdir,
                save_triton_csv=bool(job["save_triton_csv"]),
                save_triton_code=bool(job["save_triton_code"]),
            )
            print_step_summary(data)
            print_kernel_type_breakdown(data)
            print_top_kernels(data)
            write_single(data, fake_args)
            perfetto_context["a"] = _perfetto_context(data)
        else:
            data_a = compute_avgs(parse_trace(path_a, kernel_types), kernel_types)
            data_b = compute_avgs(parse_trace(path_b, kernel_types), kernel_types)
            fake_args = types.SimpleNamespace(output_dir=rdir)
            label_a = name_a or os.path.basename(path_a)
            label_b = name_b or os.path.basename(path_b)
            print_comparison(data_a, data_b, label_a, label_b)
            print_top_kernels(data_a, label=label_a)
            print_top_kernels(data_b, label=label_b)
            write_comparison(data_a, data_b, fake_args)
            perfetto_context["a"] = _perfetto_context(data_a)
            perfetto_context["b"] = _perfetto_context(data_b)

    perfetto_context = {k: v for k, v in perfetto_context.items() if v}
    if perfetto_context:
        with open(os.path.join(rdir, "perfetto_context.json"), "w") as f:
            json.dump(perfetto_context, f)

    return buf.getvalue()


# ── Background analysis task ──────────────────────────────────────────────────

async def run_analysis(job_id: str):
    db = await get_db()
    try:
        await db.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
        await db.commit()

        cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
        job = await row_to_dict(await cursor.fetchone())

        kernel_types = [p for p in (job["kernel_types"] or "").split(",") if p]
        rdir = result_dir(job_id)
        os.makedirs(rdir, exist_ok=True)

        # Resolve source paths for compare-from-history (needs DB, must happen before thread)
        if job["mode"] == "compare" and not job["file_a_path"]:
            cursor_a = await db.execute("SELECT * FROM jobs WHERE id=?", (job["source_job_a"],))
            src_a = await row_to_dict(await cursor_a.fetchone())
            cursor_b = await db.execute("SELECT * FROM jobs WHERE id=?", (job["source_job_b"],))
            src_b = await row_to_dict(await cursor_b.fetchone())
            path_a = src_a.get("file_a_gzip_path") or src_a["file_a_path"]
            path_b = src_b.get("file_a_gzip_path") or src_b["file_a_path"]
            name_a = src_a.get("file_a_name") or os.path.basename(path_a)
            name_b = src_b.get("file_a_name") or os.path.basename(path_b)
        else:
            path_a = job["file_a_path"]
            path_b = job["file_b_path"]
            name_a = job["file_a_name"]
            name_b = job["file_b_name"]

        # Run all blocking analysis in a thread pool so the event loop stays responsive
        console_out = await asyncio.to_thread(
            _run_sync_analysis, job, kernel_types, rdir, path_a, path_b, name_a, name_b
        )

        # Post-analysis: keep only .json.gz files to save storage space
        for slot in ("a", "b"):
            json_path = job.get(f"file_{slot}_path")
            gzip_path = job.get(f"file_{slot}_gzip_path")
            if not json_path:
                continue
            if not gzip_path:
                # Uploaded as uncompressed JSON — compress to .gz
                gzip_path = json_path + ".gz"
                await asyncio.to_thread(_compress_json_to_gz, json_path, gzip_path)
            # Delete the uncompressed JSON file (keep only .gz)
            if os.path.exists(json_path):
                os.remove(json_path)
            await db.execute(
                f"UPDATE jobs SET file_{slot}_path=NULL, file_{slot}_gzip_path=? WHERE id=?",
                (gzip_path, job_id),
            )

        await db.execute(
            "UPDATE jobs SET status='done', console_out=?, result_dir=? WHERE id=?",
            (console_out, rdir, job_id),
        )
        await db.commit()

    except Exception as e:
        await db.execute(
            "UPDATE jobs SET status='error', error_msg=? WHERE id=?",
            (str(e), job_id),
        )
        await db.commit()
    finally:
        await db.close()


# ── Routes: index / config ────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/api/config")
async def get_config():
    return {
        "allow_file_download": ALLOW_FILE_DOWNLOAD,
        "allow_code_execution": ALLOW_CODE_EXECUTION,
    }


# ── Routes: projects ──────────────────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects():
    db = await get_db()
    rows = await (await db.execute("""
        SELECT * FROM projects
        ORDER BY created_at DESC
    """)).fetchall()
    await db.close()
    return [dict(r) for r in rows]


@app.post("/api/projects", status_code=201)
async def create_project(body: dict):
    pid = str(uuid.uuid4())
    db = await get_db()
    await db.execute(
        "INSERT INTO projects(id, name, description, is_public) VALUES(?,?,?,?)",
        (pid, body.get("name", "新项目"), body.get("description", ""), 1),
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await cursor.fetchone()
    await db.close()
    return dict(row)


@app.put("/api/projects/{pid}")
async def update_project(pid: str, body: dict):
    db = await get_db()

    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)

    await db.execute(
        "UPDATE projects SET name=?, description=? WHERE id=?",
        (body.get("name"), body.get("description", ""), pid),
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await cursor.fetchone()
    await db.close()
    return dict(row)


@app.delete("/api/projects/{pid}", status_code=204)
async def delete_project(pid: str):
    db = await get_db()

    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)

    # Move project info to deleted_projects table for recovery
    await db.execute("""
        INSERT INTO deleted_projects(id, user_token, folder_id, name, description, password_hash, is_public, created_at, deleted_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (row["id"], row["user_token"], row.get("folder_id"), row["name"],
          row.get("description", ""), row.get("password_hash"), row.get("is_public", 0), row.get("created_at")))

    # Move all jobs to deleted_jobs table (keep files for recovery)
    cursor = await db.execute("SELECT * FROM jobs WHERE project_id=?", (pid,))
    jobs_data = await cursor.fetchall()
    for job in jobs_data:
        job_dict = dict(job)
        await db.execute("""
            INSERT INTO deleted_jobs(id, project_id, user_token, created_at, label, mode,
                file_a_name, file_a_path, file_a_gzip_path, file_a_exists,
                file_b_name, file_b_path, file_b_gzip_path, file_b_exists,
                source_job_a, source_job_b, kernel_types, save_triton_csv, save_triton_code,
                status, console_out, error_msg, result_dir, deleted_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (job_dict["id"], job_dict.get("project_id"), job_dict.get("user_token"),
              job_dict.get("created_at"), job_dict.get("label", ""), job_dict.get("mode"),
              job_dict.get("file_a_name"), job_dict.get("file_a_path"), job_dict.get("file_a_gzip_path"), job_dict.get("file_a_exists", 1),
              job_dict.get("file_b_name"), job_dict.get("file_b_path"), job_dict.get("file_b_gzip_path"), job_dict.get("file_b_exists", 1),
              job_dict.get("source_job_a"), job_dict.get("source_job_b"),
              job_dict.get("kernel_types", "gemm,embedding,pool"), job_dict.get("save_triton_csv", 0), job_dict.get("save_triton_code", 0),
              job_dict.get("status", "pending"), job_dict.get("console_out", ""), job_dict.get("error_msg", ""), job_dict.get("result_dir", "")))

    # Delete jobs from main table
    await db.execute("DELETE FROM jobs WHERE project_id=?", (pid,))

    # Delete the project
    await db.execute("DELETE FROM projects WHERE id=?", (pid,))
    await db.commit()
    await db.close()


# ── Routes: deleted projects (recovery) ───────────────────────────────────────

@app.get("/api/deleted-projects")
async def list_deleted_projects():
    """List recoverable projects deleted within the last 10 days."""
    db = await get_db()
    rows = await (await db.execute("""
        SELECT * FROM deleted_projects
        WHERE deleted_at >= datetime('now', '-10 days')
        ORDER BY deleted_at DESC
    """)).fetchall()
    await db.close()
    return [dict(r) for r in rows]


@app.post("/api/deleted-projects/{pid}/restore", status_code=200)
async def restore_project(pid: str):
    """Restore a project deleted within the last 10 days."""
    db = await get_db()

    # Get deleted project info
    cursor = await db.execute("SELECT * FROM deleted_projects WHERE id=?", (pid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404, "Deleted project not found or expired")

    # Check if project with same ID already exists (shouldn't happen, but safety check)
    cursor = await db.execute("SELECT id FROM projects WHERE id=?", (pid,))
    if await cursor.fetchone():
        await db.close()
        raise HTTPException(409, "Project with this ID already exists")

    created_at = row.get("created_at") or "CURRENT_TIMESTAMP"
    try:
        if created_at == "CURRENT_TIMESTAMP":
            await db.execute("""
                INSERT INTO projects(id, user_token, folder_id, name, description, password_hash, is_public, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (row["id"], row["user_token"], row.get("folder_id"), row["name"],
                  row.get("description", ""), row.get("password_hash"), row.get("is_public", 0)))
        else:
            await db.execute("""
                INSERT INTO projects(id, user_token, folder_id, name, description, password_hash, is_public, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """, (row["id"], row["user_token"], row.get("folder_id"), row["name"],
                  row.get("description", ""), row.get("password_hash"), row.get("is_public", 0), created_at))
    except Exception as e:
        await db.close()
        raise HTTPException(500, f"数据库错误: {e}")

    # Restore jobs from deleted_jobs table
    cursor = await db.execute("SELECT * FROM deleted_jobs WHERE project_id=?", (pid,))
    deleted_jobs = await cursor.fetchall()

    for job in deleted_jobs:
        job_dict = dict(job)
        await db.execute("""
            INSERT INTO jobs(id, project_id, user_token, created_at, label, mode,
                file_a_name, file_a_path, file_a_gzip_path, file_a_exists,
                file_b_name, file_b_path, file_b_gzip_path, file_b_exists,
                source_job_a, source_job_b, kernel_types, save_triton_csv, save_triton_code,
                status, console_out, error_msg, result_dir)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_dict["id"], pid, job_dict.get("user_token"),
              job_dict.get("created_at"), job_dict.get("label", ""), job_dict.get("mode"),
              job_dict.get("file_a_name"), job_dict.get("file_a_path"), job_dict.get("file_a_gzip_path"), job_dict.get("file_a_exists", 1),
              job_dict.get("file_b_name"), job_dict.get("file_b_path"), job_dict.get("file_b_gzip_path"), job_dict.get("file_b_exists", 1),
              job_dict.get("source_job_a"), job_dict.get("source_job_b"),
              job_dict.get("kernel_types", "gemm,embedding,pool"), job_dict.get("save_triton_csv", 0), job_dict.get("save_triton_code", 0),
              job_dict.get("status", "pending"), job_dict.get("console_out", ""), job_dict.get("error_msg", ""), job_dict.get("result_dir", "")))

    # Remove restored jobs from deleted_jobs
    await db.execute("DELETE FROM deleted_jobs WHERE project_id=?", (pid,))

    # Remove from deleted_projects
    await db.execute("DELETE FROM deleted_projects WHERE id=?", (pid,))

    await db.commit()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    restored = await cursor.fetchone()
    await db.close()

    return dict(restored)


@app.delete("/api/deleted-projects/{pid}", status_code=204)
async def permanently_delete_project(pid: str):
    """Permanently delete a project from recovery list (without restoring)."""
    db = await get_db()

    cursor = await db.execute("SELECT * FROM deleted_projects WHERE id=?", (pid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404, "Deleted project not found")

    # Get job ids from deleted_jobs to delete files
    cursor = await db.execute("SELECT id FROM deleted_jobs WHERE project_id=?", (pid,))
    job_ids = [r["id"] for r in await cursor.fetchall()]
    for jid in job_ids:
        jdir = job_dir(jid)
        if os.path.exists(jdir):
            shutil.rmtree(jdir)

    # Delete jobs from deleted_jobs table
    await db.execute("DELETE FROM deleted_jobs WHERE project_id=?", (pid,))

    # Delete from deleted_projects
    await db.execute("DELETE FROM deleted_projects WHERE id=?", (pid,))
    await db.commit()
    await db.close()


# ── Routes: jobs ──────────────────────────────────────────────────────────────

async def _with_file_exists(rows):
    # Verify actual file existence on disk (DB flag may be stale)
    # Resolve source jobs for compare jobs in batch
    src_ids = set()
    for r in rows:
        d = dict(r)
        for slot in ("a", "b"):
            if not (d.get(f"file_{slot}_gzip_path") or d.get(f"file_{slot}_path")):
                src = d.get(f"source_job_{slot}")
                if src:
                    src_ids.add(src)

    src_paths = {}
    if src_ids:
        db2 = await get_db()
        placeholders = ",".join("?" * len(src_ids))
        cur = await db2.execute(
            f"SELECT id, file_a_path, file_a_gzip_path FROM jobs WHERE id IN ({placeholders})",
            tuple(src_ids))
        async for src_row in cur:
            s = dict(src_row)
            src_paths[s["id"]] = s.get("file_a_gzip_path") or s.get("file_a_path")
        await db2.close()

    data = []
    for r in rows:
        d = dict(r)
        for slot in ("a", "b"):
            path = d.get(f"file_{slot}_gzip_path") or d.get(f"file_{slot}_path")
            if path:
                d[f"file_{slot}_exists"] = 1 if os.path.exists(path) else 0
            else:
                src_jid = d.get(f"source_job_{slot}")
                if src_jid and src_paths.get(src_jid):
                    d[f"file_{slot}_exists"] = 1 if os.path.exists(src_paths[src_jid]) else 0
                else:
                    d[f"file_{slot}_exists"] = 0
        data.append(d)
    return data


@app.get("/api/jobs")
async def list_jobs(
    project_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    # Count query
    count_sql = "SELECT COUNT(*) as total FROM jobs"
    count_params = []

    if project_id == "__none__":
        count_sql = "SELECT COUNT(*) as total FROM jobs WHERE project_id IS NULL"
    elif project_id:
        count_sql = "SELECT COUNT(*) as total FROM jobs WHERE project_id = ?"
        count_params = [project_id]

    count_cursor = await db.execute(count_sql, count_params)
    total = (await count_cursor.fetchone())[0]

    # Data query - all jobs
    if project_id == "__none__":
        rows = await (await db.execute(
            "SELECT * FROM jobs WHERE project_id IS NULL ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )).fetchall()
    elif project_id:
        rows = await (await db.execute(
            "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (project_id, limit, offset)
        )).fetchall()
    else:
        rows = await (await db.execute("""
            SELECT * FROM jobs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))).fetchall()

    await db.close()

    data = await _with_file_exists(rows)
    return {"data": data, "total": total, "limit": limit, "offset": offset}


@app.get("/api/job-groups")
async def list_job_groups(
    project_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    where_sql = ""
    where_params = []
    if project_id == "__none__":
        where_sql = "WHERE j.project_id IS NULL"
    elif project_id:
        where_sql = "WHERE j.project_id = ?"
        where_params = [project_id]

    count_cursor = await db.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT j.project_id
            FROM jobs j
            {where_sql}
            GROUP BY j.project_id
        )
        """,
        where_params,
    )
    total = (await count_cursor.fetchone())[0]

    group_rows = await (
        await db.execute(
            f"""
            SELECT
                j.project_id,
                COALESCE(p.name, '未分组') AS label
            FROM jobs j
            LEFT JOIN projects p ON p.id = j.project_id
            {where_sql}
            GROUP BY j.project_id
            ORDER BY
                CASE WHEN j.project_id IS NULL THEN 1 ELSE 0 END,
                p.name COLLATE NOCASE
            LIMIT ? OFFSET ?
            """,
            (*where_params, limit, offset),
        )
    ).fetchall()

    if not group_rows:
        await db.close()
        return {"data": [], "total": total, "limit": limit, "offset": offset}

    non_null_ids = [row["project_id"] for row in group_rows if row["project_id"] is not None]
    has_ungrouped = any(row["project_id"] is None for row in group_rows)
    clauses = []
    job_params = []
    if non_null_ids:
        clauses.append(f"project_id IN ({','.join('?' * len(non_null_ids))})")
        job_params.extend(non_null_ids)
    if has_ungrouped:
        clauses.append("project_id IS NULL")

    jobs_rows = await (
        await db.execute(
            f"""
            SELECT * FROM jobs
            WHERE {' OR '.join(clauses)}
            ORDER BY created_at DESC
            """,
            job_params,
        )
    ).fetchall()
    await db.close()

    jobs = await _with_file_exists(jobs_rows)
    jobs_by_group = defaultdict(list)
    for job in jobs:
        jobs_by_group[job.get("project_id") or "__none__"].append(job)

    data = []
    for row in group_rows:
        group_id = row["project_id"] or "__none__"
        data.append(
            {
                "id": group_id,
                "label": row["label"],
                "jobs": jobs_by_group[group_id],
            }
        )

    return {"data": data, "total": total, "limit": limit, "offset": offset}


@app.post("/api/jobs", status_code=201)
async def create_job(
    background_tasks: BackgroundTasks,
    file_a: UploadFile,
    file_b: Optional[UploadFile] = None,
    kernel_types: str = Form("gemm,embedding,pool"),
    save_triton_csv: bool = Form(False),
    save_triton_code: bool = Form(False),
    label: str = Form(""),
    project_id: Optional[str] = Form(None),
):
    jid = str(uuid.uuid4())
    jdir = job_dir(jid)

    path_a = os.path.join(jdir, "trace_a.json")
    gzip_path_a = [None]
    await save_and_extract(file_a, path_a, gzip_path_a)

    path_b = None
    name_b = None
    gzip_path_b = [None]
    mode = "single"
    if file_b and file_b.filename:
        path_b = os.path.join(jdir, "trace_b.json")
        await save_and_extract(file_b, path_b, gzip_path_b)
        name_b = file_b.filename
        mode = "compare"

    eff_label = label or file_a.filename or jid

    db = await get_db()
    await db.execute(
        """INSERT INTO jobs(id, project_id, label, mode,
               file_a_name, file_a_path, file_a_gzip_path, file_b_name, file_b_path, file_b_gzip_path,
               kernel_types, save_triton_csv, save_triton_code)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (jid, project_id or None, eff_label, mode,
         file_a.filename, path_a, gzip_path_a[0], name_b, path_b, gzip_path_b[0],
         kernel_types, int(save_triton_csv), int(save_triton_code)),
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await cursor.fetchone()
    await db.close()

    background_tasks.add_task(run_analysis, jid)
    return dict(row)


@app.post("/api/jobs/compare", status_code=201)
async def compare_jobs(body: dict, background_tasks: BackgroundTasks):
    job_id_a = body.get("job_id_a")
    job_id_b = body.get("job_id_b")
    if not job_id_a or not job_id_b:
        raise HTTPException(400, "job_id_a and job_id_b are required")

    db = await get_db()

    cursor_a = await db.execute("SELECT * FROM jobs WHERE id=?", (job_id_a,))
    src_a = await row_to_dict(await cursor_a.fetchone())
    cursor_b = await db.execute("SELECT * FROM jobs WHERE id=?", (job_id_b,))
    src_b = await row_to_dict(await cursor_b.fetchone())

    if not src_a or not src_b:
        await db.close()
        raise HTTPException(404, "Source job not found")

    # Verify actual file existence on disk (DB flag may be stale)
    path_a = src_a.get("file_a_gzip_path") or src_a.get("file_a_path")
    path_b = src_b.get("file_a_gzip_path") or src_b.get("file_a_path")
    if not path_a or not os.path.exists(path_a):
        await db.close()
        raise HTTPException(409, f"Source file A has been deleted")
    if not path_b or not os.path.exists(path_b):
        await db.close()
        raise HTTPException(409, f"Source file B has been deleted")

    jid = str(uuid.uuid4())
    kernel_types = body.get("kernel_types", "gemm,embedding,pool")
    eff_label = body.get("label") or f"{src_a['label']} vs {src_b['label']}"

    await db.execute(
        """INSERT INTO jobs(id, project_id, label, mode,
               file_a_name, file_b_name,
               source_job_a, source_job_b,
               kernel_types, save_triton_csv, save_triton_code)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (jid, body.get("project_id"), eff_label, "compare",
         src_a["file_a_name"], src_b["file_a_name"],
         job_id_a, job_id_b,
         kernel_types, 0, 0),
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await cursor.fetchone()
    await db.close()

    background_tasks.add_task(run_analysis, jid)
    return dict(row)


@app.get("/api/jobs/{jid}")
async def get_job(jid: str):
    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if row is None:
        raise HTTPException(404)

    job = dict(row)
    # Verify actual file existence on disk (DB flag may be stale)
    # For compare jobs, resolve via source jobs
    for slot in ("a", "b"):
        path = job.get(f"file_{slot}_gzip_path") or job.get(f"file_{slot}_path")
        if path:
            job[f"file_{slot}_exists"] = 1 if os.path.exists(path) else 0
        else:
            src_jid = job.get(f"source_job_{slot}")
            if src_jid:
                db2 = await get_db()
                cur2 = await db2.execute("SELECT file_a_path, file_a_gzip_path FROM jobs WHERE id=?", (src_jid,))
                src = await row_to_dict(await cur2.fetchone())
                await db2.close()
                if src:
                    src_path = src.get("file_a_gzip_path") or src.get("file_a_path")
                    job[f"file_{slot}_exists"] = 1 if src_path and os.path.exists(src_path) else 0
                else:
                    job[f"file_{slot}_exists"] = 0
            else:
                job[f"file_{slot}_exists"] = 0
    if job["status"] == "done":
        job["results"] = collect_results(jid)
        job["perfetto_context"] = await ensure_perfetto_context(job)
    return job


@app.patch("/api/jobs/{jid}")
async def patch_job(jid: str, body: dict):
    db = await get_db()

    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)

    if "label" in body:
        await db.execute("UPDATE jobs SET label=? WHERE id=?", (body["label"], jid))
    if "project_id" in body:
        await db.execute("UPDATE jobs SET project_id=? WHERE id=?",
                         (body["project_id"] or None, jid))
    await db.commit()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await cursor.fetchone()
    await db.close()
    return dict(row)


@app.delete("/api/jobs/{jid}", status_code=204)
async def delete_job(jid: str):
    db = await get_db()

    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)


    # Remove all files on disk
    jdir = job_dir(jid)
    if os.path.exists(jdir):
        shutil.rmtree(jdir)

    await db.execute("DELETE FROM jobs WHERE id=?", (jid,))
    await db.commit()


@app.post("/api/jobs/{jid}/run-triton")
async def run_job_triton(jid: str):
    """Run triton code files and append local efficiency to CSV."""
    require_code_execution_enabled()
    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if row is None:
        raise HTTPException(404)

    if row.get("status") != "done":
        raise HTTPException(400, "Job not completed")

    rdir = result_dir(jid)

    def do_run():
        results = {}
        # Find all triton code files in step_*_triton_codes directories
        if not os.path.isdir(rdir):
            return results
        for dname in sorted(os.listdir(rdir)):
            if dname.startswith("step_") and dname.endswith("_triton_codes"):
                code_dir = os.path.join(rdir, dname)
                if not os.path.isdir(code_dir):
                    continue
                for fname in sorted(os.listdir(code_dir)):
                    if fname.endswith(".py"):
                        code_path = os.path.join(code_dir, fname)
                        key = f"{dname}/{fname}"
                        efficiency = run_triton_code_and_get_efficiency(code_path)
                        results[key] = efficiency
        return results

    # Run in thread pool since subprocess is blocking
    run_results = await asyncio.to_thread(do_run)

    # If no triton code files found, return early
    if not run_results:
        return {"success": True, "message": "No triton code files found", "results": {}}

    # Check if any execution succeeded
    any_success = any(v is not None for v in run_results.values())
    if not any_success:
        return {"success": False, "message": "All triton executions failed", "results": run_results}

    # Read the step CSV files and add local efficiency column
    def update_csv_with_efficiency():
        updated = []
        for dname in sorted(os.listdir(rdir)):
            if dname.startswith("step_") and dname.endswith("_triton_kernels.csv"):
                csv_path = os.path.join(rdir, dname)
                temp_path = csv_path + ".tmp"
                with open(csv_path, "r", newline="", encoding="utf-8") as fin:
                    with open(temp_path, "w", newline="", encoding="utf-8") as fout:
                        reader = csv.reader(fin)
                        writer = csv.writer(fout)
                        header = next(reader)
                        # Check if "local efficiency" column already exists
                        if "local efficiency" not in header:
                            header.append("local efficiency")
                        writer.writerow(header)
                        # Create mapping from kernel name to efficiency
                        for row in reader:
                            if len(row) >= 1:
                                kernel_name = row[0]
                                # Try to find matching triton code file
                                matched_eff = None
                                for code_key, eff in run_results.items():
                                    if kernel_name in code_key and eff is not None:
                                        matched_eff = eff
                                        break
                                if matched_eff:
                                    row.append(matched_eff)
                                else:
                                    row.append("")
                            writer.writerow(row)
                os.replace(temp_path, csv_path)
                updated.append(dname)

                # Also update the parent triton_kernels_avg.csv if it exists
                parent_csv = os.path.join(rdir, "triton_kernels_avg.csv")
                if os.path.exists(parent_csv):
                    _update_parent_triton_csv(parent_csv, run_results)

        return updated

    def _update_parent_triton_csv(csv_path, exec_results):
        temp_path = csv_path + ".tmp"
        with open(csv_path, "r", newline="", encoding="utf-8") as fin:
            with open(temp_path, "w", newline="", encoding="utf-8") as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                header = next(reader)
                if "local efficiency" not in header:
                    header.append("local efficiency")
                writer.writerow(header)
                for row in reader:
                    if len(row) >= 1:
                        kernel_name = row[0]
                        matched_eff = None
                        for code_key, eff in exec_results.items():
                            if kernel_name in code_key and eff is not None:
                                matched_eff = eff
                                break
                        if matched_eff:
                            row.append(matched_eff)
                        else:
                            row.append("")
                    writer.writerow(row)
        os.replace(temp_path, csv_path)

    updated_files = await asyncio.to_thread(update_csv_with_efficiency)

    return {
        "success": True,
        "message": f"Updated {len(updated_files)} files",
        "results": run_results,
        "updated_files": updated_files,
    }


@app.post("/api/jobs/{jid}/clear-inductor-cache")
async def clear_inductor_cache(jid: str):
    """Clear the torchinductor cache for a job's triton runs."""
    require_code_execution_enabled()
    import shutil, glob

    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if row is None:
        raise HTTPException(404)

    def do_clear():
        import subprocess
        try:
            # Find and remove all torchinductor_* directories in /tmp
            pattern = "/tmp/torchinductor_*"
            dirs = glob.glob(pattern)
            for d in dirs:
                subprocess.run(["rm", "-rf", d], check=False)
            return {"success": True, "removed": dirs}
        except Exception as e:
            return {"success": False, "error": str(e)}

    result = await asyncio.to_thread(do_clear)
    return result


@app.post("/api/jobs/{jid}/run-triton-single")
async def run_single_triton(jid: str, body: dict):
    """Run a single triton code file and return its efficiency."""
    require_code_execution_enabled()
    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if row is None:
        raise HTTPException(404)

    if row.get("status") != "done":
        raise HTTPException(400, "Job not completed")

    code_path_rel = body.get("code_path")
    if not code_path_rel:
        raise HTTPException(400, "code_path is required")

    rdir = result_dir(jid)
    code_path = os.path.normpath(os.path.join(rdir, code_path_rel))
    # Security: ensure the resolved path is within rdir
    if not code_path.startswith(os.path.abspath(rdir)):
        raise HTTPException(400, "Invalid code_path")

    def do_run():
        import subprocess, sys
        try:
            # Get MLU device info first
            get_mlu_info = '''
import torch_mlu
import subprocess
try:
    import torch
    mlu_version = torch_mlu.get_version()
    device_name = torch.mlu.get_device_name(0) if torch.mlu.is_available() else "N/A"
    driver_version = torch_mlu.get_driver_version()
    # Get pip-installed triton version
    pip_result = subprocess.run(["pip", "show", "triton"], capture_output=True, text=True)
    triton_version = "N/A"
    for line in pip_result.stdout.split("\\n"):
        if line.startswith("Version:"):
            triton_version = line.split(":", 1)[1].strip()
            break
    print(f"MLU Device:   {device_name}")
    print(f"Driver:       {driver_version}")
    print(f"torch_mlu:    {mlu_version}")
    print(f"Triton(pip):  {triton_version}")
except Exception as e:
    print(f"[MLU Info] Failed to get MLU info: {e}")
'''
            try:
                info_result = subprocess.run(
                    [sys.executable, "-c", get_mlu_info],
                    capture_output=True, text=True, timeout=10,
                )
                mlu_info = info_result.stdout.strip() or info_result.stderr.strip()
            except subprocess.TimeoutExpired:
                mlu_info = "[MLU Info] 获取超时"

            result = subprocess.run(
                [sys.executable, code_path],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                stderr = result.stderr
                if "ModuleNotFoundError" in stderr or "ImportError" in stderr or "No module named" in stderr:
                    lines = stderr.split("\n")
                    mod_lines = [l for l in lines if "ModuleNotFoundError" in l or "ImportError" in l or "No module named" in l]
                    mod_info = " ".join(mod_lines)
                    error_msg = f"缺少依赖模块 (ImportError): {mod_info}"
                else:
                    error_msg = f"Return code {result.returncode}: {stderr}"
                full_error = f"{mlu_info}\n\n--- Execution Result ---\n{error_msg}" if mlu_info else error_msg
                return {"efficiency": None, "error": full_error}
            output = result.stdout.strip()
            if not output:
                error_msg = f"No output. stderr: {result.stderr}"
                full_error = f"{mlu_info}\n\n--- Execution Result ---\n{error_msg}" if mlu_info else error_msg
                return {"efficiency": None, "error": full_error}
            # Prepend MLU info to the output with separator
            full_output = f"{mlu_info}\n\n--- Execution Result ---\n{output}" if mlu_info else output
            return {"efficiency": full_output}
        except subprocess.TimeoutExpired:
            return {"efficiency": None, "error": "执行超时（600秒）"}
        except OSError as e:
            return {"efficiency": None, "error": str(e)}

    result = await asyncio.to_thread(do_run)
    output = result.get("efficiency")

    if output is None:
        return {"success": False, "message": result.get('error', 'unknown'), "output": None}

    # Just return the result - no CSV update needed, show in popup only
    return {"success": True, "output": output}


@app.post("/api/jobs/{jid}/run-triton-custom")
async def run_custom_triton(jid: str, body: dict):
    """Run a custom triton code string and return its efficiency."""
    require_code_execution_enabled()
    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if row is None:
        raise HTTPException(404)

    if row.get("status") != "done":
        raise HTTPException(400, "Job not completed")

    code_content = body.get("code_content")
    if not code_content:
        raise HTTPException(400, "code_content is required")

    def do_run():
        import subprocess, sys, tempfile, os
        try:
            # Get MLU device info first
            get_mlu_info = '''
import torch_mlu
import subprocess
try:
    import torch
    mlu_version = torch_mlu.get_version()
    device_name = torch.mlu.get_device_name(0) if torch.mlu.is_available() else "N/A"
    driver_version = torch_mlu.get_driver_version()
    # Get pip-installed triton version
    pip_result = subprocess.run(["pip", "show", "triton"], capture_output=True, text=True)
    triton_version = "N/A"
    for line in pip_result.stdout.split("\\n"):
        if line.startswith("Version:"):
            triton_version = line.split(":", 1)[1].strip()
            break
    print(f"MLU Device:   {device_name}")
    print(f"Driver:       {driver_version}")
    print(f"torch_mlu:    {mlu_version}")
    print(f"Triton(pip):  {triton_version}")
except Exception as e:
    print(f"[MLU Info] Failed to get MLU info: {e}")
'''
            try:
                info_result = subprocess.run(
                    [sys.executable, "-c", get_mlu_info],
                    capture_output=True, text=True, timeout=10,
                )
                mlu_info = info_result.stdout.strip() or info_result.stderr.strip()
            except subprocess.TimeoutExpired:
                mlu_info = "[MLU Info] 获取超时"

            # Write code to a temporary file and run it
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_content)
                temp_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            finally:
                os.unlink(temp_path)

            if result.returncode != 0:
                stderr = result.stderr
                if "ModuleNotFoundError" in stderr or "ImportError" in stderr or "No module named" in stderr:
                    lines = stderr.split("\n")
                    mod_lines = [l for l in lines if "ModuleNotFoundError" in l or "ImportError" in l or "No module named" in l]
                    mod_info = " ".join(mod_lines)
                    error_msg = f"缺少依赖模块 (ImportError): {mod_info}"
                else:
                    error_msg = f"Return code {result.returncode}: {stderr}"
                full_error = f"{mlu_info}\n\n--- Execution Result ---\n{error_msg}" if mlu_info else error_msg
                return {"efficiency": None, "error": full_error}
            output = result.stdout.strip()
            if not output:
                error_msg = f"No output. stderr: {result.stderr}"
                full_error = f"{mlu_info}\n\n--- Execution Result ---\n{error_msg}" if mlu_info else error_msg
                return {"efficiency": None, "error": full_error}
            # Prepend MLU info to the output with separator
            full_output = f"{mlu_info}\n\n--- Execution Result ---\n{output}" if mlu_info else output
            return {"efficiency": full_output}
        except subprocess.TimeoutExpired:
            return {"efficiency": None, "error": "执行超时（600秒）"}
        except OSError as e:
            return {"efficiency": None, "error": str(e)}

    result = await asyncio.to_thread(do_run)
    output = result.get("efficiency")

    if output is None:
        return {"success": False, "message": result.get('error', 'unknown'), "output": None}

    return {"success": True, "output": output}


@app.get("/api/jobs/{jid}/files/{slot}")
async def get_job_file(jid: str, slot: str, format: Optional[str] = None):
    """Serve trace file (a or b) for Perfetto/download.

    Query params:
      format=json  — decompress .gz content on the fly and serve as raw JSON
    """
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")

    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if not row:
        raise HTTPException(404)

    if slot not in ("a", "b"):
        raise HTTPException(400, "slot must be 'a' or 'b'")

    # Prefer gzip path if available, fall back to json path
    gzip_path = row.get(f"file_{slot}_gzip_path")
    json_path = row.get(f"file_{slot}_path")
    file_path = gzip_path if gzip_path else json_path

    # Compare-from-history jobs have no file stored directly;
    # resolve via the corresponding source job (slot a → source_job_a, slot b → source_job_b)
    if not file_path:
        src_jid = row.get(f"source_job_{slot}")
        if src_jid:
            db2 = await get_db()
            cur2 = await db2.execute("SELECT file_a_path, file_a_gzip_path FROM jobs WHERE id=?", (src_jid,))
            src = await row_to_dict(await cur2.fetchone())
            await db2.close()
            if src:
                gzip_path = src.get("file_a_gzip_path")
                json_path = src.get("file_a_path")
                file_path = gzip_path if gzip_path else json_path

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    # If the client requests raw JSON and we only have .gz, extract on the fly.
    if format == "json" and file_path.endswith(".gz"):
        buf = io.BytesIO()
        if tarfile.is_tarfile(file_path):
            with tarfile.open(file_path, "r:*") as tar:
                members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json")]
                if not members:
                    raise HTTPException(400, "Archive does not contain a JSON trace")
                member = max(members, key=lambda m: m.size)
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise HTTPException(400, "Unable to read JSON trace from archive")
                shutil.copyfileobj(extracted, buf)
        else:
            with gzip.open(file_path, "rb") as gz:
                shutil.copyfileobj(gz, buf)
        buf.seek(0)
        filename = row.get(f"file_{slot}_name") or f"trace_{slot}.json"
        return StreamingResponse(buf, media_type="application/json",
                                 headers={"Content-Disposition": f'inline; filename="{filename}"'})

    # Stream file directly — avoids loading entire file into memory
    media_type = "application/gzip" if file_path.endswith(".gz") else "application/json"
    filename = row.get(f"file_{slot}_name") or f"trace_{slot}.json"
    return FileResponse(file_path, media_type=media_type, filename=filename)


@app.delete("/api/jobs/{jid}/files/{slot}", status_code=204)
async def delete_job_file(jid: str, slot: str):
    """Delete the stored trace file (a or b) for a job."""
    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)
    if slot not in ("a", "b"):
        await db.close()
        raise HTTPException(400, "slot must be 'a' or 'b'")

    for col in (f"file_{slot}_path", f"file_{slot}_gzip_path"):
        path = row.get(col)
        if path and os.path.exists(path):
            os.remove(path)
        await db.execute(f"UPDATE jobs SET {col}=NULL WHERE id=?", (jid,))

    await db.execute(f"UPDATE jobs SET file_{slot}_exists=0 WHERE id=?", (jid,))
    await db.commit()
    await db.close()


@app.get("/api/jobs/{jid}/triton-code/{path:path}")
async def get_triton_code(jid: str, path: str):
    """Serve triton code file for display in browser."""
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")

    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if not row:
        raise HTTPException(404)

    # Prevent path traversal - ensure path is within result_dir
    full_path = os.path.normpath(os.path.join(result_dir(jid), path))
    if not full_path.startswith(os.path.abspath(result_dir(jid))):
        raise HTTPException(400, "Invalid path")

    if not os.path.exists(full_path):
        raise HTTPException(404)

    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    return {"content": content, "filename": os.path.basename(path)}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Trace Analyzer Web Server")
    parser.add_argument("--port", type=int, default=8181, help="Port to listen on (default: 8181)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--no-download", action="store_true",
                        help="Disable downloading of uploaded trace files (default: download allowed)")
    cli_args = parser.parse_args()

    if cli_args.no_download:
        os.environ["TRACE_NO_DOWNLOAD"] = "1"

    uvicorn.run("server:app", host=cli_args.host, port=cli_args.port, reload=False)
