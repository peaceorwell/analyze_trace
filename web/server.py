import argparse
import asyncio
import contextlib
import csv
import gzip
import io
import os
import shutil
import sys
import tarfile
import types
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiofiles
from fastapi import BackgroundTasks, Cookie, FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from werkzeug.security import check_password_hash

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analyze_trace import compute_avgs, parse_trace, run_triton_code_and_get_efficiency  # noqa: E402

from db import get_db, init_db, row_to_dict  # noqa: E402

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# Configured at startup via CLI; read-only after that
ALLOW_FILE_DOWNLOAD = os.environ.get("TRACE_NO_DOWNLOAD", "") == ""

# Cookie security settings
# Set FORCE_SECURE=1 or USE_HTTPS=1 to enforce secure cookies (for HTTPS deployments)
_is_https = os.environ.get("FORCE_SECURE", "") or os.environ.get("USE_HTTPS", "") or os.environ.get("HTTPS", "")
_is_production = os.environ.get("PRODUCTION", "") == "1"
COOKIE_SECURE = _is_https.lower() in ("1", "true", "yes") or (_is_production and not os.environ.get("DEV_MODE"))
COOKIE_SAMESITE = os.environ.get("COOKIE_SAMESITE", "lax")  # 'strict', 'lax', or 'none'
COOKIE_MAX_AGE = 365 * 24 * 60 * 60  # 1 year in seconds


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


# ── User token helpers ─────────────────────────────────────────────────────────

async def get_or_create_user(user_token: Optional[str] = None, x_user_token: Optional[str] = None) -> str:
    """Get existing user or create new one. Returns user_token.

    Accepts token from cookie or X-User-Token header (localStorage fallback).
    Cookie takes precedence over X-User-Token header.

    If no valid token is provided, creates a new user in both cookie and header context.
    """
    # Prefer cookie token, fall back to header token
    token = user_token if user_token else x_user_token
    db = await get_db()
    try:
        if token:
            # Token provided: ensure it exists in DB
            await db.execute("INSERT OR IGNORE INTO users(user_token) VALUES(?)", (token,))
            await db.commit()
            return token
        # No token provided: create new one (instead of returning a random user)
        new_token = str(uuid.uuid4())
        await db.execute("INSERT INTO users(user_token) VALUES(?)", (new_token,))
        await db.commit()
        return new_token
    finally:
        await db.close()


def job_dir(job_id: str) -> str:
    return os.path.join(STORAGE_DIR, job_id)


def result_dir(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "results")


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


# ── Synchronous analysis (runs in thread pool, must not await) ────────────────

def _run_sync_analysis(job, kernel_types, rdir, path_a, path_b, name_a, name_b):
    """All blocking I/O lives here so the event loop stays free."""
    from analyze_trace import (compute_avgs, parse_trace,
                               print_step_summary, print_kernel_type_breakdown, print_top_kernels,
                               write_single, print_comparison, write_comparison)

    buf = io.StringIO()
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
            path_a = src_a["file_a_path"]
            path_b = src_b["file_a_path"]
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
    return {"allow_file_download": ALLOW_FILE_DOWNLOAD}


# ── Routes: auth ──────────────────────────────────────────────────────────────

@app.post("/api/auth/guest", response_model=dict)
async def guest_login(response: JSONResponse, user_token: Optional[str] = Cookie(None)):
    """Get existing user token or create new one. Sets HttpOnly cookie."""
    token = await get_or_create_user(user_token)
    response.set_cookie(
        key="user_token",
        value=token,
        httponly=True,                    # Prevents XSS from accessing cookie
        samesite=COOKIE_SAMESITE,         # CSRF protection
        secure=COOKIE_SECURE,             # Requires HTTPS
        max_age=COOKIE_MAX_AGE,
        path="/",
        # __Host- prefix requires secure=True and provides additional security
        # (commented out to maintain compatibility with HTTP dev environments)
        # cookie_prefix="__Host-" if COOKIE_SECURE else "",
    )
    return {"user_token": token}

# ── Routes: projects ──────────────────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects():
    # No account system - all projects are public, return everything
    db = await get_db()
    rows = await (await db.execute("""
        SELECT * FROM projects
        ORDER BY created_at DESC
    """)).fetchall()
    await db.close()
    return [dict(r) for r in rows]


@app.post("/api/projects", status_code=201)
async def create_project(body: dict, user_token: Optional[str] = Cookie(None)):
    pid = str(uuid.uuid4())
    token = await get_or_create_user(user_token)
    db = await get_db()
    await db.execute(
        "INSERT INTO projects(id, user_token, name, description, is_public) VALUES(?,?,?,?,?)",
        (pid, token, body.get("name", "新项目"), body.get("description", ""), 1),
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await cursor.fetchone()
    await db.close()
    return dict(row)


@app.put("/api/projects/{pid}")
async def update_project(pid: str, body: dict, user_token: Optional[str] = Cookie(None)):
    token = await get_or_create_user(user_token)
    db = await get_db()

    # Verify ownership
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)
    if row.get("user_token") != token:
        await db.close()
        raise HTTPException(403, "Not the project owner")

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
async def delete_project(pid: str, user_token: Optional[str] = Cookie(None)):
    await get_or_create_user(user_token)  # Ensure user exists, but no ownership check (public system)
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
async def list_deleted_projects(user_token: Optional[str] = Cookie(None)):
    """List recoverable projects deleted within the last 10 days."""
    await get_or_create_user(user_token)  # Ensure user exists, but no ownership check (public system)
    db = await get_db()
    rows = await (await db.execute("""
        SELECT * FROM deleted_projects
        WHERE deleted_at >= datetime('now', '-10 days')
        ORDER BY deleted_at DESC
    """)).fetchall()
    await db.close()
    return [dict(r) for r in rows]


@app.post("/api/deleted-projects/{pid}/restore", status_code=200)
async def restore_project(pid: str, user_token: Optional[str] = Cookie(None)):
    """Restore a project deleted within the last 10 days."""
    await get_or_create_user(user_token)  # Ensure user exists, but no ownership check (public system)
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

    # Restore the project - first ensure user exists
    token = row.get("user_token")
    if token:
        await db.execute("INSERT OR IGNORE INTO users(user_token) VALUES(?)", (token,))

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
async def permanently_delete_project(pid: str, user_token: Optional[str] = Cookie(None)):
    """Permanently delete a project from recovery list (without restoring)."""
    await get_or_create_user(user_token)  # Ensure user exists, but no ownership check (public system)
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

@app.get("/api/jobs")
async def list_jobs(
    project_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    # No account system - all jobs are visible to everyone
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
    return {"data": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}


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
    user_token: Optional[str] = Cookie(None),
):
    token = await get_or_create_user(user_token)
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
        """INSERT INTO jobs(id, project_id, user_token, label, mode,
               file_a_name, file_a_path, file_a_gzip_path, file_b_name, file_b_path, file_b_gzip_path,
               kernel_types, save_triton_csv, save_triton_code)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (jid, project_id or None, token, eff_label, mode,
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
async def compare_jobs(body: dict, background_tasks: BackgroundTasks, user_token: Optional[str] = Cookie(None)):
    token = await get_or_create_user(user_token)
    job_id_a = body.get("job_id_a")
    job_id_b = body.get("job_id_b")
    if not job_id_a or not job_id_b:
        raise HTTPException(400, "job_id_a and job_id_b are required")

    db = await get_db()

    # Check user has access to both source jobs
    cursor_a = await db.execute("SELECT * FROM jobs WHERE id=?", (job_id_a,))
    src_a = await row_to_dict(await cursor_a.fetchone())
    cursor_b = await db.execute("SELECT * FROM jobs WHERE id=?", (job_id_b,))
    src_b = await row_to_dict(await cursor_b.fetchone())

    if not src_a or not src_b:
        await db.close()
        raise HTTPException(404, "Source job not found")

    # Verify user owns both source jobs
    if src_a.get("user_token") != token or src_b.get("user_token") != token:
        await db.close()
        raise HTTPException(403, "No access to source job")

    if not src_a.get("file_a_exists") or not src_b.get("file_a_exists"):
        await db.close()
        raise HTTPException(409, "Source file has been deleted")

    jid = str(uuid.uuid4())
    kernel_types = body.get("kernel_types", "gemm,embedding,pool")
    eff_label = body.get("label") or f"{src_a['label']} vs {src_b['label']}"

    await db.execute(
        """INSERT INTO jobs(id, project_id, user_token, label, mode,
               file_a_name, file_b_name,
               source_job_a, source_job_b,
               kernel_types, save_triton_csv, save_triton_code)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (jid, body.get("project_id"), token, eff_label, "compare",
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
async def get_job(jid: str, user_token: Optional[str] = Cookie(None)):
    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if row is None:
        raise HTTPException(404)

    # No account system - all jobs are public
    job = dict(row)
    if job["status"] == "done":
        job["results"] = collect_results(jid)
    return job


@app.patch("/api/jobs/{jid}")
async def patch_job(jid: str, body: dict, user_token: Optional[str] = Cookie(None), x_user_token: Optional[str] = Header(None)):
    token = await get_or_create_user(user_token, x_user_token)
    db = await get_db()

    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)

    # Only owner can update
    if row.get("user_token") != token:
        await db.close()
        raise HTTPException(403, "Not the job owner")

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
async def delete_job(jid: str, user_token: Optional[str] = Cookie(None)):
    await get_or_create_user(user_token)  # Ensure user exists, but no ownership check (public system)
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
async def run_job_triton(jid: str, user_token: Optional[str] = Cookie(None)):
    """Run triton code files and append local efficiency to CSV."""
    token = await get_or_create_user(user_token)
    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if row is None:
        raise HTTPException(404)

    if row.get("user_token") != token:
        raise HTTPException(403, "Not the job owner")

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
async def clear_inductor_cache(jid: str, user_token: Optional[str] = Cookie(None), x_user_token: Optional[str] = Header(None)):
    """Clear the torchinductor cache for a job's triton runs."""
    import shutil, glob

    await get_or_create_user(user_token, x_user_token)  # Ensure user exists, but no ownership check (public system)
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
async def run_single_triton(jid: str, body: dict, user_token: Optional[str] = Cookie(None)):
    """Run a single triton code file and return its efficiency."""
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
            info_result = subprocess.run(
                [sys.executable, "-c", get_mlu_info],
                capture_output=True, text=True, timeout=10,
            )
            mlu_info = info_result.stdout.strip() or info_result.stderr.strip()

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
async def run_custom_triton(jid: str, body: dict, user_token: Optional[str] = Cookie(None)):
    """Run a custom triton code string and return its efficiency."""
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
            info_result = subprocess.run(
                [sys.executable, "-c", get_mlu_info],
                capture_output=True, text=True, timeout=10,
            )
            mlu_info = info_result.stdout.strip() or info_result.stderr.strip()

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
async def get_job_file(jid: str, slot: str, user_token: Optional[str] = Cookie(None)):
    """Serve trace file (a or b) for Perfetto. Returns the raw file content."""
    token = await get_or_create_user(user_token)
    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if not row:
        raise HTTPException(404)

    if row.get("user_token") != token:
        raise HTTPException(403, "Not the job owner")

    if slot not in ("a", "b"):
        raise HTTPException(400, "slot must be 'a' or 'b'")

    # Prefer gzip path if available, fall back to json path
    gzip_path = row.get(f"file_{slot}_gzip_path")
    json_path = row.get(f"file_{slot}_path")

    file_path = gzip_path if gzip_path else json_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    # Send raw file (gzip or json) as-is to Perfetto
    media_type = "application/gzip" if file_path.endswith(".gz") else "application/json"

    from starlette.responses import Response

    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(500, f"Failed to read file: {e}")

    filename = row.get(f"file_{slot}_name") or f"trace_{slot}.json"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content, media_type=media_type, headers=headers)


@app.get("/api/jobs/{jid}/triton-code/{path:path}")
async def get_triton_code(jid: str, path: str, user_token: Optional[str] = Cookie(None)):
    """Serve triton code file for display in browser."""
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")

    token = await get_or_create_user(user_token)
    db = await get_db()
    cursor = await db.execute("""
        SELECT j.*, p.user_token as proj_owner, p.is_public, p.password_hash
        FROM jobs j
        LEFT JOIN projects p ON j.project_id = p.id
        WHERE j.id=?
    """, (jid,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if not row:
        raise HTTPException(404)

    def can_access_job(job):
        if job.get("user_token") == token:
            return True
        is_public = job.get("is_public")
        has_password = job.get("password_hash")
        return bool(is_public) and not has_password

    if not can_access_job(row):
        raise HTTPException(403, "No access to this file")

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