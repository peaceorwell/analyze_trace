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
from fastapi import BackgroundTasks, Cookie, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analyze_trace import compute_avgs, parse_trace  # noqa: E402

from db import get_db, init_db, row_to_dict  # noqa: E402

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# Configured at startup via CLI; read-only after that
ALLOW_FILE_DOWNLOAD = os.environ.get("TRACE_NO_DOWNLOAD", "") == ""


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


# ── User token helpers ─────────────────────────────────────────────────────────

async def get_or_create_user(user_token: Optional[str]) -> str:
    """Get existing user or create new one. Returns user_token."""
    if not user_token:
        user_token = str(uuid.uuid4())
    db = await get_db()
    try:
        await db.execute("INSERT OR IGNORE INTO users(user_token) VALUES(?)", (user_token,))
        await db.commit()
    finally:
        await db.close()
    return user_token


async def verify_project_access(db, project_id: str, user_token: str, password: Optional[str] = None) -> bool:
    """Verify user has access to project. Returns True if access granted."""
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (project_id,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        return False

    # Owner always has access
    if row.get("user_token") == user_token:
        return True

    # Public projects are accessible to all
    if row.get("is_public") and not row.get("password_hash"):
        return True

    # Password-protected projects require correct password
    if row.get("password_hash"):
        if not password:
            return False
        return check_password_hash(row["password_hash"], password)

    return False


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
    return files


# ── Synchronous analysis (runs in thread pool, must not await) ────────────────

def _run_sync_analysis(job, kernel_types, rdir, path_a, path_b, name_a, name_b):
    """All blocking I/O lives here so the event loop stays free."""
    from analyze_trace import (compute_avgs, parse_trace,
                               print_step_summary, print_kernel_type_breakdown, write_single,
                               print_comparison, write_comparison)

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
            write_single(data, fake_args)
        else:
            data_a = compute_avgs(parse_trace(path_a, kernel_types), kernel_types)
            data_b = compute_avgs(parse_trace(path_b, kernel_types), kernel_types)
            fake_args = types.SimpleNamespace(output_dir=rdir)
            label_a = name_a or os.path.basename(path_a)
            label_b = name_b or os.path.basename(path_b)
            print_comparison(data_a, data_b, label_a, label_b)
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
        httponly=True,
        samesite="lax",
        secure=False,  # Set True in production with HTTPS
        max_age=365 * 24 * 60 * 60,  # 1 year
    )
    return {"user_token": token}


@app.post("/api/auth/verify-project", response_model=dict)
async def verify_project(request: Request, body: dict):
    """Verify password for accessing a project. Returns True if password is correct."""
    project_id = body.get("project_id")
    password = body.get("password", "")

    if not project_id:
        raise HTTPException(400, "project_id is required")

    db = await get_db()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (project_id,))
    row = await row_to_dict(await cursor.fetchone())
    await db.close()

    if not row:
        raise HTTPException(404, "Project not found")

    # No password required
    if not row.get("password_hash"):
        return {"verified": True}

    # Verify password
    if password and check_password_hash(row["password_hash"], password):
        return {"verified": True}

    return {"verified": False}


# ── Routes: projects ──────────────────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects(user_token: Optional[str] = Cookie(None)):
    db = await get_db()
    token = await get_or_create_user(user_token)

    # Get user's own projects + public projects without password
    rows = await (await db.execute("""
        SELECT * FROM projects
        WHERE user_token = ? OR (is_public = 1 AND password_hash IS NULL)
        ORDER BY created_at DESC
    """, (token,))).fetchall()
    await db.close()
    return [dict(r) for r in rows]


@app.post("/api/projects", status_code=201)
async def create_project(body: dict, user_token: Optional[str] = Cookie(None)):
    pid = str(uuid.uuid4())
    token = await get_or_create_user(user_token)
    db = await get_db()
    await db.execute(
        "INSERT INTO projects(id, user_token, name, description) VALUES(?,?,?,?)",
        (pid, token, body.get("name", "新项目"), body.get("description", "")),
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


@app.put("/api/projects/{pid}/password")
async def set_project_password(
    pid: str,
    body: dict,
    user_token: Optional[str] = Cookie(None)
):
    """Set or remove password protection on a project."""
    token = await get_or_create_user(user_token)
    password = body.get("password")  # None or empty = remove password

    db = await get_db()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)
    if row.get("user_token") != token:
        await db.close()
        raise HTTPException(403, "Not the project owner")

    password_hash = generate_password_hash(password) if password else None
    await db.execute(
        "UPDATE projects SET password_hash=? WHERE id=?",
        (password_hash, pid),
    )
    await db.commit()
    await db.close()
    return {"success": True}


@app.put("/api/projects/{pid}/public")
async def set_project_public(
    pid: str,
    body: dict,
    user_token: Optional[str] = Cookie(None)
):
    """Toggle public status of a project."""
    token = await get_or_create_user(user_token)
    is_public = body.get("is_public", False)

    db = await get_db()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)
    if row.get("user_token") != token:
        await db.close()
        raise HTTPException(403, "Not the project owner")

    await db.execute(
        "UPDATE projects SET is_public=? WHERE id=?",
        (1 if is_public else 0, pid),
    )
    await db.commit()
    await db.close()
    return {"success": True}


@app.delete("/api/projects/{pid}", status_code=204)
async def delete_project(pid: str, user_token: Optional[str] = Cookie(None)):
    token = await get_or_create_user(user_token)
    db = await get_db()

    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)
    if row.get("user_token") != token:
        await db.close()
        raise HTTPException(403, "Not the project owner")

    # Delete all jobs in the project
    await db.execute("UPDATE jobs SET project_id=NULL WHERE project_id=?", (pid,))
    await db.execute("DELETE FROM projects WHERE id=?", (pid,))
    await db.commit()
    await db.close()


# ── Routes: jobs ──────────────────────────────────────────────────────────────

@app.get("/api/jobs")
async def list_jobs(
    project_id: Optional[str] = None,
    user_token: Optional[str] = Cookie(None)
):
    token = await get_or_create_user(user_token)
    db = await get_db()

    # Build query to filter by user's own jobs + jobs in accessible projects
    if project_id == "__none__":
        # Jobs without project - only user's own
        rows = await (await db.execute(
            "SELECT * FROM jobs WHERE user_token=? AND project_id IS NULL ORDER BY created_at DESC",
            (token,)
        )).fetchall()
    elif project_id:
        # Specific project - check access first
        can_access = await verify_project_access(db, project_id, token)
        if not can_access:
            await db.close()
            raise HTTPException(403, "No access to this project")
        rows = await (await db.execute(
            "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC", (project_id,)
        )).fetchall()
    else:
        # All jobs - user's own + accessible projects
        rows = await (await db.execute("""
            SELECT j.* FROM jobs j
            LEFT JOIN projects p ON j.project_id = p.id
            WHERE j.user_token = ?
               OR (p.is_public = 1 AND p.password_hash IS NULL)
            ORDER BY j.created_at DESC
        """, (token,))).fetchall()

    await db.close()
    return [dict(r) for r in rows]


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
    cursor_a = await db.execute("SELECT j.*, p.user_token as proj_owner, p.is_public, p.password_hash FROM jobs j LEFT JOIN projects p ON j.project_id = p.id WHERE j.id=?", (job_id_a,))
    src_a = await row_to_dict(await cursor_a.fetchone())
    cursor_b = await db.execute("SELECT j.*, p.user_token as proj_owner, p.is_public, p.password_hash FROM jobs j LEFT JOIN projects p ON j.project_id = p.id WHERE j.id=?", (job_id_b,))
    src_b = await row_to_dict(await cursor_b.fetchone())

    if not src_a or not src_b:
        await db.close()
        raise HTTPException(404, "Source job not found")

    # Verify access to source jobs
    def can_access_job(job):
        if job.get("user_token") == token:
            return True
        proj_owner = job.get("proj_owner")
        is_public = job.get("is_public")
        has_password = job.get("password_hash")
        return bool(is_public) and not has_password

    if not can_access_job(src_a) or not can_access_job(src_b):
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

    if row is None:
        raise HTTPException(404)

    # Check access
    def can_access_job(job):
        if job.get("user_token") == token:
            return True
        proj_owner = job.get("proj_owner")
        is_public = job.get("is_public")
        has_password = job.get("password_hash")
        return bool(is_public) and not has_password

    if not can_access_job(row):
        raise HTTPException(403, "No access to this job")

    job = dict(row)
    if job["status"] == "done":
        job["results"] = collect_results(jid)
    return job


@app.patch("/api/jobs/{jid}")
async def patch_job(jid: str, body: dict, user_token: Optional[str] = Cookie(None)):
    token = await get_or_create_user(user_token)
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
    token = await get_or_create_user(user_token)
    db = await get_db()

    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)

    # Only owner can delete
    if row.get("user_token") != token:
        await db.close()
        raise HTTPException(403, "Not the job owner")

    # Remove all files on disk
    jdir = job_dir(jid)
    if os.path.exists(jdir):
        shutil.rmtree(jdir)

    await db.execute("DELETE FROM jobs WHERE id=?", (jid,))
    await db.commit()
    await db.close()


@app.delete("/api/jobs/{jid}/files/{which}", status_code=204)
async def delete_job_file(jid: str, which: str, user_token: Optional[str] = Cookie(None)):
    token = await get_or_create_user(user_token)
    if which not in ("a", "b"):
        raise HTTPException(400, "which must be 'a' or 'b'")

    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await row_to_dict(await cursor.fetchone())
    if not row:
        await db.close()
        raise HTTPException(404)

    # Only owner can delete files
    if row.get("user_token") != token:
        await db.close()
        raise HTTPException(403, "Not the job owner")

    path_col = f"file_{which}_path"
    gzip_col = f"file_{which}_gzip_path"
    exists_col = f"file_{which}_exists"
    fpath = row.get(path_col)
    gzip_path = row.get(gzip_col)
    if fpath and os.path.exists(fpath):
        os.remove(fpath)
    if gzip_path and os.path.exists(gzip_path):
        os.remove(gzip_path)
    await db.execute(f"UPDATE jobs SET {exists_col}=0 WHERE id=?", (jid,))
    await db.commit()
    await db.close()


@app.get("/api/jobs/{jid}/files/{which}")
async def download_job_file(jid: str, which: str, user_token: Optional[str] = Cookie(None)):
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")
    if which not in ("a", "b"):
        raise HTTPException(400, "which must be 'a' or 'b'")

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

    # Check access
    def can_access_job(job):
        if job.get("user_token") == token:
            return True
        is_public = job.get("is_public")
        has_password = job.get("password_hash")
        return bool(is_public) and not has_password

    if not can_access_job(row):
        raise HTTPException(403, "No access to this file")

    gzip_path = row.get(f"file_{which}_gzip_path")
    json_path = row.get(f"file_{which}_path")

    # If original was .gz, serve the gzipped file (preserves compression for perfetto too)
    if gzip_path and os.path.exists(gzip_path):
        fname = row.get(f"file_{which}_name") or f"trace_{which}.json.gz"
        return FileResponse(gzip_path, filename=fname, media_type="application/json")

    # Otherwise serve the extracted JSON
    fname = row.get(f"file_{which}_name") or f"trace_{which}.json"
    if fname.lower().endswith(".gz"):
        fname = fname[:-3]  # serve extracted JSON with .json extension
    if not json_path or not os.path.exists(json_path):
        raise HTTPException(404, "File not found or already deleted")
    return FileResponse(json_path, filename=fname, media_type="application/json")


@app.get("/api/jobs/{jid}/results/{filename}")
async def download_result(jid: str, filename: str, user_token: Optional[str] = Cookie(None)):
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")
    # Prevent path traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(400)

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

    # Check access
    def can_access_job(job):
        if job.get("user_token") == token:
            return True
        is_public = job.get("is_public")
        has_password = job.get("password_hash")
        return bool(is_public) and not has_password

    if not can_access_job(row):
        raise HTTPException(403, "No access to this file")

    path = os.path.join(result_dir(jid), filename)
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, filename=filename, media_type="text/csv")


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