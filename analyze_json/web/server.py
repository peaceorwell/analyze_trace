import argparse
import asyncio
import contextlib
import csv
import io
import os
import sys
import types
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analyze_trace import compute_avgs, parse_trace  # noqa: E402

from db import get_db, init_db, row_to_dict  # noqa: E402

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# Configured at startup via CLI; read-only after that
ALLOW_FILE_DOWNLOAD: bool = True


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def job_dir(job_id: str) -> str:
    return os.path.join(STORAGE_DIR, job_id)


def result_dir(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "results")


async def save_upload(upload: UploadFile, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(1 << 20):  # 1 MB chunks
            await f.write(chunk)


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

        job = await row_to_dict(
            await (await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))).fetchone()
        )

        kernel_types = [p for p in (job["kernel_types"] or "").split(",") if p]
        rdir = result_dir(job_id)
        os.makedirs(rdir, exist_ok=True)

        # Resolve source paths for compare-from-history (needs DB, must happen before thread)
        if job["mode"] == "compare" and not job["file_a_path"]:
            src_a = await row_to_dict(
                await (await db.execute("SELECT * FROM jobs WHERE id=?", (job["source_job_a"],))).fetchone()
            )
            src_b = await row_to_dict(
                await (await db.execute("SELECT * FROM jobs WHERE id=?", (job["source_job_b"],))).fetchone()
            )
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


# ── Routes: projects ──────────────────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects():
    db = await get_db()
    rows = await (await db.execute("SELECT * FROM projects ORDER BY created_at DESC")).fetchall()
    await db.close()
    return [dict(r) for r in rows]


@app.post("/api/projects", status_code=201)
async def create_project(body: dict):
    pid = str(uuid.uuid4())
    db = await get_db()
    await db.execute(
        "INSERT INTO projects(id, name, description) VALUES(?,?,?)",
        (pid, body.get("name", "新项目"), body.get("description", "")),
    )
    await db.commit()
    row = await (await db.execute("SELECT * FROM projects WHERE id=?", (pid,))).fetchone()
    await db.close()
    return dict(row)


@app.put("/api/projects/{pid}")
async def update_project(pid: str, body: dict):
    db = await get_db()
    await db.execute(
        "UPDATE projects SET name=?, description=? WHERE id=?",
        (body.get("name"), body.get("description", ""), pid),
    )
    await db.commit()
    row = await (await db.execute("SELECT * FROM projects WHERE id=?", (pid,))).fetchone()
    await db.close()
    if row is None:
        raise HTTPException(404)
    return dict(row)


@app.delete("/api/projects/{pid}", status_code=204)
async def delete_project(pid: str):
    db = await get_db()
    await db.execute("UPDATE jobs SET project_id=NULL WHERE project_id=?", (pid,))
    await db.execute("DELETE FROM projects WHERE id=?", (pid,))
    await db.commit()
    await db.close()


# ── Routes: jobs ──────────────────────────────────────────────────────────────

@app.get("/api/jobs")
async def list_jobs(project_id: Optional[str] = None):
    db = await get_db()
    if project_id == "__none__":
        rows = await (await db.execute(
            "SELECT * FROM jobs WHERE project_id IS NULL ORDER BY created_at DESC"
        )).fetchall()
    elif project_id:
        rows = await (await db.execute(
            "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC", (project_id,)
        )).fetchall()
    else:
        rows = await (await db.execute("SELECT * FROM jobs ORDER BY created_at DESC")).fetchall()
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
):
    jid = str(uuid.uuid4())
    jdir = job_dir(jid)

    path_a = os.path.join(jdir, "trace_a.json")
    await save_upload(file_a, path_a)

    path_b = None
    name_b = None
    mode = "single"
    if file_b and file_b.filename:
        path_b = os.path.join(jdir, "trace_b.json")
        await save_upload(file_b, path_b)
        name_b = file_b.filename
        mode = "compare"

    eff_label = label or file_a.filename or jid

    db = await get_db()
    await db.execute(
        """INSERT INTO jobs(id, project_id, label, mode,
               file_a_name, file_a_path, file_b_name, file_b_path,
               kernel_types, save_triton_csv, save_triton_code)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (jid, project_id or None, eff_label, mode,
         file_a.filename, path_a, name_b, path_b,
         kernel_types, int(save_triton_csv), int(save_triton_code)),
    )
    await db.commit()
    row = await (await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))).fetchone()
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
    src_a = await row_to_dict(
        await (await db.execute("SELECT * FROM jobs WHERE id=?", (job_id_a,))).fetchone()
    )
    src_b = await row_to_dict(
        await (await db.execute("SELECT * FROM jobs WHERE id=?", (job_id_b,))).fetchone()
    )
    if not src_a or not src_b:
        await db.close()
        raise HTTPException(404, "Source job not found")
    if not src_a.get("file_a_exists") or not src_b.get("file_a_exists"):
        await db.close()
        raise HTTPException(409, "Source file has been deleted")

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
    row = await (await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))).fetchone()
    await db.close()

    background_tasks.add_task(run_analysis, jid)
    return dict(row)


@app.get("/api/jobs/{jid}")
async def get_job(jid: str):
    db = await get_db()
    row = await (await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))).fetchone()
    await db.close()
    if row is None:
        raise HTTPException(404)
    job = dict(row)
    if job["status"] == "done":
        job["results"] = collect_results(jid)
    return job


@app.patch("/api/jobs/{jid}")
async def patch_job(jid: str, body: dict):
    db = await get_db()
    if "label" in body:
        await db.execute("UPDATE jobs SET label=? WHERE id=?", (body["label"], jid))
    if "project_id" in body:
        await db.execute("UPDATE jobs SET project_id=? WHERE id=?",
                         (body["project_id"] or None, jid))
    await db.commit()
    row = await (await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))).fetchone()
    await db.close()
    if row is None:
        raise HTTPException(404)
    return dict(row)


@app.delete("/api/jobs/{jid}", status_code=204)
async def delete_job(jid: str):
    db = await get_db()
    row = await (await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))).fetchone()
    await db.close()
    if row is None:
        raise HTTPException(404)
    # Remove all files on disk
    jdir = job_dir(jid)
    if os.path.exists(jdir):
        import shutil
        shutil.rmtree(jdir)
    db = await get_db()
    await db.execute("DELETE FROM jobs WHERE id=?", (jid,))
    await db.commit()
    await db.close()


@app.delete("/api/jobs/{jid}/files/{which}", status_code=204)
async def delete_job_file(jid: str, which: str):
    if which not in ("a", "b"):
        raise HTTPException(400, "which must be 'a' or 'b'")
    db = await get_db()
    row = await row_to_dict(
        await (await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))).fetchone()
    )
    if row is None:
        await db.close()
        raise HTTPException(404)

    path_col = f"file_{which}_path"
    exists_col = f"file_{which}_exists"
    fpath = row.get(path_col)
    if fpath and os.path.exists(fpath):
        os.remove(fpath)
    await db.execute(f"UPDATE jobs SET {exists_col}=0 WHERE id=?", (jid,))
    await db.commit()
    await db.close()


@app.get("/api/jobs/{jid}/files/{which}")
async def download_job_file(jid: str, which: str):
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")
    if which not in ("a", "b"):
        raise HTTPException(400, "which must be 'a' or 'b'")
    db = await get_db()
    row = await row_to_dict(
        await (await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))).fetchone()
    )
    await db.close()
    if row is None:
        raise HTTPException(404)
    fpath = row.get(f"file_{which}_path")
    fname = row.get(f"file_{which}_name") or f"trace_{which}.json"
    if not fpath or not os.path.exists(fpath):
        raise HTTPException(404, "File not found or already deleted")
    return FileResponse(fpath, filename=fname, media_type="application/json")


@app.get("/api/jobs/{jid}/results/{filename}")
async def download_result(jid: str, filename: str):
    # Prevent path traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(400)
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

    global ALLOW_FILE_DOWNLOAD
    ALLOW_FILE_DOWNLOAD = not cli_args.no_download

    uvicorn.run("server:app", host=cli_args.host, port=cli_args.port, reload=False)
