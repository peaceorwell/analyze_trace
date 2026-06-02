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
from contextlib import asynccontextmanager
from typing import Optional

import aiofiles
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trace_analyzer import compute_avgs, parse_trace, run_triton_code_and_get_efficiency  # noqa: E402

from db import get_db, init_db, row_to_dict  # noqa: E402

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# Configured at startup via CLI; read-only after that
ALLOW_FILE_DOWNLOAD = os.environ.get("TRACE_NO_DOWNLOAD", "") == ""
ALLOW_CODE_EXECUTION = os.environ.get("TRACE_ENABLE_CODE_EXEC", "") == "1"
ANALYSIS_CONCURRENCY = max(1, int(os.environ.get("TRACE_ANALYSIS_CONCURRENCY", "1")))
analysis_queue: asyncio.Queue[str] = asyncio.Queue()
analysis_workers: list[asyncio.Task] = []

# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await mark_interrupted_jobs()
    await refresh_storage_cache_for_all_jobs()
    await enqueue_pending_jobs()
    start_analysis_workers()
    try:
        yield
    finally:
        await stop_analysis_workers()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


async def mark_interrupted_jobs():
    """Fail jobs that were actively running when the previous process exited."""
    db = await get_db()
    try:
        await db.execute("""
            UPDATE jobs
            SET status='error',
                error_msg='Server restarted before this analysis completed'
            WHERE status='running'
        """)
        await db.commit()
    finally:
        await db.close()


async def enqueue_pending_jobs():
    while not analysis_queue.empty():
        try:
            analysis_queue.get_nowait()
            analysis_queue.task_done()
        except asyncio.QueueEmpty:
            break
    db = await get_db()
    try:
        rows = await (
            await db.execute("SELECT id FROM jobs WHERE status='pending' ORDER BY created_at")
        ).fetchall()
    finally:
        await db.close()
    for row in rows:
        await analysis_queue.put(row["id"])


def start_analysis_workers():
    for index in range(ANALYSIS_CONCURRENCY):
        analysis_workers.append(asyncio.create_task(_analysis_worker(index)))


async def stop_analysis_workers():
    for task in analysis_workers:
        task.cancel()
    if analysis_workers:
        await asyncio.gather(*analysis_workers, return_exceptions=True)
    analysis_workers.clear()


async def enqueue_analysis_job(job_id: str):
    await analysis_queue.put(job_id)


async def _analysis_worker(index: int):
    while True:
        job_id = await analysis_queue.get()
        try:
            await run_analysis(job_id)
        finally:
            analysis_queue.task_done()


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


def _safe_download_name(filename: Optional[str], fallback: str) -> str:
    name = os.path.basename(filename or fallback).replace('"', "")
    return name or fallback


def _json_download_name(filename: Optional[str], slot: str) -> str:
    name = _safe_download_name(filename, f"trace_{slot}.json")
    lower = name.lower()
    if lower.endswith(".tar.gz"):
        return name[:-7] + ".json"
    if lower.endswith(".tgz"):
        return name[:-4] + ".json"
    if lower.endswith(".json.gz"):
        return name[:-3]
    if lower.endswith(".gz"):
        base = name[:-3]
        return base if base.lower().endswith(".json") else base + ".json"
    return name if lower.endswith(".json") else name + ".json"


def _stored_download_name(filename: Optional[str], file_path: str, slot: str) -> str:
    name = _safe_download_name(filename, f"trace_{slot}.json")
    lower = name.lower()
    if file_path.endswith(".gz") and not (lower.endswith(".gz") or lower.endswith(".tgz")):
        return name + ".gz"
    return name


def _content_disposition(filename: str, disposition: str = "attachment") -> str:
    return f'{disposition}; filename="{filename}"'


def csv_to_rows(path: str) -> dict:
    """Read a CSV file and return {fields, rows}."""
    if not os.path.exists(path):
        return {"fields": [], "rows": []}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return {"fields": reader.fieldnames or [], "rows": rows}


def _ordered_result_csv_names(rdir: str) -> list[str]:
    names = []
    for name in ["all_kernels_avg.csv", "all_kernels_cmp.csv",
                 "triton_kernels_avg.csv", "triton_kernels_cmp.csv",
                 "aten_ops_avg.csv", "aten_ops_cmp.csv",
                 "kernel_types_avg.csv", "kernel_types_cmp.csv", "kernel_types_delta.csv",
                 "cncl_ops_avg.csv", "cncl_ops_cmp.csv"]:
        if os.path.exists(os.path.join(rdir, name)):
            names.append(name)
    if os.path.isdir(rdir):
        names.extend(
            fname for fname in sorted(os.listdir(rdir))
            if fname.startswith("step_") and fname.endswith("_triton_kernels.csv")
        )
    return names


def _safe_result_csv_path(jid: str, filename: str) -> str:
    if not filename.endswith(".csv") or os.path.basename(filename) != filename:
        raise HTTPException(400, "Invalid result filename")
    rdir = result_dir(jid)
    full = os.path.abspath(os.path.join(rdir, filename))
    if os.path.commonpath([os.path.abspath(rdir), full]) != os.path.abspath(rdir):
        raise HTTPException(400, "Invalid result filename")
    if not os.path.exists(full):
        raise HTTPException(404, "Result file not found")
    return full


def collect_result_files(jid: str) -> dict:
    rdir = result_dir(jid)
    files = {}
    for name in _ordered_result_csv_names(rdir):
        full = os.path.join(rdir, name)
        fields = []
        with open(full, newline="") as f:
            reader = csv.reader(f)
            fields = next(reader, []) or []
        files[name] = {
            "fields": fields,
            "size": _path_size(full),
        }
    return files


def _csv_filter_match(row: dict, q: Optional[str], filters: dict, filter_ops: dict) -> bool:
    if q:
        ql = q.lower()
        if not any(ql in str(value).lower() for value in row.values()):
            return False

    for field, value in filters.items():
        if value in (None, ""):
            continue
        cell = row.get(field, "")
        op = filter_ops.get(field) or "~"
        text = str(value)
        if op in ("~", "!~"):
            terms = [term.lower() for term in text.split("|") if term]
            hit = any(term in str(cell).lower() for term in terms)
            if (op == "~" and not hit) or (op == "!~" and hit):
                return False
            continue

        try:
            expected = float(text)
            actual = float(cell)
        except (TypeError, ValueError):
            return False
        if op == ">=" and actual < expected:
            return False
        if op == "<=" and actual > expected:
            return False
        if op == ">" and actual <= expected:
            return False
        if op == "<" and actual >= expected:
            return False
        if op == "=" and actual != expected:
            return False
    return True


def _sort_value(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value or "").lower())


def read_csv_page(
    path: str,
    *,
    q: Optional[str] = None,
    filters: Optional[dict] = None,
    filter_ops: Optional[dict] = None,
    sort_col: Optional[str] = None,
    sort_dir: str = "asc",
    limit: int = 100,
    offset: int = 0,
) -> dict:
    filters = filters or {}
    filter_ops = filter_ops or {}
    limit = max(1, limit)
    offset = max(0, offset)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = []
        total = 0
        filtered_total = 0
        requires_materialize = bool(q or filters or sort_col)
        if requires_materialize:
            for row in reader:
                total += 1
                if _csv_filter_match(row, q, filters, filter_ops):
                    rows.append(row)
            filtered_total = len(rows)
            if sort_col and sort_col in fields:
                rows.sort(
                    key=lambda item: _sort_value(item.get(sort_col)),
                    reverse=(sort_dir == "desc"),
                )
            page_rows = rows[offset:offset + limit]
        else:
            page_rows = []
            for row in reader:
                if total >= offset and len(page_rows) < limit:
                    page_rows.append(row)
                total += 1
            filtered_total = total

    return {
        "fields": fields,
        "rows": page_rows,
        "total": total,
        "filtered_total": filtered_total,
        "limit": limit,
        "offset": offset,
    }


def collect_results(jid: str) -> dict:
    rdir = result_dir(jid)
    files = {}
    for name in ["all_kernels_avg.csv", "all_kernels_cmp.csv",
                 "triton_kernels_avg.csv", "triton_kernels_cmp.csv",
                 "aten_ops_avg.csv", "aten_ops_cmp.csv",
                 "kernel_types_avg.csv", "kernel_types_cmp.csv", "kernel_types_delta.csv",
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


def _path_size(path: Optional[str]) -> int:
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def _dir_size(path: str) -> int:
    if not path or not os.path.isdir(path):
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += _path_size(os.path.join(root, name))
    return total


def _owned_trace_bytes(job: dict) -> int:
    return sum(
        _path_size(job.get(col))
        for col in ("file_a_path", "file_a_gzip_path", "file_b_path", "file_b_gzip_path")
    )


def _job_storage_stats(job: dict) -> dict:
    return {
        "owned_bytes": _dir_size(job_dir(job["id"])),
        "result_bytes": _dir_size(result_dir(job["id"])),
        "original_trace_bytes": _owned_trace_bytes(job),
    }


async def _refresh_job_storage_cache(db, job_id: str) -> dict:
    row = await row_to_dict(
        await (await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))).fetchone()
    )
    if not row:
        return {"owned_bytes": 0, "result_bytes": 0, "original_trace_bytes": 0}
    stats = _job_storage_stats(row)
    await db.execute(
        """
        UPDATE jobs
        SET owned_bytes=?, result_bytes=?, original_trace_bytes=?
        WHERE id=?
        """,
        (stats["owned_bytes"], stats["result_bytes"], stats["original_trace_bytes"], job_id),
    )
    return stats


async def refresh_storage_cache_for_all_jobs():
    db = await get_db()
    try:
        rows = await (
            await db.execute(
                """
                SELECT id FROM jobs
                WHERE owned_bytes IS NULL
                   OR result_bytes IS NULL
                   OR original_trace_bytes IS NULL
                """
            )
        ).fetchall()
        for row in rows:
            await _refresh_job_storage_cache(db, row["id"])
        await db.commit()
    finally:
        await db.close()


def _cached_storage_value(job: dict, key: str, stats: Optional[dict] = None) -> int:
    value = job.get(key)
    if value is not None:
        return int(value)
    return int((stats or _job_storage_stats(job))[key])


async def _compare_dependents(db, source_job_id: str):
    rows = await (
        await db.execute(
            """
            SELECT id, label, created_at, project_id
            FROM jobs
            WHERE mode='compare' AND (source_job_a=? OR source_job_b=?)
            ORDER BY created_at DESC
            """,
            (source_job_id, source_job_id),
        )
    ).fetchall()
    return [dict(row) for row in rows]


async def _compare_source_summaries(job: dict):
    source_ids = [job.get("source_job_a"), job.get("source_job_b")]
    source_ids = [jid for jid in source_ids if jid]
    if not source_ids:
        return {}

    db = await get_db()
    try:
        placeholders = ",".join("?" * len(source_ids))
        rows = await (
            await db.execute(
                f"""
                SELECT j.id, j.label, j.project_id, j.created_at, j.file_a_name,
                       p.name AS project_name,
                       j.file_a_path, j.file_a_gzip_path
                FROM jobs j
                LEFT JOIN projects p ON p.id = j.project_id
                WHERE j.id IN ({placeholders})
                """,
                tuple(source_ids),
            )
        ).fetchall()
    finally:
        await db.close()

    by_id = {}
    for row in rows:
        data = dict(row)
        path = data.get("file_a_gzip_path") or data.get("file_a_path")
        data["file_a_exists"] = 1 if path and os.path.exists(path) else 0
        data.pop("file_a_path", None)
        data.pop("file_a_gzip_path", None)
        by_id[data["id"]] = data

    result = {}
    for slot in ("a", "b"):
        source_id = job.get(f"source_job_{slot}")
        if source_id and source_id in by_id:
            result[slot] = by_id[source_id]
    return result


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


def _iter_gzip_json(path: str):
    with gzip.open(path, "rb") as gz:
        while True:
            chunk = gz.read(1 << 20)
            if not chunk:
                break
            yield chunk


def _iter_tar_json(path: str):
    with tarfile.open(path, "r:*") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json")]
        if not members:
            raise ValueError("Archive does not contain a JSON trace")
        member = max(members, key=lambda m: m.size)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise ValueError("Unable to read JSON trace from archive")
        with extracted:
            while True:
                chunk = extracted.read(1 << 20)
                if not chunk:
                    break
                yield chunk


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
        cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
        job = await row_to_dict(await cursor.fetchone())
        if not job or job["status"] != "pending":
            return

        await db.execute("UPDATE jobs SET status='running', error_msg='' WHERE id=?", (job_id,))
        await db.commit()

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
        await _refresh_job_storage_cache(db, job_id)
        await db.commit()

    except Exception as e:
        await db.execute(
            "UPDATE jobs SET status='error', error_msg=? WHERE id=?",
            (str(e), job_id),
        )
        await _refresh_job_storage_cache(db, job_id)
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


def _job_search_clause(alias: str, q: Optional[str], include_project_name: bool = False):
    q = (q or "").strip()
    if not q:
        return "", []

    terms = [
        f"LOWER(COALESCE({alias}.label, '')) LIKE ?",
        f"LOWER(COALESCE({alias}.file_a_name, '')) LIKE ?",
        f"LOWER(COALESCE({alias}.file_b_name, '')) LIKE ?",
    ]
    params = [f"%{q.lower()}%"] * 3
    if include_project_name:
        terms.append("LOWER(COALESCE(p.name, '')) LIKE ?")
        params.append(f"%{q.lower()}%")
    return f"({' OR '.join(terms)})", params


def _unique_job_ids(body: dict) -> list[str]:
    job_ids = []
    for job_id in body.get("job_ids") or []:
        if job_id and job_id not in job_ids:
            job_ids.append(job_id)
    if not job_ids:
        raise HTTPException(400, "job_ids are required")
    return job_ids


async def _load_jobs_by_ids(db, job_ids: list[str]) -> list[dict]:
    placeholders = ",".join("?" * len(job_ids))
    rows = await (
        await db.execute(
            f"SELECT * FROM jobs WHERE id IN ({placeholders})",
            tuple(job_ids),
        )
    ).fetchall()
    jobs = [dict(row) for row in rows]
    if len(jobs) != len(job_ids):
        raise HTTPException(404, "Some jobs were not found")
    return jobs


def _remove_job_dir(jid: str):
    jdir = job_dir(jid)
    if os.path.exists(jdir):
        shutil.rmtree(jdir)


async def _delete_trace_files(db, row: dict, slots=("a", "b")) -> int:
    removed = 0
    for slot in slots:
        for col in (f"file_{slot}_path", f"file_{slot}_gzip_path"):
            path = row.get(col)
            if path and os.path.exists(path):
                os.remove(path)
                removed += 1
            await db.execute(f"UPDATE jobs SET {col}=NULL WHERE id=?", (row["id"],))
        await db.execute(f"UPDATE jobs SET file_{slot}_exists=0 WHERE id=?", (row["id"],))
    return removed


@app.get("/api/compare-candidates")
async def list_compare_candidates(
    project_id: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    clauses = ["j.mode='single'", "j.status='done'"]
    params = []
    if project_id == "__none__":
        clauses.append("j.project_id IS NULL")
    elif project_id:
        clauses.append("j.project_id = ?")
        params.append(project_id)

    search_sql, search_params = _job_search_clause("j", q, include_project_name=True)
    if search_sql:
        clauses.append(search_sql)
        params.extend(search_params)

    where_sql = " AND ".join(clauses)
    count_cursor = await db.execute(
        f"""
        SELECT COUNT(*)
        FROM jobs j
        LEFT JOIN projects p ON p.id = j.project_id
        WHERE {where_sql}
        """,
        params,
    )
    total = (await count_cursor.fetchone())[0]

    rows = await (
        await db.execute(
            f"""
            SELECT j.*
            FROM jobs j
            LEFT JOIN projects p ON p.id = j.project_id
            WHERE {where_sql}
            ORDER BY j.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        )
    ).fetchall()
    await db.close()

    data = await _with_file_exists(rows)
    return {"data": data, "total": total, "limit": limit, "offset": offset}


@app.patch("/api/jobs/bulk/project")
async def bulk_move_jobs(body: dict):
    job_ids = _unique_job_ids(body)
    db = await get_db()
    try:
        await _load_jobs_by_ids(db, job_ids)
        placeholders = ",".join("?" * len(job_ids))
        await db.execute(
            f"UPDATE jobs SET project_id=? WHERE id IN ({placeholders})",
            (body.get("project_id") or None, *job_ids),
        )
        await db.commit()
    finally:
        await db.close()
    return {"updated": len(job_ids)}


@app.post("/api/jobs/bulk/delete")
async def bulk_delete_jobs(body: dict):
    job_ids = _unique_job_ids(body)
    db = await get_db()
    try:
        await _load_jobs_by_ids(db, job_ids)
        for job_id in job_ids:
            _remove_job_dir(job_id)
        placeholders = ",".join("?" * len(job_ids))
        await db.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", tuple(job_ids))
        await db.commit()
    finally:
        await db.close()
    return {"deleted": len(job_ids)}


@app.post("/api/jobs/bulk/delete-files")
async def bulk_delete_job_files(body: dict):
    job_ids = _unique_job_ids(body)
    db = await get_db()
    try:
        jobs = await _load_jobs_by_ids(db, job_ids)
        files_deleted = 0
        for job in jobs:
            files_deleted += await _delete_trace_files(db, job)
            await _refresh_job_storage_cache(db, job["id"])
        await db.commit()
    finally:
        await db.close()
    return {"updated": len(job_ids), "files_deleted": files_deleted}


@app.get("/api/storage/summary")
async def storage_summary():
    db = await get_db()
    try:
        rows = await (
            await db.execute(
                """
                SELECT j.*, p.name AS project_name
                FROM jobs j
                LEFT JOIN projects p ON p.id = j.project_id
                ORDER BY j.created_at DESC
                """
            )
        ).fetchall()
        compare_counts_rows = await (
            await db.execute(
                """
                SELECT source_id, COUNT(*) AS count
                FROM (
                    SELECT source_job_a AS source_id FROM jobs WHERE source_job_a IS NOT NULL
                    UNION ALL
                    SELECT source_job_b AS source_id FROM jobs WHERE source_job_b IS NOT NULL
                )
                GROUP BY source_id
                """
            )
        ).fetchall()
    finally:
        await db.close()

    compare_counts = {row["source_id"]: row["count"] for row in compare_counts_rows}
    jobs = []
    projects = {}
    for row in rows:
        job = dict(row)
        stats = None
        if (
            job.get("owned_bytes") is None
            or job.get("result_bytes") is None
            or job.get("original_trace_bytes") is None
        ):
            stats = _job_storage_stats(job)
        owned_bytes = _cached_storage_value(job, "owned_bytes", stats)
        result_bytes = _cached_storage_value(job, "result_bytes", stats)
        original_trace_bytes = _cached_storage_value(job, "original_trace_bytes", stats)
        item = {
            "id": job["id"],
            "label": job["label"],
            "project_id": job.get("project_id"),
            "project_name": job.get("project_name") or "未分组",
            "mode": job["mode"],
            "status": job["status"],
            "created_at": job["created_at"],
            "owned_bytes": owned_bytes,
            "result_bytes": result_bytes,
            "original_trace_bytes": original_trace_bytes,
            "has_original_trace": original_trace_bytes > 0,
            "used_by_compare_count": compare_counts.get(job["id"], 0),
        }
        jobs.append(item)
        project_key = job.get("project_id") or "__none__"
        project = projects.setdefault(
            project_key,
            {
                "id": project_key,
                "name": job.get("project_name") or "未分组",
                "owned_bytes": 0,
                "original_trace_bytes": 0,
                "job_count": 0,
            },
        )
        project["owned_bytes"] += owned_bytes
        project["original_trace_bytes"] += original_trace_bytes
        project["job_count"] += 1

    jobs.sort(key=lambda item: (item["original_trace_bytes"], item["owned_bytes"]), reverse=True)
    project_list = sorted(projects.values(), key=lambda item: item["owned_bytes"], reverse=True)
    return {
        "totals": {
            "owned_bytes": sum(item["owned_bytes"] for item in jobs),
            "original_trace_bytes": sum(item["original_trace_bytes"] for item in jobs),
            "result_bytes": sum(item["result_bytes"] for item in jobs),
        },
        "projects": project_list,
        "jobs": jobs,
    }


@app.get("/api/job-groups")
async def list_job_groups(
    project_id: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    clauses = []
    where_params = []
    if project_id == "__none__":
        clauses.append("j.project_id IS NULL")
    elif project_id:
        clauses.append("j.project_id = ?")
        where_params.append(project_id)

    search_sql, search_params = _job_search_clause("j", q, include_project_name=True)
    if search_sql:
        clauses.append(search_sql)
        where_params.extend(search_params)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    count_cursor = await db.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT j.project_id
            FROM jobs j
            LEFT JOIN projects p ON p.id = j.project_id
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
                COALESCE(p.name, '未分组') AS label,
                COUNT(*) AS job_count
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

    await db.close()

    data = []
    for row in group_rows:
        group_id = row["project_id"] or "__none__"
        data.append(
            {
                "id": group_id,
                "label": row["label"],
                "job_count": row["job_count"],
            }
        )

    return {"data": data, "total": total, "limit": limit, "offset": offset}


@app.get("/api/job-groups/{group_id}/jobs")
async def list_group_jobs(
    group_id: str,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    clauses = []
    params = []
    if group_id == "__none__":
        clauses.append("j.project_id IS NULL")
    else:
        clauses.append("j.project_id = ?")
        params.append(group_id)

    search_sql, search_params = _job_search_clause("j", q, include_project_name=True)
    if search_sql:
        clauses.append(search_sql)
        params.extend(search_params)

    where_sql = " AND ".join(clauses)
    count_cursor = await db.execute(
        f"""
        SELECT COUNT(*)
        FROM jobs j
        LEFT JOIN projects p ON p.id = j.project_id
        WHERE {where_sql}
        """,
        params,
    )
    total = (await count_cursor.fetchone())[0]

    rows = await (
        await db.execute(
            f"""
            SELECT j.*
            FROM jobs j
            LEFT JOIN projects p ON p.id = j.project_id
            WHERE {where_sql}
            ORDER BY j.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        )
    ).fetchall()
    await db.close()

    data = await _with_file_exists(rows)
    return {"data": data, "total": total, "limit": limit, "offset": offset}


@app.post("/api/jobs", status_code=201)
async def create_job(
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
    await _refresh_job_storage_cache(db, jid)
    await db.commit()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await cursor.fetchone()
    await db.close()

    await enqueue_analysis_job(jid)
    return dict(row)


@app.post("/api/jobs/compare", status_code=201)
async def compare_jobs(body: dict):
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
    await _refresh_job_storage_cache(db, jid)
    await db.commit()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await cursor.fetchone()
    await db.close()

    await enqueue_analysis_job(jid)
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
        job["result_files"] = collect_result_files(jid)
        job["perfetto_context"] = collect_perfetto_context(jid)
    if job["mode"] == "compare":
        job["compare_sources"] = await _compare_source_summaries(job)
    return job


@app.get("/api/jobs/{jid}/results/{filename:path}")
async def get_job_result_table(
    jid: str,
    filename: str,
    q: Optional[str] = None,
    sort_col: Optional[str] = None,
    sort_dir: str = "asc",
    limit: int = 100,
    offset: int = 0,
    filters: Optional[str] = None,
    filter_ops: Optional[str] = None,
):
    db = await get_db()
    try:
        row = await row_to_dict(
            await (await db.execute("SELECT id, status FROM jobs WHERE id=?", (jid,))).fetchone()
        )
    finally:
        await db.close()
    if not row:
        raise HTTPException(404)
    if row["status"] != "done":
        raise HTTPException(409, "Job is not done")

    try:
        parsed_filters = json.loads(filters) if filters else {}
        parsed_ops = json.loads(filter_ops) if filter_ops else {}
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid table filters")
    if not isinstance(parsed_filters, dict) or not isinstance(parsed_ops, dict):
        raise HTTPException(400, "Invalid table filters")

    path = _safe_result_csv_path(jid, filename)
    return read_csv_page(
        path,
        q=q,
        filters=parsed_filters,
        filter_ops=parsed_ops,
        sort_col=sort_col,
        sort_dir="desc" if sort_dir == "desc" else "asc",
        limit=limit,
        offset=offset,
    )


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


    _remove_job_dir(jid)

    await db.execute("DELETE FROM jobs WHERE id=?", (jid,))
    await db.commit()
    await db.close()


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

    filename = row.get(f"file_{slot}_name")

    # If the client requests raw JSON, return parseable JSON even when storage
    # keeps only a compressed copy after analysis.
    if format == "json":
        json_filename = _json_download_name(filename, slot)
        if file_path.endswith(".gz"):
            try:
                chunks = _iter_tar_json(file_path) if tarfile.is_tarfile(file_path) else _iter_gzip_json(file_path)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            return StreamingResponse(
                chunks,
                media_type="application/json",
                headers={"Content-Disposition": _content_disposition(json_filename)},
            )
        return FileResponse(file_path, media_type="application/json", filename=json_filename)

    # Stream file directly — avoids loading entire file into memory
    media_type = "application/gzip" if file_path.endswith(".gz") else "application/json"
    stored_filename = _stored_download_name(filename, file_path, slot)
    return FileResponse(file_path, media_type=media_type, filename=stored_filename)


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

    await _delete_trace_files(db, row, slots=(slot,))
    await _refresh_job_storage_cache(db, jid)
    await db.commit()
    await db.close()


@app.get("/api/jobs/{jid}/files/{slot}/delete-impact")
async def get_delete_file_impact(jid: str, slot: str):
    if slot not in ("a", "b"):
        raise HTTPException(400, "slot must be 'a' or 'b'")

    db = await get_db()
    try:
        row = await row_to_dict(
            await (await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))).fetchone()
        )
        if not row:
            raise HTTPException(404)
        dependents = await _compare_dependents(db, jid) if slot == "a" else []
    finally:
        await db.close()
    return {"dependent_compare_jobs": dependents, "count": len(dependents)}


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
    parser.add_argument("--analysis-concurrency", type=int, default=1,
                        help="Concurrent analysis jobs (default: 1)")
    parser.add_argument("--no-download", action="store_true",
                        help="Disable downloading of uploaded trace files (default: download allowed)")
    cli_args = parser.parse_args()

    if cli_args.no_download:
        os.environ["TRACE_NO_DOWNLOAD"] = "1"
    os.environ["TRACE_ANALYSIS_CONCURRENCY"] = str(max(1, cli_args.analysis_concurrency))

    uvicorn.run("server:app", host=cli_args.host, port=cli_args.port, reload=False)
