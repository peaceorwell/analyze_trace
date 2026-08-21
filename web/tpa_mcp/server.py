"""TPA MCP server.

Exposes the TPA (trace performance analyzer) REST API at tpa.cambricon.com as
Model Context Protocol tools so Claude can search jobs/projects, read analysis
results and AI reports, and push trace profiles to trigger new analysis.

Auth: the ``TPA_API_KEY`` environment variable must hold a user access token
(large random string, created in the TPA web UI under user settings).
Optionally override the host with ``TPA_BASE_URL`` (default http://tpa.cambricon.com).

Run as an MCP stdio server and register with::

    claude mcp add tpa --env TPA_API_KEY=$TPA_API_KEY -- python3 <this file>
"""
from __future__ import annotations

import os
from typing import Any, Optional

from fastmcp import FastMCP

from .tpa_api import DEFAULT_BASE_URL, TpaClient

mcp = FastMCP(
    "tpa",
    instructions=(
        "TPA (trace performance analyzer) at tpa.cambricon.com. Manage and inspect "
        "PyTorch profiler analysis jobs: list/query jobs and projects, read kernel/op "
        "result tables and AI analysis reports, and upload trace profiles to start "
        "new single or compare analysis. Jobs can be referenced by their numeric seq "
        "or their UUID id. Auth is via the TPA access token."
    ),
)


def _client() -> TpaClient:
    return TpaClient(base_url=os.environ.get("TPA_BASE_URL", DEFAULT_BASE_URL))


# ---------------------------------------------------------------- jobs/projects


@mcp.tool()
def tpa_list_jobs(
    project_id: Optional[str] = None,
    q: Optional[str] = None,
    statuses: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Any:
    """List analysis jobs accessible to the token user.

    Args:
        project_id: Filter to a project UUID; "__none__" lists jobs with no project.
        q: Free-text search across job labels and file names.
        statuses: Comma-separated subset of pending,running,done,error.
        limit: Max rows (1-100).
        offset: Pagination offset.
    """
    return _client().get(
        "/api/jobs",
        {
            "project_id": project_id,
            "q": q,
            "statuses": statuses,
            "limit": min(limit, 100),
            "offset": offset,
        },
    )


@mcp.tool()
def tpa_get_job(job: str) -> Any:
    """Get full details for one analysis job by numeric seq or UUID id.

    Includes the console summary (per-step / kernel-type comparison when it is a
    compare job), status, file info and available result files.
    """
    return _client().get(f"/api/jobs/{job}")


@mcp.tool()
def tpa_get_job_status(job: str) -> Any:
    """Compact status for a job — use this to poll after tpa_upload_trace.

    Returns status (pending/running/done/error), the job's mode, and the AI
    analysis progress. Poll with this until status is 'done' (or 'error'), then
    call tpa_get_job or tpa_get_job_result for the actual results.
    """
    try:
        full = _client().get(f"/api/jobs/{job}")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    if not isinstance(full, dict):
        return {"error": f"unexpected response: {full!r}"}
    ai = full.get("ai_analysis") or {}
    return {
        "seq": full.get("seq"),
        "job_id": full.get("id"),
        "status": full.get("status"),
        "mode": full.get("mode"),
        "label": full.get("label"),
        "is_owner": full.get("is_owner"),
        "ai_analysis": {
            "status": ai.get("status"),
            "progress": ai.get("progress"),
            "error": ai.get("error"),
            "report_exists": ai.get("report_exists"),
        },
    }


@mcp.tool()
def tpa_list_projects(q: Optional[str] = None) -> Any:
    """List projects accessible to the token user.

    Args:
        q: Optional free-text search on project name.
    """
    return _client().get("/api/projects", {"q": q})


# ------------------------------------------------------------- results / reports


@mcp.tool()
def tpa_start_ai_analysis(job: str, force: bool = False, prompt: Optional[str] = None) -> Any:
    """Trigger (or re-run) the AI analysis report for a completed job.

    Args:
        job: Job numeric seq or UUID id. Must already have status 'done'.
        force: Re-run even if a report already exists.
        prompt: Optional custom instruction/prompt for the AI analysis.
    """
    body: dict[str, Any] = {"force": force}
    if prompt:
        body["prompt"] = prompt
    return _client().post_json(f"/api/jobs/{job}/ai-analysis", body)


@mcp.tool()
def tpa_get_job_result(
    job: str,
    filename: str,
    q: Optional[str] = None,
    sort_col: Optional[str] = None,
    sort_dir: str = "asc",
    limit: int = 100,
    offset: int = 0,
) -> Any:
    """Read a result table (CSV-backed) for a completed job.

    Typical filenames: all_kernels, triton_kernels, aten_ops, kernel_types
    (compare jobs also have *_cmp variants). The response has 'fields', 'rows'
    and 'total'.

    Args:
        job: Job numeric seq or UUID id.
        filename: Result file name inside the job results dir. Use one of the
            job's result_files keys, e.g. all_kernels_cmp.csv, triton_kernels.csv,
            aten_ops.csv, kernel_types_cmp.csv (compare jobs add _cmp variants).
            The .csv extension is optional.
        q: Search filter on table rows.
        sort_col: Column to sort by.
        sort_dir: asc or desc.
        limit: Max rows.
        offset: Pagination offset.
    """
    if not filename.lower().endswith(".csv"):
        filename = filename + ".csv"
    return _client().get(
        f"/api/jobs/{job}/results/{filename}",
        {
            "q": q,
            "sort_col": sort_col,
            "sort_dir": sort_dir,
            "limit": min(limit, 200),
            "offset": offset,
        },
    )


@mcp.tool()
def tpa_get_ai_report(job: str) -> str:
    """Return the auto-generated AI analysis report (Markdown) for a job.

    Returns the report body as text, or a message explaining it is not available
    yet (the job may not have finished, or AI analysis has not been triggered).
    """
    try:
        text = _client().get(f"/api/jobs/{job}/ai-analysis/report.md")
        return text if isinstance(text, str) else str(text)
    except Exception as exc:  # noqa: BLE001 - surface as readable message
        return f"[AI report not available for job {job}] {exc}"


@mcp.tool()
def tpa_get_ai_analysis_status(job: str) -> Any:
    """Check the AI analysis status/progress for a job."""
    return _client().get(f"/api/jobs/{job}/ai-analysis")


# ------------------------------------------------------------------- upload


def _cg(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".gz"):
        return "application/gzip"
    if lower.endswith(".json") or lower.endswith(".json.gz"):
        return "application/json"
    return "application/octet-stream"


def _read_file_or_error(path: str) -> tuple[Optional[bytes], Optional[str]]:
    if not os.path.isfile(path):
        return None, f"file not found: {path}"
    try:
        with open(path, "rb") as fh:
            return fh.read(), None
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"


@mcp.tool()
def tpa_upload_trace(
    file_a: str,
    file_b: Optional[str] = None,
    label: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Any:
    """Upload one or two trace profile files and start analysis (async).

    Provide a single file for single analysis, or both file_a and file_b for a
    compare analysis. The analysis runs asynchronously; this returns immediately
    with the new job's seq/job_id/status. Poll with tpa_get_job_status until the
    status is 'done', then use tpa_get_job / tpa_get_job_result to read results,
    and tpa_start_ai_analysis to request the AI report.

    Args:
        file_a: Local path to the first trace file (json / gz).
        file_b: Optional local path to the second trace file (compare mode).
        label: Optional human-readable label for the job.
        project_id: Optional project UUID to attach the job to.
    """
    client = _client()
    data_a, err_a = _read_file_or_error(file_a)
    if err_a:
        return {"error": err_a}
    files: dict[str, tuple[str, bytes, str]] = {
        "file_a": (os.path.basename(file_a), data_a, _cg(file_a))
    }
    if file_b:
        data_b, err_b = _read_file_or_error(file_b)
        if err_b:
            return {"error": err_b}
        files["file_b"] = (os.path.basename(file_b), data_b, _cg(file_b))

    fields: dict[str, Any] = {"save_triton_csv": "true"}
    if label:
        fields["label"] = label
    if project_id:
        fields["project_id"] = project_id

    job = client.post_multipart("/api/jobs", fields, files)
    if not isinstance(job, dict):
        return {"error": f"unexpected upload response: {job!r}"}
    return {
        "job_id": job.get("id"),
        "seq": job.get("seq"),
        "status": job.get("status"),
        "mode": job.get("mode"),
        "label": job.get("label"),
        "hint": "Analysis runs async. Poll tpa_get_job_status until status='done'.",
    }


@mcp.tool()
def tpa_get_job_report_md(job: str) -> str:
    """Return the plain-text summary Markdown report (report.md) for a job."""
    try:
        text = _client().get(f"/api/jobs/{job}/report.md")
        return text if isinstance(text, str) else str(text)
    except Exception as exc:  # noqa: BLE001
        return f"[report.md not available for job {job}] {exc}"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
