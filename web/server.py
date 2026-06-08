import argparse
import asyncio
import base64
import contextlib
import csv
from datetime import datetime, timedelta, timezone
import gzip
import html
import io
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import smtplib
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import uuid
import zipfile
import zlib
from contextlib import asynccontextmanager
from email.message import EmailMessage
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trace_analyzer import compute_avgs, extract_kernel_family, parse_trace, run_triton_code_and_get_efficiency  # noqa: E402

from db import get_db, init_db, row_to_dict  # noqa: E402
import auth as ldap_auth  # noqa: E402

WEB_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(WEB_DIR)
PROJECT_CLAUDE_SKILLS_DIR = os.path.join(PROJECT_ROOT, ".claude", "skills")
DEFAULT_STORAGE_DIR = os.path.join(WEB_DIR, "storage")
STORAGE_DIR = os.environ.get("TRACE_STORAGE_DIR", DEFAULT_STORAGE_DIR)
APP_VERSION = "0.2.35"
BACKUP_DIR = os.environ.get("TRACE_BACKUP_DIR", os.path.join(WEB_DIR, "backups"))
MAX_BATCH_COMPARE_JOBS = 50
FEEDBACK_DIRNAME = "feedback"
FEEDBACK_MAX_IMAGES = 4
FEEDBACK_MAX_IMAGE_BYTES = 8 * 1024 * 1024
FEEDBACK_MENTION_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,62}$")
FEEDBACK_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_.%-])@([A-Za-z][A-Za-z0-9_.-]{0,62})(?=$|[^A-Za-z0-9_.-])")
AI_ANALYSIS_DIRNAME = "ai_analysis"
AI_ANALYSIS_STATUS_FILE = "ai_analysis_status.json"
AI_ANALYSIS_REPORT_FILE = "ai_analysis.md"
AI_ANALYSIS_ARTIFACT_MAX_BYTES = 256 * 1024
AI_ANALYSIS_ARTIFACT_MAX_TOTAL_BYTES = 1024 * 1024
AI_ANALYSIS_ARTIFACT_EXTENSIONS = {
    ".csv", ".json", ".log", ".md", ".text", ".tsv", ".txt", ".yaml", ".yml",
}
AI_ANALYSIS_INTERNAL_FILES = {
    AI_ANALYSIS_STATUS_FILE,
    "ai_diagnostics.json",
    "claude_tool_probe.txt",
    "command.json",
}
AI_ANALYSIS_FAILURE_PATTERNS = (
    "all tool calls are being denied",
    "cannot proceed because the execution environment has restricted permissions",
    "claude did not create the bash probe file",
    "eacces - cannot create",
    "error: tool denied",
    "permission denied for all files outside",
    "tool permissions may be denied",
)

# Configured at startup via CLI; read-only after that
AUTH_MODE = os.environ.get("AUTH_MODE", "none").strip().lower()
AUTH_ENABLED = AUTH_MODE == "ldap"
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-only-change-me")
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "0") == "1"
ALLOW_FILE_DOWNLOAD = os.environ.get("TRACE_NO_DOWNLOAD", "") == ""
ALLOW_CODE_EXECUTION = os.environ.get("TRACE_ENABLE_CODE_EXEC", "") == "1"
ANALYSIS_CONCURRENCY = max(1, int(os.environ.get("TRACE_ANALYSIS_CONCURRENCY", "1")))
CLAUDE_ANALYSIS_ENABLED = os.environ.get("TRACE_ENABLE_CLAUDE_ANALYSIS", "") == "1"
CLAUDE_ANALYSIS_COMMAND = os.environ.get("TRACE_CLAUDE_COMMAND", "claude")
CLAUDE_ANALYSIS_COMMAND_TEMPLATE = os.environ.get("TRACE_CLAUDE_COMMAND_TEMPLATE", "")
DEFAULT_CLAUDE_ANALYSIS_EXTRA_ARGS = "--dangerously-skip-permissions"
CLAUDE_ANALYSIS_EXTRA_ARGS = os.environ.get("TRACE_CLAUDE_EXTRA_ARGS", DEFAULT_CLAUDE_ANALYSIS_EXTRA_ARGS)
CLAUDE_ANALYSIS_TIMEOUT_SECONDS = max(30, int(os.environ.get("TRACE_CLAUDE_TIMEOUT_SECONDS", "1800")))
CLAUDE_DIAGNOSTIC_TIMEOUT_SECONDS = max(5, int(os.environ.get("TRACE_CLAUDE_DIAGNOSTIC_TIMEOUT_SECONDS", "60")))
CLAUDE_ANALYSIS_SKILLS_DIR = os.environ.get("TRACE_CLAUDE_SKILLS_DIR", PROJECT_CLAUDE_SKILLS_DIR)
CLAUDE_SINGLE_TRACE_SKILL = os.environ.get("TRACE_CLAUDE_SINGLE_SKILL", "e2e-profiling-analyzer")
CLAUDE_COMPARE_TRACE_SKILL = os.environ.get("TRACE_CLAUDE_COMPARE_SKILL", "e2e-profiling-comparator")


def _parse_user_list_env(value: str) -> set[str]:
    return {
        item.strip().lower()
        for item in (value or "").replace(";", ",").split(",")
        if item.strip()
    }


def _truthy_env(value: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _normalize_email(value: str, domain: str = "cambricon.com") -> str:
    token = (value or "").strip().strip("<>,;")
    if not token:
        return ""
    if "@" not in token:
        token = f"{token}@{domain}"
    token = token.lower()
    if not re.fullmatch(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", token):
        return ""
    return token


def _parse_email_list_env(value: str, domain: str = "cambricon.com") -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in (value or "").replace(";", ",").split(","):
        email = _normalize_email(item, domain)
        if email and email not in seen:
            seen.add(email)
            result.append(email)
    return result


ADMIN_USERS = _parse_user_list_env(os.environ.get("TRACE_ADMIN_USERS", ""))
FEEDBACK_MENTION_DOMAIN = os.environ.get("TRACE_FEEDBACK_MENTION_DOMAIN", "cambricon.com").strip().lower() or "cambricon.com"
FEEDBACK_NOTIFICATION_ADMIN_EMAILS = _parse_email_list_env(
    os.environ.get("TRACE_FEEDBACK_ADMIN_EMAILS", os.environ.get("TRACE_FEEDBACK_ADMIN_EMAIL", "zhouyusong@cambricon.com")),
    FEEDBACK_MENTION_DOMAIN,
)
FEEDBACK_EMAIL_NOTIFICATIONS_ENABLED = not _truthy_env(os.environ.get("TRACE_DISABLE_FEEDBACK_EMAIL", ""), False)
PUBLIC_BASE_URL = os.environ.get("TRACE_PUBLIC_BASE_URL", "").strip().rstrip("/")
SMTP_HOST = os.environ.get("TRACE_SMTP_HOST", os.environ.get("SMTP_HOST", "")).strip()
SMTP_PORT = _int_env("TRACE_SMTP_PORT", _int_env("SMTP_PORT", 25))
SMTP_USERNAME = os.environ.get("TRACE_SMTP_USERNAME", os.environ.get("SMTP_USERNAME", "")).strip()
SMTP_PASSWORD = os.environ.get("TRACE_SMTP_PASSWORD", os.environ.get("SMTP_PASSWORD", ""))
SMTP_FROM_RAW = (
    os.environ.get("TRACE_SMTP_FROM", "")
    or SMTP_USERNAME
    or f"trace-analyzer@{FEEDBACK_MENTION_DOMAIN}"
).strip()
SMTP_FROM = _normalize_email(SMTP_FROM_RAW, FEEDBACK_MENTION_DOMAIN) or f"trace-analyzer@{FEEDBACK_MENTION_DOMAIN}"
SMTP_USE_SSL = _truthy_env(os.environ.get("TRACE_SMTP_SSL", os.environ.get("SMTP_SSL", "")), False)
SMTP_USE_STARTTLS = _truthy_env(os.environ.get("TRACE_SMTP_STARTTLS", os.environ.get("SMTP_STARTTLS", "")), False)
SMTP_TIMEOUT_SECONDS = max(1, _int_env("TRACE_SMTP_TIMEOUT_SECONDS", _int_env("SMTP_TIMEOUT_SECONDS", 10)))
SENDMAIL_COMMAND = os.environ.get("TRACE_SENDMAIL_COMMAND", "").strip()
SENDMAIL_AUTODETECT = _truthy_env(os.environ.get("TRACE_ENABLE_SENDMAIL_AUTODETECT", ""), False)
if not SENDMAIL_COMMAND and SENDMAIL_AUTODETECT:
    for _sendmail_candidate in ("/usr/sbin/sendmail", "/usr/lib/sendmail"):
        if os.path.exists(_sendmail_candidate) and os.access(_sendmail_candidate, os.X_OK):
            SENDMAIL_COMMAND = _sendmail_candidate
            break
LOG_TIMEZONE_NAME = os.environ.get("TRACE_LOG_TIMEZONE", os.environ.get("TZ", "Asia/Shanghai")).strip()
analysis_queue: asyncio.Queue[str] = asyncio.Queue()
analysis_workers: list[asyncio.Task] = []
ai_analysis_tasks: dict[str, asyncio.Task] = {}
APP_STARTED_AT = time.time()
HTTP_METRICS: dict[tuple[str, str, int], dict[str, float]] = {}
LOGIN_CAPTCHA_THRESHOLD = max(1, int(os.environ.get("LOGIN_CAPTCHA_THRESHOLD", "5")))
LOGIN_CAPTCHA_TTL_SECONDS = max(60, int(os.environ.get("LOGIN_CAPTCHA_TTL_SECONDS", "300")))
LOGIN_FAILURE_TTL_SECONDS = max(60, int(os.environ.get("LOGIN_FAILURE_TTL_SECONDS", "900")))
LOGIN_FAILURES: dict[str, dict[str, float]] = {}
LOGIN_CAPTCHA_CHALLENGES: dict[str, dict[str, float | str]] = {}
CAPTCHA_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _resolve_log_timezone(name: str):
    value = (name or "").strip()
    if not value or value.lower() in {"local", "system"}:
        return datetime.now().astimezone().tzinfo or timezone.utc
    if value.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError:
        fallback = (
            timezone(timedelta(hours=8), name="CST")
            if value in {"Asia/Shanghai", "Asia/Chongqing", "PRC", "CST"}
            else datetime.now().astimezone().tzinfo or timezone.utc
        )
        logging.getLogger("analyze_trace.web").warning(
            "log_timezone_invalid",
            extra={"event": "log_timezone_invalid", "timezone": value},
        )
        return fallback


class JsonLogFormatter(logging.Formatter):
    def __init__(self, tzinfo=None):
        super().__init__()
        self.tzinfo = tzinfo or timezone.utc

    def format(self, record):
        payload = {
            "time": datetime.fromtimestamp(record.created, self.tzinfo).isoformat(),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _LOG_RESERVED_KEYS:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_LOG_RESERVED_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process",
}


def configure_logging():
    logger = logging.getLogger("analyze_trace.web")
    logger.setLevel(os.environ.get("TRACE_LOG_LEVEL", "INFO").upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = JsonLogFormatter(_resolve_log_timezone(LOG_TIMEZONE_NAME))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file = os.environ.get("TRACE_LOG_FILE")
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.warning(
                "log_file_unavailable",
                extra={"event": "log_file_unavailable", "path": log_file, "error": str(e)},
            )
    return logger


logger = configure_logging()


def request_user(request: Request) -> str:
    session_user = getattr(request.state, "user", None) or {}
    if session_user.get("username"):
        return session_user["username"]
    return (
        request.headers.get("X-Remote-User")
        or request.headers.get("X-Forwarded-User")
        or request.headers.get("X-User")
        or "local"
    )


def request_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else ""


def _login_username_key(username: str) -> str:
    return (username or "").strip().lower()


def _login_failure_key(request: Request, username: str) -> str:
    return f"{request_ip(request)}:{_login_username_key(username)}"


def _prune_login_state(now: Optional[float] = None):
    now = now or time.time()
    for key, state in list(LOGIN_FAILURES.items()):
        if now - float(state.get("updated_at", 0)) > LOGIN_FAILURE_TTL_SECONDS:
            LOGIN_FAILURES.pop(key, None)
    for key, state in list(LOGIN_CAPTCHA_CHALLENGES.items()):
        if now > float(state.get("expires_at", 0)):
            LOGIN_CAPTCHA_CHALLENGES.pop(key, None)


def _login_failure_count(key: str) -> int:
    _prune_login_state()
    state = LOGIN_FAILURES.get(key) or {}
    return int(state.get("count", 0))


def _record_login_failure(key: str) -> int:
    _prune_login_state()
    state = LOGIN_FAILURES.get(key) or {}
    count = int(state.get("count", 0)) + 1
    LOGIN_FAILURES[key] = {"count": count, "updated_at": time.time()}
    return count


def _clear_login_failure(key: str):
    LOGIN_FAILURES.pop(key, None)


def _captcha_required_for(request: Request, username: str) -> bool:
    return _login_failure_count(_login_failure_key(request, username)) >= LOGIN_CAPTCHA_THRESHOLD


def _clear_login_captcha(request: Request):
    captcha_id = request.session.pop("login_captcha_id", None)
    request.session.pop("login_captcha_user", None)
    if captcha_id:
        LOGIN_CAPTCHA_CHALLENGES.pop(captcha_id, None)


def _captcha_svg_data_url(code: str) -> str:
    lines = []
    for idx in range(5):
        y = 8 + idx * 7 + secrets.randbelow(5)
        lines.append(
            f'<path d="M8 {y} C 42 {secrets.randbelow(42)}, '
            f'88 {secrets.randbelow(42)}, 142 {y + secrets.randbelow(9) - 4}" />'
        )
    chars = []
    for idx, char in enumerate(code):
        x = 18 + idx * 24 + secrets.randbelow(4)
        y = 30 + secrets.randbelow(8)
        angle = secrets.randbelow(19) - 9
        chars.append(
            f'<text x="{x}" y="{y}" transform="rotate({angle} {x} {y})">'
            f'{html.escape(char)}</text>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="150" height="44" viewBox="0 0 150 44">'
        '<rect width="150" height="44" rx="8" fill="#eef2ff"/>'
        '<g fill="none" stroke="#c7d2fe" stroke-width="1.2" stroke-linecap="round">'
        + "".join(lines)
        + '</g><g font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="24" '
        'font-weight="800" fill="#4f46e5">'
        + "".join(chars)
        + "</g></svg>"
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _issue_login_captcha(request: Request, username: str) -> dict:
    _clear_login_captcha(request)
    code = "".join(CAPTCHA_ALPHABET[secrets.randbelow(len(CAPTCHA_ALPHABET))] for _ in range(5))
    captcha_id = secrets.token_urlsafe(24)
    username_key = _login_username_key(username)
    LOGIN_CAPTCHA_CHALLENGES[captcha_id] = {
        "answer": code,
        "username": username_key,
        "expires_at": time.time() + LOGIN_CAPTCHA_TTL_SECONDS,
    }
    request.session["login_captcha_id"] = captcha_id
    request.session["login_captcha_user"] = username_key
    return {
        "captcha_required": True,
        "captcha_image": _captcha_svg_data_url(code),
        "captcha_ttl_seconds": LOGIN_CAPTCHA_TTL_SECONDS,
    }


def _verify_login_captcha(request: Request, username: str, answer: str) -> bool:
    _prune_login_state()
    captcha_id = request.session.get("login_captcha_id")
    challenge = LOGIN_CAPTCHA_CHALLENGES.get(captcha_id or "")
    if not captcha_id or not challenge:
        return False
    if challenge.get("username") != _login_username_key(username):
        return False
    if time.time() > float(challenge.get("expires_at", 0)):
        _clear_login_captcha(request)
        return False
    expected = str(challenge.get("answer") or "")
    actual = (answer or "").strip().upper()
    if not actual or not secrets.compare_digest(actual, expected):
        return False
    _clear_login_captcha(request)
    return True


def _login_error_response(
    request: Request,
    username: str,
    detail: str,
    status_code: int = 401,
    include_captcha: bool = False,
) -> JSONResponse:
    payload = {"detail": detail, "captcha_required": bool(include_captcha)}
    if include_captcha:
        payload.update(_issue_login_captcha(request, username))
    return JSONResponse(payload, status_code=status_code)


def _auth_public_path(path: str) -> bool:
    return (
        path == "/"
        or path.startswith("/static/")
        or path in {
            "/healthz",
            "/readyz",
            "/metrics",
            "/api/config",
            "/api/login",
            "/api/login-captcha",
            "/api/logout",
            "/api/me",
        }
    )


def current_user(request: Request) -> dict:
    return getattr(request.state, "user", None) or {}


def current_user_token(request: Request) -> Optional[str]:
    if not AUTH_ENABLED:
        return None
    username = current_user(request).get("username")
    if not username:
        raise HTTPException(401, "Authentication required")
    return username


def is_admin_user(user_token: str) -> bool:
    return (user_token or "").strip().lower() in ADMIN_USERS


def admin_identity_candidates(request: Request) -> list[str]:
    user = current_user(request)
    candidates = [
        user.get("username"),
        user.get("email"),
        user.get("display_name"),
    ]
    if not user.get("username"):
        candidates.append(request_user(request))

    seen = set()
    identities = []
    for candidate in candidates:
        token = (candidate or "").strip()
        key = token.lower()
        if token and key not in seen:
            seen.add(key)
            identities.append(token)
    return identities


def request_admin_identity(request: Request) -> Optional[str]:
    user = current_user(request)
    if AUTH_ENABLED and not user.get("username"):
        return None
    for token in admin_identity_candidates(request):
        if is_admin_user(token):
            return token
    return None


def request_is_admin(request: Request) -> bool:
    return request_admin_identity(request) is not None


def require_admin(request: Request) -> str:
    if AUTH_ENABLED:
        current_user_token(request)
    admin_identity = request_admin_identity(request)
    if not admin_identity:
        raise HTTPException(403, "需要管理员权限")
    user = current_user(request)
    return user.get("username") or admin_identity


def _owner_clause(request: Request, alias: str = "") -> tuple[str, list[str]]:
    token = current_user_token(request)
    if not token:
        return "", []
    prefix = f"{alias}." if alias else ""
    return f"{prefix}user_token = ?", [token]


def _project_access_clause(request: Request, alias: str = "") -> tuple[str, list[str]]:
    token = current_user_token(request)
    if not token:
        return "", []
    prefix = f"{alias}." if alias else ""
    return f"({prefix}user_token = ? OR {prefix}is_public = 1)", [token]


def _job_access_clause(request: Request, alias: str = "") -> tuple[str, list[str]]:
    token = current_user_token(request)
    if not token:
        return "", []
    prefix = f"{alias}." if alias else ""
    project_id = f"{prefix}project_id"
    return (
        f"({prefix}user_token = ? OR EXISTS ("
        f"SELECT 1 FROM projects p_access "
        f"WHERE p_access.id = {project_id} AND p_access.is_public = 1"
        f"))",
        [token],
    )


def _project_response(row: dict, request: Request) -> dict:
    data = dict(row)
    data["is_public"] = 1 if data.get("is_public") else 0
    token = current_user_token(request)
    data["is_owner"] = True if not token else data.get("user_token") == token
    data["visibility"] = "shared" if data["is_public"] else "personal"
    return data


def _job_response(row: dict, request: Request) -> dict:
    data = dict(row)
    token = current_user_token(request)
    data["is_owner"] = True if not token else data.get("user_token") == token
    data["is_pinned"] = 1 if data.get("is_pinned") else 0
    return data


async def ensure_user_row(db, request: Request) -> Optional[str]:
    token = current_user_token(request)
    if token:
        await db.execute("INSERT OR IGNORE INTO users(user_token) VALUES(?)", (token,))
    return token


async def load_owned_job(db, request: Request, job_id: str, columns: str = "*") -> Optional[dict]:
    owner_sql, owner_params = _owner_clause(request)
    clauses = ["id=?"]
    params = [job_id]
    if owner_sql:
        clauses.append(owner_sql)
        params.extend(owner_params)
    row = await row_to_dict(
        await (await db.execute(f"SELECT {columns} FROM jobs WHERE {' AND '.join(clauses)}", params)).fetchone()
    )
    return row


async def load_accessible_job(db, request: Request, job_id: str, columns: str = "*") -> Optional[dict]:
    access_sql, access_params = _job_access_clause(request)
    clauses = ["id=?"]
    params = [job_id]
    if access_sql:
        clauses.append(access_sql)
        params.extend(access_params)
    row = await row_to_dict(
        await (await db.execute(f"SELECT {columns} FROM jobs WHERE {' AND '.join(clauses)}", params)).fetchone()
    )
    return row


async def load_owned_project(db, request: Request, project_id: Optional[str]) -> Optional[dict]:
    if not project_id:
        return None
    owner_sql, owner_params = _owner_clause(request)
    clauses = ["id=?"]
    params = [project_id]
    if owner_sql:
        clauses.append(owner_sql)
        params.extend(owner_params)
    return await row_to_dict(
        await (await db.execute(f"SELECT * FROM projects WHERE {' AND '.join(clauses)}", params)).fetchone()
    )


async def load_accessible_project(db, request: Request, project_id: Optional[str]) -> Optional[dict]:
    if not project_id:
        return None
    access_sql, access_params = _project_access_clause(request)
    clauses = ["id=?"]
    params = [project_id]
    if access_sql:
        clauses.append(access_sql)
        params.extend(access_params)
    return await row_to_dict(
        await (await db.execute(f"SELECT * FROM projects WHERE {' AND '.join(clauses)}", params)).fetchone()
    )


async def validate_project_access(db, request: Request, project_id: Optional[str]):
    if not project_id:
        return
    if not await load_accessible_project(db, request, project_id):
        raise HTTPException(404, "Project not found")


def _record_http_metric(method: str, path: str, status: int, duration: float):
    key = (method, path, status)
    item = HTTP_METRICS.setdefault(key, {"count": 0.0, "sum": 0.0})
    item["count"] += 1
    item["sum"] += duration


async def write_audit(
    db,
    request: Request,
    action: str,
    *,
    resource_type: str = "",
    resource_id: str = "",
    details: Optional[dict] = None,
):
    await db.execute(
        """
        INSERT INTO audit_logs(id, user, action, resource_type, resource_id, ip, detail_json)
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            str(uuid.uuid4()),
            request_user(request),
            action,
            resource_type,
            resource_id,
            request_ip(request),
            json.dumps(details or {}, ensure_ascii=False, default=str),
        ),
    )

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


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    try:
        request.state.user = request.session.get("user")
    except AssertionError:
        request.state.user = None

    if AUTH_ENABLED and not _auth_public_path(request.url.path) and not request.state.user:
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return await call_next(request)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    status_code = 500
    response = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception(
            "http_request_failed",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "user": request_user(request),
                "ip": request_ip(request),
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
            },
        )
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        _record_http_metric(request.method, route_path, status_code, duration_ms / 1000)
        if response is not None:
            response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "user": request_user(request),
                "ip": request_ip(request),
                "method": request.method,
                "path": route_path,
                "raw_path": request.url.path,
                "status": status_code,
                "duration_ms": round(duration_ms, 3),
            },
        )


app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=SESSION_COOKIE_SECURE,
)


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


def ai_analysis_dir(job_id: str) -> str:
    return os.path.join(result_dir(job_id), AI_ANALYSIS_DIRNAME)


def ai_analysis_status_path(job_id: str) -> str:
    return os.path.join(ai_analysis_dir(job_id), AI_ANALYSIS_STATUS_FILE)


def ai_analysis_report_path(job_id: str) -> str:
    return os.path.join(ai_analysis_dir(job_id), AI_ANALYSIS_REPORT_FILE)


def feedback_dir() -> str:
    return os.path.join(STORAGE_DIR, FEEDBACK_DIRNAME)


def feedback_message_dir(message_id: str) -> str:
    return os.path.join(feedback_dir(), message_id)


def _safe_feedback_storage_path(path: str) -> Optional[str]:
    if not path:
        return None
    root = os.path.abspath(feedback_dir())
    full = os.path.abspath(path)
    try:
        if os.path.commonpath([root, full]) != root:
            return None
    except ValueError:
        return None
    return full


def _remove_feedback_storage(paths: list[str], message_ids: list[str]):
    for path in paths:
        full = _safe_feedback_storage_path(path)
        if full and os.path.isfile(full):
            with contextlib.suppress(OSError):
                os.remove(full)
    for message_id in message_ids:
        full = _safe_feedback_storage_path(feedback_message_dir(message_id))
        if full and os.path.isdir(full):
            shutil.rmtree(full, ignore_errors=True)


def require_code_execution_enabled():
    if not ALLOW_CODE_EXECUTION:
        raise HTTPException(403, "Code execution is disabled")


def _text_or_empty(value: Optional[str]) -> str:
    text = (value or "").strip()
    return text or "<empty>"


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_triton_execution_error(
    result,
    *,
    command: list[str],
    code_path: Optional[str] = None,
    no_output: bool = False,
) -> str:
    if no_output:
        lines = [
            "脚本执行成功，但没有输出可解析的结果。",
            "",
            "可能原因:",
            "- 该 Triton 代码只包含 kernel/function 定义，没有实际运行 benchmark。",
            "- 脚本运行后没有 print 性能结果，前端无法提取 GB/s。",
            '- 如果是手动编辑代码，请确认脚本末尾会执行并打印结果，例如 "123.45 GB/s"。',
            "",
        ]
    else:
        lines = [
            f"脚本执行失败，返回码: {result.returncode}",
            "",
        ]

    lines.extend([
        f"Command: {_format_command(command)}",
    ])
    if code_path:
        lines.append(f"Code path: {code_path}")
    lines.extend([
        f"stdout:\n{_text_or_empty(result.stdout)}",
        f"stderr:\n{_text_or_empty(result.stderr)}",
    ])
    return "\n".join(lines)


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


def _extract_zip_to_json(zip_path: str, dest_path: str):
    """Extract the largest .json file from a zip archive into dest_path."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".json")
        ]
        if not members:
            raise ValueError("压缩包中未找到 .json 文件")
        member = max(members, key=lambda info: info.file_size)
        with archive.open(member) as src:
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(src, out)


async def save_and_extract(upload: UploadFile, dest_json: str, gzip_path: list):
    """Save upload and prepare a JSON file for analysis.

    Compressed uploads keep a .json.gz copy for download/storage. Zip archives are
    normalized to gzip-compressed JSON so downloaded traces stay tool-friendly.
    """
    filename = (upload.filename or "").lower()
    if filename.endswith(".zip"):
        temp_zip = dest_json + ".zip"
        gzip_path[0] = dest_json + ".gz"
        await save_upload(upload, temp_zip)
        try:
            await asyncio.to_thread(_extract_zip_to_json, temp_zip, dest_json)
            await asyncio.to_thread(_compress_json_to_gz, dest_json, gzip_path[0])
        except Exception as e:
            raise HTTPException(400, f"解压失败: {e}")
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(temp_zip)
    elif filename.endswith((".gz", ".tgz")):
        # Plain gzip can be kept as-is; tar.gz/tgz is normalized to gzip JSON.
        gzip_path[0] = dest_json + ".gz"
        await save_upload(upload, gzip_path[0])
        try:
            is_tar_archive = await asyncio.to_thread(_is_tar_archive, gzip_path[0])
            await asyncio.to_thread(_extract_gz_to_json, gzip_path[0], dest_json)
            if is_tar_archive:
                await asyncio.to_thread(_compress_json_to_gz, dest_json, gzip_path[0])
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
    if lower.endswith(".zip"):
        base = name[:-4]
        return base if base.lower().endswith(".json") else base + ".json"
    return name if lower.endswith(".json") else name + ".json"


def _gzip_json_download_name(filename: Optional[str], slot: str) -> str:
    return _json_download_name(filename, slot) + ".gz"


def _content_disposition(filename: str, disposition: str = "attachment") -> str:
    return f'{disposition}; filename="{filename}"'


def _app_base_url(request: Request) -> str:
    host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host")
    proto = request.headers.get("X-Forwarded-Proto") or request.url.scheme
    if host:
        return f"{proto}://{host}"
    return str(request.base_url).rstrip("/")


def _job_share_url(request: Request, jid: str) -> str:
    return f"{_app_base_url(request)}/#/job/{jid}"


def _markdown_cell(value, max_len: int = 160) -> str:
    text = str(value if value is not None else "").replace("\n", " ").replace("\r", " ")
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text.replace("|", "\\|")


def _markdown_table(fields: list[str], rows: list[dict]) -> str:
    if not fields:
        return "_无数据_"
    lines = [
        "| " + " | ".join(_markdown_cell(field, 80) for field in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def _report_csv_sections(jid: str, mode: str) -> list[str]:
    title_by_file = {
        "kernel_types_avg.csv": "Kernel 类型汇总",
        "kernel_types_cmp.csv": "Kernel 类型对比",
        "all_kernels_avg.csv": "热点 Kernel",
        "all_kernels_cmp.csv": "Kernel 对比",
        "triton_kernels_avg.csv": "Triton Kernel",
        "triton_kernels_cmp.csv": "Triton Kernel 对比",
        "aten_ops_avg.csv": "Aten Ops",
        "aten_ops_cmp.csv": "Aten Ops 对比",
        "cncl_ops_avg.csv": "CNCL Ops",
        "cncl_ops_cmp.csv": "CNCL Ops 对比",
    }
    preferred = (
        ["kernel_types_cmp.csv", "all_kernels_cmp.csv", "triton_kernels_cmp.csv", "aten_ops_cmp.csv", "cncl_ops_cmp.csv"]
        if mode == "compare"
        else ["kernel_types_avg.csv", "all_kernels_avg.csv", "triton_kernels_avg.csv", "aten_ops_avg.csv", "cncl_ops_avg.csv"]
    )
    sections = []
    for filename in preferred:
        path = os.path.join(result_dir(jid), filename)
        if not os.path.exists(path):
            continue
        page = read_csv_page(path, limit=10, offset=0)
        sections.append(f"### {title_by_file.get(filename, filename)}\n\n{_markdown_table(page['fields'], page['rows'])}")
    return sections


def _console_excerpt(console_out: str, max_lines: int = 120) -> str:
    lines = (console_out or "").splitlines()
    if not lines:
        return "无控制台输出。"
    suffix = ""
    if len(lines) > max_lines:
        suffix = f"\n... 省略 {len(lines) - max_lines} 行"
    return "\n".join(lines[:max_lines]) + suffix


def csv_to_rows(path: str) -> dict:
    """Read a CSV file and return {fields, rows}."""
    if not os.path.exists(path):
        return {"fields": [], "rows": []}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = _result_csv_fields(os.path.basename(path), reader.fieldnames or [])
        rows = [_result_csv_row(os.path.basename(path), row) for row in reader]
        return {"fields": fields, "rows": rows}


def _is_percent_csv_field(field: str) -> bool:
    name = (field or "").strip().lower()
    return (
        name == "pct"
        or name.startswith("pct_")
        or name.endswith("_pct")
        or "_pct_" in name
        or "percent" in name
        or "percentage" in name
    )


def _filter_result_csv_fields(fields: list[str]) -> list[str]:
    return [field for field in fields if not _is_percent_csv_field(field)]


def _filter_result_csv_row(row: dict) -> dict:
    return {field: value for field, value in row.items() if not _is_percent_csv_field(field)}


def _augment_result_csv_fields(filename: str, fields: list[str]) -> list[str]:
    if filename != "all_kernels_cmp.csv" or "family" in fields or "kernel_name" not in fields:
        return fields
    idx = fields.index("kernel_name") + 1
    return fields[:idx] + ["family"] + fields[idx:]


def _augment_result_csv_row(filename: str, row: dict) -> dict:
    if filename == "all_kernels_cmp.csv" and "family" not in row and row.get("kernel_name"):
        row = dict(row)
        row["family"] = extract_kernel_family(row["kernel_name"])
    return row


def _result_csv_fields(filename: str, fields: list[str]) -> list[str]:
    return _filter_result_csv_fields(_augment_result_csv_fields(filename, fields))


def _result_csv_row(filename: str, row: dict) -> dict:
    return _filter_result_csv_row(_augment_result_csv_row(filename, row))


def _ordered_result_csv_names(rdir: str) -> list[str]:
    names = []
    for name in ["all_kernels_avg.csv", "all_kernels_cmp.csv",
                 "triton_kernels_avg.csv", "triton_kernels_cmp.csv",
                 "aten_ops_avg.csv", "aten_ops_cmp.csv",
                 "kernel_types_avg.csv", "kernel_types_cmp.csv",
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


def _safe_child_path(root: str, rel_path: str, error_message: str) -> str:
    root_abs = os.path.abspath(root)
    full = os.path.abspath(os.path.normpath(os.path.join(root_abs, rel_path)))
    try:
        in_root = os.path.commonpath([root_abs, full]) == root_abs
    except ValueError:
        in_root = False
    if not in_root:
        raise HTTPException(400, error_message)
    return full


def collect_result_files(jid: str) -> dict:
    rdir = result_dir(jid)
    files = {}
    for name in _ordered_result_csv_names(rdir):
        full = os.path.join(rdir, name)
        fields = []
        with open(full, newline="") as f:
            reader = csv.reader(f)
            fields = _result_csv_fields(name, next(reader, []) or [])
        files[name] = {
            "fields": fields,
            "size": _path_size(full),
        }
    return files


def _read_ai_analysis_status(jid: str) -> dict:
    path = ai_analysis_status_path(jid)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_ai_analysis_status(jid: str, status: dict):
    os.makedirs(ai_analysis_dir(jid), exist_ok=True)
    path = ai_analysis_status_path(jid)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _find_latest_ai_report(analysis_dir: str) -> Optional[str]:
    candidates = []
    for root, dirs, files in os.walk(analysis_dir):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for filename in files:
            if not filename.lower().endswith(".md") or filename == AI_ANALYSIS_REPORT_FILE:
                continue
            path = os.path.join(root, filename)
            priority = 1 if filename == "report.md" else 0
            candidates.append((priority, os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _read_text_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _detect_ai_analysis_failure(content: str) -> str:
    lowered = (content or "").lower()
    for pattern in AI_ANALYSIS_FAILURE_PATTERNS:
        if pattern in lowered:
            return pattern
    return ""


def _ai_artifact_priority(path: str) -> tuple[int, str]:
    basename = os.path.basename(path)
    if path == AI_ANALYSIS_REPORT_FILE:
        return (0, path)
    if basename == "report.md":
        return (1, path)
    if basename == "stdout.txt":
        return (2, path)
    if basename == "stderr.txt":
        return (3, path)
    if basename.lower().endswith(".md"):
        return (4, path)
    return (5, path)


def _collect_ai_analysis_artifacts(jid: str) -> list[dict]:
    analysis_dir = ai_analysis_dir(jid)
    if not os.path.isdir(analysis_dir):
        return []

    root_abs = os.path.abspath(analysis_dir)
    artifacts = []
    total_read = 0
    for root, dirs, files in os.walk(root_abs):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for filename in sorted(files):
            if filename.startswith(".") or filename.endswith(".tmp"):
                continue
            if filename in AI_ANALYSIS_INTERNAL_FILES:
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in AI_ANALYSIS_ARTIFACT_EXTENSIONS:
                continue

            full = os.path.abspath(os.path.join(root, filename))
            try:
                if os.path.commonpath([root_abs, full]) != root_abs:
                    continue
            except ValueError:
                continue
            rel_path = os.path.relpath(full, root_abs)
            size = _path_size(full)
            remaining = max(0, AI_ANALYSIS_ARTIFACT_MAX_TOTAL_BYTES - total_read)
            read_limit = min(size, AI_ANALYSIS_ARTIFACT_MAX_BYTES, remaining)
            content = ""
            truncated = size > read_limit
            if read_limit > 0:
                with open(full, "rb") as f:
                    data = f.read(read_limit)
                total_read += len(data)
                content = data.decode("utf-8", errors="replace")
            artifacts.append({
                "path": rel_path,
                "name": filename,
                "size": size,
                "mtime": datetime.fromtimestamp(os.path.getmtime(full), timezone.utc).isoformat(),
                "truncated": truncated,
                "content": content,
            })

    artifacts.sort(key=lambda item: _ai_artifact_priority(item["path"]))
    return artifacts


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        normalized = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _ai_analysis_progress(status: dict) -> int:
    explicit = status.get("progress")
    try:
        if explicit is not None:
            return max(0, min(100, int(explicit)))
    except (TypeError, ValueError):
        pass
    state = status.get("status")
    phase = status.get("phase")
    if state == "done":
        return 100
    if state == "error":
        return 100
    if state == "running":
        return {"queued": 5, "diagnosing": 20, "analyzing": 55}.get(phase, 35)
    return 0


def _ai_analysis_timing(status: dict) -> dict:
    started = _parse_iso_datetime(status.get("started_at") or "")
    if not started:
        return {"elapsed_ms": 0}
    finished = _parse_iso_datetime(status.get("finished_at") or "")
    end = finished or datetime.now(timezone.utc)
    elapsed_ms = max(0, int((end - started).total_seconds() * 1000))
    timing = {"elapsed_ms": elapsed_ms}
    if finished:
        timing["duration_ms"] = elapsed_ms
    return timing


def collect_ai_analysis(jid: str) -> dict:
    status = _read_ai_analysis_status(jid)
    report_path = ai_analysis_report_path(jid)
    report_exists = os.path.exists(report_path)
    if not status and report_exists:
        status = {"status": "done", "updated_at": datetime.fromtimestamp(os.path.getmtime(report_path), timezone.utc).isoformat()}
    if not status:
        status = {"status": "not_started"}
    if status.get("status") == "running" and jid not in ai_analysis_tasks:
        status = {
            **status,
            "status": "error",
            "error": "Server restarted before this AI analysis completed",
        }
    timing = _ai_analysis_timing(status)
    return {
        "enabled": CLAUDE_ANALYSIS_ENABLED,
        "status": status.get("status", "not_started"),
        "error": status.get("error", ""),
        "mode": status.get("mode", ""),
        "skill": status.get("skill", ""),
        "phase": status.get("phase", ""),
        "progress": _ai_analysis_progress(status),
        "diagnostics": status.get("diagnostics"),
        "started_at": status.get("started_at", ""),
        "finished_at": status.get("finished_at", ""),
        "updated_at": status.get("updated_at", ""),
        **timing,
        "report_exists": report_exists,
    }


def _read_ai_analysis_report(jid: str) -> str:
    return _read_text_file(ai_analysis_report_path(jid))


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _build_claude_command(prompt: str, values: dict) -> list[str]:
    values = {
        **values,
        "prompt": prompt,
        "single_skill": CLAUDE_SINGLE_TRACE_SKILL,
        "compare_skill": CLAUDE_COMPARE_TRACE_SKILL,
    }
    if CLAUDE_ANALYSIS_COMMAND_TEMPLATE.strip():
        raw_parts = shlex.split(CLAUDE_ANALYSIS_COMMAND_TEMPLATE)
        command = [part.format_map(_SafeFormatDict(values)) for part in raw_parts]
        if "{prompt}" not in CLAUDE_ANALYSIS_COMMAND_TEMPLATE:
            command.append(prompt)
        return _normalize_claude_command(command)

    command = shlex.split(CLAUDE_ANALYSIS_COMMAND) or ["claude"]
    if CLAUDE_ANALYSIS_EXTRA_ARGS.strip():
        command.extend(
            part.format_map(_SafeFormatDict(values))
            for part in shlex.split(CLAUDE_ANALYSIS_EXTRA_ARGS)
        )
    command.extend(["-p", prompt])
    return _normalize_claude_command(command)


def _normalize_claude_command(command: list[str]) -> list[str]:
    normalized = []
    has_skip_permissions = "--dangerously-skip-permissions" in command
    i = 0
    while i < len(command):
        part = command[i]
        if part == "--permission-mode" and i + 1 < len(command) and command[i + 1] == "bypassPermissions":
            if not has_skip_permissions:
                normalized.append("--dangerously-skip-permissions")
                has_skip_permissions = True
            i += 2
            continue
        if part == "--permission-mode=bypassPermissions":
            if not has_skip_permissions:
                normalized.append("--dangerously-skip-permissions")
                has_skip_permissions = True
            i += 1
            continue
        normalized.append(part)
        i += 1
    return normalized


def _validate_claude_command(command: list[str]) -> None:
    executable = command[0] if command else ""
    if not executable:
        raise RuntimeError("Claude analysis command is empty. Check TRACE_CLAUDE_COMMAND_TEMPLATE.")
    if os.path.sep in executable or (os.path.altsep and os.path.altsep in executable):
        if not os.path.exists(executable):
            raise RuntimeError(
                f"Claude Code command not found: {executable}. "
                "Install Claude Code on the deployment host or set TRACE_CLAUDE_COMMAND to a valid absolute path."
            )
        if not os.access(executable, os.X_OK):
            raise RuntimeError(f"Claude Code command is not executable: {executable}")
        return
    if shutil.which(executable):
        return
    raise RuntimeError(
        f"Claude Code command not found: {executable}. "
        "Install Claude Code on the deployment host, add it to the service PATH, "
        "or set TRACE_CLAUDE_COMMAND / TRACE_CLAUDE_COMMAND_TEMPLATE to the absolute command path."
    )


def _resolve_claude_skills_dir() -> str:
    skills_dir = (CLAUDE_ANALYSIS_SKILLS_DIR or "").strip()
    if not skills_dir:
        return ""
    return os.path.abspath(os.path.expanduser(skills_dir))


def _validate_claude_skill(skill: str, skills_dir: str) -> None:
    if not skills_dir:
        return
    if not os.path.isdir(skills_dir):
        raise RuntimeError(
            f"Claude skills directory not found: {skills_dir}. "
            "Set TRACE_CLAUDE_SKILLS_DIR to the directory containing e2e-profiling-analyzer "
            "and e2e-profiling-comparator."
        )
    skill_file = os.path.join(skills_dir, skill, "SKILL.md")
    if not os.path.isfile(skill_file):
        raise RuntimeError(
            f"Claude skill not found: {skill} in {skills_dir}. "
            "Check TRACE_CLAUDE_SINGLE_SKILL / TRACE_CLAUDE_COMPARE_SKILL and the skills directory."
        )


def _mount_claude_skills_for_analysis(analysis_dir: str, skills_dir: str) -> None:
    if not skills_dir:
        return
    claude_dir = os.path.join(analysis_dir, ".claude")
    target = os.path.join(claude_dir, "skills")
    if os.path.islink(target):
        os.unlink(target)
    if os.path.exists(target):
        return
    os.makedirs(claude_dir, exist_ok=True)
    shutil.copytree(skills_dir, target, dirs_exist_ok=True)


def _build_claude_env(base_env: dict, analysis_dir: str, extra: Optional[dict] = None) -> dict:
    env = base_env.copy()
    if extra:
        env.update(extra)
    claude_dir = os.path.join(analysis_dir, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    env.setdefault("CLAUDE_PROJECT_DIR", analysis_dir)
    env.setdefault("CLAUDE_CODE_PROJECT_DIR", analysis_dir)
    env.setdefault("TRACE_CLAUDE_PROJECT_DIR", analysis_dir)
    return env


def _tail_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _diagnostic_check(name: str, label: str, status: str, detail: str = "", **extra) -> dict:
    return {
        "name": name,
        "label": label,
        "status": status,
        "detail": detail,
        **extra,
    }


async def _run_claude_diagnostic_prompt(
    name: str,
    label: str,
    prompt: str,
    cwd: str,
    env: dict,
    values: dict,
) -> dict:
    started = time.time()
    command = _build_claude_command(prompt, values)
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=CLAUDE_DIAGNOSTIC_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            return _diagnostic_check(
                name,
                label,
                "error",
                f"Timed out after {CLAUDE_DIAGNOSTIC_TIMEOUT_SECONDS}s",
                exit_code=None,
                duration_ms=round((time.time() - started) * 1000, 3),
                command=_format_command(command),
                stdout_tail=_tail_text(stdout_text),
                stderr_tail=_tail_text(stderr_text),
            )

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        ok = proc.returncode == 0
        return _diagnostic_check(
            name,
            label,
            "ok" if ok else "error",
            "Command completed" if ok else f"Command exited with code {proc.returncode}",
            exit_code=proc.returncode,
            duration_ms=round((time.time() - started) * 1000, 3),
            command=_format_command(command),
            stdout_tail=_tail_text(stdout_text),
            stderr_tail=_tail_text(stderr_text),
        )
    except FileNotFoundError as e:
        executable = command[0] if command else str(e)
        return _diagnostic_check(
            name,
            label,
            "error",
            f"Claude Code command not found: {executable}",
            exit_code=None,
            duration_ms=round((time.time() - started) * 1000, 3),
            command=_format_command(command),
            stdout_tail="",
            stderr_tail="",
        )
    except Exception as e:
        return _diagnostic_check(
            name,
            label,
            "error",
            str(e),
            exit_code=None,
            duration_ms=round((time.time() - started) * 1000, 3),
            command=_format_command(command),
            stdout_tail="",
            stderr_tail="",
        )


async def _run_claude_tool_probe(cwd: str, env: dict, values: dict) -> dict:
    probe_file = "claude_tool_probe.txt"
    probe_path = os.path.join(cwd, probe_file)
    probe_token = secrets.token_hex(16)
    probe_env = {**env, "TRACE_AI_TOOL_PROBE_TOKEN": probe_token}
    with contextlib.suppress(FileNotFoundError):
        os.remove(probe_path)
    probe_python = (
        "import os; "
        f"open({probe_file!r}, 'w', encoding='utf-8').write(os.environ['TRACE_AI_TOOL_PROBE_TOKEN'])"
    )
    probe_command = f"python3 -c {shlex.quote(probe_python)}"
    prompt = (
        "这是一个 Claude Code 工具权限诊断，请不要使用任何 skill。\n"
        "环境变量 TRACE_AI_TOOL_PROBE_TOKEN 中有一个服务端随机 token，提示词里没有该 token 的值。\n"
        "请必须调用 Bash 工具，在当前工作目录执行下面这条命令，把 token 写入探针文件。\n"
        "命令执行成功后只回复一行 OK；如果不能调用 Bash 工具，请回复 ERROR: tool denied。\n\n"
        f"{probe_command}"
    )
    check = await _run_claude_diagnostic_prompt(
        "tool_probe",
        "工具权限探针",
        prompt,
        cwd,
        probe_env,
        values,
    )
    probe_ok = False
    try:
        with open(probe_path, encoding="utf-8") as f:
            probe_ok = secrets.compare_digest(f.read().strip(), probe_token)
    except OSError:
        probe_ok = False
    if check.get("status") == "ok" and probe_ok:
        return {
            **check,
            "detail": "Command completed and Bash probe file was created with the expected token",
            "probe_path": probe_path,
        }
    hint = (
        "Claude did not create the Bash probe file. Tool permissions may be denied; "
        "keep TRACE_CLAUDE_EXTRA_ARGS='--dangerously-skip-permissions' for server-side analysis, "
        "or use TRACE_CLAUDE_COMMAND_TEMPLATE with equivalent non-interactive permissions."
    )
    return {
        **check,
        "status": "error",
        "detail": f"{check.get('detail') or 'Command completed'}; {hint}",
        "probe_path": probe_path,
    }


async def _run_ai_diagnostics_payload(
    *,
    analysis_dir: Optional[str] = None,
    job_id: str = "diagnostic",
    mode: str = "diagnostic",
    skill: Optional[str] = None,
    result_dir_path: str = "",
    report_path: str = "",
) -> dict:
    started = time.time()
    started_at = _utc_now_iso()
    skills_dir = _resolve_claude_skills_dir()
    smoke_skill = skill or CLAUDE_SINGLE_TRACE_SKILL
    values = {
        "job_id": job_id,
        "mode": mode,
        "skill": smoke_skill,
        "skills_dir": skills_dir,
        "trace_a": "",
        "trace_b": "",
        "results_dir": result_dir_path,
        "analysis_dir": analysis_dir or "",
        "report_path": report_path,
    }
    checks: list[dict] = []

    base_prompt = "请不要使用任何 skill，不要调用工具，只回复一行 OK。"
    base_command = _build_claude_command(base_prompt, values)
    command_ok = False
    try:
        _validate_claude_command(base_command)
        command_ok = True
        checks.append(_diagnostic_check(
            "command",
            "Claude 命令",
            "ok",
            f"Resolved command: {_format_command(base_command)}",
            command=_format_command(base_command),
        ))
    except Exception as e:
        checks.append(_diagnostic_check(
            "command",
            "Claude 命令",
            "error",
            str(e),
            command=_format_command(base_command),
        ))

    skills_ok = True
    if skills_dir:
        if os.path.isdir(skills_dir):
            checks.append(_diagnostic_check("skills_dir", "Skills 目录", "ok", skills_dir))
        else:
            skills_ok = False
            checks.append(_diagnostic_check("skills_dir", "Skills 目录", "error", f"Directory not found: {skills_dir}"))
    else:
        checks.append(_diagnostic_check("skills_dir", "Skills 目录", "skipped", "TRACE_CLAUDE_SKILLS_DIR is empty"))

    for name, label, skill_name in (
        ("single_skill", "单 trace skill", CLAUDE_SINGLE_TRACE_SKILL),
        ("compare_skill", "对比 skill", CLAUDE_COMPARE_TRACE_SKILL),
    ):
        try:
            _validate_claude_skill(skill_name, skills_dir)
            checks.append(_diagnostic_check(name, label, "ok", f"{skill_name}"))
        except Exception as e:
            skills_ok = False
            checks.append(_diagnostic_check(name, label, "error", str(e)))

    async def run_in_dir(tmp_dir: str) -> None:
        nonlocal skills_ok
        env = _build_claude_env(os.environ, tmp_dir, {
            "TRACE_AI_JOB_ID": job_id,
            "TRACE_AI_MODE": mode,
            "TRACE_AI_SKILL": smoke_skill,
            "TRACE_AI_TRACE_A": "",
            "TRACE_AI_TRACE_B": "",
            "TRACE_AI_RESULT_DIR": result_dir_path or tmp_dir,
            "TRACE_AI_ANALYSIS_DIR": tmp_dir,
            "TRACE_AI_REPORT_PATH": report_path or os.path.join(tmp_dir, "diagnostic.md"),
        })
        if skills_dir:
            env["TRACE_CLAUDE_SKILLS_DIR"] = skills_dir
        if skills_ok:
            try:
                _mount_claude_skills_for_analysis(tmp_dir, skills_dir)
            except Exception as e:
                skills_ok = False
                checks.append(_diagnostic_check("skills_mount", "Skills 挂载", "error", str(e)))
            else:
                checks.append(_diagnostic_check(
                    "skills_mount",
                    "Skills 挂载",
                    "ok",
                    os.path.join(tmp_dir, ".claude", "skills"),
                ))

        base_smoke_ok = False
        if command_ok:
            base_smoke = await _run_claude_diagnostic_prompt(
                "base_smoke",
                "基础 Claude 调用",
                base_prompt,
                tmp_dir,
                env,
                values,
            )
            base_smoke_ok = base_smoke.get("status") == "ok"
            checks.append(base_smoke)
        else:
            checks.append(_diagnostic_check("base_smoke", "基础 Claude 调用", "skipped", "Claude command is invalid"))

        tool_probe_ok = False
        if command_ok and base_smoke_ok:
            tool_probe = await _run_claude_tool_probe(tmp_dir, env, values)
            tool_probe_ok = tool_probe.get("status") == "ok"
            checks.append(tool_probe)
        else:
            reason = "基础 Claude 调用失败" if not base_smoke_ok else "Claude command is invalid"
            checks.append(_diagnostic_check("tool_probe", "工具权限探针", "skipped", reason))

        if command_ok and skills_ok and base_smoke_ok and tool_probe_ok:
            skill_values = {**values, "skill": smoke_skill, "analysis_dir": tmp_dir, "results_dir": result_dir_path or tmp_dir}
            skill_prompt = (
                f"请使用 Claude Code skill `{smoke_skill}`，automatic-final 模式。"
                "这是环境诊断，不要读取 trace，不要运行重型分析脚本；如果 skill 可用，只回复一行 OK。"
            )
            checks.append(await _run_claude_diagnostic_prompt(
                "skill_smoke",
                "AI skill 调用",
                skill_prompt,
                tmp_dir,
                env,
                skill_values,
            ))
        else:
            if not base_smoke_ok:
                reason = "基础 Claude 调用失败"
            elif not tool_probe_ok:
                reason = "工具权限探针失败"
            else:
                reason = "skills check failed"
            checks.append(_diagnostic_check("skill_smoke", "AI skill 调用", "skipped", reason))

    if analysis_dir:
        os.makedirs(analysis_dir, exist_ok=True)
        await run_in_dir(analysis_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="analyze_trace_ai_diag_") as tmp_dir:
            await run_in_dir(tmp_dir)

    ok = all(check["status"] in {"ok", "skipped"} for check in checks) and any(
        check["name"] == "base_smoke" and check["status"] == "ok" for check in checks
    ) and any(
        check["name"] == "tool_probe" and check["status"] == "ok" for check in checks
    )
    finished_at = _utc_now_iso()
    return {
        "ok": ok,
        "enabled": CLAUDE_ANALYSIS_ENABLED,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": round((time.time() - started) * 1000, 3),
        "timeout_seconds": CLAUDE_DIAGNOSTIC_TIMEOUT_SECONDS,
        "skills_dir": skills_dir,
        "single_skill": CLAUDE_SINGLE_TRACE_SKILL,
        "compare_skill": CLAUDE_COMPARE_TRACE_SKILL,
        "checks": checks,
    }


def _format_ai_diagnostics_markdown(payload: dict) -> str:
    lines = [
        "# AI 环境诊断未通过" if not payload.get("ok") else "# AI 环境诊断通过",
        "",
        f"- 诊断结果: {'通过' if payload.get('ok') else '未通过'}",
        f"- Skills 目录: `{payload.get('skills_dir') or '-'}`",
        f"- 单 trace skill: `{payload.get('single_skill') or '-'}`",
        f"- 对比 skill: `{payload.get('compare_skill') or '-'}`",
        f"- 耗时: `{payload.get('duration_ms', '-')}` ms",
        "",
        "## 检查明细",
        "",
        "| 状态 | 检查项 | 说明 |",
        "|---|---|---|",
    ]
    status_labels = {"ok": "OK", "error": "失败", "skipped": "跳过"}
    for check in payload.get("checks") or []:
        status = status_labels.get(check.get("status"), check.get("status") or "")
        label = str(check.get("label") or check.get("name") or "").replace("|", "\\|")
        detail = str(check.get("detail") or "").replace("\n", "<br>").replace("|", "\\|")
        lines.append(f"| {status} | {label} | {detail} |")
    for check in payload.get("checks") or []:
        if not (check.get("command") or check.get("stdout_tail") or check.get("stderr_tail")):
            continue
        lines.extend([
            "",
            f"### {check.get('label') or check.get('name')}",
            "",
        ])
        if check.get("command"):
            lines.extend(["命令:", "", "```text", check["command"], "```"])
        if check.get("stdout_tail"):
            lines.extend(["stdout:", "", "```text", check["stdout_tail"], "```"])
        if check.get("stderr_tail"):
            lines.extend(["stderr:", "", "```text", check["stderr_tail"], "```"])
    lines.extend([
        "",
        "## 处理建议",
        "",
        "- 确认服务进程使用的 `TRACE_CLAUDE_COMMAND` 指向可执行的 Claude Code。",
        "- 确认 `TRACE_CLAUDE_EXTRA_ARGS` 包含 `--dangerously-skip-permissions`，并在修改后重启服务。",
        "- 确认服务运行用户对 storage、AI 分析目录和 `.claude` 临时目录有写权限。",
        "- 修复后重新点击“重新分析”，系统会再次自动诊断并继续生成报告。",
    ])
    return "\n".join(lines) + "\n"


async def _source_trace_for_ai(db, source_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not source_id:
        return None, None
    source = await row_to_dict(
        await (await db.execute(
            "SELECT file_a_name, file_a_path, file_a_gzip_path FROM jobs WHERE id=?",
            (source_id,),
        )).fetchone()
    )
    if not source:
        return None, None
    path = source.get("file_a_gzip_path") or source.get("file_a_path")
    return path, source.get("file_a_name")


async def _resolve_ai_trace_inputs(job: dict) -> dict:
    db = await get_db()
    try:
        trace_a = job.get("file_a_gzip_path") or job.get("file_a_path")
        name_a = job.get("file_a_name")
        trace_b = job.get("file_b_gzip_path") or job.get("file_b_path")
        name_b = job.get("file_b_name")

        if not trace_a:
            trace_a, source_name = await _source_trace_for_ai(db, job.get("source_job_a"))
            name_a = name_a or source_name
        if job.get("mode") == "compare" and not trace_b:
            trace_b, source_name = await _source_trace_for_ai(db, job.get("source_job_b"))
            name_b = name_b or source_name
    finally:
        await db.close()

    if not trace_a or not os.path.exists(trace_a):
        raise ValueError(f"Trace A file not found: {trace_a or '<empty>'}")
    if job.get("mode") == "compare" and (not trace_b or not os.path.exists(trace_b)):
        raise ValueError(f"Trace B file not found: {trace_b or '<empty>'}")

    return {
        "trace_a": trace_a,
        "trace_b": trace_b if job.get("mode") == "compare" else "",
        "name_a": name_a or os.path.basename(trace_a),
        "name_b": name_b or (os.path.basename(trace_b) if trace_b else ""),
    }


def _render_claude_prompt(job: dict, trace_inputs: dict, skill: str, report_path: str, skills_dir: str = "") -> str:
    rdir = result_dir(job["id"])
    mode_text = "双 trace 对比" if job.get("mode") == "compare" else "单 trace 分析"
    lines = [
        f"请使用 Claude Code skill `{skill}` 执行一次{mode_text}，不要向用户追问。",
        "如果 skill 文档区分交互模式和自动模式，请选择 automatic-final / 自动最终报告模式。",
        "分析完成后，必须生成一份用户可直接阅读的最终 Markdown 报告。",
        f"请把同一份最终报告写入 `{report_path}` 和当前工作目录的 `report.md`，并把最终报告内容输出到 stdout。",
        "最终报告中不要包含 Claude 执行过程、工具权限解释、原始 prompt、命令行流水或无关调试日志。",
        "如果工具权限、文件权限或输入文件导致无法分析，请明确写成失败报告，不要伪造成性能结论。",
        "",
        "任务信息:",
        f"- job_id: {job['id']}",
        f"- label: {job.get('label') or job['id']}",
        f"- mode: {job.get('mode')}",
        f"- results_dir: {rdir}",
        f"- final_report_path: {report_path}",
        "",
        "Trace 输入:",
        f"- A name: {trace_inputs['name_a']}",
        f"- A path: {trace_inputs['trace_a']}",
    ]
    if job.get("mode") == "compare":
        lines.extend([
            f"- B name: {trace_inputs['name_b']}",
            f"- B path: {trace_inputs['trace_b']}",
            "",
            "对比约定: A 为 baseline，B 为 current，Delta = B - A。",
        ])
    lines.extend([
        "",
        "报告要求:",
        "- 先给出 3-6 条关键结论，说明主要性能热点、回退或改善点。",
        "- 尽量引用现有 CSV / console 摘要 / trace 中的可验证数字。",
        "- 对不确定结论明确写出依据和不确定性。",
        "- 最后给出可执行优化建议，按收益和排查成本排序。",
    ])
    if skills_dir:
        lines.extend([
            "",
            f"自定义 skills 目录提示: {skills_dir}",
            f"当前工作目录已挂载 `.claude/skills`，其中应包含 `{skill}`。",
        ])
    return "\n".join(lines)


def _write_ai_report(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")
    os.replace(tmp_path, path)


def _track_ai_analysis_task(jid: str, task: asyncio.Task):
    ai_analysis_tasks[jid] = task

    def _done(done_task: asyncio.Task):
        ai_analysis_tasks.pop(jid, None)
        with contextlib.suppress(asyncio.CancelledError):
            exc = done_task.exception()
            if exc:
                logger.error(
                    "ai_analysis_task_failed",
                    extra={"event": "ai_analysis_task_failed", "job_id": jid},
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    task.add_done_callback(_done)


async def _run_ai_analysis_task(jid: str):
    analysis_dir = ai_analysis_dir(jid)
    report_path = ai_analysis_report_path(jid)
    os.makedirs(analysis_dir, exist_ok=True)
    stdout_text = ""
    stderr_text = ""

    try:
        db = await get_db()
        try:
            job = await row_to_dict(
                await (await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))).fetchone()
            )
        finally:
            await db.close()

        if not job:
            raise ValueError("Job not found")
        if job.get("status") != "done":
            raise ValueError("Job is not completed")

        trace_inputs = await _resolve_ai_trace_inputs(job)
        skill = CLAUDE_COMPARE_TRACE_SKILL if job.get("mode") == "compare" else CLAUDE_SINGLE_TRACE_SKILL
        skills_dir = _resolve_claude_skills_dir()
        status = {
            "status": "running",
            "phase": "diagnosing",
            "progress": 20,
            "mode": job.get("mode"),
            "skill": skill,
            "skills_dir": skills_dir,
            "started_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        _write_ai_analysis_status(jid, status)

        diagnostics = await _run_ai_diagnostics_payload(
            analysis_dir=analysis_dir,
            job_id=jid,
            mode=job.get("mode") or "",
            skill=skill,
            result_dir_path=result_dir(jid),
            report_path=report_path,
        )
        with open(os.path.join(analysis_dir, "ai_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2)
        if not diagnostics.get("ok"):
            content = _format_ai_diagnostics_markdown(diagnostics)
            _write_ai_report(report_path, content)
            _write_ai_analysis_status(jid, {
                **status,
                "status": "error",
                "phase": "diagnostics_failed",
                "progress": 100,
                "error": "AI environment diagnostics failed",
                "diagnostics": diagnostics,
                "finished_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            })
            return

        status = {
            **status,
            "phase": "analyzing",
            "progress": 55,
            "diagnostics": diagnostics,
            "updated_at": _utc_now_iso(),
        }
        _write_ai_analysis_status(jid, status)

        prompt = _render_claude_prompt(job, trace_inputs, skill, report_path, skills_dir)
        command = _build_claude_command(prompt, {
            "job_id": jid,
            "mode": job.get("mode") or "",
            "skill": skill,
            "skills_dir": skills_dir,
            "trace_a": trace_inputs["trace_a"],
            "trace_b": trace_inputs["trace_b"],
            "results_dir": result_dir(jid),
            "analysis_dir": analysis_dir,
            "report_path": report_path,
        })

        with open(os.path.join(analysis_dir, "command.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "command": command,
                    "display_command": _format_command(command),
                    "skill": skill,
                    "skills_dir": skills_dir,
                    "trace_a": trace_inputs["trace_a"],
                    "trace_b": trace_inputs["trace_b"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        _validate_claude_command(command)

        env = _build_claude_env(os.environ, analysis_dir, {
            "TRACE_AI_JOB_ID": jid,
            "TRACE_AI_MODE": job.get("mode") or "",
            "TRACE_AI_SKILL": skill,
            "TRACE_AI_TRACE_A": trace_inputs["trace_a"],
            "TRACE_AI_TRACE_B": trace_inputs["trace_b"],
            "TRACE_AI_RESULT_DIR": result_dir(jid),
            "TRACE_AI_ANALYSIS_DIR": analysis_dir,
            "TRACE_AI_REPORT_PATH": report_path,
        })
        if skills_dir:
            env["TRACE_CLAUDE_SKILLS_DIR"] = skills_dir

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=analysis_dir,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            executable = command[0] if command else str(e)
            raise RuntimeError(
                f"Claude Code command not found: {executable}. "
                "Install Claude Code on the deployment host, add it to the service PATH, "
                "or set TRACE_CLAUDE_COMMAND / TRACE_CLAUDE_COMMAND_TEMPLATE to the absolute command path."
            ) from e
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=CLAUDE_ANALYSIS_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Claude analysis timed out after {CLAUDE_ANALYSIS_TIMEOUT_SECONDS}s")

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        _write_ai_report(os.path.join(analysis_dir, "stdout.txt"), stdout_text or "")
        _write_ai_report(os.path.join(analysis_dir, "stderr.txt"), stderr_text or "")

        if proc.returncode != 0:
            tail = stderr_text[-2000:] or stdout_text[-2000:] or "no output"
            raise RuntimeError(f"Claude analysis failed with exit code {proc.returncode}: {tail}")

        content = _read_text_file(report_path).strip()
        if not content:
            candidate_report = _find_latest_ai_report(analysis_dir)
            if candidate_report:
                content = _read_text_file(candidate_report).strip()
            else:
                content = stdout_text.strip()
        failure_pattern = _detect_ai_analysis_failure("\n".join([content, stdout_text]))
        if failure_pattern:
            raise RuntimeError(
                "Claude produced a failure report instead of a usable analysis "
                f"(matched: {failure_pattern})"
            )
        if not content:
            content = "Claude command completed, but no report content was produced."
        _write_ai_report(report_path, content)
        _write_ai_analysis_status(jid, {
            **status,
            "status": "done",
            "progress": 100,
            "finished_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        })
    except Exception as e:
        error_report = [
            "# AI 分析失败",
            "",
            f"- 任务 ID: `{jid}`",
            f"- 错误: `{e}`",
        ]
        if stdout_text.strip():
            error_report.extend(["", "## stdout", "", "```text", stdout_text[-4000:], "```"])
        if stderr_text.strip():
            error_report.extend(["", "## stderr", "", "```text", stderr_text[-4000:], "```"])
        _write_ai_report(report_path, "\n".join(error_report))
        status = _read_ai_analysis_status(jid)
        _write_ai_analysis_status(jid, {
            **status,
            "status": "error",
            "progress": 100,
            "error": str(e),
            "finished_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        })
        logger.exception(
            "ai_analysis_failed",
            extra={"event": "ai_analysis_failed", "job_id": jid, "error": str(e)},
        )


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
        filename = os.path.basename(path)
        reader = csv.DictReader(f)
        fields = _result_csv_fields(filename, reader.fieldnames or [])
        filters = {key: value for key, value in filters.items() if key in fields}
        filter_ops = {key: value for key, value in filter_ops.items() if key in fields}
        rows = []
        total = 0
        filtered_total = 0
        requires_materialize = bool(q or filters or sort_col)
        if requires_materialize:
            for row in reader:
                row = _result_csv_row(filename, row)
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
                row = _result_csv_row(filename, row)
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


async def _compare_dependents(db, request: Request, source_job_id: str):
    access_sql, access_params = _job_access_clause(request)
    access_filter = f" AND {access_sql}" if access_sql else ""
    rows = await (
        await db.execute(
            f"""
            SELECT id, label, created_at, project_id
            FROM jobs
            WHERE mode='compare' AND (source_job_a=? OR source_job_b=?){access_filter}
            ORDER BY created_at DESC
            """,
            (source_job_id, source_job_id, *access_params),
        )
    ).fetchall()
    return [dict(row) for row in rows]


async def _compare_source_summaries(request: Request, job: dict):
    source_ids = [job.get("source_job_a"), job.get("source_job_b")]
    source_ids = [jid for jid in source_ids if jid]
    if not source_ids:
        return {}

    db = await get_db()
    try:
        placeholders = ",".join("?" * len(source_ids))
        access_sql, access_params = _job_access_clause(request, "j")
        access_filter = f" AND {access_sql}" if access_sql else ""
        rows = await (
            await db.execute(
                f"""
                SELECT j.id, j.label, j.project_id, j.created_at, j.file_a_name,
                       p.name AS project_name,
                       j.file_a_path, j.file_a_gzip_path
                FROM jobs j
                LEFT JOIN projects p ON p.id = j.project_id
                WHERE j.id IN ({placeholders}){access_filter}
                """,
                (*source_ids, *access_params),
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

    return _perfetto_context(compute_avgs(parse_trace(path)))


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


def _iter_zip_json(path: str):
    with zipfile.ZipFile(path) as archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".json")
        ]
        if not members:
            raise ValueError("Archive does not contain a JSON trace")
        member = max(members, key=lambda info: info.file_size)
        with archive.open(member) as extracted:
            while True:
                chunk = extracted.read(1 << 20)
                if not chunk:
                    break
                yield chunk


def _iter_file_chunks(path: str):
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            yield chunk


def _is_tar_archive(path: str) -> bool:
    try:
        return tarfile.is_tarfile(path)
    except Exception:
        return False


def _iter_json_chunks(path: str):
    lower = path.lower()
    if lower.endswith(".zip"):
        return _iter_zip_json(path)
    if lower.endswith((".gz", ".tgz")):
        return _iter_tar_json(path) if _is_tar_archive(path) else _iter_gzip_json(path)
    return _iter_file_chunks(path)


def _iter_gzip_encoded(chunks):
    compressor = zlib.compressobj(level=6, wbits=16 + zlib.MAX_WBITS)
    for chunk in chunks:
        compressed = compressor.compress(chunk)
        if compressed:
            yield compressed
    tail = compressor.flush()
    if tail:
        yield tail


# ── Synchronous analysis (runs in thread pool, must not await) ────────────────

def _run_sync_analysis(job, rdir, path_a, path_b, name_a, name_b):
    """All blocking I/O lives here so the event loop stays free."""
    from trace_analyzer import (compute_avgs, parse_trace,
                                print_step_summary, print_kernel_type_breakdown, print_top_kernels,
                                write_single, print_comparison, write_comparison)

    buf = io.StringIO()
    perfetto_context = {}
    with contextlib.redirect_stdout(buf):
        if job["mode"] == "single":
            data = compute_avgs(parse_trace(path_a))
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
            data_a = compute_avgs(parse_trace(path_a))
            data_b = compute_avgs(parse_trace(path_b))
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

        logger.info("analysis_started", extra={"event": "analysis_started", "job_id": job_id, "mode": job["mode"]})
        await db.execute("UPDATE jobs SET status='running', error_msg='' WHERE id=?", (job_id,))
        await db.commit()

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
            _run_sync_analysis, job, rdir, path_a, path_b, name_a, name_b
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
        logger.info("analysis_finished", extra={"event": "analysis_finished", "job_id": job_id, "mode": job["mode"]})

    except Exception as e:
        await db.execute(
            "UPDATE jobs SET status='error', error_msg=? WHERE id=?",
            (str(e), job_id),
        )
        await _refresh_job_storage_cache(db, job_id)
        await db.commit()
        logger.exception("analysis_failed", extra={"event": "analysis_failed", "job_id": job_id})
    finally:
        await db.close()


def _prom_label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _backup_manifest() -> dict:
    path = os.path.join(BACKUP_DIR, "latest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


async def _job_status_counts() -> dict[str, int]:
    db = await get_db()
    try:
        rows = await (
            await db.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status")
        ).fetchall()
        return {row["status"]: row["count"] for row in rows}
    finally:
        await db.close()


def _probe_writable_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        probe_path = os.path.join(path, f".readyz-{uuid.uuid4().hex}")
        with open(probe_path, "w") as f:
            f.write("ok")
        os.remove(probe_path)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def _probe_writable_file(path: str) -> str:
    if not path:
        return "disabled"
    try:
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(path, "a"):
            pass
        return "ok"
    except Exception as e:
        return f"error: {e}"


# ── Routes: index / config ────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/readyz")
async def readyz():
    checks = {}
    paths = {
        "storage": STORAGE_DIR,
        "backup": BACKUP_DIR,
        "log_file": os.environ.get("TRACE_LOG_FILE", ""),
    }
    db = None
    try:
        db = await get_db()
        await (await db.execute("SELECT 1")).fetchone()
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {e}"
    finally:
        if db is not None:
            await db.close()

    checks["storage"] = _probe_writable_dir(STORAGE_DIR)
    checks["backup"] = _probe_writable_dir(BACKUP_DIR)
    checks["log_file"] = _probe_writable_file(os.environ.get("TRACE_LOG_FILE", ""))

    ready = all(value in ("ok", "disabled") for value in checks.values())
    return JSONResponse(
        {"status": "ok" if ready else "error", "checks": checks, "paths": paths, "version": APP_VERSION},
        status_code=200 if ready else 503,
    )


@app.get("/metrics")
async def metrics():
    job_counts = await _job_status_counts()
    disk = shutil.disk_usage(STORAGE_DIR if os.path.exists(STORAGE_DIR) else os.path.dirname(STORAGE_DIR))
    backup = _backup_manifest()

    lines = [
        "# HELP analyze_trace_app_uptime_seconds Web application uptime.",
        "# TYPE analyze_trace_app_uptime_seconds gauge",
        f"analyze_trace_app_uptime_seconds {time.time() - APP_STARTED_AT:.3f}",
        "# HELP analyze_trace_http_requests_total HTTP requests by method, route and status.",
        "# TYPE analyze_trace_http_requests_total counter",
    ]
    for (method, path, status), item in sorted(HTTP_METRICS.items()):
        lines.append(
            f'analyze_trace_http_requests_total{{method="{_prom_label(method)}",'
            f'path="{_prom_label(path)}",status="{status}"}} {int(item["count"])}'
        )

    lines.extend([
        "# HELP analyze_trace_http_request_duration_seconds_sum Total HTTP request duration.",
        "# TYPE analyze_trace_http_request_duration_seconds_sum counter",
    ])
    for (method, path, status), item in sorted(HTTP_METRICS.items()):
        lines.append(
            f'analyze_trace_http_request_duration_seconds_sum{{method="{_prom_label(method)}",'
            f'path="{_prom_label(path)}",status="{status}"}} {item["sum"]:.6f}'
        )

    lines.extend([
        "# HELP analyze_trace_http_request_duration_seconds_count HTTP request duration sample count.",
        "# TYPE analyze_trace_http_request_duration_seconds_count counter",
    ])
    for (method, path, status), item in sorted(HTTP_METRICS.items()):
        lines.append(
            f'analyze_trace_http_request_duration_seconds_count{{method="{_prom_label(method)}",'
            f'path="{_prom_label(path)}",status="{status}"}} {int(item["count"])}'
        )

    lines.extend([
        "# HELP analyze_trace_jobs_total Jobs by status.",
        "# TYPE analyze_trace_jobs_total gauge",
    ])
    for status, count in sorted(job_counts.items()):
        lines.append(f'analyze_trace_jobs_total{{status="{_prom_label(status)}"}} {count}')

    lines.extend([
        "# HELP analyze_trace_analysis_queue_size Pending analysis queue size in memory.",
        "# TYPE analyze_trace_analysis_queue_size gauge",
        f"analyze_trace_analysis_queue_size {analysis_queue.qsize()}",
        "# HELP analyze_trace_storage_free_bytes Free bytes on the storage filesystem.",
        "# TYPE analyze_trace_storage_free_bytes gauge",
        f"analyze_trace_storage_free_bytes {disk.free}",
        "# HELP analyze_trace_storage_total_bytes Total bytes on the storage filesystem.",
        "# TYPE analyze_trace_storage_total_bytes gauge",
        f"analyze_trace_storage_total_bytes {disk.total}",
        "# HELP analyze_trace_backup_last_success_timestamp_seconds Last successful backup timestamp.",
        "# TYPE analyze_trace_backup_last_success_timestamp_seconds gauge",
        f"analyze_trace_backup_last_success_timestamp_seconds {float(backup.get('created_at_epoch', 0) or 0):.0f}",
        "# HELP analyze_trace_backup_last_size_bytes Last successful backup archive size.",
        "# TYPE analyze_trace_backup_last_size_bytes gauge",
        f"analyze_trace_backup_last_size_bytes {int(backup.get('size_bytes', 0) or 0)}",
    ])
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/api/config")
async def get_config():
    return {
        "version": APP_VERSION,
        "auth_mode": AUTH_MODE,
        "auth_required": AUTH_ENABLED,
        "allow_file_download": ALLOW_FILE_DOWNLOAD,
        "allow_code_execution": ALLOW_CODE_EXECUTION,
        "claude_analysis_enabled": CLAUDE_ANALYSIS_ENABLED,
    }


@app.get("/api/me")
async def get_me(request: Request):
    user = current_user(request)
    return {
        "authenticated": bool(user),
        "auth_mode": AUTH_MODE,
        "user": user or None,
        "is_admin": request_is_admin(request),
    }


@app.get("/api/login-captcha")
async def get_login_captcha(request: Request, username: str = ""):
    if not AUTH_ENABLED or not _captcha_required_for(request, username):
        _clear_login_captcha(request)
        return {"captcha_required": False}
    return _issue_login_captcha(request, username)


@app.post("/api/login")
async def login(request: Request, body: dict):
    if not AUTH_ENABLED:
        return {
            "authenticated": True,
            "auth_mode": AUTH_MODE,
            "user": None,
            "is_admin": request_is_admin(request),
        }
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    failure_key = _login_failure_key(request, username)
    if _captcha_required_for(request, username):
        captcha = body.get("captcha") or ""
        if not _verify_login_captcha(request, username, captcha):
            logger.info(
                "login_captcha_failed",
                extra={"event": "login_captcha_failed", "username": username, "ip": request_ip(request)},
            )
            return _login_error_response(
                request, username, "验证码错误或已过期", status_code=400, include_captcha=True
            )
    try:
        user = await asyncio.to_thread(ldap_auth.authenticate, username, password)
    except ldap_auth.AuthError as e:
        failed_count = _record_login_failure(failure_key)
        captcha_required = failed_count >= LOGIN_CAPTCHA_THRESHOLD
        logger.info(
            "login_failed",
            extra={
                "event": "login_failed",
                "username": username,
                "ip": request_ip(request),
                "failed_count": failed_count,
                "captcha_required": captcha_required,
            },
        )
        return _login_error_response(request, username, str(e), status_code=401, include_captcha=captcha_required)
    _clear_login_failure(failure_key)
    _clear_login_captcha(request)
    request.session["user"] = {
        "username": user["username"],
        "display_name": user.get("display_name") or user["username"],
        "email": user.get("email") or "",
    }
    request.state.user = request.session["user"]
    db = await get_db()
    try:
        await ensure_user_row(db, request)
        await write_audit(
            db, request, "auth.login",
            resource_type="user", resource_id=user["username"],
            details={"display_name": user.get("display_name"), "email": user.get("email")},
        )
        await db.commit()
    finally:
        await db.close()
    logger.info("login_success", extra={"event": "login_success", "username": user["username"]})
    return {
        "authenticated": True,
        "auth_mode": AUTH_MODE,
        "user": request.session["user"],
        "is_admin": request_is_admin(request),
    }


@app.post("/api/logout")
async def logout(request: Request):
    user = request.session.get("user") or {}
    if user.get("username"):
        db = await get_db()
        try:
            await write_audit(
                db, request, "auth.logout",
                resource_type="user", resource_id=user["username"],
            )
            await db.commit()
        finally:
            await db.close()
    request.session.clear()
    return {"ok": True}


@app.get("/api/audit-logs")
async def list_audit_logs(
    request: Request,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    clauses = []
    params = []
    owner = current_user_token(request)
    if owner:
        clauses.append("user = ?")
        params.append(owner)
    if action:
        clauses.append("action = ?")
        params.append(action)
    if resource_type:
        clauses.append("resource_type = ?")
        params.append(resource_type)
    if resource_id:
        clauses.append("resource_id = ?")
        params.append(resource_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    db = await get_db()
    try:
        total_row = await (
            await db.execute(f"SELECT COUNT(*) AS total FROM audit_logs {where}", params)
        ).fetchone()
        rows = await (
            await db.execute(
                f"""
                SELECT *
                FROM audit_logs
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            )
        ).fetchall()
    finally:
        await db.close()
    return {
        "data": [dict(row) for row in rows],
        "total": total_row["total"],
        "limit": limit,
        "offset": offset,
    }


# ── Routes: feedback board ───────────────────────────────────────────────────

def _feedback_author(request: Request) -> tuple[str, str]:
    user = current_user(request)
    token = user.get("username") or request_user(request)
    display = user.get("display_name") or token
    return token, display


def _detect_image_type(data: bytes, content_type: str = "") -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    raise HTTPException(400, "只支持 PNG、JPEG、GIF 或 WebP 图片")


async def _read_feedback_image(upload: UploadFile) -> tuple[bytes, str, str]:
    data = await upload.read(FEEDBACK_MAX_IMAGE_BYTES + 1)
    if len(data) > FEEDBACK_MAX_IMAGE_BYTES:
        raise HTTPException(413, f"图片不能超过 {FEEDBACK_MAX_IMAGE_BYTES // 1024 // 1024}MB")
    if not data:
        raise HTTPException(400, "图片内容为空")
    content_type, ext = _detect_image_type(data, (upload.content_type or "").lower())
    return data, content_type, ext


def _feedback_attachment_response(row: dict) -> dict:
    return {
        "id": row["id"],
        "filename": row.get("filename") or "",
        "content_type": row.get("content_type") or "",
        "size_bytes": row.get("size_bytes") or 0,
        "url": f"/api/feedback/images/{row['id']}",
    }


def _feedback_message_response(
    row: dict,
    attachments_by_message: dict,
    replies: Optional[list] = None,
    reply_count: Optional[int] = None,
    last_activity_at: Optional[str] = None,
) -> dict:
    normalized_replies = replies or []
    return {
        "id": row["id"],
        "parent_id": row.get("parent_id"),
        "user_display": row.get("user_display") or row.get("user_token") or "local",
        "body": row.get("body") or "",
        "created_at": row.get("created_at") or "",
        "updated_at": row.get("updated_at") or "",
        "attachments": attachments_by_message.get(row["id"], []),
        "replies": normalized_replies,
        "reply_count": reply_count if reply_count is not None else len(normalized_replies),
        "last_activity_at": last_activity_at or row.get("last_activity_at") or row.get("updated_at") or row.get("created_at") or "",
    }


def _feedback_title(body: str, fallback: str = "无标题留言") -> str:
    first_line = " ".join((body or "").strip().splitlines()).strip()
    if not first_line:
        return fallback
    return first_line[:60] + ("..." if len(first_line) > 60 else "")


def _feedback_body_preview(body: str, max_len: int = 240) -> str:
    text = " ".join((body or "").strip().split())
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _feedback_mentioned_emails(body: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for match in FEEDBACK_MENTION_RE.finditer(body or ""):
        email = _normalize_email(match.group(1), FEEDBACK_MENTION_DOMAIN)
        if email and email not in seen:
            seen.add(email)
            result.append(email)
    return result


def _feedback_notification_recipients(body: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for email in [*FEEDBACK_NOTIFICATION_ADMIN_EMAILS, *_feedback_mentioned_emails(body)]:
        normalized = _normalize_email(email, FEEDBACK_MENTION_DOMAIN)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _feedback_notification_subject(payload: dict) -> str:
    kind = "回复" if payload.get("parent_id") else "新帖子"
    title = payload.get("post_title") or _feedback_title(payload.get("body", ""))
    return f"[Torch Profiler Analyzer] 留言板{kind}: {title}"


def _feedback_notification_body(payload: dict) -> str:
    kind = "回复" if payload.get("parent_id") else "新帖子"
    lines = [
        f"留言板有一条{kind}",
        "",
        f"作者: {payload.get('author') or 'local'}",
        f"时间: {payload.get('created_at') or _utc_now_iso()}",
        f"帖子: {payload.get('post_title') or _feedback_title(payload.get('body', ''))}",
        f"帖子 ID: {payload.get('post_id') or ''}",
        f"消息 ID: {payload.get('message_id') or ''}",
        f"图片: {payload.get('image_count', 0)} 张",
        "",
        "内容:",
        payload.get("body") or "(仅图片)",
    ]
    mentions = payload.get("mentioned_emails") or []
    if mentions:
        lines.extend(["", "提及:", ", ".join(mentions)])
    if PUBLIC_BASE_URL:
        lines.extend(["", f"打开应用: {PUBLIC_BASE_URL}"])
    else:
        lines.extend(["", "请打开 Torch Profiler Analyzer 右上角的“改进留言板”查看详情。"])
    return "\n".join(lines)


def _feedback_email_transport() -> str:
    if SMTP_HOST:
        return "smtp"
    if SENDMAIL_COMMAND:
        return "sendmail"
    return ""


def _feedback_notification_status(recipients: list[str]) -> dict:
    if not FEEDBACK_EMAIL_NOTIFICATIONS_ENABLED:
        return {"status": "disabled", "detail": "留言板邮件通知已关闭"}
    if not recipients:
        return {"status": "no_recipients", "detail": "没有可通知的收件人"}
    transport = _feedback_email_transport()
    if not transport:
        return {
            "status": "missing_transport",
            "detail": "未配置 TRACE_SMTP_HOST 或 TRACE_SENDMAIL_COMMAND，邮件未发送",
            "recipients": recipients,
        }
    return {"status": "ready", "detail": "邮件通知可发送", "transport": transport, "recipients": recipients}


def _email_error_message(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or str(exc)
        return detail[:500]
    return str(exc)[:500]


def _send_email_sync(to_addrs: list[str], subject: str, body: str):
    if not FEEDBACK_EMAIL_NOTIFICATIONS_ENABLED:
        logger.info("feedback_email_disabled", extra={"event": "feedback_email_disabled"})
        return
    recipients = []
    seen = set()
    for addr in to_addrs:
        email = _normalize_email(addr, FEEDBACK_MENTION_DOMAIN)
        if email and email not in seen:
            seen.add(email)
            recipients.append(email)
    if not recipients:
        return
    if not SMTP_HOST and not SENDMAIL_COMMAND:
        logger.warning(
            "feedback_email_smtp_missing",
            extra={"event": "feedback_email_smtp_missing", "recipients": recipients},
        )
        return

    message = EmailMessage()
    message["From"] = SMTP_FROM
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message.set_content(body, charset="utf-8")

    if SMTP_HOST:
        smtp_cls = smtplib.SMTP_SSL if SMTP_USE_SSL else smtplib.SMTP
        with smtp_cls(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as smtp:
            if SMTP_USE_STARTTLS and not SMTP_USE_SSL:
                smtp.starttls()
            if SMTP_USERNAME:
                smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)
        return

    subprocess.run(
        [SENDMAIL_COMMAND, "-t", "-i"],
        input=message.as_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=SMTP_TIMEOUT_SECONDS,
        check=True,
    )


async def _send_feedback_email_notification(payload: dict):
    recipients = payload.get("recipients") or []
    status = _feedback_notification_status(recipients)
    if status["status"] != "ready":
        logger.warning(
            "feedback_email_not_sent",
            extra={"event": "feedback_email_not_sent", **status},
        )
        return status
    try:
        await asyncio.to_thread(
            _send_email_sync,
            recipients,
            _feedback_notification_subject(payload),
            _feedback_notification_body(payload),
        )
        sent_status = {
            **status,
            "status": "sent",
            "detail": "邮件通知已发送",
        }
        logger.info(
            "feedback_email_sent",
            extra={
                "event": "feedback_email_sent",
                "message_id": payload.get("message_id"),
                "recipients": recipients,
                "transport": status.get("transport", ""),
            },
        )
        return sent_status
    except Exception as exc:
        error = _email_error_message(exc)
        logger.warning(
            "feedback_email_failed",
            extra={
                "event": "feedback_email_failed",
                "message_id": payload.get("message_id"),
                "error": error,
                "recipients": recipients,
                "transport": status.get("transport", ""),
            },
            exc_info=True,
        )
        return {
            **status,
            "status": "failed",
            "detail": f"邮件通知发送失败: {error}",
            "error": error,
        }


async def _feedback_attachments_for(db, message_ids: list[str]) -> dict:
    if not message_ids:
        return {}
    placeholders = ",".join("?" * len(message_ids))
    rows = await (
        await db.execute(
            f"""
            SELECT id, message_id, filename, content_type, size_bytes
            FROM feedback_attachments
            WHERE message_id IN ({placeholders})
            ORDER BY created_at ASC
            """,
            tuple(message_ids),
        )
    ).fetchall()
    result: dict[str, list[dict]] = {}
    for row in rows:
        item = dict(row)
        result.setdefault(item["message_id"], []).append(_feedback_attachment_response(item))
    return result


def _mention_username(value: str) -> str:
    token = (value or "").strip().lstrip("@")
    if "@" in token:
        token = token.split("@", 1)[0]
    return token if FEEDBACK_MENTION_TOKEN_RE.fullmatch(token or "") else ""


def _mention_candidate(username: str, display_name: str = "", email: str = "", source: str = "local") -> Optional[dict]:
    token = _mention_username(username) or _mention_username(email)
    if not token:
        return None
    normalized_email = _normalize_email(email or token, FEEDBACK_MENTION_DOMAIN)
    return {
        "username": token,
        "display_name": (display_name or token).strip(),
        "email": normalized_email,
        "source": source,
    }


def _mention_candidate_rank(candidate: dict, query: str) -> tuple:
    q = (query or "").lower()
    username = (candidate.get("username") or "").lower()
    display = (candidate.get("display_name") or "").lower()
    email = (candidate.get("email") or "").lower()
    source_rank = 0 if candidate.get("source") == "ldap" else 1
    if username == q:
        rank = 0
    elif username.startswith(q):
        rank = 1
    elif display.startswith(q):
        rank = 2
    elif email.startswith(q):
        rank = 3
    elif q in username:
        rank = 4
    elif q in display:
        rank = 5
    elif q in email:
        rank = 6
    else:
        rank = 7
    return (rank, source_rank, username)


async def _local_mention_candidates(query: str, limit: int) -> list[dict]:
    q = (query or "").strip().lower()
    if not q:
        return []
    like = f"%{q}%"
    rows: list[dict] = []
    db = await get_db()
    try:
        feedback_rows = await (
            await db.execute(
                """
                SELECT user_token, user_display, MAX(created_at) AS last_seen
                FROM feedback_messages
                WHERE user_token <> ''
                  AND (lower(user_token) LIKE ? OR lower(user_display) LIKE ?)
                GROUP BY user_token, user_display
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (like, like, max(limit * 3, 20)),
            )
        ).fetchall()
        rows.extend(dict(row) for row in feedback_rows)
        job_rows = await (
            await db.execute(
                """
                SELECT user_token, user_token AS user_display, MAX(created_at) AS last_seen
                FROM jobs
                WHERE user_token <> '' AND lower(user_token) LIKE ?
                GROUP BY user_token
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (like, max(limit * 2, 20)),
            )
        ).fetchall()
        rows.extend(dict(row) for row in job_rows)
        user_rows = await (
            await db.execute(
                """
                SELECT user_token, user_token AS user_display, created_at AS last_seen
                FROM users
                WHERE user_token <> '' AND lower(user_token) LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (like, max(limit * 2, 20)),
            )
        ).fetchall()
        rows.extend(dict(row) for row in user_rows)
    finally:
        await db.close()

    candidates = []
    for row in rows:
        candidate = _mention_candidate(
            row.get("user_token") or "",
            row.get("user_display") or "",
            source="local",
        )
        if candidate:
            candidates.append(candidate)
    return candidates


def _merge_mention_candidates(candidates: list[dict], query: str, limit: int) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in candidates:
        candidate = _mention_candidate(
            item.get("username") or "",
            item.get("display_name") or "",
            item.get("email") or "",
            item.get("source") or "local",
        )
        if not candidate:
            continue
        key = (candidate["username"] or candidate["email"]).lower()
        existing = merged.get(key)
        if not existing or existing.get("source") != "ldap" and candidate.get("source") == "ldap":
            merged[key] = candidate
        elif existing and not existing.get("email") and candidate.get("email"):
            existing["email"] = candidate["email"]
    result = list(merged.values())
    result.sort(key=lambda item: _mention_candidate_rank(item, query))
    return result[:limit]


@app.get("/api/mention-candidates")
async def mention_candidates(request: Request, q: str = "", limit: int = 8):
    if AUTH_ENABLED:
        current_user_token(request)
    query = (q or "").strip()
    limit = max(1, min(limit, 20))
    if not query:
        return {"data": [], "query": query}

    candidates: list[dict] = []
    if AUTH_ENABLED:
        try:
            ldap_users = await asyncio.to_thread(ldap_auth.search_users, query, limit)
            for user in ldap_users:
                candidate = _mention_candidate(
                    user.get("username") or "",
                    user.get("display_name") or "",
                    user.get("email") or "",
                    "ldap",
                )
                if candidate:
                    candidates.append(candidate)
        except Exception as exc:
            logger.info(
                "mention_ldap_search_failed",
                extra={"event": "mention_ldap_search_failed", "error": str(exc)},
            )

    candidates.extend(await _local_mention_candidates(query, limit))
    return {"data": _merge_mention_candidates(candidates, query, limit), "query": query}


@app.get("/api/feedback")
async def list_feedback(request: Request, limit: int = 30, offset: int = 0, sort: str = "updated"):
    if AUTH_ENABLED:
        current_user_token(request)
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    sort_key = (sort or "updated").strip().lower()
    if sort_key not in {"updated", "created", "hot"}:
        sort_key = "updated"
    order_by = {
        "updated": "last_activity_at DESC, p.created_at DESC",
        "created": "p.created_at DESC",
        "hot": "reply_count DESC, last_activity_at DESC, p.created_at DESC",
    }[sort_key]
    db = await get_db()
    try:
        total_row = await (
            await db.execute("SELECT COUNT(*) AS total FROM feedback_messages WHERE parent_id IS NULL")
        ).fetchone()
        parent_rows = await (
            await db.execute(
                f"""
                SELECT p.*,
                       (
                           SELECT COUNT(*)
                           FROM feedback_messages r
                           WHERE r.parent_id = p.id
                       ) AS reply_count,
                       COALESCE(
                           (
                               SELECT MAX(r.created_at)
                               FROM feedback_messages r
                               WHERE r.parent_id = p.id
                           ),
                           p.updated_at,
                           p.created_at
                       ) AS last_activity_at
                FROM feedback_messages p
                WHERE p.parent_id IS NULL
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        ).fetchall()
        parents = [dict(row) for row in parent_rows]
        parent_ids = [row["id"] for row in parents]

        reply_rows = []
        if parent_ids:
            placeholders = ",".join("?" * len(parent_ids))
            reply_rows = await (
                await db.execute(
                    f"""
                    SELECT *
                    FROM feedback_messages
                    WHERE parent_id IN ({placeholders})
                    ORDER BY created_at ASC
                    """,
                    tuple(parent_ids),
                )
            ).fetchall()

        replies_by_parent: dict[str, list[dict]] = {}
        replies = [dict(row) for row in reply_rows]
        message_ids = parent_ids + [row["id"] for row in replies]
        attachments_by_message = await _feedback_attachments_for(db, message_ids)
        for reply in replies:
            replies_by_parent.setdefault(reply["parent_id"], []).append(
                _feedback_message_response(reply, attachments_by_message)
            )
    finally:
        await db.close()

    return {
        "data": [
            _feedback_message_response(
                parent,
                attachments_by_message,
                replies_by_parent.get(parent["id"], []),
                reply_count=parent.get("reply_count"),
                last_activity_at=parent.get("last_activity_at"),
            )
            for parent in parents
        ],
        "total": total_row["total"] if total_row else 0,
        "limit": limit,
        "offset": offset,
        "sort": sort_key,
    }


@app.get("/api/feedback/{post_id}")
async def get_feedback_post(request: Request, post_id: str):
    if AUTH_ENABLED:
        current_user_token(request)
    db = await get_db()
    try:
        found = await row_to_dict(
            await (await db.execute(
                "SELECT id, parent_id FROM feedback_messages WHERE id=?",
                (post_id,),
            )).fetchone()
        )
        if not found:
            raise HTTPException(404, "帖子不存在")
        root_id = found.get("parent_id") or found["id"]
        parent = await row_to_dict(
            await (await db.execute(
                """
                SELECT p.*,
                       (
                           SELECT COUNT(*)
                           FROM feedback_messages r
                           WHERE r.parent_id = p.id
                       ) AS reply_count,
                       COALESCE(
                           (
                               SELECT MAX(r.created_at)
                               FROM feedback_messages r
                               WHERE r.parent_id = p.id
                           ),
                           p.updated_at,
                           p.created_at
                       ) AS last_activity_at
                FROM feedback_messages p
                WHERE p.id=?
                """,
                (root_id,),
            )).fetchone()
        )
        reply_rows = await (
            await db.execute(
                """
                SELECT *
                FROM feedback_messages
                WHERE parent_id=?
                ORDER BY created_at ASC
                """,
                (root_id,),
            )
        ).fetchall()
        replies = [dict(row) for row in reply_rows]
        message_ids = [root_id] + [row["id"] for row in replies]
        attachments_by_message = await _feedback_attachments_for(db, message_ids)
    finally:
        await db.close()

    return _feedback_message_response(
        parent,
        attachments_by_message,
        [_feedback_message_response(reply, attachments_by_message) for reply in replies],
        reply_count=parent.get("reply_count"),
        last_activity_at=parent.get("last_activity_at"),
    )


@app.post("/api/feedback", status_code=201)
async def create_feedback(
    request: Request,
    body: str = Form(""),
    parent_id: Optional[str] = Form(None),
    images: Optional[list[UploadFile]] = File(None),
):
    if AUTH_ENABLED:
        current_user_token(request)
    text = (body or "").strip()
    image_files = [img for img in (images or []) if img and img.filename]
    if len(image_files) > FEEDBACK_MAX_IMAGES:
        raise HTTPException(400, f"最多上传 {FEEDBACK_MAX_IMAGES} 张图片")
    if not text and not image_files:
        raise HTTPException(400, "请输入留言内容或上传图片")

    db = await get_db()
    message_id = str(uuid.uuid4())
    saved_paths: list[str] = []
    notification_payload: Optional[dict] = None
    notification_status: Optional[dict] = None
    try:
        root_parent_id = parent_id or None
        parent_post: Optional[dict] = None
        if root_parent_id:
            parent = await row_to_dict(
                await (await db.execute(
                    "SELECT id, parent_id FROM feedback_messages WHERE id=?",
                    (root_parent_id,),
                )).fetchone()
            )
            if not parent:
                raise HTTPException(404, "留言不存在")
            if parent.get("parent_id"):
                root_parent_id = parent["parent_id"]
            parent_post = await row_to_dict(
                await (await db.execute(
                    "SELECT id, body FROM feedback_messages WHERE id=?",
                    (root_parent_id,),
                )).fetchone()
            )

        user_token, user_display = _feedback_author(request)
        await db.execute(
            """
            INSERT INTO feedback_messages(id, parent_id, user_token, user_display, body)
            VALUES(?,?,?,?,?)
            """,
            (message_id, root_parent_id, user_token, user_display, text),
        )

        target_dir = feedback_message_dir(message_id)
        os.makedirs(target_dir, exist_ok=True)
        for upload in image_files:
            data, content_type, ext = await _read_feedback_image(upload)
            attachment_id = str(uuid.uuid4())
            stored_path = os.path.join(target_dir, f"{attachment_id}{ext}")
            async with aiofiles.open(stored_path, "wb") as f:
                await f.write(data)
            saved_paths.append(stored_path)
            await db.execute(
                """
                INSERT INTO feedback_attachments(id, message_id, filename, stored_path, content_type, size_bytes)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    attachment_id,
                    message_id,
                    _safe_download_name(upload.filename, f"image{ext}")[:240],
                    stored_path,
                    content_type,
                    len(data),
                ),
            )

        if root_parent_id:
            await db.execute(
                "UPDATE feedback_messages SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (root_parent_id,),
            )

        await write_audit(
            db, request, "feedback.create",
            resource_type="feedback", resource_id=message_id,
            details={"parent_id": root_parent_id, "image_count": len(image_files)},
        )
        await db.commit()

        message = await row_to_dict(
            await (await db.execute("SELECT * FROM feedback_messages WHERE id=?", (message_id,))).fetchone()
        )
        attachments_by_message = await _feedback_attachments_for(db, [message_id])
        recipients = _feedback_notification_recipients(text)
        notification_payload = {
            "message_id": message_id,
            "post_id": root_parent_id or message_id,
            "parent_id": root_parent_id,
            "author": user_display,
            "user_token": user_token,
            "body": text,
            "created_at": message.get("created_at") if message else _utc_now_iso(),
            "image_count": len(image_files),
            "post_title": _feedback_title((parent_post or {}).get("body") or text),
            "mentioned_emails": _feedback_mentioned_emails(text),
            "recipients": recipients,
        }
    except HTTPException:
        await db.rollback()
        for path in saved_paths:
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
        raise
    except Exception:
        await db.rollback()
        for path in saved_paths:
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
        raise
    finally:
        await db.close()

    if notification_payload:
        notification_status = await _send_feedback_email_notification(notification_payload)
        logger.info(
            "feedback_created",
            extra={
                "event": "feedback_created",
                "feedback_id": message_id,
                "feedback_kind": "reply" if notification_payload.get("parent_id") else "post",
                "post_id": notification_payload.get("post_id"),
                "parent_id": notification_payload.get("parent_id"),
                "user": notification_payload.get("user_token"),
                "author": notification_payload.get("author"),
                "ip": request_ip(request),
                "image_count": notification_payload.get("image_count", 0),
                "body_length": len(text),
                "body_preview": _feedback_body_preview(text),
                "mentioned_emails": notification_payload.get("mentioned_emails", []),
                "recipients": notification_payload.get("recipients", []),
                "notification_status": (notification_status or {}).get("status", ""),
                "notification_transport": (notification_status or {}).get("transport", ""),
            },
        )

    response = _feedback_message_response(message, attachments_by_message)
    if notification_status:
        response["notification"] = notification_status
    return response


@app.delete("/api/feedback/{message_id}")
async def delete_feedback_message(request: Request, message_id: str):
    admin_user = require_admin(request)
    db = await get_db()
    deleted_ids: list[str] = []
    deleted_paths: list[str] = []
    root_parent_id: Optional[str] = None
    try:
        target = await row_to_dict(
            await (await db.execute(
                "SELECT id, parent_id FROM feedback_messages WHERE id=?",
                (message_id,),
            )).fetchone()
        )
        if not target:
            raise HTTPException(404, "帖子不存在")

        if target.get("parent_id"):
            deleted_ids = [target["id"]]
            root_parent_id = target["parent_id"]
            delete_scope = "reply"
        else:
            reply_rows = await (
                await db.execute(
                    "SELECT id FROM feedback_messages WHERE parent_id=?",
                    (target["id"],),
                )
            ).fetchall()
            deleted_ids = [target["id"]] + [row["id"] for row in reply_rows]
            delete_scope = "post"

        placeholders = ",".join("?" * len(deleted_ids))
        attachment_rows = await (
            await db.execute(
                f"SELECT stored_path FROM feedback_attachments WHERE message_id IN ({placeholders})",
                tuple(deleted_ids),
            )
        ).fetchall()
        deleted_paths = [row["stored_path"] for row in attachment_rows if row["stored_path"]]

        await db.execute(
            f"DELETE FROM feedback_attachments WHERE message_id IN ({placeholders})",
            tuple(deleted_ids),
        )
        await db.execute(
            f"DELETE FROM feedback_messages WHERE id IN ({placeholders})",
            tuple(deleted_ids),
        )
        if root_parent_id:
            await db.execute(
                "UPDATE feedback_messages SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (root_parent_id,),
            )

        await write_audit(
            db, request, "feedback.delete",
            resource_type="feedback", resource_id=target["id"],
            details={
                "admin_user": admin_user,
                "scope": delete_scope,
                "deleted_ids": deleted_ids,
                "root_parent_id": root_parent_id,
            },
        )
        await db.commit()
    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

    _remove_feedback_storage(deleted_paths, deleted_ids)
    return {"ok": True, "deleted": len(deleted_ids), "ids": deleted_ids}


@app.get("/api/feedback/images/{attachment_id}")
async def get_feedback_image(request: Request, attachment_id: str):
    if AUTH_ENABLED:
        current_user_token(request)
    db = await get_db()
    try:
        row = await row_to_dict(
            await (await db.execute(
                "SELECT filename, stored_path, content_type FROM feedback_attachments WHERE id=?",
                (attachment_id,),
            )).fetchone()
        )
    finally:
        await db.close()
    if not row:
        raise HTTPException(404)
    path = row["stored_path"]
    root = os.path.abspath(feedback_dir())
    full = os.path.abspath(path)
    if os.path.commonpath([root, full]) != root or not os.path.exists(full):
        raise HTTPException(404)
    return FileResponse(
        full,
        media_type=row.get("content_type") or "application/octet-stream",
        headers={
            "Content-Disposition": _content_disposition(
                _safe_download_name(row.get("filename"), os.path.basename(full)),
                disposition="inline",
            ),
        },
    )


# ── Routes: projects ──────────────────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects(request: Request):
    db = await get_db()
    access_sql, access_params = _project_access_clause(request)
    where_sql = f"WHERE {access_sql}" if access_sql else ""
    rows = await (await db.execute(f"""
        SELECT * FROM projects
        {where_sql}
        ORDER BY is_public DESC, created_at DESC
    """, access_params)).fetchall()
    await db.close()
    return [_project_response(dict(r), request) for r in rows]


@app.post("/api/projects", status_code=201)
async def create_project(request: Request, body: dict):
    pid = str(uuid.uuid4())
    db = await get_db()
    user_token = await ensure_user_row(db, request)
    is_public = 1 if body.get("is_public") else 0
    await db.execute(
        "INSERT INTO projects(id, user_token, name, description, is_public) VALUES(?,?,?,?,?)",
        (pid, user_token, body.get("name", "新项目"), body.get("description", ""), is_public),
    )
    await write_audit(
        db, request, "project.create",
        resource_type="project", resource_id=pid,
        details={"name": body.get("name", "新项目"), "is_public": is_public},
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await cursor.fetchone()
    await db.close()
    return _project_response(dict(row), request)


@app.put("/api/projects/{pid}")
async def update_project(request: Request, pid: str, body: dict):
    db = await get_db()

    row = await load_owned_project(db, request, pid)
    if not row:
        await db.close()
        raise HTTPException(404)

    next_name = body.get("name", row.get("name"))
    next_description = body.get("description", row.get("description", ""))
    next_is_public = 1 if body.get("is_public", row.get("is_public", 0)) else 0
    await db.execute(
        "UPDATE projects SET name=?, description=?, is_public=? WHERE id=?",
        (next_name, next_description, next_is_public, pid),
    )
    await write_audit(
        db, request, "project.update",
        resource_type="project", resource_id=pid,
        details={
            "old_name": row.get("name"),
            "new_name": next_name,
            "old_is_public": row.get("is_public", 0),
            "new_is_public": next_is_public,
        },
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    row = await cursor.fetchone()
    await db.close()
    return _project_response(dict(row), request)


@app.delete("/api/projects/{pid}", status_code=204)
async def delete_project(request: Request, pid: str):
    db = await get_db()

    row = await load_owned_project(db, request, pid)
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
                is_pinned,
                file_a_name, file_a_path, file_a_gzip_path, file_a_exists,
                file_b_name, file_b_path, file_b_gzip_path, file_b_exists,
                source_job_a, source_job_b, save_triton_csv, save_triton_code,
                status, console_out, error_msg, result_dir, deleted_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (job_dict["id"], job_dict.get("project_id"), job_dict.get("user_token"),
              job_dict.get("created_at"), job_dict.get("label", ""), job_dict.get("mode"),
              job_dict.get("is_pinned", 0),
              job_dict.get("file_a_name"), job_dict.get("file_a_path"), job_dict.get("file_a_gzip_path"), job_dict.get("file_a_exists", 1),
              job_dict.get("file_b_name"), job_dict.get("file_b_path"), job_dict.get("file_b_gzip_path"), job_dict.get("file_b_exists", 1),
              job_dict.get("source_job_a"), job_dict.get("source_job_b"),
              job_dict.get("save_triton_csv", 0), job_dict.get("save_triton_code", 0),
              job_dict.get("status", "pending"), job_dict.get("console_out", ""), job_dict.get("error_msg", ""), job_dict.get("result_dir", "")))

    # Delete jobs from main table
    await db.execute("DELETE FROM jobs WHERE project_id=?", (pid,))

    # Delete the project
    await db.execute("DELETE FROM projects WHERE id=?", (pid,))
    await write_audit(
        db, request, "project.delete",
        resource_type="project", resource_id=pid,
        details={"name": row.get("name"), "jobs": len(jobs_data)},
    )
    await db.commit()
    await db.close()


# ── Routes: deleted projects (recovery) ───────────────────────────────────────

@app.get("/api/deleted-projects")
async def list_deleted_projects(request: Request):
    """List recoverable projects deleted within the last 10 days."""
    db = await get_db()
    owner_sql, owner_params = _owner_clause(request)
    owner_filter = f"AND {owner_sql}" if owner_sql else ""
    rows = await (await db.execute(f"""
        SELECT * FROM deleted_projects
        WHERE deleted_at >= datetime('now', '-10 days')
          {owner_filter}
        ORDER BY deleted_at DESC
    """, owner_params)).fetchall()
    await db.close()
    return [dict(r) for r in rows]


@app.post("/api/deleted-projects/{pid}/restore", status_code=200)
async def restore_project(request: Request, pid: str):
    """Restore a project deleted within the last 10 days."""
    db = await get_db()

    # Get deleted project info
    owner_sql, owner_params = _owner_clause(request)
    row = await row_to_dict(
        await (
            await db.execute(
                f"SELECT * FROM deleted_projects WHERE id=? {'AND ' + owner_sql if owner_sql else ''}",
                (pid, *owner_params),
            )
        ).fetchone()
    )
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
                is_pinned,
                file_a_name, file_a_path, file_a_gzip_path, file_a_exists,
                file_b_name, file_b_path, file_b_gzip_path, file_b_exists,
                source_job_a, source_job_b, save_triton_csv, save_triton_code,
                status, console_out, error_msg, result_dir)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_dict["id"], pid, job_dict.get("user_token"),
              job_dict.get("created_at"), job_dict.get("label", ""), job_dict.get("mode"),
              job_dict.get("is_pinned", 0),
              job_dict.get("file_a_name"), job_dict.get("file_a_path"), job_dict.get("file_a_gzip_path"), job_dict.get("file_a_exists", 1),
              job_dict.get("file_b_name"), job_dict.get("file_b_path"), job_dict.get("file_b_gzip_path"), job_dict.get("file_b_exists", 1),
              job_dict.get("source_job_a"), job_dict.get("source_job_b"),
              job_dict.get("save_triton_csv", 0), job_dict.get("save_triton_code", 0),
              job_dict.get("status", "pending"), job_dict.get("console_out", ""), job_dict.get("error_msg", ""), job_dict.get("result_dir", "")))

    # Remove restored jobs from deleted_jobs
    await db.execute("DELETE FROM deleted_jobs WHERE project_id=?", (pid,))

    # Remove from deleted_projects
    await db.execute("DELETE FROM deleted_projects WHERE id=?", (pid,))
    await write_audit(
        db, request, "project.restore",
        resource_type="project", resource_id=pid,
        details={"name": row.get("name"), "jobs": len(deleted_jobs)},
    )

    await db.commit()
    cursor = await db.execute("SELECT * FROM projects WHERE id=?", (pid,))
    restored = await cursor.fetchone()
    await db.close()

    return dict(restored)


@app.delete("/api/deleted-projects/{pid}", status_code=204)
async def permanently_delete_project(request: Request, pid: str):
    """Permanently delete a project from recovery list (without restoring)."""
    db = await get_db()

    owner_sql, owner_params = _owner_clause(request)
    row = await row_to_dict(
        await (
            await db.execute(
                f"SELECT * FROM deleted_projects WHERE id=? {'AND ' + owner_sql if owner_sql else ''}",
                (pid, *owner_params),
            )
        ).fetchone()
    )
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
    await write_audit(
        db, request, "project.permanent_delete",
        resource_type="project", resource_id=pid,
        details={"name": row.get("name"), "jobs": len(job_ids)},
    )
    await db.commit()
    await db.close()


# ── Routes: jobs ──────────────────────────────────────────────────────────────

async def _with_file_exists(rows, request: Optional[Request] = None):
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
        params = list(src_ids)
        access_filter = ""
        if request is not None:
            access_sql, access_params = _job_access_clause(request)
            if access_sql:
                access_filter = f" AND {access_sql}"
                params.extend(access_params)
        cur = await db2.execute(
            f"SELECT id, file_a_path, file_a_gzip_path FROM jobs WHERE id IN ({placeholders}){access_filter}",
            tuple(params))
        async for src_row in cur:
            s = dict(src_row)
            src_paths[s["id"]] = s.get("file_a_gzip_path") or s.get("file_a_path")
        await db2.close()

    data = []
    for r in rows:
        d = _job_response(dict(r), request) if request is not None else dict(r)
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
    request: Request,
    project_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    clauses = []
    params = []
    access_sql, access_params = _job_access_clause(request)
    if access_sql:
        clauses.append(access_sql)
        params.extend(access_params)
    if project_id == "__none__":
        clauses.append("project_id IS NULL")
    elif project_id:
        await validate_project_access(db, request, project_id)
        clauses.append("project_id = ?")
        params.append(project_id)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    count_cursor = await db.execute(f"SELECT COUNT(*) as total FROM jobs {where_sql}", params)
    total = (await count_cursor.fetchone())[0]

    rows = await (await db.execute(f"""
            SELECT * FROM jobs
            {where_sql}
            ORDER BY COALESCE(is_pinned, 0) DESC, created_at DESC
            LIMIT ? OFFSET ?
        """, (*params, limit, offset))).fetchall()

    await db.close()

    data = await _with_file_exists(rows, request)
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


async def _load_jobs_by_ids(db, job_ids: list[str], request: Optional[Request] = None) -> list[dict]:
    placeholders = ",".join("?" * len(job_ids))
    params = list(job_ids)
    owner_filter = ""
    if request is not None:
        owner_sql, owner_params = _owner_clause(request)
        if owner_sql:
            owner_filter = f" AND {owner_sql}"
            params.extend(owner_params)
    rows = await (
        await db.execute(
            f"SELECT * FROM jobs WHERE id IN ({placeholders}){owner_filter}",
            tuple(params),
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
    request: Request,
    project_id: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    clauses = ["j.mode='single'", "j.status='done'"]
    params = []
    access_sql, access_params = _job_access_clause(request, "j")
    if access_sql:
        clauses.append(access_sql)
        params.extend(access_params)
    if project_id == "__none__":
        clauses.append("j.project_id IS NULL")
    elif project_id:
        await validate_project_access(db, request, project_id)
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
            ORDER BY COALESCE(j.is_pinned, 0) DESC, j.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        )
    ).fetchall()
    await db.close()

    data = await _with_file_exists(rows, request)
    return {"data": data, "total": total, "limit": limit, "offset": offset}


@app.patch("/api/jobs/bulk/project")
async def bulk_move_jobs(request: Request, body: dict):
    job_ids = _unique_job_ids(body)
    db = await get_db()
    try:
        await _load_jobs_by_ids(db, job_ids, request)
        await validate_project_access(db, request, body.get("project_id") or None)
        placeholders = ",".join("?" * len(job_ids))
        await db.execute(
            f"UPDATE jobs SET project_id=? WHERE id IN ({placeholders})",
            (body.get("project_id") or None, *job_ids),
        )
        await write_audit(
            db, request, "job.bulk_move",
            resource_type="job", resource_id=",".join(job_ids[:10]),
            details={"job_count": len(job_ids), "project_id": body.get("project_id") or None},
        )
        await db.commit()
    finally:
        await db.close()
    return {"updated": len(job_ids)}


@app.post("/api/jobs/bulk/delete")
async def bulk_delete_jobs(request: Request, body: dict):
    job_ids = _unique_job_ids(body)
    db = await get_db()
    try:
        await _load_jobs_by_ids(db, job_ids, request)
        for job_id in job_ids:
            _remove_job_dir(job_id)
        placeholders = ",".join("?" * len(job_ids))
        await db.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", tuple(job_ids))
        await write_audit(
            db, request, "job.bulk_delete",
            resource_type="job", resource_id=",".join(job_ids[:10]),
            details={"job_count": len(job_ids)},
        )
        await db.commit()
    finally:
        await db.close()
    return {"deleted": len(job_ids)}


@app.post("/api/jobs/bulk/delete-files")
async def bulk_delete_job_files(request: Request, body: dict):
    job_ids = _unique_job_ids(body)
    db = await get_db()
    try:
        jobs = await _load_jobs_by_ids(db, job_ids, request)
        files_deleted = 0
        for job in jobs:
            files_deleted += await _delete_trace_files(db, job)
            await _refresh_job_storage_cache(db, job["id"])
        await write_audit(
            db, request, "job.bulk_delete_files",
            resource_type="job", resource_id=",".join(job_ids[:10]),
            details={"job_count": len(job_ids), "files_deleted": files_deleted},
        )
        await db.commit()
    finally:
        await db.close()
    return {"updated": len(job_ids), "files_deleted": files_deleted}


@app.get("/api/storage/summary")
async def storage_summary(request: Request):
    db = await get_db()
    try:
        access_sql, access_params = _job_access_clause(request, "j")
        where_sql = f"WHERE {access_sql}" if access_sql else ""
        rows = await (
            await db.execute(
                f"""
                SELECT j.*, p.name AS project_name
                FROM jobs j
                LEFT JOIN projects p ON p.id = j.project_id
                {where_sql}
                ORDER BY j.created_at DESC
                """,
                access_params,
            )
        ).fetchall()
        access_source_sql, access_source_params = _job_access_clause(request)
        access_source_filter = f"WHERE {access_source_sql}" if access_source_sql else ""
        compare_counts_rows = await (
            await db.execute(
                f"""
                SELECT source_id, COUNT(*) AS count
                FROM (
                    SELECT source_job_a AS source_id FROM jobs {access_source_filter}
                    {'AND' if access_source_filter else 'WHERE'} source_job_a IS NOT NULL
                    UNION ALL
                    SELECT source_job_b AS source_id FROM jobs {access_source_filter}
                    {'AND' if access_source_filter else 'WHERE'} source_job_b IS NOT NULL
                )
                GROUP BY source_id
                """,
                (*access_source_params, *access_source_params),
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
    request: Request,
    project_id: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    clauses = []
    where_params = []
    access_sql, access_params = _job_access_clause(request, "j")
    if access_sql:
        clauses.append(access_sql)
        where_params.extend(access_params)
    if project_id == "__none__":
        clauses.append("j.project_id IS NULL")
    elif project_id:
        await validate_project_access(db, request, project_id)
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
    request: Request,
    group_id: str,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    clauses = []
    params = []
    access_sql, access_params = _job_access_clause(request, "j")
    if access_sql:
        clauses.append(access_sql)
        params.extend(access_params)
    if group_id == "__none__":
        clauses.append("j.project_id IS NULL")
    else:
        await validate_project_access(db, request, group_id)
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
            ORDER BY COALESCE(j.is_pinned, 0) DESC, j.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        )
    ).fetchall()
    await db.close()

    data = await _with_file_exists(rows, request)
    return {"data": data, "total": total, "limit": limit, "offset": offset}


@app.post("/api/jobs", status_code=201)
async def create_job(
    request: Request,
    file_a: UploadFile,
    file_b: Optional[UploadFile] = None,
    save_triton_csv: bool = Form(False),
    save_triton_code: bool = Form(False),
    label: str = Form(""),
    project_id: Optional[str] = Form(None),
):
    jid = str(uuid.uuid4())
    jdir = job_dir(jid)

    db = await get_db()
    try:
        user_token = await ensure_user_row(db, request)
        await validate_project_access(db, request, project_id or None)
        await db.commit()
    finally:
        await db.close()

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

    eff_label = label or (
        f"{file_a.filename} vs {name_b}" if mode == "compare" and name_b else file_a.filename
    ) or jid

    db = await get_db()
    await db.execute(
        """INSERT INTO jobs(id, project_id, user_token, label, mode,
               file_a_name, file_a_path, file_a_gzip_path, file_b_name, file_b_path, file_b_gzip_path,
               save_triton_csv, save_triton_code)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (jid, project_id or None, user_token, eff_label, mode,
         file_a.filename, path_a, gzip_path_a[0], name_b, path_b, gzip_path_b[0],
         int(save_triton_csv), int(save_triton_code)),
    )
    await _refresh_job_storage_cache(db, jid)
    await write_audit(
        db, request, "job.create",
        resource_type="job", resource_id=jid,
        details={
            "mode": mode,
            "project_id": project_id,
            "label": eff_label,
            "file_a_name": file_a.filename,
            "file_b_name": name_b,
        },
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await cursor.fetchone()
    await db.close()

    await enqueue_analysis_job(jid)
    return dict(row)


@app.post("/api/jobs/compare", status_code=201)
async def compare_jobs(request: Request, body: dict):
    job_id_a = body.get("job_id_a")
    job_id_b = body.get("job_id_b")
    if not job_id_a or not job_id_b:
        raise HTTPException(400, "job_id_a and job_id_b are required")

    db = await get_db()
    user_token = await ensure_user_row(db, request)

    src_a = await load_accessible_job(db, request, job_id_a)
    src_b = await load_accessible_job(db, request, job_id_b)

    if not src_a or not src_b:
        await db.close()
        raise HTTPException(404, "Source job not found")

    if not _trace_path_exists(src_a):
        await db.close()
        raise HTTPException(409, f"Source file A has been deleted")
    if not _trace_path_exists(src_b):
        await db.close()
        raise HTTPException(409, f"Source file B has been deleted")

    jid = str(uuid.uuid4())
    eff_label = body.get("label") or f"{src_a['label']} vs {src_b['label']}"
    await validate_project_access(db, request, body.get("project_id") or None)

    await db.execute(
        """INSERT INTO jobs(id, project_id, user_token, label, mode,
               file_a_name, file_b_name,
               source_job_a, source_job_b,
               save_triton_csv, save_triton_code)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (jid, body.get("project_id"), user_token, eff_label, "compare",
         src_a["file_a_name"], src_b["file_a_name"],
         job_id_a, job_id_b,
         0, 0),
    )
    await _refresh_job_storage_cache(db, jid)
    await write_audit(
        db, request, "job.compare_create",
        resource_type="job", resource_id=jid,
        details={
            "source_job_a": job_id_a,
            "source_job_b": job_id_b,
            "project_id": body.get("project_id"),
            "label": eff_label,
        },
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await cursor.fetchone()
    await db.close()

    await enqueue_analysis_job(jid)
    return dict(row)


@app.post("/api/jobs/batch-compare", status_code=201)
async def batch_compare_jobs(request: Request, body: dict):
    baseline_job_id = body.get("baseline_job_id")
    candidate_job_ids = []
    for job_id in body.get("candidate_job_ids") or []:
        if job_id and job_id not in candidate_job_ids:
            candidate_job_ids.append(job_id)

    if not baseline_job_id or not candidate_job_ids:
        raise HTTPException(400, "baseline_job_id and candidate_job_ids are required")
    if baseline_job_id in candidate_job_ids:
        raise HTTPException(400, "baseline job cannot also be a candidate")
    if len(candidate_job_ids) > MAX_BATCH_COMPARE_JOBS:
        raise HTTPException(400, f"Too many candidates; max is {MAX_BATCH_COMPARE_JOBS}")

    db = await get_db()
    user_token = await ensure_user_row(db, request)
    project_id = body.get("project_id") or None
    await validate_project_access(db, request, project_id)

    baseline = await load_accessible_job(db, request, baseline_job_id)
    if not baseline:
        await db.close()
        raise HTTPException(404, "Baseline job not found")
    if not _trace_path_exists(baseline):
        await db.close()
        raise HTTPException(409, "Baseline source file has been deleted")

    candidates = []
    for candidate_id in candidate_job_ids:
        src = await load_accessible_job(db, request, candidate_id)
        if not src:
            await db.close()
            raise HTTPException(404, f"Candidate job not found: {candidate_id}")
        if not _trace_path_exists(src):
            await db.close()
            raise HTTPException(409, f"Candidate source file has been deleted: {src.get('label') or candidate_id}")
        candidates.append(src)

    label_prefix = str(body.get("label_prefix") or "").strip()
    created = []
    for src_b in candidates:
        jid = str(uuid.uuid4())
        pair_label = f"{baseline['label']} vs {src_b['label']}"
        eff_label = f"{label_prefix} - {pair_label}" if label_prefix else pair_label
        await db.execute(
            """INSERT INTO jobs(id, project_id, user_token, label, mode,
                   file_a_name, file_b_name,
                   source_job_a, source_job_b,
                   save_triton_csv, save_triton_code)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (jid, project_id, user_token, eff_label, "compare",
             baseline["file_a_name"], src_b["file_a_name"],
             baseline_job_id, src_b["id"],
             0, 0),
        )
        await _refresh_job_storage_cache(db, jid)
        await write_audit(
            db, request, "job.batch_compare_create",
            resource_type="job", resource_id=jid,
            details={
                "baseline_job_id": baseline_job_id,
                "candidate_job_id": src_b["id"],
                "project_id": project_id,
                "label": eff_label,
                "batch_size": len(candidates),
            },
        )
        cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
        created.append(dict(await cursor.fetchone()))

    await db.commit()
    await db.close()

    for job in created:
        await enqueue_analysis_job(job["id"])
    return {"data": created, "count": len(created)}


def _primary_trace_path(row: dict, slot: str = "a") -> Optional[str]:
    return row.get(f"file_{slot}_gzip_path") or row.get(f"file_{slot}_path")


def _trace_path_exists(row: dict, slot: str = "a") -> bool:
    path = _primary_trace_path(row, slot)
    return bool(path and os.path.exists(path))


def _copy_trace_reference(src_path: str, dest_json: str) -> tuple[Optional[str], Optional[str]]:
    os.makedirs(os.path.dirname(dest_json), exist_ok=True)
    if src_path.lower().endswith((".gz", ".tgz")):
        dest_gzip = dest_json + ".gz"
        shutil.copyfile(src_path, dest_gzip)
        return None, dest_gzip
    shutil.copyfile(src_path, dest_json)
    return dest_json, None


@app.post("/api/jobs/{jid}/rerun-swapped", status_code=201)
async def rerun_compare_swapped(request: Request, jid: str):
    db = await get_db()
    try:
        user_token = await ensure_user_row(db, request)
        job = await load_accessible_job(db, request, jid)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.get("mode") != "compare":
            raise HTTPException(400, "Only compare jobs can be swapped")

        new_jid = str(uuid.uuid4())

        if job.get("source_job_a") and job.get("source_job_b"):
            source_a = await load_accessible_job(db, request, job["source_job_a"])
            source_b = await load_accessible_job(db, request, job["source_job_b"])
            if not source_a or not source_b:
                raise HTTPException(404, "Source job not found")

            path_a = _primary_trace_path(source_a)
            path_b = _primary_trace_path(source_b)
            if not path_a or not os.path.exists(path_a):
                raise HTTPException(409, "Source file A has been deleted")
            if not path_b or not os.path.exists(path_b):
                raise HTTPException(409, "Source file B has been deleted")

            label = f"{source_b.get('label') or source_b['id'][:8]} vs {source_a.get('label') or source_a['id'][:8]}"
            await db.execute(
                """INSERT INTO jobs(id, project_id, user_token, label, mode,
                       file_a_name, file_b_name,
                       source_job_a, source_job_b,
                       save_triton_csv, save_triton_code)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_jid, job.get("project_id"), user_token, label, "compare",
                    source_b.get("file_a_name"), source_a.get("file_a_name"),
                    source_b["id"], source_a["id"],
                    0, 0,
                ),
            )
        else:
            path_a = _primary_trace_path(job, "a")
            path_b = _primary_trace_path(job, "b")
            if not path_a or not os.path.exists(path_a):
                raise HTTPException(409, "File A has been deleted")
            if not path_b or not os.path.exists(path_b):
                raise HTTPException(409, "File B has been deleted")

            new_dir = job_dir(new_jid)
            try:
                new_a_path, new_a_gzip_path = _copy_trace_reference(path_b, os.path.join(new_dir, "trace_a.json"))
                new_b_path, new_b_gzip_path = _copy_trace_reference(path_a, os.path.join(new_dir, "trace_b.json"))
            except OSError as e:
                with contextlib.suppress(FileNotFoundError):
                    shutil.rmtree(new_dir)
                raise HTTPException(500, f"Failed to copy trace files: {e}") from e

            label_a = job.get("file_b_name") or os.path.basename(path_b)
            label_b = job.get("file_a_name") or os.path.basename(path_a)
            await db.execute(
                """INSERT INTO jobs(id, project_id, user_token, label, mode,
                       file_a_name, file_a_path, file_a_gzip_path,
                       file_b_name, file_b_path, file_b_gzip_path,
                       save_triton_csv, save_triton_code)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_jid, job.get("project_id"), user_token, f"{label_a} vs {label_b}", "compare",
                    label_a, new_a_path, new_a_gzip_path,
                    label_b, new_b_path, new_b_gzip_path,
                    0, 0,
                ),
            )

        await _refresh_job_storage_cache(db, new_jid)
        await write_audit(
            db, request, "job.compare_swap",
            resource_type="job", resource_id=new_jid,
            details={"source_compare_job": jid},
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (new_jid,))
        row = await cursor.fetchone()
    except HTTPException:
        await db.rollback()
        raise
    finally:
        await db.close()

    await enqueue_analysis_job(new_jid)
    return dict(row)


@app.post("/api/jobs/{jid}/share")
async def share_job(request: Request, jid: str):
    db = await get_db()
    changed = False
    created_project = False
    try:
        row = await load_accessible_job(db, request, jid)
        if row is None:
            raise HTTPException(404)

        job = _job_response(row, request)
        project_id = job.get("project_id")
        project_is_public = False
        if AUTH_ENABLED:
            if project_id:
                project = await load_accessible_project(db, request, project_id)
                if not project:
                    raise HTTPException(404, "Project not found")
                project_is_public = bool(project.get("is_public"))
                if not project_is_public:
                    if not job.get("is_owner"):
                        raise HTTPException(404)
                    await db.execute("UPDATE projects SET is_public=1 WHERE id=?", (project_id,))
                    project_is_public = True
                    changed = True
            else:
                if not job.get("is_owner"):
                    raise HTTPException(404)
                user_token = await ensure_user_row(db, request)
                project_id = str(uuid.uuid4())
                project_name = (job.get("label") or "共享任务")[:80]
                await db.execute(
                    "INSERT INTO projects(id, user_token, name, description, is_public) VALUES(?,?,?,?,1)",
                    (project_id, user_token, project_name, "由分享任务自动创建"),
                )
                await db.execute("UPDATE jobs SET project_id=? WHERE id=?", (project_id, jid))
                project_is_public = True
                changed = True
                created_project = True

        await write_audit(
            db, request, "job.share",
            resource_type="job", resource_id=jid,
            details={
                "project_id": project_id,
                "project_is_public": project_is_public,
                "changed": changed,
                "created_project": created_project,
            },
        )
        await db.commit()
    except HTTPException:
        await db.rollback()
        raise
    finally:
        await db.close()

    return {
        "url": _job_share_url(request, jid),
        "path": f"/#/job/{jid}",
        "project_id": project_id,
        "project_is_public": project_is_public,
        "changed": changed,
        "created_project": created_project,
    }


@app.get("/api/jobs/{jid}")
async def get_job(request: Request, jid: str):
    db = await get_db()
    row = await load_accessible_job(db, request, jid)
    await db.close()

    if row is None:
        raise HTTPException(404)

    job = _job_response(dict(row), request)
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
                src = await load_accessible_job(db2, request, src_jid, "file_a_path, file_a_gzip_path")
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
        job["ai_analysis"] = collect_ai_analysis(jid)
    if job["mode"] == "compare":
        job["compare_sources"] = await _compare_source_summaries(request, job)
    return job


@app.get("/api/jobs/{jid}/ai-analysis")
async def get_job_ai_analysis(request: Request, jid: str):
    db = await get_db()
    try:
        row = await load_accessible_job(db, request, jid, "id, status")
    finally:
        await db.close()
    if not row:
        raise HTTPException(404)

    payload = collect_ai_analysis(jid)
    payload["content"] = _read_ai_analysis_report(jid) if payload["report_exists"] else ""
    payload["artifacts"] = _collect_ai_analysis_artifacts(jid)
    return payload


@app.get("/api/jobs/{jid}/ai-analysis/report.md")
async def download_job_ai_analysis_report(request: Request, jid: str):
    db = await get_db()
    try:
        row = await load_accessible_job(db, request, jid, "id, label, status")
    finally:
        await db.close()
    if not row:
        raise HTTPException(404)

    content = _read_ai_analysis_report(jid)
    if not content:
        raise HTTPException(404, "AI analysis report not found")

    filename = _safe_download_name(row.get("label"), f"{jid}-ai-analysis")
    lower = filename.lower()
    for suffix in (".markdown", ".md"):
        if lower.endswith(suffix):
            filename = filename[:-len(suffix)]
            break
    filename = f"{filename}-ai-analysis.md"
    return PlainTextResponse(
        content.rstrip() + "\n",
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": _content_disposition(filename)},
    )


@app.post("/api/ai/diagnostics")
async def run_ai_diagnostics(request: Request):
    if not CLAUDE_ANALYSIS_ENABLED:
        raise HTTPException(403, "Claude analysis is disabled")
    if AUTH_ENABLED:
        current_user_token(request)
    return await _run_ai_diagnostics_payload()


@app.post("/api/jobs/{jid}/ai-analysis")
async def start_job_ai_analysis(request: Request, jid: str, body: Optional[dict] = None):
    if not CLAUDE_ANALYSIS_ENABLED:
        raise HTTPException(403, "Claude analysis is disabled")

    force = bool((body or {}).get("force"))
    db = await get_db()
    try:
        row = await load_accessible_job(db, request, jid)
        if not row:
            raise HTTPException(404)
        if row.get("status") != "done":
            raise HTTPException(409, "Job is not completed")

        current = collect_ai_analysis(jid)
        if current.get("status") == "running":
            raise HTTPException(409, "AI analysis is already running")
        if current.get("status") == "done" and not force:
            current["content"] = _read_ai_analysis_report(jid)
            current["artifacts"] = _collect_ai_analysis_artifacts(jid)
            return current

        _write_ai_analysis_status(jid, {
            "status": "running",
            "phase": "diagnosing",
            "progress": 5,
            "mode": row.get("mode"),
            "skill": CLAUDE_COMPARE_TRACE_SKILL if row.get("mode") == "compare" else CLAUDE_SINGLE_TRACE_SKILL,
            "started_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        })
        await write_audit(
            db, request, "job.ai_analysis_start",
            resource_type="job", resource_id=jid,
            details={"mode": row.get("mode"), "force": force},
        )
        await db.commit()
    except HTTPException:
        await db.rollback()
        raise
    finally:
        await db.close()

    task = asyncio.create_task(_run_ai_analysis_task(jid))
    _track_ai_analysis_task(jid, task)
    payload = collect_ai_analysis(jid)
    payload["content"] = _read_ai_analysis_report(jid) if payload["report_exists"] else ""
    payload["artifacts"] = _collect_ai_analysis_artifacts(jid)
    return JSONResponse(payload, status_code=202)


@app.get("/api/jobs/{jid}/report.md")
async def download_job_report(request: Request, jid: str):
    db = await get_db()
    row = await load_accessible_job(db, request, jid)
    project_name = ""
    if row and row.get("project_id"):
        cursor = await db.execute("SELECT name FROM projects WHERE id=?", (row["project_id"],))
        project = await cursor.fetchone()
        if project:
            project_name = project["name"]
    await db.close()

    if row is None:
        raise HTTPException(404)

    job = _job_response(dict(row), request)
    compare_sources = await _compare_source_summaries(request, job) if job.get("mode") == "compare" else {}
    lines = [
        f"# {job.get('label') or job['id']}",
        "",
        "## 任务信息",
        "",
        f"- 任务 ID: `{job['id']}`",
        f"- 类型: `{job.get('mode')}`",
        f"- 状态: `{job.get('status')}`",
        f"- 项目: {project_name or '未分组'}",
        f"- 创建时间: {job.get('created_at') or ''}",
        f"- 导出版本: {APP_VERSION}",
        "",
        "## Trace 文件",
        "",
        f"- A: {job.get('file_a_name') or '无'}",
    ]
    if job.get("file_b_name"):
        lines.append(f"- B: {job.get('file_b_name')}")
    if compare_sources:
        lines.extend(["", "## 对比来源", ""])
        for slot in ("a", "b"):
            source = compare_sources.get(slot)
            if source:
                lines.append(
                    f"- {slot.upper()}: {source.get('label') or source.get('id')} "
                    f"({source.get('project_name') or '未分组'}, {source.get('created_at') or ''})"
                )

    lines.extend([
        "",
        "## 控制台摘要",
        "",
        "```text",
        _console_excerpt(job.get("console_out", "")),
        "```",
    ])

    csv_sections = _report_csv_sections(jid, job.get("mode"))
    if csv_sections:
        lines.extend(["", "## 关键结果", "", *csv_sections])
    else:
        lines.extend(["", "## 关键结果", "", "_暂无可导出的结果表格。_"])

    filename = _safe_download_name(job.get("label"), f"{jid}.md")
    if not filename.lower().endswith(".md"):
        filename += ".md"
    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": _content_disposition(filename)},
    )


@app.get("/api/jobs/{jid}/results/{filename:path}")
async def get_job_result_table(
    request: Request,
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
        row = await load_accessible_job(db, request, jid, "id, status")
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
async def patch_job(request: Request, jid: str, body: dict):
    db = await get_db()

    row = await load_owned_job(db, request, jid)
    if not row:
        await db.close()
        raise HTTPException(404)

    if "project_id" in body:
        await validate_project_access(db, request, body["project_id"] or None)
    if "label" in body:
        await db.execute("UPDATE jobs SET label=? WHERE id=?", (body["label"], jid))
    if "project_id" in body:
        await db.execute("UPDATE jobs SET project_id=? WHERE id=?",
                         (body["project_id"] or None, jid))
    if "is_pinned" in body:
        await db.execute("UPDATE jobs SET is_pinned=? WHERE id=?", (1 if body["is_pinned"] else 0, jid))
    await write_audit(
        db, request, "job.update",
        resource_type="job", resource_id=jid,
        details={
            "old_label": row.get("label"),
            "new_label": body.get("label", row.get("label")),
            "old_project_id": row.get("project_id"),
            "new_project_id": body.get("project_id", row.get("project_id")),
            "old_is_pinned": row.get("is_pinned", 0),
            "new_is_pinned": 1 if body.get("is_pinned", row.get("is_pinned", 0)) else 0,
        },
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
    row = await cursor.fetchone()
    await db.close()
    return dict(row)


@app.delete("/api/jobs/{jid}", status_code=204)
async def delete_job(request: Request, jid: str):
    db = await get_db()

    row = await load_owned_job(db, request, jid)
    if not row:
        await db.close()
        raise HTTPException(404)


    _remove_job_dir(jid)

    await db.execute("DELETE FROM jobs WHERE id=?", (jid,))
    await write_audit(
        db, request, "job.delete",
        resource_type="job", resource_id=jid,
        details={"label": row.get("label"), "mode": row.get("mode")},
    )
    await db.commit()
    await db.close()


@app.post("/api/jobs/{jid}/run-triton")
async def run_job_triton(request: Request, jid: str):
    """Run triton code files and append local efficiency to CSV."""
    require_code_execution_enabled()
    db = await get_db()
    row = await load_owned_job(db, request, jid)
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
async def clear_inductor_cache(request: Request, jid: str):
    """Clear the torchinductor cache for a job's triton runs."""
    require_code_execution_enabled()
    import shutil, glob

    db = await get_db()
    row = await load_owned_job(db, request, jid)
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
async def run_single_triton(request: Request, jid: str, body: dict):
    """Run a single triton code file and return its efficiency."""
    require_code_execution_enabled()
    db = await get_db()
    row = await load_owned_job(db, request, jid)
    await db.close()

    if row is None:
        raise HTTPException(404)

    if row.get("status") != "done":
        raise HTTPException(400, "Job not completed")

    code_path_rel = body.get("code_path")
    if not code_path_rel:
        raise HTTPException(400, "code_path is required")

    rdir = result_dir(jid)
    code_path = _safe_child_path(rdir, code_path_rel, "Invalid code_path")

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

            command = [sys.executable, code_path]
            result = subprocess.run(
                command,
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
                    diagnostic = _format_triton_execution_error(result, command=command, code_path=code_path)
                    error_msg = f"缺少依赖模块 (ImportError): {mod_info}\n\n{diagnostic}"
                else:
                    error_msg = _format_triton_execution_error(result, command=command, code_path=code_path)
                full_error = f"{mlu_info}\n\n--- Execution Result ---\n{error_msg}" if mlu_info else error_msg
                return {"efficiency": None, "error": full_error}
            output = result.stdout.strip()
            if not output:
                error_msg = _format_triton_execution_error(result, command=command, code_path=code_path, no_output=True)
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
async def run_custom_triton(request: Request, jid: str, body: dict):
    """Run a custom triton code string and return its efficiency."""
    require_code_execution_enabled()
    db = await get_db()
    row = await load_owned_job(db, request, jid)
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
                command = [sys.executable, temp_path]
                result = subprocess.run(
                    command,
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
                    diagnostic = _format_triton_execution_error(result, command=command, code_path=temp_path)
                    error_msg = f"缺少依赖模块 (ImportError): {mod_info}\n\n{diagnostic}"
                else:
                    error_msg = _format_triton_execution_error(result, command=command, code_path=temp_path)
                full_error = f"{mlu_info}\n\n--- Execution Result ---\n{error_msg}" if mlu_info else error_msg
                return {"efficiency": None, "error": full_error}
            output = result.stdout.strip()
            if not output:
                error_msg = _format_triton_execution_error(result, command=command, code_path=temp_path, no_output=True)
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
async def get_job_file(request: Request, jid: str, slot: str, format: Optional[str] = None):
    """Serve trace file (a or b) for Perfetto/download.

    Query params:
      format=json  — decompress .gz content on the fly and serve as raw JSON
    """
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")

    db = await get_db()
    row = await load_accessible_job(db, request, jid)
    await db.close()

    if not row:
        raise HTTPException(404)

    if slot not in ("a", "b"):
        raise HTTPException(400, "slot must be 'a' or 'b'")

    # Prefer gzip path if available, fall back to json path
    gzip_path = row.get(f"file_{slot}_gzip_path")
    json_path = row.get(f"file_{slot}_path")
    file_path = gzip_path if gzip_path else json_path
    filename = row.get(f"file_{slot}_name")

    # Compare-from-history jobs have no file stored directly;
    # resolve via the corresponding source job (slot a → source_job_a, slot b → source_job_b)
    if not file_path:
        src_jid = row.get(f"source_job_{slot}")
        if src_jid:
            db2 = await get_db()
            src = await load_accessible_job(db2, request, src_jid, "file_a_name, file_a_path, file_a_gzip_path")
            await db2.close()
            if src:
                gzip_path = src.get("file_a_gzip_path")
                json_path = src.get("file_a_path")
                file_path = gzip_path if gzip_path else json_path
                filename = src.get("file_a_name")

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    audit_db = await get_db()
    try:
        await write_audit(
            audit_db, request, "job.file_download",
            resource_type="job", resource_id=jid,
            details={"slot": slot, "format": format or "gzip", "filename": filename},
        )
        await audit_db.commit()
    finally:
        await audit_db.close()

    # If the client requests raw JSON, return parseable JSON even when storage
    # keeps only a compressed copy after analysis.
    if format == "json":
        json_filename = _json_download_name(filename, slot)
        if file_path.lower().endswith((".gz", ".tgz", ".zip")):
            chunks = _iter_json_chunks(file_path)
            return StreamingResponse(
                chunks,
                media_type="application/json",
                headers={"Content-Disposition": _content_disposition(json_filename)},
            )
        return FileResponse(file_path, media_type="application/json", filename=json_filename)

    gzip_filename = _gzip_json_download_name(filename, slot)
    if file_path.lower().endswith(".gz") and not _is_tar_archive(file_path):
        return FileResponse(file_path, media_type="application/gzip", filename=gzip_filename)

    return StreamingResponse(
        _iter_gzip_encoded(_iter_json_chunks(file_path)),
        media_type="application/gzip",
        headers={"Content-Disposition": _content_disposition(gzip_filename)},
    )


@app.delete("/api/jobs/{jid}/files/{slot}", status_code=204)
async def delete_job_file(request: Request, jid: str, slot: str):
    """Delete the stored trace file (a or b) for a job."""
    db = await get_db()
    row = await load_owned_job(db, request, jid)
    if not row:
        await db.close()
        raise HTTPException(404)
    if slot not in ("a", "b"):
        await db.close()
        raise HTTPException(400, "slot must be 'a' or 'b'")

    await _delete_trace_files(db, row, slots=(slot,))
    await _refresh_job_storage_cache(db, jid)
    await write_audit(
        db, request, "job.file_delete",
        resource_type="job", resource_id=jid,
        details={"slot": slot, "label": row.get("label")},
    )
    await db.commit()
    await db.close()


@app.get("/api/jobs/{jid}/files/{slot}/delete-impact")
async def get_delete_file_impact(request: Request, jid: str, slot: str):
    if slot not in ("a", "b"):
        raise HTTPException(400, "slot must be 'a' or 'b'")

    db = await get_db()
    try:
        row = await load_owned_job(db, request, jid)
        if not row:
            raise HTTPException(404)
        dependents = await _compare_dependents(db, request, jid) if slot == "a" else []
    finally:
        await db.close()
    return {"dependent_compare_jobs": dependents, "count": len(dependents)}


@app.get("/api/jobs/{jid}/triton-code/{path:path}")
async def get_triton_code(request: Request, jid: str, path: str):
    """Serve triton code file for display in browser."""
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")

    db = await get_db()
    row = await load_accessible_job(db, request, jid)
    await db.close()

    if not row:
        raise HTTPException(404)

    full_path = _safe_child_path(result_dir(jid), path, "Invalid path")

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
