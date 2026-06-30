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
import math
import mimetypes
import os
import re
import secrets
import shlex
import shutil
import smtplib
import socket
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
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trace_analyzer import (  # noqa: E402
    compute_avgs,
    extract_kernel_family,
    filter_parsed_steps,
    fmt3,
    parse_step_filter,
    parse_trace,
    run_triton_code_and_get_efficiency,
)

from db import get_db, init_db, row_to_dict  # noqa: E402
import auth as ldap_auth  # noqa: E402

WEB_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(WEB_DIR)
PROJECT_CLAUDE_SKILLS_DIR = os.path.join(PROJECT_ROOT, ".claude", "skills")
DEFAULT_STORAGE_DIR = os.path.join(WEB_DIR, "storage")
STORAGE_DIR = os.environ.get("TRACE_STORAGE_DIR", DEFAULT_STORAGE_DIR)
APP_VERSION = "0.4.22"
INTERRUPTED_ANALYSIS_ERROR = "Server restarted before this analysis completed"
BACKUP_DIR = os.environ.get("TRACE_BACKUP_DIR", os.path.join(WEB_DIR, "backups"))
MAX_BATCH_COMPARE_JOBS = 50
PROJECT_METRIC_TARGET_KEYS = {
    "compute_ms",
    "e2e_ms",
    "comm_ms",
    "kernel_count",
    "aten_ops_ms",
    "aten_ops_count",
    "step_dur_ms",
}
FEEDBACK_DIRNAME = "feedback"
FEEDBACK_MAX_IMAGES = 4
FEEDBACK_MAX_IMAGE_BYTES = 8 * 1024 * 1024
EXPERIMENT_NODE_ATTACHMENTS_DIRNAME = "experiment_node_attachments"
EXPERIMENT_NODE_ATTACHMENTS_META_FILE = "attachments.json"
EXPERIMENT_NODE_ATTACHMENT_MAX_BYTES = int(os.environ.get("TRACE_EXPERIMENT_NODE_ATTACHMENT_MAX_BYTES", str(500 * 1024 * 1024)))
FEEDBACK_MENTION_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,62}$")
FEEDBACK_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_.%-])@([A-Za-z][A-Za-z0-9_.-]{0,62})(?=$|[^A-Za-z0-9_.-])")
FEEDBACK_REACTION_EMOJIS = {
    "👍", "👎", "😄", "🎉", "🚀", "❤️", "👀", "💡", "✅", "🙏",
    "🔥", "🤔", "😕", "👏", "🙌", "💯", "🧠", "🛠️", "📌", "❓",
}
AI_ANALYSIS_DIRNAME = "ai_analysis"
AI_ANALYSIS_STATUS_FILE = "ai_analysis_status.json"
AI_ANALYSIS_REPORT_FILE = "ai_analysis.md"
AI_ANALYSIS_VERSIONS_DIR = "versions"
AI_ANALYSIS_VERSION_META_FILE = "metadata.json"
AI_ANALYSIS_VERSION_REPORT_FILE = "report.md"
AI_ANALYSIS_USER_PROMPT_MAX_CHARS = 20000
AI_ANALYSIS_ARTIFACT_MAX_BYTES = 256 * 1024
AI_ANALYSIS_ARTIFACT_MAX_TOTAL_BYTES = 1024 * 1024
AI_ANALYSIS_CODE_PREVIEW_MAX_BYTES = int(os.environ.get("TRACE_AI_CODE_PREVIEW_MAX_BYTES", str(2 * 1024 * 1024)))
AI_ANALYSIS_ARTIFACT_EXTENSIONS = {
    ".csv", ".json", ".log", ".md", ".py", ".text", ".tsv", ".txt", ".yaml", ".yml",
}
AI_ANALYSIS_DOWNLOAD_ARTIFACT_EXTENSIONS = AI_ANALYSIS_ARTIFACT_EXTENSIONS | {".db"}
AI_ANALYSIS_INTERNAL_FILES = {
    AI_ANALYSIS_STATUS_FILE,
    AI_ANALYSIS_VERSION_META_FILE,
    "ai_diagnostics.json",
    "claude_tool_probe.txt",
    "command.json",
}
AI_ANALYSIS_VERSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
EXPERIMENT_NODE_ATTACHMENT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
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
UPLOAD_CONCURRENCY = max(1, int(os.environ.get("TRACE_UPLOAD_CONCURRENCY", "3")))
MAX_UPLOAD_BYTES = max(0, int(os.environ.get("TRACE_MAX_UPLOAD_BYTES", "0")))
MIN_STORAGE_FREE_BYTES = max(0, int(os.environ.get("TRACE_MIN_STORAGE_FREE_BYTES", "0")))
MAX_TRACE_JSON_BYTES = max(
    0,
    int(os.environ.get("TRACE_MAX_TRACE_JSON_BYTES", "0")),
)
STRICT_GZIP_SIZE_CHECK = os.environ.get("TRACE_STRICT_GZIP_SIZE_CHECK", "") == "1"
CLAUDE_ANALYSIS_ENABLED = os.environ.get("TRACE_ENABLE_CLAUDE_ANALYSIS", "") == "1"
AI_ANALYSIS_CONCURRENCY = max(1, int(os.environ.get("TRACE_AI_ANALYSIS_CONCURRENCY", "1")))
CLAUDE_ANALYSIS_COMMAND = os.environ.get("TRACE_CLAUDE_COMMAND", "claude")
CLAUDE_ANALYSIS_COMMAND_TEMPLATE = os.environ.get("TRACE_CLAUDE_COMMAND_TEMPLATE", "")
DEFAULT_CLAUDE_ANALYSIS_EXTRA_ARGS = "--dangerously-skip-permissions"
CLAUDE_ANALYSIS_EXTRA_ARGS = os.environ.get("TRACE_CLAUDE_EXTRA_ARGS", DEFAULT_CLAUDE_ANALYSIS_EXTRA_ARGS)
TRACE_CLAUDE_CUSTOM_HEADERS = os.environ.get("TRACE_CLAUDE_CUSTOM_HEADERS", "").strip()
CLAUDE_CUSTOM_HEADERS = (
    TRACE_CLAUDE_CUSTOM_HEADERS
    or os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").strip()
    or "x-project: torch_mlu"
)
CLAUDE_ANALYSIS_TIMEOUT_SECONDS = max(30, int(os.environ.get("TRACE_CLAUDE_TIMEOUT_SECONDS", "1800")))
CLAUDE_DIAGNOSTIC_TIMEOUT_SECONDS = max(5, int(os.environ.get("TRACE_CLAUDE_DIAGNOSTIC_TIMEOUT_SECONDS", "60")))
CLAUDE_ANALYSIS_SKILLS_DIR = os.environ.get("TRACE_CLAUDE_SKILLS_DIR", PROJECT_CLAUDE_SKILLS_DIR)
CLAUDE_SINGLE_TRACE_SKILL = os.environ.get("TRACE_CLAUDE_SINGLE_SKILL", "e2e-profiling-analyzer")
CLAUDE_COMPARE_TRACE_SKILL = os.environ.get("TRACE_CLAUDE_COMPARE_SKILL", "e2e-profiling-comparator")
CLAUDE_ANALYSIS_MODEL = (
    os.environ.get("TRACE_CLAUDE_MODEL")
    or os.environ.get("ANTHROPIC_MODEL")
    or "Claude Code default"
).strip()


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
USAGE_TIMEZONE_NAME = os.environ.get("TRACE_USAGE_TIMEZONE", LOG_TIMEZONE_NAME).strip()
analysis_queue: asyncio.Queue[str] = asyncio.Queue()
analysis_queue_loop: Optional[asyncio.AbstractEventLoop] = None
analysis_workers: list[asyncio.Task] = []
upload_semaphore = asyncio.Semaphore(UPLOAD_CONCURRENCY)
ai_analysis_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
ai_analysis_queue_loop: Optional[asyncio.AbstractEventLoop] = None
ai_analysis_workers: list[asyncio.Task] = []
ai_analysis_tasks: dict[str, asyncio.Task] = {}
ai_analysis_queued_jobs: set[str] = set()
ai_analysis_locks: dict[str, asyncio.Lock] = {}
ai_analysis_start_lock = asyncio.Lock()
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


USAGE_TZINFO = _resolve_log_timezone(USAGE_TIMEZONE_NAME)


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


def _favorite_user_token(request: Request) -> str:
    return current_user_token(request) or "local"


def _project_favorite_select(alias: str = "p") -> str:
    return (
        "CASE WHEN EXISTS ("
        "SELECT 1 FROM project_favorites pf "
        f"WHERE pf.project_id = {alias}.id AND pf.user_token = ?"
        ") THEN 1 ELSE 0 END AS is_favorite"
    )


def _project_view_clause(request: Request, view: Optional[str], alias: str = "p") -> tuple[str, list[str]]:
    view = (view or "all").strip().lower()
    if view not in {"all", "favorite", "mine", "shared"}:
        view = "all"
    prefix = f"{alias}." if alias else ""
    token = current_user_token(request)
    if view == "favorite":
        return (
            "EXISTS (SELECT 1 FROM project_favorites pf_view "
            f"WHERE pf_view.project_id = {prefix}id AND pf_view.user_token = ?)",
            [_favorite_user_token(request)],
        )
    if view == "mine":
        if not token:
            return "", []
        return f"{prefix}user_token = ?", [token]
    if view == "shared":
        if token:
            return f"({prefix}is_public = 1 AND COALESCE({prefix}user_token, '') <> ?)", [token]
        return f"{prefix}is_public = 1", []
    return "", []


def _project_response(row: dict, request: Request) -> dict:
    data = dict(row)
    data["is_public"] = 1 if data.get("is_public") else 0
    data["is_favorite"] = 1 if data.get("is_favorite") else 0
    metric_targets = _project_metric_targets(data)
    data["metric_targets"] = metric_targets
    data["compute_target_ms"] = metric_targets.get("compute_ms")
    token = current_user_token(request)
    data["is_owner"] = True if not token else data.get("user_token") == token
    data["visibility"] = "shared" if data["is_public"] else "personal"
    return data


def _coerce_project_metric_target(value):
    if value in (None, ""):
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _parse_project_metric_target(value, field_name: str = "target"):
    if value in (None, ""):
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"{field_name} must be a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise HTTPException(400, f"{field_name} must be a positive number")
    return round(number, 6)


def _parse_project_metric_targets(value) -> dict:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "metric_targets must be an object") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, "metric_targets must be an object")
    targets = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if key not in PROJECT_METRIC_TARGET_KEYS:
            raise HTTPException(400, f"Unsupported metric target: {key}")
        target = _parse_project_metric_target(raw_value, f"metric_targets.{key}")
        if target is not None:
            targets[key] = target
    return targets


def _project_metric_targets(row: dict) -> dict:
    targets = {}
    raw_json = row.get("metric_targets_json")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if key in PROJECT_METRIC_TARGET_KEYS:
                    target = _coerce_project_metric_target(value)
                    if target is not None:
                        targets[key] = target
    legacy_compute_target = _coerce_project_metric_target(row.get("compute_target_ms"))
    if legacy_compute_target is not None and "compute_ms" not in targets:
        targets["compute_ms"] = legacy_compute_target
    return targets


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


async def resolve_job_pk(db, handle: str) -> Optional[str]:
    """Resolve a public job handle to the canonical jobs.id UUID."""
    if handle is None:
        return None
    text = str(handle).strip()
    if not text:
        return None
    if text.isdigit():
        row = await (await db.execute("SELECT id FROM jobs WHERE seq=?", (int(text),))).fetchone()
        return row[0] if row else None
    return text


async def load_owned_job(db, request: Request, job_id: str, columns: str = "*") -> Optional[dict]:
    job_id = await resolve_job_pk(db, job_id)
    if not job_id:
        return None
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
    job_id = await resolve_job_pk(db, job_id)
    if not job_id:
        return None
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


def _usage_day(dt: Optional[datetime] = None) -> str:
    value = dt or datetime.now(USAGE_TZINFO)
    if value.tzinfo is None:
        value = value.replace(tzinfo=USAGE_TZINFO)
    return value.astimezone(USAGE_TZINFO).date().isoformat()


def _usage_sqlite_day_expr(column: str) -> str:
    # The deployment target uses Asia/Shanghai. Keep historical DB timestamps comparable
    # to the daily usage buckets even when SQLite stores CURRENT_TIMESTAMP in UTC.
    if USAGE_TIMEZONE_NAME in {"Asia/Shanghai", "Asia/Chongqing", "PRC", "CST"}:
        return f"date({column}, '+8 hours')"
    return f"date({column}, 'localtime')"


def _usage_identity(request: Request) -> Optional[tuple[str, str]]:
    user = current_user(request)
    token = (user.get("username") or "").strip()
    display = (user.get("display_name") or token).strip()
    if not token:
        token = (
            request.headers.get("X-Remote-User")
            or request.headers.get("X-Forwarded-User")
            or request.headers.get("X-User")
            or ""
        ).strip()
        display = token
    if not token and not AUTH_ENABLED:
        token = "local"
        display = "local"
    if not token:
        return None
    return token, display or token


def _should_record_usage(request: Request) -> bool:
    path = request.url.path
    if path.startswith("/static/"):
        return False
    if path in {"/healthz", "/readyz", "/metrics"}:
        return False
    if path.startswith("/api/admin/usage"):
        return False
    return True


async def record_usage_daily(request: Request, status_code: int):
    if not _should_record_usage(request):
        return
    identity = _usage_identity(request)
    if not identity:
        return
    token, display = identity
    day = _usage_day()
    now = datetime.now(timezone.utc).isoformat()
    path = request.url.path[:240]
    method = request.method[:16]
    db = await get_db()
    try:
        await db.execute(
            """
            INSERT INTO usage_daily(day, user_token, display_name, request_count,
                                    last_path, last_method, last_status, last_seen_at)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(day, user_token) DO UPDATE SET
                display_name=excluded.display_name,
                request_count=usage_daily.request_count + 1,
                last_path=excluded.last_path,
                last_method=excluded.last_method,
                last_status=excluded.last_status,
                last_seen_at=excluded.last_seen_at
            """,
            (day, token, display, 1, path, method, int(status_code or 0), now),
        )
        await db.commit()
    finally:
        await db.close()


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
    await recover_interrupted_jobs()
    mark_interrupted_ai_analysis()
    await refresh_storage_cache_for_all_jobs()
    await enqueue_pending_jobs()
    start_analysis_workers()
    start_ai_analysis_workers()
    try:
        yield
    finally:
        await stop_ai_analysis_workers()
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
        try:
            await record_usage_daily(request, status_code)
        except Exception as e:
            logger.warning(
                "usage_record_failed",
                extra={
                    "event": "usage_record_failed",
                    "request_id": request_id,
                    "user": request_user(request),
                    "path": request.url.path,
                    "error": str(e),
                },
            )


app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=SESSION_COOKIE_SECURE,
)


async def recover_interrupted_jobs():
    """Requeue analysis jobs interrupted by a previous process restart."""
    db = await get_db()
    try:
        cursor = await db.execute("""
            UPDATE jobs
            SET status='pending',
                error_msg=''
            WHERE status='running'
               OR (status='error' AND error_msg=?)
        """, (INTERRUPTED_ANALYSIS_ERROR,))
        if cursor.rowcount:
            logger.info(
                "analysis_jobs_recovered",
                extra={"event": "analysis_jobs_recovered", "count": cursor.rowcount},
            )
        await db.commit()
    finally:
        await db.close()


def mark_interrupted_ai_analysis():
    """Fail AI analysis status files left in queued/running after a restart."""
    if not os.path.isdir(STORAGE_DIR):
        return
    for jid in os.listdir(STORAGE_DIR):
        status_path = ai_analysis_status_path(jid)
        if not os.path.exists(status_path):
            continue
        status = _read_ai_analysis_status(jid)
        if status.get("status") not in {"queued", "running"}:
            continue
        _write_ai_analysis_status(jid, {
            **status,
            "status": "error",
            "progress": 100,
            "error": "Server restarted before this AI analysis completed",
            "finished_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        })


def _get_analysis_queue() -> asyncio.Queue[str]:
    global analysis_queue, analysis_queue_loop
    loop = asyncio.get_running_loop()
    if analysis_queue_loop is not loop:
        analysis_queue = asyncio.Queue()
        analysis_queue_loop = loop
        analysis_workers.clear()
    return analysis_queue


def _get_ai_analysis_queue() -> asyncio.Queue[tuple[str, dict]]:
    global ai_analysis_queue, ai_analysis_queue_loop
    loop = asyncio.get_running_loop()
    if ai_analysis_queue_loop is not loop:
        ai_analysis_queue = asyncio.Queue()
        ai_analysis_queue_loop = loop
        ai_analysis_workers.clear()
        ai_analysis_tasks.clear()
        ai_analysis_queued_jobs.clear()
        ai_analysis_locks.clear()
    return ai_analysis_queue


async def enqueue_pending_jobs():
    queue = _get_analysis_queue()
    while not queue.empty():
        try:
            queue.get_nowait()
            queue.task_done()
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
        await queue.put(row["id"])


def start_analysis_workers():
    queue = _get_analysis_queue()
    for index in range(ANALYSIS_CONCURRENCY):
        analysis_workers.append(asyncio.create_task(_analysis_worker(index, queue)))


def start_ai_analysis_workers():
    if not CLAUDE_ANALYSIS_ENABLED:
        return
    queue = _get_ai_analysis_queue()
    ai_analysis_workers[:] = [task for task in ai_analysis_workers if not task.done()]
    for index in range(len(ai_analysis_workers), AI_ANALYSIS_CONCURRENCY):
        ai_analysis_workers.append(asyncio.create_task(_ai_analysis_worker(index, queue)))


async def stop_analysis_workers():
    global analysis_queue_loop
    for task in analysis_workers:
        task.cancel()
    if analysis_workers:
        await asyncio.gather(*analysis_workers, return_exceptions=True)
    analysis_workers.clear()
    analysis_queue_loop = None


async def stop_ai_analysis_workers():
    global ai_analysis_queue_loop
    for task in ai_analysis_workers:
        task.cancel()
    if ai_analysis_workers:
        await asyncio.gather(*ai_analysis_workers, return_exceptions=True)
    ai_analysis_workers.clear()
    ai_analysis_tasks.clear()
    ai_analysis_queued_jobs.clear()
    while not ai_analysis_queue.empty():
        try:
            ai_analysis_queue.get_nowait()
            ai_analysis_queue.task_done()
        except asyncio.QueueEmpty:
            break
    ai_analysis_queue_loop = None


async def enqueue_analysis_job(job_id: str):
    await _get_analysis_queue().put(job_id)


async def enqueue_ai_analysis_job(job_id: str, context: Optional[dict] = None):
    queue = _get_ai_analysis_queue()
    ai_analysis_queued_jobs.add(job_id)
    await queue.put((job_id, dict(context or {})))


async def _analysis_worker(index: int, queue: asyncio.Queue[str]):
    while True:
        job_id = await queue.get()
        try:
            await run_analysis(job_id)
        finally:
            queue.task_done()


async def _ai_analysis_worker(index: int, queue: asyncio.Queue[tuple[str, dict]]):
    while True:
        job_id, context = await queue.get()
        ai_analysis_queued_jobs.discard(job_id)
        task = asyncio.current_task()
        lock = ai_analysis_locks.setdefault(job_id, asyncio.Lock())
        try:
            async with lock:
                if task is not None:
                    ai_analysis_tasks[job_id] = task
                try:
                    await _run_ai_analysis_task(job_id, context)
                finally:
                    if task is not None and ai_analysis_tasks.get(job_id) is task:
                        ai_analysis_tasks.pop(job_id, None)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(
                "ai_analysis_worker_failed",
                extra={
                    "event": "ai_analysis_worker_failed",
                    "job_id": job_id,
                    "worker": index,
                    "error": str(e),
                },
            )
        finally:
            queue.task_done()


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


def ai_analysis_versions_dir(job_id: str) -> str:
    return os.path.join(ai_analysis_dir(job_id), AI_ANALYSIS_VERSIONS_DIR)


def ai_analysis_version_dir(job_id: str, version_id: str) -> str:
    return os.path.join(ai_analysis_versions_dir(job_id), version_id)


def ai_analysis_version_report_path(job_id: str, version_id: str) -> str:
    return os.path.join(ai_analysis_version_dir(job_id, version_id), AI_ANALYSIS_VERSION_REPORT_FILE)


def ai_analysis_version_meta_path(job_id: str, version_id: str) -> str:
    return os.path.join(ai_analysis_version_dir(job_id, version_id), AI_ANALYSIS_VERSION_META_FILE)


def feedback_dir() -> str:
    return os.path.join(STORAGE_DIR, FEEDBACK_DIRNAME)


def feedback_message_dir(message_id: str) -> str:
    return os.path.join(feedback_dir(), message_id)


def experiment_node_attachments_dir(job_id: str) -> str:
    return os.path.join(job_dir(job_id), EXPERIMENT_NODE_ATTACHMENTS_DIRNAME)


def experiment_node_attachments_meta_path(job_id: str) -> str:
    return os.path.join(experiment_node_attachments_dir(job_id), EXPERIMENT_NODE_ATTACHMENTS_META_FILE)


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
    written = 0
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(1 << 20):  # 1 MB chunks
            written += len(chunk)
            if MAX_UPLOAD_BYTES and written > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"上传文件超过限制: {MAX_UPLOAD_BYTES} bytes")
            await f.write(chunk)


def _storage_disk_path() -> str:
    if os.path.exists(STORAGE_DIR):
        return STORAGE_DIR
    parent = os.path.dirname(STORAGE_DIR) or "."
    return parent if os.path.exists(parent) else "."


def _ensure_storage_capacity():
    if not MIN_STORAGE_FREE_BYTES:
        return
    disk = shutil.disk_usage(_storage_disk_path())
    if disk.free < MIN_STORAGE_FREE_BYTES:
        raise HTTPException(
            507,
            f"存储空间不足，当前可用 {disk.free} bytes，低于保留阈值 {MIN_STORAGE_FREE_BYTES} bytes",
        )


def _format_size(num_bytes: int) -> str:
    value = float(max(0, num_bytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


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
            if not is_tar_archive:
                return
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
    raw_name = os.path.basename(str(filename or "download")).replace("\\", "_")
    raw_name = re.sub(r"[\r\n\x00-\x1f\x7f]+", "_", raw_name).replace('"', "").strip()
    raw_name = raw_name or "download"
    fallback = raw_name.encode("ascii", "ignore").decode("ascii")
    fallback = re.sub(r"[^A-Za-z0-9._ -]+", "_", fallback).strip(" .") or "download"
    if len(fallback) > 180:
        base, ext = os.path.splitext(fallback)
        fallback = (base[: max(1, 180 - len(ext))].rstrip(" .") or "download") + ext
    return f'{disposition}; filename="{fallback}"; filename*=UTF-8\'\'{quote(raw_name)}'


def _app_base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host")
    proto = request.headers.get("X-Forwarded-Proto") or request.url.scheme
    if host:
        return f"{proto}://{host}"
    return str(request.base_url).rstrip("/")


def _job_share_url(request: Request, jid: str) -> str:
    return f"{_app_base_url(request)}/#/job/{jid}"


def _job_public_handle(job: dict) -> str:
    seq = (job or {}).get("seq")
    return str(seq) if seq is not None else str((job or {}).get("id") or "")


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
        "non_triton_kernel_efficiency_avg.csv": "非 Triton Kernel 效率",
        "aten_ops_avg.csv": "Aten Ops",
        "aten_ops_cmp.csv": "Aten Ops 对比",
        "tf_ops_avg.csv": "TensorFlow Ops",
        "tf_ops_cmp.csv": "TensorFlow Ops 对比",
        "cncl_ops_avg.csv": "CNCL Ops",
        "cncl_ops_cmp.csv": "CNCL Ops 对比",
    }
    preferred = (
        [
            "kernel_types_cmp.csv",
            "all_kernels_cmp.csv",
            "triton_kernels_cmp.csv",
            "aten_ops_cmp.csv",
            "tf_ops_cmp.csv",
            "cncl_ops_cmp.csv",
        ]
        if mode == "compare"
        else [
            "kernel_types_avg.csv",
            "all_kernels_avg.csv",
            "triton_kernels_avg.csv",
            "non_triton_kernel_efficiency_avg.csv",
            "aten_ops_avg.csv",
            "tf_ops_avg.csv",
            "cncl_ops_avg.csv",
        ]
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
        filename = os.path.basename(path)
        fields = _result_csv_fields(filename, reader.fieldnames or [])
        rows = [_result_csv_row(filename, row) for row in reader]
        rows = _aggregate_kernel_type_rows(filename, fields, rows)
        return {"fields": fields, "rows": rows}


def _prune_empty_tensorflow_result_csvs(files: dict) -> dict:
    """Hide legacy empty PyTorch-only CSVs from TensorFlow trace results."""
    has_tf_rows = any(
        files.get(name, {}).get("rows")
        for name in ("tf_ops_avg.csv", "tf_ops_cmp.csv")
    )
    if not has_tf_rows:
        return files
    for name in (
        "triton_kernels_avg.csv",
        "triton_kernels_cmp.csv",
        "non_triton_kernel_efficiency_avg.csv",
        "aten_ops_avg.csv",
        "aten_ops_cmp.csv",
        "cncl_ops_avg.csv",
        "cncl_ops_cmp.csv",
    ):
        if name in files and not files[name].get("rows"):
            files.pop(name, None)
    return files


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
    fields = list(fields)
    if filename == "all_kernels_cmp.csv" and "family" not in fields and "kernel_name" in fields:
        idx = fields.index("kernel_name") + 1
        fields = fields[:idx] + ["family"] + fields[idx:]
    if (
        filename == "kernel_types_cmp.csv"
        and "delta_count" not in fields
        and "avg_count_A" in fields
        and "avg_count_B" in fields
    ):
        idx = fields.index("avg_count_B") + 1
        fields = fields[:idx] + ["delta_count"] + fields[idx:]
    return fields


def _normalize_result_category_label(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    return "triton_other" if text == "triton" else text


def _augment_result_csv_row(filename: str, row: dict) -> dict:
    if filename == "all_kernels_cmp.csv" and "family" not in row and row.get("kernel_name"):
        row = dict(row)
        row["family"] = extract_kernel_family(row["kernel_name"])
    if "family" in row:
        row = dict(row)
        row["family"] = _normalize_result_category_label(row.get("family"))
    if filename in {"kernel_types_avg.csv", "kernel_types_cmp.csv"} and "type" in row:
        row = dict(row)
        row["type"] = _normalize_result_category_label(row.get("type"))
    return row


def _is_kernel_type_result_csv(filename: str) -> bool:
    return filename in {"kernel_types_avg.csv", "kernel_types_cmp.csv"}


def _sum_csv_number(rows: list[dict], field: str) -> float:
    total = 0.0
    for row in rows:
        value = _float_or_none(row.get(field))
        if value is not None:
            total += value
    return total


def _aggregate_kernel_type_rows(filename: str, fields: list[str], rows: list[dict]) -> list[dict]:
    if not _is_kernel_type_result_csv(filename) or "type" not in fields:
        return rows

    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        key = _normalize_result_category_label(row.get("type"))
        if not key:
            continue
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append({**row, "type": key})

    merged = []
    for key in order:
        bucket = grouped[key]
        row = {field: bucket[0].get(field, "") for field in fields}
        row["type"] = key
        if filename == "kernel_types_avg.csv":
            for field in ("avg_count", "avg_dur_ms"):
                if field in fields:
                    row[field] = fmt3(_sum_csv_number(bucket, field))
        else:
            for field in ("avg_dur_ms_A", "avg_dur_ms_B", "avg_count_A", "avg_count_B"):
                if field in fields:
                    row[field] = fmt3(_sum_csv_number(bucket, field))
            dur_a = _sum_csv_number(bucket, "avg_dur_ms_A")
            dur_b = _sum_csv_number(bucket, "avg_dur_ms_B")
            count_a = _sum_csv_number(bucket, "avg_count_A")
            count_b = _sum_csv_number(bucket, "avg_count_B")
            if "delta_dur_ms" in fields:
                row["delta_dur_ms"] = fmt3(dur_b - dur_a)
            if "delta_count" in fields:
                row["delta_count"] = fmt3(count_b - count_a)
        merged.append(row)

    return merged


def _result_csv_fields(filename: str, fields: list[str]) -> list[str]:
    return _filter_result_csv_fields(_augment_result_csv_fields(filename, fields))


def _result_csv_row(filename: str, row: dict) -> dict:
    return _filter_result_csv_row(_augment_result_csv_row(filename, row))


def _ordered_result_csv_names(rdir: str) -> list[str]:
    names = []
    for name in ["all_kernels_avg.csv", "all_kernels_cmp.csv",
                 "triton_kernels_avg.csv", "triton_kernels_cmp.csv",
                 "non_triton_kernel_efficiency_avg.csv",
                 "aten_ops_avg.csv", "aten_ops_cmp.csv",
                 "tf_ops_avg.csv", "tf_ops_cmp.csv",
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


def _normalize_ai_user_prompt(value) -> str:
    text = str(value or "").strip()
    if len(text) > AI_ANALYSIS_USER_PROMPT_MAX_CHARS:
        text = text[:AI_ANALYSIS_USER_PROMPT_MAX_CHARS].rstrip()
    return text


def _ai_backend_model_name() -> str:
    return CLAUDE_ANALYSIS_MODEL or "Claude Code default"


def _new_ai_analysis_version_id(generated_at: str = "") -> str:
    dt = _parse_iso_datetime(generated_at) if generated_at else None
    stamp = (dt or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def _safe_ai_analysis_version_id(version_id: str) -> str:
    value = (version_id or "").strip()
    if not value or not AI_ANALYSIS_VERSION_ID_RE.match(value):
        return ""
    return value


def _write_ai_version_metadata(path: str, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _ai_version_report_path_from_metadata(jid: str, metadata: dict) -> str:
    analysis_root = os.path.abspath(ai_analysis_dir(jid))
    rel_path = metadata.get("report_path") or ""
    if not rel_path and metadata.get("id") == "current":
        return ai_analysis_report_path(jid)
    if not rel_path and metadata.get("id"):
        rel_path = os.path.join(
            AI_ANALYSIS_VERSIONS_DIR,
            metadata["id"],
            AI_ANALYSIS_VERSION_REPORT_FILE,
        )
    full = os.path.abspath(os.path.join(analysis_root, rel_path))
    try:
        if os.path.commonpath([analysis_root, full]) != analysis_root:
            return ""
    except ValueError:
        return ""
    return full


def _save_ai_analysis_version(
    jid: str,
    content: str,
    status: dict,
    job: Optional[dict] = None,
    notification_context: Optional[dict] = None,
) -> dict:
    normalized_content = _finalize_ai_report_markdown(jid, content or "")
    if not normalized_content.strip():
        return {}

    context = notification_context or {}
    job = job or {}
    generated_at = status.get("finished_at") or status.get("updated_at") or _utc_now_iso()
    for _ in range(5):
        version_id = _new_ai_analysis_version_id(generated_at)
        version_dir = ai_analysis_version_dir(jid, version_id)
        try:
            os.makedirs(version_dir, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        version_id = f"{_new_ai_analysis_version_id(generated_at)}-{secrets.token_hex(2)}"
        version_dir = ai_analysis_version_dir(jid, version_id)
        os.makedirs(version_dir, exist_ok=False)

    report_path = ai_analysis_version_report_path(jid, version_id)
    _write_ai_report(report_path, normalized_content)
    timing = _ai_analysis_timing(status)
    metadata = {
        "id": version_id,
        "job_id": jid,
        "status": status.get("status", ""),
        "error": status.get("error", ""),
        "mode": status.get("mode") or job.get("mode") or "",
        "skill": status.get("skill", ""),
        "model": status.get("model") or _ai_backend_model_name(),
        "user_prompt": status.get("user_prompt") or context.get("user_prompt") or "",
        "generated_at": generated_at,
        "started_at": status.get("started_at", ""),
        "finished_at": status.get("finished_at", generated_at),
        "updated_at": status.get("updated_at", generated_at),
        "duration_ms": timing.get("duration_ms") or timing.get("elapsed_ms") or status.get("duration_ms"),
        "trigger_user_token": status.get("trigger_user_token") or context.get("trigger_user_token") or "",
        "trigger_user_display": context.get("trigger_user_display") or status.get("trigger_user_display") or "",
        "trigger_user_email": context.get("trigger_user_email") or status.get("trigger_user_email") or "",
        "owner_user_token": status.get("owner_user_token") or context.get("owner_user_token") or job.get("user_token") or "",
        "report_path": os.path.join(
            AI_ANALYSIS_VERSIONS_DIR,
            version_id,
            AI_ANALYSIS_VERSION_REPORT_FILE,
        ),
        "report_exists": True,
    }
    _write_ai_version_metadata(ai_analysis_version_meta_path(jid, version_id), metadata)
    logger.info(
        "ai_analysis_version_saved",
        extra={
            "event": "ai_analysis_version_saved",
            "job_id": jid,
            "version_id": version_id,
            "status": metadata["status"],
            "model": metadata["model"],
        },
    )
    return metadata


def _legacy_ai_analysis_version(jid: str, status: dict) -> dict:
    report_path = ai_analysis_report_path(jid)
    generated_at = (
        status.get("finished_at")
        or status.get("updated_at")
        or datetime.fromtimestamp(os.path.getmtime(report_path), timezone.utc).isoformat()
    )
    timing = _ai_analysis_timing({
        "started_at": status.get("started_at", ""),
        "finished_at": status.get("finished_at") or generated_at,
    })
    return {
        "id": "current",
        "job_id": jid,
        "status": status.get("status") or "done",
        "error": status.get("error", ""),
        "mode": status.get("mode", ""),
        "skill": status.get("skill", ""),
        "model": status.get("model") or "未知",
        "user_prompt": status.get("user_prompt", ""),
        "generated_at": generated_at,
        "started_at": status.get("started_at", ""),
        "finished_at": status.get("finished_at") or generated_at,
        "updated_at": status.get("updated_at") or generated_at,
        "duration_ms": timing.get("duration_ms") or timing.get("elapsed_ms"),
        "trigger_user_token": status.get("trigger_user_token", ""),
        "trigger_user_display": status.get("trigger_user_display", ""),
        "trigger_user_email": status.get("trigger_user_email", ""),
        "owner_user_token": status.get("owner_user_token", ""),
        "report_path": AI_ANALYSIS_REPORT_FILE,
        "report_exists": True,
        "legacy": True,
    }


def _collect_ai_analysis_versions(jid: str, status: Optional[dict] = None) -> list[dict]:
    status = status or _read_ai_analysis_status(jid)
    versions: list[dict] = []
    versions_root = ai_analysis_versions_dir(jid)
    if os.path.isdir(versions_root):
        for version_id in sorted(os.listdir(versions_root)):
            if not _safe_ai_analysis_version_id(version_id):
                continue
            meta_path = ai_analysis_version_meta_path(jid, version_id)
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            metadata["id"] = version_id
            report_path = _ai_version_report_path_from_metadata(jid, metadata)
            metadata["report_exists"] = bool(report_path and os.path.exists(report_path))
            metadata.setdefault("model", "未知")
            metadata.setdefault("generated_at", metadata.get("finished_at") or metadata.get("updated_at") or "")
            with contextlib.suppress(OSError):
                metadata["_sort_mtime"] = os.path.getmtime(meta_path)
            versions.append(metadata)

    if not versions and os.path.exists(ai_analysis_report_path(jid)):
        versions.append(_legacy_ai_analysis_version(jid, status))

    def sort_key(item: dict) -> tuple[datetime, float, str]:
        dt = _parse_iso_datetime(item.get("generated_at") or item.get("finished_at") or "")
        return (dt or datetime.fromtimestamp(0, timezone.utc), float(item.get("_sort_mtime") or 0), item.get("id", ""))

    versions.sort(key=sort_key, reverse=True)
    for item in versions:
        item.pop("_sort_mtime", None)
    return versions


def _select_ai_analysis_version(jid: str, version_id: str = "", status: Optional[dict] = None) -> Optional[dict]:
    versions = _collect_ai_analysis_versions(jid, status)
    if not versions:
        return None
    if not version_id:
        return versions[0]
    safe_id = _safe_ai_analysis_version_id(version_id)
    if not safe_id:
        return None
    return next((item for item in versions if item.get("id") == safe_id), None)


def _ensure_current_ai_report_versioned(jid: str, status: Optional[dict] = None) -> None:
    versions_root = ai_analysis_versions_dir(jid)
    if os.path.isdir(versions_root):
        for version_id in os.listdir(versions_root):
            if _safe_ai_analysis_version_id(version_id) and os.path.isfile(ai_analysis_version_meta_path(jid, version_id)):
                return
    content = _read_text_file(ai_analysis_report_path(jid))
    if not content.strip():
        return
    status = status or _read_ai_analysis_status(jid) or {"status": "done"}
    _save_ai_analysis_version(jid, content, status)


def _find_latest_ai_report(analysis_dir: str) -> Optional[str]:
    candidates = []
    for root, dirs, files in os.walk(analysis_dir):
        dirs[:] = [
            name for name in dirs
            if not name.startswith(".") and name != AI_ANALYSIS_VERSIONS_DIR
        ]
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
        dirs[:] = [
            name for name in dirs
            if not name.startswith(".") and not (root == root_abs and name == AI_ANALYSIS_VERSIONS_DIR)
        ]
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


def _safe_ai_analysis_artifact_path(jid: str, rel_path: str) -> tuple[str, str]:
    raw = (rel_path or "").strip()
    if not raw or os.path.isabs(raw):
        raise HTTPException(400, "Invalid AI analysis artifact path")
    normalized = raw.replace("\\", "/")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} or part.startswith(".") for part in parts):
        raise HTTPException(400, "Invalid AI analysis artifact path")
    if parts[0] == AI_ANALYSIS_VERSIONS_DIR:
        raise HTTPException(400, "Invalid AI analysis artifact path")

    filename = parts[-1]
    if filename.endswith(".tmp") or filename in AI_ANALYSIS_INTERNAL_FILES:
        raise HTTPException(400, "Invalid AI analysis artifact path")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in AI_ANALYSIS_DOWNLOAD_ARTIFACT_EXTENSIONS:
        raise HTTPException(400, "Unsupported AI analysis artifact type")

    root = ai_analysis_dir(jid)
    full = _safe_child_path(root, normalized, "Invalid AI analysis artifact path")
    if not os.path.isfile(full):
        raise HTTPException(404, "AI analysis artifact not found")
    return full, normalized


def _is_ai_code_preview_artifact(normalized_path: str) -> bool:
    lowered = normalized_path.replace("\\", "/").lower()
    filename = os.path.basename(lowered)
    ext = os.path.splitext(filename)[1]
    if ext == ".py":
        return True
    if ext != ".txt":
        return False
    return (
        "triton_output_code" in filename
        or "/triton_output_code/" in f"/{lowered}"
        or filename.startswith("output_code_")
    )


def _read_text_preview(path: str, max_bytes: int) -> tuple[str, bool, int]:
    size = _path_size(path)
    read_limit = min(size, max_bytes)
    with open(path, "rb") as f:
        data = f.read(read_limit)
    return data.decode("utf-8", errors="replace"), size > read_limit, size


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
    if state == "queued":
        return 5
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
    versions = _collect_ai_analysis_versions(jid, status)
    selected_version = versions[0] if versions else None
    report_exists = report_exists or bool(selected_version and selected_version.get("report_exists"))
    return {
        "enabled": CLAUDE_ANALYSIS_ENABLED,
        "status": status.get("status", "not_started"),
        "error": status.get("error", ""),
        "mode": status.get("mode", ""),
        "skill": status.get("skill", ""),
        "model": status.get("model") or _ai_backend_model_name(),
        "phase": status.get("phase", ""),
        "progress": _ai_analysis_progress(status),
        "diagnostics": status.get("diagnostics"),
        "started_at": status.get("started_at", ""),
        "finished_at": status.get("finished_at", ""),
        "updated_at": status.get("updated_at", ""),
        **timing,
        "report_exists": report_exists,
        "generated_at": selected_version.get("generated_at", "") if selected_version else "",
        "latest_version_id": selected_version.get("id", "") if selected_version else "",
        "selected_version_id": selected_version.get("id", "") if selected_version else "",
        "selected_version": selected_version,
        "versions": versions,
    }


_AI_FINDING_BULLET_RE = re.compile(
    r"^\s{0,3}[-*+]\s+(?:\*\*)?(结论|证据|建议)(?:\*\*)?\s*[:：]\s*(.*?)\s*$"
)


def _clean_ai_table_text(value: object) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([，。；：、])", r"\1", text)
    text = text.replace(".；", "；").replace("。；", "；")
    return text.strip("；; ")


def _split_ai_table_items(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = re.split(r"\s*[；;]\s*", str(value or ""))
    items: list[str] = []
    for item in raw_items:
        cleaned = _clean_ai_table_text(item)
        if cleaned:
            items.append(cleaned.replace("|", "\\|"))
    return items


def _format_ai_table_cell(value: object) -> str:
    items = _split_ai_table_items(value)
    if not items:
        return "-"
    bullets = []
    for item in items:
        cleaned = re.sub(r"^\s*(?:[-*+]|\d+\.|•)\s+", "", item)
        bullets.append(f"• {cleaned}")
    return "<br>".join(bullets)


def _format_triton_compute_rate(profile: object) -> str:
    if not isinstance(profile, dict):
        return "-"
    parts = []
    compute_ops = profile.get("compute_ops")
    if compute_ops:
        try:
            value = float(compute_ops)
            if value >= 1e12:
                parts.append(f"计算量 {value / 1e12:.2f} Tops")
            elif value >= 1e9:
                parts.append(f"计算量 {value / 1e9:.2f} Gops")
            elif value >= 1e6:
                parts.append(f"计算量 {value / 1e6:.2f} Mops")
            else:
                parts.append(f"计算量 {value:.0f} ops")
        except (TypeError, ValueError):
            pass
    try:
        compute_gops = profile.get("compute_gops")
        if compute_gops is not None:
            parts.append(f"速率 {float(compute_gops):.2f} GOPS")
    except (TypeError, ValueError):
        pass
    if parts:
        return "；".join(parts)

    # Historical triton_code_optimization.json may contain only a mixed summary
    # such as "IO ...；计算 ...；AI ...". Keep only the compute segment here.
    summary = str(profile.get("summary") or profile.get("mac_summary") or "")
    match = re.search(r"(计算\s*[^；;]+(?:[/／]\s*[^；;]+)?)", summary)
    return match.group(1).replace("|", "\\|") if match else "-"


def _triton_action_items(candidate: dict) -> list[str]:
    items: list[str] = []
    raw_strategies = candidate.get("strategies") or []
    if isinstance(raw_strategies, str):
        strategies = raw_strategies
    else:
        strategies = ", ".join(str(strategy) for strategy in raw_strategies if strategy)
    if strategies:
        items.append(f"方向：{strategies}")
    recommendations = candidate.get("recommendation_items") or candidate.get("recommendation")
    items.extend(_split_ai_table_items(recommendations))
    return items


def _split_triton_strategy_text(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = re.split(r"\s*/\s*|\s*,\s*", str(value or ""))
    items: list[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in items:
            items.append(text)
    return items


def _triton_candidates_from_kernels(kernels: object) -> list[dict]:
    if not isinstance(kernels, list):
        return []

    candidates: list[dict] = []
    for kernel in kernels:
        if not isinstance(kernel, dict):
            continue
        findings = kernel.get("findings") or []
        if not isinstance(findings, list):
            findings = []

        strategies: list[str] = []
        evidence_items: list[str] = []
        recommendation_items: list[str] = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            for strategy in _split_triton_strategy_text(finding.get("strategy")):
                if strategy not in strategies:
                    strategies.append(strategy)
            evidence = finding.get("evidence")
            if evidence and len(evidence_items) < 3:
                evidence_items.append(str(evidence))
            recommendation = finding.get("recommendation")
            if recommendation and len(recommendation_items) < 2:
                recommendation_items.append(str(recommendation))

        if not findings and not kernel.get("kernel_name"):
            continue
        candidates.append({
            "kernel_name": kernel.get("kernel_name") or "-",
            "file": kernel.get("file") or kernel.get("output_code_file") or "-",
            "priority": kernel.get("priority"),
            "total_ms": kernel.get("total_ms"),
            "bandwidth_utilization": kernel.get("bandwidth_utilization"),
            "estimated_profile": kernel.get("estimated_profile"),
            "strategies": strategies,
            "evidence_items": evidence_items,
            "recommendation_items": recommendation_items,
            "evidence": "; ".join(evidence_items),
            "recommendation": recommendation_items[0] if recommendation_items else "",
        })
    return candidates


def _triton_code_summary_from_payload(payload: dict, candidates: list[dict]) -> str:
    summary = payload.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    scanned = summary.get("scanned_files")
    kernels = summary.get("kernels_with_findings")
    findings = summary.get("finding_count")
    strategies_raw = summary.get("top_strategies") or []
    strategies = []
    if isinstance(strategies_raw, list):
        for item in strategies_raw[:4]:
            if isinstance(item, dict):
                strategy = item.get("strategy")
            else:
                strategy = item
            if strategy:
                strategies.append(str(strategy))

    if scanned is None:
        scanned = len(candidates)
    if kernels is None:
        kernels = len(candidates)
    base = f"扫描 {scanned} 个 Triton output_code，{kernels} 个 kernel 存在代码级候选"
    if findings is not None:
        base += f"，共 {findings} 条优化信号"
    if strategies:
        base += f"，主要方向为 {', '.join(strategies)}"
    return base + "。"


def _format_ai_finding_blocks(items: list[tuple[str, str]]) -> Optional[list[str]]:
    if len(items) < 3:
        return None
    entries: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    for label, text in items:
        if label == "结论" and current:
            entries.append(current)
            current = {}
        current.setdefault(label, []).append(text)
    if current:
        entries.append(current)

    meaningful = [
        entry for entry in entries
        if entry.get("结论") and (entry.get("证据") or entry.get("建议"))
    ]
    if not meaningful:
        return None

    lines: list[str] = []
    for idx, entry in enumerate(entries, 1):
        conclusion = (entry.get("结论") or ["关键问题"])[0].strip()
        title = re.sub(r"[。；;，,.\s]+$", "", conclusion)
        if len(title) > 52:
            title = title[:52].rstrip() + "..."
        lines.extend([f"### 发现 {idx}：{title or '关键问题'}", ""])
        for label in ("结论", "证据", "建议"):
            values = [value.strip() for value in entry.get(label, []) if value.strip()]
            if not values:
                continue
            if len(values) == 1:
                lines.append(f"**{label}：** {values[0]}")
            else:
                lines.append(f"**{label}：**")
                lines.extend(f"- {value}" for value in values)
            lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _normalize_ai_report_markdown(content: str) -> str:
    if not content:
        return ""
    source = str(content).replace("\r\n", "\n").replace("\r", "\n")
    lines = source.split("\n")
    output: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            output.append(lines[i])
            i += 1
            continue
        if in_fence:
            output.append(lines[i])
            i += 1
            continue

        match = _AI_FINDING_BULLET_RE.match(lines[i])
        if not match:
            output.append(lines[i])
            i += 1
            continue

        start = i
        items: list[tuple[str, str]] = []
        while i < len(lines):
            bullet = _AI_FINDING_BULLET_RE.match(lines[i])
            if bullet:
                items.append((bullet.group(1), bullet.group(2).strip()))
                i += 1
                continue
            if not lines[i].strip() and i + 1 < len(lines) and _AI_FINDING_BULLET_RE.match(lines[i + 1]):
                i += 1
                continue
            break

        formatted = _format_ai_finding_blocks(items)
        if not formatted:
            output.extend(lines[start:i])
            continue
        if output and output[-1].strip():
            output.append("")
        output.extend(formatted)
        if i < len(lines) and lines[i].strip():
            output.append("")

    normalized = "\n".join(output).rstrip()
    return normalized + ("\n" if source.endswith("\n") else "")


def _build_triton_code_optimization_section(jid: str) -> str:
    path = os.path.join(ai_analysis_dir(jid), "triton_code_optimization.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    if not payload.get("has_findings"):
        return ""

    guidance = payload.get("final_report_guidance") or {}
    summary = guidance.get("summary_cn") or ""
    candidates = guidance.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []
    kernel_candidates = _triton_candidates_from_kernels(payload.get("kernels"))
    if kernel_candidates and len(kernel_candidates) > len(candidates):
        candidates = kernel_candidates
    if not summary and candidates:
        summary = _triton_code_summary_from_payload(payload, candidates)
    if candidates:
        table_lines = [
            "| Kernel | 代码文件 | 耗时 | BW 利用率 | 计算速率估算 | 优化方向与建议 |",
            "|---|---|---:|---:|---|---|",
        ]
        for candidate in candidates:
            kernel_name = str(candidate.get("kernel_name") or "-").replace("|", "\\|")
            code_file = os.path.basename(str(candidate.get("file") or "-")).replace("|", "\\|")
            total_ms = candidate.get("total_ms")
            try:
                total_text = f"{float(total_ms):.2f} ms"
            except (TypeError, ValueError):
                total_text = "-"
            bw_util = candidate.get("bandwidth_utilization")
            try:
                bw_text = f"{float(bw_util) * 100:.1f}%"
            except (TypeError, ValueError):
                bw_text = "-"
            estimated = _format_triton_compute_rate(candidate.get("estimated_profile"))
            action = _format_ai_table_cell(_triton_action_items(candidate))
            table_lines.append(
                f"| `{kernel_name}` | `{code_file}` | {total_text} | {bw_text} | {estimated} | {action} |"
            )
    else:
        table_md = guidance.get("required_table_md") or ""
        table_lines = []
        for line in str(table_md).splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not stripped and not table_lines:
                continue
            table_lines.append(line)

    lines = ["## Triton Kernel 代码优化", ""]
    if summary:
        lines.extend([summary, ""])
    lines.extend(table_lines)
    lines.extend([
        "",
        "> 该章节由 `triton_code_optimization.json` 生成，属于静态代码级候选；具体收益需要用单 kernel benchmark 或重新采集 trace 验证。",
    ])
    return "\n".join(lines).rstrip()


def _has_top_level_triton_code_section(content: str) -> bool:
    return bool(re.search(r"(?m)^##\s+Triton Kernel 代码优化\s*$", content or ""))


def _replace_top_level_triton_code_section(content: str, section: str) -> str:
    match = re.search(r"(?m)^##\s+Triton Kernel 代码优化\s*$", content or "")
    if not match:
        return content
    next_match = re.search(r"(?m)^##\s+\S", content[match.end():])
    end = match.end() + next_match.start() if next_match else len(content)
    before = content[:match.start()].rstrip()
    after = content[end:].lstrip()
    parts = []
    if before:
        parts.append(before)
    parts.append(section)
    if after:
        parts.append(after)
    return "\n\n".join(parts).rstrip() + "\n"


def _strip_nested_triton_code_candidate_sections(content: str) -> str:
    lines = str(content or "").splitlines()
    output: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"^#{3,6}\s+Triton Kernel 代码优化候选\s*$", lines[i].strip()):
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if re.match(r"^##\s+\S", stripped) or stripped == "---":
                    break
                i += 1
            continue
        output.append(lines[i])
        i += 1
    normalized = "\n".join(output).rstrip()
    return normalized + ("\n" if normalized else "")


def _inject_triton_code_optimization_section(jid: str, content: str) -> str:
    if not content:
        return content
    section = _build_triton_code_optimization_section(jid)
    if not section:
        return content

    source = _strip_nested_triton_code_candidate_sections(content)
    if _has_top_level_triton_code_section(source):
        return _replace_top_level_triton_code_section(source, section)

    match = re.search(r"(?m)^##\s+不确定性与下一步\s*$", source)
    if not match:
        match = re.search(r"(?m)^##\s+产物\s*$", source)
    if not match:
        return source.rstrip() + "\n\n" + section + "\n"
    before = source[:match.start()].rstrip()
    after = source[match.start():].lstrip()
    return before + "\n\n" + section + "\n\n" + after


def _finalize_ai_report_markdown(jid: str, content: str) -> str:
    normalized = _normalize_ai_report_markdown(content)
    return _inject_triton_code_optimization_section(jid, normalized)


def _read_ai_analysis_report(jid: str, version_id: str = "") -> str:
    version = _select_ai_analysis_version(jid, version_id)
    if not version:
        return ""
    path = _ai_version_report_path_from_metadata(jid, version)
    if not path:
        return ""
    return _finalize_ai_report_markdown(jid, _read_text_file(path))


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


def _write_claude_env_settings(claude_dir: str, custom_headers: str) -> None:
    if not custom_headers:
        return
    settings_path = os.path.join(claude_dir, "settings.local.json")
    settings: dict = {}
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                settings = loaded
        except Exception:
            settings = {}
    env_settings = settings.get("env")
    if not isinstance(env_settings, dict):
        env_settings = {}
    env_settings["ANTHROPIC_CUSTOM_HEADERS"] = custom_headers
    settings["env"] = env_settings
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _build_claude_env(base_env: dict, analysis_dir: str, extra: Optional[dict] = None) -> dict:
    env = base_env.copy()
    if extra:
        env.update(extra)
    claude_dir = os.path.join(analysis_dir, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    env.setdefault("CLAUDE_PROJECT_DIR", analysis_dir)
    env.setdefault("CLAUDE_CODE_PROJECT_DIR", analysis_dir)
    env.setdefault("TRACE_CLAUDE_PROJECT_DIR", analysis_dir)
    if TRACE_CLAUDE_CUSTOM_HEADERS:
        env["ANTHROPIC_CUSTOM_HEADERS"] = TRACE_CLAUDE_CUSTOM_HEADERS
    elif CLAUDE_CUSTOM_HEADERS:
        env.setdefault("ANTHROPIC_CUSTOM_HEADERS", CLAUDE_CUSTOM_HEADERS)
    custom_headers = str(env.get("ANTHROPIC_CUSTOM_HEADERS", "")).strip()
    if custom_headers:
        env["ANTHROPIC_CUSTOM_HEADERS"] = custom_headers
        _write_claude_env_settings(claude_dir, custom_headers)
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


def _render_claude_prompt(
    job: dict,
    trace_inputs: dict,
    skill: str,
    report_path: str,
    skills_dir: str = "",
    user_prompt: str = "",
) -> str:
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
        "- 默认生成适合 Web 阅读的短报告：单 trace 在 `## 产物` 前不超过约 1200 个中文字符，对比报告不超过约 1500 个中文字符。",
        "- 先给出 3-4 个关键发现，说明主要性能热点、回退或改善点；合并相同根因，不要重复展开。",
        "- `## 结论概览` 不要输出平铺的 `- 结论` / `- 证据` / `- 建议` 同级列表。",
        "- 每个关键发现请使用 `### 发现 N：一句话标题`，下面分别写 `**结论：** ...`、`**证据：** ...`、`**建议：** ...`，每段只写一句短句。",
        "- 不要同时写一套详细的 `主要发现` / `主要回退` 来重复 `结论概览`；详细证据、脚本输出、长表格和调用栈放到产物文件中引用。",
        "- 例外：如果生成了 `triton_code_optimization.json` 且其中 `has_findings=true`，必须在最终报告正文中加入独立顶级章节 `## Triton Kernel 代码优化`，展示 `final_report_guidance.candidates` / `required_table_md` 中的全部候选，包含静态计算速率估算与合并后的优化方向/建议，不要新增单独的 `证据` 列；位置放在 `## 优先行动` 之后、`## 不确定性与下一步` 之前，不要只把它放到 `## 产物`。",
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
    user_prompt = _normalize_ai_user_prompt(user_prompt)
    if user_prompt:
        lines.extend([
            "",
            "用户补充 Prompt:",
            "```text",
            user_prompt,
            "```",
            "请将用户补充 Prompt 纳入分析；如果它与上面的报告格式要求冲突，以上面的报告格式要求为准。",
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


async def _run_ai_analysis_task(jid: str, notification_context: Optional[dict] = None):
    analysis_dir = ai_analysis_dir(jid)
    report_path = ai_analysis_report_path(jid)
    os.makedirs(analysis_dir, exist_ok=True)
    previous_status = _read_ai_analysis_status(jid)
    user_prompt = _normalize_ai_user_prompt(
        previous_status.get("user_prompt") or (notification_context or {}).get("user_prompt") or ""
    )
    with contextlib.suppress(OSError):
        os.remove(report_path)
    stdout_text = ""
    stderr_text = ""
    job = None

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
            "model": _ai_backend_model_name(),
            "user_prompt": user_prompt,
            "skills_dir": skills_dir,
            "trigger_user_token": previous_status.get("trigger_user_token") or (notification_context or {}).get("trigger_user_token") or "",
            "trigger_user_display": previous_status.get("trigger_user_display") or (notification_context or {}).get("trigger_user_display") or "",
            "trigger_user_email": previous_status.get("trigger_user_email") or (notification_context or {}).get("trigger_user_email") or "",
            "owner_user_token": previous_status.get("owner_user_token") or (notification_context or {}).get("owner_user_token") or job.get("user_token") or "",
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
            final_status = {
                **status,
                "status": "error",
                "phase": "diagnostics_failed",
                "progress": 100,
                "error": "AI environment diagnostics failed",
                "diagnostics": diagnostics,
                "finished_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
            version_meta = _save_ai_analysis_version(jid, content, final_status, job, notification_context)
            if version_meta:
                final_status["latest_version_id"] = version_meta["id"]
            _write_ai_analysis_status(jid, final_status)
            return

        status = {
            **status,
            "phase": "analyzing",
            "progress": 55,
            "diagnostics": diagnostics,
            "updated_at": _utc_now_iso(),
        }
        _write_ai_analysis_status(jid, status)

        prompt = _render_claude_prompt(job, trace_inputs, skill, report_path, skills_dir, user_prompt)
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
        content = _finalize_ai_report_markdown(jid, content)
        _write_ai_report(report_path, content)
        final_status = {
            **status,
            "status": "done",
            "progress": 100,
            "finished_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        version_meta = _save_ai_analysis_version(jid, content, final_status, job, notification_context)
        if version_meta:
            final_status["latest_version_id"] = version_meta["id"]
        _write_ai_analysis_status(jid, final_status)
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
        content = "\n".join(error_report)
        _write_ai_report(report_path, content)
        status = _read_ai_analysis_status(jid)
        final_status = {
            **status,
            "status": "error",
            "progress": 100,
            "error": str(e),
            "finished_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        version_meta = _save_ai_analysis_version(jid, content, final_status, job, notification_context)
        if version_meta:
            final_status["latest_version_id"] = version_meta["id"]
        _write_ai_analysis_status(jid, final_status)
        logger.exception(
            "ai_analysis_failed",
            extra={"event": "ai_analysis_failed", "job_id": jid, "error": str(e)},
        )
    finally:
        try:
            final_status = _read_ai_analysis_status(jid)
            if final_status.get("status") in {"done", "error"}:
                await _send_ai_analysis_completion_notification(
                    jid,
                    job or {},
                    final_status,
                    notification_context or {},
                )
        except Exception as notify_exc:
            logger.warning(
                "ai_analysis_email_notify_failed",
                extra={
                    "event": "ai_analysis_email_notify_failed",
                    "job_id": jid,
                    "error": str(notify_exc),
                },
                exc_info=True,
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
        if op in ("~", "!~", "=="):
            terms = [term.lower() for term in text.split("|") if term]
            if op == "==":
                hit = any(term == str(cell).lower() for term in terms)
            else:
                hit = any(term in str(cell).lower() for term in terms)
            if (op == "~" and not hit) or (op == "!~" and hit) or (op == "==" and not hit):
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
        if _is_kernel_type_result_csv(filename):
            rows = _aggregate_kernel_type_rows(
                filename,
                fields,
                [_result_csv_row(filename, row) for row in reader],
            )
            total = len(rows)
            rows = [row for row in rows if _csv_filter_match(row, q, filters, filter_ops)]
            filtered_total = len(rows)
            if sort_col and sort_col in fields:
                rows.sort(
                    key=lambda item: _sort_value(item.get(sort_col)),
                    reverse=(sort_dir == "desc"),
                )
            page_rows = rows[offset:offset + limit]
            return {
                "fields": fields,
                "rows": page_rows,
                "total": total,
                "filtered_total": filtered_total,
                "limit": limit,
                "offset": offset,
            }
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
                 "non_triton_kernel_efficiency_avg.csv",
                 "aten_ops_avg.csv", "aten_ops_cmp.csv",
                 "tf_ops_avg.csv", "tf_ops_cmp.csv",
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
    return _prune_empty_tensorflow_result_csvs(files)


def collect_perfetto_context(jid: str) -> dict:
    path = os.path.join(result_dir(jid), "perfetto_context.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _round_ms(value):
    return round(value, 2) if isinstance(value, (int, float)) else None


def _float_or_none(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _delta_pct(parent, child):
    if parent in (None, 0) or child is None:
        return None
    return round((child - parent) / parent * 100, 1)


def _read_perfetto_e2e_ms(job_id: str):
    path = os.path.join(result_dir(job_id), "perfetto_context.json")
    try:
        with open(path) as f:
            payload = json.load(f)
        dur_ns = (payload.get("a") or {}).get("dur_ns")
        if dur_ns is None:
            return None
        return _round_ms(float(dur_ns) / 1_000_000)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _read_avg_ms_csv(job_id: str, filename: str, key_field: str) -> tuple[float | None, dict, float | None]:
    path = os.path.join(result_dir(job_id), filename)
    total = 0.0
    count_total = 0.0
    by_key = {}
    seen = False
    seen_count = False
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    avg_ms = float(row.get("avg_dur_ms") or "")
                except (TypeError, ValueError):
                    continue
                seen = True
                total += avg_ms
                try:
                    avg_count = float(row.get("avg_count") or "")
                    seen_count = True
                    count_total += avg_count
                except (TypeError, ValueError):
                    pass
                key = (row.get(key_field) or "").strip()
                if key:
                    by_key[key] = by_key.get(key, 0.0) + avg_ms
    except OSError:
        return None, {}, None
    rounded_by_key = {key: _round_ms(value) for key, value in by_key.items()}
    return (_round_ms(total) if seen else None), rounded_by_key, (_round_ms(count_total) if seen_count else None)


def _step_summary_from_parts(headers, parts) -> dict:
    def column_value(name):
        try:
            return _float_or_none(parts[headers.index(name)])
        except (ValueError, IndexError):
            return None

    return {
        "step_dur_ms": column_value("step_dur(ms)"),
        "aten_ops_count": column_value("aten_ops"),
        "aten_ops_ms": column_value("aten_dur(ms)"),
    }


def _read_step_summary_from_console(console_out: str | None) -> dict:
    if not console_out:
        return {}
    headers = None
    step_rows = []
    step_count = None
    for raw_line in str(console_out).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        step_count_match = re.search(r"Per-Step Summary\s*\((\d+)\s+steps?\)", line)
        if step_count_match:
            step_count = int(step_count_match.group(1))
            continue
        if "step_dur(ms)" in line and "aten_ops" in line:
            headers = line.split()
            continue
        if not headers:
            continue
        if line.startswith("-"):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0].lower() == "avg":
            summary = {
                key: _round_ms(value)
                for key, value in _step_summary_from_parts(headers, parts).items()
                if value is not None
            }
            if step_count is not None:
                summary["step_count"] = step_count
            return summary
        if parts[0].isdigit():
            row = _step_summary_from_parts(headers, parts)
            if any(value is not None for value in row.values()):
                step_rows.append(row)
        elif line.startswith("===") and step_rows:
            break
    if not step_rows:
        return {}
    averaged = {}
    for key in ("step_dur_ms", "aten_ops_count", "aten_ops_ms"):
        values = [row[key] for row in step_rows if row.get(key) is not None]
        if values:
            averaged[key] = _round_ms(sum(values) / len(values))
    if step_count is not None:
        averaged["step_count"] = step_count
    elif step_rows:
        averaged["step_count"] = len(step_rows)
    return averaged


def _read_hot_kernel_metrics(job_id: str, limit: int = 5) -> tuple[list[dict], dict]:
    path = os.path.join(result_dir(job_id), "all_kernels_avg.csv")
    by_name = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("kernel_name") or "").strip()
                if not name:
                    continue
                avg_ms = _float_or_none(row.get("avg_dur_ms"))
                if avg_ms is None:
                    continue
                avg_count = _float_or_none(row.get("avg_count"))
                item = by_name.setdefault(name, {
                    "name": name,
                    "family": (row.get("family") or "").strip(),
                    "avg_count": 0.0,
                    "avg_dur_ms": 0.0,
                })
                item["avg_dur_ms"] += avg_ms
                if avg_count is not None:
                    item["avg_count"] += avg_count
    except OSError:
        return [], {}
    rows = []
    for item in by_name.values():
        rows.append({
            "name": item["name"],
            "family": item["family"],
            "avg_count": _round_ms(item["avg_count"]),
            "avg_dur_ms": _round_ms(item["avg_dur_ms"]),
        })
    rows.sort(key=lambda item: item.get("avg_dur_ms") or 0, reverse=True)
    kernel_durations = {item["name"]: item["avg_dur_ms"] for item in rows}
    return rows[:limit], kernel_durations


def _read_node_metrics(job_id: str, console_out: str | None = None) -> dict:
    compute_ms, by_type, kernel_count = _read_avg_ms_csv(job_id, "kernel_types_avg.csv", "type")
    comm_ms, _, _ = _read_avg_ms_csv(job_id, "cncl_ops_avg.csv", "op_name")
    aten_ops_ms, _, aten_ops_count = _read_avg_ms_csv(job_id, "aten_ops_avg.csv", "op_name")
    step_summary = _read_step_summary_from_console(console_out)
    top_kernels, kernel_durations = _read_hot_kernel_metrics(job_id)
    return {
        "e2e_ms": _read_perfetto_e2e_ms(job_id),
        "step_dur_ms": step_summary.get("step_dur_ms"),
        "step_count": step_summary.get("step_count"),
        "compute_ms": compute_ms,
        "comm_ms": comm_ms,
        "kernel_count": kernel_count,
        "aten_ops_count": step_summary.get("aten_ops_count", aten_ops_count),
        "aten_ops_ms": step_summary.get("aten_ops_ms", aten_ops_ms),
        "by_type": by_type,
        "top_kernels": top_kernels,
        "kernel_durations": kernel_durations,
    }


def _step_meta_for(metrics: dict) -> dict:
    return {
        "step_count": metrics.get("step_count"),
        "step_dur_ms": metrics.get("step_dur_ms"),
    }


def _step_meta_warning(parent: dict, child: dict) -> str:
    parent_count = parent.get("step_count")
    child_count = child.get("step_count")
    if parent_count is not None and child_count is not None and parent_count != child_count:
        return "两端 step 口径不一致，跨度对比仅供参考"
    parent_step = parent.get("step_dur_ms")
    child_step = child.get("step_dur_ms")
    if parent_step not in (None, 0) and child_step is not None:
        ratio = abs(child_step - parent_step) / abs(parent_step)
        if ratio > 0.5:
            return "两端单步耗时差异较大，跨度对比仅供参考"
    return ""


def compute_perf_delta(
    parent_job_id: str,
    child_job_id: str,
    parent_console_out: str | None = None,
    child_console_out: str | None = None,
) -> dict:
    parent = _read_node_metrics(parent_job_id, parent_console_out)
    child = _read_node_metrics(child_job_id, child_console_out)

    metrics = {}
    incomplete = False
    for key in ("e2e_ms", "compute_ms", "comm_ms"):
        parent_value = parent.get(key)
        child_value = child.get(key)
        if key in {"e2e_ms", "compute_ms"} and (parent_value is None or child_value is None):
            incomplete = True
        metrics[key] = {
            "parent": parent_value,
            "child": child_value,
            "delta_pct": _delta_pct(parent_value, child_value),
        }

    top_movers = []
    parent_types = parent.get("by_type") or {}
    child_types = child.get("by_type") or {}
    for kernel_type in set(parent_types) | set(child_types):
        parent_ms = parent_types.get(kernel_type, 0.0)
        child_ms = child_types.get(kernel_type, 0.0)
        delta_ms = child_ms - parent_ms
        top_movers.append({
            "type": kernel_type,
            "parent_ms": _round_ms(parent_ms),
            "child_ms": _round_ms(child_ms),
            "delta_ms": _round_ms(delta_ms),
            "delta_pct": _delta_pct(parent_ms, child_ms),
        })
    top_movers.sort(key=lambda item: abs(item.get("delta_ms") or 0), reverse=True)
    notes = []
    if incomplete:
        notes.append("部分结果文件缺失或尚未生成")
    step_warning = _step_meta_warning(parent, child)
    if step_warning:
        notes.append(step_warning)

    return {
        "schema": 1,
        "generated_at": _utc_now_iso(),
        "metrics": metrics,
        "top_movers": top_movers[:5],
        "step_meta": {
            "parent": _step_meta_for(parent),
            "child": _step_meta_for(child),
            "warning": step_warning,
        },
        "incomplete": incomplete,
        "notes": "；".join(notes),
    }


async def _read_node_metrics_async(job_id: str, console_out: str | None = None) -> dict:
    return await run_in_threadpool(_read_node_metrics, job_id, console_out)


async def _load_experiment_console_outs(parent_job_id: str, child_job_id: str) -> dict:
    db = await get_db()
    try:
        rows = await (
            await db.execute(
                "SELECT id, console_out FROM jobs WHERE id IN (?,?)",
                (parent_job_id, child_job_id),
            )
        ).fetchall()
        return {row["id"]: row["console_out"] for row in rows}
    finally:
        await db.close()


async def compute_perf_delta_async(
    parent_job_id: str,
    child_job_id: str,
    parent_console_out: str | None = None,
    child_console_out: str | None = None,
) -> dict:
    if parent_console_out is None or child_console_out is None:
        console_outs = await _load_experiment_console_outs(parent_job_id, child_job_id)
        if parent_console_out is None:
            parent_console_out = console_outs.get(parent_job_id)
        if child_console_out is None:
            child_console_out = console_outs.get(child_job_id)
    return await run_in_threadpool(
        compute_perf_delta,
        parent_job_id,
        child_job_id,
        parent_console_out,
        child_console_out,
    )


def _parse_json_field(value, fallback):
    if value in (None, ""):
        return fallback
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback
    return parsed


def _normalize_experiment_variables(raw) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HTTPException(400, "variables must be an array")
    variables = []
    for item in raw:
        if not isinstance(item, dict):
            raise HTTPException(400, "variables items must be objects")
        name = str(item.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "variables items require name")
        variables.append({
            "name": name,
            "from": "" if item.get("from") is None else str(item.get("from")).strip(),
            "to": "" if item.get("to") is None else str(item.get("to")).strip(),
        })
    return variables


def _experiment_edge_response(row: dict) -> dict:
    data = dict(row)
    data["variables"] = _parse_json_field(data.get("variables"), [])
    if not isinstance(data["variables"], list):
        data["variables"] = []
    perf = _parse_json_field(data.get("perf_summary"), None)
    data["perf"] = perf if isinstance(perf, dict) else None
    data.pop("perf_summary", None)
    data["perf_summary_edited"] = 1 if data.get("perf_summary_edited") else 0
    return data


def _normalize_edge_label_layout(raw) -> dict:
    if not isinstance(raw, dict):
        raise HTTPException(400, "label_layout must be an object")
    normalized = {}
    for key in ("label_x", "label_y", "label_width", "label_height"):
        if key not in raw:
            continue
        try:
            value = float(raw.get(key))
        except (TypeError, ValueError):
            raise HTTPException(400, "标签布局必须是数字")
        if key in {"label_x", "label_y"}:
            if not (-1_000_000 <= value <= 1_000_000):
                raise HTTPException(400, "标签坐标超出范围")
        elif key == "label_width":
            if not (120 <= value <= 900):
                raise HTTPException(400, "标签宽度超出范围")
        elif key == "label_height":
            if not (36 <= value <= 420):
                raise HTTPException(400, "标签高度超出范围")
        normalized[key] = value
    return normalized


def _would_create_cycle(edges, parent: str, child: str) -> bool:
    adj = {}
    for edge in edges:
        adj.setdefault(edge["parent_job_id"], []).append(edge["child_job_id"])
    seen = set()
    queue = [child]
    while queue:
        node = queue.pop(0)
        if node == parent:
            return True
        if node in seen:
            continue
        seen.add(node)
        queue.extend(adj.get(node, []))
    return False


async def _load_experiment_edge(db, edge_id: str) -> Optional[dict]:
    return await row_to_dict(
        await (await db.execute("SELECT * FROM experiment_edges WHERE id=?", (edge_id,))).fetchone()
    )


async def _clear_experiment_job_links(db, job_ids: list[str]):
    if not job_ids:
        return
    placeholders = ",".join("?" * len(job_ids))
    params = tuple(job_ids)
    await db.execute(
        f"DELETE FROM experiment_edges WHERE parent_job_id IN ({placeholders}) OR child_job_id IN ({placeholders})",
        (*params, *params),
    )
    await db.execute(
        f"DELETE FROM experiment_node_layout WHERE job_id IN ({placeholders})",
        params,
    )
    await db.execute(
        f"UPDATE experiment_edges SET compare_job_id=NULL WHERE compare_job_id IN ({placeholders})",
        params,
    )


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
            SELECT id, seq, label, created_at, project_id
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
                SELECT j.id, j.seq, j.label, j.project_id, j.created_at, j.file_a_name,
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


def _perfetto_context_from_trace(path, source_name=None):
    from trace_analyzer import compute_avgs, parse_trace

    return _perfetto_context(compute_avgs(parse_trace(path, source_name=source_name)))


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


def _largest_archive_json_size(path: str) -> Optional[int]:
    lower = path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            sizes = [
                info.file_size
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".json")
            ]
        if not sizes:
            raise ValueError("Archive does not contain a JSON trace")
        return max(sizes)
    if lower.endswith((".tar.gz", ".tgz")) or (lower.endswith(".gz") and _is_tar_archive(path)):
        with tarfile.open(path, "r:*") as tar:
            sizes = [
                member.size
                for member in tar.getmembers()
                if member.isfile() and member.name.lower().endswith(".json")
            ]
        if not sizes:
            raise ValueError("Archive does not contain a JSON trace")
        return max(sizes)
    if not lower.endswith(".gz"):
        return os.path.getsize(path)
    return None


def _assert_trace_json_size_supported(path: str, slot: str = "trace") -> None:
    if not MAX_TRACE_JSON_BYTES:
        return
    label = f"Trace {slot.upper()}" if slot else "Trace"
    try:
        known_size = _largest_archive_json_size(path)
        if known_size is not None:
            if known_size > MAX_TRACE_JSON_BYTES:
                raise ValueError(
                    f"{label} 解压后 JSON 大小 {_format_size(known_size)} 超过当前分析上限 "
                    f"{_format_size(MAX_TRACE_JSON_BYTES)}。为了避免服务内存耗尽，已停止分析；"
                    "可缩短 profiler 采样窗口、指定 step 重新导出，或在确认机器内存足够后设置 "
                    "TRACE_MAX_TRACE_JSON_BYTES 调大/设为 0 关闭保护。"
                )
            return

        # Plain gzip does not expose a reliable uncompressed size for >4GiB
        # files.  The parser handles gzip traces in a streaming-safe way, so
        # avoid a full pre-scan by default; otherwise large traces are
        # decompressed twice before producing any result.
        if not STRICT_GZIP_SIZE_CHECK:
            return

        total = 0
        for chunk in _iter_json_chunks(path):
            total += len(chunk)
            if total > MAX_TRACE_JSON_BYTES:
                raise ValueError(
                    f"{label} 解压后 JSON 大小超过当前分析上限 {_format_size(MAX_TRACE_JSON_BYTES)}。"
                    "为了避免服务内存耗尽，已停止分析；可缩短 profiler 采样窗口、指定 step 重新导出，"
                    "或在确认机器内存足够后设置 TRACE_MAX_TRACE_JSON_BYTES 调大/设为 0 关闭保护。"
                )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"{label} 文件预检失败: {e}") from e


# ── Synchronous analysis (runs in thread pool, must not await) ────────────────

def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    if minutes:
        return f"{minutes}分{sec}秒"
    return f"{sec}秒"


async def _write_analysis_progress(job_id: str, message: str) -> None:
    progress_db = None
    try:
        progress_db = await get_db()
        await progress_db.execute(
            "UPDATE jobs SET console_out=? WHERE id=? AND status='running'",
            (message, job_id),
        )
        await progress_db.commit()
    except Exception:
        logger.debug(
            "analysis_progress_update_failed",
            extra={"event": "analysis_progress_update_failed", "job_id": job_id},
            exc_info=True,
        )
    finally:
        if progress_db is not None:
            await progress_db.close()


def _run_sync_analysis(job, rdir, path_a, path_b, name_a, name_b, progress_callback=None):
    """All blocking I/O lives here so the event loop stays free."""
    from trace_analyzer import (print_step_summary, print_kernel_type_breakdown, print_top_kernels,
                                write_single, print_comparison, write_comparison)

    def stage(message: str):
        if progress_callback:
            progress_callback(message)

    def compute_trace(path, step_filter, slot, source_name=None):
        trace_label = source_name or os.path.basename(path)

        def parse_progress(events_seen, records_seen):
            stage(
                f"解析 Trace {slot.upper()}: {trace_label} · "
                f"已读 {events_seen:,} events · 命中 {records_seen:,} 条"
            )

        stage(f"预检 Trace {slot.upper()}")
        _assert_trace_json_size_supported(path, slot)
        stage(f"解析 Trace {slot.upper()}: {trace_label}")
        keep_triton_code = bool(job["save_triton_csv"] or job["save_triton_code"])
        parsed = parse_trace(
            path,
            keep_triton_code=keep_triton_code,
            progress_callback=parse_progress,
            source_name=source_name,
        )
        steps = parse_step_filter(step_filter)
        if steps:
            stage(f"筛选 Trace {slot.upper()} step: {step_filter}")
            parsed = filter_parsed_steps(parsed, steps)
        stage(f"聚合 Trace {slot.upper()} 指标")
        return compute_avgs(parsed)

    buf = io.StringIO()
    perfetto_context = {}
    with contextlib.redirect_stdout(buf):
        if job["mode"] == "single":
            step_filter_a = (job.get("step_filter_a") or "").strip()
            if step_filter_a:
                print(f"Step filter A: {step_filter_a}")
            data = compute_trace(path_a, step_filter_a, "a", name_a)
            fake_args = types.SimpleNamespace(
                output_dir=rdir,
                save_triton_csv=bool(job["save_triton_csv"]),
                save_triton_code=bool(job["save_triton_code"]),
            )
            stage("生成单 trace 结果")
            print_step_summary(data)
            print_kernel_type_breakdown(data)
            print_top_kernels(data)
            write_single(data, fake_args)
            perfetto_context["a"] = _perfetto_context(data)
        else:
            step_filter_a = (job.get("step_filter_a") or "").strip()
            step_filter_b = (job.get("step_filter_b") or "").strip()
            if step_filter_a:
                print(f"Step filter A: {step_filter_a}")
            if step_filter_b:
                print(f"Step filter B: {step_filter_b}")
            data_a = compute_trace(path_a, step_filter_a, "a", name_a)
            data_b = compute_trace(path_b, step_filter_b, "b", name_b)
            fake_args = types.SimpleNamespace(output_dir=rdir)
            label_a = name_a or os.path.basename(path_a)
            label_b = name_b or os.path.basename(path_b)
            stage("生成对比结果")
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
    progress_task = None
    progress_state = None
    try:
        claim = await db.execute(
            """
            UPDATE jobs
            SET status='running',
                error_msg='',
                console_out='准备分析任务 · 已用 0秒'
            WHERE id=? AND status='pending'
            """,
            (job_id,),
        )
        await db.commit()
        if claim.rowcount != 1:
            return

        cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
        job = await row_to_dict(await cursor.fetchone())
        if not job:
            return

        logger.info("analysis_started", extra={"event": "analysis_started", "job_id": job_id, "mode": job["mode"]})
        loop = asyncio.get_running_loop()
        progress_started_at = time.perf_counter()
        progress_state = {"message": "准备分析任务", "done": False}

        def set_progress(message: str) -> None:
            # Called from the analysis worker thread; keep it non-blocking so
            # SQLite/file I/O can never stall trace parsing.
            loop.call_soon_threadsafe(progress_state.__setitem__, "message", message)

        async def progress_heartbeat():
            while not progress_state["done"]:
                elapsed = _format_elapsed(time.perf_counter() - progress_started_at)
                await _write_analysis_progress(job_id, f"{progress_state['message']} · 已用 {elapsed}")
                await asyncio.sleep(5)

        progress_task = asyncio.create_task(progress_heartbeat())
        set_progress("准备分析任务")

        rdir = result_dir(job_id)
        os.makedirs(rdir, exist_ok=True)

        # Resolve source paths for compare-from-history (needs DB, must happen before thread)
        if job["mode"] == "compare" and not _primary_trace_path(job, "a"):
            cursor_a = await db.execute("SELECT * FROM jobs WHERE id=?", (job["source_job_a"],))
            src_a = await row_to_dict(await cursor_a.fetchone())
            cursor_b = await db.execute("SELECT * FROM jobs WHERE id=?", (job["source_job_b"],))
            src_b = await row_to_dict(await cursor_b.fetchone())
            if not src_a or not src_b:
                raise FileNotFoundError("Source job not found")
            path_a = _primary_trace_path(src_a)
            path_b = _primary_trace_path(src_b)
            if not path_a or not os.path.exists(path_a):
                raise FileNotFoundError("Trace file A not found")
            if not path_b or not os.path.exists(path_b):
                raise FileNotFoundError("Trace file B not found")
            name_a = src_a.get("file_a_name") or os.path.basename(path_a)
            name_b = src_b.get("file_a_name") or os.path.basename(path_b)
        else:
            path_a = _primary_trace_path(job, "a")
            path_b = _primary_trace_path(job, "b") if job["mode"] == "compare" else None
            if not path_a or not os.path.exists(path_a):
                raise FileNotFoundError("Trace file A not found")
            if job["mode"] == "compare" and (not path_b or not os.path.exists(path_b)):
                raise FileNotFoundError("Trace file B not found")
            name_a = job["file_a_name"]
            name_b = job["file_b_name"]

        # Run all blocking analysis in a thread pool so the event loop stays responsive
        console_out = await asyncio.to_thread(
            _run_sync_analysis,
            job, rdir, path_a, path_b, name_a, name_b,
            set_progress,
        )
        set_progress("压缩并整理 trace 文件")

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
        if progress_state is not None:
            progress_state["done"] = True
        if progress_task is not None:
            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task
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
        "# HELP analyze_trace_ai_analysis_queue_size Pending Claude AI analysis queue size in memory.",
        "# TYPE analyze_trace_ai_analysis_queue_size gauge",
        f"analyze_trace_ai_analysis_queue_size {ai_analysis_queue.qsize()}",
        "# HELP analyze_trace_ai_analysis_active Active Claude AI analysis tasks in memory.",
        "# TYPE analyze_trace_ai_analysis_active gauge",
        f"analyze_trace_ai_analysis_active {len(ai_analysis_tasks)}",
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


@app.get("/api/email/diagnostics")
async def email_diagnostics(request: Request):
    if AUTH_ENABLED:
        require_admin(request)
    return await asyncio.to_thread(_run_email_diagnostics_payload)


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


@app.get("/api/admin/usage")
async def admin_usage_stats(request: Request, days: int = 14):
    if AUTH_ENABLED:
        require_admin(request)
    days = max(1, min(int(days or 14), 90))
    today_date = datetime.now(USAGE_TZINFO).date()
    start_date = today_date - timedelta(days=days - 1)
    seven_start_date = today_date - timedelta(days=6)
    today = today_date.isoformat()
    start_day = start_date.isoformat()
    seven_start_day = seven_start_date.isoformat()
    audit_day_expr = _usage_sqlite_day_expr("created_at")
    business_actions = (
        "job.create",
        "job.compare_create",
        "job.batch_compare_create",
        "job.ai_analysis_start",
        "feedback.create",
    )

    db = await get_db()
    try:
        usage_rows = await (
            await db.execute(
                """
                SELECT day,
                       COUNT(*) AS dau,
                       COALESCE(SUM(request_count), 0) AS requests,
                       MAX(last_seen_at) AS last_seen_at
                FROM usage_daily
                WHERE day >= ?
                GROUP BY day
                """,
                (start_day,),
            )
        ).fetchall()
        audit_rows = await (
            await db.execute(
                f"""
                SELECT {audit_day_expr} AS day,
                       SUM(CASE WHEN action='job.create' THEN 1 ELSE 0 END) AS upload_jobs,
                       SUM(CASE WHEN action IN ('job.compare_create','job.batch_compare_create') THEN 1 ELSE 0 END) AS compare_jobs,
                       SUM(CASE WHEN action='job.ai_analysis_start' THEN 1 ELSE 0 END) AS ai_runs,
                       SUM(CASE WHEN action='feedback.create' THEN 1 ELSE 0 END) AS feedback_messages
                FROM audit_logs
                WHERE {audit_day_expr} >= ?
                  AND action IN ({','.join('?' for _ in business_actions)})
                GROUP BY day
                """,
                (start_day, *business_actions),
            )
        ).fetchall()
        top_users_today = await (
            await db.execute(
                """
                SELECT user_token, display_name, request_count, last_seen_at, last_path
                FROM usage_daily
                WHERE day=?
                ORDER BY request_count DESC, last_seen_at DESC
                LIMIT 10
                """,
                (today,),
            )
        ).fetchall()
        seven_usage = await (
            await db.execute(
                """
                SELECT COUNT(DISTINCT user_token) AS active_users,
                       COALESCE(SUM(request_count), 0) AS requests
                FROM usage_daily
                WHERE day >= ?
                """,
                (seven_start_day,),
            )
        ).fetchone()
        total_usage = await (
            await db.execute(
                """
                SELECT COUNT(DISTINCT user_token) AS users,
                       COALESCE(SUM(request_count), 0) AS requests
                FROM usage_daily
                """
            )
        ).fetchone()
    finally:
        await db.close()

    by_day = {}
    for i in range(days):
        day = (start_date + timedelta(days=i)).isoformat()
        by_day[day] = {
            "day": day,
            "dau": 0,
            "requests": 0,
            "upload_jobs": 0,
            "compare_jobs": 0,
            "ai_runs": 0,
            "feedback_messages": 0,
            "last_seen_at": "",
        }
    for row in usage_rows:
        day = row["day"]
        if day in by_day:
            by_day[day].update({
                "dau": int(row["dau"] or 0),
                "requests": int(row["requests"] or 0),
                "last_seen_at": row["last_seen_at"] or "",
            })
    for row in audit_rows:
        day = row["day"]
        if day in by_day:
            by_day[day].update({
                "upload_jobs": int(row["upload_jobs"] or 0),
                "compare_jobs": int(row["compare_jobs"] or 0),
                "ai_runs": int(row["ai_runs"] or 0),
                "feedback_messages": int(row["feedback_messages"] or 0),
            })
    seven_rows = [item for item in by_day.values() if item["day"] >= seven_start_day]
    seven_summary = {
        "active_users": int(seven_usage["active_users"] or 0) if seven_usage else 0,
        "requests": int(seven_usage["requests"] or 0) if seven_usage else 0,
        "upload_jobs": sum(item["upload_jobs"] for item in seven_rows),
        "compare_jobs": sum(item["compare_jobs"] for item in seven_rows),
        "ai_runs": sum(item["ai_runs"] for item in seven_rows),
        "feedback_messages": sum(item["feedback_messages"] for item in seven_rows),
    }
    return {
        "timezone": USAGE_TIMEZONE_NAME or "local",
        "days": days,
        "today": dict(by_day.get(today, {"day": today})),
        "seven_days": seven_summary,
        "all_time": {
            "active_users": int(total_usage["users"] or 0) if total_usage else 0,
            "requests": int(total_usage["requests"] or 0) if total_usage else 0,
        },
        "daily": list(reversed(list(by_day.values()))),
        "top_users_today": [dict(row) for row in top_users_today],
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


def _safe_experiment_node_attachment_id(attachment_id: str) -> str:
    value = (attachment_id or "").strip()
    if not value or not EXPERIMENT_NODE_ATTACHMENT_ID_RE.match(value):
        return ""
    return value


def _experiment_node_attachment_response(jid: str, item: dict) -> dict:
    attachment_id = item.get("id") or ""
    return {
        "id": attachment_id,
        "filename": item.get("filename") or "",
        "content_type": item.get("content_type") or "",
        "size_bytes": int(item.get("size_bytes") or 0),
        "uploaded_at": item.get("uploaded_at") or "",
        "uploaded_by": item.get("uploaded_by") or "",
        "url": f"/api/jobs/{quote(jid)}/experiment-attachments/{quote(attachment_id)}",
    }


def _read_experiment_node_attachment_records(jid: str) -> list[dict]:
    meta_path = experiment_node_attachments_meta_path(jid)
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = []
    if not isinstance(data, list):
        return []

    root = experiment_node_attachments_dir(jid)
    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        attachment_id = _safe_experiment_node_attachment_id(item.get("id") or "")
        stored_name = os.path.basename(str(item.get("stored_name") or ""))
        if not attachment_id or not stored_name:
            continue
        try:
            full = _safe_child_path(root, stored_name, "Invalid experiment node attachment path")
        except HTTPException:
            continue
        if not os.path.isfile(full):
            continue
        result.append({
            **item,
            "id": attachment_id,
            "stored_name": stored_name,
            "size_bytes": _path_size(full),
        })
    return result


def _read_experiment_node_attachments(jid: str) -> list[dict]:
    return [
        _experiment_node_attachment_response(jid, item)
        for item in _read_experiment_node_attachment_records(jid)
    ]


def _write_experiment_node_attachments(jid: str, attachments: list[dict]) -> None:
    meta_path = experiment_node_attachments_meta_path(jid)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    stored = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        attachment_id = _safe_experiment_node_attachment_id(item.get("id") or "")
        stored_name = os.path.basename(str(item.get("stored_name") or ""))
        if not attachment_id or not stored_name:
            continue
        stored.append({
            "id": attachment_id,
            "filename": item.get("filename") or "",
            "stored_name": stored_name,
            "content_type": item.get("content_type") or "",
            "size_bytes": int(item.get("size_bytes") or 0),
            "uploaded_at": item.get("uploaded_at") or "",
            "uploaded_by": item.get("uploaded_by") or "",
        })
    tmp_path = f"{meta_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(stored, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, meta_path)


def _feedback_message_response(
    row: dict,
    attachments_by_message: dict,
    reactions_by_message: Optional[dict] = None,
    replies: Optional[list] = None,
    reply_count: Optional[int] = None,
    last_activity_at: Optional[str] = None,
) -> dict:
    normalized_replies = replies or []
    reactions_by_message = reactions_by_message or {}
    return {
        "id": row["id"],
        "parent_id": row.get("parent_id"),
        "user_token": row.get("user_token") or "",
        "user_display": row.get("user_display") or row.get("user_token") or "local",
        "body": row.get("body") or "",
        "created_at": row.get("created_at") or "",
        "updated_at": row.get("updated_at") or "",
        "edited_at": row.get("edited_at") or "",
        "edit_count": int(row.get("edit_count") or 0),
        "attachments": attachments_by_message.get(row["id"], []),
        "reactions": reactions_by_message.get(row["id"], []),
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


def _feedback_message_url(payload: dict) -> str:
    if not PUBLIC_BASE_URL:
        return ""
    post_id = str(payload.get("post_id") or payload.get("message_id") or "").strip()
    message_id = str(payload.get("message_id") or "").strip()
    if not post_id:
        return PUBLIC_BASE_URL
    url = f"{PUBLIC_BASE_URL}/#/feedback/{quote(post_id, safe='')}"
    if message_id and message_id != post_id:
        url += f"?message={quote(message_id, safe='')}"
    return url


def _feedback_notification_body(payload: dict) -> str:
    kind = "回复" if payload.get("parent_id") else "新帖子"
    lines = [
        f"留言板有一条{kind}",
        "",
        f"作者: {payload.get('author') or 'local'}",
        f"时间: {payload.get('created_at') or _utc_now_iso()}",
        f"帖子: {payload.get('post_title') or _feedback_title(payload.get('body', ''))}",
        "",
        "内容:",
        payload.get("body") or "(仅图片)",
    ]
    mentions = payload.get("mentioned_emails") or []
    if mentions:
        lines.extend(["", "提及:", ", ".join(mentions)])
    if PUBLIC_BASE_URL:
        lines.extend(["", f"打开留言: {_feedback_message_url(payload)}"])
    else:
        lines.extend(["", "请打开 Torch Profiler Analyzer 右上角的“灵感社区”查看详情。"])
    return "\n".join(lines)


def _feedback_email_transport() -> str:
    if SMTP_HOST:
        return "smtp"
    if SENDMAIL_COMMAND:
        return "sendmail"
    return ""


def _email_notification_status(recipients: list[str], disabled_detail: str) -> dict:
    if not FEEDBACK_EMAIL_NOTIFICATIONS_ENABLED:
        return {"status": "disabled", "detail": disabled_detail}
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


def _feedback_notification_status(recipients: list[str]) -> dict:
    return _email_notification_status(recipients, "留言板邮件通知已关闭")


def _email_error_message(exc: Exception) -> str:
    if isinstance(exc, socket.gaierror):
        host = SMTP_HOST or "未配置"
        return f"SMTP 主机无法解析: {host}。请确认 TRACE_SMTP_HOST 是 IT 提供的真实 SMTP 地址，并检查服务器 DNS。原始错误: {exc}"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return f"连接邮件服务器超时: {SMTP_HOST}:{SMTP_PORT}。请确认服务器网络和 SMTP 端口可达。"
    if isinstance(exc, ConnectionRefusedError):
        return f"邮件服务器拒绝连接: {SMTP_HOST}:{SMTP_PORT}。请确认 SMTP 端口、SSL/STARTTLS 配置是否正确。"
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return "SMTP 认证失败。请确认 TRACE_SMTP_USERNAME / TRACE_SMTP_PASSWORD 是否正确。"
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return "SMTP 拒收所有收件人。请确认收件人邮箱或 SMTP 中继策略。"
    if isinstance(exc, smtplib.SMTPSenderRefused):
        return "SMTP 拒收发件人。请确认 TRACE_SMTP_FROM 是否被允许发信。"
    if isinstance(exc, smtplib.SMTPException):
        return f"SMTP 发送失败: {exc}"
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or str(exc)
        return detail[:500]
    return str(exc)[:500]


def _email_diag_check(status: str, label: str, detail: str = "", **extra) -> dict:
    return {"status": status, "label": label, "detail": detail, **extra}


def _run_email_diagnostics_payload() -> dict:
    checks: list[dict] = []
    transport = _feedback_email_transport()
    checks.append(_email_diag_check(
        "ok" if FEEDBACK_EMAIL_NOTIFICATIONS_ENABLED else "fail",
        "邮件通知开关",
        "已启用" if FEEDBACK_EMAIL_NOTIFICATIONS_ENABLED else "已通过 TRACE_DISABLE_FEEDBACK_EMAIL 关闭",
    ))
    checks.append(_email_diag_check(
        "ok" if FEEDBACK_NOTIFICATION_ADMIN_EMAILS else "warn",
        "默认管理员收件人",
        ", ".join(FEEDBACK_NOTIFICATION_ADMIN_EMAILS) if FEEDBACK_NOTIFICATION_ADMIN_EMAILS else "未配置 TRACE_FEEDBACK_ADMIN_EMAILS",
    ))

    if SMTP_HOST:
        checks.append(_email_diag_check("ok", "邮件通道", f"SMTP {SMTP_HOST}:{SMTP_PORT}"))
        try:
            addr_infos = socket.getaddrinfo(SMTP_HOST, SMTP_PORT, type=socket.SOCK_STREAM)
            addresses = sorted({info[4][0] for info in addr_infos})
            checks.append(_email_diag_check("ok", "SMTP DNS 解析", ", ".join(addresses[:6]) or "解析成功"))
        except Exception as exc:
            checks.append(_email_diag_check("fail", "SMTP DNS 解析", _email_error_message(exc)))
            return {
                "ok": False,
                "transport": "smtp",
                "smtp_host": SMTP_HOST,
                "smtp_port": SMTP_PORT,
                "checks": checks,
            }

        try:
            with socket.create_connection((SMTP_HOST, SMTP_PORT), timeout=min(SMTP_TIMEOUT_SECONDS, 8)):
                pass
            checks.append(_email_diag_check("ok", "SMTP TCP 连通性", f"{SMTP_HOST}:{SMTP_PORT} 可连接"))
        except Exception as exc:
            checks.append(_email_diag_check("fail", "SMTP TCP 连通性", _email_error_message(exc)))
    elif SENDMAIL_COMMAND:
        exists = os.path.exists(SENDMAIL_COMMAND)
        executable = os.access(SENDMAIL_COMMAND, os.X_OK)
        checks.append(_email_diag_check(
            "ok" if exists and executable else "fail",
            "sendmail 命令",
            SENDMAIL_COMMAND if exists and executable else f"{SENDMAIL_COMMAND} 不存在或不可执行",
        ))
        checks.append(_email_diag_check(
            "warn",
            "sendmail 投递能力",
            "只能确认命令存在，仍需在服务器上确认该 sendmail 能外发到公司邮箱。",
        ))
    else:
        checks.append(_email_diag_check(
            "fail",
            "邮件通道",
            "未配置 TRACE_SMTP_HOST 或 TRACE_SENDMAIL_COMMAND",
        ))

    return {
        "ok": all(item["status"] != "fail" for item in checks),
        "transport": transport,
        "smtp_host": SMTP_HOST,
        "smtp_port": SMTP_PORT,
        "smtp_ssl": SMTP_USE_SSL,
        "smtp_starttls": SMTP_USE_STARTTLS,
        "smtp_from": SMTP_FROM,
        "sendmail_command": SENDMAIL_COMMAND,
        "checks": checks,
    }


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


def _dedupe_emails(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        email = _normalize_email(value, FEEDBACK_MENTION_DOMAIN)
        if email and email not in seen:
            seen.add(email)
            result.append(email)
    return result


def _dedupe_identity_values(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        token = str(value or "").strip()
        key = token.lower()
        if token and key not in seen:
            seen.add(key)
            result.append(token)
    return result


async def _compare_source_owner_tokens(job: dict) -> list[str]:
    if (job or {}).get("mode") != "compare":
        return []
    source_ids = _dedupe_identity_values([job.get("source_job_a"), job.get("source_job_b")])
    if not source_ids:
        return []

    placeholders = ",".join("?" * len(source_ids))
    db = await get_db()
    try:
        rows = await (
            await db.execute(
                f"SELECT user_token FROM jobs WHERE id IN ({placeholders})",
                source_ids,
            )
        ).fetchall()
    finally:
        await db.close()
    return _dedupe_identity_values([row["user_token"] for row in rows if row["user_token"]])


def _job_ai_analysis_url(jid: str) -> str:
    if not PUBLIC_BASE_URL:
        return ""
    return f"{PUBLIC_BASE_URL}/#/job/{quote(jid, safe='')}/ai"


def _job_ai_report_url(jid: str) -> str:
    if not PUBLIC_BASE_URL:
        return ""
    return f"{PUBLIC_BASE_URL}/api/jobs/{quote(jid, safe='')}/ai-analysis/report.md"


def _format_duration_ms_brief(value) -> str:
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return "-"
    if ms < 1000:
        return f"{int(ms)} ms"
    seconds = int(round(ms / 1000))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def _ai_analysis_notification_recipients(payload: dict) -> list[str]:
    values = [payload.get("trigger_user_email", ""), payload.get("owner_user_email", "")]
    values.extend(payload.get("owner_user_emails") or [])
    for key in ("trigger_user_token", "owner_user_token"):
        token = str(payload.get(key) or "").strip()
        if token and token.lower() not in {"local", "anonymous"}:
            values.append(token)
    for token in payload.get("owner_user_tokens") or []:
        token = str(token or "").strip()
        if token and token.lower() not in {"local", "anonymous"}:
            values.append(token)
    return _dedupe_emails(values)


def _ai_analysis_notification_subject(payload: dict) -> str:
    status_text = "完成" if payload.get("status") == "done" else "失败"
    label = payload.get("job_label") or payload.get("job_id") or "AI 分析"
    return f"[Torch Profiler Analyzer] AI分析{status_text}: {label}"


def _ai_analysis_notification_body(payload: dict) -> str:
    status_text = "完成" if payload.get("status") == "done" else "失败"
    mode_text = "对比" if payload.get("job_mode") == "compare" else "单 trace"
    ai_url = payload.get("ai_url") or ""
    report_url = payload.get("report_url") or ""
    owner_tokens = _dedupe_identity_values(payload.get("owner_user_tokens") or [payload.get("owner_user_token")])
    owner_text = ", ".join(owner_tokens) if owner_tokens else "local"
    lines = [
        f"AI 分析{status_text}",
        "",
        f"任务: {payload.get('job_label') or payload.get('job_id') or ''}",
        f"任务 ID: {payload.get('job_id') or ''}",
        f"类型: {mode_text}",
        f"Skill: {payload.get('skill') or ''}",
        f"触发人: {payload.get('trigger_user_display') or payload.get('trigger_user_token') or 'local'}",
        f"任务所属人: {owner_text}",
        f"状态: {payload.get('status') or ''}",
        f"开始时间: {payload.get('started_at') or ''}",
        f"结束时间: {payload.get('finished_at') or ''}",
        f"总耗时: {_format_duration_ms_brief(payload.get('duration_ms'))}",
    ]
    if payload.get("error"):
        lines.extend(["", "错误:", str(payload.get("error"))])
    if ai_url:
        lines.extend(["", f"打开 AI 分析结果: {ai_url}"])
    else:
        lines.extend(["", "打开应用的对应任务 AI 分析页查看结果；建议配置 TRACE_PUBLIC_BASE_URL 后生成邮件直达链接。"])
    if report_url and payload.get("report_exists"):
        lines.append(f"下载 Markdown 报告: {report_url}")
    if PUBLIC_BASE_URL:
        lines.append(f"打开应用: {PUBLIC_BASE_URL}")
    return "\n".join(lines)


async def _send_ai_analysis_email_notification(payload: dict) -> dict:
    recipients = _ai_analysis_notification_recipients(payload)
    status = _email_notification_status(recipients, "AI 分析邮件通知已关闭")
    if status["status"] != "ready":
        logger.warning(
            "ai_analysis_email_not_sent",
            extra={
                "event": "ai_analysis_email_not_sent",
                "job_id": payload.get("job_id"),
                **status,
            },
        )
        return status
    try:
        await asyncio.to_thread(
            _send_email_sync,
            recipients,
            _ai_analysis_notification_subject(payload),
            _ai_analysis_notification_body(payload),
        )
        sent_status = {**status, "status": "sent", "detail": "AI 分析邮件通知已发送"}
        logger.info(
            "ai_analysis_email_sent",
            extra={
                "event": "ai_analysis_email_sent",
                "job_id": payload.get("job_id"),
                "analysis_status": payload.get("status"),
                "recipients": recipients,
                "transport": status.get("transport", ""),
            },
        )
        return sent_status
    except Exception as exc:
        error = _email_error_message(exc)
        logger.warning(
            "ai_analysis_email_failed",
            extra={
                "event": "ai_analysis_email_failed",
                "job_id": payload.get("job_id"),
                "analysis_status": payload.get("status"),
                "error": error,
                "recipients": recipients,
                "transport": status.get("transport", ""),
            },
            exc_info=True,
        )
        return {
            **status,
            "status": "failed",
            "detail": f"AI 分析邮件通知发送失败: {error}",
            "error": error,
        }


async def _send_ai_analysis_completion_notification(
    jid: str,
    job: dict,
    status: dict,
    notification_context: dict,
) -> dict:
    if status.get("status") not in {"done", "error"}:
        return {"status": "skipped", "detail": "AI 分析尚未结束"}
    timing = _ai_analysis_timing(status)
    owner_tokens = _dedupe_identity_values([
        notification_context.get("owner_user_token"),
        job.get("user_token"),
        status.get("owner_user_token"),
    ])
    owner_tokens = _dedupe_identity_values([*owner_tokens, *await _compare_source_owner_tokens(job)])
    public_handle = _job_public_handle(job)
    payload = {
        "job_id": jid,
        "job_label": notification_context.get("job_label") or job.get("label") or jid,
        "job_mode": notification_context.get("job_mode") or job.get("mode") or status.get("mode") or "",
        "owner_user_token": owner_tokens[0] if owner_tokens else "",
        "owner_user_tokens": owner_tokens,
        "owner_user_email": notification_context.get("owner_user_email") or "",
        "trigger_user_token": notification_context.get("trigger_user_token") or "",
        "trigger_user_display": notification_context.get("trigger_user_display") or "",
        "trigger_user_email": notification_context.get("trigger_user_email") or "",
        "status": status.get("status") or "",
        "error": status.get("error") or "",
        "skill": status.get("skill") or "",
        "started_at": status.get("started_at") or "",
        "finished_at": status.get("finished_at") or _utc_now_iso(),
        "duration_ms": timing.get("duration_ms") or timing.get("elapsed_ms"),
        "report_exists": os.path.exists(ai_analysis_report_path(jid)),
        "ai_url": _job_ai_analysis_url(public_handle),
        "report_url": _job_ai_report_url(public_handle),
    }
    return await _send_ai_analysis_email_notification(payload)


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


async def _feedback_reactions_for(db, message_ids: list[str], viewer_token: str = "") -> dict:
    if not message_ids:
        return {}
    placeholders = ",".join("?" * len(message_ids))
    rows = await (
        await db.execute(
            f"""
            SELECT message_id,
                   emoji,
                   COUNT(*) AS count,
                   SUM(CASE WHEN user_token=? THEN 1 ELSE 0 END) AS reacted,
                   MIN(created_at) AS first_reacted_at
            FROM feedback_reactions
            WHERE message_id IN ({placeholders})
            GROUP BY message_id, emoji
            ORDER BY count DESC, first_reacted_at ASC
            """,
            (viewer_token or "", *message_ids),
        )
    ).fetchall()
    result: dict[str, list[dict]] = {}
    for row in rows:
        item = dict(row)
        result.setdefault(item["message_id"], []).append({
            "emoji": item.get("emoji") or "",
            "count": int(item.get("count") or 0),
            "reacted": bool(item.get("reacted")),
        })
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
    viewer_token, _ = _feedback_author(request)
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
                               SELECT MAX(COALESCE(r.updated_at, r.created_at))
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
        reactions_by_message = await _feedback_reactions_for(db, message_ids, viewer_token)
        for reply in replies:
            replies_by_parent.setdefault(reply["parent_id"], []).append(
                _feedback_message_response(reply, attachments_by_message, reactions_by_message)
            )
    finally:
        await db.close()

    return {
        "data": [
            _feedback_message_response(
                parent,
                attachments_by_message,
                reactions_by_message,
                replies=replies_by_parent.get(parent["id"], []),
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
    viewer_token, _ = _feedback_author(request)
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
                               SELECT MAX(COALESCE(r.updated_at, r.created_at))
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
        reactions_by_message = await _feedback_reactions_for(db, message_ids, viewer_token)
    finally:
        await db.close()

    return _feedback_message_response(
        parent,
        attachments_by_message,
        reactions_by_message,
        replies=[_feedback_message_response(reply, attachments_by_message, reactions_by_message) for reply in replies],
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

    response = _feedback_message_response(message, attachments_by_message, {})
    if notification_status:
        response["notification"] = notification_status
    return response


@app.patch("/api/feedback/{message_id}")
async def update_feedback_message(request: Request, message_id: str, payload: Optional[dict] = None):
    if AUTH_ENABLED:
        current_user_token(request)
    user_token, _ = _feedback_author(request)
    text = str((payload or {}).get("body") or "").strip()
    if len(text) > 2000:
        raise HTTPException(400, "留言内容不能超过 2000 字")

    db = await get_db()
    try:
        target = await row_to_dict(
            await (await db.execute(
                "SELECT * FROM feedback_messages WHERE id=?",
                (message_id,),
            )).fetchone()
        )
        if not target:
            raise HTTPException(404, "留言不存在")
        if (target.get("user_token") or "") != user_token and not request_is_admin(request):
            raise HTTPException(403, "只能编辑自己发布的内容")

        attachment_count_row = await (
            await db.execute(
                "SELECT COUNT(*) AS count FROM feedback_attachments WHERE message_id=?",
                (message_id,),
            )
        ).fetchone()
        if not text and int(attachment_count_row["count"] or 0) == 0:
            raise HTTPException(400, "留言内容不能为空")

        await db.execute(
            """
            UPDATE feedback_messages
            SET body=?,
                updated_at=CURRENT_TIMESTAMP,
                edited_at=CURRENT_TIMESTAMP,
                edit_count=COALESCE(edit_count, 0) + 1
            WHERE id=?
            """,
            (text, message_id),
        )
        root_parent_id = target.get("parent_id")
        if root_parent_id:
            await db.execute(
                "UPDATE feedback_messages SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (root_parent_id,),
            )
        await write_audit(
            db, request, "feedback.edit",
            resource_type="feedback", resource_id=message_id,
            details={"parent_id": root_parent_id, "body_length": len(text)},
        )
        await db.commit()

        updated = await row_to_dict(
            await (await db.execute("SELECT * FROM feedback_messages WHERE id=?", (message_id,))).fetchone()
        )
        root_id = root_parent_id or message_id
        message_ids = [message_id]
        attachments_by_message = await _feedback_attachments_for(db, message_ids)
        reactions_by_message = await _feedback_reactions_for(db, message_ids, user_token)
    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

    response = _feedback_message_response(updated, attachments_by_message, reactions_by_message)
    response["post_id"] = root_id
    return response


@app.post("/api/feedback/{message_id}/reactions")
async def toggle_feedback_reaction(request: Request, message_id: str, payload: Optional[dict] = None):
    if AUTH_ENABLED:
        current_user_token(request)
    user_token, _ = _feedback_author(request)
    emoji = str((payload or {}).get("emoji") or "").strip()
    if emoji not in FEEDBACK_REACTION_EMOJIS:
        raise HTTPException(400, "不支持该表情")

    db = await get_db()
    active = False
    try:
        target = await row_to_dict(
            await (await db.execute(
                "SELECT id FROM feedback_messages WHERE id=?",
                (message_id,),
            )).fetchone()
        )
        if not target:
            raise HTTPException(404, "留言不存在")
        existing = await row_to_dict(
            await (await db.execute(
                "SELECT 1 FROM feedback_reactions WHERE message_id=? AND user_token=? AND emoji=?",
                (message_id, user_token, emoji),
            )).fetchone()
        )
        if existing:
            await db.execute(
                "DELETE FROM feedback_reactions WHERE message_id=? AND user_token=? AND emoji=?",
                (message_id, user_token, emoji),
            )
        else:
            active = True
            await db.execute(
                "INSERT INTO feedback_reactions(message_id, user_token, emoji) VALUES(?,?,?)",
                (message_id, user_token, emoji),
            )
        await write_audit(
            db, request, "feedback.reaction",
            resource_type="feedback", resource_id=message_id,
            details={"emoji": emoji, "active": active},
        )
        await db.commit()
        reactions_by_message = await _feedback_reactions_for(db, [message_id], user_token)
    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

    return {
        "ok": True,
        "message_id": message_id,
        "emoji": emoji,
        "active": active,
        "reactions": reactions_by_message.get(message_id, []),
    }


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
            f"DELETE FROM feedback_reactions WHERE message_id IN ({placeholders})",
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
    access_sql, access_params = _project_access_clause(request, "p")
    where_sql = f"WHERE {access_sql}" if access_sql else ""
    rows = await (await db.execute(f"""
        SELECT p.*, {_project_favorite_select("p")}, po.sort_order
        FROM projects p
        LEFT JOIN project_orders po ON po.project_id = p.id AND po.user_token = ?
        {where_sql}
        ORDER BY
            CASE WHEN po.sort_order IS NULL THEN 1 ELSE 0 END,
            po.sort_order ASC,
            is_favorite DESC,
            p.is_public DESC,
            p.created_at DESC
    """, (_favorite_user_token(request), _favorite_user_token(request), *access_params))).fetchall()
    await db.close()
    return [_project_response(dict(r), request) for r in rows]


@app.post("/api/projects", status_code=201)
async def create_project(request: Request, body: dict):
    pid = str(uuid.uuid4())
    metric_targets = _parse_project_metric_targets(body.get("metric_targets")) if "metric_targets" in body else {}
    if "compute_target_ms" in body:
        compute_target = _parse_project_metric_target(body.get("compute_target_ms"), "compute_target_ms")
        if compute_target is None:
            metric_targets.pop("compute_ms", None)
        else:
            metric_targets["compute_ms"] = compute_target
    compute_target_ms = metric_targets.get("compute_ms")
    metric_targets_json = json.dumps(metric_targets, ensure_ascii=False)
    db = await get_db()
    user_token = await ensure_user_row(db, request)
    is_public = 1 if body.get("is_public") else 0
    await db.execute(
        """
        INSERT INTO projects(id, user_token, name, description, is_public, compute_target_ms, metric_targets_json)
        VALUES(?,?,?,?,?,?,?)
        """,
        (pid, user_token, body.get("name", "新项目"), body.get("description", ""), is_public, compute_target_ms, metric_targets_json),
    )
    await write_audit(
        db, request, "project.create",
        resource_type="project", resource_id=pid,
        details={"name": body.get("name", "新项目"), "is_public": is_public},
    )
    await db.commit()
    cursor = await db.execute(
        f"SELECT p.*, {_project_favorite_select('p')} FROM projects p WHERE p.id=?",
        (_favorite_user_token(request), pid),
    )
    row = await cursor.fetchone()
    await db.close()
    return _project_response(dict(row), request)


@app.put("/api/projects/{pid}")
async def update_project(request: Request, pid: str, body: dict):
    has_compute_target = "compute_target_ms" in body
    parsed_compute_target_ms = _parse_project_metric_target(body.get("compute_target_ms"), "compute_target_ms") if has_compute_target else None
    has_metric_targets = "metric_targets" in body
    parsed_metric_targets = _parse_project_metric_targets(body.get("metric_targets")) if has_metric_targets else None
    db = await get_db()

    target_only_update = set(body.keys()).issubset({"metric_targets", "compute_target_ms"})
    row = await (load_accessible_project(db, request, pid) if target_only_update else load_owned_project(db, request, pid))
    if not row:
        await db.close()
        raise HTTPException(404)

    next_name = body.get("name", row.get("name"))
    next_description = body.get("description", row.get("description", ""))
    next_is_public = 1 if body.get("is_public", row.get("is_public", 0)) else 0
    next_metric_targets = parsed_metric_targets if has_metric_targets else _project_metric_targets(row)
    if has_compute_target:
        if parsed_compute_target_ms is None:
            next_metric_targets.pop("compute_ms", None)
        else:
            next_metric_targets["compute_ms"] = parsed_compute_target_ms
    next_compute_target_ms = next_metric_targets.get("compute_ms")
    next_metric_targets_json = json.dumps(next_metric_targets, ensure_ascii=False)
    await db.execute(
        "UPDATE projects SET name=?, description=?, is_public=?, compute_target_ms=?, metric_targets_json=? WHERE id=?",
        (next_name, next_description, next_is_public, next_compute_target_ms, next_metric_targets_json, pid),
    )
    await write_audit(
        db, request, "project.update",
        resource_type="project", resource_id=pid,
        details={
            "old_name": row.get("name"),
            "new_name": next_name,
            "old_is_public": row.get("is_public", 0),
            "new_is_public": next_is_public,
            "old_compute_target_ms": row.get("compute_target_ms"),
            "new_compute_target_ms": next_compute_target_ms,
            "metric_targets": next_metric_targets,
        },
    )
    await db.commit()
    cursor = await db.execute(
        f"SELECT p.*, {_project_favorite_select('p')} FROM projects p WHERE p.id=?",
        (_favorite_user_token(request), pid),
    )
    row = await cursor.fetchone()
    await db.close()
    return _project_response(dict(row), request)


@app.put("/api/projects/{pid}/favorite")
async def update_project_favorite(request: Request, pid: str, body: dict):
    db = await get_db()
    try:
        await validate_project_access(db, request, pid)

        favorite_token = _favorite_user_token(request)
        is_favorite = 1 if body.get("is_favorite") else 0
        if is_favorite:
            await db.execute(
                "INSERT OR IGNORE INTO project_favorites(project_id, user_token) VALUES(?, ?)",
                (pid, favorite_token),
            )
        else:
            await db.execute(
                "DELETE FROM project_favorites WHERE project_id=? AND user_token=?",
                (pid, favorite_token),
            )
        await write_audit(
            db, request, "project.favorite",
            resource_type="project", resource_id=pid,
            details={"is_favorite": is_favorite},
        )
        await db.commit()
        cursor = await db.execute(
            f"SELECT p.*, {_project_favorite_select('p')} FROM projects p WHERE p.id=?",
            (favorite_token, pid),
        )
        refreshed = await cursor.fetchone()
        return _project_response(dict(refreshed), request)
    finally:
        await db.close()


@app.post("/api/projects/order")
async def update_project_order(request: Request, body: dict):
    raw_project_ids = body.get("project_ids") or []
    project_ids: list[str] = []
    for raw_id in raw_project_ids:
        project_id = str(raw_id or "").strip()
        if not project_id or project_id == "__none__" or project_id in project_ids:
            continue
        project_ids.append(project_id)

    db = await get_db()
    try:
        user_token = _favorite_user_token(request)
        for project_id in project_ids:
            await validate_project_access(db, request, project_id)

        if project_ids:
            await db.executemany(
                """
                INSERT INTO project_orders(project_id, user_token, sort_order, updated_at)
                VALUES(?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(project_id, user_token) DO UPDATE SET
                    sort_order=excluded.sort_order,
                    updated_at=CURRENT_TIMESTAMP
                """,
                [(project_id, user_token, index) for index, project_id in enumerate(project_ids)],
            )
        await write_audit(
            db, request, "project.reorder",
            resource_type="project",
            resource_id=",".join(project_ids[:10]),
            details={"project_count": len(project_ids)},
        )
        await db.commit()
    finally:
        await db.close()
    return {"updated": len(project_ids)}


@app.delete("/api/projects/{pid}", status_code=204)
async def delete_project(request: Request, pid: str):
    db = await get_db()

    row = await load_owned_project(db, request, pid)
    if not row:
        await db.close()
        raise HTTPException(404)

    # Move project info to deleted_projects table for recovery
    await db.execute("""
        INSERT INTO deleted_projects(
            id, user_token, folder_id, name, description, password_hash,
            is_public, compute_target_ms, metric_targets_json, created_at, deleted_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (row["id"], row["user_token"], row.get("folder_id"), row["name"],
          row.get("description", ""), row.get("password_hash"), row.get("is_public", 0),
          row.get("compute_target_ms"), row.get("metric_targets_json") or "{}",
          row.get("created_at")))

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
                source_job_a, source_job_b, step_filter_a, step_filter_b,
                save_triton_csv, save_triton_code,
                status, console_out, error_msg, result_dir, deleted_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (job_dict["id"], job_dict.get("project_id"), job_dict.get("user_token"),
              job_dict.get("created_at"), job_dict.get("label", ""), job_dict.get("mode"),
              job_dict.get("is_pinned", 0),
              job_dict.get("file_a_name"), job_dict.get("file_a_path"), job_dict.get("file_a_gzip_path"), job_dict.get("file_a_exists", 1),
              job_dict.get("file_b_name"), job_dict.get("file_b_path"), job_dict.get("file_b_gzip_path"), job_dict.get("file_b_exists", 1),
              job_dict.get("source_job_a"), job_dict.get("source_job_b"),
              job_dict.get("step_filter_a", ""), job_dict.get("step_filter_b", ""),
              job_dict.get("save_triton_csv", 0), job_dict.get("save_triton_code", 0),
              job_dict.get("status", "pending"), job_dict.get("console_out", ""), job_dict.get("error_msg", ""), job_dict.get("result_dir", "")))

    await db.execute("DELETE FROM experiment_edges WHERE project_id=?", (pid,))
    await db.execute("DELETE FROM experiment_node_layout WHERE project_id=?", (pid,))

    # Delete jobs from main table
    await db.execute("DELETE FROM jobs WHERE project_id=?", (pid,))

    # Remove personal shortcuts to this project.
    await db.execute("DELETE FROM project_favorites WHERE project_id=?", (pid,))

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
                INSERT INTO projects(
                    id, user_token, folder_id, name, description, password_hash,
                    is_public, compute_target_ms, metric_targets_json, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (row["id"], row["user_token"], row.get("folder_id"), row["name"],
                  row.get("description", ""), row.get("password_hash"), row.get("is_public", 0),
                  row.get("compute_target_ms"), row.get("metric_targets_json") or "{}"))
        else:
            await db.execute("""
                INSERT INTO projects(
                    id, user_token, folder_id, name, description, password_hash,
                    is_public, compute_target_ms, metric_targets_json, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (row["id"], row["user_token"], row.get("folder_id"), row["name"],
                  row.get("description", ""), row.get("password_hash"), row.get("is_public", 0),
                  row.get("compute_target_ms"), row.get("metric_targets_json") or "{}",
                  created_at))
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
                source_job_a, source_job_b, step_filter_a, step_filter_b,
                save_triton_csv, save_triton_code,
                status, console_out, error_msg, result_dir)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_dict["id"], pid, job_dict.get("user_token"),
              job_dict.get("created_at"), job_dict.get("label", ""), job_dict.get("mode"),
              job_dict.get("is_pinned", 0),
              job_dict.get("file_a_name"), job_dict.get("file_a_path"), job_dict.get("file_a_gzip_path"), job_dict.get("file_a_exists", 1),
              job_dict.get("file_b_name"), job_dict.get("file_b_path"), job_dict.get("file_b_gzip_path"), job_dict.get("file_b_exists", 1),
              job_dict.get("source_job_a"), job_dict.get("source_job_b"),
              job_dict.get("step_filter_a", ""), job_dict.get("step_filter_b", ""),
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
    project_view: Optional[str] = None,
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
    if project_id == "__none__":
        clauses.append("j.project_id IS NULL")
    elif project_id:
        await validate_project_access(db, request, project_id)
        clauses.append("j.project_id = ?")
        params.append(project_id)
    else:
        view_sql, view_params = _project_view_clause(request, project_view, "p")
        if view_sql:
            clauses.append(view_sql)
            params.extend(view_params)
    search_sql, search_params = _job_search_clause("j", q, include_project_name=True)
    if search_sql:
        clauses.append(search_sql)
        params.extend(search_params)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    count_cursor = await db.execute(
        f"""
        SELECT COUNT(*) as total
        FROM jobs j
        LEFT JOIN projects p ON p.id = j.project_id
        {where_sql}
        """,
        params,
    )
    total = (await count_cursor.fetchone())[0]

    rows = await (await db.execute(f"""
            SELECT j.*, p.name AS project_name
            FROM jobs j
            LEFT JOIN projects p ON p.id = j.project_id
            {where_sql}
            ORDER BY COALESCE(j.is_pinned, 0) DESC, j.created_at DESC
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


def _ai_analysis_is_active(jid: str) -> bool:
    if jid in ai_analysis_tasks or jid in ai_analysis_queued_jobs:
        return True
    return _read_ai_analysis_status(jid).get("status") in {"queued", "running"}


def _job_delete_locked(row: dict) -> bool:
    return row.get("status") == "running" or _ai_analysis_is_active(row["id"])


def _job_files_locked(row: dict) -> bool:
    return row.get("status") in {"pending", "running"} or _ai_analysis_is_active(row["id"])


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
        jobs = await _load_jobs_by_ids(db, job_ids, request)
        target_project_id = body.get("project_id") or None
        await validate_project_access(db, request, target_project_id)
        changed_job_ids = [job["id"] for job in jobs if job.get("project_id") != target_project_id]
        await _clear_experiment_job_links(db, changed_job_ids)
        placeholders = ",".join("?" * len(job_ids))
        await db.execute(
            f"UPDATE jobs SET project_id=? WHERE id IN ({placeholders})",
            (target_project_id, *job_ids),
        )
        await write_audit(
            db, request, "job.bulk_move",
            resource_type="job", resource_id=",".join(job_ids[:10]),
            details={"job_count": len(job_ids), "project_id": target_project_id},
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
        jobs = await _load_jobs_by_ids(db, job_ids, request)
        busy = [job["id"] for job in jobs if _job_delete_locked(job)]
        if busy:
            raise HTTPException(409, f"任务仍在分析中，暂不能删除: {', '.join(busy[:5])}")
        for job_id in job_ids:
            _remove_job_dir(job_id)
        placeholders = ",".join("?" * len(job_ids))
        await _clear_experiment_job_links(db, job_ids)
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
        busy = [job["id"] for job in jobs if _job_files_locked(job)]
        if busy:
            raise HTTPException(409, f"任务仍在分析中，暂不能删除文件: {', '.join(busy[:5])}")
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
    project_view: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    db = await get_db()

    q_text = (q or "").strip()
    like = f"%{q_text.lower()}%"
    active_project_view = (project_view or "all").strip().lower()
    if active_project_view not in {"all", "favorite", "mine", "shared"}:
        active_project_view = "all"

    groups: list[dict] = []

    if project_id != "__none__":
        if project_id:
            await validate_project_access(db, request, project_id)

        project_clauses = []
        project_params: list = []
        project_access_sql, project_access_params = _project_access_clause(request, "p")
        if project_access_sql:
            project_clauses.append(project_access_sql)
            project_params.extend(project_access_params)
        if project_id:
            project_clauses.append("p.id = ?")
            project_params.append(project_id)
        elif active_project_view != "all":
            view_sql, view_params = _project_view_clause(request, active_project_view, "p")
            if view_sql:
                project_clauses.append(view_sql)
                project_params.extend(view_params)

        job_count_clauses = ["j.project_id = p.id"]
        job_count_params: list = []
        job_access_sql, job_access_params = _job_access_clause(request, "j")
        if job_access_sql:
            job_count_clauses.append(job_access_sql)
            job_count_params.extend(job_access_params)
        if q_text:
            job_count_clauses.append(
                "("
                "LOWER(COALESCE(j.label, '')) LIKE ? OR "
                "LOWER(COALESCE(j.file_a_name, '')) LIKE ? OR "
                "LOWER(COALESCE(j.file_b_name, '')) LIKE ? OR "
                "LOWER(COALESCE(p.name, '')) LIKE ?"
                ")"
            )
            job_count_params.extend([like, like, like, like])
            project_clauses.append(
                "("
                "LOWER(COALESCE(p.name, '')) LIKE ? OR EXISTS ("
                "SELECT 1 FROM jobs j "
                "WHERE j.project_id = p.id"
                f"{' AND ' + job_access_sql if job_access_sql else ''}"
                " AND ("
                "LOWER(COALESCE(j.label, '')) LIKE ? OR "
                "LOWER(COALESCE(j.file_a_name, '')) LIKE ? OR "
                "LOWER(COALESCE(j.file_b_name, '')) LIKE ?"
                ")"
                ")"
                ")"
            )
            project_params.append(like)
            project_params.extend(job_access_params)
            project_params.extend([like, like, like])

        project_where_sql = f"WHERE {' AND '.join(project_clauses)}" if project_clauses else ""
        job_count_where_sql = " AND ".join(job_count_clauses)

        project_rows = await (
            await db.execute(
                f"""
                SELECT
                    p.id,
                    p.name AS label,
                    p.user_token,
                    p.is_public,
                    po.sort_order,
                    {_project_favorite_select("p")},
                    EXISTS (
                        SELECT 1
                        FROM experiment_edges ee
                        WHERE ee.project_id = p.id
                        LIMIT 1
                    ) AS has_experiment_tree,
                    (
                        SELECT COUNT(*)
                        FROM jobs j
                        WHERE {job_count_where_sql}
                    ) AS job_count
                FROM projects p
                LEFT JOIN project_orders po ON po.project_id = p.id AND po.user_token = ?
                {project_where_sql}
                ORDER BY
                    CASE WHEN po.sort_order IS NULL THEN 1 ELSE 0 END,
                    po.sort_order ASC,
                    is_favorite DESC,
                    p.name COLLATE NOCASE
                """,
                (_favorite_user_token(request), *job_count_params, _favorite_user_token(request), *project_params),
            )
        ).fetchall()
        groups.extend(
            {
                "id": row["id"],
                "label": row["label"],
                "job_count": row["job_count"],
                "sort_order": row["sort_order"],
                "is_public": 1 if row["is_public"] else 0,
                "is_owner": True if not current_user_token(request) else row["user_token"] == current_user_token(request),
                "is_favorite": 1 if row["is_favorite"] else 0,
                "has_experiment_tree": 1 if row["has_experiment_tree"] else 0,
                "visibility": "shared" if row["is_public"] else "personal",
            }
            for row in project_rows
        )

    if (not project_id or project_id == "__none__") and (project_id == "__none__" or active_project_view == "all"):
        ungrouped_clauses = ["j.project_id IS NULL"]
        ungrouped_params: list = []
        access_sql, access_params = _job_access_clause(request, "j")
        if access_sql:
            ungrouped_clauses.append(access_sql)
            ungrouped_params.extend(access_params)
        search_sql, search_params = _job_search_clause("j", q_text, include_project_name=False)
        if search_sql:
            ungrouped_clauses.append(search_sql)
            ungrouped_params.extend(search_params)
        ungrouped_where_sql = " AND ".join(ungrouped_clauses)
        cursor = await db.execute(
            f"SELECT COUNT(*) FROM jobs j WHERE {ungrouped_where_sql}",
            ungrouped_params,
        )
        ungrouped_count = (await cursor.fetchone())[0]
        if ungrouped_count:
            groups.append({"id": "__none__", "label": "未分组", "job_count": ungrouped_count})

    await db.close()

    total = len(groups)
    data = groups[offset:offset + limit]

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
    row = None
    db_row_committed = False

    try:
        async with upload_semaphore:
            _ensure_storage_capacity()

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
            try:
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
                db_row_committed = True
                cursor = await db.execute("SELECT * FROM jobs WHERE id=?", (jid,))
                row = await cursor.fetchone()
            except Exception:
                await db.rollback()
                raise
            finally:
                await db.close()
    except Exception:
        if row is None:
            if db_row_committed:
                with contextlib.suppress(Exception):
                    cleanup_db = await get_db()
                    try:
                        await cleanup_db.execute("DELETE FROM jobs WHERE id=?", (jid,))
                        await cleanup_db.commit()
                    finally:
                        await cleanup_db.close()
            with contextlib.suppress(Exception):
                _remove_job_dir(jid)
        raise

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
    job_id_a = src_a["id"]
    job_id_b = src_b["id"]
    if job_id_a == job_id_b:
        await db.close()
        raise HTTPException(400, "不能自比")

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


@app.get("/api/projects/{pid}/experiments")
async def get_project_experiments(request: Request, pid: str):
    db = await get_db()
    try:
        project = await load_accessible_project(db, request, pid)
        if not project:
            raise HTTPException(404, "Project not found")

        edge_rows = [
            dict(row)
            for row in await (
                await db.execute(
                    "SELECT * FROM experiment_edges WHERE project_id=? ORDER BY created_at, id",
                    (pid,),
                )
            ).fetchall()
        ]
        for edge in edge_rows:
            perf = _parse_json_field(edge.get("perf_summary"), None)
            if isinstance(perf, dict) and perf.get("incomplete") and not edge.get("perf_summary_edited"):
                refreshed = await compute_perf_delta_async(edge["parent_job_id"], edge["child_job_id"])
                refreshed_json = json.dumps(refreshed, ensure_ascii=False)
                if refreshed_json != edge.get("perf_summary"):
                    edge["perf_summary"] = refreshed_json
                    await db.execute(
                        "UPDATE experiment_edges SET perf_summary=? WHERE id=?",
                        (edge["perf_summary"], edge["id"]),
                    )

        layout_rows = await (
            await db.execute(
                "SELECT job_id, x, y, scale, width, height, pinned FROM experiment_node_layout WHERE project_id=?",
                (pid,),
            )
        ).fetchall()
        layout_by_job = {row["job_id"]: dict(row) for row in layout_rows}

        node_ids = []
        for edge in edge_rows:
            for job_id in (edge.get("parent_job_id"), edge.get("child_job_id")):
                if job_id and job_id not in node_ids:
                    node_ids.append(job_id)

        nodes = []
        kernel_durations_by_job = {}
        if node_ids:
            placeholders = ",".join("?" * len(node_ids))
            clauses = ["j.project_id=?", f"j.id IN ({placeholders})"]
            params = [pid, *node_ids]
            access_sql, access_params = _job_access_clause(request, "j")
            if access_sql:
                clauses.append(access_sql)
                params.extend(access_params)
            rows = await (
                await db.execute(
                    f"""
                    SELECT j.id, j.seq, j.label, j.file_a_name, j.created_at, j.status, j.mode, j.console_out
                    FROM jobs j
                    WHERE {' AND '.join(clauses)}
                    ORDER BY j.created_at, j.id
                    """,
                    params,
                )
            ).fetchall()
            for row in rows:
                item = dict(row)
                layout = layout_by_job.get(item["id"]) or {}
                metrics = await _read_node_metrics_async(item["id"], item.get("console_out"))
                item.pop("console_out", None)
                item["e2e_ms"] = metrics.get("e2e_ms")
                item["step_dur_ms"] = metrics.get("step_dur_ms")
                item["compute_ms"] = metrics.get("compute_ms")
                item["comm_ms"] = metrics.get("comm_ms")
                item["kernel_count"] = metrics.get("kernel_count")
                item["aten_ops_count"] = metrics.get("aten_ops_count")
                item["aten_ops_ms"] = metrics.get("aten_ops_ms")
                item["top_kernels"] = metrics.get("top_kernels") or []
                item["attachments"] = _read_experiment_node_attachments(item["id"])
                kernel_durations_by_job[item["id"]] = metrics.get("kernel_durations") or {}
                item["x"] = layout.get("x")
                item["y"] = layout.get("y")
                item["scale"] = layout.get("scale") if layout.get("scale") is not None else 1.0
                item["width"] = layout.get("width")
                item["height"] = layout.get("height")
                item["pinned"] = 1 if layout.get("pinned") else 0 if layout else None
                nodes.append(item)

        first_parent_by_child = {}
        for edge in edge_rows:
            first_parent_by_child.setdefault(edge.get("child_job_id"), edge.get("parent_job_id"))
        for item in nodes:
            parent_id = first_parent_by_child.get(item["id"])
            if not parent_id:
                continue
            parent_kernels = kernel_durations_by_job.get(parent_id) or {}
            enriched = []
            for kernel in item.get("top_kernels") or []:
                child_ms = _float_or_none(kernel.get("avg_dur_ms"))
                parent_ms = _float_or_none(parent_kernels.get(kernel.get("name"))) or 0.0
                enriched.append({
                    **kernel,
                    "parent_delta_ms": _round_ms(child_ms - parent_ms) if child_ms is not None else None,
                })
            item["top_kernels"] = enriched

        unconnected_clauses = [
            "j.project_id=?",
            "j.mode='single'",
            """
            NOT EXISTS (
                SELECT 1 FROM experiment_edges ee
                WHERE ee.project_id=? AND (ee.parent_job_id=j.id OR ee.child_job_id=j.id)
            )
            """,
        ]
        unconnected_params = [pid, pid]
        access_sql, access_params = _job_access_clause(request, "j")
        if access_sql:
            unconnected_clauses.append(access_sql)
            unconnected_params.extend(access_params)
        unconnected_rows = await (
            await db.execute(
                f"""
                SELECT j.id, j.seq, j.label, j.file_a_name, j.created_at, j.status, j.mode, j.console_out
                FROM jobs j
                WHERE {' AND '.join(unconnected_clauses)}
                ORDER BY j.created_at DESC, j.id
                """,
                unconnected_params,
            )
        ).fetchall()
        unconnected = []
        for row in unconnected_rows:
            item = dict(row)
            metrics = await _read_node_metrics_async(item["id"], item.get("console_out"))
            item.pop("console_out", None)
            item["e2e_ms"] = metrics.get("e2e_ms")
            item["step_dur_ms"] = metrics.get("step_dur_ms")
            item["compute_ms"] = metrics.get("compute_ms")
            item["comm_ms"] = metrics.get("comm_ms")
            item["kernel_count"] = metrics.get("kernel_count")
            item["aten_ops_count"] = metrics.get("aten_ops_count")
            item["aten_ops_ms"] = metrics.get("aten_ops_ms")
            item["top_kernels"] = metrics.get("top_kernels") or []
            item["attachments"] = _read_experiment_node_attachments(item["id"])
            unconnected.append(item)

        await db.commit()
        return {
            "project_id": pid,
            "project": _project_response(project, request),
            "nodes": nodes,
            "unconnected": unconnected,
            "edges": [_experiment_edge_response(edge) for edge in edge_rows],
        }
    finally:
        await db.close()


@app.post("/api/jobs/{jid}/experiment-attachments")
async def upload_experiment_node_attachment(
    request: Request,
    jid: str,
    file: UploadFile = File(...),
):
    db = await get_db()
    try:
        row = await load_accessible_job(db, request, jid, "id, label, status, user_token")
    finally:
        await db.close()
    if not row:
        raise HTTPException(404)
    jid = row["id"]

    filename = _safe_download_name(file.filename, "attachment")[:240]
    extension = re.sub(r"[^A-Za-z0-9.]+", "", os.path.splitext(filename)[1])[:24]
    attachment_id = str(uuid.uuid4())
    stored_name = f"{attachment_id}{extension}"
    target_dir = experiment_node_attachments_dir(jid)
    target_path = _safe_child_path(target_dir, stored_name, "Invalid experiment node attachment path")
    os.makedirs(target_dir, exist_ok=True)
    written = 0
    try:
        async with aiofiles.open(target_path, "wb") as f:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > EXPERIMENT_NODE_ATTACHMENT_MAX_BYTES:
                    raise HTTPException(413, f"附件不能超过 {EXPERIMENT_NODE_ATTACHMENT_MAX_BYTES // 1024 // 1024}MB")
                await f.write(chunk)
    except HTTPException:
        with contextlib.suppress(OSError):
            os.remove(target_path)
        raise
    if written <= 0:
        with contextlib.suppress(OSError):
            os.remove(target_path)
        raise HTTPException(400, "附件内容为空")

    user = request_user(request)
    records = _read_experiment_node_attachment_records(jid)
    records.append({
        "id": attachment_id,
        "filename": filename,
        "stored_name": stored_name,
        "content_type": file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
        "size_bytes": written,
        "uploaded_at": _utc_now_iso(),
        "uploaded_by": user,
    })
    _write_experiment_node_attachments(jid, records)

    audit_db = await get_db()
    try:
        await write_audit(
            audit_db, request, "job.experiment_node_attachment_upload",
            resource_type="job", resource_id=jid,
            details={"attachment_id": attachment_id, "filename": filename, "size": written},
        )
        await audit_db.commit()
    finally:
        await audit_db.close()

    return {"attachments": _read_experiment_node_attachments(jid)}


@app.get("/api/jobs/{jid}/experiment-attachments/{attachment_id}")
async def download_experiment_node_attachment(request: Request, jid: str, attachment_id: str):
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")

    db = await get_db()
    try:
        row = await load_accessible_job(db, request, jid, "id, label, status")
    finally:
        await db.close()
    if not row:
        raise HTTPException(404)
    jid = row["id"]

    safe_attachment_id = _safe_experiment_node_attachment_id(attachment_id)
    if not safe_attachment_id:
        raise HTTPException(404, "Experiment node attachment not found")
    record = next(
        (item for item in _read_experiment_node_attachment_records(jid) if item.get("id") == safe_attachment_id),
        None,
    )
    if not record:
        raise HTTPException(404, "Experiment node attachment not found")
    full_path = _safe_child_path(
        experiment_node_attachments_dir(jid),
        record["stored_name"],
        "Invalid experiment node attachment path",
    )
    if not os.path.isfile(full_path):
        raise HTTPException(404, "Experiment node attachment not found")

    audit_db = await get_db()
    try:
        await write_audit(
            audit_db, request, "job.experiment_node_attachment_download",
            resource_type="job", resource_id=jid,
            details={"attachment_id": safe_attachment_id, "size": _path_size(full_path)},
        )
        await audit_db.commit()
    finally:
        await audit_db.close()

    filename = _safe_download_name(record.get("filename"), os.path.basename(full_path))
    media_type = record.get("content_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(
        full_path,
        media_type=media_type,
        headers={"Content-Disposition": _content_disposition(filename)},
    )


@app.delete("/api/jobs/{jid}/experiment-attachments/{attachment_id}", status_code=204)
async def delete_experiment_node_attachment(request: Request, jid: str, attachment_id: str):
    db = await get_db()
    try:
        row = await load_accessible_job(db, request, jid, "id, label, status, user_token")
    finally:
        await db.close()
    if not row:
        raise HTTPException(404)
    jid = row["id"]

    safe_attachment_id = _safe_experiment_node_attachment_id(attachment_id)
    if not safe_attachment_id:
        raise HTTPException(404, "Experiment node attachment not found")
    records = _read_experiment_node_attachment_records(jid)
    record = next((item for item in records if item.get("id") == safe_attachment_id), None)
    if not record:
        raise HTTPException(404, "Experiment node attachment not found")

    actor = request_user(request)
    can_delete = (
        not AUTH_ENABLED
        or request_is_admin(request)
        or actor == (row.get("user_token") or "")
        or actor == (record.get("uploaded_by") or "")
    )
    if not can_delete:
        raise HTTPException(403, "No permission to delete this attachment")

    full_path = _safe_child_path(
        experiment_node_attachments_dir(jid),
        record["stored_name"],
        "Invalid experiment node attachment path",
    )
    if os.path.isfile(full_path):
        with contextlib.suppress(OSError):
            os.remove(full_path)
    _write_experiment_node_attachments(
        jid,
        [item for item in records if item.get("id") != safe_attachment_id],
    )

    audit_db = await get_db()
    try:
        await write_audit(
            audit_db, request, "job.experiment_node_attachment_delete",
            resource_type="job", resource_id=jid,
            details={"attachment_id": safe_attachment_id},
        )
        await audit_db.commit()
    finally:
        await audit_db.close()

    return None


@app.post("/api/projects/{pid}/experiments/edges", status_code=201)
async def create_experiment_edge(request: Request, pid: str, body: dict):
    parent_job_id = (body.get("parent_job_id") or "").strip()
    child_job_id = (body.get("child_job_id") or "").strip()
    if not parent_job_id or not child_job_id:
        raise HTTPException(400, "parent_job_id and child_job_id are required")
    if parent_job_id == child_job_id:
        raise HTTPException(400, "不能自连")

    variables = _normalize_experiment_variables(body.get("variables") or [])
    db = await get_db()
    try:
        user_token = await ensure_user_row(db, request)
        await validate_project_access(db, request, pid)
        parent = await load_accessible_job(db, request, parent_job_id, "id, project_id, mode, console_out")
        child = await load_accessible_job(db, request, child_job_id, "id, project_id, mode, console_out")
        if not parent or not child:
            raise HTTPException(404, "节点不存在")
        parent_job_id = parent["id"]
        child_job_id = child["id"]
        if parent_job_id == child_job_id:
            raise HTTPException(400, "不能自连")
        if (
            parent.get("project_id") != pid
            or child.get("project_id") != pid
            or parent.get("mode") != "single"
            or child.get("mode") != "single"
        ):
            raise HTTPException(400, "节点必须是本项目的单文件任务")

        existing = await (
            await db.execute(
                """
                SELECT 1 FROM experiment_edges
                WHERE project_id=? AND parent_job_id=? AND child_job_id=?
                """,
                (pid, parent_job_id, child_job_id),
            )
        ).fetchone()
        if existing:
            raise HTTPException(409, "关系已存在")

        edge_pairs = [
            dict(row)
            for row in await (
                await db.execute(
                    "SELECT parent_job_id, child_job_id FROM experiment_edges WHERE project_id=?",
                    (pid,),
                )
            ).fetchall()
        ]
        if _would_create_cycle(edge_pairs, parent_job_id, child_job_id):
            raise HTTPException(409, "会形成循环依赖")

        edge_id = str(uuid.uuid4())
        perf = await compute_perf_delta_async(
            parent_job_id,
            child_job_id,
            parent.get("console_out"),
            child.get("console_out"),
        )
        await db.execute(
            """
            INSERT INTO experiment_edges(
                id, project_id, user_token, parent_job_id, child_job_id,
                title, description, perf_summary, perf_summary_edited, variables
            )
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                edge_id,
                pid,
                user_token,
                parent_job_id,
                child_job_id,
                str(body.get("title") or ""),
                str(body.get("description") or ""),
                json.dumps(perf, ensure_ascii=False),
                0,
                json.dumps(variables, ensure_ascii=False),
            ),
        )
        await write_audit(
            db, request, "experiment.edge_create",
            resource_type="experiment_edge", resource_id=edge_id,
            details={"project_id": pid, "parent": parent_job_id, "child": child_job_id},
        )
        await db.commit()
        row = await _load_experiment_edge(db, edge_id)
        return _experiment_edge_response(row)
    finally:
        await db.close()


@app.patch("/api/experiments/edges/{eid}")
async def patch_experiment_edge(request: Request, eid: str, body: dict):
    db = await get_db()
    try:
        edge = await _load_experiment_edge(db, eid)
        if not edge:
            raise HTTPException(404, "关系不存在")
        await validate_project_access(db, request, edge.get("project_id"))

        updates = []
        params = []
        if "title" in body:
            updates.append("title=?")
            params.append(str(body.get("title") or ""))
        if "description" in body:
            updates.append("description=?")
            params.append(str(body.get("description") or ""))
        if "variables" in body:
            variables = _normalize_experiment_variables(body.get("variables"))
            updates.append("variables=?")
            params.append(json.dumps(variables, ensure_ascii=False))
        if "label_layout" in body:
            label_layout = _normalize_edge_label_layout(body.get("label_layout"))
            for key, value in label_layout.items():
                updates.append(f"{key}=?")
                params.append(value)
        if "perf_summary" in body:
            perf_summary = body.get("perf_summary")
            if not isinstance(perf_summary, dict):
                raise HTTPException(400, "perf_summary must be an object")
            updates.append("perf_summary=?")
            params.append(json.dumps(perf_summary, ensure_ascii=False))
            updates.append("perf_summary_edited=1")
        if "compare_job_id" in body:
            compare_job_id = (body.get("compare_job_id") or "").strip()
            if compare_job_id:
                compare_job = await load_accessible_job(db, request, compare_job_id, "id, mode")
                if not compare_job:
                    raise HTTPException(404, "对比任务不存在")
                if compare_job.get("mode") != "compare":
                    raise HTTPException(400, "compare_job_id 必须指向对比任务")
                updates.append("compare_job_id=?")
                params.append(compare_job_id)
            else:
                updates.append("compare_job_id=NULL")

        if updates:
            await db.execute(
                f"UPDATE experiment_edges SET {', '.join(updates)} WHERE id=?",
                (*params, eid),
            )
        await write_audit(
            db, request, "experiment.edge_update",
            resource_type="experiment_edge", resource_id=eid,
            details={"fields": list(body.keys())},
        )
        await db.commit()
        row = await _load_experiment_edge(db, eid)
        return _experiment_edge_response(row)
    finally:
        await db.close()


@app.delete("/api/experiments/edges/{eid}", status_code=204)
async def delete_experiment_edge(request: Request, eid: str):
    db = await get_db()
    try:
        edge = await _load_experiment_edge(db, eid)
        if not edge:
            raise HTTPException(404, "关系不存在")
        await validate_project_access(db, request, edge.get("project_id"))
        await db.execute("DELETE FROM experiment_edges WHERE id=?", (eid,))
        await write_audit(
            db, request, "experiment.edge_delete",
            resource_type="experiment_edge", resource_id=eid,
            details={
                "project_id": edge.get("project_id"),
                "parent": edge.get("parent_job_id"),
                "child": edge.get("child_job_id"),
            },
        )
        await db.commit()
    finally:
        await db.close()


@app.post("/api/experiments/edges/{eid}/refresh-perf")
async def refresh_experiment_edge_perf(request: Request, eid: str):
    db = await get_db()
    try:
        edge = await _load_experiment_edge(db, eid)
        if not edge:
            raise HTTPException(404, "关系不存在")
        await validate_project_access(db, request, edge.get("project_id"))
        if edge.get("perf_summary_edited"):
            response = _experiment_edge_response(edge)
            response["skipped"] = True
            return response
        perf = await compute_perf_delta_async(edge["parent_job_id"], edge["child_job_id"])
        await db.execute(
            "UPDATE experiment_edges SET perf_summary=? WHERE id=?",
            (json.dumps(perf, ensure_ascii=False), eid),
        )
        await write_audit(
            db, request, "experiment.edge_refresh_perf",
            resource_type="experiment_edge", resource_id=eid,
            details={"project_id": edge.get("project_id")},
        )
        await db.commit()
        row = await _load_experiment_edge(db, eid)
        response = _experiment_edge_response(row)
        response["skipped"] = False
        return response
    finally:
        await db.close()


@app.post("/api/experiments/edges/{eid}/compare", status_code=201)
async def create_experiment_edge_compare(request: Request, eid: str):
    db = await get_db()
    try:
        user_token = await ensure_user_row(db, request)
        edge = await _load_experiment_edge(db, eid)
        if not edge:
            raise HTTPException(404, "关系不存在")
        await validate_project_access(db, request, edge.get("project_id"))
        if edge.get("compare_job_id"):
            return {"compare_job_id": edge["compare_job_id"]}

        parent = await load_accessible_job(db, request, edge["parent_job_id"])
        child = await load_accessible_job(db, request, edge["child_job_id"])
        if not parent or not child:
            raise HTTPException(404, "源任务不存在")
        if not _trace_path_exists(parent):
            raise HTTPException(409, "父节点源文件已被删除")
        if not _trace_path_exists(child):
            raise HTTPException(409, "子节点源文件已被删除")

        compare_job_id = str(uuid.uuid4())
        label = f"{parent.get('label') or parent.get('file_a_name') or edge['parent_job_id']} vs {child.get('label') or child.get('file_a_name') or edge['child_job_id']}"
        await db.execute(
            """INSERT INTO jobs(id, project_id, user_token, label, mode,
                   file_a_name, file_b_name, source_job_a, source_job_b,
                   save_triton_csv, save_triton_code)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                compare_job_id,
                edge.get("project_id"),
                user_token,
                label,
                "compare",
                parent.get("file_a_name"),
                child.get("file_a_name"),
                edge["parent_job_id"],
                edge["child_job_id"],
                0,
                0,
            ),
        )
        await _refresh_job_storage_cache(db, compare_job_id)
        await db.execute(
            "UPDATE experiment_edges SET compare_job_id=? WHERE id=?",
            (compare_job_id, eid),
        )
        await write_audit(
            db, request, "experiment.edge_compare_create",
            resource_type="experiment_edge", resource_id=eid,
            details={"compare_job_id": compare_job_id, "project_id": edge.get("project_id")},
        )
        await db.commit()
    finally:
        await db.close()

    await enqueue_analysis_job(compare_job_id)
    return {"compare_job_id": compare_job_id}


@app.post("/api/projects/{pid}/experiments/compare", status_code=201)
async def create_experiment_node_compare(request: Request, pid: str, body: dict):
    job_id_a = (body.get("job_id_a") or "").strip()
    job_id_b = (body.get("job_id_b") or "").strip()
    if not job_id_a or not job_id_b:
        raise HTTPException(400, "job_id_a and job_id_b are required")
    if job_id_a == job_id_b:
        raise HTTPException(400, "不能自比")

    compare_job_id = ""
    db = await get_db()
    try:
        user_token = await ensure_user_row(db, request)
        await validate_project_access(db, request, pid)
        src_a = await load_accessible_job(db, request, job_id_a)
        src_b = await load_accessible_job(db, request, job_id_b)
        if not src_a or not src_b:
            raise HTTPException(404, "Source job not found")
        job_id_a = src_a["id"]
        job_id_b = src_b["id"]
        if (
            src_a.get("project_id") != pid
            or src_b.get("project_id") != pid
            or src_a.get("mode") != "single"
            or src_b.get("mode") != "single"
        ):
            raise HTTPException(400, "节点必须是本项目的单文件任务")
        if not _trace_path_exists(src_a):
            raise HTTPException(409, "节点 A 源文件已被删除")
        if not _trace_path_exists(src_b):
            raise HTTPException(409, "节点 B 源文件已被删除")

        clauses = [
            "j.mode='compare'",
            "j.project_id=?",
            "j.source_job_a=?",
            "j.source_job_b=?",
        ]
        params = [pid, job_id_a, job_id_b]
        access_sql, access_params = _job_access_clause(request, "j")
        if access_sql:
            clauses.append(access_sql)
            params.extend(access_params)
        existing = await row_to_dict(
            await (
                await db.execute(
                    f"""
                    SELECT j.id, j.seq
                    FROM jobs j
                    WHERE {' AND '.join(clauses)}
                    ORDER BY j.created_at DESC, j.id DESC
                    LIMIT 1
                    """,
                    params,
                )
            ).fetchone()
        )
        if existing:
            await db.commit()
            return {"compare_job_id": existing["id"], "compare_job_seq": existing.get("seq"), "existing": True}

        compare_job_id = str(uuid.uuid4())
        label = (
            f"{src_a.get('label') or src_a.get('file_a_name') or job_id_a}"
            f" vs {src_b.get('label') or src_b.get('file_a_name') or job_id_b}"
        )
        await db.execute(
            """INSERT INTO jobs(id, project_id, user_token, label, mode,
                   file_a_name, file_b_name, source_job_a, source_job_b,
                   save_triton_csv, save_triton_code)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                compare_job_id,
                pid,
                user_token,
                label,
                "compare",
                src_a.get("file_a_name"),
                src_b.get("file_a_name"),
                job_id_a,
                job_id_b,
                0,
                0,
            ),
        )
        await _refresh_job_storage_cache(db, compare_job_id)
        await write_audit(
            db, request, "experiment.node_compare_create",
            resource_type="job", resource_id=compare_job_id,
            details={"project_id": pid, "source_job_a": job_id_a, "source_job_b": job_id_b},
        )
        await db.commit()
    finally:
        await db.close()

    await enqueue_analysis_job(compare_job_id)
    db = await get_db()
    try:
        created = await row_to_dict(
            await (await db.execute("SELECT id, seq FROM jobs WHERE id=?", (compare_job_id,))).fetchone()
        )
    finally:
        await db.close()
    return {
        "compare_job_id": compare_job_id,
        "compare_job_seq": (created or {}).get("seq"),
        "existing": False,
    }


@app.put("/api/projects/{pid}/experiments/layout")
async def update_experiment_layout(request: Request, pid: str, body: dict):
    positions = body.get("positions") or []
    if not isinstance(positions, list):
        raise HTTPException(400, "positions must be an array")
    db = await get_db()
    try:
        await ensure_user_row(db, request)
        await validate_project_access(db, request, pid)
        normalized = []
        for item in positions:
            if not isinstance(item, dict):
                raise HTTPException(400, "positions items must be objects")
            job_id = str(item.get("job_id") or "").strip()
            if not job_id:
                raise HTTPException(400, "job_id is required")
            job = await load_accessible_job(db, request, job_id, "id, project_id")
            if not job or job.get("project_id") != pid:
                raise HTTPException(400, "节点必须属于本项目")
            job_id = job["id"]
            try:
                x = float(item.get("x"))
                y = float(item.get("y"))
            except (TypeError, ValueError):
                raise HTTPException(400, "坐标必须是数字")
            if not (-1_000_000 <= x <= 1_000_000 and -1_000_000 <= y <= 1_000_000):
                raise HTTPException(400, "坐标超出范围")
            try:
                scale = float(item.get("scale", 1.0))
            except (TypeError, ValueError):
                raise HTTPException(400, "缩放比例必须是数字")
            if not (0.5 <= scale <= 2.5):
                raise HTTPException(400, "缩放比例超出范围")
            width = None
            if item.get("width") not in (None, ""):
                try:
                    width = float(item.get("width"))
                except (TypeError, ValueError):
                    raise HTTPException(400, "节点宽度必须是数字")
                if not (120 <= width <= 900):
                    raise HTTPException(400, "节点宽度超出范围")
            height = None
            if item.get("height") not in (None, ""):
                try:
                    height = float(item.get("height"))
                except (TypeError, ValueError):
                    raise HTTPException(400, "节点高度必须是数字")
                if not (72 <= height <= 640):
                    raise HTTPException(400, "节点高度超出范围")
            pinned = 1 if item.get("pinned", 1) else 0
            normalized.append((pid, job_id, x, y, scale, width, height, pinned))

        await db.executemany(
            """
            INSERT INTO experiment_node_layout(project_id, job_id, x, y, scale, width, height, pinned, updated_at)
            VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(project_id, job_id) DO UPDATE SET
                x=excluded.x,
                y=excluded.y,
                scale=excluded.scale,
                width=excluded.width,
                height=excluded.height,
                pinned=excluded.pinned,
                updated_at=CURRENT_TIMESTAMP
            """,
            normalized,
        )
        await write_audit(
            db, request, "experiment.layout_update",
            resource_type="project", resource_id=pid,
            details={"positions": len(normalized)},
        )
        await db.commit()
        return {"updated": len(normalized)}
    finally:
        await db.close()


@app.delete("/api/projects/{pid}/experiments/layout", status_code=204)
async def reset_experiment_layout(request: Request, pid: str):
    db = await get_db()
    try:
        await ensure_user_row(db, request)
        await validate_project_access(db, request, pid)
        await db.execute("DELETE FROM experiment_node_layout WHERE project_id=?", (pid,))
        await db.execute(
            "UPDATE experiment_edges SET label_x=NULL, label_y=NULL, label_width=NULL, label_height=NULL WHERE project_id=?",
            (pid,),
        )
        await write_audit(
            db, request, "experiment.layout_update",
            resource_type="project", resource_id=pid,
            details={"reset": True},
        )
        await db.commit()
    finally:
        await db.close()


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
    baseline_job_id = baseline["id"]
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


def _normalize_step_filter(value, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parse_step_filter(text)
    except ValueError as e:
        raise HTTPException(400, f"{label} step 选择格式错误: {e}") from e
    return text


def _step_reanalysis_suffix(mode: str, step_filter_a: str, step_filter_b: str = "") -> str:
    if mode == "single":
        return f"step {step_filter_a}"
    return f"A step {step_filter_a or '全部'} / B step {step_filter_b or '全部'}"


async def _resolve_reanalysis_trace(db, request: Request, job: dict, slot: str) -> tuple[str, str, Optional[str]]:
    path = _primary_trace_path(job, slot)
    name = job.get(f"file_{slot}_name") or ""
    source_id = None
    source_job_id = job.get(f"source_job_{slot}")
    if not path and source_job_id:
        source = await load_accessible_job(db, request, source_job_id)
        if not source:
            raise HTTPException(404, f"Source job {slot.upper()} not found")
        source_id = source["id"]
        path = _primary_trace_path(source, "a")
        name = source.get("file_a_name") or source.get("label") or ""

    if not path or not os.path.exists(path):
        raise HTTPException(409, f"Trace file {slot.upper()} has been deleted")
    return path, name or os.path.basename(path), source_id


@app.post("/api/jobs/{jid}/analyze-trace-slot", status_code=201)
async def analyze_compare_trace_slot(request: Request, jid: str, body: dict):
    db = await get_db()
    new_jid = str(uuid.uuid4())
    new_dir = job_dir(new_jid)
    row = None
    try:
        user_token = await ensure_user_row(db, request)
        job = await load_accessible_job(db, request, jid)
        if not job:
            raise HTTPException(404, "Job not found")
        jid = job["id"]
        if job.get("mode") != "compare":
            raise HTTPException(400, "Only compare jobs can derive a single trace analysis")
        if job.get("status") in {"pending", "running"}:
            raise HTTPException(409, "任务仍在分析中，完成后再单独分析 trace")

        slot = str(body.get("slot") or "").strip().lower()
        if slot not in {"a", "b"}:
            raise HTTPException(400, "slot must be 'a' or 'b'")

        project_id = body.get("project_id") if "project_id" in body else job.get("project_id")
        await validate_project_access(db, request, project_id or None)

        trace_path, trace_name, source_id = await _resolve_reanalysis_trace(db, request, job, slot)
        try:
            new_a_path, new_a_gzip_path = _copy_trace_reference(trace_path, os.path.join(new_dir, "trace_a.json"))
        except OSError as e:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(new_dir)
            raise HTTPException(500, f"Failed to copy trace file: {e}") from e

        label = str(body.get("label") or "").strip() or f"{trace_name} · 单独分析"
        await db.execute(
            """INSERT INTO jobs(id, project_id, user_token, label, mode,
                   file_a_name, file_a_path, file_a_gzip_path,
                   source_job_a,
                   save_triton_csv, save_triton_code)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                new_jid, project_id or None, user_token, label, "single",
                trace_name, new_a_path, new_a_gzip_path,
                source_id,
                int(job.get("save_triton_csv", 0) or 0),
                int(job.get("save_triton_code", 0) or 0),
            ),
        )
        await _refresh_job_storage_cache(db, new_jid)
        await write_audit(
            db, request, "job.trace_slot_analysis_create",
            resource_type="job", resource_id=new_jid,
            details={
                "source_compare_job": jid,
                "slot": slot,
                "source_job": source_id,
                "project_id": project_id,
                "label": label,
            },
        )
        await db.commit()
        row = await row_to_dict(await (await db.execute("SELECT * FROM jobs WHERE id=?", (new_jid,))).fetchone())
    except HTTPException:
        await db.rollback()
        if row is None:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(new_dir)
        raise
    except Exception:
        await db.rollback()
        if row is None:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(new_dir)
        raise
    finally:
        await db.close()

    await enqueue_analysis_job(new_jid)
    return row


@app.post("/api/jobs/{jid}/reanalyze-steps", status_code=201)
async def reanalyze_job_steps(request: Request, jid: str, body: dict):
    db = await get_db()
    new_jid = str(uuid.uuid4())
    new_dir = job_dir(new_jid)
    row = None
    try:
        user_token = await ensure_user_row(db, request)
        job = await load_accessible_job(db, request, jid)
        if not job:
            raise HTTPException(404, "Job not found")
        jid = job["id"]
        if job.get("status") in {"pending", "running"}:
            raise HTTPException(409, "任务仍在分析中，暂不能指定 step 重分析")

        mode = job.get("mode")
        step_filter_a = _normalize_step_filter(body.get("step_filter_a"), "A")
        step_filter_b = _normalize_step_filter(body.get("step_filter_b"), "B") if mode == "compare" else ""
        if mode == "single" and not step_filter_a:
            raise HTTPException(400, "单 trace 重分析需要指定 A step")
        if mode == "compare" and not (step_filter_a or step_filter_b):
            raise HTTPException(400, "对比重分析需要至少指定 A 或 B 的 step")

        project_id = body.get("project_id") if "project_id" in body else job.get("project_id")
        await validate_project_access(db, request, project_id or None)

        path_a, name_a, source_a = await _resolve_reanalysis_trace(db, request, job, "a")
        try:
            new_a_path, new_a_gzip_path = _copy_trace_reference(path_a, os.path.join(new_dir, "trace_a.json"))
            if mode == "compare":
                path_b, name_b, source_b = await _resolve_reanalysis_trace(db, request, job, "b")
                new_b_path, new_b_gzip_path = _copy_trace_reference(path_b, os.path.join(new_dir, "trace_b.json"))
            else:
                name_b = None
                source_b = None
                new_b_path = None
                new_b_gzip_path = None
        except OSError as e:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(new_dir)
            raise HTTPException(500, f"Failed to copy trace files: {e}") from e

        suffix = _step_reanalysis_suffix(mode, step_filter_a, step_filter_b)
        label = str(body.get("label") or "").strip() or f"{job.get('label') or jid[:8]} · {suffix}"
        await db.execute(
            """INSERT INTO jobs(id, project_id, user_token, label, mode,
                   file_a_name, file_a_path, file_a_gzip_path,
                   file_b_name, file_b_path, file_b_gzip_path,
                   source_job_a, source_job_b,
                   step_filter_a, step_filter_b,
                   save_triton_csv, save_triton_code)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                new_jid, project_id or None, user_token, label, mode,
                name_a, new_a_path, new_a_gzip_path,
                name_b, new_b_path, new_b_gzip_path,
                source_a, source_b,
                step_filter_a, step_filter_b,
                int(job.get("save_triton_csv", 0) or 0),
                int(job.get("save_triton_code", 0) or 0),
            ),
        )
        await _refresh_job_storage_cache(db, new_jid)
        await write_audit(
            db, request, "job.step_reanalysis_create",
            resource_type="job", resource_id=new_jid,
            details={
                "source_job": jid,
                "mode": mode,
                "project_id": project_id,
                "step_filter_a": step_filter_a,
                "step_filter_b": step_filter_b,
                "label": label,
            },
        )
        await db.commit()
        row = await row_to_dict(await (await db.execute("SELECT * FROM jobs WHERE id=?", (new_jid,))).fetchone())
    except HTTPException:
        await db.rollback()
        if row is None:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(new_dir)
        raise
    except Exception:
        await db.rollback()
        if row is None:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(new_dir)
        raise
    finally:
        await db.close()

    await enqueue_analysis_job(new_jid)
    return row


@app.post("/api/jobs/{jid}/rerun-swapped", status_code=201)
async def rerun_compare_swapped(request: Request, jid: str):
    db = await get_db()
    try:
        user_token = await ensure_user_row(db, request)
        job = await load_accessible_job(db, request, jid)
        if not job:
            raise HTTPException(404, "Job not found")
        jid = job["id"]
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
        jid = row["id"]

        job = _job_response(row, request)
        public_handle = _job_public_handle(job)
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
        "url": _job_share_url(request, public_handle),
        "path": f"/#/job/{public_handle}",
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
    jid = job["id"]
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
    jid = row["id"]

    version_id = request.query_params.get("version_id", "")
    payload = collect_ai_analysis(jid)
    selected_version = _select_ai_analysis_version(jid, version_id)
    if version_id and not selected_version:
        raise HTTPException(404, "AI analysis version not found")
    if selected_version:
        payload["selected_version"] = selected_version
        payload["selected_version_id"] = selected_version.get("id", "")
        payload["content"] = _read_ai_analysis_report(jid, selected_version.get("id", ""))
    else:
        payload["content"] = ""
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
    jid = row["id"]

    version_id = request.query_params.get("version_id", "")
    selected_version = _select_ai_analysis_version(jid, version_id)
    if version_id and not selected_version:
        raise HTTPException(404, "AI analysis version not found")
    content = _read_ai_analysis_report(jid, selected_version.get("id", "") if selected_version else "")
    if not content:
        raise HTTPException(404, "AI analysis report not found")

    filename = _safe_download_name(row.get("label"), f"{jid}-ai-analysis")
    lower = filename.lower()
    for suffix in (".markdown", ".md"):
        if lower.endswith(suffix):
            filename = filename[:-len(suffix)]
            break
    generated = ""
    if selected_version and selected_version.get("generated_at"):
        generated = selected_version["generated_at"].replace(":", "").replace("-", "")[:15]
    suffix = f"-{generated}" if generated else ""
    filename = f"{filename}-ai-analysis{suffix}.md"
    return PlainTextResponse(
        content.rstrip() + "\n",
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": _content_disposition(filename)},
    )


@app.get("/api/jobs/{jid}/ai-analysis/artifacts/{artifact_path:path}")
async def download_job_ai_analysis_artifact(request: Request, jid: str, artifact_path: str):
    if not ALLOW_FILE_DOWNLOAD:
        raise HTTPException(403, "File download is disabled")

    db = await get_db()
    try:
        row = await load_accessible_job(db, request, jid, "id, label, status")
    finally:
        await db.close()
    if not row:
        raise HTTPException(404)
    jid = row["id"]

    full_path, normalized_path = _safe_ai_analysis_artifact_path(jid, artifact_path)

    audit_db = await get_db()
    try:
        await write_audit(
            audit_db, request, "job.ai_analysis_artifact_download",
            resource_type="job", resource_id=jid,
            details={"path": normalized_path, "size": _path_size(full_path)},
        )
        await audit_db.commit()
    finally:
        await audit_db.close()

    filename = _safe_download_name(
        normalized_path.replace("/", "__"),
        os.path.basename(normalized_path),
    )
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(
        full_path,
        media_type=media_type,
        headers={"Content-Disposition": _content_disposition(filename)},
    )


@app.get("/api/jobs/{jid}/ai-analysis/artifact-content/{artifact_path:path}")
async def read_job_ai_analysis_artifact_content(request: Request, jid: str, artifact_path: str):
    db = await get_db()
    try:
        row = await load_accessible_job(db, request, jid, "id, label, status")
    finally:
        await db.close()
    if not row:
        raise HTTPException(404)
    jid = row["id"]

    full_path, normalized_path = _safe_ai_analysis_artifact_path(jid, artifact_path)
    if not _is_ai_code_preview_artifact(normalized_path):
        raise HTTPException(400, "Unsupported AI analysis code preview type")

    content, truncated, size = await asyncio.to_thread(
        _read_text_preview,
        full_path,
        AI_ANALYSIS_CODE_PREVIEW_MAX_BYTES,
    )

    audit_db = await get_db()
    try:
        await write_audit(
            audit_db, request, "job.ai_analysis_artifact_preview",
            resource_type="job", resource_id=jid,
            details={"path": normalized_path, "size": size, "truncated": truncated},
        )
        await audit_db.commit()
    finally:
        await audit_db.close()

    return {
        "path": normalized_path,
        "name": os.path.basename(normalized_path),
        "size": size,
        "truncated": truncated,
        "content": content,
        "language": "python",
    }


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
    start_ai_analysis_workers()

    body = body or {}
    force = bool(body.get("force"))
    user_prompt = _normalize_ai_user_prompt(body.get("prompt") if "prompt" in body else body.get("promt"))
    notification_context = {}
    async with ai_analysis_start_lock:
        db = await get_db()
        try:
            row = await load_accessible_job(db, request, jid)
            if not row:
                raise HTTPException(404)
            jid = row["id"]
            if row.get("status") != "done":
                raise HTTPException(409, "Job is not completed")

            current = collect_ai_analysis(jid)
            if current.get("status") in {"queued", "running"}:
                raise HTTPException(409, "AI analysis is already queued or running")
            if current.get("status") == "done" and not force and not user_prompt:
                current["content"] = _read_ai_analysis_report(jid)
                current["artifacts"] = _collect_ai_analysis_artifacts(jid)
                return current
            _ensure_current_ai_report_versioned(jid, _read_ai_analysis_status(jid))

            user = current_user(request)
            trigger_user_token = user.get("username") or request_user(request)
            notification_context = {
                "job_label": row.get("label") or row.get("file_a_name") or jid,
                "job_mode": row.get("mode") or "",
                "owner_user_token": row.get("user_token") or "",
                "trigger_user_token": trigger_user_token,
                "trigger_user_display": user.get("display_name") or trigger_user_token or "local",
                "trigger_user_email": user.get("email") or "",
                "user_prompt": user_prompt,
            }
            queued_at = _utc_now_iso()
            _write_ai_analysis_status(jid, {
                "status": "queued",
                "phase": "queued",
                "progress": 5,
                "mode": row.get("mode"),
                "skill": CLAUDE_COMPARE_TRACE_SKILL if row.get("mode") == "compare" else CLAUDE_SINGLE_TRACE_SKILL,
                "model": _ai_backend_model_name(),
                "user_prompt": user_prompt,
                "trigger_user_token": notification_context["trigger_user_token"],
                "trigger_user_display": notification_context["trigger_user_display"],
                "trigger_user_email": notification_context["trigger_user_email"],
                "owner_user_token": notification_context["owner_user_token"],
                "queued_at": queued_at,
                "started_at": queued_at,
                "updated_at": queued_at,
            })
            await write_audit(
                db, request, "job.ai_analysis_start",
                resource_type="job", resource_id=jid,
                details={"mode": row.get("mode"), "force": force, "has_user_prompt": bool(user_prompt)},
            )
            await db.commit()
        except HTTPException:
            await db.rollback()
            raise
        finally:
            await db.close()

        await enqueue_ai_analysis_job(jid, notification_context)
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
    jid = job["id"]
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
    ]
    if job.get("step_filter_a") or job.get("step_filter_b"):
        lines.append(f"- Step 过滤 A: `{job.get('step_filter_a') or '全部'}`")
        if job.get("mode") == "compare":
            lines.append(f"- Step 过滤 B: `{job.get('step_filter_b') or '全部'}`")
    lines.extend([
        "",
        "## Trace 文件",
        "",
        f"- A: {job.get('file_a_name') or '无'}",
    ])
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
    jid = row["id"]
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
    jid = row["id"]

    if "project_id" in body:
        await validate_project_access(db, request, body["project_id"] or None)
    if "label" in body:
        await db.execute("UPDATE jobs SET label=? WHERE id=?", (body["label"], jid))
    if "project_id" in body:
        next_project_id = body["project_id"] or None
        if row.get("project_id") != next_project_id:
            await _clear_experiment_job_links(db, [jid])
        await db.execute("UPDATE jobs SET project_id=? WHERE id=?",
                         (next_project_id, jid))
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
    jid = row["id"]

    if _job_delete_locked(row):
        await db.close()
        raise HTTPException(409, "任务仍在分析中，暂不能删除")

    _remove_job_dir(jid)

    await _clear_experiment_job_links(db, [jid])
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
    jid = row["id"]

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
    jid = row["id"]

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
    jid = row["id"]

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
    jid = row["id"]
    if slot not in ("a", "b"):
        await db.close()
        raise HTTPException(400, "slot must be 'a' or 'b'")
    if _job_files_locked(row):
        await db.close()
        raise HTTPException(409, "任务仍在分析中，暂不能删除文件")

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
        jid = row["id"]
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
    jid = row["id"]

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
