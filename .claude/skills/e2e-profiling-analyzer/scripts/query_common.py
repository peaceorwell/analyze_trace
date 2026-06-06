#!/usr/bin/env python3
"""Shared SQLite query helpers for E2E profiling scripts."""

import argparse
import json
import re
import sqlite3
import sys


def parse_extra(extra_str):
    try:
        return json.loads(extra_str) if extra_str else {}
    except Exception:
        return {}


def load_string_map(cursor):
    cursor.execute("SELECT ID, string FROM string_table")
    return {row[0]: row[1] for row in cursor.fetchall()}


def get_name(string_map, name_id, width=None):
    name = string_map.get(name_id, f"nameId={name_id}")
    if width:
        return name[:width]
    return name


def simplify_task_name(name):
    if not name or name.startswith("nameId="):
        return name

    simplified = name.strip()
    simplified = re.sub(r"\(.*\)$", "", simplified)
    simplified = re.sub(r"<.*>", "", simplified)
    simplified = simplified.rstrip("< ")
    match = re.search(r"([A-Za-z_~][A-Za-z0-9_:]*)\s*$", simplified)
    if match:
        simplified = match.group(1)
    return simplified or name


def get_invoke_info(cursor, corr_id):
    cursor.execute(
        """
        SELECT nameId, start, end, self, threadId, processId
        FROM function_data
        WHERE correlationId = ?
        """,
        (corr_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "name_id": row[0],
        "start": row[1],
        "end": row[2],
        "self": row[3],
        "tid": row[4],
        "process_id": row[5],
    }


def get_function_call_stack(cursor, invoke, max_depth=16):
    """Infer the host function stack by same-thread time containment."""
    process_id = invoke.get("process_id")
    thread_id = invoke["tid"]
    start = invoke["start"]
    end = invoke["end"]

    if process_id is None:
        cursor.execute(
            """
            SELECT correlationId, nameId, processId, threadId, start, end, self
            FROM function_data
            WHERE threadId = ? AND start <= ? AND end >= ?
            ORDER BY start ASC, end DESC
            LIMIT ?
            """,
            (thread_id, start, end, max_depth),
        )
    else:
        cursor.execute(
            """
            SELECT correlationId, nameId, processId, threadId, start, end, self
            FROM function_data
            WHERE processId = ? AND threadId = ? AND start <= ? AND end >= ?
            ORDER BY start ASC, end DESC
            LIMIT ?
            """,
            (process_id, thread_id, start, end, max_depth),
        )

    return [
        {
            "correlation_id": row[0],
            "name_id": row[1],
            "process_id": row[2],
            "tid": row[3],
            "start": row[4],
            "end": row[5],
            "self": row[6],
        }
        for row in cursor.fetchall()
    ]


def table_exists(cursor, table_name):
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def get_host_context_stack(cursor, invoke, max_depth=32):
    """Infer host-side context stack from function, CNPX, and framework-op ranges."""
    process_id = invoke.get("process_id")
    thread_id = invoke["tid"]
    start = invoke["start"]
    end = invoke["end"]
    frames = []

    def append_frames(source, query, params, mapper):
        cursor.execute(query, params)
        for row in cursor.fetchall():
            item = mapper(row)
            item["source"] = source
            frames.append(item)

    append_frames(
        "function",
        """
        SELECT correlationId, nameId, processId, threadId, start, end, self
        FROM function_data
        WHERE processId = ? AND threadId = ? AND start <= ? AND end >= ?
        """,
        (process_id, thread_id, start, end),
        lambda row: {
            "correlation_id": row[0],
            "name_id": row[1],
            "process_id": row[2],
            "tid": row[3],
            "start": row[4],
            "end": row[5],
            "self": row[6],
        },
    )

    if table_exists(cursor, "cnpx_data"):
        append_frames(
            "cnpx",
            """
            SELECT ID, stringId, processId, threadId, start, end, extra_data
            FROM cnpx_data
            WHERE processId = ? AND threadId = ? AND start <= ? AND end >= ?
            """,
            (process_id, thread_id, start, end),
            lambda row: {
                "correlation_id": row[0],
                "name_id": row[1],
                "name": parse_extra(row[6]).get("name"),
                "process_id": row[2],
                "tid": row[3],
                "start": row[4],
                "end": row[5],
                "self": None,
            },
        )

    cnpx_tables = [
        ("cnpx_framework", "INTERNAL_CNPX_FRAMEWORK_RANGES_V1"),
        ("cnpx_op", "INTERNAL_CNPX_OP_RANGES_V1"),
        ("cnpx_comm", "INTERNAL_CNPX_COMM_RANGES_V1"),
    ]
    for source, table_name in cnpx_tables:
        if not table_exists(cursor, table_name):
            continue
        append_frames(
            source,
            f"""
            SELECT cnpxId, nameId, processId, threadId, start, end, extra
            FROM {table_name}
            WHERE processId = ? AND threadId = ? AND start <= ? AND end >= ?
            """,
            (process_id, thread_id, start, end),
            lambda row: {
                "correlation_id": row[0],
                "name_id": row[1],
                "name": parse_extra(row[6]).get("name"),
                "process_id": row[2],
                "tid": row[3],
                "start": row[4],
                "end": row[5],
                "self": None,
            },
        )

    if table_exists(cursor, "Internal_operation_range_data"):
        append_frames(
            "framework_op",
            """
            SELECT extraId, nameId, processId, threadId, start, end, extra
            FROM Internal_operation_range_data
            WHERE processId = ? AND threadId = ? AND start <= ? AND end >= ?
            """,
            (process_id, thread_id, start, end),
            lambda row: {
                "correlation_id": row[0],
                "name_id": row[1],
                "name": parse_extra(row[6]).get("name"),
                "process_id": row[2],
                "tid": row[3],
                "start": row[4],
                "end": row[5],
                "self": None,
            },
        )

    if table_exists(cursor, "Internal_python_function_trace_data"):
        append_frames(
            "python",
            """
            SELECT stringId, pid, tid, start, end
            FROM Internal_python_function_trace_data
            WHERE pid = ? AND tid = ? AND start <= ? AND end >= ?
            """,
            (process_id, thread_id, start, end),
            lambda row: {
                "correlation_id": None,
                "name_id": row[0],
                "process_id": row[1],
                "tid": row[2],
                "start": row[3],
                "end": row[4],
                "self": None,
            },
        )

    frames.sort(key=lambda item: (item["start"], -item["end"], item["source"]))
    return frames[:max_depth]


def format_host_frame(frame, string_map):
    name = simplify_task_name(frame.get("name") or get_name(string_map, frame["name_id"]))
    corr = f" corr={frame['correlation_id']}" if frame.get("correlation_id") is not None else ""
    return (
        f"[{frame['source']}] {name} start={frame['start']/1e6:.2f}ms "
        f"(+{(frame['end'] - frame['start'])/1e6:.4f}ms){corr} tid={frame['tid']}"
    )


def host_frame_payload(frame, string_map):
    return {
        "source": frame["source"],
        "name": simplify_task_name(frame.get("name") or get_name(string_map, frame["name_id"])),
        "start_ms": frame["start"] / 1e6,
        "duration_ms": (frame["end"] - frame["start"]) / 1e6,
        "correlation_id": frame.get("correlation_id"),
        "process_id": frame.get("process_id"),
        "tid": frame["tid"],
    }


def emit_host_stack_text(frames, string_map, out):
    if not frames:
        print("(empty)", file=out)
        return
    for frame in frames:
        print(format_host_frame(frame, string_map), file=out)


def emit_host_stack_json(frames, string_map, out):
    json.dump([host_frame_payload(frame, string_map) for frame in frames], out, indent=2)
    print(file=out)


def build_invoke_from_args(cursor, args):
    invoke = get_invoke_info(cursor, args.host_stack)
    if not invoke:
        raise ValueError(f"function invoke not found: corr_id={args.host_stack}")
    return invoke


def main():
    parser = argparse.ArgumentParser(description="Common query utilities for E2E profiling DBs")
    parser.add_argument("db_path", help="数据库路径")
    parser.add_argument("--host-stack", type=int, metavar="CORR_ID", help="查询host调用栈的function_data correlationId")
    parser.add_argument("--max-depth", type=int, default=32, help="最大栈深度, 默认32")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="输出格式")

    args = parser.parse_args()
    if args.host_stack is None:
        parser.error("--host-stack is required")

    conn = sqlite3.connect(args.db_path)
    try:
        cursor = conn.cursor()
        string_map = load_string_map(cursor)
        invoke = build_invoke_from_args(cursor, args)
        frames = get_host_context_stack(cursor, invoke, max_depth=args.max_depth)
        if args.format == "json":
            emit_host_stack_json(frames, string_map, sys.stdout)
            return
        emit_host_stack_text(frames, string_map, sys.stdout)
    except ValueError as exc:
        parser.error(str(exc))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
