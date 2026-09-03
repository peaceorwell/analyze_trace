#!/usr/bin/env python3
"""Host-side operator and runtime-API cost breakdown.

A high device-stream gap ratio says the host is not keeping the device fed; it does
not say which host work is responsible. This script answers that: it computes host
operator self time from nested ranges, separates operators that launch device work
from host-only operators, and summarizes runtime launch/sync APIs.

Usage:
    python3 host_op_breakdown.py cnperf_data.db
    python3 host_op_breakdown.py cnperf_data.db --format json --start-ns 5 --end-ns 9
    python3 host_op_breakdown.py cnperf_data.db --top 30 --min-self-ms 1
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

try:
    from query_common import add_window_args, load_string_map, table_exists, window_payload, window_sql
except ImportError:  # allow importing this module from another sys.path
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from query_common import add_window_args, load_string_map, table_exists, window_payload, window_sql

# Runtime APIs whose cost is launch overhead rather than useful host work.
LAUNCH_API_HINTS = ("invokekernel", "launch", "kernel")
SYNC_API_HINTS = ("sync", "wait", "query")


def ms(value_ns) -> float:
    return round((value_ns or 0) / 1e6, 3)


def us(value_ns) -> float:
    return round((value_ns or 0) / 1e3, 3)


def _classify_api(name: str) -> str:
    lowered = name.lower()
    if any(hint in lowered for hint in SYNC_API_HINTS):
        return "sync"
    if any(hint in lowered for hint in LAUNCH_API_HINTS):
        return "launch"
    return "other"


def compute_self_times(ranges):
    """Attribute exclusive (self) time to nested host ranges per thread.

    Ranges nest; summing inclusive durations double counts. Each range keeps the
    time not covered by its direct children, which is the cost that actually
    belongs to that operator.
    """
    by_thread: dict[tuple, list] = {}
    for row in ranges:
        by_thread.setdefault((row["process_id"], row["thread_id"]), []).append(row)

    for rows in by_thread.values():
        rows.sort(key=lambda row: (row["start_ns"], -row["end_ns"]))
        stack: list[dict] = []
        for row in rows:
            while stack and stack[-1]["end_ns"] <= row["start_ns"]:
                stack.pop()
            row["depth"] = len(stack)
            if stack:
                parent = stack[-1]
                covered = min(row["end_ns"], parent["end_ns"]) - max(row["start_ns"], parent["start_ns"])
                parent["child_ns"] = parent.get("child_ns", 0) + max(0, covered)
            stack.append(row)

    for row in ranges:
        duration = max(0, row["end_ns"] - row["start_ns"])
        row["duration_ns"] = duration
        row["self_ns"] = max(0, duration - row.get("child_ns", 0))
    return ranges


def load_ranges(cursor, process_id, start_ns, end_ns):
    if not table_exists(cursor, "Internal_operation_range_data"):
        return []
    clauses, params = [], []
    if process_id is not None:
        clauses.append("processId = ?")
        params.append(process_id)
    window_clauses, window_params = window_sql(start_ns, end_ns, mode="overlap")
    clauses.extend(window_clauses)
    params.extend(window_params)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    cursor.execute(
        "SELECT processId, threadId, start, end, extraId, nameId "
        f"FROM Internal_operation_range_data{where}",
        params,
    )
    return [
        {
            "process_id": pid,
            "thread_id": tid,
            "start_ns": int(start),
            "end_ns": int(end),
            "extra_id": extra_id,
            "name_id": name_id,
        }
        for pid, tid, start, end, extra_id, name_id in cursor.fetchall()
        if start is not None and end is not None and int(end) >= int(start)
    ]


def load_launch_links(cursor, start_ns, end_ns):
    """Map host range extraId -> device kernel time launched through it."""
    if not (table_exists(cursor, "Internal_op_range_relations") and table_exists(cursor, "function_data")):
        return {}, {}
    cursor.execute("SELECT externalCorrelationId, correlationId FROM Internal_op_range_relations")
    corr_by_extra: dict[int, list[int]] = {}
    for extra_id, corr in cursor.fetchall():
        corr_by_extra.setdefault(extra_id, []).append(corr)

    kernel_ns_by_corr: dict[int, int] = {}
    if table_exists(cursor, "device_task_kernel_data"):
        clauses, params = window_sql(start_ns, end_ns, mode="start")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cursor.execute(
            f"SELECT correlationId, start, end FROM device_task_kernel_data{where}", params
        )
        for corr, start, end in cursor.fetchall():
            if corr is None or start is None or end is None:
                continue
            kernel_ns_by_corr[corr] = kernel_ns_by_corr.get(corr, 0) + max(0, int(end) - int(start))

    device_ns_by_extra: dict[int, int] = {}
    kernel_count_by_extra: dict[int, int] = {}
    for extra_id, corrs in corr_by_extra.items():
        total = 0
        count = 0
        for corr in corrs:
            if corr in kernel_ns_by_corr:
                total += kernel_ns_by_corr[corr]
                count += 1
        device_ns_by_extra[extra_id] = total
        kernel_count_by_extra[extra_id] = count
    return device_ns_by_extra, kernel_count_by_extra


def summarize_runtime_apis(cursor, string_map, process_id, start_ns, end_ns, top):
    if not table_exists(cursor, "function_data"):
        return {"available": False, "reason": "missing function_data", "apis": []}
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(function_data)")}
    clauses, params = [], []
    if process_id is not None and "processId" in columns:
        clauses.append("processId = ?")
        params.append(process_id)
    window_clauses, window_params = window_sql(start_ns, end_ns, mode="start")
    clauses.extend(window_clauses)
    params.extend(window_params)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    self_expr = "self" if "self" in columns else "(end - start)"
    cursor.execute(
        f"SELECT nameId, COUNT(*), SUM({self_expr}), SUM(end - start) FROM function_data{where} "
        "GROUP BY nameId",
        params,
    )
    apis = []
    for name_id, count, self_ns, total_ns in cursor.fetchall():
        name = string_map.get(name_id, f"nameId={name_id}")
        self_ns = self_ns or total_ns or 0
        apis.append(
            {
                "api": name,
                "kind": _classify_api(name),
                "count": count,
                "self_ms": ms(self_ns),
                "total_ms": ms(total_ns),
                "avg_self_us": us(self_ns / count) if count else 0.0,
            }
        )
    apis.sort(key=lambda item: -item["self_ms"])
    launch = [item for item in apis if item["kind"] == "launch"]
    sync = [item for item in apis if item["kind"] == "sync"]
    return {
        "available": True,
        "apis": apis[:top],
        "launch_calls": sum(item["count"] for item in launch),
        "launch_self_ms": round(sum(item["self_ms"] for item in launch), 3),
        "avg_launch_us": (
            us(sum(item["self_ms"] for item in launch) * 1e6 / sum(item["count"] for item in launch))
            if sum(item["count"] for item in launch)
            else None
        ),
        "sync_self_ms": round(sum(item["self_ms"] for item in sync), 3),
        "total_self_ms": round(sum(item["self_ms"] for item in apis), 3),
    }


def analyze_db(db_path, top=20, min_self_ms=0.0, process_id=None, start_ns=None, end_ns=None):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        string_map = load_string_map(cursor) if table_exists(cursor, "string_table") else {}
        ranges = load_ranges(cursor, process_id, start_ns, end_ns)
        if not ranges:
            return {
                "db": db_path,
                "window": window_payload(start_ns, end_ns),
                "error": "no host operator ranges (Internal_operation_range_data is missing or empty)",
                "runtime_apis": summarize_runtime_apis(
                    cursor, string_map, process_id, start_ns, end_ns, top
                ),
            }

        compute_self_times(ranges)
        device_ns_by_extra, kernel_count_by_extra = load_launch_links(cursor, start_ns, end_ns)
        linkage_available = bool(device_ns_by_extra)

        by_name: dict[str, dict] = {}
        for row in ranges:
            name = string_map.get(row["name_id"], f"nameId={row['name_id']}")
            entry = by_name.setdefault(
                name,
                {
                    "op": name,
                    "count": 0,
                    "self_ns": 0,
                    "total_ns": 0,
                    "max_self_ns": 0,
                    "device_ns": 0,
                    "kernel_count": 0,
                    "min_depth": row.get("depth", 0),
                },
            )
            entry["count"] += 1
            entry["self_ns"] += row["self_ns"]
            entry["total_ns"] += row["duration_ns"]
            entry["max_self_ns"] = max(entry["max_self_ns"], row["self_ns"])
            entry["min_depth"] = min(entry["min_depth"], row.get("depth", 0))
            extra_id = row.get("extra_id")
            if extra_id is not None:
                entry["device_ns"] += device_ns_by_extra.get(extra_id, 0)
                entry["kernel_count"] += kernel_count_by_extra.get(extra_id, 0)

        operators = []
        for entry in by_name.values():
            operators.append(
                {
                    "op": entry["op"],
                    "count": entry["count"],
                    "self_ms": ms(entry["self_ns"]),
                    "total_ms": ms(entry["total_ns"]),
                    "avg_self_us": us(entry["self_ns"] / entry["count"]),
                    "max_self_us": us(entry["max_self_ns"]),
                    "device_ms": ms(entry["device_ns"]) if linkage_available else None,
                    "kernel_count": entry["kernel_count"] if linkage_available else None,
                    "device_work": (
                        None if not linkage_available else bool(entry["kernel_count"])
                    ),
                    "min_depth": entry["min_depth"],
                }
            )
        operators.sort(key=lambda item: -item["self_ms"])
        filtered = [item for item in operators if item["self_ms"] >= min_self_ms]

        # "host-only" is about a range's own self time: a wrapper whose children launch
        # kernels still counts as host-only for the time it spends outside those children.
        host_only = [item for item in filtered if item["device_work"] is False]
        total_self_ms = round(sum(item["self_ms"] for item in operators), 3)
        span_ns = max(row["end_ns"] for row in ranges) - min(row["start_ns"] for row in ranges)

        limitations = []
        if not linkage_available:
            limitations.append(
                "no usable Internal_op_range_relations/function_data linkage; "
                "host-only versus device-launching operators cannot be separated"
            )
        if linkage_available:
            limitations.append(
                "host-only operators are ranges whose own self time launches no device work; "
                "their children may still launch kernels"
            )
        if start_ns is None and end_ns is None:
            limitations.append(
                "whole-capture scope: run step_window.py and pass --start-ns/--end-ns "
                "before making steady-state claims"
            )

        return {
            "db": db_path,
            "window": window_payload(start_ns, end_ns),
            "host_range_span_ms": ms(span_ns),
            "host_op_self_total_ms": total_self_ms,
            "host_op_range_count": len(ranges),
            "top_operators": filtered[:top],
            "top_host_only_operators": host_only[:top],
            "host_only_self_ms": round(sum(item["self_ms"] for item in host_only), 3),
            "runtime_apis": summarize_runtime_apis(
                cursor, string_map, process_id, start_ns, end_ns, top
            ),
            "limitations": limitations,
        }
    finally:
        conn.close()


def print_text(payload) -> None:
    print(f"DB: {payload['db']}")
    if payload.get("error"):
        print(f"Error: {payload['error']}")
    else:
        print(
            f"Host range span: {payload['host_range_span_ms']}ms | "
            f"operator self total: {payload['host_op_self_total_ms']}ms | "
            f"ranges: {payload['host_op_range_count']}"
        )
        print(f"Host-only operator self time: {payload['host_only_self_ms']}ms")
        print()
        print(f"{'self_ms':>10} {'count':>7} {'avg_us':>9} {'device_ms':>10} {'kernels':>8}  op")
        for item in payload["top_operators"]:
            print(
                f"{item['self_ms']:>10} {item['count']:>7} {item['avg_self_us']:>9} "
                f"{str(item['device_ms']):>10} {str(item['kernel_count']):>8}  {item['op'][:52]}"
            )
    apis = payload.get("runtime_apis", {})
    if apis.get("available"):
        print()
        print(
            f"Runtime APIs: launch calls={apis['launch_calls']} "
            f"launch self={apis['launch_self_ms']}ms avg={apis['avg_launch_us']}us "
            f"sync self={apis['sync_self_ms']}ms"
        )
        for item in apis["apis"][:10]:
            print(f"  {item['self_ms']:>10}ms {item['count']:>7} {item['kind']:<7} {item['api'][:48]}")
    for item in payload.get("limitations", []):
        print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Host operator and runtime API cost breakdown")
    parser.add_argument("db", nargs="+", help="cnperf SQLite DB path(s)")
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--min-self-ms", type=float, default=0.0, help="drop operators below this self time"
    )
    add_window_args(parser)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    payloads = [
        analyze_db(db, args.top, args.min_self_ms, args.process_id, args.start_ns, args.end_ns)
        for db in args.db
    ]
    if args.format == "json":
        json.dump(
            {"window": window_payload(args.start_ns, args.end_ns), "profiles": payloads},
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        print()
    else:
        for payload in payloads:
            print_text(payload)
            print()


if __name__ == "__main__":
    main()
