#!/usr/bin/env python3
"""Collect breakdown and summary tables for one cnperf DB time range."""

import argparse
import bisect
import json
import math
import re
import sqlite3
from pathlib import Path
from urllib.parse import quote

DEFAULT_GAP_THRESHOLD_NS = 100_000
DEFAULT_INVOKE_THRESHOLD_NS = 100_000
DEFAULT_HOST_ROWS = 100

# Theoretical (peak) bandwidth per MLU model, same unit as io_efficiency (folded
# bandwidth, GB/s). Matched by substring on the device model name.
THEORETICAL_BANDWIDTH = {"590": 2000, "580": 1200}

COMPILE_REGION_REGEX = re.compile(
    r"Torch-Compiled Region|CompiledFunction|TorchDynamo|Inductor|"
    r"compiled_fn|AOTAutograd|graph\s*break",
    re.IGNORECASE,
)
RECOMPILE_REGEX = re.compile(r"TorchDynamo Cache Lookup|recompile|guard", re.IGNORECASE)
IO_EFF_KEYS = ("io_efficiency", "io_eff", "memory_efficiency", "mem_efficiency")
BANDWIDTH_KEYS = ("achieved_bandwidth", "bandwidth", "gbps", "effective_bandwidth")

def ms(ns):
    return ns / 1e6 if ns is not None else None


def fmt_ms(value):
    if value is None:
        return "n/a"
    return f"{value:,.3f}"


def normalize_input(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix != ".db":
        raise ValueError(f"Expected a cnperf DB input. Convert upstream first: {path}")
    return path


def connect_db(path):
    uri = f"file:{quote(str(Path(path).resolve()), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def table_exists(cur, table):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,))
    return cur.fetchone() is not None


def table_columns(cur, table):
    if not table_exists(cur, table):
        return set()
    return {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}


def load_strings(cur):
    if not table_exists(cur, "string_table"):
        return {}
    return {row[0]: row[1] for row in cur.execute("SELECT ID, string FROM string_table")}


def get_name(strings, name_id):
    return strings.get(name_id, f"nameId={name_id}")


def simplify_name(name):
    return " ".join(str(name).replace("\n", " ").split())


def empty_summary():
    return {"count": 0, "unique_names": 0, "total_ms": 0.0, "avg_ms": 0.0, "p90_ms": 0.0, "max_ms": 0.0, "top": []}


def window_params(start, end):
    return {"window_start": start, "window_end": end}


def get_db_time_range(cur):
    """Infer global [start, end) from all relevant tables."""
    tables = [
        "function_data",
        "Internal_operation_range_data",
        "device_task_kernel_data",
        "device_task_memcpy_data",
        "device_task_memset_data",
        "device_task_atomic_operation_data",
        "device_task_notifier_data",
    ]
    all_min = []
    all_max = []
    for table in tables:
        if not table_exists(cur, table):
            continue
        row = cur.execute(f"SELECT MIN(start), MAX(end) FROM {table}").fetchone()
        if row and row[0] is not None:
            all_min.append(row[0])
            all_max.append(row[1])
    if not all_min:
        return 0, 0
    return min(all_min), max(all_max)


def percentile_ns(values, percentile):
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile / 100.0) - 1))
    return ordered[index]


def clipped_duration(row_start, row_end, window_start, window_end):
    return max(0, min(row_end, window_end) - max(row_start, window_start))


def start_in_range_where():
    return "start >= :window_start AND start < :window_end"


def clipped_intervals(cur, table, start, end, where_extra=""):
    if not table_exists(cur, table):
        return []
    where = start_in_range_where()
    if where_extra:
        where += f" AND {where_extra}"
    query = f"SELECT start, end FROM {table} WHERE {where}"
    intervals = []
    for row_start, row_end in cur.execute(query, window_params(start, end)):
        interval_start = max(row_start, start)
        interval_end = min(row_end, end)
        if interval_end > interval_start:
            intervals.append((interval_start, interval_end))
    return intervals


def merge_intervals(intervals):
    merged = []
    for interval_start, interval_end in sorted(intervals):
        if not merged or interval_start > merged[-1][1]:
            merged.append([interval_start, interval_end])
        else:
            merged[-1][1] = max(merged[-1][1], interval_end)
    return [(interval_start, interval_end) for interval_start, interval_end in merged]


def interval_index(intervals):
    return {"intervals": intervals, "starts": [item[0] for item in intervals]}


def interval_total(intervals):
    return sum(interval_end - interval_start for interval_start, interval_end in intervals)


def complement_intervals(start, end, covered_intervals):
    gaps = []
    cursor = start
    for interval_start, interval_end in merge_intervals(covered_intervals):
        if interval_start > cursor:
            gaps.append((cursor, interval_start))
        cursor = max(cursor, interval_end)
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def uncovered_ns(row_start, row_end, covering_intervals, window_start, window_end):
    interval_start = max(row_start, window_start)
    interval_end = min(row_end, window_end)
    if interval_end <= interval_start:
        return 0
    uncovered = interval_end - interval_start
    if isinstance(covering_intervals, dict):
        intervals = covering_intervals["intervals"]
        starts = covering_intervals["starts"]
    else:
        intervals = covering_intervals
        starts = [item[0] for item in intervals]
    index = max(0, bisect.bisect_right(starts, interval_start) - 1)
    for cover_start, cover_end in intervals[index:]:
        if cover_start >= interval_end:
            break
        overlap = min(interval_end, cover_end) - max(interval_start, cover_start)
        if overlap > 0:
            uncovered -= overlap
    return max(0, uncovered)


def summary_from_durations(durations):
    count = len(durations)
    total = sum(durations)
    return {
        "count": count,
        "total_ms": ms(total),
        "avg_ms": ms(total / count) if count else 0.0,
        "p90_ms": ms(percentile_ns(durations, 90)) if count else 0.0,
        "max_ms": ms(max(durations)) if count else 0.0,
    }


def summary_from_intervals(intervals):
    durations = [interval_end - interval_start for interval_start, interval_end in intervals]
    return summary_from_durations(durations)


def aggregate_named(cur, table, strings, start, end, where_extra="", top=None, preloaded_rows=None):
    if preloaded_rows is None:
        if not table_exists(cur, table):
            return empty_summary()
        where = start_in_range_where()
        if where_extra:
            where += f" AND {where_extra}"
        query = f"SELECT nameId, start, end FROM {table} WHERE {where}"
        preloaded_rows = cur.execute(query, window_params(start, end))
    groups = {}
    durations = []
    for name_id, row_start, row_end in preloaded_rows:
        duration = clipped_duration(row_start, row_end, start, end)
        if duration <= 0:
            continue
        name = simplify_name(get_name(strings, name_id))
        groups.setdefault(name, []).append(duration)
        durations.append(duration)
    total = sum(durations)
    rows = []
    for name, group_durations in groups.items():
        item = summary_from_durations(group_durations)
        item["name"] = name
        item["share_pct"] = sum(group_durations) / total * 100 if total else 0.0
        rows.append(item)
    rows.sort(key=lambda item: item["total_ms"], reverse=True)
    summary = summary_from_durations(durations)
    return {
        "count": summary["count"],
        "unique_names": len(groups),
        "total_ms": summary["total_ms"],
        "avg_ms": summary["avg_ms"],
        "p90_ms": summary["p90_ms"],
        "max_ms": summary["max_ms"],
        "top": rows[:top] if top is not None else rows,
    }


def communication_kernel_summary(cur, strings, start, end, compute_intervals, preloaded_rows, top=20):
    compute_intervals = compute_intervals or []
    groups = {}
    durations = []
    uncovered_total = 0
    for name_id, row_start, row_end in preloaded_rows:
        duration = clipped_duration(row_start, row_end, start, end)
        if duration <= 0:
            continue
        uncovered = uncovered_ns(row_start, row_end, compute_intervals, start, end)
        name = simplify_name(get_name(strings, name_id))
        group = groups.setdefault(name, {"durations": [], "uncovered": 0})
        group["durations"].append(duration)
        group["uncovered"] += uncovered
        durations.append(duration)
        uncovered_total += uncovered
    total = sum(durations)
    rows = []
    for name, item in groups.items():
        row_total = sum(item["durations"])
        row = summary_from_durations(item["durations"])
        row.update(
            {
                "name": name,
                "uncovered_ms": ms(item["uncovered"]),
                "share_pct": row_total / total * 100 if total else 0.0,
                "uncovered_share_pct": item["uncovered"] / row_total * 100 if row_total else 0.0,
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: item["share_pct"], reverse=True)
    summary = summary_from_durations(durations)
    return {
        "count": summary["count"],
        "unique_names": len(groups),
        "total_ms": summary["total_ms"],
        "uncovered_ms": ms(uncovered_total),
        "avg_ms": summary["avg_ms"],
        "p90_ms": summary["p90_ms"],
        "max_ms": summary["max_ms"],
        "top": rows,
    }


def parse_extra(extra):
    if not extra:
        return {}
    try:
        return json.loads(extra)
    except Exception:
        return {}

def memcpy_summary(cur, start, end, compute_intervals=None, top=20):
    table = "device_task_memcpy_data"
    if not table_exists(cur, table):
        return {
            "count": 0,
            "total_ms": 0.0,
            "uncovered_ms": 0.0,
            "avg_ms": 0.0,
            "p90_ms": 0.0,
            "max_ms": 0.0,
            "bytes": 0,
            "top": [],
        }
    compute_intervals = compute_intervals or []
    cols = table_columns(cur, table)
    select_bytes = "bytes" if "bytes" in cols else "NULL AS bytes"
    select_type = "type" if "type" in cols else "NULL AS type"
    rows = cur.execute(
        f"""
        SELECT start, end, {select_type}, {select_bytes}, extra
        FROM {table}
        WHERE start >= :window_start AND start < :window_end
        """,
        window_params(start, end),
    ).fetchall()
    groups = {}
    durations = []
    byte_total = 0
    uncovered_total = 0
    for row_start, row_end, copy_type_id, byte_count, extra in rows:
        duration = clipped_duration(row_start, row_end, start, end)
        if duration <= 0:
            continue
        extra_obj = parse_extra(extra)
        copy_type = extra_obj.get("copy_type") or (f"type={copy_type_id}" if copy_type_id is not None else "unknown")
        bytes_value = byte_count or extra_obj.get("bytes", 0) or 0
        uncovered = uncovered_ns(row_start, row_end, compute_intervals, start, end)
        group = groups.setdefault(copy_type, {"name": copy_type, "durations": [], "uncovered": 0, "bytes": 0})
        group["durations"].append(duration)
        group["uncovered"] += uncovered
        group["bytes"] += bytes_value
        durations.append(duration)
        uncovered_total += uncovered
        byte_total += bytes_value
    total = sum(durations)
    top_rows = sorted(groups.values(), key=lambda item: sum(item["durations"]), reverse=True)
    summary = summary_from_durations(durations)
    return {
        "count": summary["count"],
        "total_ms": summary["total_ms"],
        "uncovered_ms": ms(uncovered_total),
        "avg_ms": summary["avg_ms"],
        "p90_ms": summary["p90_ms"],
        "max_ms": summary["max_ms"],
        "bytes": byte_total,
        "top": [
            {
                "name": item["name"],
                "count": len(item["durations"]),
                "total_ms": ms(sum(item["durations"])),
                "uncovered_ms": ms(item["uncovered"]),
                "avg_ms": ms(sum(item["durations"]) / len(item["durations"])) if item["durations"] else 0.0,
                "p90_ms": ms(percentile_ns(item["durations"], 90)) if item["durations"] else 0.0,
                "max_ms": ms(max(item["durations"])) if item["durations"] else 0.0,
                "bytes": item["bytes"],
                "avg_bytes": item["bytes"] / len(item["durations"]) if item["durations"] else 0.0,
                "bandwidth_gbps": (item["bytes"] / 1e9) / (sum(item["durations"]) / 1e9) if sum(item["durations"]) else None,
                "share_pct": sum(item["durations"]) / total * 100 if total else 0.0,
                "uncovered_share_pct": item["uncovered"] / sum(item["durations"]) * 100 if sum(item["durations"]) else 0.0,
            }
            for item in top_rows
        ],
    }


def other_activity_summary(cur, start, end, preloaded_intervals):
    intervals = []
    durations = []
    for table in ("device_task_memset_data", "device_task_atomic_operation_data", "device_task_notifier_data"):
        table_intervals = preloaded_intervals.get(table, [])
        intervals.extend(table_intervals)
        durations.extend(interval_end - interval_start for interval_start, interval_end in table_intervals)
    return summary_from_durations(durations)


def pure_gap_summary(cur, start, end, preloaded_intervals):
    covered = []
    for table in ("device_task_kernel_data", "device_task_memcpy_data", "device_task_memset_data",
                  "device_task_atomic_operation_data", "device_task_notifier_data"):
        covered.extend(preloaded_intervals.get(table, []))
    return summary_from_intervals(complement_intervals(start, end, covered))


def get_invoke_end(cur, corr_id):
    if not table_exists(cur, "function_data"):
        return None
    row = cur.execute("SELECT end FROM function_data WHERE correlationId = ?", (corr_id,)).fetchone()
    return row[0] if row else None


def blocking_event_reason(cur, process_id, device_id, queue_id, gap_start, gap_end):
    candidates = []
    if table_exists(cur, "device_task_notifier_data"):
        candidates.append(
            (
                "notifier_blocking",
                """
                SELECT start, end
                FROM device_task_notifier_data
                WHERE processId = ? AND deviceId = ? AND queueId = ?
                  AND type = 0 AND start >= ? AND start <= ?
                ORDER BY start DESC, end DESC
                LIMIT 1
                """,
            )
        )
    if table_exists(cur, "device_task_atomic_operation_data"):
        candidates.append(
            (
                "atomicOp_blocking",
                """
                SELECT start, end
                FROM device_task_atomic_operation_data
                WHERE processId = ? AND deviceId = ? AND queueId = ?
                  AND start >= ? AND start <= ?
                ORDER BY start DESC, end DESC
                LIMIT 1
                """,
            )
        )
    if table_exists(cur, "device_task_memcpy_data"):
        async_clause = "AND isAsync = 1" if "isAsync" in table_columns(cur, "device_task_memcpy_data") else ""
        candidates.append(
            (
                "memcpy_blocking",
                f"""
                SELECT start, end
                FROM device_task_memcpy_data
                WHERE processId = ? AND deviceId = ? AND queueId = ?
                  {async_clause}
                  AND start >= ? AND start <= ?
                ORDER BY start DESC, end DESC
                LIMIT 1
                """,
            )
        )
    best = None
    for reason, query in candidates:
        row = cur.execute(query, (process_id, device_id, queue_id, gap_start, gap_end)).fetchone()
        if not row:
            continue
        if best is None or row[0] > best[1]:
            best = (reason, row[0])
    return best[0] if best else "other"


def compute_gap_summary(cur, strings, start, end, preloaded_rows, top=20):
    coverage_by_group = {}
    reasons = {}
    gaps = []
    for process_id, device_id, queue_id, k_start, k_end, corr_id, name_id in preloaded_rows:
        group = (process_id, device_id)
        coverage = coverage_by_group.get(group)
        if coverage is None:
            coverage_by_group[group] = {
                "end": k_end,
                "end_kernel": (queue_id, k_end, corr_id, name_id),
            }
            continue
        if k_start <= coverage["end"]:
            if k_end > coverage["end"]:
                coverage["end"] = k_end
                coverage["end_kernel"] = (queue_id, k_end, corr_id, name_id)
            continue
        prev_queue, prev_end, prev_corr, prev_name = coverage["end_kernel"]
        gap_start = max(prev_end, start)
        gap_end = min(k_start, end)
        gap = gap_end - gap_start
        coverage["end"] = k_end
        coverage["end_kernel"] = (queue_id, k_end, corr_id, name_id)
        if gap <= 0:
            continue
        if gap < DEFAULT_GAP_THRESHOLD_NS:
            reason = "mini_gap"
        else:
            invoke_end = get_invoke_end(cur, corr_id)
            if invoke_end is not None and k_start - invoke_end <= DEFAULT_INVOKE_THRESHOLD_NS:
                reason = "host_blocking"
            else:
                reason = blocking_event_reason(cur, process_id, device_id, queue_id, prev_end, k_start)
        item = reasons.setdefault(reason, {"count": 0, "total": 0, "max": 0})
        item["count"] += 1
        item["total"] += gap
        item["max"] = max(item["max"], gap)
        gaps.append(
            {
                "reason": reason,
                "gap_type": "compute_coverage_gap",
                "duration_ms": ms(gap),
                "start_ms": ms(gap_start),
                "process_id": process_id,
                "device_id": device_id,
                "prev": simplify_name(get_name(strings, prev_name)),
                "curr": simplify_name(get_name(strings, name_id)),
                "prev_corr": prev_corr,
                "curr_corr": corr_id,
            }
        )
    total = sum(item["total"] for item in reasons.values())
    gap_durations = [int(item["duration_ms"] * 1e6) for item in gaps]
    formatted_reasons = {
        reason: {
            "count": item["count"],
            "total_ms": ms(item["total"]),
            "avg_ms": ms(item["total"] / item["count"]) if item["count"] else 0.0,
            "max_ms": ms(item["max"]),
            "share_pct": item["total"] / total * 100 if total else 0.0,
        }
        for reason, item in sorted(reasons.items(), key=lambda kv: kv[1]["total"], reverse=True)
    }
    return {
        "count": len(gaps),
        "total_ms": ms(total),
        "avg_ms": ms(total / len(gaps)) if gaps else 0.0,
        "p90_ms": ms(percentile_ns(gap_durations, 90)) if gaps else 0.0,
        "max_ms": max((item["duration_ms"] for item in gaps), default=0.0),
        "reasons": formatted_reasons,
        "top": sorted(gaps, key=lambda item: item["duration_ms"], reverse=True)[:top],
    }


def host_summary(cur, strings, start, end, top=DEFAULT_HOST_ROWS):
    function_stats = aggregate_named(cur, "function_data", strings, start, end, top=top)
    internal_stats = aggregate_named(cur, "Internal_operation_range_data", strings, start, end, top=top)
    return {
        "function": function_stats,
        "annotation": internal_stats,
        "overlap_warning": "function total duration can double count overlapping threads and nested calls",
    }


def _load_device_data(cur, start, end):
    """Load all device-side data in one pass. Returns raw rows and intervals."""
    # 1. 统一查询 kernel 数据
    kernel_rows = []
    if table_exists(cur, "device_task_kernel_data"):
        kernel_rows = cur.execute(
            """
            SELECT nameId, start, end, isComputation, processId, deviceId, queueId, correlationId
            FROM device_task_kernel_data
            WHERE start >= :window_start AND start < :window_end
            ORDER BY processId, deviceId, start
            """,
            window_params(start, end),
        ).fetchall()

    compute_rows = [(name_id, k_start, k_end)
                    for name_id, k_start, k_end, is_comp, *_ in kernel_rows
                    if is_comp == 1]
    comm_rows = [(name_id, k_start, k_end)
                 for name_id, k_start, k_end, is_comp, *_ in kernel_rows
                 if is_comp == 0]

    # 2. 预加载小表 intervals
    preloaded_intervals = {}
    for table in ("device_task_memset_data", "device_task_atomic_operation_data", "device_task_notifier_data"):
        if table_exists(cur, table):
            preloaded_intervals[table] = clipped_intervals(cur, table, start, end)
    if table_exists(cur, "device_task_kernel_data"):
        preloaded_intervals["device_task_kernel_data"] = [
            (max(s, start), min(e, end))
            for _, s, e, *_ in kernel_rows
            if min(e, end) > max(s, start)
        ]
    if table_exists(cur, "device_task_memcpy_data"):
        preloaded_intervals["device_task_memcpy_data"] = clipped_intervals(cur, "device_task_memcpy_data", start, end)

    return {
        "kernel_rows": kernel_rows,
        "compute_rows": compute_rows,
        "comm_rows": comm_rows,
        "compute_intervals": merge_intervals([(s, e) for _, s, e in compute_rows]),
        "preloaded_intervals": preloaded_intervals,
    }


def classify_kernel(name):
    low = name.lower()
    is_triton = "triton" in low
    is_fused = "fused" in low
    if is_triton and is_fused:
        return "triton_fused"
    if is_triton:
        return "triton_other"
    return "non_triton"


def find_key(d, candidates):
    lowered = {k.lower(): k for k in d.keys()}
    for cand in candidates:
        if cand in lowered:
            return d[lowered[cand]]
    return None


def to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def get_device_name(cur):
    if table_exists(cur, "device_information"):
        row = cur.execute(
            "SELECT name FROM device_information WHERE name IS NOT NULL LIMIT 1"
        ).fetchone()
        if row and row[0]:
            return row[0]
    if table_exists(cur, "meta_information"):
        row = cur.execute(
            "SELECT value FROM meta_information WHERE type = 'deviceInfo'"
        ).fetchone()
        if row:
            try:
                return json.loads(row[0]).get("m_dev_basic_info", {}).get("dev_name")
            except Exception:
                return None
    return None


def resolve_peak_bandwidth(cur):
    """Prefer MLU-model theoretical bandwidth; fall back to meta max_bandwidth."""
    device_name = get_device_name(cur)
    if device_name:
        for key, bw in THEORETICAL_BANDWIDTH.items():
            if key in device_name:
                return bw, device_name, "theoretical(MLU model)"
    if table_exists(cur, "meta_information"):
        row = cur.execute(
            "SELECT value FROM meta_information WHERE type = 'deviceInfo'"
        ).fetchone()
        if row:
            try:
                meta = json.loads(row[0]).get("m_dev_basic_info", {}).get("max_bandwidth")
                if meta is not None:
                    return meta, device_name, "meta max_bandwidth"
            except Exception:
                pass
    return None, device_name, None


def _exec_tasks_with_queue(cur, start, end):
    rows = []
    if table_exists(cur, "device_task_kernel_data"):
        rows.extend(
            (pid, did, qid, s, e, isc)
            for pid, did, qid, s, e, isc in cur.execute(
                "SELECT processId, deviceId, queueId, start, end, isComputation "
                "FROM device_task_kernel_data WHERE start >= :window_start AND start < :window_end",
                window_params(start, end),
            )
        )
    for table in ("device_task_memcpy_data", "device_task_memset_data", "device_task_atomic_operation_data"):
        if not table_exists(cur, table):
            continue
        rows.extend(
            (pid, did, qid, s, e, 0)
            for pid, did, qid, s, e in cur.execute(
                f"SELECT processId, deviceId, queueId, start, end "
                f"FROM {table} WHERE start >= :window_start AND start < :window_end",
                window_params(start, end),
            )
        )
    return rows


def device_stream_gap_summary(cur, start, end):
    """Per-queue (device stream) gap ratio within the window. Key host-overhead signal."""
    rows = _exec_tasks_with_queue(cur, start, end)
    queues = {}
    device_intervals = {}
    for pid, did, qid, s, e, isc in rows:
        cs, ce = max(s, start), min(e, end)
        if ce <= cs:
            continue
        item = queues.setdefault((pid, did, qid), {"intervals": [], "compute": 0})
        item["intervals"].append((cs, ce))
        if isc == 1:
            item["compute"] += ce - cs
        device_intervals.setdefault((pid, did), []).append((cs, ce))

    queue_rows = []
    for (pid, did, qid), item in queues.items():
        span = max(e for _, e in item["intervals"]) - min(s for s, _ in item["intervals"])
        busy = interval_total(merge_intervals(item["intervals"]))
        gap = span - busy
        queue_rows.append(
            {
                "process_id": pid,
                "device_id": did,
                "queue_id": qid,
                "span_ms": ms(span),
                "busy_ms": ms(busy),
                "gap_ms": ms(gap),
                "gap_pct": (gap / span * 100) if span > 0 else 0.0,
                "compute_ms": ms(item["compute"]),
            }
        )
    queue_rows.sort(key=lambda r: -(r["compute_ms"] or 0.0))
    for i, row in enumerate(queue_rows):
        row["is_main_compute_stream"] = i == 0
    main_stream_gap_pct = queue_rows[0]["gap_pct"] if queue_rows else 0.0

    window = end - start
    device_busy = sum(
        interval_total(merge_intervals(iv)) for iv in device_intervals.values()
    ) / max(1, len(device_intervals))
    device_gap_pct = ((window - device_busy) / window * 100) if window > 0 else 0.0

    return {
        "main_stream_gap_pct": main_stream_gap_pct,
        "device_gap_pct": device_gap_pct,
        "queues": queue_rows,
    }


def triton_fusion_summary(cur, strings, start, end, compute_rows, top=20):
    by_class = {"triton_fused": 0, "triton_other": 0, "non_triton": 0}
    by_name = {}
    for name_id, row_start, row_end in compute_rows:
        duration = clipped_duration(row_start, row_end, start, end)
        if duration <= 0:
            continue
        name = simplify_name(get_name(strings, name_id))
        cls = classify_kernel(name)
        by_class[cls] += duration
        item = by_name.setdefault(name, {"class": cls, "total": 0, "count": 0})
        item["total"] += duration
        item["count"] += 1
    total = sum(by_class.values())
    non_fused = sorted(
        (
            {
                "name": name,
                "class": item["class"],
                "total_ms": ms(item["total"]),
                "count": item["count"],
                "share_pct": item["total"] / total * 100 if total else 0.0,
            }
            for name, item in by_name.items()
            if item["class"] != "triton_fused"
        ),
        key=lambda r: -(r["total_ms"] or 0.0),
    )
    return {
        "uses_torch_compile": bool(by_class["triton_fused"] or by_class["triton_other"]),
        "compute_total_ms": ms(total),
        "fusion_coverage_pct": by_class["triton_fused"] / total * 100 if total else 0.0,
        "triton_fused_ms": ms(by_class["triton_fused"]),
        "triton_other_ms": ms(by_class["triton_other"]),
        "non_triton_ms": ms(by_class["non_triton"]),
        "top_non_fused": non_fused[:top],
    }


def compile_segmentation_summary(cur, strings, start, end, kernel_rows, stream_gap):
    regions_by_thread = {}
    region_inventory = {}
    recompile_count = 0
    if table_exists(cur, "Internal_operation_range_data"):
        for pid, tid, r_start, r_end, name_id in cur.execute(
            "SELECT processId, threadId, start, end, nameId FROM Internal_operation_range_data "
            "WHERE start < :window_end AND end > :window_start",
            window_params(start, end),
        ):
            name = simplify_name(get_name(strings, name_id))
            if RECOMPILE_REGEX.search(name):
                recompile_count += 1
            if not COMPILE_REGION_REGEX.search(name):
                continue
            regions_by_thread.setdefault((pid, tid), []).append((r_start, r_end))
            norm = re.sub(r"[:#]\s*\d+\s*$", "", name).strip() or name
            inv = region_inventory.setdefault(norm, {"count": 0, "host_ns": 0})
            inv["count"] += 1
            inv["host_ns"] += max(0, min(r_end, end) - max(r_start, start))

    region_sets = {}
    for key, intervals in regions_by_thread.items():
        merged = merge_intervals(intervals)
        region_sets[key] = (merged, [s for s, _ in merged])

    def in_region(pid, tid, point):
        entry = region_sets.get((pid, tid))
        if not entry:
            return False
        merged, starts = entry
        idx = bisect.bisect_right(starts, point) - 1
        if idx < 0:
            return False
        s, e = merged[idx]
        return s <= point < e

    launch_by_corr = {}
    if table_exists(cur, "function_data"):
        for corr, pid, tid, f_start, self_ns in cur.execute(
            "SELECT correlationId, processId, threadId, start, self FROM function_data "
            "WHERE start >= :window_start AND start < :window_end",
            window_params(start, end),
        ):
            launch_by_corr[corr] = (pid, tid, f_start, self_ns)

    inside_ns = outside_ns = 0
    inside_cnt = outside_cnt = 0
    total_launch_self_ns = 0
    launched = 0
    for name_id, k_start, k_end, is_comp, pid, did, qid, corr in kernel_rows:
        if is_comp != 1:
            continue
        dur = clipped_duration(k_start, k_end, start, end)
        if dur <= 0:
            continue
        launch = launch_by_corr.get(corr)
        if launch is None:
            outside_ns += dur
            outside_cnt += 1
            continue
        lpid, ltid, lstart, self_ns = launch
        total_launch_self_ns += self_ns or 0
        launched += 1
        if in_region(lpid, ltid, lstart):
            inside_ns += dur
            inside_cnt += 1
        else:
            outside_ns += dur
            outside_cnt += 1

    device_compute_ns = inside_ns + outside_ns
    return {
        "has_compiled_regions": bool(region_inventory),
        "compiled_region_count": sum(v["count"] for v in region_inventory.values()),
        "recompile_indicator_count": recompile_count,
        "inside_region_compute_ms": ms(inside_ns),
        "inside_region_kernel_count": inside_cnt,
        "outside_region_compute_ms": ms(outside_ns),
        "outside_region_kernel_count": outside_cnt,
        "outside_region_share_pct": (outside_ns / device_compute_ns * 100) if device_compute_ns else 0.0,
        "host_launch_overhead": {
            "main_stream_gap_pct": stream_gap.get("main_stream_gap_pct", 0.0),
            "launched_kernel_count": launched,
            "avg_launch_self_us": (total_launch_self_ns / launched / 1000.0) if launched else 0.0,
            "total_launch_self_ms": ms(total_launch_self_ns),
            "device_compute_ms": ms(device_compute_ns),
            "launch_self_to_compute_ratio": (total_launch_self_ns / device_compute_ns) if device_compute_ns else 0.0,
        },
        "region_inventory": sorted(
            ({"region": k, "count": v["count"], "host_ms": ms(v["host_ns"])} for k, v in region_inventory.items()),
            key=lambda r: -(r["host_ms"] or 0.0),
        ),
    }


def triton_kernel_efficiency_summary(cur, strings, start, end, peak_bandwidth, top=20):
    if not table_exists(cur, "device_task_kernel_data"):
        return {"has_io_metadata": False, "observed_keys": [], "kernels": []}
    observed = set()
    by_name = {}
    for name_id, k_start, k_end, extra in cur.execute(
        "SELECT nameId, start, end, extra FROM device_task_kernel_data "
        "WHERE isComputation = 1 AND start >= :window_start AND start < :window_end",
        window_params(start, end),
    ):
        name = simplify_name(get_name(strings, name_id))
        if "triton" not in name.lower():
            continue
        extra_obj = parse_extra(extra)
        if not extra_obj:
            continue
        io_val = to_float(find_key(extra_obj, IO_EFF_KEYS))
        bw_val = to_float(find_key(extra_obj, BANDWIDTH_KEYS))
        if io_val is None and bw_val is None:
            continue
        for cand_set, present in ((IO_EFF_KEYS, io_val), (BANDWIDTH_KEYS, bw_val)):
            if present is not None:
                for k in cand_set:
                    if k in {kk.lower() for kk in extra_obj}:
                        observed.add(k)
                        break
        dur = clipped_duration(k_start, k_end, start, end)
        if dur <= 0:
            continue
        item = by_name.setdefault(name, {"dur": 0, "count": 0, "io": [], "bw": []})
        item["dur"] += dur
        item["count"] += 1
        if io_val is not None:
            item["io"].append((io_val, dur))
        if bw_val is not None:
            item["bw"].append((bw_val, dur))

    if not by_name:
        return {"has_io_metadata": False, "observed_keys": [], "kernels": []}

    def wavg(pairs):
        w = sum(x[1] for x in pairs)
        return sum(v * wt for v, wt in pairs) / w if w else None

    kernels = []
    for name, item in by_name.items():
        io_eff = wavg(item["io"]) if item["io"] else None
        util = (io_eff / float(peak_bandwidth)) if (io_eff is not None and peak_bandwidth) else None
        kernels.append(
            {
                "name": name,
                "count": item["count"],
                "total_ms": ms(item["dur"]),
                "avg_io_efficiency": io_eff,
                "avg_achieved_bandwidth": wavg(item["bw"]) if item["bw"] else None,
                "bandwidth_utilization": util,
            }
        )
    kernels.sort(key=lambda k: (k["bandwidth_utilization"] if k["bandwidth_utilization"] is not None else float("inf"), -(k["total_ms"] or 0.0)))
    return {
        "has_io_metadata": True,
        "observed_keys": sorted(observed),
        "kernels": kernels[:top],
    }


def device_summary(cur, strings, start, end):
    data = _load_device_data(cur, start, end)
    kernel_rows = data["kernel_rows"]
    compute_rows = data["compute_rows"]
    comm_rows = data["comm_rows"]
    compute_index = interval_index(data["compute_intervals"])
    preloaded_intervals = data["preloaded_intervals"]

    return {
        "compute_kernel": aggregate_named(
            cur, "device_task_kernel_data", strings, start, end, "isComputation = 1",
            preloaded_rows=compute_rows,
        ),
        "communication_kernel": communication_kernel_summary(
            cur, strings, start, end, compute_index,
            preloaded_rows=comm_rows,
        ),
        "memcpy": memcpy_summary(cur, start, end, compute_index),
        "compute_gap": compute_gap_summary(
            cur, strings, start, end,
            preloaded_rows=[(pid, did, qid, k_start, k_end, corr_id, name_id)
                           for name_id, k_start, k_end, is_comp, pid, did, qid, corr_id in kernel_rows
                           if is_comp == 1],
        ),
        "pure_gap": pure_gap_summary(cur, start, end, preloaded_intervals),
        "other_activity": other_activity_summary(cur, start, end, preloaded_intervals),
        "device_stream_gap": device_stream_gap_summary(cur, start, end),
    }


def torch_compile_summary(cur, strings, start, end):
    data = _load_device_data(cur, start, end)
    kernel_rows = data["kernel_rows"]
    compute_rows = data["compute_rows"]
    stream_gap = device_stream_gap_summary(cur, start, end)
    peak_bandwidth, device_name, peak_source = resolve_peak_bandwidth(cur)
    return {
        "device_name": device_name,
        "peak_bandwidth": peak_bandwidth,
        "peak_bandwidth_source": peak_source,
        "fusion": triton_fusion_summary(cur, strings, start, end, compute_rows),
        "segmentation": compile_segmentation_summary(cur, strings, start, end, kernel_rows, stream_gap),
        "kernel_efficiency": triton_kernel_efficiency_summary(cur, strings, start, end, peak_bandwidth),
    }


def analyze_db(db_path, start_ms, end_ms):
    if end_ms <= start_ms:
        raise ValueError("--end-ms must be greater than --start-ms")
    start = int(start_ms * 1_000_000)
    end = int(end_ms * 1_000_000)
    with connect_db(db_path) as conn:
        cur = conn.cursor()
        strings = load_strings(cur)
        return {
            "label": Path(db_path).stem,
            "db": str(db_path),
            "strings_available": bool(strings),
            "range": {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
            },
            "host": host_summary(cur, strings, start, end),
            "device": device_summary(cur, strings, start, end),
            "torch_compile": torch_compile_summary(cur, strings, start, end),
        }


def print_metric_basic(title, rows):
    print(f"\n### {title}")
    print("| Metric | Value |")
    print("|---|---:|")
    for name, value in rows:
        print(f"| {name} | {value:,.3f} |")


def range_share(item, profile):
    duration = profile["range"].get("duration_ms", 0.0) or 0.0
    return (item.get("total_ms", 0.0) or 0.0) / duration * 100 if duration else 0.0


def print_device_overview_table(side):
    print(f"\n### {side['label']} Device Breakdown Overview")
    print("| Category | Total ms | Count | Avg ms | Max ms | Range share |")
    print("|---|---:|---:|---:|---:|---:|")
    rows = (
        ("compute kernel", side["device"]["compute_kernel"]),
        ("communication kernel", side["device"]["communication_kernel"]),
        ("memcpy", side["device"]["memcpy"]),
        ("compute gap", side["device"]["compute_gap"]),
        ("pure gap", side["device"]["pure_gap"]),
        ("other activity", side["device"]["other_activity"]),
    )
    for category, item in rows:
        print(
            f"| {category} | {fmt_ms(item.get('total_ms', 0.0))} | "
            f"{item.get('count', 0):,.0f} | {fmt_ms(item.get('avg_ms', 0.0))} | "
            f"{fmt_ms(item.get('max_ms', 0.0))} | {range_share(item, side):.2f}% |"
        )


def compact_rows_to_share(rows, threshold=95.0):
    output = []
    other = None
    cumulative = 0.0
    for row in rows:
        share = row.get("share_pct", 0.0) or 0.0
        if cumulative < threshold:
            output.append(row)
            cumulative += share
            continue
        if other is None:
            other = {
                "name": "other",
                "count": 0,
                "total_ms": 0.0,
                "uncovered_ms": 0.0,
                "max_ms": 0.0,
                "bytes": 0,
                "share_pct": 0.0,
            }
        other["count"] += row.get("count", 0) or 0
        other["total_ms"] += row.get("total_ms", 0.0) or 0.0
        other["uncovered_ms"] += row.get("uncovered_ms", 0.0) or 0.0
        other["max_ms"] = max(other["max_ms"], row.get("max_ms", 0.0) or 0.0)
        other["bytes"] += row.get("bytes", 0) or 0
        other["share_pct"] += share
    if other and other["count"]:
        other["avg_ms"] = other["total_ms"] / other["count"]
        other["p90_ms"] = None
        other["avg_bytes"] = other["bytes"] / other["count"] if other["count"] else 0.0
        other["bandwidth_gbps"] = (other["bytes"] / 1e9) / (other["total_ms"] / 1000.0) if other["total_ms"] else None
        other["uncovered_share_pct"] = other["uncovered_ms"] / other["total_ms"] * 100 if other["total_ms"] else 0.0
        output.append(other)
    return output


def visible_rows(rows):
    return compact_rows_to_share(rows)


def print_named_basic_table(title, rows, kind=None):
    print(f"\n### {title}")
    if not rows:
        print("No rows.")
        return
    if kind is None:
        print("| Name | Total ms | Count | Avg ms | Max ms |")
        print("|---|---:|---:|---:|---:|")
        for row in rows:
            print(
                f"| {row['name']} | {fmt_ms(row.get('total_ms', 0.0))} | "
                f"{row.get('count', 0):,.0f} | {fmt_ms(row.get('avg_ms', 0.0))} | "
                f"{fmt_ms(row.get('max_ms', 0.0))} |"
            )
        return
    print("| Kind | Name | Total ms | Count | Avg ms | Max ms |")
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {kind} | {row['name']} | {fmt_ms(row.get('total_ms', 0.0))} | "
            f"{row.get('count', 0):,.0f} | {fmt_ms(row.get('avg_ms', 0.0))} | "
            f"{fmt_ms(row.get('max_ms', 0.0))} |"
        )


def print_compute_kernel_summary(side):
    print(f"\n### {side['label']} Compute Kernel Summary")
    rows = visible_rows(side["device"]["compute_kernel"]["top"])
    if not rows:
        print("No rows.")
        return
    print("| Kernel name | Total ms | Count | Avg ms | P90 ms | Max ms | Share |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['name']} | {fmt_ms(row.get('total_ms', 0.0))} | {row.get('count', 0):,.0f} | "
            f"{fmt_ms(row.get('avg_ms', 0.0))} | {fmt_ms(row.get('p90_ms'))} | "
            f"{fmt_ms(row.get('max_ms', 0.0))} | {row.get('share_pct', 0.0):.2f}% |"
        )


def print_communication_kernel_summary(side):
    print(f"\n### {side['label']} Communication Kernel Summary")
    rows = visible_rows(side["device"]["communication_kernel"]["top"])
    if not rows:
        print("No rows.")
        return
    print("| Kernel name | Total ms | Uncovered ms | Count | Avg ms | P90 ms | Max ms | Share | Uncovered share |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['name']} | {fmt_ms(row.get('total_ms', 0.0))} | {fmt_ms(row.get('uncovered_ms', 0.0))} | "
            f"{row.get('count', 0):,.0f} | {fmt_ms(row.get('avg_ms', 0.0))} | "
            f"{fmt_ms(row.get('p90_ms'))} | {fmt_ms(row.get('max_ms', 0.0))} | "
            f"{row.get('share_pct', 0.0):.2f}% | {row.get('uncovered_share_pct', 0.0):.2f}% |"
        )


def print_memcpy_summary(side):
    print(f"\n### {side['label']} Memcpy Summary")
    rows = visible_rows(side["device"]["memcpy"]["top"])
    if not rows:
        print("No rows.")
        return
    print("| Copy type | Total ms | Uncovered ms | Count | Avg ms | P90 ms | Max ms | Total bytes | Avg bytes | Bandwidth GB/s | Share | Uncovered share |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        bandwidth = row.get("bandwidth_gbps")
        bandwidth_text = "n/a" if bandwidth is None else f"{bandwidth:,.3f}"
        print(
            f"| {row['name']} | {fmt_ms(row.get('total_ms', 0.0))} | {fmt_ms(row.get('uncovered_ms', 0.0))} | "
            f"{row.get('count', 0):,.0f} | {fmt_ms(row.get('avg_ms', 0.0))} | "
            f"{fmt_ms(row.get('p90_ms'))} | {fmt_ms(row.get('max_ms', 0.0))} | "
            f"{row.get('bytes', 0):,.0f} | {row.get('avg_bytes', 0.0):,.1f} | {bandwidth_text} | "
            f"{row.get('share_pct', 0.0):.2f}% | {row.get('uncovered_share_pct', 0.0):.2f}% |"
        )


def print_family_basic_table(title, family_map):
    print(f"\n### {title}")
    if not family_map:
        print("No rows.")
        return
    print("| Family | Total ms | Count | Unique names | Avg ms | Share |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, row in family_map.items():
        print(
            f"| {name} | {fmt_ms(row.get('total_ms', 0.0))} | "
            f"{row.get('count', 0):,.0f} | {row.get('unique_names', 0)} | "
            f"{fmt_ms(row.get('avg_ms', 0.0))} | {row.get('share_pct', 0.0):.2f}% |"
        )


def print_device_stream_gap(side):
    sg = side["device"].get("device_stream_gap")
    if not sg:
        return
    print(f"\n### {side['label']} Device Stream Gap Ratio  <-- key host-overhead indicator")
    print(
        f"- main compute stream gap ratio: {sg['main_stream_gap_pct']:.2f}%  "
        f"(device-level gap ratio: {sg['device_gap_pct']:.2f}%)"
    )
    print("| Queue | Span ms | Gap ms | Gap% | Compute ms | Main |")
    print("|---:|---:|---:|---:|---:|:--:|")
    for row in sg["queues"]:
        main = "*" if row.get("is_main_compute_stream") else ""
        print(
            f"| {row['queue_id']} | {fmt_ms(row.get('span_ms', 0.0))} | "
            f"{fmt_ms(row.get('gap_ms', 0.0))} | {row.get('gap_pct', 0.0):.2f}% | "
            f"{fmt_ms(row.get('compute_ms', 0.0))} | {main} |"
        )


def print_torch_compile(side):
    tc = side.get("torch_compile")
    if not tc:
        return
    fusion = tc.get("fusion", {})
    seg = tc.get("segmentation", {})
    ke = tc.get("kernel_efficiency", {})
    if not (fusion.get("uses_torch_compile") or seg.get("has_compiled_regions") or ke.get("has_io_metadata")):
        return
    print(f"\n### {side['label']} torch.compile / triton")
    print(
        f"- fusion coverage: {fusion.get('fusion_coverage_pct', 0.0):.2f}% "
        f"(fused {fmt_ms(fusion.get('triton_fused_ms', 0.0))} ms, "
        f"non-triton {fmt_ms(fusion.get('non_triton_ms', 0.0))} ms)"
    )
    hlo = seg.get("host_launch_overhead", {})
    print(
        f"- host launch overhead: main stream gap {hlo.get('main_stream_gap_pct', 0.0):.2f}%, "
        f"avg launch self {hlo.get('avg_launch_self_us', 0.0):.2f} us, "
        f"launch/compute ratio {hlo.get('launch_self_to_compute_ratio', 0.0):.2f}"
    )
    print(
        f"- compiled regions: {seg.get('compiled_region_count', 0)}, "
        f"outside/eager compute {fmt_ms(seg.get('outside_region_compute_ms', 0.0))} ms "
        f"({seg.get('outside_region_share_pct', 0.0):.1f}%), recompile indicators {seg.get('recompile_indicator_count', 0)}"
    )
    print(
        f"- device: {tc.get('device_name')}, peak bandwidth {tc.get('peak_bandwidth')} "
        f"(source: {tc.get('peak_bandwidth_source')})"
    )
    if fusion.get("top_non_fused"):
        print("\n#### Top Non-Fused Kernels")
        print("| Kernel | Class | Total ms | Count | Share |")
        print("|---|---|---:|---:|---:|")
        for row in fusion["top_non_fused"][:10]:
            print(
                f"| {row['name']} | {row['class']} | {fmt_ms(row.get('total_ms', 0.0))} | "
                f"{row.get('count', 0):,.0f} | {row.get('share_pct', 0.0):.2f}% |"
            )
    if ke.get("has_io_metadata") and ke.get("kernels"):
        print("\n#### Triton Kernel IO Efficiency (io_efficiency = folded bandwidth, not %)")
        print("| Kernel | Count | Total ms | Folded BW (io_eff) | Utilization |")
        print("|---|---:|---:|---:|---:|")
        for row in ke["kernels"][:10]:
            io_eff = "n/a" if row.get("avg_io_efficiency") is None else f"{row['avg_io_efficiency']:,.2f}"
            util = "n/a" if row.get("bandwidth_utilization") is None else f"{row['bandwidth_utilization']*100:.1f}%"
            print(f"| {row['name']} | {row.get('count', 0):,.0f} | {fmt_ms(row.get('total_ms', 0.0))} | {io_eff} | {util} |")


def print_single_file_tables(side):
    print(f"\n## {side['label']} Breakdown Tables")
    print_device_overview_table(side)
    print_device_stream_gap(side)
    print_compute_kernel_summary(side)
    print_communication_kernel_summary(side)
    print_memcpy_summary(side)
    print_torch_compile(side)
    print_named_basic_table(f"{side['label']} Host Function Summary", side["host"]["function"]["top"])
    print_named_basic_table(
        f"{side['label']} Host annotation Summary",
        side["host"]["annotation"]["top"],
    )


def print_text_report(report):
    profile = report["profile"]
    selected_range = profile["range"]
    print("# E2E Profiling Breakdown Tables")
    print("\n## Inputs")
    print(f"- {profile['label']}: {profile['db']}")
    print(
        f"- Range: {fmt_ms(selected_range['start_ms'])} ms -> "
        f"{fmt_ms(selected_range['end_ms'])} ms "
        f"(duration {fmt_ms(selected_range['duration_ms'])} ms)"
    )
    print_single_file_tables(profile)


def build_report(args):
    db_path = normalize_input(args.input)
    start_ms, end_ms = args.start_ms, args.end_ms
    if start_ms is None or end_ms is None:
        with connect_db(db_path) as conn:
            cur = conn.cursor()
            db_start, db_end = get_db_time_range(cur)
        db_start_ms = db_start / 1e6
        db_end_ms = db_end / 1e6
        start_ms = start_ms if start_ms is not None else db_start_ms
        end_ms = end_ms if end_ms is not None else db_end_ms
    profile = analyze_db(db_path, start_ms, end_ms)
    return {
        "profile": profile,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="cnperf SQLite DB")
    parser.add_argument("--start-ms", type=float, default=None, help="Selected range start timestamp in milliseconds (auto-detected if omitted)")
    parser.add_argument("--end-ms", type=float, default=None, help="Selected range end timestamp in milliseconds (auto-detected if omitted)")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Report format")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
