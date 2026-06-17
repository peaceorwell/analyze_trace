#!/usr/bin/env python3
"""Compare per-process/device progress metrics across cnperf DBs."""

import argparse
import bisect
import json
import statistics
import sqlite3
from pathlib import Path


def table_exists(cur, table):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,))
    return cur.fetchone() is not None


def ms(ns):
    return ns / 1e6 if ns is not None else 0.0


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_index(intervals):
    merged = merge_intervals(intervals)
    return {"intervals": merged, "starts": [item[0] for item in merged]}


def interval_total(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def uncovered_ns(start, end, covering):
    if end <= start:
        return 0
    intervals = covering["intervals"]
    starts = covering["starts"]
    uncovered = end - start
    index = max(0, bisect.bisect_right(starts, start) - 1)
    for cover_start, cover_end in intervals[index:]:
        if cover_start >= end:
            break
        overlap = min(end, cover_end) - max(start, cover_start)
        if overlap > 0:
            uncovered -= overlap
    return max(0, uncovered)


def coverage_gaps(compute_intervals):
    gaps = []
    merged = merge_intervals(compute_intervals)
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        if next_start > prev_end:
            gaps.append((prev_end, next_start))
    return gaps


def analyze_db(db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        if not table_exists(cur, "device_task_kernel_data"):
            return [{"db": db_path, "label": Path(db_path).stem, "error": "missing device_task_kernel_data"}]
        rows = cur.execute(
            """
            SELECT processId, deviceId, start, end, isComputation
            FROM device_task_kernel_data
            ORDER BY processId, deviceId, start
            """
        ).fetchall()
        function_rows = []
        if table_exists(cur, "function_data"):
            function_rows = cur.execute(
                "SELECT processId, MIN(start), MAX(end), COUNT(*) FROM function_data GROUP BY processId"
            ).fetchall()

    host_by_process = {
        process_id: {"host_start_ms": ms(start), "host_end_ms": ms(end), "host_function_count": count}
        for process_id, start, end, count in function_rows
    }
    groups = {}
    for process_id, device_id, start, end, is_comp in rows:
        item = groups.setdefault(
            (process_id, device_id),
            {
                "compute_intervals": [],
                "comm_intervals": [],
                "kernel_count": 0,
                "compute_kernel_count": 0,
                "comm_kernel_count": 0,
                "device_start": None,
                "device_end": None,
            },
        )
        item["kernel_count"] += 1
        item["device_start"] = start if item["device_start"] is None else min(item["device_start"], start)
        item["device_end"] = end if item["device_end"] is None else max(item["device_end"], end)
        if is_comp == 1:
            item["compute_kernel_count"] += 1
            item["compute_intervals"].append((start, end))
        else:
            item["comm_kernel_count"] += 1
            item["comm_intervals"].append((start, end))

    output = []
    for (process_id, device_id), item in sorted(groups.items()):
        compute_index = interval_index(item["compute_intervals"])
        comm_uncovered = sum(uncovered_ns(start, end, compute_index) for start, end in item["comm_intervals"])
        compute_gaps = coverage_gaps(item["compute_intervals"])
        device_span = (item["device_end"] or 0) - (item["device_start"] or 0)
        row = {
            "db": db_path,
            "label": Path(db_path).stem,
            "process_id": process_id,
            "device_id": device_id,
            "device_start_ms": ms(item["device_start"]),
            "device_end_ms": ms(item["device_end"]),
            "device_span_ms": ms(device_span),
            "kernel_count": item["kernel_count"],
            "compute_kernel_count": item["compute_kernel_count"],
            "comm_kernel_count": item["comm_kernel_count"],
            "compute_total_ms": ms(sum(end - start for start, end in item["compute_intervals"])),
            "compute_coverage_ms": ms(interval_total(item["compute_intervals"])),
            "comm_total_ms": ms(sum(end - start for start, end in item["comm_intervals"])),
            "comm_uncovered_ms": ms(comm_uncovered),
            "compute_gap_total_ms": ms(sum(end - start for start, end in compute_gaps)),
            "max_compute_gap_ms": ms(max((end - start for start, end in compute_gaps), default=0)),
        }
        row.update(host_by_process.get(process_id, {}))
        output.append(row)
    return output


def add_skew(rows, metric):
    values = [row.get(metric, 0.0) for row in rows if not row.get("error")]
    median = statistics.median(values) if values else 0.0
    fastest = min(values) if values else 0.0
    for row in rows:
        value = row.get(metric, 0.0)
        row[f"{metric}_delta_vs_median"] = value - median
        row[f"{metric}_delta_vs_fastest"] = value - fastest


def build_payload(db_paths):
    rows = []
    for db_path in db_paths:
        rows.extend(analyze_db(db_path))
    for metric in ("device_span_ms", "compute_total_ms", "comm_uncovered_ms", "compute_gap_total_ms"):
        add_skew(rows, metric)
    rows.sort(key=lambda row: (row.get("device_span_ms", 0.0), row.get("compute_total_ms", 0.0)), reverse=True)
    return {"ranks": rows}


def print_markdown(payload):
    print("# Rank / Device Comparison")
    rows = payload["ranks"]
    if not rows:
        print("No rows.")
        return
    print("| DB | PID | Device | Span ms | Compute ms | Comm uncovered ms | Compute gap ms | Kernels | Span vs median | Compute vs median |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row.get("error"):
            print(f"| {row['label']} | - | - | - | - | - | - | - | - | - |")
            continue
        print(
            f"| {row['label']} | {row['process_id']} | {row['device_id']} | "
            f"{row['device_span_ms']:,.3f} | {row['compute_total_ms']:,.3f} | "
            f"{row['comm_uncovered_ms']:,.3f} | {row['compute_gap_total_ms']:,.3f} | "
            f"{row['kernel_count']:,} | {row['device_span_ms_delta_vs_median']:+,.3f} | "
            f"{row['compute_total_ms_delta_vs_median']:+,.3f} |"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", nargs="+", help="cnperf SQLite DB path(s)")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_payload(args.db)
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_markdown(payload)


if __name__ == "__main__":
    main()
