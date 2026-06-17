#!/usr/bin/env python3
"""Communication-kernel exposed-time breakdown for one or more cnperf DBs."""

import argparse
import bisect
import json
import math
import sqlite3
from pathlib import Path


def table_exists(cur, table):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,))
    return cur.fetchone() is not None


def load_strings(cur):
    if not table_exists(cur, "string_table"):
        return {}
    return {row[0]: row[1] for row in cur.execute("SELECT ID, string FROM string_table")}


def name_of(strings, name_id):
    return strings.get(name_id, f"nameId={name_id}")


def ms(ns):
    return ns / 1e6 if ns is not None else 0.0


def percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * pct / 100.0) - 1))
    return ordered[index]


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


def summarize(durations):
    total = sum(durations)
    count = len(durations)
    return {
        "count": count,
        "total_ms": ms(total),
        "avg_ms": ms(total / count) if count else 0.0,
        "p90_ms": ms(percentile(durations, 90)) if count else 0.0,
        "max_ms": ms(max(durations)) if durations else 0.0,
    }


def analyze_db(db_path, top):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        strings = load_strings(cur)
        if not table_exists(cur, "device_task_kernel_data"):
            return {"db": db_path, "error": "missing device_task_kernel_data"}
        rows = cur.execute(
            """
            SELECT processId, deviceId, queueId, correlationId, nameId, start, end, isComputation
            FROM device_task_kernel_data
            ORDER BY processId, deviceId, start
            """
        ).fetchall()

    compute_by_group = {}
    comm_rows = []
    for process_id, device_id, queue_id, corr_id, name_id, start, end, is_comp in rows:
        group = (process_id, device_id)
        if is_comp == 1:
            compute_by_group.setdefault(group, []).append((start, end))
        else:
            comm_rows.append((process_id, device_id, queue_id, corr_id, name_id, start, end))
    compute_index = {group: interval_index(intervals) for group, intervals in compute_by_group.items()}

    by_name = {}
    by_group = {}
    long_events = []
    total_duration = 0
    total_uncovered = 0
    for process_id, device_id, queue_id, corr_id, name_id, start, end in comm_rows:
        duration = max(0, end - start)
        group_key = (process_id, device_id)
        uncovered = uncovered_ns(start, end, compute_index.get(group_key, interval_index([])))
        name = name_of(strings, name_id)
        total_duration += duration
        total_uncovered += uncovered
        item = by_name.setdefault(name, {"durations": [], "uncovered": 0})
        item["durations"].append(duration)
        item["uncovered"] += uncovered
        group_item = by_group.setdefault(group_key, {"durations": [], "uncovered": 0})
        group_item["durations"].append(duration)
        group_item["uncovered"] += uncovered
        long_events.append(
            {
                "process_id": process_id,
                "device_id": device_id,
                "queue_id": queue_id,
                "correlation_id": corr_id,
                "kernel_name": name,
                "start_ms": ms(start),
                "duration_ms": ms(duration),
                "uncovered_ms": ms(uncovered),
            }
        )

    top_names = []
    for name, item in by_name.items():
        row = summarize(item["durations"])
        row.update(
            {
                "kernel_name": name,
                "uncovered_ms": ms(item["uncovered"]),
                "share_of_comm_pct": sum(item["durations"]) / total_duration * 100 if total_duration else 0.0,
                "share_of_uncovered_pct": item["uncovered"] / total_uncovered * 100 if total_uncovered else 0.0,
                "uncovered_share_pct": item["uncovered"] / sum(item["durations"]) * 100
                if item["durations"]
                else 0.0,
            }
        )
        top_names.append(row)
    top_names.sort(key=lambda row: row["uncovered_ms"], reverse=True)

    per_group = []
    for (process_id, device_id), item in sorted(by_group.items()):
        row = summarize(item["durations"])
        row.update(
            {
                "process_id": process_id,
                "device_id": device_id,
                "uncovered_ms": ms(item["uncovered"]),
                "share_of_uncovered_pct": item["uncovered"] / total_uncovered * 100 if total_uncovered else 0.0,
                "uncovered_share_pct": item["uncovered"] / sum(item["durations"]) * 100
                if item["durations"]
                else 0.0,
            }
        )
        per_group.append(row)

    return {
        "db": db_path,
        "label": Path(db_path).stem,
        "communication_summary": {
            "count": len(comm_rows),
            "total_ms": ms(total_duration),
            "uncovered_ms": ms(total_uncovered),
            "uncovered_share_pct": total_uncovered / total_duration * 100 if total_duration else 0.0,
            "unique_kernel_names": len(by_name),
        },
        "top_communication_kernels": top_names[:top],
        "per_process_device": per_group,
        "top_long_events": sorted(long_events, key=lambda row: row["uncovered_ms"], reverse=True)[:top],
    }


def print_markdown(payload):
    print("# Communication Breakdown")
    for profile in payload["profiles"]:
        print(f"\n## {profile.get('label', profile.get('db'))}")
        if profile.get("error"):
            print(f"- Error: {profile['error']}")
            continue
        summary = profile["communication_summary"]
        print(
            f"- Communication total: {summary['total_ms']:,.3f} ms, "
            f"uncovered: {summary['uncovered_ms']:,.3f} ms "
            f"({summary['uncovered_share_pct']:.2f}%), count: {summary['count']:,}"
        )
        print("\n### Top Communication Kernels By Uncovered Time")
        print("| Kernel name | Total ms | Uncovered ms | Count | Uncovered share | Share of uncovered | Avg ms | Max ms |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in profile["top_communication_kernels"]:
            print(
                f"| {row['kernel_name']} | {row['total_ms']:,.3f} | {row['uncovered_ms']:,.3f} | "
                f"{row['count']:,} | {row['uncovered_share_pct']:.2f}% | "
                f"{row['share_of_uncovered_pct']:.2f}% | {row['avg_ms']:,.3f} | {row['max_ms']:,.3f} |"
            )
        print("\n### Per Process/Device")
        print("| Process | Device | Total ms | Uncovered ms | Count | Uncovered share | Share of uncovered |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for row in profile["per_process_device"]:
            print(
                f"| {row['process_id']} | {row['device_id']} | {row['total_ms']:,.3f} | "
                f"{row['uncovered_ms']:,.3f} | {row['count']:,} | "
                f"{row['uncovered_share_pct']:.2f}% | {row['share_of_uncovered_pct']:.2f}% |"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", nargs="+", help="cnperf SQLite DB path(s)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {"profiles": [analyze_db(db_path, args.top) for db_path in args.db]}
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_markdown(payload)


if __name__ == "__main__":
    main()
