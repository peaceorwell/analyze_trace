#!/usr/bin/env python3
"""Compute-kernel breakdown for one or more cnperf DBs."""

import argparse
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


def summarize_durations(durations):
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
            SELECT processId, deviceId, nameId, start, end
            FROM device_task_kernel_data
            WHERE isComputation = 1
            ORDER BY processId, deviceId, start
            """
        ).fetchall()

    groups = {}
    name_groups = {}
    all_durations = []
    for process_id, device_id, name_id, start, end in rows:
        duration = max(0, end - start)
        key = (process_id, device_id)
        name = name_of(strings, name_id)
        group = groups.setdefault(key, {"durations": [], "names": {}})
        group["durations"].append(duration)
        group["names"].setdefault(name, []).append(duration)
        name_groups.setdefault(name, []).append(duration)
        all_durations.append(duration)

    total_compute = sum(all_durations)
    per_device = []
    for (process_id, device_id), item in sorted(groups.items()):
        group_total = sum(item["durations"])
        top_names = []
        for name, durations in item["names"].items():
            row = summarize_durations(durations)
            row["kernel_name"] = name
            row["share_of_group_pct"] = sum(durations) / group_total * 100 if group_total else 0.0
            top_names.append(row)
        top_names.sort(key=lambda row: row["total_ms"], reverse=True)
        summary = summarize_durations(item["durations"])
        summary.update(
            {
                "process_id": process_id,
                "device_id": device_id,
                "share_of_compute_pct": group_total / total_compute * 100 if total_compute else 0.0,
                "unique_kernel_names": len(item["names"]),
                "top_kernels": top_names[:top],
            }
        )
        per_device.append(summary)

    top_kernels = []
    for name, durations in name_groups.items():
        row = summarize_durations(durations)
        row["kernel_name"] = name
        row["share_of_compute_pct"] = sum(durations) / total_compute * 100 if total_compute else 0.0
        top_kernels.append(row)
    top_kernels.sort(key=lambda row: row["total_ms"], reverse=True)

    totals = summarize_durations(all_durations)
    totals["unique_kernel_names"] = len(name_groups)
    return {
        "db": db_path,
        "label": Path(db_path).stem,
        "compute_summary": totals,
        "top_compute_kernels": top_kernels[:top],
        "per_process_device": per_device,
    }


def print_markdown(payload):
    print("# Compute Kernel Breakdown")
    for profile in payload["profiles"]:
        print(f"\n## {profile.get('label', profile.get('db'))}")
        if profile.get("error"):
            print(f"- Error: {profile['error']}")
            continue
        summary = profile["compute_summary"]
        print(
            f"- Total compute: {summary['total_ms']:,.3f} ms, "
            f"count: {summary['count']:,}, unique names: {summary['unique_kernel_names']:,}"
        )
        print("\n### Top Compute Kernels")
        print("| Kernel name | Total ms | Count | Share | Avg ms | P90 ms | Max ms |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for row in profile["top_compute_kernels"]:
            print(
                f"| {row['kernel_name']} | {row['total_ms']:,.3f} | {row['count']:,} | "
                f"{row['share_of_compute_pct']:.2f}% | {row['avg_ms']:,.3f} | "
                f"{row['p90_ms']:,.3f} | {row['max_ms']:,.3f} |"
            )
        print("\n### Per Process/Device")
        print("| Process | Device | Total ms | Count | Share | Unique names | Top kernel |")
        print("|---:|---:|---:|---:|---:|---:|---|")
        for row in profile["per_process_device"]:
            top_kernel = row["top_kernels"][0]["kernel_name"] if row["top_kernels"] else "-"
            print(
                f"| {row['process_id']} | {row['device_id']} | {row['total_ms']:,.3f} | "
                f"{row['count']:,} | {row['share_of_compute_pct']:.2f}% | "
                f"{row['unique_kernel_names']:,} | {top_kernel} |"
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
