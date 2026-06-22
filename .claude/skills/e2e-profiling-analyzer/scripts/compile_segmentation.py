#!/usr/bin/env python3
"""
torch.compile segmentation + host-launch-overhead (cpp_wrapper) analysis.

torch.compile 编译区域分段分析，以及 host launch overhead / cpp_wrapper 检查。

The primary host-overhead indicator is the device-stream (queue) gap ratio: the
fraction of the main compute stream's span spent idle between device tasks. A high
gap ratio together with small kernels and high per-launch host self-time is the
signature of Python-wrapper launch overhead (cpp_wrapper disabled). The torch
trace rarely records the cpp_wrapper config flag, so this is inferred, not stated
as fact.
"""

import argparse
import bisect
import json
import re
import sqlite3
from pathlib import Path


COMPILE_REGION_REGEX = re.compile(
    r"Torch-Compiled Region|CompiledFunction|TorchDynamo|Inductor|"
    r"compiled_fn|AOTAutograd|graph\s*break",
    re.IGNORECASE,
)
RECOMPILE_REGEX = re.compile(r"TorchDynamo Cache Lookup|recompile|guard", re.IGNORECASE)


def table_exists(cur, table):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def load_strings(cur):
    if not table_exists(cur, "string_table"):
        return {}
    return {row[0]: row[1] for row in cur.execute("SELECT ID, string FROM string_table")}


def name_of(strings, name_id):
    return strings.get(name_id, f"nameId={name_id}")


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


def interval_total(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def normalize_region(name):
    n = re.sub(r"[:#]\s*\d+\s*$", "", name).strip()
    n = re.sub(r"\s+\d+\s*$", "", n).strip()
    return n or name


class IntervalSet:
    def __init__(self, intervals):
        self.merged = merge_intervals(intervals)
        self.starts = [s for s, _ in self.merged]

    def contains(self, point):
        idx = bisect.bisect_right(self.starts, point) - 1
        if idx < 0:
            return False
        s, e = self.merged[idx]
        return s <= point < e


def analyze_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        strings = load_strings(cur)
        if not table_exists(cur, "device_task_kernel_data"):
            return {"db": db_path, "label": Path(db_path).stem, "error": "missing device_task_kernel_data"}

        # --- Compiled region ranges (host side) ---
        regions_by_thread = {}
        region_inventory = {}
        recompile_count = 0
        has_region_table = table_exists(cur, "Internal_operation_range_data")
        if has_region_table:
            for pid, tid, start, end, name_id in cur.execute(
                "SELECT processId, threadId, start, end, nameId FROM Internal_operation_range_data"
            ):
                name = name_of(strings, name_id)
                if RECOMPILE_REGEX.search(name):
                    recompile_count += 1
                if not COMPILE_REGION_REGEX.search(name):
                    continue
                regions_by_thread.setdefault((pid, tid), []).append((start, end))
                norm = normalize_region(name)
                inv = region_inventory.setdefault(norm, {"count": 0, "host_ns": 0})
                inv["count"] += 1
                inv["host_ns"] += max(0, end - start)

        region_sets = {key: IntervalSet(iv) for key, iv in regions_by_thread.items()}
        has_regions = bool(region_inventory)

        # --- host launch (function_data) keyed by correlationId ---
        launch_by_corr = {}
        if table_exists(cur, "function_data"):
            for corr, pid, tid, start, end, self_ns in cur.execute(
                "SELECT correlationId, processId, threadId, start, end, self FROM function_data"
            ):
                launch_by_corr[corr] = (pid, tid, start, end, self_ns)

        # --- device tasks for queue gap ratio (all exec tasks) ---
        queues = {}  # (pid, did, qid) -> {intervals, compute}
        for table, is_kernel in (
            ("device_task_kernel_data", True),
            ("device_task_memcpy_data", False),
            ("device_task_memset_data", False),
            ("device_task_atomic_operation_data", False),
        ):
            if not table_exists(cur, table):
                continue
            cols = "processId, deviceId, queueId, start, end" + (", isComputation" if is_kernel else "")
            for row in cur.execute(f"SELECT {cols} FROM {table}"):
                if is_kernel:
                    pid, did, qid, start, end, is_comp = row
                else:
                    pid, did, qid, start, end = row
                    is_comp = 0
                item = queues.setdefault((pid, did, qid), {"intervals": [], "compute": 0})
                item["intervals"].append((start, end))
                if is_comp == 1:
                    item["compute"] += max(0, end - start)

        queue_rows = []
        for (pid, did, qid), item in queues.items():
            starts = [s for s, _ in item["intervals"]]
            ends = [e for _, e in item["intervals"]]
            span = max(ends) - min(starts)
            busy = interval_total(item["intervals"])
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
        queue_rows.sort(key=lambda r: -r["compute_ms"])
        for i, row in enumerate(queue_rows):
            row["is_main_compute_stream"] = i == 0
        main_stream_gap_pct = queue_rows[0]["gap_pct"] if queue_rows else 0.0

        # --- attribute compute kernels inside/outside compiled regions ---
        inside = {"count": 0, "device_ns": 0, "launch_self_ns": 0}
        outside = {"count": 0, "device_ns": 0, "launch_self_ns": 0}
        unknown_launch = {"count": 0, "device_ns": 0}
        outside_by_name = {}
        total_launch_self_ns = 0
        launched_kernel_count = 0

        for corr, start, end, name_id in cur.execute(
            "SELECT correlationId, start, end, nameId FROM device_task_kernel_data WHERE isComputation = 1"
        ):
            dur = max(0, end - start)
            launch = launch_by_corr.get(corr)
            if launch is None:
                unknown_launch["count"] += 1
                unknown_launch["device_ns"] += dur
                bucket = outside  # treat unknown as outside for region accounting only
            else:
                pid, tid, lstart, lend, self_ns = launch
                total_launch_self_ns += self_ns or 0
                launched_kernel_count += 1
                rset = region_sets.get((pid, tid))
                in_region = rset.contains(lstart) if rset else False
                bucket = inside if in_region else outside
                bucket["launch_self_ns"] += self_ns or 0
            bucket["count"] += 1
            bucket["device_ns"] += dur
            if bucket is outside:
                nm = name_of(strings, name_id)
                o = outside_by_name.setdefault(nm, {"count": 0, "device_ns": 0})
                o["count"] += 1
                o["device_ns"] += dur

        top_outside = sorted(
            (
                {"kernel_name": nm, "count": v["count"], "total_ms": ms(v["device_ns"])}
                for nm, v in outside_by_name.items()
            ),
            key=lambda r: -r["total_ms"],
        )[:15]

        device_compute_ns = inside["device_ns"] + outside["device_ns"] + unknown_launch["device_ns"]
        avg_launch_self_us = (
            (total_launch_self_ns / launched_kernel_count / 1000.0)
            if launched_kernel_count
            else 0.0
        )
        launch_to_compute_ratio = (
            total_launch_self_ns / device_compute_ns if device_compute_ns > 0 else 0.0
        )

        region_inv_rows = sorted(
            (
                {"region": nm, "count": v["count"], "host_ms": ms(v["host_ns"])}
                for nm, v in region_inventory.items()
            ),
            key=lambda r: -r["host_ms"],
        )

        return {
            "db": db_path,
            "label": Path(db_path).stem,
            "has_compiled_regions": has_regions,
            "region_inventory": region_inv_rows,
            "recompile_indicator_count": recompile_count,
            "segmentation": {
                "compiled_region_count": sum(r["count"] for r in region_inv_rows),
                "inside_region_compute_ms": ms(inside["device_ns"]),
                "inside_region_kernel_count": inside["count"],
                "outside_region_compute_ms": ms(outside["device_ns"]),
                "outside_region_kernel_count": outside["count"],
                "unknown_launch_kernel_count": unknown_launch["count"],
                "outside_region_share_pct": (
                    outside["device_ns"] / device_compute_ns * 100
                    if device_compute_ns > 0
                    else 0.0
                ),
            },
            "top_outside_region_kernels": top_outside,
            "host_launch_overhead": {
                "main_stream_gap_pct": main_stream_gap_pct,
                "launched_kernel_count": launched_kernel_count,
                "avg_launch_self_us": avg_launch_self_us,
                "total_launch_self_ms": ms(total_launch_self_ns),
                "device_compute_ms": ms(device_compute_ns),
                "launch_self_to_compute_ratio": launch_to_compute_ratio,
                "note": (
                    "High main_stream_gap_pct with small kernels + high avg_launch_self_us "
                    "and launch_self_to_compute_ratio indicates Python-wrapper launch overhead "
                    "(cpp_wrapper likely disabled). Recommend enabling cpp_wrapper. "
                    "cpp_wrapper mode is inferred, not read from a config flag."
                ),
            },
            "queues": queue_rows,
        }
    finally:
        conn.close()


def print_markdown(payload):
    print("# torch.compile Segmentation")
    for profile in payload["profiles"]:
        print(f"\n## {profile.get('label', profile.get('db'))}")
        if profile.get("error"):
            print(f"- Error: {profile['error']}")
            continue
        if not profile["has_compiled_regions"]:
            print("- No compiled-region annotations found; workload likely does not use torch.compile.")
        seg = profile["segmentation"]
        print(
            f"- Compiled regions: {seg['compiled_region_count']}, "
            f"recompile indicators: {profile['recompile_indicator_count']}"
        )
        print(
            f"- Compute inside regions: {seg['inside_region_compute_ms']:,.2f} ms "
            f"({seg['inside_region_kernel_count']} kernels); "
            f"outside/eager: {seg['outside_region_compute_ms']:,.2f} ms "
            f"({seg['outside_region_kernel_count']} kernels, "
            f"{seg['outside_region_share_pct']:.1f}% of compute)"
        )

        hlo = profile["host_launch_overhead"]
        print("\n### Host Launch Overhead / cpp_wrapper Check")
        print(f"- main compute stream gap ratio: **{hlo['main_stream_gap_pct']:.2f}%** (key host-overhead indicator)")
        print(
            f"- avg host launch self-time per kernel: {hlo['avg_launch_self_us']:.2f} us "
            f"over {hlo['launched_kernel_count']:,} launches"
        )
        print(
            f"- launch self-time vs device compute: {hlo['total_launch_self_ms']:,.2f} ms / "
            f"{hlo['device_compute_ms']:,.2f} ms (ratio {hlo['launch_self_to_compute_ratio']:.2f})"
        )

        if profile["region_inventory"]:
            print("\n### Compiled Region Inventory")
            print("| Region | Count | Host ms |")
            print("|---|---:|---:|")
            for row in profile["region_inventory"]:
                print(f"| {row['region']} | {row['count']:,} | {row['host_ms']:,.2f} |")

        if profile["top_outside_region_kernels"]:
            print("\n### Top Outside-Region (Eager) Compute Kernels")
            print("| Kernel | Count | Total ms |")
            print("|---|---:|---:|")
            for row in profile["top_outside_region_kernels"]:
                print(f"| {row['kernel_name']} | {row['count']:,} | {row['total_ms']:,.2f} |")

        print("\n### Device Stream Gap Ratio")
        print("| Process | Device | Queue | Span ms | Gap ms | Gap% | Compute ms | Main |")
        print("|---:|---:|---:|---:|---:|---:|---:|:--:|")
        for row in profile["queues"]:
            main = "*" if row.get("is_main_compute_stream") else ""
            print(
                f"| {row['process_id']} | {row['device_id']} | {row['queue_id']} | "
                f"{row['span_ms']:,.2f} | {row['gap_ms']:,.2f} | {row['gap_pct']:.2f}% | "
                f"{row['compute_ms']:,.2f} | {main} |"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", nargs="+", help="cnperf SQLite DB path(s)")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {"profiles": [analyze_db(db_path) for db_path in args.db]}
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_markdown(payload)


if __name__ == "__main__":
    main()
