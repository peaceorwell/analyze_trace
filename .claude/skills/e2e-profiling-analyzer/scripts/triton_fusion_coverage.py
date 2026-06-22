#!/usr/bin/env python3
"""
Triton fusion coverage: how much compute time is in inductor-fused triton kernels
versus non-fused library/eager kernels.

triton 融合覆盖率：计算时间中有多少落在 inductor 融合的 triton 核心上，多少落在
未融合的库/eager 核心上。未融合核心是潜在的融合缺口候选。
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path


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


def percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * pct / 100.0) - 1))
    return ordered[index]


def classify(name):
    low = name.lower()
    is_triton = "triton" in low
    is_fused = "fused" in low
    if is_triton and is_fused:
        return "triton_fused"
    if is_triton:
        return "triton_other"
    return "non_triton"


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
            return {"db": db_path, "label": Path(db_path).stem, "error": "missing device_task_kernel_data"}
        rows = cur.execute(
            "SELECT processId, deviceId, nameId, start, end "
            "FROM device_task_kernel_data WHERE isComputation = 1"
        ).fetchall()

    by_class = {"triton_fused": [], "triton_other": [], "non_triton": []}
    by_name = {}
    by_group = {}  # (pid, did) -> {class: durations}
    for pid, did, name_id, start, end in rows:
        dur = max(0, end - start)
        name = name_of(strings, name_id)
        cls = classify(name)
        by_class[cls].append(dur)
        item = by_name.setdefault(name, {"class": cls, "durations": []})
        item["durations"].append(dur)
        group = by_group.setdefault((pid, did), {"triton_fused": 0, "triton_other": 0, "non_triton": 0, "total": 0})
        group[cls] += dur
        group["total"] += dur

    total_compute = sum(sum(d) for d in by_class.values())
    fused_total = sum(by_class["triton_fused"])
    coverage = fused_total / total_compute * 100 if total_compute else 0.0

    class_summary = {cls: summarize(durs) for cls, durs in by_class.items()}
    for cls, s in class_summary.items():
        s["share_of_compute_pct"] = (
            sum(by_class[cls]) / total_compute * 100 if total_compute else 0.0
        )

    non_fused_names = [
        {
            "kernel_name": nm,
            "class": item["class"],
            **summarize(item["durations"]),
            "share_of_compute_pct": sum(item["durations"]) / total_compute * 100
            if total_compute
            else 0.0,
        }
        for nm, item in by_name.items()
        if item["class"] != "triton_fused"
    ]
    non_fused_names.sort(key=lambda r: -r["total_ms"])

    per_group = []
    for (pid, did), g in sorted(by_group.items()):
        per_group.append(
            {
                "process_id": pid,
                "device_id": did,
                "fusion_coverage_pct": g["triton_fused"] / g["total"] * 100 if g["total"] else 0.0,
                "triton_fused_ms": ms(g["triton_fused"]),
                "triton_other_ms": ms(g["triton_other"]),
                "non_triton_ms": ms(g["non_triton"]),
                "compute_total_ms": ms(g["total"]),
            }
        )

    return {
        "db": db_path,
        "label": Path(db_path).stem,
        "compute_total_ms": ms(total_compute),
        "fusion_coverage_pct": coverage,
        "class_summary": class_summary,
        "top_non_fused_kernels": non_fused_names[:top],
        "per_process_device": per_group,
        "uses_torch_compile": bool(by_class["triton_fused"] or by_class["triton_other"]),
    }


def print_markdown(payload):
    print("# Triton Fusion Coverage")
    for profile in payload["profiles"]:
        print(f"\n## {profile.get('label', profile.get('db'))}")
        if profile.get("error"):
            print(f"- Error: {profile['error']}")
            continue
        if not profile["uses_torch_compile"]:
            print("- No triton kernels found; workload likely does not use torch.compile/inductor.")
        print(
            f"- Compute total: {profile['compute_total_ms']:,.2f} ms, "
            f"fusion coverage: **{profile['fusion_coverage_pct']:.2f}%**"
        )
        print("\n### Compute Time By Class")
        print("| Class | Total ms | Share | Count | Avg ms | Max ms |")
        print("|---|---:|---:|---:|---:|---:|")
        for cls in ("triton_fused", "triton_other", "non_triton"):
            s = profile["class_summary"][cls]
            print(
                f"| {cls} | {s['total_ms']:,.2f} | {s['share_of_compute_pct']:.2f}% | "
                f"{s['count']:,} | {s['avg_ms']:,.4f} | {s['max_ms']:,.4f} |"
            )
        if profile["top_non_fused_kernels"]:
            print("\n### Top Non-Fused Kernels (fusion-miss / fallback candidates)")
            print("| Kernel | Class | Total ms | Share | Count | Avg ms | Max ms |")
            print("|---|---|---:|---:|---:|---:|---:|")
            for row in profile["top_non_fused_kernels"]:
                print(
                    f"| {row['kernel_name']} | {row['class']} | {row['total_ms']:,.2f} | "
                    f"{row['share_of_compute_pct']:.2f}% | {row['count']:,} | "
                    f"{row['avg_ms']:,.4f} | {row['max_ms']:,.4f} |"
                )
        if len(profile["per_process_device"]) > 1:
            print("\n### Per Process/Device Fusion Coverage")
            print("| Process | Device | Coverage | Fused ms | Non-triton ms | Compute ms |")
            print("|---:|---:|---:|---:|---:|---:|")
            for row in profile["per_process_device"]:
                print(
                    f"| {row['process_id']} | {row['device_id']} | "
                    f"{row['fusion_coverage_pct']:.2f}% | {row['triton_fused_ms']:,.2f} | "
                    f"{row['non_triton_ms']:,.2f} | {row['compute_total_ms']:,.2f} |"
                )
        print(
            "\n_Note: non-triton library compute primitives (GEMM/conv) are often the "
            "intended fast path; flag fusion misses mainly for elementwise/reduction/pointwise ops._"
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
