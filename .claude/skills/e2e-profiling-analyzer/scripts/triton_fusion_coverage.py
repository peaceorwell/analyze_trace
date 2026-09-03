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
import re
import sqlite3
from pathlib import Path

try:
    from query_common import add_window_args, window_payload, window_sql
except ImportError:  # allow importing this module from another sys.path
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from query_common import add_window_args, window_payload, window_sql


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


COMMUNICATION_RE = re.compile(
    r"allreduce|all_reduce|allgather|all_gather|reduce_scatter|barrier|(?:^|[_:])broadcast|"
    r"send|recv|nccl|cncl|tccl|tcpipe|tcdp|ring|collective",
    re.IGNORECASE,
)
LIBRARY_RE = re.compile(
    r"gemm|matmul|bmm|mmkernel|conv|convolution|flash[_-]?attention|attention|"
    r"cudnn|cnnl|mluop|mlublas|embedding|gather|scatter|sort|topk|pool",
    re.IGNORECASE,
)
REDUCE_RE = re.compile(
    r"triton[_-]red|triton[_-]reduce|reduce|reduction|sum|mean|amax|amin|"
    r"(?<!all)reduce|layernorm|layer_norm|rmsnorm|norm|softmax|argmax|argmin|"
    r"variance|var|std|prod",
    re.IGNORECASE,
)
POINTWISE_RE = re.compile(
    r"triton[_-]poi|pointwise|elementwise|elemwise|where|masked|binary|unary|"
    r"add|sub|mul|div|pow|neg|abs|exp|log|sqrt|rsqrt|sigmoid|silu|gelu|relu|"
    r"tanh|erf|clamp|cast|convert|copy|fill|zero|maximum|minimum|equal|less|greater",
    re.IGNORECASE,
)
FUSION_SENSITIVE_FAMILIES = {"pointwise", "reduce"}


def classify_fusion_family(name):
    if COMMUNICATION_RE.search(name):
        return "communication"
    if REDUCE_RE.search(name):
        return "reduce"
    if POINTWISE_RE.search(name):
        return "pointwise"
    if LIBRARY_RE.search(name):
        return "library_or_gemm"
    if "triton" in name.lower():
        return "triton_other"
    return "other"


def highlight_reason(cls, family):
    if cls == "triton_fused" or family not in FUSION_SENSITIVE_FAMILIES:
        return ""
    if family == "pointwise":
        return "pointwise-like kernel did not appear as triton-fused; likely missed Inductor fusion or eager fallback"
    if family == "reduce":
        return "reduction-like kernel did not appear as triton-fused; inspect reduce fusion or graph breaks"
    return "fusion-sensitive kernel did not appear as triton-fused"


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


def analyze_db(db_path, top, start_ns=None, end_ns=None):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        strings = load_strings(cur)
        if not table_exists(cur, "device_task_kernel_data"):
            return {"db": db_path, "label": Path(db_path).stem, "error": "missing device_task_kernel_data"}
        clauses, params = window_sql(start_ns, end_ns, mode="start")
        where = " AND ".join(["isComputation = 1"] + clauses)
        rows = cur.execute(
            "SELECT processId, deviceId, nameId, start, end "
            f"FROM device_task_kernel_data WHERE {where}",
            params,
        ).fetchall()

    by_class = {"triton_fused": [], "triton_other": [], "non_triton": []}
    by_family = {}
    by_name = {}
    by_group = {}  # (pid, did) -> {class: durations}
    for pid, did, name_id, start, end in rows:
        dur = max(0, end - start)
        name = name_of(strings, name_id)
        cls = classify(name)
        family = classify_fusion_family(name)
        by_class[cls].append(dur)
        family_item = by_family.setdefault(
            family,
            {"total": 0, "count": 0, "fused": 0, "fused_count": 0, "unfused": 0, "unfused_count": 0},
        )
        family_item["total"] += dur
        family_item["count"] += 1
        if cls == "triton_fused":
            family_item["fused"] += dur
            family_item["fused_count"] += 1
        else:
            family_item["unfused"] += dur
            family_item["unfused_count"] += 1
        item = by_name.setdefault(name, {"class": cls, "family": family, "durations": []})
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
            "fusion_family": item["family"],
            "fusion_sensitive": item["family"] in FUSION_SENSITIVE_FAMILIES,
            "highlight_unfused": bool(highlight_reason(item["class"], item["family"])),
            "highlight_reason": highlight_reason(item["class"], item["family"]),
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

    family_rows = []
    for family, item in sorted(by_family.items()):
        family_rows.append(
            {
                "name": family,
                "family": family,
                "total_ms": ms(item["total"]),
                "fused_ms": ms(item["fused"]),
                "unfused_ms": ms(item["unfused"]),
                "unfused_share_pct": item["unfused"] / item["total"] * 100 if item["total"] else 0.0,
                "count": item["count"],
                "fused_count": item["fused_count"],
                "unfused_count": item["unfused_count"],
                "highlight": family in FUSION_SENSITIVE_FAMILIES and item["unfused"] > 0,
            }
        )
    family_rows.sort(key=lambda r: (-r["unfused_ms"], r["family"]))
    sensitive_unfused = [row for row in non_fused_names if row["highlight_unfused"]]

    return {
        "db": db_path,
        "label": Path(db_path).stem,
        "compute_total_ms": ms(total_compute),
        "fusion_coverage_pct": coverage,
        "class_summary": class_summary,
        "top_non_fused_kernels": non_fused_names[:top],
        "fusion_granularity": {
            "families": family_rows,
            "unfused_pointwise_ms": sum(
                item["unfused"] for family, item in by_family.items() if family == "pointwise"
            )
            / 1e6,
            "unfused_reduce_ms": sum(
                item["unfused"] for family, item in by_family.items() if family == "reduce"
            )
            / 1e6,
            "highlight_unfused_pointwise": any(
                row["family"] == "pointwise" and row["highlight"] for row in family_rows
            ),
            "top_unfused_fusion_sensitive_kernels": sensitive_unfused[:top],
            "note": (
                "Family classification is name-based. Treat highlighted pointwise/reduce rows "
                "as Inductor fusion candidates to confirm with graph-break and generated-code evidence."
            ),
        },
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
        granularity = profile.get("fusion_granularity", {})
        families = granularity.get("families", [])
        if families:
            print("\n### Inductor Fusion Granularity")
            print("| Family | Total ms | Fused ms | Unfused ms | Unfused share | Count | Unfused count | Highlight |")
            print("|---|---:|---:|---:|---:|---:|---:|:--:|")
            for row in families:
                mark = "yes" if row.get("highlight") else ""
                print(
                    f"| {row['family']} | {row['total_ms']:,.2f} | {row['fused_ms']:,.2f} | "
                    f"{row['unfused_ms']:,.2f} | {row['unfused_share_pct']:.2f}% | "
                    f"{row['count']:,} | {row['unfused_count']:,} | {mark} |"
                )
        sensitive = granularity.get("top_unfused_fusion_sensitive_kernels", [])
        if sensitive:
            print("\n### Highlighted Unfused Pointwise/Reduce Candidates")
            print("| Kernel | Family | Total ms | Share | Count | Reason |")
            print("|---|---|---:|---:|---:|---|")
            for row in sensitive:
                print(
                    f"| {row['kernel_name']} | {row['fusion_family']} | {row['total_ms']:,.2f} | "
                    f"{row['share_of_compute_pct']:.2f}% | {row['count']:,} | {row['highlight_reason']} |"
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
    add_window_args(parser)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {
        "window": window_payload(args.start_ns, args.end_ns),
        "profiles": [
            analyze_db(db_path, args.top, args.start_ns, args.end_ns)
            for db_path in args.db
        ],
    }
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_markdown(payload)


if __name__ == "__main__":
    main()
