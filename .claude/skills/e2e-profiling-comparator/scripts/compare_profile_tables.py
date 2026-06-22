#!/usr/bin/env python3
"""Compare two collect_profile_tables.py JSON outputs."""

import argparse
import json
from pathlib import Path


OVERVIEW_CATEGORIES = (
    ("compute_kernel", "compute kernel"),
    ("communication_kernel", "communication kernel"),
    ("memcpy", "memcpy"),
    ("compute_gap", "compute gap"),
    ("pure_gap", "pure gap"),
    ("other_activity", "other activity"),
)

NAMED_TABLES = (
    ("compute_kernel", ("device", "compute_kernel", "top"), "total_ms"),
    ("communication_kernel", ("device", "communication_kernel", "top"), "uncovered_ms"),
    ("memcpy", ("device", "memcpy", "top"), "uncovered_ms"),
    ("host_function", ("host", "function", "top"), "total_ms"),
    ("host_annotation", ("host", "annotation", "top"), "total_ms"),
)


def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("profile", payload)


def nested_get(obj, path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def num(value):
    return float(value or 0.0)


def pct_delta(a_value, b_value):
    if not a_value:
        return None
    return (b_value - a_value) / a_value * 100.0


def delta_values(a_value, b_value):
    a_num = num(a_value)
    b_num = num(b_value)
    return {
        "baseline": a_num,
        "current": b_num,
        "delta": b_num - a_num,
        "delta_pct": pct_delta(a_num, b_num),
    }


def classify_delta(delta, threshold=1e-9):
    if delta > threshold:
        return "regression"
    if delta < -threshold:
        return "improvement"
    return "flat"


def status_for(delta, higher_is_worse=True, threshold=1e-9):
    """Direction-aware status. Some metrics regress when they go down (e.g. fusion
    coverage, bandwidth utilization), others when they go up (e.g. gap ratio)."""
    if abs(delta) <= threshold:
        return "flat"
    worse = delta > 0 if higher_is_worse else delta < 0
    return "regression" if worse else "improvement"


def scalar_delta_rows(baseline, current, specs):
    """specs: list of (label, path_tuple, higher_is_worse)."""
    rows = []
    for label, path, higher_is_worse in specs:
        a = num(nested_get(baseline, path))
        b = num(nested_get(current, path))
        row = delta_values(a, b)
        row["metric"] = label
        row["higher_is_worse"] = higher_is_worse
        row["status"] = status_for(row["delta"], higher_is_worse)
        rows.append(row)
    return rows


def compare_kernel_efficiency(baseline_rows, current_rows):
    """Per triton-kernel-name IO efficiency delta. Lower utilization/bandwidth is worse."""
    a_map = row_map(baseline_rows)
    b_map = row_map(current_rows)
    rows = []
    for name in sorted(set(a_map) | set(b_map)):
        a_item = a_map.get(name, {})
        b_item = b_map.get(name, {})
        util = delta_values(a_item.get("bandwidth_utilization"), b_item.get("bandwidth_utilization"))
        rows.append(
            {
                "name": name,
                "bandwidth_utilization": util,
                "avg_io_efficiency": delta_values(
                    a_item.get("avg_io_efficiency"), b_item.get("avg_io_efficiency")
                ),
                "total_ms": delta_values(a_item.get("total_ms"), b_item.get("total_ms")),
                "status": status_for(util["delta"], higher_is_worse=False),
                "presence": (
                    "both" if name in a_map and name in b_map
                    else "current_only" if name in b_map else "baseline_only"
                ),
            }
        )
    # Worst regressions first: largest utilization drop.
    rows.sort(key=lambda r: r["bandwidth_utilization"]["delta"])
    return rows


def compare_overview(baseline, current):
    rows = []
    baseline_duration = num(nested_get(baseline, ("range", "duration_ms"), 0.0))
    current_duration = num(nested_get(current, ("range", "duration_ms"), 0.0))
    for key, label in OVERVIEW_CATEGORIES:
        a_item = nested_get(baseline, ("device", key), {}) or {}
        b_item = nested_get(current, ("device", key), {}) or {}
        total = delta_values(a_item.get("total_ms"), b_item.get("total_ms"))
        count = delta_values(a_item.get("count"), b_item.get("count"))
        a_share = total["baseline"] / baseline_duration * 100.0 if baseline_duration else 0.0
        b_share = total["current"] / current_duration * 100.0 if current_duration else 0.0
        row = {
            "category": label,
            "total_ms": total,
            "count": count,
            "avg_ms": delta_values(a_item.get("avg_ms"), b_item.get("avg_ms")),
            "max_ms": delta_values(a_item.get("max_ms"), b_item.get("max_ms")),
            "range_share_pct": {
                "baseline": a_share,
                "current": b_share,
                "delta": b_share - a_share,
                "delta_pct": pct_delta(a_share, b_share),
            },
            "status": classify_delta(total["delta"]),
        }
        if "uncovered_ms" in a_item or "uncovered_ms" in b_item:
            row["uncovered_ms"] = delta_values(a_item.get("uncovered_ms"), b_item.get("uncovered_ms"))
        rows.append(row)
    rows.sort(key=lambda item: item["total_ms"]["delta"], reverse=True)
    return rows


def row_map(rows):
    mapped = {}
    for row in rows or []:
        name = row.get("name")
        if not name:
            continue
        mapped[name] = row
    return mapped


def compare_named_rows(baseline_rows, current_rows, primary_metric):
    a_map = row_map(baseline_rows)
    b_map = row_map(current_rows)
    names = sorted(set(a_map) | set(b_map))
    rows = []
    for name in names:
        a_item = a_map.get(name, {})
        b_item = b_map.get(name, {})
        primary = delta_values(a_item.get(primary_metric), b_item.get(primary_metric))
        row = {
            "name": name,
            "primary_metric": primary_metric,
            primary_metric: primary,
            "total_ms": delta_values(a_item.get("total_ms"), b_item.get("total_ms")),
            "count": delta_values(a_item.get("count"), b_item.get("count")),
            "avg_ms": delta_values(a_item.get("avg_ms"), b_item.get("avg_ms")),
            "p90_ms": delta_values(a_item.get("p90_ms"), b_item.get("p90_ms")),
            "max_ms": delta_values(a_item.get("max_ms"), b_item.get("max_ms")),
            "share_pct": delta_values(a_item.get("share_pct"), b_item.get("share_pct")),
            "status": classify_delta(primary["delta"]),
            "presence": (
                "both"
                if name in a_map and name in b_map
                else "current_only"
                if name in b_map
                else "baseline_only"
            ),
        }
        if "uncovered_ms" in a_item or "uncovered_ms" in b_item:
            row["uncovered_ms"] = delta_values(a_item.get("uncovered_ms"), b_item.get("uncovered_ms"))
            row["uncovered_share_pct"] = delta_values(
                a_item.get("uncovered_share_pct"), b_item.get("uncovered_share_pct")
            )
        if "bytes" in a_item or "bytes" in b_item:
            row["bytes"] = delta_values(a_item.get("bytes"), b_item.get("bytes"))
            row["bandwidth_gbps"] = delta_values(
                a_item.get("bandwidth_gbps"), b_item.get("bandwidth_gbps")
            )
        for metric in ("fused_ms", "unfused_ms", "unfused_share_pct", "fused_count", "unfused_count"):
            if metric == primary_metric:
                continue
            if metric in a_item or metric in b_item:
                row[metric] = delta_values(a_item.get(metric), b_item.get(metric))
        rows.append(row)
    rows.sort(key=lambda item: item[primary_metric]["delta"], reverse=True)
    return rows


def top_rows(rows, limit):
    regressions = [row for row in rows if row.get("status") == "regression"]
    improvements = [row for row in rows if row.get("status") == "improvement"]
    return {
        "regressions": regressions[:limit],
        "improvements": improvements[-limit:][::-1],
    }


def build_comparison(baseline, current, limit):
    named = {}
    for table_name, path, primary_metric in NAMED_TABLES:
        rows = compare_named_rows(
            nested_get(baseline, path, []) or [],
            nested_get(current, path, []) or [],
            primary_metric,
        )
        named[table_name] = {
            "primary_metric": primary_metric,
            "rows": rows,
            "top": top_rows(rows, limit),
        }
    host_overhead_delta = scalar_delta_rows(
        baseline,
        current,
        [
            ("main compute stream gap %", ("device", "device_stream_gap", "main_stream_gap_pct"), True),
            ("device-level gap %", ("device", "device_stream_gap", "device_gap_pct"), True),
            ("avg launch self (us)", ("torch_compile", "segmentation", "host_launch_overhead", "avg_launch_self_us"), True),
            ("launch/compute ratio", ("torch_compile", "segmentation", "host_launch_overhead", "launch_self_to_compute_ratio"), True),
        ],
    )
    torch_compile_delta = {
        "fusion_scalar": scalar_delta_rows(
            baseline,
            current,
            [
                ("fusion coverage %", ("torch_compile", "fusion", "fusion_coverage_pct"), False),
                ("triton fused ms", ("torch_compile", "fusion", "triton_fused_ms"), False),
                ("non-triton ms", ("torch_compile", "fusion", "non_triton_ms"), True),
                ("unfused pointwise ms", ("torch_compile", "fusion", "fusion_granularity", "unfused_pointwise_ms"), True),
                ("unfused reduce ms", ("torch_compile", "fusion", "fusion_granularity", "unfused_reduce_ms"), True),
                ("outside-region compute ms", ("torch_compile", "segmentation", "outside_region_compute_ms"), True),
                ("recompile indicators", ("torch_compile", "segmentation", "recompile_indicator_count"), True),
            ],
        ),
        "fusion_granularity_families": compare_named_rows(
            nested_get(baseline, ("torch_compile", "fusion", "fusion_granularity", "families"), []) or [],
            nested_get(current, ("torch_compile", "fusion", "fusion_granularity", "families"), []) or [],
            "unfused_ms",
        ),
        "unfused_fusion_sensitive_kernels": compare_named_rows(
            nested_get(
                baseline,
                ("torch_compile", "fusion", "fusion_granularity", "top_unfused_fusion_sensitive"),
                [],
            )
            or [],
            nested_get(
                current,
                ("torch_compile", "fusion", "fusion_granularity", "top_unfused_fusion_sensitive"),
                [],
            )
            or [],
            "total_ms",
        ),
        "non_fused_kernels": compare_named_rows(
            nested_get(baseline, ("torch_compile", "fusion", "top_non_fused"), []) or [],
            nested_get(current, ("torch_compile", "fusion", "top_non_fused"), []) or [],
            "total_ms",
        ),
        "kernel_efficiency": compare_kernel_efficiency(
            nested_get(baseline, ("torch_compile", "kernel_efficiency", "kernels"), []) or [],
            nested_get(current, ("torch_compile", "kernel_efficiency", "kernels"), []) or [],
        ),
    }

    return {
        "comparison": {
            "baseline": {
                "label": baseline.get("label"),
                "db": baseline.get("db"),
                "range": baseline.get("range"),
                "device_name": nested_get(baseline, ("torch_compile", "device_name")),
                "peak_bandwidth": nested_get(baseline, ("torch_compile", "peak_bandwidth")),
            },
            "current": {
                "label": current.get("label"),
                "db": current.get("db"),
                "range": current.get("range"),
                "device_name": nested_get(current, ("torch_compile", "device_name")),
                "peak_bandwidth": nested_get(current, ("torch_compile", "peak_bandwidth")),
            },
            "delta_definition": "current - baseline",
        },
        "device_overview_delta": compare_overview(baseline, current),
        "named_deltas": named,
        "host_overhead_delta": host_overhead_delta,
        "torch_compile_delta": torch_compile_delta,
    }


def fmt(value):
    if value is None:
        return "n/a"
    return f"{value:,.3f}"


def fmt_delta(delta_obj):
    delta_pct = delta_obj.get("delta_pct")
    pct_text = "n/a" if delta_pct is None else f"{delta_pct:+.2f}%"
    return (
        f"{fmt(delta_obj.get('baseline'))} | {fmt(delta_obj.get('current'))} | "
        f"{fmt(delta_obj.get('delta'))} | {pct_text}"
    )


def print_delta_table(title, rows, metric_key, name_key="name", limit=20):
    print(f"\n## {title}")
    if not rows:
        print("No rows.")
        return
    print(f"| {name_key} | A | B | Delta | Delta % | Status |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows[:limit]:
        values = fmt_delta(row[metric_key]).split(" | ")
        print(
            f"| {row[name_key]} | {values[0]} | {values[1]} | "
            f"{values[2]} | {values[3]} | {row.get('status', '')} |"
        )


def print_scalar_delta_table(title, rows):
    print(f"\n## {title}")
    if not rows:
        print("No rows.")
        return
    print("| Metric | A | B | Delta | Delta % | Status |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        delta_pct = row.get("delta_pct")
        pct_text = "n/a" if delta_pct is None else f"{delta_pct:+.2f}%"
        print(
            f"| {row['metric']} | {fmt(row.get('baseline'))} | {fmt(row.get('current'))} | "
            f"{fmt(row.get('delta'))} | {pct_text} | {row.get('status', '')} |"
        )


def print_markdown(payload, limit):
    cmp_info = payload["comparison"]
    print("# E2E Profiling Comparison Delta")
    print("\n## Inputs")
    print(f"- Baseline A: {cmp_info['baseline'].get('label')} ({cmp_info['baseline'].get('db')})")
    print(f"- Current B: {cmp_info['current'].get('label')} ({cmp_info['current'].get('db')})")
    print("- Delta: B - A")

    print_delta_table(
        "Device Overview Delta",
        payload["device_overview_delta"],
        "total_ms",
        name_key="category",
        limit=limit,
    )

    print_scalar_delta_table(
        "Host Overhead Delta (device-stream gap is the key indicator)",
        payload.get("host_overhead_delta", []),
    )

    tc = payload.get("torch_compile_delta", {})
    print_scalar_delta_table("torch.compile / Fusion Delta", tc.get("fusion_scalar", []))
    print_delta_table(
        "Inductor Fusion Granularity Delta (unfused_ms)",
        tc.get("fusion_granularity_families", []),
        "unfused_ms",
        limit=limit,
    )
    print_delta_table(
        "Highlighted Unfused Pointwise/Reduce Delta (total_ms)",
        [r for r in tc.get("unfused_fusion_sensitive_kernels", []) if r.get("status") == "regression"],
        "total_ms",
        limit=limit,
    )
    print_delta_table(
        "Non-Fused Kernels Delta (total_ms)",
        [r for r in tc.get("non_fused_kernels", []) if r.get("status") == "regression"],
        "total_ms",
        limit=limit,
    )
    eff_rows = [r for r in tc.get("kernel_efficiency", []) if r.get("status") == "regression"]
    if eff_rows:
        print("\n## Triton Kernel IO Efficiency Delta (utilization; lower is worse)")
        print("| Kernel | Util A | Util B | Delta | io_eff A | io_eff B | Status |")
        print("|---|---:|---:|---:|---:|---:|---|")
        for row in eff_rows[:limit]:
            u = row["bandwidth_utilization"]
            e = row["avg_io_efficiency"]
            print(
                f"| {row['name']} | {fmt(u.get('baseline'))} | {fmt(u.get('current'))} | "
                f"{fmt(u.get('delta'))} | {fmt(e.get('baseline'))} | {fmt(e.get('current'))} | "
                f"{row.get('status', '')} |"
            )

    for table_name, table in payload["named_deltas"].items():
        primary = table["primary_metric"]
        title = f"{table_name.replace('_', ' ').title()} Delta ({primary})"
        print_delta_table(title, table["top"]["regressions"], primary, limit=limit)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="baseline collect_profile_tables.py JSON")
    parser.add_argument("current", help="current collect_profile_tables.py JSON")
    parser.add_argument("--format", choices=("json", "text"), default="text")
    parser.add_argument("--limit", type=int, default=20, help="max rows for text/top outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_comparison(load_profile(args.baseline), load_profile(args.current), args.limit)
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_markdown(payload, args.limit)


if __name__ == "__main__":
    main()
