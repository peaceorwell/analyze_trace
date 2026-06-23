#!/usr/bin/env python3
"""
Triton kernel IO efficiency + output_code analysis.

triton 核心 IO 效率与 output_code 分析。

IMPORTANT: `io_efficiency` here is NOT a normalized 0-1 ratio. It is a
bandwidth-equivalent value (the kernel's effective/folded bandwidth). Judge
efficiency by comparing it against the device peak bandwidth
(meta_information deviceInfo.m_dev_basic_info.max_bandwidth); never compute
`1 - io_efficiency` on the raw value. Only the dimensionless
bandwidth_utilization = io_efficiency / peak_bandwidth may take a `1 - x` term.

This metadata is optional inductor/profiler enrichment stored in
device_task_kernel_data.extra (JSON) and is frequently absent.
"""

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path


# Theoretical (peak) bandwidth per MLU model, in the same unit as io_efficiency
# (folded bandwidth, GB/s). Used to compute bandwidth_utilization. Matched by
# substring against the device model name.
THEORETICAL_BANDWIDTH = {
    "590": 2000,
    "580": 1200,
}

IO_EFF_KEYS = (
    "io_efficiency",
    "io_eff",
    "memory_efficiency",
    "mem_efficiency",
    "IO efficiency(GB/s)",
    "io efficiency",
)
BANDWIDTH_KEYS = (
    "achieved_bandwidth",
    "bandwidth",
    "gbps",
    "effective_bandwidth",
    "achieved bandwidth(GB/s)",
)
OUTPUT_CODE_KEYS = (
    "output_code",
    "triton output code",
    "triton_code",
    "source_code",
    "kernel_code",
)
BYTES_KEYS = ("bytes", "nbytes", "num_bytes", "kernel num(GB)", "kernel_num_gb")


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


def parse_extra(extra_str):
    try:
        return json.loads(extra_str) if extra_str else {}
    except Exception:
        return {}


def normalize_metadata_key(key):
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def find_key(d, candidates):
    """Robust top-level key lookup; returns (matched_key, value) or (None, None)."""
    lowered = {k.lower(): k for k in d.keys()}
    for cand in candidates:
        key = lowered.get(cand.lower())
        if key is not None:
            return key, d[key]
    normalized = {normalize_metadata_key(k): k for k in d.keys()}
    for cand in candidates:
        key = normalized.get(normalize_metadata_key(cand))
        if key is not None:
            return key, d[key]
    return None, None


def to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if m:
            return float(m.group(0))
    return None


def get_device_name(cur):
    if table_exists(cur, "device_information"):
        row = cur.execute("SELECT name FROM device_information WHERE name IS NOT NULL LIMIT 1").fetchone()
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


def theoretical_bandwidth(device_name):
    """Peak bandwidth from the MLU model name (e.g. MLU590 -> 2000, MLU580 -> 1200)."""
    if not device_name:
        return None
    for key, bw in THEORETICAL_BANDWIDTH.items():
        if key in device_name:
            return bw
    return None


def get_meta_max_bandwidth(cur):
    if not table_exists(cur, "meta_information"):
        return None
    row = cur.execute(
        "SELECT value FROM meta_information WHERE type = 'deviceInfo'"
    ).fetchone()
    if not row:
        return None
    try:
        info = json.loads(row[0])
        return info.get("m_dev_basic_info", {}).get("max_bandwidth")
    except Exception:
        return None


def resolve_peak_bandwidth(cur):
    """Prefer the MLU-model theoretical bandwidth; fall back to meta max_bandwidth.

    Returns (peak_bandwidth, device_name, source).
    """
    device_name = get_device_name(cur)
    theo = theoretical_bandwidth(device_name)
    if theo is not None:
        return theo, device_name, "theoretical(MLU model)"
    meta = get_meta_max_bandwidth(cur)
    if meta is not None:
        return meta, device_name, "meta max_bandwidth"
    return None, device_name, None


def analyze_db(db_path, top, dump_dir):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        strings = load_strings(cur)
        if not table_exists(cur, "device_task_kernel_data"):
            return {"db": db_path, "label": Path(db_path).stem, "error": "missing device_task_kernel_data"}
        peak_bandwidth, device_name, peak_source = resolve_peak_bandwidth(cur)
        rows = cur.execute(
            "SELECT nameId, start, end, extra "
            "FROM device_task_kernel_data WHERE isComputation = 1"
        ).fetchall()

    observed_keys = set()
    by_name = {}
    for name_id, start, end, extra_str in rows:
        name = name_of(strings, name_id)
        if "triton" not in name.lower():
            continue
        extra = parse_extra(extra_str)
        if not extra:
            continue
        io_key, io_val = find_key(extra, IO_EFF_KEYS)
        bw_key, bw_val = find_key(extra, BANDWIDTH_KEYS)
        oc_key, oc_val = find_key(extra, OUTPUT_CODE_KEYS)
        for k in (io_key, bw_key, oc_key):
            if k:
                observed_keys.add(k)
        io_eff = to_float(io_val)
        bw = to_float(bw_val)
        if io_eff is None and bw is None and oc_val is None:
            continue
        dur = max(0, end - start)
        item = by_name.setdefault(
            name,
            {"durations": [], "io_eff": [], "bw": [], "output_code": None},
        )
        item["durations"].append(dur)
        if io_eff is not None:
            item["io_eff"].append((io_eff, dur))
        if bw is not None:
            item["bw"].append((bw, dur))
        if oc_val is not None and item["output_code"] is None:
            item["output_code"] = oc_val if isinstance(oc_val, str) else json.dumps(oc_val)

    has_metadata = bool(by_name)
    if not has_metadata:
        return {
            "db": db_path,
            "label": Path(db_path).stem,
            "has_io_metadata": False,
            "observed_metadata_keys": [],
            "device_name": device_name,
            "peak_bandwidth": peak_bandwidth,
            "peak_bandwidth_source": peak_source,
            "kernels": [],
        }

    def weighted_avg(pairs):
        wsum = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / wsum if wsum else None

    kernels = []
    for name, item in by_name.items():
        total_ns = sum(item["durations"])
        avg_io_eff = weighted_avg(item["io_eff"]) if item["io_eff"] else None
        min_io_eff = min((v for v, _ in item["io_eff"]), default=None)
        avg_bw = weighted_avg(item["bw"]) if item["bw"] else None

        utilization = None
        if avg_io_eff is not None and peak_bandwidth:
            try:
                utilization = avg_io_eff / float(peak_bandwidth)
            except (TypeError, ZeroDivisionError):
                utilization = None

        if utilization is not None:
            improvement_target = ms(total_ns) * (1 - min(1.0, max(0.0, utilization)))
        else:
            improvement_target = None

        kernels.append(
            {
                "kernel_name": name,
                "count": len(item["durations"]),
                "total_ms": ms(total_ns),
                "avg_io_efficiency": avg_io_eff,
                "min_io_efficiency": min_io_eff,
                "avg_achieved_bandwidth": avg_bw,
                "bandwidth_utilization": utilization,
                "improvement_target": improvement_target,
                "_output_code": item["output_code"],
            }
        )

    # Rank: by improvement_target when available, else by lowest folded bandwidth
    # (avg_io_efficiency ascending) weighted by total time.
    if any(k["improvement_target"] is not None for k in kernels):
        kernels.sort(key=lambda k: -(k["improvement_target"] or 0.0))
        ranking = "improvement_target = total_ms * (1 - bandwidth_utilization)"
    else:
        kernels.sort(
            key=lambda k: (
                k["avg_io_efficiency"] if k["avg_io_efficiency"] is not None else float("inf"),
                -k["total_ms"],
            )
        )
        ranking = "lowest folded bandwidth (avg_io_efficiency asc), weighted by total_ms"

    top_kernels = kernels[:top]

    # Dump output_code excerpts.
    for idx, k in enumerate(top_kernels):
        code = k.pop("_output_code", None)
        if not code:
            k["output_code_excerpt"] = None
            k["output_code_file"] = None
            continue
        excerpt = code[:1200]
        k["output_code_excerpt"] = excerpt
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", k["kernel_name"])[:80]
            fname = os.path.join(dump_dir, f"triton_output_code_{idx:02d}_{safe}.txt")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(code)
            k["output_code_file"] = fname
        else:
            k["output_code_file"] = None
    for k in kernels[top:]:
        k.pop("_output_code", None)

    return {
        "db": db_path,
        "label": Path(db_path).stem,
        "has_io_metadata": True,
        "observed_metadata_keys": sorted(observed_keys),
        "device_name": device_name,
        "peak_bandwidth": peak_bandwidth,
        "peak_bandwidth_source": peak_source,
        "io_efficiency_is_bandwidth_not_percentage": True,
        "ranking_method": ranking,
        "kernel_count_with_metadata": len(kernels),
        "top_low_bandwidth_kernels": top_kernels,
    }


def print_markdown(payload):
    print("# Triton Kernel IO Efficiency")
    print(
        "\n_io_efficiency is a folded/effective BANDWIDTH value, not a 0-1 percentage. "
        "Compared against device peak bandwidth._"
    )
    for profile in payload["profiles"]:
        print(f"\n## {profile.get('label', profile.get('db'))}")
        if profile.get("error"):
            print(f"- Error: {profile['error']}")
            continue
        if not profile["has_io_metadata"]:
            print("- No output_code / io_efficiency metadata on any triton kernel; branch should be skipped.")
            continue
        print(f"- Observed metadata keys: {', '.join(profile['observed_metadata_keys']) or '(none)'}")
        print(
            f"- Device: {profile.get('device_name')}, peak bandwidth: {profile['peak_bandwidth']} "
            f"(source: {profile.get('peak_bandwidth_source')}; verify units match io_efficiency)"
        )
        print(f"- Kernels with metadata: {profile['kernel_count_with_metadata']}")
        print(f"- Ranking: {profile['ranking_method']}")
        print("\n### Top Low-Bandwidth Kernels")
        print("| Kernel | Count | Total ms | Folded BW (io_eff) | Min io_eff | Achieved BW | Util | Improvement target |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k in profile["top_low_bandwidth_kernels"]:
            util = f"{k['bandwidth_utilization']*100:.1f}%" if k["bandwidth_utilization"] is not None else "-"
            tgt = f"{k['improvement_target']:.2f}" if k["improvement_target"] is not None else "-"
            io_eff = f"{k['avg_io_efficiency']:,.2f}" if k["avg_io_efficiency"] is not None else "-"
            min_eff = f"{k['min_io_efficiency']:,.2f}" if k["min_io_efficiency"] is not None else "-"
            bw = f"{k['avg_achieved_bandwidth']:,.2f}" if k["avg_achieved_bandwidth"] is not None else "-"
            print(
                f"| {k['kernel_name']} | {k['count']:,} | {k['total_ms']:,.2f} | "
                f"{io_eff} | {min_eff} | {bw} | {util} | {tgt} |"
            )
            if k.get("output_code_file"):
                print(f"  - output_code: {k['output_code_file']}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", nargs="+", help="cnperf SQLite DB path(s)")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--dump-dir", help="directory to write full output_code per top kernel")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {"profiles": [analyze_db(db_path, args.top, args.dump_dir) for db_path in args.db]}
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_markdown(payload)


if __name__ == "__main__":
    main()
