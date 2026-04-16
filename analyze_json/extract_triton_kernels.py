import argparse
import csv
import json
import os
from collections import defaultdict
from decimal import Decimal


def fmt3(val):
    """Format a number to 3 significant figures without scientific notation."""
    if val is None:
        return ""
    return format(Decimal(f"{float(val):.3g}"), "f")


def classify_kernel(name, args, kernel_types):
    """Classify a GPU kernel by checking kernel_types patterns in order.

    kernel_types is a list of pattern strings; the first pattern found as a
    case-insensitive substring of name is returned as the type.
    Triton kernels (identified via args) are always classified as "triton".
    Unmatched kernels return "other".
    """
    if "triton output code" in args:
        return "triton"
    nl = name.lower()
    for pattern in kernel_types:
        if pattern in nl:
            return pattern
    return "other"


def parse_trace(trace_file, kernel_types):
    """Parse a PyTorch profiler trace JSON.

    Returns:
        step_to_triton:       step -> [{kernel_name, dur(ms), total io(GB), IO efficiency(GB/s),
                                        tiling config, triton_output_code}]
        step_to_kernels:      step -> {kernel_name -> {"count": int, "dur_ms": float}}
        step_to_kernel_types: step -> {type -> {"count": int, "dur_ms": float}}
                              types: triton, gemm, embedding, pooling, other
        step_to_aten:         step -> {op_name -> {"count": int, "dur_ms": float}}
        step_to_cncl:         step -> {op_name -> {"count": int, "dur_ms": float}}
        step_durations:       step -> wall-clock duration in ms (from ProfilerStep# event)
    """
    with open(trace_file) as f:
        trace = json.load(f)

    events = trace["traceEvents"]

    # step_num -> (start_ts, end_ts)
    step_ranges    = {}
    step_durations = {}   # step_num -> ms
    all_kernel_events = []
    aten_events       = []
    cncl_events       = []

    for e in events:
        name = e.get("name", "")
        cat  = e.get("cat", "")
        if name.startswith("ProfilerStep#") and cat == "user_annotation":
            ts  = e.get("ts", 0)
            dur = e.get("dur", 0)
            step_num = int(name.split("#")[-1])
            step_ranges[step_num]    = (ts, ts + dur)
            step_durations[step_num] = dur / 1000
        elif cat == "kernel":
            all_kernel_events.append(e)
        elif name.startswith("aten::"):
            aten_events.append(e)
        elif cat == "gpu_user_annotation" and name.startswith("cncl"):
            cncl_events.append(e)

    def find_step(ts):
        for step_num, (start, end) in step_ranges.items():
            if start <= ts <= end:
                return step_num
        return None

    step_to_triton       = defaultdict(list)
    step_to_kernels      = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))
    step_to_kernel_types = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))
    step_to_aten         = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))
    step_to_cncl         = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))

    for e in all_kernel_events:
        step = find_step(e.get("ts", 0))
        if step is None:
            continue
        name    = e.get("name", "")
        raw_dur = e.get("dur")
        dur_ms  = raw_dur / 1000 if raw_dur is not None else 0.0
        step_to_kernels[step][name]["count"]  += 1
        step_to_kernels[step][name]["dur_ms"] += dur_ms

        args  = e.get("args", {})
        ktype = classify_kernel(name, args, kernel_types)
        step_to_kernel_types[step][ktype]["count"]  += 1
        step_to_kernel_types[step][ktype]["dur_ms"] += dur_ms

        if "triton output code" in args:
            step_to_triton[step].append({
                "kernel_name":         name,
                "dur(ms)":             dur_ms if raw_dur is not None else None,
                "total io(GB)":        float(args["kernel num(GB)"]),
                "IO efficiency(GB/s)": float(args["IO efficiency(GB/s)"]),
                "tiling config":       args["kernel kwargs"],
                "triton_output_code":  args["triton output code"],
            })

    for e in aten_events:
        step = find_step(e.get("ts", 0))
        if step is None:
            continue
        name    = e.get("name", "")
        raw_dur = e.get("dur")
        dur_ms  = raw_dur / 1000 if raw_dur is not None else 0.0
        step_to_aten[step][name]["count"]  += 1
        step_to_aten[step][name]["dur_ms"] += dur_ms

    for e in cncl_events:
        step = find_step(e.get("ts", 0))
        if step is None:
            continue
        name    = e.get("name", "")
        raw_dur = e.get("dur")
        dur_ms  = raw_dur / 1000 if raw_dur is not None else 0.0
        step_to_cncl[step][name]["count"]  += 1
        step_to_cncl[step][name]["dur_ms"] += dur_ms

    return step_to_triton, step_to_kernels, step_to_kernel_types, step_to_aten, step_to_cncl, step_durations


def avg_stats(step_to_dict, steps):
    """Average {name -> {count, dur_ms}} across steps.

    Returns {name -> {avg_count, avg_dur_ms}}, sorted by avg_dur_ms descending.
    """
    all_names = set()
    for s in steps:
        all_names.update(step_to_dict[s])
    n = len(steps)
    result = {}
    for name in all_names:
        result[name] = {
            "avg_count":  sum(step_to_dict[s].get(name, {"count": 0})["count"]  for s in steps) / n,
            "avg_dur_ms": sum(step_to_dict[s].get(name, {"dur_ms": 0.0})["dur_ms"] for s in steps) / n,
        }
    return dict(sorted(result.items(), key=lambda x: -x[1]["avg_dur_ms"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a PyTorch profiler trace JSON and extract kernel/op info per ProfilerStep."
    )
    parser.add_argument("trace_file", help="Path to the profiler trace JSON file (e.g. trace.json)")
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to write output files (default: current directory)",
    )
    parser.add_argument(
        "--kernel-types",
        default="gemm,embedding,pool",
        metavar="PATTERN,...",
        help="Comma-separated list of name patterns for kernel classification (case-insensitive substring match). "
             "Each pattern becomes its own category. First match wins. "
             "Default: gemm,embedding,pool",
    )
    args = parser.parse_args()
    kernel_types = [p for p in args.kernel_types.split(",") if p]

    step_to_triton, step_to_kernels, step_to_kernel_types, step_to_aten, step_to_cncl, step_durations = parse_trace(args.trace_file, kernel_types)
    all_steps = sorted(set(step_durations) | set(step_to_kernels) | set(step_to_aten) | set(step_to_cncl))
    n_steps   = len(all_steps)

    def mean(vals): return sum(vals) / n_steps if n_steps else 0.0

    # ── Per-step summary ──────────────────────────────────────────────────────
    print(f"\n=== Per-Step Summary ({n_steps} steps) ===")
    hdr = (f"{'step':<8} {'step_dur(ms)':<14} {'kernels':<10} {'kernel_dur(ms)':<16}"
           f" {'triton':<10} {'triton_dur(ms)':<16} {'aten_ops':<10} {'aten_dur(ms)':<14}"
           f" {'cncl':<8} {'cncl_dur(ms)':<14}")
    print(hdr)
    print("-" * len(hdr))
    for step in all_steps:
        sd = step_durations.get(step, 0.0)
        kc = sum(v["count"]  for v in step_to_kernels[step].values())
        kd = sum(v["dur_ms"] for v in step_to_kernels[step].values())
        tc = len(step_to_triton[step])
        td = sum((k["dur(ms)"] or 0.0) for k in step_to_triton[step])
        ac = sum(v["count"]  for v in step_to_aten[step].values())
        ad = sum(v["dur_ms"] for v in step_to_aten[step].values())
        cc = sum(v["count"]  for v in step_to_cncl[step].values())
        cd = sum(v["dur_ms"] for v in step_to_cncl[step].values())
        print(f"{step:<8} {sd:<14.3f} {kc:<10} {kd:<16.3f} {tc:<10} {td:<16.3f}"
              f" {ac:<10} {ad:<14.3f} {cc:<8} {cd:<14.3f}")

    avg_sd = mean([step_durations.get(s, 0.0) for s in all_steps])
    avg_kc = mean([sum(v["count"]  for v in step_to_kernels[s].values()) for s in all_steps])
    avg_kd = mean([sum(v["dur_ms"] for v in step_to_kernels[s].values()) for s in all_steps])
    avg_tc = mean([len(step_to_triton[s]) for s in all_steps])
    avg_td = mean([sum((k["dur(ms)"] or 0.0) for k in step_to_triton[s]) for s in all_steps])
    avg_ac = mean([sum(v["count"]  for v in step_to_aten[s].values()) for s in all_steps])
    avg_ad = mean([sum(v["dur_ms"] for v in step_to_aten[s].values()) for s in all_steps])
    avg_cc = mean([sum(v["count"]  for v in step_to_cncl[s].values()) for s in all_steps])
    avg_cd = mean([sum(v["dur_ms"] for v in step_to_cncl[s].values()) for s in all_steps])
    print("-" * len(hdr))
    print(f"{'avg':<8} {avg_sd:<14.3f} {avg_kc:<10.1f} {avg_kd:<16.3f} {avg_tc:<10.1f} {avg_td:<16.3f}"
          f" {avg_ac:<10.1f} {avg_ad:<14.3f} {avg_cc:<8.1f} {avg_cd:<14.3f}")

    # ── Kernel type breakdown (averaged) ─────────────────────────────────────
    KERNEL_TYPES = ["triton"] + kernel_types + ["other"]
    print(f"\n=== Kernel Type Breakdown (avg across {n_steps} steps) ===")
    hdr2 = f"{'type':<12} {'avg_count':<12} {'avg_dur_ms':<14}"
    print(hdr2)
    print("-" * len(hdr2))
    for ktype in KERNEL_TYPES:
        ac = mean([step_to_kernel_types[s].get(ktype, {"count": 0})["count"]   for s in all_steps])
        ad = mean([step_to_kernel_types[s].get(ktype, {"dur_ms": 0.0})["dur_ms"] for s in all_steps])
        print(f"{ktype:<12} {ac:<12.1f} {ad:<14.3f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Per-step triton kernel CSVs + triton source files ─────────────────────
    triton_fields = ["kernel_name", "dur(ms)", "total io(GB)", "IO efficiency(GB/s)", "tiling config", "triton_code_file"]
    for step in all_steps:
        kernels = step_to_triton[step]
        if not kernels:
            continue
        csv_path = os.path.join(args.output_dir, f"step_{step}_triton_kernels.csv")
        code_dir = os.path.join(args.output_dir, f"step_{step}_triton_codes")
        os.makedirs(code_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=triton_fields)
            writer.writeheader()
            for idx, kernel in enumerate(kernels):
                safe_name     = kernel["kernel_name"].replace("/", "_").replace(" ", "_")
                code_filename = f"kernel_{idx}_{safe_name}.py"
                with open(os.path.join(code_dir, code_filename), "w") as cf:
                    cf.write(kernel["triton_output_code"])
                writer.writerow({
                    "kernel_name":          kernel["kernel_name"],
                    "dur(ms)":              fmt3(kernel["dur(ms)"]),
                    "total io(GB)":         fmt3(kernel["total io(GB)"]),
                    "IO efficiency(GB/s)":  fmt3(kernel["IO efficiency(GB/s)"]),
                    "tiling config":        kernel["tiling config"].replace("\n", "\\n").replace("\r", ""),
                    "triton_code_file":     os.path.join(f"step_{step}_triton_codes", code_filename),
                })
        print(f"Wrote {csv_path} ({len(kernels)} rows)")

    # ── Averaged all-kernel stats ─────────────────────────────────────────────
    avg_kernels = avg_stats(step_to_kernels, all_steps)
    path = os.path.join(args.output_dir, "all_kernels_avg.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel_name", "avg_count", "avg_dur_ms"])
        writer.writeheader()
        for name, s in avg_kernels.items():
            writer.writerow({"kernel_name": name, "avg_count": fmt3(s["avg_count"]), "avg_dur_ms": fmt3(s["avg_dur_ms"])})
    print(f"Wrote {path} ({len(avg_kernels)} rows)")

    # ── Averaged triton kernel stats ──────────────────────────────────────────
    # Aggregate per step by kernel name so we can average across steps
    step_triton_agg = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0, "io_gb": 0.0, "io_eff": 0.0}))
    for step, kernels in step_to_triton.items():
        for k in kernels:
            a = step_triton_agg[step][k["kernel_name"]]
            a["count"]  += 1
            a["dur_ms"] += k["dur(ms)"] or 0.0
            a["io_gb"]  += k["total io(GB)"]
            a["io_eff"] += k["IO efficiency(GB/s)"]

    all_triton_names = set()
    for s in all_steps:
        all_triton_names.update(step_triton_agg[s])

    avg_triton = {}
    for name in all_triton_names:
        avg_triton[name] = {
            "avg_count":  mean([step_triton_agg[s].get(name, {"count": 0})["count"]   for s in all_steps]),
            "avg_dur_ms": mean([step_triton_agg[s].get(name, {"dur_ms": 0.0})["dur_ms"] for s in all_steps]),
            "avg_io_gb":  mean([step_triton_agg[s].get(name, {"io_gb": 0.0})["io_gb"]  for s in all_steps]),
            "avg_io_eff": mean([step_triton_agg[s].get(name, {"io_eff": 0.0})["io_eff"] for s in all_steps]),
        }
    avg_triton = dict(sorted(avg_triton.items(), key=lambda x: -x[1]["avg_dur_ms"]))

    path = os.path.join(args.output_dir, "triton_kernels_avg.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel_name", "avg_count", "avg_dur_ms", "avg_io_gb", "avg_io_efficiency"])
        writer.writeheader()
        for name, s in avg_triton.items():
            writer.writerow({
                "kernel_name":       name,
                "avg_count":         fmt3(s["avg_count"]),
                "avg_dur_ms":        fmt3(s["avg_dur_ms"]),
                "avg_io_gb":         fmt3(s["avg_io_gb"]),
                "avg_io_efficiency": fmt3(s["avg_io_eff"]),
            })
    print(f"Wrote {path} ({len(avg_triton)} rows)")

    # ── Averaged aten:: op stats ──────────────────────────────────────────────
    avg_aten = avg_stats(step_to_aten, all_steps)
    path = os.path.join(args.output_dir, "aten_ops_avg.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["op_name", "avg_count", "avg_dur_ms"])
        writer.writeheader()
        for name, s in avg_aten.items():
            writer.writerow({"op_name": name, "avg_count": fmt3(s["avg_count"]), "avg_dur_ms": fmt3(s["avg_dur_ms"])})
    print(f"Wrote {path} ({len(avg_aten)} rows)")

    # ── Averaged kernel type stats ────────────────────────────────────────────
    path = os.path.join(args.output_dir, "kernel_types_avg.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "avg_count", "avg_dur_ms"])
        writer.writeheader()
        for ktype in KERNEL_TYPES:
            ac = mean([step_to_kernel_types[s].get(ktype, {"count": 0})["count"]    for s in all_steps])
            ad = mean([step_to_kernel_types[s].get(ktype, {"dur_ms": 0.0})["dur_ms"] for s in all_steps])
            writer.writerow({"type": ktype, "avg_count": fmt3(ac), "avg_dur_ms": fmt3(ad)})
    print(f"Wrote {path} ({len(KERNEL_TYPES)} rows)")

    # ── Averaged cncl op stats ────────────────────────────────────────────────
    avg_cncl = avg_stats(step_to_cncl, all_steps)
    path = os.path.join(args.output_dir, "cncl_ops_avg.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["op_name", "avg_count", "avg_dur_ms"])
        writer.writeheader()
        for name, s in avg_cncl.items():
            writer.writerow({"op_name": name, "avg_count": fmt3(s["avg_count"]), "avg_dur_ms": fmt3(s["avg_dur_ms"])})
    print(f"Wrote {path} ({len(avg_cncl)} rows)")
