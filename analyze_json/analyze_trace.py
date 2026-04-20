import argparse
import bisect
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


def pct(a, b):
    """Percentage change from a to b. Returns empty string if a is zero."""
    if not a:
        return ""
    return f"{(b - a) / a * 100:+.1f}%"


def classify_kernel(name, args, kernel_types):
    """Classify a GPU kernel by checking kernel_types patterns in order.

    kernel_types is a list of pattern strings; the first pattern found as a
    case-insensitive substring of name is returned as the type.
    Triton kernels (identified via args) are always classified as "triton".
    Unmatched kernels return "other".
    """
    if name.startswith("triton_"):
        return "triton"
    nl = name.lower()
    for pattern in kernel_types:
        if pattern in nl:
            return pattern
    if args.get("Collective name"):
        return "collective"
    return "other"


def write_triton_code_file(code_dir, idx, kernel):
    """Write kernel["triton_output_code"] to a .py file; return the filename."""
    safe_name = kernel["kernel_name"].replace("/", "_").replace(" ", "_")
    code_filename = f"kernel_{idx}_{safe_name}.py"
    with open(os.path.join(code_dir, code_filename), "w") as cf:
        cf.write(kernel["triton_output_code"])
    return code_filename


def write_avg_csv(path, data, name_field):
    """Write {name -> {avg_count, avg_dur_ms}} to a CSV and print confirmation."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[name_field, "avg_count", "avg_dur_ms"])
        writer.writeheader()
        for name, s in data.items():
            writer.writerow({name_field: name, "avg_count": fmt3(s["avg_count"]), "avg_dur_ms": fmt3(s["avg_dur_ms"])})
    print(f"Wrote {path} ({len(data)} rows)")


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
        elif cat == "gpu_user_annotation" and (name.startswith("cncl") or name.startswith("nccl")):
            cncl_events.append(e)

    sorted_steps = sorted(step_ranges.items(), key=lambda x: x[1][0])
    step_starts  = [v[0] for _, v in sorted_steps]
    step_ends    = [v[1] for _, v in sorted_steps]
    step_nums    = [k    for k, _ in sorted_steps]

    def find_step(ts):
        i = bisect.bisect_right(step_starts, ts) - 1
        if i >= 0 and ts <= step_ends[i]:
            return step_nums[i]
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

        # For triton kernels, additionally accumulate any matching user patterns
        # as sub-categories (a kernel can appear in both "triton" and e.g. "triton_red")
        if ktype == "triton":
            nl = name.lower()
            for pattern in kernel_types:
                if pattern in nl:
                    step_to_kernel_types[step][pattern]["count"]  += 1
                    step_to_kernel_types[step][pattern]["dur_ms"] += dur_ms

        if name.startswith("triton_"):
            has_code = "triton output code" in args
            step_to_triton[step].append({
                "kernel_name":         name,
                "dur(ms)":             dur_ms if raw_dur is not None else None,
                "total io(GB)":        float(args["kernel num(GB)"])      if has_code else None,
                "IO efficiency(GB/s)": float(args["IO efficiency(GB/s)"]) if has_code else None,
                "tiling config":       args["kernel kwargs"]              if has_code else None,
                "triton_output_code":  args["triton output code"]         if has_code else None,
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
    if not n:
        return {}
    result = {}
    zero = {"count": 0, "dur_ms": 0.0}
    for name in all_names:
        entries = [step_to_dict[s].get(name) or zero for s in steps]
        result[name] = {
            "avg_count":  sum(e["count"]  for e in entries) / n,
            "avg_dur_ms": sum(e["dur_ms"] for e in entries) / n,
        }
    return dict(sorted(result.items(), key=lambda x: -x[1]["avg_dur_ms"]))


def compute_avgs(parsed, kernel_types):
    """Compute all average stats from a parsed trace. Returns a data dict."""
    step_to_triton, step_to_kernels, step_to_kernel_types, step_to_aten, step_to_cncl, step_durations = parsed
    all_steps = sorted(set(step_durations) | set(step_to_kernels) | set(step_to_aten) | set(step_to_cncl))
    n_steps   = len(all_steps)
    mean      = lambda vals: sum(vals) / n_steps if n_steps else 0.0

    KERNEL_TYPES = ["triton"] + kernel_types + ["collective", "other"]

    step_stats = {}
    for step in all_steps:
        sd  = step_durations.get(step, 0.0)
        kc  = sum(v["count"]  for v in step_to_kernels[step].values())
        kd  = sum(v["dur_ms"] for v in step_to_kernels[step].values())
        tc  = len(step_to_triton[step])
        td  = sum((k["dur(ms)"] or 0.0) for k in step_to_triton[step])
        ac  = sum(v["count"]  for v in step_to_aten[step].values())
        ad  = sum(v["dur_ms"] for v in step_to_aten[step].values())
        cc  = sum(v["count"]  for v in step_to_cncl[step].values())
        cd  = sum(v["dur_ms"] for v in step_to_cncl[step].values())
        ckd = kd - cd
        step_stats[step] = (sd, kc, ckd, tc, td, ac, ad, cc, cd)

    avg_row = tuple(mean([step_stats[s][i] for s in all_steps]) for i in range(9))

    kt_avgs = {
        ktype: (
            mean([step_to_kernel_types[s].get(ktype, {"count": 0})["count"]    for s in all_steps]),
            mean([step_to_kernel_types[s].get(ktype, {"dur_ms": 0.0})["dur_ms"] for s in all_steps]),
        )
        for ktype in KERNEL_TYPES
    }

    # Triton aggregation: per step by kernel name
    step_triton_agg = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0, "io_gb": 0.0, "io_eff": 0.0}))
    for step, kernels in step_to_triton.items():
        for k in kernels:
            a = step_triton_agg[step][k["kernel_name"]]
            a["count"]  += 1
            a["dur_ms"] += k["dur(ms)"] or 0.0
            if k["total io(GB)"] is not None:
                a["io_gb"]  += k["total io(GB)"]
                a["io_eff"] += k["IO efficiency(GB/s)"]

    all_triton_names = set()
    for s in all_steps:
        all_triton_names.update(step_triton_agg[s])

    avg_triton = {}
    for name in all_triton_names:
        avg_triton[name] = {
            "avg_count":  mean([step_triton_agg[s].get(name, {"count": 0})["count"]    for s in all_steps]),
            "avg_dur_ms": mean([step_triton_agg[s].get(name, {"dur_ms": 0.0})["dur_ms"] for s in all_steps]),
            "avg_io_gb":  mean([step_triton_agg[s].get(name, {"io_gb": 0.0})["io_gb"]   for s in all_steps]),
            "avg_io_eff": mean([step_triton_agg[s].get(name, {"io_eff": 0.0})["io_eff"] for s in all_steps]),
        }
    avg_triton = dict(sorted(avg_triton.items(), key=lambda x: -x[1]["avg_dur_ms"]))

    return {
        "all_steps":      all_steps,
        "n_steps":        n_steps,
        "step_stats":     step_stats,
        "avg_row":        avg_row,
        "KERNEL_TYPES":   KERNEL_TYPES,
        "kt_avgs":        kt_avgs,
        "avg_kernels":    avg_stats(step_to_kernels, all_steps),
        "avg_aten":       avg_stats(step_to_aten, all_steps),
        "avg_cncl":       avg_stats(step_to_cncl, all_steps),
        "avg_triton":     avg_triton,
        "step_to_triton": step_to_triton,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

_HDR = (f"{'step':<8} {'step_dur(ms)':<14} {'kernels':<10} {'compute_kernel_dur(ms)':<24}"
        f" {'triton':<10} {'triton_dur(ms)':<16} {'aten_ops':<10} {'aten_dur(ms)':<14}"
        f" {'cncl':<8} {'cncl_dur(ms)':<14}")


def print_step_summary(data, label=""):
    title = f"=== Per-Step Summary ({data['n_steps']} steps)"
    if label:
        title += f" — {label}"
    print(f"\n{title} ===")
    print(_HDR)
    print("-" * len(_HDR))
    for step in data["all_steps"]:
        sd, kc, ckd, tc, td, ac, ad, cc, cd = data["step_stats"][step]
        print(f"{step:<8} {sd:<14.3f} {kc:<10} {ckd:<24.3f} {tc:<10} {td:<16.3f}"
              f" {ac:<10} {ad:<14.3f} {cc:<8} {cd:<14.3f}")
    avg_sd, avg_kc, avg_ckd, avg_tc, avg_td, avg_ac, avg_ad, avg_cc, avg_cd = data["avg_row"]
    print("-" * len(_HDR))
    print(f"{'avg':<8} {avg_sd:<14.3f} {avg_kc:<10.1f} {avg_ckd:<24.3f} {avg_tc:<10.1f} {avg_td:<16.3f}"
          f" {avg_ac:<10.1f} {avg_ad:<14.3f} {avg_cc:<8.1f} {avg_cd:<14.3f}")


def print_kernel_type_breakdown(data, label=""):
    title = f"=== Kernel Type Breakdown (avg across {data['n_steps']} steps)"
    if label:
        title += f" — {label}"
    print(f"\n{title} ===")
    hdr = f"{'type':<12} {'avg_count':<12} {'avg_dur_ms':<14}"
    print(hdr)
    print("-" * len(hdr))
    for ktype in data["KERNEL_TYPES"]:
        ac, ad = data["kt_avgs"][ktype]
        print(f"{ktype:<12} {ac:<12.1f} {ad:<14.3f}")


def print_comparison(data_a, data_b, label_a, label_b):
    # Per-step summaries
    print_step_summary(data_a, label_a)
    print_step_summary(data_b, label_b)

    # Avg row comparison
    METRICS = [
        "step_dur(ms)", "kernels", "compute_kernel_dur(ms)", "triton",
        "triton_dur(ms)", "aten_ops", "aten_dur(ms)", "cncl", "cncl_dur(ms)",
    ]
    la, lb = label_a[:16], label_b[:16]
    print(f"\n=== Avg Comparison ({label_a} vs {label_b}) ===")
    hdr = f"{'metric':<26} {la:<18} {lb:<18} {'delta':<14} {'pct_change':<12}"
    print(hdr)
    print("-" * len(hdr))
    for i, metric in enumerate(METRICS):
        va, vb = data_a["avg_row"][i], data_b["avg_row"][i]
        print(f"{metric:<26} {va:<18.3f} {vb:<18.3f} {vb - va:<+14.3f} {pct(va, vb):<12}")

    # Kernel type comparison
    print(f"\n=== Kernel Type Comparison ({label_a} vs {label_b}) ===")
    hdr2 = (f"{'type':<12} {'count_A':<10} {'count_B':<10} {'dur_A(ms)':<12}"
            f" {'dur_B(ms)':<12} {'delta_dur':<12} {'pct':<10}")
    print(hdr2)
    print("-" * len(hdr2))
    for ktype in data_a["KERNEL_TYPES"]:
        ac_a, ad_a = data_a["kt_avgs"].get(ktype, (0.0, 0.0))
        ac_b, ad_b = data_b["kt_avgs"].get(ktype, (0.0, 0.0))
        print(f"{ktype:<12} {ac_a:<10.1f} {ac_b:<10.1f} {ad_a:<12.3f}"
              f" {ad_b:<12.3f} {ad_b - ad_a:<+12.3f} {pct(ad_a, ad_b):<10}")


# ── CSV write helpers ─────────────────────────────────────────────────────────

def _write_triton_avg_csv(path, avg_triton):
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


def _write_kernel_types_csv(path, kernel_types, kt_avgs):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "avg_count", "avg_dur_ms"])
        writer.writeheader()
        for ktype in kernel_types:
            ac, ad = kt_avgs[ktype]
            writer.writerow({"type": ktype, "avg_count": fmt3(ac), "avg_dur_ms": fmt3(ad)})
    print(f"Wrote {path} ({len(kernel_types)} rows)")


def _write_cmp_avg_csv(path, data_a, data_b, name_field):
    """Comparison CSV for avg stats (kernels or ops). Sorted by |delta_dur_ms| desc."""
    zero = {"avg_count": 0.0, "avg_dur_ms": 0.0}
    rows = []
    for name in set(data_a) | set(data_b):
        a, b  = data_a.get(name, zero), data_b.get(name, zero)
        delta = b["avg_dur_ms"] - a["avg_dur_ms"]
        rows.append({
            name_field:     name,
            "avg_dur_ms_A": fmt3(a["avg_dur_ms"]),
            "avg_dur_ms_B": fmt3(b["avg_dur_ms"]),
            "delta_dur_ms": fmt3(delta),
            "pct_change":   pct(a["avg_dur_ms"], b["avg_dur_ms"]),
            "avg_count_A":  fmt3(a["avg_count"]),
            "avg_count_B":  fmt3(b["avg_count"]),
            "_sort":        abs(delta),
        })
    rows.sort(key=lambda r: -r["_sort"])
    fields = [name_field, "avg_dur_ms_A", "avg_dur_ms_B", "delta_dur_ms", "pct_change", "avg_count_A", "avg_count_B"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _write_triton_cmp_csv(path, avg_triton_a, avg_triton_b):
    zero = {"avg_count": 0.0, "avg_dur_ms": 0.0, "avg_io_gb": 0.0, "avg_io_eff": 0.0}
    rows = []
    for name in set(avg_triton_a) | set(avg_triton_b):
        a, b  = avg_triton_a.get(name, zero), avg_triton_b.get(name, zero)
        delta = b["avg_dur_ms"] - a["avg_dur_ms"]
        rows.append({
            "kernel_name":  name,
            "avg_dur_ms_A": fmt3(a["avg_dur_ms"]),
            "avg_dur_ms_B": fmt3(b["avg_dur_ms"]),
            "delta_dur_ms": fmt3(delta),
            "pct_change":   pct(a["avg_dur_ms"], b["avg_dur_ms"]),
            "avg_count_A":  fmt3(a["avg_count"]),
            "avg_count_B":  fmt3(b["avg_count"]),
            "avg_io_gb_A":  fmt3(a["avg_io_gb"]),
            "avg_io_gb_B":  fmt3(b["avg_io_gb"]),
            "_sort":        abs(delta),
        })
    rows.sort(key=lambda r: -r["_sort"])
    fields = ["kernel_name", "avg_dur_ms_A", "avg_dur_ms_B", "delta_dur_ms", "pct_change",
              "avg_count_A", "avg_count_B", "avg_io_gb_A", "avg_io_gb_B"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _write_kernel_types_cmp_csv(path, data_a, data_b):
    rows = []
    for ktype in data_a["KERNEL_TYPES"]:
        ac_a, ad_a = data_a["kt_avgs"].get(ktype, (0.0, 0.0))
        ac_b, ad_b = data_b["kt_avgs"].get(ktype, (0.0, 0.0))
        rows.append({
            "type":         ktype,
            "avg_dur_ms_A": fmt3(ad_a),
            "avg_dur_ms_B": fmt3(ad_b),
            "delta_dur_ms": fmt3(ad_b - ad_a),
            "pct_change":   pct(ad_a, ad_b),
            "avg_count_A":  fmt3(ac_a),
            "avg_count_B":  fmt3(ac_b),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "avg_dur_ms_A", "avg_dur_ms_B",
                                               "delta_dur_ms", "pct_change", "avg_count_A", "avg_count_B"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


# ── Top-level write functions ─────────────────────────────────────────────────

def write_single(data, args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Per-step triton CSVs + source files
    if args.save_triton_csv or args.save_triton_code:
        triton_fields = ["kernel_name", "dur(ms)", "total io(GB)", "IO efficiency(GB/s)", "tiling config", "triton_code_file"]
        for step in data["all_steps"]:
            kernels = [k for k in data["step_to_triton"][step] if k["triton_output_code"] is not None]
            if not kernels:
                continue
            code_dir = os.path.join(args.output_dir, f"step_{step}_triton_codes")
            if args.save_triton_code:
                os.makedirs(code_dir, exist_ok=True)
            if args.save_triton_csv:
                csv_path = os.path.join(args.output_dir, f"step_{step}_triton_kernels.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=triton_fields)
                    writer.writeheader()
                    for idx, kernel in enumerate(kernels):
                        row = {
                            "kernel_name":         kernel["kernel_name"],
                            "dur(ms)":             fmt3(kernel["dur(ms)"]),
                            "total io(GB)":        fmt3(kernel["total io(GB)"]),
                            "IO efficiency(GB/s)": fmt3(kernel["IO efficiency(GB/s)"]),
                            "tiling config":       kernel["tiling config"].replace("\n", "\\n").replace("\r", ""),
                            "triton_code_file":    "",
                        }
                        if args.save_triton_code:
                            fname = write_triton_code_file(code_dir, idx, kernel)
                            row["triton_code_file"] = os.path.join(f"step_{step}_triton_codes", fname)
                        writer.writerow(row)
                print(f"Wrote {csv_path} ({len(kernels)} rows)")
            elif args.save_triton_code:
                for idx, kernel in enumerate(kernels):
                    write_triton_code_file(code_dir, idx, kernel)
                print(f"Wrote {code_dir}/ ({len(kernels)} files)")

    write_avg_csv(os.path.join(args.output_dir, "all_kernels_avg.csv"), data["avg_kernels"], "kernel_name")
    _write_triton_avg_csv(os.path.join(args.output_dir, "triton_kernels_avg.csv"), data["avg_triton"])
    write_avg_csv(os.path.join(args.output_dir, "aten_ops_avg.csv"),    data["avg_aten"],    "op_name")
    _write_kernel_types_csv(os.path.join(args.output_dir, "kernel_types_avg.csv"), data["KERNEL_TYPES"], data["kt_avgs"])
    write_avg_csv(os.path.join(args.output_dir, "cncl_ops_avg.csv"),    data["avg_cncl"],    "op_name")


def write_comparison(data_a, data_b, args):
    os.makedirs(args.output_dir, exist_ok=True)
    _write_cmp_avg_csv(os.path.join(args.output_dir, "all_kernels_cmp.csv"),
                       data_a["avg_kernels"], data_b["avg_kernels"], "kernel_name")
    _write_triton_cmp_csv(os.path.join(args.output_dir, "triton_kernels_cmp.csv"),
                          data_a["avg_triton"], data_b["avg_triton"])
    _write_cmp_avg_csv(os.path.join(args.output_dir, "aten_ops_cmp.csv"),
                       data_a["avg_aten"], data_b["avg_aten"], "op_name")
    _write_kernel_types_cmp_csv(os.path.join(args.output_dir, "kernel_types_cmp.csv"), data_a, data_b)
    _write_cmp_avg_csv(os.path.join(args.output_dir, "cncl_ops_cmp.csv"),
                       data_a["avg_cncl"], data_b["avg_cncl"], "op_name")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse PyTorch profiler trace JSON(s) and extract kernel/op info per ProfilerStep. "
                    "Provide two files to compare them."
    )
    parser.add_argument("trace_files", nargs="+", metavar="trace_file",
                        help="One or two profiler trace JSON files.")
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to write output files (default: current directory)",
    )
    parser.add_argument(
        "-c", "--save-triton-code",
        action="store_true",
        help="Save triton output code for each kernel to individual .py files (default: off)",
    )
    parser.add_argument(
        "-s", "--save-triton-csv",
        action="store_true",
        help="Save per-step triton kernel CSV files (default: off)",
    )
    parser.add_argument(
        "-k", "--kernel-types",
        default="gemm,embedding,pool",
        metavar="PATTERN,...",
        help="Comma-separated list of name patterns for kernel classification (case-insensitive substring match). "
             "Each pattern becomes its own category. First match wins. "
             "Default: gemm,embedding,pool",
    )
    args = parser.parse_args()
    if len(args.trace_files) > 2:
        parser.error("At most two trace files can be provided.")
    kernel_types = [p for p in args.kernel_types.split(",") if p]

    if len(args.trace_files) == 1:
        data = compute_avgs(parse_trace(args.trace_files[0], kernel_types), kernel_types)
        print_step_summary(data)
        print_kernel_type_breakdown(data)
        write_single(data, args)
    else:
        label_a = os.path.basename(args.trace_files[0])
        label_b = os.path.basename(args.trace_files[1])
        data_a = compute_avgs(parse_trace(args.trace_files[0], kernel_types), kernel_types)
        data_b = compute_avgs(parse_trace(args.trace_files[1], kernel_types), kernel_types)
        print_comparison(data_a, data_b, label_a, label_b)
        write_comparison(data_a, data_b, args)
