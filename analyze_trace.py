import argparse
import bisect
import csv
import json
import os
import re
import subprocess
import sys
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


# ── Auto kernel classification ────────────────────────────────────────────────

# Ordered (keywords, family_label) pairs for semantic kernel family detection.
# First matching keyword wins. Keywords are checked as lowercase substrings.
_FAMILY_PATTERNS = [
    (["gemm", "sgemm", "dgemm", "hgemm", "igemm", "bgemm", "cutlass", "matmul", "cublas"], "gemm"),
    (["flash_attn", "flash_attention", "fmha", "scaled_dot_product", "self_attention"],     "attention"),
    (["layer_norm", "layernorm", "rms_norm", "rmsnorm", "group_norm", "groupnorm",
      "batch_norm", "batchnorm"],                                                            "norm"),
    (["elementwise", "pointwise", "eltwise"],                                               "elementwise"),
    (["embedding", "lookup_table"],                                                         "embedding"),
    (["conv2d", "conv1d", "conv3d", "convolution", "scudnn", "cudnn_conv", "winograd"],    "conv"),
    (["softmax", "log_softmax"],                                                            "softmax"),
    (["reduce_", "cub::device_reduce", "sum_kernel", "mean_kernel"],                        "reduce"),
    (["dropout"],                                                                           "dropout"),
    (["index_", "scatter", "gather_", "take_"],                                             "index_op"),
    (["sort_", "topk", "argsort"],                                                          "sort"),
    (["copy_", "memcpy", "fill_", "zeros_", "ones_"],                                       "memory"),
]

# Pre-compiled regexes for kernel name normalization (used in fallback)
_STRIP_TEMPLATE_RE = re.compile(r'<.*')
_STRIP_LEADING_RE  = re.compile(r'^(void\s+|at::native::|\w+::)+', re.IGNORECASE)


def extract_kernel_family(name: str) -> str:
    """Map a GPU kernel name to a semantic family label.

    Priority order:
    1. triton_ prefix  → triton sub-type (triton_reduce / triton_pointwise / triton_<sub>)
    2. Collective / communication keywords  (checked BEFORE semantic patterns to avoid
       misclassifying e.g. TCDP_RING_ALLREDUCE as "reduce")
    3. Known semantic patterns from _FAMILY_PATTERNS
    4. Fallback: first meaningful token from the cleaned name
    """
    nl = name.lower()

    # Triton kernels — group by sub-type token
    if nl.startswith("triton_"):
        parts = name.split("_")
        if len(parts) >= 2:
            sub = parts[1].lower()
            if sub in ("red", "per"):    # reduction / persistent-reduction
                return "triton_reduce"
            if sub in ("poi", "tem"):    # pointwise / template-pointwise
                return "triton_pointwise"
            if sub == "mm":
                return "triton_mm"
            return f"triton_{sub}"
        return "triton"

    # Collective / communication — must come before _FAMILY_PATTERNS so that names like
    # TCDP_RING_ALLREDUCE_* are not matched by the "reduce_" pattern first.
    if nl.startswith("tcdp") or any(kw in nl for kw in (
            "nccl", "cncl", "collective",
            "allreduce", "allgather", "reducescatter", "broadcast_")):
        return "collective"

    # Known semantic families
    for keywords, family in _FAMILY_PATTERNS:
        for kw in keywords:
            if kw in nl:
                return family

    # Fallback: strip templates / namespaces / "void", take first meaningful token.
    # Preserve the original case — the token IS the type name, not a synthetic label.
    clean = _STRIP_TEMPLATE_RE.sub("", name)
    clean = _STRIP_LEADING_RE.sub("", clean)
    tokens = [t for t in re.split(r'[_\s:]+', clean) if t and not t.isdigit() and len(t) > 1]
    if tokens:
        return tokens[0][:24]
    return "other"


def auto_classify_kernels(avg_kernels: dict) -> tuple:
    """Classify kernel families from aggregated per-kernel stats.

    Every distinct non-collective family with any duration gets its own category,
    sorted by avg_dur_ms descending.  "other" is always last.
    Collective kernels are excluded from KERNEL_TYPES — they are handled separately
    and should not appear in Kernel Type Breakdown / chart / kernel_types_avg.csv.

    Returns:
        (KERNEL_TYPES, kt_avgs)
        KERNEL_TYPES : list of compute family labels sorted by avg_dur_ms desc,
                       "other" last.  "collective" is NOT included.
        kt_avgs      : {type -> (avg_count, avg_dur_ms)}  — includes "collective"
                       so callers can still reference it for percentage calculations.
    """
    if not avg_kernels:
        return ["other"], {"other": (0.0, 0.0)}

    # Aggregate per-kernel stats into families
    family_dur   = defaultdict(float)
    family_count = defaultdict(float)
    for name, stats in avg_kernels.items():
        fam = extract_kernel_family(name)
        family_dur[fam]   += stats["avg_dur_ms"]
        family_count[fam] += stats["avg_count"]

    # All distinct compute families (exclude collective and other; other goes last)
    compute_fams = [
        f for f in family_dur
        if f not in ("collective", "other") and family_dur[f] > 0
    ]
    compute_fams.sort(key=lambda f: -family_dur[f])

    KERNEL_TYPES = compute_fams + ["other"]

    kt_avgs: dict = {f: (family_count[f], family_dur[f]) for f in compute_fams}
    kt_avgs["other"]      = (family_count.get("other", 0.0),      family_dur.get("other", 0.0))
    # Keep collective in kt_avgs for callers that need it in percentage calculations
    kt_avgs["collective"] = (family_count.get("collective", 0.0), family_dur.get("collective", 0.0))

    return KERNEL_TYPES, kt_avgs


def write_triton_code_file(code_dir, idx, kernel):
    """Write kernel["triton_output_code"] to a .py file; return the filename."""
    safe_name = kernel["kernel_name"].replace("/", "_").replace(" ", "_")
    code_filename = f"kernel_{idx}_{safe_name}.py"
    with open(os.path.join(code_dir, code_filename), "w") as cf:
        cf.write(kernel["triton_output_code"])
    return code_filename


def run_triton_code_and_get_efficiency(code_path):
    """Execute a triton .py file and return its efficiency output (GB/s).

    Returns the efficiency value as a string on success, or None if execution fails.
    The script is run with stdout captured; if it fails (non-zero return code),
    None is returned and no modification to CSV should occur.
    """
    try:
        result = subprocess.run(
            [sys.executable, code_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"Triton execution failed with return code {result.returncode}: {result.stderr[:500]}", file=sys.stderr)
            return None
        # Output format: "{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s"
        # We want the GB/s value (3rd field)
        output = result.stdout.strip()
        if not output:
            print(f"Triton execution produced no output. stderr: {result.stderr[:500]}", file=sys.stderr)
            return None
        parts = output.split()
        if len(parts) >= 3:
            # Last part is like "GB/s", e.g., "8.00GB/s"
            try:
                efficiency = float(parts[-1].replace("GB/s", ""))
                return f"{efficiency:.2f}"
            except (ValueError, IndexError):
                print(f"Failed to parse efficiency from output: {output}", file=sys.stderr)
                return None
        print(f"Unexpected output format: {output}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"Triton execution timed out after 120s: {code_path}", file=sys.stderr)
        return None
    except OSError as e:
        print(f"OSError running triton code {code_path}: {e}", file=sys.stderr)
        return None


def write_avg_csv(path, data, name_field):
    """Write {name -> {avg_count, avg_dur_ms}} to a CSV and print confirmation."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[name_field, "avg_count", "avg_dur_ms"])
        writer.writeheader()
        for name, s in data.items():
            writer.writerow({name_field: name, "avg_count": fmt3(s["avg_count"]), "avg_dur_ms": fmt3(s["avg_dur_ms"])})
    print(f"Wrote {path} ({len(data)} rows)")


def _write_kernels_avg_csv(path, avg_kernels):
    """Write all_kernels_avg.csv with family, dur_pct, count_pct, and avg_us_per_call.

    dur_pct / count_pct for compute kernels are relative to the compute total
    (collective excluded).  For collective kernels they are relative to all-kernel total.
    """
    # Pre-compute family for each kernel (avoid calling extract_kernel_family twice)
    families = {name: extract_kernel_family(name) for name in avg_kernels}
    total_dur     = sum(v["avg_dur_ms"] for v in avg_kernels.values()) or 1.0
    total_count   = sum(v["avg_count"]  for v in avg_kernels.values()) or 1.0
    compute_dur   = sum(v["avg_dur_ms"] for name, v in avg_kernels.items()
                        if families[name] != "collective") or 1.0
    compute_count = sum(v["avg_count"]  for name, v in avg_kernels.items()
                        if families[name] != "collective") or 1.0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "kernel_name", "family", "avg_count", "count_pct",
            "avg_dur_ms", "dur_pct", "avg_us_per_call",
        ])
        writer.writeheader()
        for name, s in avg_kernels.items():
            fam = families[name]
            cnt = s["avg_count"]
            dur = s["avg_dur_ms"]
            is_collective = (fam == "collective")
            d_denom = total_dur   if is_collective else compute_dur
            c_denom = total_count if is_collective else compute_count
            writer.writerow({
                "kernel_name":     name,
                "family":          fam,
                "avg_count":       fmt3(cnt),
                "count_pct":       f"{cnt / c_denom * 100:.1f}%",
                "avg_dur_ms":      fmt3(dur),
                "dur_pct":         f"{dur / d_denom * 100:.1f}%",
                "avg_us_per_call": fmt3(dur / cnt * 1000) if cnt > 0 else "",
            })
    print(f"Wrote {path} ({len(avg_kernels)} rows)")


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
            step_to_triton[step].append({
                "kernel_name":         name,
                "dur(ms)":             dur_ms if raw_dur is not None else None,
                "total io(GB)":        float(args["kernel num(GB)"])      if "kernel num(GB)" in args else None,
                "IO efficiency(GB/s)": float(args["IO efficiency(GB/s)"]) if "IO efficiency(GB/s)" in args else None,
                "tiling config":       args.get("kernel kwargs", None),
                "triton_output_code":  args.get("triton output code"),
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

    return {
        "step_to_triton":       step_to_triton,
        "step_to_kernels":      step_to_kernels,
        "step_to_kernel_types": step_to_kernel_types,
        "step_to_aten":         step_to_aten,
        "step_to_cncl":         step_to_cncl,
        "step_durations":       step_durations,
    }


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
    # Support both dict (new API) and tuple (old API) for backward compatibility
    if isinstance(parsed, dict):
        step_to_triton       = parsed["step_to_triton"]
        step_to_kernels      = parsed["step_to_kernels"]
        step_to_kernel_types = parsed["step_to_kernel_types"]
        step_to_aten         = parsed["step_to_aten"]
        step_to_cncl         = parsed["step_to_cncl"]
        step_durations       = parsed["step_durations"]
    else:
        step_to_triton, step_to_kernels, step_to_kernel_types, step_to_aten, step_to_cncl, step_durations = parsed
    all_steps = sorted(set(step_durations) | set(step_to_kernels) | set(step_to_aten) | set(step_to_cncl))
    n_steps   = len(all_steps)
    mean      = lambda vals: sum(vals) / n_steps if n_steps else 0.0

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

    # Auto-classify kernel families from aggregated per-kernel stats
    avg_kernels_data = avg_stats(step_to_kernels, all_steps)
    KERNEL_TYPES, kt_avgs = auto_classify_kernels(avg_kernels_data)

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
        "avg_kernels":    avg_kernels_data,
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
    total_dur     = sum(ad for _, ad in data["kt_avgs"].values()) or 1.0
    total_count   = sum(ac for ac, _ in data["kt_avgs"].values()) or 1.0
    coll_dur      = data["kt_avgs"].get("collective", (0.0, 0.0))[1]
    coll_count    = data["kt_avgs"].get("collective", (0.0, 0.0))[0]
    compute_dur   = (total_dur   - coll_dur)   or 1.0
    compute_count = (total_count - coll_count) or 1.0
    type_w = max(16, max((len(k) for k in data["KERNEL_TYPES"]), default=16))
    hdr = f"{'type':<{type_w}} {'avg_count':<12} {'count_pct':<11} {'avg_dur_ms':<14} {'dur_pct':<10}"
    print(hdr)
    print("-" * len(hdr))
    # KERNEL_TYPES contains only compute families (no collective)
    for ktype in data["KERNEL_TYPES"]:
        ac, ad = data["kt_avgs"][ktype]
        pct_d = f"{ad / compute_dur   * 100:.1f}%"
        pct_c = f"{ac / compute_count * 100:.1f}%"
        print(f"{ktype:<{type_w}} {ac:<12.1f} {pct_c:<11} {ad:<14.3f} {pct_d:<10}")


def print_top_kernels(data, top_n=10, label=""):
    """Print the top-N compute hotspot kernels (collective excluded) with family and duration %."""
    avg_kernels = data["avg_kernels"]
    if not avg_kernels:
        return

    # Build compute-only list (exclude collective kernels)
    compute_kernels = [
        (name, stats, extract_kernel_family(name))
        for name, stats in avg_kernels.items()
        if extract_kernel_family(name) != "collective"
    ]
    if not compute_kernels:
        return

    total_dur = sum(stats["avg_dur_ms"] for _, stats, _ in compute_kernels) or 1.0
    candidates = compute_kernels[:top_n]

    # Dynamic family column width
    fam_w = max(12, max(len(fam) for _, _, fam in candidates))

    title = f"=== Top {top_n} Compute Hotspot Kernels"
    if label:
        title += f" — {label}"
    print(f"\n{title} ===")
    hdr = (f"{'#':<4} {'family':<{fam_w}} {'dur_pct':<9} "
           f"{'avg_dur_ms':<14} {'avg_count':<12} kernel_name")
    print(hdr)
    print("-" * len(hdr))
    for i, (name, stats, family) in enumerate(candidates, 1):
        dur   = stats["avg_dur_ms"]
        cnt   = stats["avg_count"]
        pct_s = f"{dur / total_dur * 100:.1f}%"
        short = name if len(name) <= 55 else name[:52] + "..."
        print(f"{i:<4} {family:<{fam_w}} {pct_s:<9} {dur:<14.3f} {cnt:<12.1f} {short}")


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

    # Kernel type comparison — union of both auto-classified type lists
    all_types = list(dict.fromkeys(
        [t for t in data_a["KERNEL_TYPES"] if t != "other"] +
        [t for t in data_b["KERNEL_TYPES"] if t != "other"]
    ))
    all_types.sort(key=lambda t: -(
        data_a["kt_avgs"].get(t, (0.0, 0.0))[1] + data_b["kt_avgs"].get(t, (0.0, 0.0))[1]
    ))
    all_types.append("other")
    print(f"\n=== Kernel Type Comparison ({label_a} vs {label_b}) ===")
    hdr2 = (f"{'type':<16} {'count_A':<10} {'count_B':<10} {'dur_A(ms)':<12}"
            f" {'dur_B(ms)':<12} {'delta_dur':<12} {'pct':<10}")
    print(hdr2)
    print("-" * len(hdr2))
    for ktype in all_types:
        ac_a, ad_a = data_a["kt_avgs"].get(ktype, (0.0, 0.0))
        ac_b, ad_b = data_b["kt_avgs"].get(ktype, (0.0, 0.0))
        print(f"{ktype:<16} {ac_a:<10.1f} {ac_b:<10.1f} {ad_a:<12.3f}"
              f" {ad_b:<12.3f} {ad_b - ad_a:<+12.3f} {pct(ad_a, ad_b):<10}")


# ── CSV write helpers ─────────────────────────────────────────────────────────

def _write_triton_avg_csv(path, avg_triton):
    total_dur = sum(s["avg_dur_ms"] for s in avg_triton.values()) or 1.0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "kernel_name", "avg_count", "avg_dur_ms", "dur_pct",
            "avg_us_per_call", "avg_io_gb", "avg_io_efficiency",
        ])
        writer.writeheader()
        for name, s in avg_triton.items():
            cnt = s["avg_count"]
            dur = s["avg_dur_ms"]
            writer.writerow({
                "kernel_name":       name,
                "avg_count":         fmt3(cnt),
                "avg_dur_ms":        fmt3(dur),
                "dur_pct":           f"{dur / total_dur * 100:.1f}%",
                "avg_us_per_call":   fmt3(dur / cnt * 1000) if cnt > 0 else "",
                "avg_io_gb":         fmt3(s["avg_io_gb"]),
                "avg_io_efficiency": fmt3(s["avg_io_eff"]),
            })
    print(f"Wrote {path} ({len(avg_triton)} rows)")


def _write_kernel_types_csv(path, kernel_types, kt_avgs):
    total_dur     = sum(v[1] for v in kt_avgs.values()) or 1.0
    total_count   = sum(v[0] for v in kt_avgs.values()) or 1.0
    coll_dur      = kt_avgs.get("collective", (0.0, 0.0))[1]
    coll_count    = kt_avgs.get("collective", (0.0, 0.0))[0]
    compute_dur   = (total_dur   - coll_dur)   or 1.0
    compute_count = (total_count - coll_count) or 1.0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "avg_count", "count_pct", "avg_dur_ms", "dur_pct"])
        writer.writeheader()
        # kernel_types contains only compute families (no collective)
        for ktype in kernel_types:
            ac, ad = kt_avgs[ktype]
            writer.writerow({
                "type":       ktype,
                "avg_count":  fmt3(ac),
                "count_pct":  f"{ac / compute_count * 100:.1f}%",
                "avg_dur_ms": fmt3(ad),
                "dur_pct":    f"{ad / compute_dur   * 100:.1f}%",
            })
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
    # Use union of both type lists (excluding "other"), then append "other" at the end
    all_types = list(dict.fromkeys(
        [t for t in data_a["KERNEL_TYPES"] if t != "other"] +
        [t for t in data_b["KERNEL_TYPES"] if t != "other"]
    ))
    # Sort by max duration across A/B descending, then append "other"
    all_types.sort(key=lambda t: -(
        data_a["kt_avgs"].get(t, (0.0, 0.0))[1] + data_b["kt_avgs"].get(t, (0.0, 0.0))[1]
    ))
    all_types.append("other")
    total_a    = sum(v[1] for v in data_a["kt_avgs"].values()) or 1.0
    total_b    = sum(v[1] for v in data_b["kt_avgs"].values()) or 1.0
    compute_a  = (total_a - data_a["kt_avgs"].get("collective", (0.0, 0.0))[1]) or 1.0
    compute_b  = (total_b - data_b["kt_avgs"].get("collective", (0.0, 0.0))[1]) or 1.0
    # all_types contains only compute families (no collective); use compute totals
    rows = []
    for ktype in all_types:
        ac_a, ad_a = data_a["kt_avgs"].get(ktype, (0.0, 0.0))
        ac_b, ad_b = data_b["kt_avgs"].get(ktype, (0.0, 0.0))
        rows.append({
            "type":         ktype,
            "dur_pct_A":    f"{ad_a / compute_a * 100:.1f}%",
            "avg_dur_ms_A": fmt3(ad_a),
            "dur_pct_B":    f"{ad_b / compute_b * 100:.1f}%",
            "avg_dur_ms_B": fmt3(ad_b),
            "delta_dur_ms": fmt3(ad_b - ad_a),
            "pct_change":   pct(ad_a, ad_b),
            "avg_count_A":  fmt3(ac_a),
            "avg_count_B":  fmt3(ac_b),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "type", "dur_pct_A", "avg_dur_ms_A", "dur_pct_B", "avg_dur_ms_B",
            "delta_dur_ms", "pct_change", "avg_count_A", "avg_count_B",
        ])
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
                code_file_paths = []  # (idx, kernel_name, code_rel_path, code_abs_path)
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=triton_fields)
                    writer.writeheader()
                    for idx, kernel in enumerate(kernels):
                        row = {
                            "kernel_name":         kernel["kernel_name"],
                            "dur(ms)":             fmt3(kernel["dur(ms)"]),
                            "total io(GB)":        fmt3(kernel["total io(GB)"]),
                            "IO efficiency(GB/s)": fmt3(kernel["IO efficiency(GB/s)"]),
                            "tiling config":       (kernel["tiling config"] or "").replace("\n", "\\n").replace("\r", ""),
                            "triton_code_file":    "",
                        }
                        if args.save_triton_code:
                            fname = write_triton_code_file(code_dir, idx, kernel)
                            code_rel_path = os.path.join(f"step_{step}_triton_codes", fname)
                            code_abs_path = os.path.join(code_dir, fname)
                            row["triton_code_file"] = code_rel_path
                            code_file_paths.append((idx, kernel["kernel_name"], code_rel_path, code_abs_path))
                        writer.writerow(row)
                print(f"Wrote {csv_path} ({len(kernels)} rows)")
            elif args.save_triton_code:
                for idx, kernel in enumerate(kernels):
                    write_triton_code_file(code_dir, idx, kernel)
                print(f"Wrote {code_dir}/ ({len(kernels)} files)")

    _write_kernels_avg_csv(os.path.join(args.output_dir, "all_kernels_avg.csv"), data["avg_kernels"])
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
        print_top_kernels(data)
        write_single(data, args)
    else:
        label_a = os.path.basename(args.trace_files[0])
        label_b = os.path.basename(args.trace_files[1])
        data_a = compute_avgs(parse_trace(args.trace_files[0], kernel_types), kernel_types)
        data_b = compute_avgs(parse_trace(args.trace_files[1], kernel_types), kernel_types)
        print_comparison(data_a, data_b, label_a, label_b)
        print_top_kernels(data_a, label=label_a)
        print_top_kernels(data_b, label=label_b)
        write_comparison(data_a, data_b, args)
