import argparse
import bisect
import csv
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
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


def classify_kernel(name, args=None):
    """Classify a GPU kernel into the automatic family taxonomy."""
    args = args or {}
    if args.get("Collective name"):
        return "collective"
    return extract_kernel_family(name)


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

# Pre-compiled regex for stripping leading "void " and C++ namespace prefixes (used in fallback)
_STRIP_LEADING_RE  = re.compile(r'^(void\s+|at::native::|\w+::)+', re.IGNORECASE)

_STEP_NAME_PATTERNS = [
    re.compile(r"^ProfilerStep#\s*(\d+)$", re.IGNORECASE),
    re.compile(r"^ProfilerStep\s*#?\s*(\d+)$", re.IGNORECASE),
    re.compile(r"^step[_\s#:-]*(\d+)$", re.IGNORECASE),
]


def extract_step_number(name: str):
    """Return a numeric step id for common profiler step marker names."""
    text = (name or "").strip()
    for pattern in _STEP_NAME_PATTERNS:
        match = pattern.match(text)
        if match:
            return int(match.group(1))
    return None


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

    # Fallback: strip "void " and C++ namespace prefixes, then take the identifier
    # up to (but not including) the first '<' or '('.  Preserve original case.
    # e.g. "void mlu::PoolingForwardKernel<float, long>" → "PoolingForwardKernel"
    #      "cudaKernel(int, float*)"                     → "cudaKernel"
    clean = _STRIP_LEADING_RE.sub("", name)
    cut = len(clean)
    if '<' in clean:
        cut = min(cut, clean.index('<'))
    if '(' in clean:
        cut = min(cut, clean.index('('))
    clean = clean[:cut].strip()
    if clean:
        return clean
    return "other"


def safe_float(value):
    """Parse a profiler numeric field, returning None for missing or malformed values."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def auto_classify_kernels(avg_kernels: dict, kernel_families: dict | None = None) -> tuple:
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
    kernel_families = kernel_families or {}
    for name, stats in avg_kernels.items():
        fam = kernel_families.get(name) or extract_kernel_family(name)
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
    prefix = f"kernel_{idx}_"
    suffix = ".py"
    code_filename = f"{prefix}{safe_name}{suffix}"
    max_filename_bytes = 240
    if len(code_filename.encode("utf-8")) > max_filename_bytes:
        digest = hashlib.sha1(kernel["kernel_name"].encode("utf-8")).hexdigest()[:10]
        reserved_bytes = len(prefix.encode("utf-8")) + len(suffix.encode("utf-8")) + len(digest) + 1
        name_bytes = safe_name.encode("utf-8")[: max_filename_bytes - reserved_bytes]
        short_name = name_bytes.decode("utf-8", "ignore")
        code_filename = f"{prefix}{short_name}_{digest}{suffix}"
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


def _write_kernels_avg_csv(path, avg_kernels, kernel_families=None):
    """Write all_kernels_avg.csv with family, dur_pct, count_pct, and avg_us_per_call.

    dur_pct / count_pct for compute kernels are relative to the compute total
    (collective excluded).  For collective kernels they are relative to all-kernel total.
    """
    # Pre-compute family for each kernel (avoid calling extract_kernel_family twice)
    kernel_families = kernel_families or {}
    families = {name: kernel_families.get(name) or extract_kernel_family(name) for name in avg_kernels}
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


def _load_trace_json(trace_file):
    """Load trace JSON from plain JSON, gzip JSON, tar.gz, or zip archives."""
    trace_name = str(trace_file).lower()
    if trace_name.endswith(".zip"):
        with zipfile.ZipFile(trace_file) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".json")
            ]
            if not members:
                raise ValueError(f"No JSON trace found in archive: {trace_file}")
            member = max(members, key=lambda info: info.file_size)
            with archive.open(member) as extracted:
                return json.load(extracted)

    if not trace_name.endswith((".gz", ".tgz")):
        with open(trace_file) as f:
            return json.load(f)

    if tarfile.is_tarfile(trace_file):
        with tarfile.open(trace_file, "r:*") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".json"):
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        with extracted:
                            return json.load(extracted)
        raise ValueError(f"No JSON trace found in archive: {trace_file}")

    with gzip.open(trace_file, "rt", encoding="utf-8") as f:
        return json.load(f)


def parse_trace(trace_file):
    """Parse a PyTorch profiler trace JSON.

    Returns:
        step_to_triton:       step -> [{kernel_name, dur(ms), total io(GB), IO efficiency(GB/s),
                                        tiling config, triton_output_code}]
        step_to_kernels:      step -> {kernel_name -> {"count": int, "dur_ms": float}}
        step_to_aten:         step -> {op_name -> {"count": int, "dur_ms": float}}
        step_to_cncl:         step -> {op/kernel_name -> {"count": int, "dur_ms": float}}
        step_durations:       step -> wall-clock duration in ms (from ProfilerStep#/step_N event)
        kernel_families:      kernel_name -> automatic family label
    """
    trace = _load_trace_json(trace_file)

    events = trace["traceEvents"]

    # step_num -> (start_ts, end_ts)
    step_ranges    = {}
    step_durations = {}   # step_num -> ms
    all_kernel_events = []
    aten_events       = []
    cncl_events       = []
    kernel_families   = {}

    def add_step_range(step_num, ts, dur):
        start = ts
        end = ts + dur
        if step_num in step_ranges:
            old_start, old_end = step_ranges[step_num]
            start = min(old_start, start)
            end = max(old_end, end)
        step_ranges[step_num] = (start, end)
        step_durations[step_num] = (end - start) / 1000

    for e in events:
        name = e.get("name", "")
        cat  = e.get("cat", "")
        step_num = extract_step_number(name)
        if step_num is not None and cat != "kernel":
            ts  = e.get("ts", 0)
            dur = e.get("dur", 0)
            add_step_range(step_num, ts, dur)
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
        family = classify_kernel(name, args)
        if name not in kernel_families or family == "collective":
            kernel_families[name] = family
        if family == "collective":
            step_to_cncl[step][name]["count"]  += 1
            step_to_cncl[step][name]["dur_ms"] += dur_ms

        if name.startswith("triton_"):
            step_to_triton[step].append({
                "kernel_name":         name,
                "dur(ms)":             dur_ms if raw_dur is not None else None,
                "total io(GB)":        safe_float(args.get("kernel num(GB)")),
                "IO efficiency(GB/s)": safe_float(args.get("IO efficiency(GB/s)")),
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
        "step_to_aten":         step_to_aten,
        "step_to_cncl":         step_to_cncl,
        "step_durations":       step_durations,
        "step_ranges":          step_ranges,
        "kernel_families":      kernel_families,
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


def compute_avgs(parsed):
    """Compute all average stats from a parsed trace. Returns a data dict."""
    # Support both dict (new API) and tuple (old API) for backward compatibility
    if isinstance(parsed, dict):
        step_to_triton       = parsed["step_to_triton"]
        step_to_kernels      = parsed["step_to_kernels"]
        step_to_aten         = parsed["step_to_aten"]
        step_to_cncl         = parsed["step_to_cncl"]
        step_durations       = parsed["step_durations"]
        step_ranges          = parsed.get("step_ranges", {})
        kernel_families      = parsed.get("kernel_families", {})
    else:
        if len(parsed) == 6:
            step_to_triton, step_to_kernels, _, step_to_aten, step_to_cncl, step_durations = parsed
        else:
            step_to_triton, step_to_kernels, step_to_aten, step_to_cncl, step_durations = parsed
        step_ranges = {}
        kernel_families = {}
    all_steps = sorted(set(step_durations) | set(step_to_kernels) | set(step_to_aten) | set(step_to_cncl))
    n_steps   = len(all_steps)
    mean      = lambda vals: sum(vals) / n_steps if n_steps else 0.0

    step_stats = {}
    for step in all_steps:
        sd  = step_durations.get(step, 0.0)
        kc  = sum(v["count"]  for v in step_to_kernels[step].values())
        collective_names = set(step_to_cncl[step])
        ckd = sum(
            v["dur_ms"]
            for name, v in step_to_kernels[step].items()
            if (
                name not in collective_names
                and (kernel_families.get(name) or extract_kernel_family(name)) != "collective"
            )
        )
        tc  = len(step_to_triton[step])
        td  = sum((k["dur(ms)"] or 0.0) for k in step_to_triton[step])
        ac  = sum(v["count"]  for v in step_to_aten[step].values())
        ad  = sum(v["dur_ms"] for v in step_to_aten[step].values())
        cc  = sum(v["count"]  for v in step_to_cncl[step].values())
        cd  = sum(v["dur_ms"] for v in step_to_cncl[step].values())
        step_stats[step] = (sd, kc, ckd, tc, td, ac, ad, cc, cd)

    avg_row = tuple(mean([step_stats[s][i] for s in all_steps]) for i in range(9))

    # Auto-classify kernel families from aggregated per-kernel stats
    avg_kernels_data = avg_stats(step_to_kernels, all_steps)
    KERNEL_TYPES, kt_avgs = auto_classify_kernels(avg_kernels_data, kernel_families)

    # Triton aggregation: per step by kernel name
    step_triton_agg = defaultdict(lambda: defaultdict(lambda: {
        "count": 0,
        "dur_ms": 0.0,
        "io_gb": 0.0,
        "io_eff_sum": 0.0,
        "io_eff_count": 0,
    }))
    for step, kernels in step_to_triton.items():
        for k in kernels:
            a = step_triton_agg[step][k["kernel_name"]]
            a["count"]  += 1
            a["dur_ms"] += k["dur(ms)"] or 0.0
            if k["total io(GB)"] is not None:
                a["io_gb"]  += k["total io(GB)"]
            if k["IO efficiency(GB/s)"] is not None:
                a["io_eff_sum"] += k["IO efficiency(GB/s)"]
                a["io_eff_count"] += 1

    all_triton_names = set()
    for s in all_steps:
        all_triton_names.update(step_triton_agg[s])

    avg_triton = {}
    for name in all_triton_names:
        io_eff_sum = sum(step_triton_agg[s].get(name, {"io_eff_sum": 0.0})["io_eff_sum"] for s in all_steps)
        io_eff_count = sum(step_triton_agg[s].get(name, {"io_eff_count": 0})["io_eff_count"] for s in all_steps)
        avg_triton[name] = {
            "avg_count":  mean([step_triton_agg[s].get(name, {"count": 0})["count"] for s in all_steps]),
            "avg_dur_ms": mean([step_triton_agg[s].get(name, {"dur_ms": 0.0})["dur_ms"] for s in all_steps]),
            "avg_io_gb":  mean([step_triton_agg[s].get(name, {"io_gb": 0.0})["io_gb"] for s in all_steps]),
            "avg_io_eff": (io_eff_sum / io_eff_count) if io_eff_count else None,
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
        "step_ranges":    step_ranges,
        "kernel_families": kernel_families,
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

    kernel_families = data.get("kernel_families", {})
    # Build compute-only list (exclude collective kernels)
    compute_kernels = [
        (name, stats, kernel_families.get(name) or extract_kernel_family(name))
        for name, stats in avg_kernels.items()
        if (kernel_families.get(name) or extract_kernel_family(name)) != "collective"
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
    hdr = f"{'metric':<26} {la:<18} {lb:<18} {'delta':<14}"
    print(hdr)
    print("-" * len(hdr))
    for i, metric in enumerate(METRICS):
        va, vb = data_a["avg_row"][i], data_b["avg_row"][i]
        print(f"{metric:<26} {va:<18.3f} {vb:<18.3f} {vb - va:<+14.3f}")

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
    type_w = max(16, max((len(t) for t in all_types), default=16))
    hdr2 = (f"{'type':<{type_w}} {'count_A':<10} {'count_B':<10} {'dur_A(ms)':<12}"
            f" {'dur_B(ms)':<12} {'delta_dur':<12} {'pct':<10}")
    print(hdr2)
    print("-" * len(hdr2))
    for ktype in all_types:
        ac_a, ad_a = data_a["kt_avgs"].get(ktype, (0.0, 0.0))
        ac_b, ad_b = data_b["kt_avgs"].get(ktype, (0.0, 0.0))
        print(f"{ktype:<{type_w}} {ac_a:<10.1f} {ac_b:<10.1f} {ad_a:<12.3f}"
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


def _write_kernel_types_csv(path, kernel_type_names, kt_avgs):
    total_dur     = sum(v[1] for v in kt_avgs.values()) or 1.0
    total_count   = sum(v[0] for v in kt_avgs.values()) or 1.0
    coll_dur      = kt_avgs.get("collective", (0.0, 0.0))[1]
    coll_count    = kt_avgs.get("collective", (0.0, 0.0))[0]
    compute_dur   = (total_dur   - coll_dur)   or 1.0
    compute_count = (total_count - coll_count) or 1.0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "avg_count", "count_pct", "avg_dur_ms", "dur_pct"])
        writer.writeheader()
        # kernel_type_names contains only compute families (no collective)
        for ktype in kernel_type_names:
            ac, ad = kt_avgs[ktype]
            writer.writerow({
                "type":       ktype,
                "avg_count":  fmt3(ac),
                "count_pct":  f"{ac / compute_count * 100:.1f}%",
                "avg_dur_ms": fmt3(ad),
                "dur_pct":    f"{ad / compute_dur   * 100:.1f}%",
            })
    print(f"Wrote {path} ({len(kernel_type_names)} rows)")


def _write_cmp_avg_csv(path, data_a, data_b, name_field):
    """Comparison CSV for avg stats (kernels or ops). Sorted by |delta_dur_ms| desc."""
    zero = {"avg_count": 0.0, "avg_dur_ms": 0.0}
    rows = []
    for name in set(data_a) | set(data_b):
        a, b  = data_a.get(name, zero), data_b.get(name, zero)
        delta = b["avg_dur_ms"] - a["avg_dur_ms"]
        delta_cnt = b["avg_count"] - a["avg_count"]
        rows.append({
            name_field:     name,
            "avg_dur_ms_A": fmt3(a["avg_dur_ms"]),
            "avg_dur_ms_B": fmt3(b["avg_dur_ms"]),
            "delta_dur_ms": fmt3(delta),
            "avg_count_A":  fmt3(a["avg_count"]),
            "avg_count_B":  fmt3(b["avg_count"]),
            "delta_count":  fmt3(delta_cnt),
            "_sort":        abs(delta),
        })
    rows.sort(key=lambda r: -r["_sort"])
    fields = [name_field, "avg_dur_ms_A", "avg_dur_ms_B", "delta_dur_ms",
              "avg_count_A", "avg_count_B", "delta_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _write_kernels_cmp_csv(path, data_a, data_b):
    """Write all_kernels_cmp.csv with family for type drill-down."""
    avg_a = data_a["avg_kernels"]
    avg_b = data_b["avg_kernels"]
    families_a = data_a.get("kernel_families", {})
    families_b = data_b.get("kernel_families", {})
    zero = {"avg_count": 0.0, "avg_dur_ms": 0.0}
    rows = []
    for name in set(avg_a) | set(avg_b):
        a, b  = avg_a.get(name, zero), avg_b.get(name, zero)
        delta = b["avg_dur_ms"] - a["avg_dur_ms"]
        delta_cnt = b["avg_count"] - a["avg_count"]
        rows.append({
            "kernel_name":  name,
            "family":       families_b.get(name) or families_a.get(name) or extract_kernel_family(name),
            "avg_dur_ms_A": fmt3(a["avg_dur_ms"]),
            "avg_dur_ms_B": fmt3(b["avg_dur_ms"]),
            "delta_dur_ms": fmt3(delta),
            "avg_count_A":  fmt3(a["avg_count"]),
            "avg_count_B":  fmt3(b["avg_count"]),
            "delta_count":  fmt3(delta_cnt),
            "_sort":        abs(delta),
        })
    rows.sort(key=lambda r: -r["_sort"])
    fields = ["kernel_name", "family", "avg_dur_ms_A", "avg_dur_ms_B", "delta_dur_ms",
              "avg_count_A", "avg_count_B", "delta_count"]
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
            "avg_count_A":  fmt3(a["avg_count"]),
            "avg_count_B":  fmt3(b["avg_count"]),
            "avg_io_gb_A":  fmt3(a["avg_io_gb"]),
            "avg_io_gb_B":  fmt3(b["avg_io_gb"]),
            "_sort":        abs(delta),
        })
    rows.sort(key=lambda r: -r["_sort"])
    fields = ["kernel_name", "avg_dur_ms_A", "avg_dur_ms_B", "delta_dur_ms",
              "avg_count_A", "avg_count_B", "avg_io_gb_A", "avg_io_gb_B"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _write_kernel_types_cmp_csv(path, data_a, data_b):
    rows = _kernel_type_cmp_rows(data_a, data_b, sort_by="combined")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "type", "dur_pct_A", "avg_dur_ms_A", "dur_pct_B", "avg_dur_ms_B",
            "delta_dur_ms", "avg_count_A", "avg_count_B",
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _kernel_type_cmp_rows(data_a, data_b, sort_by="combined"):
    """Build per-kernel-family comparison rows.

    sort_by="combined" keeps the legacy order by A+B duration; sort_by="delta"
    highlights the largest duration changes first.
    """
    all_types = list(dict.fromkeys(
        [t for t in data_a["KERNEL_TYPES"] if t != "other"] +
        [t for t in data_b["KERNEL_TYPES"] if t != "other"]
    ))
    all_types.append("other")
    total_a    = sum(v[1] for v in data_a["kt_avgs"].values()) or 1.0
    total_b    = sum(v[1] for v in data_b["kt_avgs"].values()) or 1.0
    compute_a  = (total_a - data_a["kt_avgs"].get("collective", (0.0, 0.0))[1]) or 1.0
    compute_b  = (total_b - data_b["kt_avgs"].get("collective", (0.0, 0.0))[1]) or 1.0
    rows = []
    for ktype in all_types:
        ac_a, ad_a = data_a["kt_avgs"].get(ktype, (0.0, 0.0))
        ac_b, ad_b = data_b["kt_avgs"].get(ktype, (0.0, 0.0))
        delta_dur = ad_b - ad_a
        delta_count = ac_b - ac_a
        rows.append({
            "type":         ktype,
            "dur_pct_A":    f"{ad_a / compute_a * 100:.1f}%",
            "avg_dur_ms_A": fmt3(ad_a),
            "dur_pct_B":    f"{ad_b / compute_b * 100:.1f}%",
            "avg_dur_ms_B": fmt3(ad_b),
            "delta_dur_ms": fmt3(delta_dur),
            "delta_abs_ms": fmt3(abs(delta_dur)),
            "delta_pct":    pct(ad_a, ad_b),
            "avg_count_A":  fmt3(ac_a),
            "avg_count_B":  fmt3(ac_b),
            "delta_count":  fmt3(delta_count),
            "_combined":    ad_a + ad_b,
            "_delta_abs":   abs(delta_dur),
        })

    if sort_by == "delta":
        rows.sort(key=lambda row: (-row["_delta_abs"], row["type"] == "other", row["type"]))
    else:
        rows.sort(key=lambda row: (row["type"] == "other", -row["_combined"], row["type"]))
    return rows


def _write_kernel_types_delta_csv(path, data_a, data_b):
    rows = _kernel_type_cmp_rows(data_a, data_b, sort_by="delta")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "type", "delta_dur_ms", "delta_abs_ms", "delta_pct",
            "avg_dur_ms_A", "avg_dur_ms_B", "dur_pct_A", "dur_pct_B",
            "avg_count_A", "avg_count_B", "delta_count",
        ], extrasaction="ignore")
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

    _write_kernels_avg_csv(
        os.path.join(args.output_dir, "all_kernels_avg.csv"),
        data["avg_kernels"],
        data.get("kernel_families", {}),
    )
    _write_triton_avg_csv(os.path.join(args.output_dir, "triton_kernels_avg.csv"), data["avg_triton"])
    write_avg_csv(os.path.join(args.output_dir, "aten_ops_avg.csv"),    data["avg_aten"],    "op_name")
    _write_kernel_types_csv(os.path.join(args.output_dir, "kernel_types_avg.csv"), data["KERNEL_TYPES"], data["kt_avgs"])
    write_avg_csv(os.path.join(args.output_dir, "cncl_ops_avg.csv"),    data["avg_cncl"],    "op_name")


def write_comparison(data_a, data_b, args):
    os.makedirs(args.output_dir, exist_ok=True)
    _write_kernels_cmp_csv(os.path.join(args.output_dir, "all_kernels_cmp.csv"), data_a, data_b)
    _write_triton_cmp_csv(os.path.join(args.output_dir, "triton_kernels_cmp.csv"),
                          data_a["avg_triton"], data_b["avg_triton"])
    _write_cmp_avg_csv(os.path.join(args.output_dir, "aten_ops_cmp.csv"),
                       data_a["avg_aten"], data_b["avg_aten"], "op_name")
    _write_kernel_types_cmp_csv(os.path.join(args.output_dir, "kernel_types_cmp.csv"), data_a, data_b)
    _write_kernel_types_delta_csv(os.path.join(args.output_dir, "kernel_types_delta.csv"), data_a, data_b)
    _write_cmp_avg_csv(os.path.join(args.output_dir, "cncl_ops_cmp.csv"),
                       data_a["avg_cncl"], data_b["avg_cncl"], "op_name")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
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
    args = parser.parse_args(argv)
    if len(args.trace_files) > 2:
        parser.error("At most two trace files can be provided.")

    if len(args.trace_files) == 1:
        data = compute_avgs(parse_trace(args.trace_files[0]))
        print_step_summary(data)
        print_kernel_type_breakdown(data)
        print_top_kernels(data)
        write_single(data, args)
    else:
        label_a = os.path.basename(args.trace_files[0])
        label_b = os.path.basename(args.trace_files[1])
        data_a = compute_avgs(parse_trace(args.trace_files[0]))
        data_b = compute_avgs(parse_trace(args.trace_files[1]))
        print_comparison(data_a, data_b, label_a, label_b)
        print_top_kernels(data_a, label=label_a)
        print_top_kernels(data_b, label=label_b)
        write_comparison(data_a, data_b, args)


if __name__ == "__main__":
    main()
