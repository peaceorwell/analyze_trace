import argparse
import bisect
import csv
import gzip
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from collections import defaultdict
from decimal import Decimal

import ijson
import orjson


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
    (["union1bmm", "blockbmm", "stridebatchedgemm"], "bmm"),
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
    re.compile(r"^profile[_\s-]*step[_\s#:-]*(\d+)$", re.IGNORECASE),
    re.compile(r"^step[_\s#:-]*(\d+)$", re.IGNORECASE),
]

_RUN_STEP_RE = re.compile(r"\brun[_\s-]*step\b", re.IGNORECASE)

_SPOOL_BATCH_SIZE = 10000
_SPOOL_MEMORY_BYTES = 128 * 1024 * 1024
_FAST_TRACE_JSON_BYTES_DEFAULT = 256 * 1024 * 1024
_EVENT_KERNEL = 1
_EVENT_ATEN = 2
_EVENT_KERNEL_EFFICIENCY = 4
_KERNEL_EFFICIENCY_COUNTER_NAMES = {
    "Compute Efficiency(%)",
    "IO Efficiency(%)",
    "OP Efficiency(%)",
}
_KERNEL_EFFICIENCY_TS_TOLERANCE_US = 0.25


def extract_step_number(name: str):
    """Return a numeric step id for common profiler step marker names."""
    if not name:
        return None
    if "step" not in name and "Step" not in name:
        return None
    first = name[0]
    if first not in ("P", "p", "s", "S", " "):
        return None
    text = name.strip()
    for pattern in _STEP_NAME_PATTERNS:
        match = pattern.match(text)
        if match:
            return int(match.group(1))
    return None


def is_fallback_step_marker(name: str) -> bool:
    """Return True for common non-numbered model step wrappers."""
    if not name:
        return False
    if "step" not in name and "Step" not in name:
        return False
    return bool(_RUN_STEP_RE.search(name))


def extract_kernel_family(name: str) -> str:
    """Map a GPU kernel name to a semantic family label.

    Priority order:
    1. triton_ prefix  -> triton sub-type (triton_reduce / triton_pointwise / triton_other)
    2. Collective / communication keywords  (checked BEFORE semantic patterns to avoid
       misclassifying e.g. TCDP_RING_ALLREDUCE as "reduce")
    3. Known semantic patterns from _FAMILY_PATTERNS
    4. Fallback: first meaningful token from the cleaned name
    """
    nl = name.lower()

    # Triton kernels — group by sub-type token
    if nl.startswith("triton_"):
        if "poi" in nl:    # pointwise / template-pointwise
            return "triton_pointwise"
        if "tem" in nl:
            if "_tritonfusion_" in nl:
                return "tritonfusion"
            if "_mm_" in nl:
                return "triton_mm"
            if "_bmm_" in nl:
                return "triton_bmm"
        if any(x in nl for x in ("red", "per")):
            return "triton_reduce"
        return "triton_other"

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


def _iter_metric_values(value):
    """Yield (key, value) pairs from nested profiler metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _iter_metric_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_metric_values(child)


def _metric_from_args(args, predicate):
    if not isinstance(args, dict):
        return None
    for key, value in _iter_metric_values(args):
        key_lower = key.lower()
        if predicate(key_lower):
            parsed = safe_float(value)
            if parsed is not None:
                return parsed
    return None


def _metric_key_has_all(key, *tokens):
    return all(token in key for token in tokens)


def _compute_efficiency_from_args(args):
    return _metric_from_args(
        args,
        lambda key: _metric_key_has_all(key, "compute", "efficiency"),
    )


def _op_efficiency_from_args(args):
    return _metric_from_args(
        args,
        lambda key: _metric_key_has_all(key, "op", "efficiency"),
    )


def _io_efficiency_pct_from_args(args):
    return _metric_from_args(
        args,
        lambda key: (
            _metric_key_has_all(key, "io", "efficiency")
            and ("%" in key or "pct" in key or "percent" in key)
        ),
    )


def _is_triton_kernel_event(name, args):
    """Return True for generated or hand-written Triton kernels in Chrome trace."""
    return name.startswith("triton_") or bool(isinstance(args, dict) and args.get("triton output code"))


def _is_empty_category(cat):
    return cat is None or cat == ""


def _tensorflow_group_step(args):
    if not isinstance(args, dict):
        return None
    group_id = args.get("group_id")
    if group_id is None:
        return None
    try:
        step = int(str(group_id).strip())
    except (TypeError, ValueError):
        return None
    return step if step >= 0 else None


def _is_tensorflow_step_event(name, cat, args):
    """TensorFlow traces usually use SessionRun + group_id instead of ProfilerStep."""
    return _is_empty_category(cat) and name == "SessionRun" and _tensorflow_group_step(args) is not None


def _is_tensorflow_device_kernel_event(e, name, cat, args):
    """Return True for TensorFlow MLU device kernel/memcpy timeline events."""
    if not _is_empty_category(cat) or e.get("ph") != "X" or not isinstance(args, dict):
        return False
    if str(e.get("pid", "")) != "1":
        return False
    if name == "cnInvokeKernel":
        return False
    return bool(args.get("tf_op") and args.get("correlation_id"))


def _is_tensorflow_op_event(e, name, cat, args):
    """Return True for host-side TensorFlow op spans with graph node names."""
    if not _is_empty_category(cat) or e.get("ph") != "X" or not isinstance(args, dict):
        return False
    if _is_tensorflow_step_event(name, cat, args) or _is_tensorflow_device_kernel_event(e, name, cat, args):
        return False
    if name in {"ExecutorState::Process", "cnInvokeKernel"}:
        return False
    return bool(args.get("long_name"))


def _tensorflow_op_name(name, args):
    if isinstance(args, dict):
        return str(args.get("long_name") or args.get("tf_op") or name)
    return str(name)


def _external_id_key(value):
    if value is None:
        return ""
    return str(value)


def _compact_detail_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _first_arg_value(args, *keys):
    if not isinstance(args, dict):
        return None
    for key in keys:
        if key in args:
            return args.get(key)
    lowered = {str(key).lower(): value for key, value in args.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None:
            return value
    return None


def _operator_detail_from_event(name, cat, args):
    if cat != "cpu_op" or not isinstance(args, dict):
        return None
    external_id = _external_id_key(args.get("External id"))
    if not external_id:
        return None
    input_dims = _compact_detail_value(_first_arg_value(args, "Input Dims", "Input dims", "input_dims"))
    input_types = _compact_detail_value(_first_arg_value(args, "Input type", "Input Types", "input_types"))
    input_strides = _compact_detail_value(_first_arg_value(args, "Input Strides", "input_strides"))
    concrete_inputs = _compact_detail_value(_first_arg_value(args, "Concrete Inputs", "concrete_inputs"))
    operator_details = _compact_detail_value(_first_arg_value(args, "Operator Details", "operator_details"))
    if not any((input_dims, input_types, input_strides, concrete_inputs, operator_details)):
        return None
    signature = "|".join((name, input_dims, input_types, concrete_inputs, operator_details))
    return {
        "operator": name,
        "input_dims": input_dims,
        "input_types": input_types,
        "input_strides": input_strides,
        "concrete_inputs": concrete_inputs,
        "operator_details": operator_details,
        "operator_signature": signature,
    }


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


_TRITON_NUMERIC_SUFFIX_RE = re.compile(r"(_\d+)+$")
_TRITON_DEF_NAME_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _normalize_triton_kernel_name(name):
    text = str(name or "").strip()
    return _TRITON_NUMERIC_SUFFIX_RE.sub("", text)


def _stable_text_hash(value):
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = re.sub(r"\s+", "", text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _stable_triton_code_signature_hash(value):
    text = str(value or "").strip()
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        stripped = _TRITON_DEF_NAME_RE.sub("def KERNEL(", stripped)
        lines.append(stripped)
    normalized = re.sub(r"\s+", "", "\n".join(lines))
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _stable_triton_tiling_hash(value):
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if parts and all("=" in part for part in parts):
        pairs = []
        for part in parts:
            key, val = part.split("=", 1)
            pairs.append((key.strip(), val.strip()))
        text = ",".join(f"{key}={val}" for key, val in sorted(pairs))
    return _stable_text_hash(text)


def _triton_hash_values(stats, plural_field, singular_field):
    values = stats.get(plural_field)
    if values is None:
        values = []
    if isinstance(values, str):
        values = [values]
    try:
        result = {str(value) for value in values if value}
    except TypeError:
        result = set()
    single = stats.get(singular_field) or ""
    if single:
        result.add(str(single))
    return sorted(result)


def _triton_match_candidates(name, stats):
    normalized_name = stats.get("triton_normalized_name") or _normalize_triton_kernel_name(name)
    normalized_key = normalized_name.lower()
    candidates = [("exact_name", f"exact:{name}")]
    for code_hash in _triton_hash_values(stats, "triton_code_hashes", "triton_code_hash"):
        candidates.append(("code_hash", f"code:{code_hash}"))
    for signature_hash in _triton_hash_values(
            stats, "triton_code_signature_hashes", "triton_code_signature_hash"):
        candidates.append(("code_signature", f"code_sig:{signature_hash}"))
    for tiling_hash in _triton_hash_values(stats, "triton_tiling_hashes", "triton_tiling_hash"):
        if not normalized_key:
            continue
        candidates.append(("normalized_name_tiling", f"name_tiling:{normalized_key}:{tiling_hash}"))
    if normalized_key:
        candidates.append(("normalized_name", f"name:{normalized_key}"))
    return candidates


def _triton_display_name(name_a, name_b, stats_a, stats_b):
    if name_a and name_a == name_b:
        return name_a
    norm_a = stats_a.get("triton_normalized_name") or _normalize_triton_kernel_name(name_a)
    norm_b = stats_b.get("triton_normalized_name") or _normalize_triton_kernel_name(name_b)
    if norm_a and norm_b and norm_a.lower() == norm_b.lower():
        return norm_a
    return name_b or name_a


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
    """Write all_kernels_avg.csv with family and avg_us_per_call."""
    # Pre-compute family for each kernel (avoid calling extract_kernel_family twice)
    kernel_families = kernel_families or {}
    families = {name: kernel_families.get(name) or extract_kernel_family(name) for name in avg_kernels}
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "kernel_name", "family", "avg_count", "avg_dur_ms", "avg_us_per_call",
        ])
        writer.writeheader()
        for name, s in avg_kernels.items():
            fam = families[name]
            cnt = s["avg_count"]
            dur = s["avg_dur_ms"]
            writer.writerow({
                "kernel_name":     name,
                "family":          fam,
                "avg_count":       fmt3(cnt),
                "avg_dur_ms":      fmt3(dur),
                "avg_us_per_call": fmt3(dur / cnt * 1000) if cnt > 0 else "",
            })
    print(f"Wrote {path} ({len(avg_kernels)} rows)")


def _fast_trace_json_limit():
    raw = os.environ.get("TRACE_FAST_TRACE_JSON_BYTES")
    if raw is None:
        return _FAST_TRACE_JSON_BYTES_DEFAULT
    try:
        return max(int(raw), 0)
    except ValueError:
        return _FAST_TRACE_JSON_BYTES_DEFAULT


def _gzip_footer_uncompressed_size(path):
    try:
        with open(path, "rb") as f:
            f.seek(-4, os.SEEK_END)
            return int.from_bytes(f.read(4), "little")
    except OSError:
        return None


def _trace_events_from_payload(payload):
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
        return events if isinstance(events, list) else []
    return []


def _load_trace_events_fast(file_obj):
    return _trace_events_from_payload(orjson.loads(file_obj.read()))


def _load_trace_events_fast_limited(file_obj, limit):
    data = file_obj.read(limit + 1)
    if len(data) > limit:
        return None
    return _trace_events_from_payload(orjson.loads(data))


def _read_trace_events_fast(trace_file):
    """Return traceEvents via a fast whole-file JSON parser, else None.

    Whole-file parsing is much faster than object-by-object streaming for
    normal-size traces.  Keep streaming as the large-file fallback so 10GB+
    traces do not need to be fully materialized in memory.
    """
    limit = _fast_trace_json_limit()
    if limit <= 0:
        return None

    trace_name = str(trace_file).lower()
    file_size = os.path.getsize(trace_file)

    if trace_name.endswith(".zip"):
        with zipfile.ZipFile(trace_file) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".json")
            ]
            if not members:
                raise ValueError(f"No JSON trace found in archive: {trace_file}")
            member = max(members, key=lambda info: info.file_size)
            if member.file_size > limit:
                return None
            with archive.open(member) as extracted:
                return _load_trace_events_fast(extracted)

    if trace_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(trace_file, "r:*") as tar:
            members = [
                member for member in tar.getmembers()
                if member.isfile() and member.name.lower().endswith(".json")
            ]
            if not members:
                raise ValueError(f"No JSON trace found in archive: {trace_file}")
            member = max(members, key=lambda item: item.size)
            if member.size > limit:
                return None
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"Unable to read JSON trace from archive: {trace_file}")
            with extracted:
                return _load_trace_events_fast(extracted)

    if trace_name.endswith(".gz"):
        # The gzip footer stores the uncompressed size modulo 4GB.  It is safe
        # as a rejection guard when it is already above the threshold, but not
        # as proof that a trace is small.  The limit+1 read below is the final
        # safety check before whole-file parsing.
        if file_size > limit:
            return None
        footer_size = _gzip_footer_uncompressed_size(trace_file)
        if footer_size is None or footer_size > limit:
            return None
        with gzip.open(trace_file, "rb") as f:
            return _load_trace_events_fast_limited(f, limit)

    if file_size > limit:
        return None
    with open(trace_file, "rb") as f:
        return _load_trace_events_fast(f)


def _flush_spool_batch(spool, batch):
    if batch:
        pickle.dump(batch, spool, protocol=pickle.HIGHEST_PROTOCOL)
        batch.clear()


def _spool_record(spool, batch, record):
    batch.append(record)
    if len(batch) >= _SPOOL_BATCH_SIZE:
        _flush_spool_batch(spool, batch)


def _iter_spooled_records(spool):
    spool.seek(0)
    while True:
        try:
            batch = pickle.load(spool)
        except EOFError:
            return
        yield from batch


def _iter_trace_events(trace_file):
    """Yield traceEvents one by one from plain JSON, gzip JSON, tar.gz, or zip."""
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
                yield from ijson.items(extracted, "traceEvents.item", use_float=True)
        return

    if not trace_name.endswith((".gz", ".tgz")):
        with open(trace_file, "rb") as f:
            yield from ijson.items(f, "traceEvents.item", use_float=True)
        return

    if trace_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(trace_file, "r:*") as tar:
            members = [
                member for member in tar.getmembers()
                if member.isfile() and member.name.lower().endswith(".json")
            ]
            if not members:
                raise ValueError(f"No JSON trace found in archive: {trace_file}")
            member = max(members, key=lambda item: item.size)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"Unable to read JSON trace from archive: {trace_file}")
            with extracted:
                yield from ijson.items(extracted, "traceEvents.item", use_float=True)
        return

    with gzip.open(trace_file, "rb") as f:
        yield from ijson.items(f, "traceEvents.item", use_float=True)


def _looks_like_tensorflow_trace_name(name):
    base = os.path.basename(str(name or "")).lower()
    return base.startswith("tf_") or ".tf.trace" in base or "tensorflow" in base


def _detect_trace_format(trace_file, sample_limit=100000, source_name=None):
    """Sniff the trace format without mixing framework-specific parse rules."""
    trace_name = os.path.basename(str(trace_file)).lower()
    hint_name = os.path.basename(str(source_name or "")).lower()
    if any(_looks_like_tensorflow_trace_name(name) for name in (trace_name, hint_name)):
        return "tensorflow"

    torch_score = 0
    tf_score = 0
    events = _read_trace_events_fast(trace_file)
    if events is None:
        return "pytorch"
    try:
        for idx, event in enumerate(events):
            if idx >= sample_limit:
                break
            name = event.get("name", "")
            cat = event.get("cat") or ""
            args = event.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}
            if cat == "kernel" or name.startswith("aten::") or extract_step_number(name) is not None:
                torch_score += 1
            if _is_tensorflow_step_event(name, cat, args) or _is_tensorflow_device_kernel_event(event, name, cat, args):
                tf_score += 2
            elif _is_tensorflow_op_event(event, name, cat, args):
                tf_score += 1
            if torch_score >= 3 and tf_score == 0:
                return "pytorch"
            if tf_score >= 3 and torch_score == 0:
                return "tensorflow"
    except Exception:
        return "pytorch"
    return "tensorflow" if tf_score > torch_score else "pytorch"


def parse_trace(trace_file, *, keep_triton_code=False, progress_callback=None, source_name=None):
    """Parse a Chrome trace with framework-specific parsers."""
    if _detect_trace_format(trace_file, source_name=source_name) == "tensorflow":
        return _parse_tensorflow_trace(trace_file, keep_triton_code=keep_triton_code, progress_callback=progress_callback)
    return _parse_pytorch_trace(trace_file, keep_triton_code=keep_triton_code, progress_callback=progress_callback)


def _parse_tensorflow_trace(trace_file, *, keep_triton_code=False, progress_callback=None):
    """Parse TensorFlow Chrome traces without reusing PyTorch event rules."""
    step_ranges = {}
    step_durations = {}
    step_cache_dirty = True
    cached_step_starts = []
    cached_step_ends = []
    cached_step_nums = []
    analyzable_intervals = []

    step_to_triton = defaultdict(list)
    step_to_kernels = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))
    step_to_aten = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))
    step_to_tf_ops = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))
    step_to_non_triton_kernel_events = defaultdict(list)
    step_to_kernel_efficiency_counters = defaultdict(list)
    kernel_families = {}
    family_cache = {}
    parse_events_seen = 0
    parse_records_seen = 0
    last_progress_at = time.perf_counter()
    last_progress_snapshot = None

    def maybe_report_parse_progress(*, force=False):
        nonlocal last_progress_at, last_progress_snapshot
        if not progress_callback:
            return
        now = time.perf_counter()
        if not force and now - last_progress_at < 5:
            return
        snapshot = (parse_events_seen, parse_records_seen)
        if force and snapshot == last_progress_snapshot:
            return
        progress_callback(parse_events_seen, parse_records_seen)
        last_progress_snapshot = snapshot
        last_progress_at = now

    def event_interval(e):
        raw_ts = e.get("ts")
        if raw_ts is None:
            return None
        ts = float(raw_ts) if isinstance(raw_ts, (int, float)) else safe_float(raw_ts)
        if ts is None:
            return None
        raw_dur = e.get("dur")
        if raw_dur is None:
            dur = 0.0
        elif isinstance(raw_dur, (int, float)):
            dur = float(raw_dur)
        else:
            dur = safe_float(raw_dur) or 0.0
        return ts, ts + max(dur, 0.0)

    def add_step_range(step_num, ts, dur):
        nonlocal step_cache_dirty
        start = ts
        end = ts + dur
        if step_num in step_ranges:
            old_start, old_end = step_ranges[step_num]
            start = min(old_start, start)
            end = max(old_end, end)
        step_ranges[step_num] = (start, end)
        step_durations[step_num] = (end - start) / 1000
        step_cache_dirty = True

    def find_known_step(ts):
        nonlocal step_cache_dirty, cached_step_starts, cached_step_ends, cached_step_nums
        if step_cache_dirty:
            sorted_steps = sorted(step_ranges.items(), key=lambda x: x[1][0])
            cached_step_starts = [v[0] for _, v in sorted_steps]
            cached_step_ends = [v[1] for _, v in sorted_steps]
            cached_step_nums = [k for k, _ in sorted_steps]
            step_cache_dirty = False
        i = bisect.bisect_right(cached_step_starts, ts) - 1
        if i >= 0 and ts <= cached_step_ends[i]:
            return cached_step_nums[i]
        return None

    def is_tf_record_event(e, name, cat, args):
        return (
            _is_tensorflow_device_kernel_event(e, name, cat, args)
            or _is_tensorflow_op_event(e, name, cat, args)
            or name in _KERNEL_EFFICIENCY_COUNTER_NAMES
        )

    def compact_record(e, name, cat, interval):
        args = e.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}
        ts = interval[0]
        dur_ms = max(interval[1] - interval[0], 0.0) / 1000 if e.get("dur") is not None else 0.0
        if _is_tensorflow_device_kernel_event(e, name, cat, args):
            triton_output_code = args.get("triton output code")
            triton_code_hash = _stable_text_hash(triton_output_code) if triton_output_code else ""
            triton_code_signature_hash = (
                _stable_triton_code_signature_hash(triton_output_code) if triton_output_code else ""
            )
            return (
                "kernel",
                name,
                ts,
                dur_ms,
                e.get("dur") is not None,
                safe_float(args.get("kernel num(GB)")),
                safe_float(args.get("IO efficiency(GB/s)")),
                args.get("kernel kwargs"),
                triton_code_hash,
                triton_code_signature_hash,
                triton_output_code if keep_triton_code else None,
                _compute_efficiency_from_args(args),
                _io_efficiency_pct_from_args(args),
                _op_efficiency_from_args(args),
                e.get("tid"),
                _compact_detail_value(args.get("tf_op")),
                _is_triton_kernel_event(name, args),
            )
        if _is_tensorflow_op_event(e, name, cat, args):
            return ("tf_op", _tensorflow_op_name(name, args), ts, dur_ms)
        value = safe_float(args.get("utils")) if isinstance(args, dict) else None
        return ("efficiency", name, ts, value, e.get("tid"))

    def aggregate_record(record, step):
        kind = record[0]
        if kind == "kernel":
            (
                _,
                name,
                start_ts,
                dur_ms,
                has_dur,
                total_io_gb,
                io_efficiency,
                tiling_config,
                triton_code_hash,
                triton_code_signature_hash,
                triton_output_code,
                compute_efficiency,
                io_efficiency_pct,
                op_efficiency,
                tid,
                operator,
                is_triton_kernel,
            ) = record
            step_to_kernels[step][name]["count"] += 1
            step_to_kernels[step][name]["dur_ms"] += dur_ms

            family = family_cache.get(name)
            if family is None:
                family = classify_kernel(name)
                family_cache[name] = family
            if is_triton_kernel and family != "collective" and not family.startswith("triton"):
                family = "triton_custom"
            if name not in kernel_families or family == "collective":
                kernel_families[name] = family

            if is_triton_kernel:
                step_to_triton[step].append({
                    "kernel_name": name,
                    "dur(ms)": dur_ms if has_dur else None,
                    "total io(GB)": total_io_gb,
                    "IO efficiency(GB/s)": io_efficiency,
                    "tiling config": tiling_config,
                    "triton_code_hash": triton_code_hash,
                    "triton_code_signature_hash": triton_code_signature_hash,
                    "triton_output_code": triton_output_code,
                })
            else:
                step_to_non_triton_kernel_events[step].append({
                    "kernel_name": name,
                    "family": family,
                    "start_ts": start_ts,
                    "end_ts": start_ts + dur_ms * 1000 if has_dur else start_ts,
                    "dur(ms)": dur_ms if has_dur else None,
                    "compute_efficiency": compute_efficiency,
                    "io_efficiency": io_efficiency_pct,
                    "op_efficiency": op_efficiency,
                    "tid": tid,
                    "operator": operator,
                })
            return

        if kind == "tf_op":
            _, name, _ts, dur_ms = record
            step_to_tf_ops[step][name]["count"] += 1
            step_to_tf_ops[step][name]["dur_ms"] += dur_ms
            return

        _, counter_name, ts, value, tid = record
        if value is not None:
            step_to_kernel_efficiency_counters[step].append({
                "name": counter_name,
                "ts": ts,
                "value": value,
                "tid": tid,
            })

    def scan_event(e, record_sink=None, *, allow_direct=False):
        name = e.get("name", "")
        cat = e.get("cat") or ""
        args = e.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}
        if _is_tensorflow_step_event(name, cat, args):
            interval = event_interval(e)
            if interval is not None:
                add_step_range(_tensorflow_group_step(args), interval[0], interval[1] - interval[0])
            return False
        if not is_tf_record_event(e, name, cat, args):
            return False
        interval = event_interval(e)
        if interval is None:
            return False
        if name not in _KERNEL_EFFICIENCY_COUNTER_NAMES:
            analyzable_intervals.append(interval)
        record = compact_record(e, name, cat, interval)
        if allow_direct and step_ranges:
            step = find_known_step(interval[0])
            if step is not None:
                aggregate_record(record, step)
                return True
        if record_sink is not None:
            record_sink(record)
        return True

    def scan_events_with_progress(events, record_sink=None, *, allow_direct=False):
        nonlocal parse_events_seen, parse_records_seen
        for event in events:
            parse_events_seen += 1
            if scan_event(event, record_sink, allow_direct=allow_direct):
                parse_records_seen += 1
            if parse_events_seen % 50000 == 0:
                maybe_report_parse_progress()

    def fast_records(events):
        for event in events:
            name = event.get("name", "")
            cat = event.get("cat") or ""
            args = event.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}
            interval = event_interval(event)
            if interval is not None and is_tf_record_event(event, name, cat, args):
                yield compact_record(event, name, cat, interval)

    fast_events = _read_trace_events_fast(trace_file)
    spool = None
    spool_batch = []
    try:
        if fast_events is not None:
            scan_events_with_progress(fast_events)
        else:
            spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES)

            def record_to_spool(record):
                _spool_record(spool, spool_batch, record)

            scan_events_with_progress(_iter_trace_events(trace_file), record_to_spool, allow_direct=True)
            _flush_spool_batch(spool, spool_batch)
        maybe_report_parse_progress(force=True)

        if not step_ranges and analyzable_intervals:
            start = min(interval[0] for interval in analyzable_intervals)
            end = max(interval[1] for interval in analyzable_intervals)
            add_step_range(0, start, max(end - start, 0.0))

        sorted_steps = sorted(step_ranges.items(), key=lambda x: x[1][0])
        step_starts = [v[0] for _, v in sorted_steps]
        step_ends = [v[1] for _, v in sorted_steps]
        step_nums = [k for k, _ in sorted_steps]

        def find_step(ts):
            i = bisect.bisect_right(step_starts, ts) - 1
            if i >= 0 and ts <= step_ends[i]:
                return step_nums[i]
            return None

        record_iter = fast_records(fast_events) if fast_events is not None else _iter_spooled_records(spool)
        for record in record_iter:
            ts = record[2]
            step = find_step(ts)
            if step is not None:
                aggregate_record(record, step)

        def build_non_triton_kernel_efficiency():
            counter_field = {
                "Compute Efficiency(%)": "compute_efficiency",
                "IO Efficiency(%)": "io_efficiency",
                "OP Efficiency(%)": "op_efficiency",
            }
            result = defaultdict(list)
            for step, kernels in step_to_non_triton_kernel_events.items():
                counters_by_tid = defaultdict(list)
                for counter in step_to_kernel_efficiency_counters.get(step, []):
                    counters_by_tid[counter.get("tid")].append(counter)
                for counters in counters_by_tid.values():
                    counters.sort(key=lambda item: item["ts"])
                ts_cache = {
                    tid: [counter["ts"] for counter in counters]
                    for tid, counters in counters_by_tid.items()
                }
                for kernel in kernels:
                    metrics = {
                        "compute_efficiency": kernel.get("compute_efficiency"),
                        "io_efficiency": kernel.get("io_efficiency"),
                        "op_efficiency": kernel.get("op_efficiency"),
                    }
                    counters = counters_by_tid.get(kernel.get("tid"), [])
                    ts_values = ts_cache.get(kernel.get("tid"), [])
                    if counters and ts_values:
                        start = kernel["start_ts"]
                        end = max(kernel.get("end_ts") or start, start)
                        left = bisect.bisect_left(ts_values, start - _KERNEL_EFFICIENCY_TS_TOLERANCE_US)
                        right = bisect.bisect_right(ts_values, start + _KERNEL_EFFICIENCY_TS_TOLERANCE_US)
                        candidates = counters[left:right]
                        if not candidates:
                            left = bisect.bisect_left(ts_values, start)
                            right = bisect.bisect_right(ts_values, end)
                            candidates = counters[left:right]
                        values = defaultdict(list)
                        for counter in candidates:
                            field = counter_field.get(counter["name"])
                            if field:
                                values[field].append(counter["value"])
                        for field, field_values in values.items():
                            if metrics.get(field) is None and field_values:
                                metrics[field] = max(field_values)
                    if not any(value not in (None, 0, 0.0) for value in metrics.values()):
                        continue
                    operator = kernel.get("operator", "")
                    result[step].append({
                        "kernel_name": kernel["kernel_name"],
                        "family": kernel["family"],
                        "dur(ms)": kernel.get("dur(ms)"),
                        "operator": operator,
                        "input_dims": "",
                        "input_types": "",
                        "input_strides": "",
                        "concrete_inputs": "",
                        "operator_details": operator,
                        "operator_signature": operator,
                        **metrics,
                    })
            return result

        return {
            "framework": "tensorflow",
            "step_to_triton": step_to_triton,
            "step_to_kernels": step_to_kernels,
            "step_to_aten": step_to_aten,
            "step_to_tf_ops": step_to_tf_ops,
            "step_to_non_triton_kernel_efficiency": build_non_triton_kernel_efficiency(),
            "step_durations": step_durations,
            "step_ranges": step_ranges,
            "kernel_families": kernel_families,
        }
    finally:
        if spool is not None:
            spool.close()


def _parse_pytorch_trace(trace_file, *, keep_triton_code=False, progress_callback=None):
    """Parse a PyTorch profiler trace JSON.

    Returns:
        step_to_triton:       step -> [{kernel_name, dur(ms), total io(GB), IO efficiency(GB/s),
                                        tiling config, triton hashes, optional triton_output_code}]
        step_to_kernels:      step -> {kernel_name -> {"count": int, "dur_ms": float}}
        step_to_aten:         step -> {op_name -> {"count": int, "dur_ms": float}}
        step_durations:       step -> wall-clock duration in ms (from ProfilerStep#/step_N event)
        kernel_families:      kernel_name -> automatic family label

    Traces that do not contain ProfilerStep#/step_N markers fall back to
    non-numbered run_step wrappers, then to one synthetic step spanning all
    analyzable events.  This keeps valid traces from producing empty results.
    """
    # step_num -> (start_ts, end_ts)
    step_ranges    = {}
    step_durations = {}   # step_num -> ms
    kernel_families   = {}
    analyzable_intervals = []
    fallback_intervals = []
    step_cache_dirty = True
    cached_step_starts = []
    cached_step_ends = []
    cached_step_nums = []

    step_to_triton       = defaultdict(list)
    step_to_kernels      = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))
    step_to_aten         = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))
    step_to_non_triton_kernel_events = defaultdict(list)
    step_to_kernel_efficiency_counters = defaultdict(list)
    operator_details_by_external_id = {}
    family_cache = {}
    parse_events_seen = 0
    parse_records_seen = 0
    last_progress_at = time.perf_counter()
    last_progress_snapshot = None

    def maybe_report_parse_progress(*, force=False):
        nonlocal last_progress_at, last_progress_snapshot
        if not progress_callback:
            return
        now = time.perf_counter()
        if not force and now - last_progress_at < 5:
            return
        snapshot = (parse_events_seen, parse_records_seen)
        if force and snapshot == last_progress_snapshot:
            return
        progress_callback(parse_events_seen, parse_records_seen)
        last_progress_snapshot = snapshot
        last_progress_at = now

    def event_interval(e):
        raw_ts = e.get("ts")
        if raw_ts is None:
            return None
        ts = float(raw_ts) if isinstance(raw_ts, (int, float)) else safe_float(raw_ts)
        if ts is None:
            return None
        raw_dur = e.get("dur")
        if raw_dur is None:
            dur = 0.0
        elif isinstance(raw_dur, (int, float)):
            dur = float(raw_dur)
        else:
            dur = safe_float(raw_dur) or 0.0
        return ts, ts + max(dur, 0.0)

    def add_step_range(step_num, ts, dur):
        nonlocal step_cache_dirty
        start = ts
        end = ts + dur
        if step_num in step_ranges:
            old_start, old_end = step_ranges[step_num]
            start = min(old_start, start)
            end = max(old_end, end)
        step_ranges[step_num] = (start, end)
        step_durations[step_num] = (end - start) / 1000
        step_cache_dirty = True

    def find_known_step(ts):
        nonlocal step_cache_dirty, cached_step_starts, cached_step_ends, cached_step_nums
        if step_cache_dirty:
            sorted_steps = sorted(step_ranges.items(), key=lambda x: x[1][0])
            cached_step_starts = [v[0] for _, v in sorted_steps]
            cached_step_ends = [v[1] for _, v in sorted_steps]
            cached_step_nums = [k for k, _ in sorted_steps]
            step_cache_dirty = False
        i = bisect.bisect_right(cached_step_starts, ts) - 1
        if i >= 0 and ts <= cached_step_ends[i]:
            return cached_step_nums[i]
        return None

    def is_interesting_event(e, name, cat, args):
        return (
            cat == "kernel"
            or name.startswith("aten::")
            or name in _KERNEL_EFFICIENCY_COUNTER_NAMES
        )

    def compact_record(e, name, cat, interval):
        args = e.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}
        raw_dur = e.get("dur")
        has_dur = raw_dur is not None
        dur_ms = max(interval[1] - interval[0], 0.0) / 1000 if has_dur else 0.0
        ts = interval[0]
        if cat == "kernel":
            triton_output_code = args.get("triton output code")
            triton_code_hash = _stable_text_hash(triton_output_code) if triton_output_code else ""
            triton_code_signature_hash = (
                _stable_triton_code_signature_hash(triton_output_code) if triton_output_code else ""
            )
            is_triton_kernel = _is_triton_kernel_event(name, args)
            return (
                _EVENT_KERNEL,
                name,
                ts,
                dur_ms,
                has_dur,
                args.get("Collective name"),
                safe_float(args.get("kernel num(GB)")),
                safe_float(args.get("IO efficiency(GB/s)")),
                args.get("kernel kwargs"),
                triton_code_hash,
                triton_code_signature_hash,
                triton_output_code if keep_triton_code else None,
                _compute_efficiency_from_args(args),
                _io_efficiency_pct_from_args(args),
                _op_efficiency_from_args(args),
                e.get("tid"),
                _external_id_key(args.get("External id")),
                is_triton_kernel,
            )
        if name.startswith("aten::"):
            return (_EVENT_ATEN, name, ts, dur_ms)
        value = safe_float(args.get("utils")) if isinstance(args, dict) else None
        return (_EVENT_KERNEL_EFFICIENCY, name, ts, value, e.get("tid"))

    def aggregate_record(record, step):
        kind = record[0]
        if kind == _EVENT_KERNEL:
            (
                _,
                name,
                _ts,
                dur_ms,
                has_dur,
                collective_name,
                total_io_gb,
                io_efficiency,
                tiling_config,
                triton_code_hash,
                triton_code_signature_hash,
                triton_output_code,
                compute_efficiency,
                io_efficiency_pct,
                op_efficiency,
                tid,
                external_id,
                is_triton_kernel,
            ) = record
            step_to_kernels[step][name]["count"] += 1
            step_to_kernels[step][name]["dur_ms"] += dur_ms

            family_key = (name, collective_name or "")
            family = family_cache.get(family_key)
            if family is None:
                args = {"Collective name": collective_name} if collective_name else {}
                family = classify_kernel(name, args)
                family_cache[family_key] = family
            if is_triton_kernel and family != "collective" and not family.startswith("triton"):
                family = "triton_custom"
            if name not in kernel_families or family == "collective":
                kernel_families[name] = family

            if is_triton_kernel:
                step_to_triton[step].append({
                    "kernel_name":         name,
                    "dur(ms)":             dur_ms if has_dur else None,
                    "total io(GB)":        total_io_gb,
                    "IO efficiency(GB/s)": io_efficiency,
                    "tiling config":       tiling_config,
                    "triton_code_hash":    triton_code_hash,
                    "triton_code_signature_hash": triton_code_signature_hash,
                    "triton_output_code":  triton_output_code,
                })
            else:
                step_to_non_triton_kernel_events[step].append({
                    "kernel_name":              name,
                    "family":                   family,
                    "start_ts":                 _ts,
                    "end_ts":                   _ts + dur_ms * 1000 if has_dur else _ts,
                    "dur(ms)":                  dur_ms if has_dur else None,
                    "compute_efficiency":       compute_efficiency,
                    "io_efficiency":            io_efficiency_pct,
                    "op_efficiency":            op_efficiency,
                    "tid":                      tid,
                    "external_id":              external_id,
                })
            return

        if kind == _EVENT_KERNEL_EFFICIENCY:
            _, counter_name, ts, value, tid = record
            if value is not None:
                step_to_kernel_efficiency_counters[step].append({
                    "name": counter_name,
                    "ts": ts,
                    "value": value,
                    "tid": tid,
                })
            return

        _, name, _ts, dur_ms = record
        step_to_aten[step][name]["count"] += 1
        step_to_aten[step][name]["dur_ms"] += dur_ms

    def scan_event(e, record_sink=None, *, allow_direct=False):
        name = e.get("name", "")
        cat  = e.get("cat") or ""
        args = e.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}
        detail = _operator_detail_from_event(name, cat, args)
        if detail:
            operator_details_by_external_id.setdefault(_external_id_key(args.get("External id")), detail)
        step_num = extract_step_number(name)
        if step_num is not None and cat != "kernel":
            interval = event_interval(e)
            if interval is not None:
                add_step_range(step_num, interval[0], interval[1] - interval[0])
            return
        if cat != "kernel" and is_fallback_step_marker(name):
            interval = event_interval(e)
            if interval and interval[1] > interval[0] and not step_ranges:
                fallback_intervals.append(interval)
            return
        if not is_interesting_event(e, name, cat, args):
            return

        interval = event_interval(e)
        if interval is not None:
            if not step_ranges:
                analyzable_intervals.append(interval)
            record = compact_record(e, name, cat, interval)
            if allow_direct and step_ranges:
                step = find_known_step(interval[0])
                if step is not None:
                    aggregate_record(record, step)
                    return True
            if record_sink is not None:
                record_sink(record)
            return True
        return False

    def scan_events_with_progress(events, record_sink=None, *, allow_direct=False):
        nonlocal parse_events_seen, parse_records_seen
        for event in events:
            parse_events_seen += 1
            if scan_event(event, record_sink, allow_direct=allow_direct):
                parse_records_seen += 1
            if parse_events_seen % 50000 == 0:
                maybe_report_parse_progress()

    def fast_records(events):
        for e in events:
            name = e.get("name", "")
            cat = e.get("cat") or ""
            args = e.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}
            interval = event_interval(e)
            if is_interesting_event(e, name, cat, args) and interval is not None:
                yield compact_record(e, name, cat, interval)

    fast_events = _read_trace_events_fast(trace_file)
    spool = None
    spool_batch = []
    try:
        if fast_events is not None:
            scan_events_with_progress(fast_events)
        else:
            spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES)

            def record_to_spool(record):
                _spool_record(spool, spool_batch, record)

            scan_events_with_progress(_iter_trace_events(trace_file), record_to_spool, allow_direct=True)
            _flush_spool_batch(spool, spool_batch)
        maybe_report_parse_progress(force=True)

        def add_fallback_step_ranges():
            fallback_intervals.sort(key=lambda item: item[0])
            analyzable_intervals.sort(key=lambda item: item[0])
            analyzable_starts = [item[0] for item in analyzable_intervals]

            if fallback_intervals:
                for index, (start, end) in enumerate(fallback_intervals):
                    next_start = fallback_intervals[index + 1][0] if index + 1 < len(fallback_intervals) else None
                    left = bisect.bisect_left(analyzable_starts, start)
                    right = (
                        bisect.bisect_left(analyzable_starts, next_start)
                        if next_start is not None else len(analyzable_intervals)
                    )
                    if right > left:
                        end = max(end, max(event_end for _, event_end in analyzable_intervals[left:right]))
                    if next_start is not None:
                        end = min(end, next_start)
                    add_step_range(index, start, max(end - start, 0.0))
                return

            if analyzable_intervals:
                start = analyzable_intervals[0][0]
                end = max(interval[1] for interval in analyzable_intervals)
                add_step_range(0, start, max(end - start, 0.0))

        if not step_ranges:
            add_fallback_step_ranges()

        sorted_steps = sorted(step_ranges.items(), key=lambda x: x[1][0])
        step_starts  = [v[0] for _, v in sorted_steps]
        step_ends    = [v[1] for _, v in sorted_steps]
        step_nums    = [k    for k, _ in sorted_steps]

        def find_step(ts):
            i = bisect.bisect_right(step_starts, ts) - 1
            if i >= 0 and ts <= step_ends[i]:
                return step_nums[i]
            return None

        record_iter = fast_records(fast_events) if fast_events is not None else _iter_spooled_records(spool)
        for record in record_iter:
            ts = record[2]
            step = find_step(ts)
            if step is None:
                continue
            aggregate_record(record, step)

        def build_non_triton_kernel_efficiency():
            counter_field = {
                "Compute Efficiency(%)": "compute_efficiency",
                "IO Efficiency(%)": "io_efficiency",
                "OP Efficiency(%)": "op_efficiency",
            }
            result = defaultdict(list)
            for step, kernels in step_to_non_triton_kernel_events.items():
                counters_by_tid = defaultdict(list)
                for counter in step_to_kernel_efficiency_counters.get(step, []):
                    counters_by_tid[counter.get("tid")].append(counter)
                for counters in counters_by_tid.values():
                    counters.sort(key=lambda item: item["ts"])

                ts_cache = {
                    tid: [counter["ts"] for counter in counters]
                    for tid, counters in counters_by_tid.items()
                }

                for kernel in kernels:
                    metrics = {
                        "compute_efficiency": kernel.get("compute_efficiency"),
                        "io_efficiency": kernel.get("io_efficiency"),
                        "op_efficiency": kernel.get("op_efficiency"),
                    }
                    counters = counters_by_tid.get(kernel.get("tid"), [])
                    ts_values = ts_cache.get(kernel.get("tid"), [])
                    if counters and ts_values:
                        start = kernel["start_ts"]
                        end = max(kernel.get("end_ts") or start, start)
                        left = bisect.bisect_left(ts_values, start - _KERNEL_EFFICIENCY_TS_TOLERANCE_US)
                        right = bisect.bisect_right(ts_values, start + _KERNEL_EFFICIENCY_TS_TOLERANCE_US)
                        candidates = counters[left:right]
                        if not candidates:
                            left = bisect.bisect_left(ts_values, start)
                            right = bisect.bisect_right(ts_values, end)
                            candidates = counters[left:right]

                        values = defaultdict(list)
                        for counter in candidates:
                            field = counter_field.get(counter["name"])
                            if field:
                                values[field].append(counter["value"])
                        for field, field_values in values.items():
                            if metrics.get(field) is None and field_values:
                                metrics[field] = max(field_values)

                    if not any(value not in (None, 0, 0.0) for value in metrics.values()):
                        continue
                    operator_detail = operator_details_by_external_id.get(kernel.get("external_id"), {})
                    result[step].append({
                        "kernel_name": kernel["kernel_name"],
                        "family": kernel["family"],
                        "dur(ms)": kernel.get("dur(ms)"),
                        "operator": operator_detail.get("operator", "") or kernel.get("operator", ""),
                        "input_dims": operator_detail.get("input_dims", ""),
                        "input_types": operator_detail.get("input_types", ""),
                        "input_strides": operator_detail.get("input_strides", ""),
                        "concrete_inputs": operator_detail.get("concrete_inputs", ""),
                        "operator_details": operator_detail.get("operator_details", ""),
                        "operator_signature": operator_detail.get("operator_signature", ""),
                        **metrics,
                    })
            return result

        return {
            "framework": "pytorch",
            "step_to_triton":       step_to_triton,
            "step_to_kernels":      step_to_kernels,
            "step_to_aten":         step_to_aten,
            "step_to_non_triton_kernel_efficiency": build_non_triton_kernel_efficiency(),
            "step_durations":       step_durations,
            "step_ranges":          step_ranges,
            "kernel_families":      kernel_families,
        }
    finally:
        if spool is not None:
            spool.close()


def parse_step_filter(value):
    """Parse a user-facing step selector.

    Supported forms:
      - "3"
      - "0,2,5"
      - "1-4"
      - "0, 2-4, 8"
    """
    if value is None:
        return tuple()

    if isinstance(value, (list, tuple, set)):
        steps = []
        for item in value:
            try:
                step = int(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid step value: {item!r}") from exc
            if step < 0:
                raise ValueError(f"Step must be non-negative: {step}")
            if step not in steps:
                steps.append(step)
        return tuple(steps)

    text = str(value).strip()
    if not text:
        return tuple()

    normalized = text.translate(str.maketrans({
        "，": ",",
        "；": ",",
        ";": ",",
        "、": ",",
    }))
    steps = []
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue

        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if range_match:
            start, end = map(int, range_match.groups())
            if end < start:
                raise ValueError(f"Invalid step range: {token}")
            for step in range(start, end + 1):
                if step not in steps:
                    steps.append(step)
            continue

        single_match = re.fullmatch(r"(?:(?:ProfilerStep#?|profile_step_|step_?))?(\d+)", token, re.IGNORECASE)
        if single_match:
            step = int(single_match.group(1))
            if step not in steps:
                steps.append(step)
            continue

        raise ValueError(f"Invalid step selector: {token}")

    return tuple(steps)


def _same_step_mapping(mapping):
    if isinstance(mapping, defaultdict):
        return defaultdict(mapping.default_factory)
    return {}


def filter_parsed_steps(parsed, steps, *, require_match=True):
    """Return a parsed trace limited to the selected step numbers."""
    selected_steps = parse_step_filter(steps)
    if not selected_steps:
        return parsed
    if not isinstance(parsed, dict):
        raise TypeError("filter_parsed_steps expects the dict returned by parse_trace")

    available_steps = set(parsed.get("step_durations", {}))
    for key in ("step_to_triton", "step_to_kernels", "step_to_aten", "step_to_tf_ops"):
        available_steps.update(parsed.get(key, {}))

    missing = [step for step in selected_steps if step not in available_steps]
    if require_match and missing:
        available = ", ".join(str(step) for step in sorted(available_steps)) or "none"
        missing_text = ", ".join(str(step) for step in missing)
        raise ValueError(f"Selected step(s) not found: {missing_text}. Available steps: {available}")

    selected_set = set(selected_steps)

    def filter_mapping(key):
        src = parsed.get(key, {})
        dst = _same_step_mapping(src)
        for step in selected_steps:
            if step in src:
                dst[step] = src[step]
        return dst

    filtered = dict(parsed)
    filtered["step_durations"] = {
        step: parsed.get("step_durations", {})[step]
        for step in selected_steps
        if step in parsed.get("step_durations", {})
    }
    filtered["step_ranges"] = {
        step: parsed.get("step_ranges", {})[step]
        for step in selected_steps
        if step in parsed.get("step_ranges", {})
    }
    filtered["step_to_triton"] = filter_mapping("step_to_triton")
    filtered["step_to_kernels"] = filter_mapping("step_to_kernels")
    filtered["step_to_aten"] = filter_mapping("step_to_aten")
    filtered["step_to_tf_ops"] = filter_mapping("step_to_tf_ops")
    filtered["step_to_non_triton_kernel_efficiency"] = filter_mapping("step_to_non_triton_kernel_efficiency")

    kernel_names = set()
    for step in selected_set:
        kernel_names.update(filtered["step_to_kernels"][step])
    filtered["kernel_families"] = {
        name: family
        for name, family in parsed.get("kernel_families", {}).items()
        if name in kernel_names
    }
    return filtered


def avg_stats(step_to_dict, steps):
    """Average {name -> {count, dur_ms}} across steps.

    Returns {name -> {avg_count, avg_dur_ms}}, sorted by avg_dur_ms descending.
    """
    all_names = set()
    for s in steps:
        all_names.update(step_to_dict.get(s, {}))
    n = len(steps)
    if not n:
        return {}
    result = {}
    zero = {"count": 0, "dur_ms": 0.0}
    for name in all_names:
        entries = [step_to_dict.get(s, {}).get(name) or zero for s in steps]
        result[name] = {
            "avg_count":  sum(e["count"]  for e in entries) / n,
            "avg_dur_ms": sum(e["dur_ms"] for e in entries) / n,
        }
    return dict(sorted(result.items(), key=lambda x: -x[1]["avg_dur_ms"]))


def compute_avgs(parsed):
    """Compute all average stats from a parsed trace. Returns a data dict."""
    # Support both dict (new API) and tuple (old API) for backward compatibility
    if isinstance(parsed, dict):
        framework = parsed.get("framework", "pytorch")
        step_to_triton       = parsed["step_to_triton"]
        step_to_kernels      = parsed["step_to_kernels"]
        step_to_aten         = parsed["step_to_aten"]
        step_to_tf_ops       = parsed.get("step_to_tf_ops", defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0})))
        step_to_non_triton_kernel_efficiency = parsed.get("step_to_non_triton_kernel_efficiency", defaultdict(list))
        step_durations       = parsed["step_durations"]
        step_ranges          = parsed.get("step_ranges", {})
        kernel_families      = parsed.get("kernel_families", {})
    else:
        framework = "pytorch"
        if len(parsed) == 6:
            step_to_triton, step_to_kernels, _, step_to_aten, step_to_cncl, step_durations = parsed
        else:
            step_to_triton, step_to_kernels, step_to_aten, step_to_cncl, step_durations = parsed
        step_to_tf_ops = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ms": 0.0}))
        step_to_non_triton_kernel_efficiency = defaultdict(list)
        step_ranges = {}
        kernel_families = {}
    all_steps = sorted(
        set(step_durations)
        | set(step_to_kernels)
        | set(step_to_aten)
        | set(step_to_tf_ops)
        | set(step_to_non_triton_kernel_efficiency)
    )
    n_steps   = len(all_steps)
    mean      = lambda vals: sum(vals) / n_steps if n_steps else 0.0

    step_stats = {}
    for step in all_steps:
        sd  = step_durations.get(step, 0.0)
        step_kernels = step_to_kernels.get(step, {})
        step_triton = step_to_triton.get(step, [])
        step_aten = step_to_aten.get(step, {})
        step_tf_ops = step_to_tf_ops.get(step, {})
        kc  = sum(v["count"]  for v in step_kernels.values())
        ckd = sum(
            v["dur_ms"]
            for name, v in step_kernels.items()
            if (kernel_families.get(name) or extract_kernel_family(name)) != "collective"
        )
        tc  = len(step_triton)
        td  = sum((k["dur(ms)"] or 0.0) for k in step_triton)
        ac  = sum(v["count"]  for v in step_aten.values())
        ad  = sum(v["dur_ms"] for v in step_aten.values())
        fc  = sum(v["count"]  for v in step_tf_ops.values())
        fd  = sum(v["dur_ms"] for v in step_tf_ops.values())
        step_stats[step] = (sd, kc, ckd, tc, td, ac, ad, fc, fd)

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
        "code_hashes": set(),
        "code_signature_hashes": set(),
        "tiling_hashes": set(),
    }))
    triton_meta_cache = {}
    for step, kernels in step_to_triton.items():
        for k in kernels:
            kernel_name = k["kernel_name"]
            a = step_triton_agg[step][kernel_name]
            a["count"]  += 1
            a["dur_ms"] += k["dur(ms)"] or 0.0
            if k["total io(GB)"] is not None:
                a["io_gb"]  += k["total io(GB)"]
            if k["IO efficiency(GB/s)"] is not None:
                a["io_eff_sum"] += k["IO efficiency(GB/s)"]
                a["io_eff_count"] += 1
            meta = triton_meta_cache.setdefault(kernel_name, {})
            code_hash = k.get("triton_code_hash") or ""
            code_signature_hash = k.get("triton_code_signature_hash") or ""
            code_text = k.get("triton_output_code")
            if code_text and "code_hash" not in meta:
                meta["code_hash"] = _stable_text_hash(code_text)
                meta["code_signature_hash"] = _stable_triton_code_signature_hash(code_text)
            elif code_hash and "code_hash" not in meta:
                meta["code_hash"] = code_hash
                meta["code_signature_hash"] = code_signature_hash
            tiling_text = k.get("tiling config")
            if tiling_text and "tiling_hash" not in meta:
                meta["tiling_hash"] = _stable_triton_tiling_hash(tiling_text)
            if meta.get("code_hash"):
                a["code_hashes"].add(meta["code_hash"])
            if meta.get("code_signature_hash"):
                a["code_signature_hashes"].add(meta["code_signature_hash"])
            if meta.get("tiling_hash"):
                a["tiling_hashes"].add(meta["tiling_hash"])

    all_triton_names = set()
    for s in all_steps:
        all_triton_names.update(step_triton_agg[s])

    avg_triton = {}
    for name in all_triton_names:
        code_hashes = set()
        code_signature_hashes = set()
        tiling_hashes = set()
        for step in all_steps:
            stats = step_triton_agg[step].get(name)
            if not stats:
                continue
            code_hashes.update(stats.get("code_hashes", set()))
            code_signature_hashes.update(stats.get("code_signature_hashes", set()))
            tiling_hashes.update(stats.get("tiling_hashes", set()))
        io_eff_sum = sum(step_triton_agg[s].get(name, {"io_eff_sum": 0.0})["io_eff_sum"] for s in all_steps)
        io_eff_count = sum(step_triton_agg[s].get(name, {"io_eff_count": 0})["io_eff_count"] for s in all_steps)
        avg_triton[name] = {
            "avg_count":  mean([step_triton_agg[s].get(name, {"count": 0})["count"] for s in all_steps]),
            "avg_dur_ms": mean([step_triton_agg[s].get(name, {"dur_ms": 0.0})["dur_ms"] for s in all_steps]),
            "avg_io_gb":  mean([step_triton_agg[s].get(name, {"io_gb": 0.0})["io_gb"] for s in all_steps]),
            "avg_io_eff": (io_eff_sum / io_eff_count) if io_eff_count else None,
            "triton_normalized_name": _normalize_triton_kernel_name(name),
            "triton_code_hash": next(iter(code_hashes)) if len(code_hashes) == 1 else "",
            "triton_code_hashes": sorted(code_hashes),
            "triton_code_signature_hash": (
                next(iter(code_signature_hashes)) if len(code_signature_hashes) == 1 else ""
            ),
            "triton_code_signature_hashes": sorted(code_signature_hashes),
            "triton_tiling_hash": next(iter(tiling_hashes)) if len(tiling_hashes) == 1 else "",
            "triton_tiling_hashes": sorted(tiling_hashes),
        }
    avg_triton = dict(sorted(avg_triton.items(), key=lambda x: -x[1]["avg_dur_ms"]))

    # Non-Triton kernel efficiency aggregation.  These metrics can be emitted as
    # Chrome trace counter events instead of kernel args, so parse_trace already
    # associates them with nearby non-Triton kernel intervals.
    step_non_triton_eff_agg = defaultdict(lambda: defaultdict(lambda: {
        "kernel_name": "",
        "family": "",
        "operator": "",
        "input_dims": "",
        "input_types": "",
        "input_strides": "",
        "concrete_inputs": "",
        "operator_details": "",
        "count": 0,
        "dur_ms": 0.0,
        "compute_eff_sum": 0.0,
        "compute_eff_count": 0,
        "io_eff_sum": 0.0,
        "io_eff_count": 0,
        "op_eff_sum": 0.0,
        "op_eff_count": 0,
    }))
    for step, kernels in step_to_non_triton_kernel_efficiency.items():
        for kernel in kernels:
            name = kernel["kernel_name"]
            signature = kernel.get("operator_signature") or ""
            row_key = name if not signature else f"{name}||{signature}"
            stats = step_non_triton_eff_agg[step][row_key]
            stats["kernel_name"] = name
            stats["family"] = kernel.get("family") or kernel_families.get(name) or extract_kernel_family(name)
            for detail_key in (
                "operator",
                "input_dims",
                "input_types",
                "input_strides",
                "concrete_inputs",
                "operator_details",
            ):
                if kernel.get(detail_key):
                    stats[detail_key] = kernel.get(detail_key)
            stats["count"] += 1
            stats["dur_ms"] += kernel.get("dur(ms)") or 0.0
            for source_key, sum_key, count_key in (
                ("compute_efficiency", "compute_eff_sum", "compute_eff_count"),
                ("io_efficiency", "io_eff_sum", "io_eff_count"),
                ("op_efficiency", "op_eff_sum", "op_eff_count"),
            ):
                value = kernel.get(source_key)
                if value is not None:
                    stats[sum_key] += value
                    stats[count_key] += 1

    all_non_triton_eff_names = set()
    for step in all_steps:
        all_non_triton_eff_names.update(step_non_triton_eff_agg[step])

    avg_non_triton_kernel_efficiency = {}
    for row_key in all_non_triton_eff_names:
        compute_sum = sum(
            step_non_triton_eff_agg[s].get(row_key, {"compute_eff_sum": 0.0})["compute_eff_sum"]
            for s in all_steps
        )
        compute_count = sum(
            step_non_triton_eff_agg[s].get(row_key, {"compute_eff_count": 0})["compute_eff_count"]
            for s in all_steps
        )
        io_sum = sum(
            step_non_triton_eff_agg[s].get(row_key, {"io_eff_sum": 0.0})["io_eff_sum"]
            for s in all_steps
        )
        io_count = sum(
            step_non_triton_eff_agg[s].get(row_key, {"io_eff_count": 0})["io_eff_count"]
            for s in all_steps
        )
        op_sum = sum(
            step_non_triton_eff_agg[s].get(row_key, {"op_eff_sum": 0.0})["op_eff_sum"]
            for s in all_steps
        )
        op_count = sum(
            step_non_triton_eff_agg[s].get(row_key, {"op_eff_count": 0})["op_eff_count"]
            for s in all_steps
        )
        family = ""
        kernel_name = ""
        detail_values = {
            "operator": "",
            "input_dims": "",
            "input_types": "",
            "input_strides": "",
            "concrete_inputs": "",
            "operator_details": "",
        }
        for step in all_steps:
            stats = step_non_triton_eff_agg[step].get(row_key)
            if stats and stats.get("family"):
                family = stats["family"]
            if stats and stats.get("kernel_name"):
                kernel_name = stats["kernel_name"]
            if stats:
                for detail_key in detail_values:
                    if stats.get(detail_key):
                        detail_values[detail_key] = stats.get(detail_key)
        kernel_name = kernel_name or str(row_key).split("||", 1)[0]
        avg_non_triton_kernel_efficiency[row_key] = {
            "kernel_name": kernel_name,
            "family": family or kernel_families.get(kernel_name) or extract_kernel_family(kernel_name),
            **detail_values,
            "avg_count": mean([step_non_triton_eff_agg[s].get(row_key, {"count": 0})["count"] for s in all_steps]),
            "avg_dur_ms": mean([step_non_triton_eff_agg[s].get(row_key, {"dur_ms": 0.0})["dur_ms"] for s in all_steps]),
            "avg_compute_efficiency": (compute_sum / compute_count) if compute_count else None,
            "avg_io_efficiency": (io_sum / io_count) if io_count else None,
            "avg_op_efficiency": (op_sum / op_count) if op_count else None,
        }
    avg_non_triton_kernel_efficiency = dict(
        sorted(avg_non_triton_kernel_efficiency.items(), key=lambda x: -x[1]["avg_dur_ms"])
    )

    return {
        "framework":      framework,
        "all_steps":      all_steps,
        "n_steps":        n_steps,
        "step_stats":     step_stats,
        "avg_row":        avg_row,
        "KERNEL_TYPES":   KERNEL_TYPES,
        "kt_avgs":        kt_avgs,
        "avg_kernels":    avg_kernels_data,
        "avg_aten":       avg_stats(step_to_aten, all_steps),
        "avg_tf_ops":     avg_stats(step_to_tf_ops, all_steps),
        "avg_triton":     avg_triton,
        "avg_non_triton_kernel_efficiency": avg_non_triton_kernel_efficiency,
        "step_to_triton": step_to_triton,
        "step_ranges":    step_ranges,
        "kernel_families": kernel_families,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

_HDR = (f"{'step':<8} {'step_dur(ms)':<14} {'kernels':<10} {'compute_kernel_dur(ms)':<24}"
        f" {'triton':<10} {'triton_dur(ms)':<16} {'aten_ops':<10} {'aten_dur(ms)':<14}"
        f" {'tf_ops':<10} {'tf_dur(ms)':<12}")


def print_step_summary(data, label=""):
    title = f"=== Per-Step Summary ({data['n_steps']} steps)"
    if label:
        title += f" — {label}"
    print(f"\n{title} ===")
    print(_HDR)
    print("-" * len(_HDR))
    for step in data["all_steps"]:
        sd, kc, ckd, tc, td, ac, ad, fc, fd = data["step_stats"][step]
        print(f"{step:<8} {sd:<14.3f} {kc:<10} {ckd:<24.3f} {tc:<10} {td:<16.3f}"
              f" {ac:<10} {ad:<14.3f} {fc:<10} {fd:<12.3f}")
    avg_sd, avg_kc, avg_ckd, avg_tc, avg_td, avg_ac, avg_ad, avg_fc, avg_fd = data["avg_row"]
    print("-" * len(_HDR))
    print(f"{'avg':<8} {avg_sd:<14.3f} {avg_kc:<10.1f} {avg_ckd:<24.3f} {avg_tc:<10.1f} {avg_td:<16.3f}"
          f" {avg_ac:<10.1f} {avg_ad:<14.3f} {avg_fc:<10.1f} {avg_fd:<12.3f}")


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
        "triton_dur(ms)", "aten_ops", "aten_dur(ms)", "tf_ops", "tf_dur(ms)",
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
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "kernel_name", "avg_count", "avg_dur_ms", "avg_us_per_call",
            "avg_io_gb", "avg_io_efficiency",
        ])
        writer.writeheader()
        for name, s in avg_triton.items():
            cnt = s["avg_count"]
            dur = s["avg_dur_ms"]
            writer.writerow({
                "kernel_name":       name,
                "avg_count":         fmt3(cnt),
                "avg_dur_ms":        fmt3(dur),
                "avg_us_per_call":   fmt3(dur / cnt * 1000) if cnt > 0 else "",
                "avg_io_gb":         fmt3(s["avg_io_gb"]),
                "avg_io_efficiency": fmt3(s["avg_io_eff"]),
            })
    print(f"Wrote {path} ({len(avg_triton)} rows)")


def _write_non_triton_kernel_efficiency_avg_csv(path, avg_efficiency):
    """Write non-Triton kernel hardware efficiency metrics when present."""
    fields = [
        "kernel_name",
        "family",
        "operator",
        "input_dims",
        "input_types",
        "input_strides",
        "concrete_inputs",
        "avg_count",
        "avg_dur_ms",
        "avg_us_per_call",
        "avg_compute_efficiency",
        "avg_io_efficiency",
        "avg_op_efficiency",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, s in avg_efficiency.items():
            cnt = s["avg_count"]
            dur = s["avg_dur_ms"]
            writer.writerow({
                "kernel_name": s.get("kernel_name") or name,
                "family": s.get("family", ""),
                "operator": s.get("operator", ""),
                "input_dims": s.get("input_dims", ""),
                "input_types": s.get("input_types", ""),
                "input_strides": s.get("input_strides", ""),
                "concrete_inputs": s.get("concrete_inputs", ""),
                "avg_count": fmt3(cnt),
                "avg_dur_ms": fmt3(dur),
                "avg_us_per_call": fmt3(dur / cnt * 1000) if cnt > 0 else "",
                "avg_compute_efficiency": fmt3(s.get("avg_compute_efficiency")),
                "avg_io_efficiency": fmt3(s.get("avg_io_efficiency")),
                "avg_op_efficiency": fmt3(s.get("avg_op_efficiency")),
            })
    print(f"Wrote {path} ({len(avg_efficiency)} rows)")


def _write_kernel_types_csv(path, kernel_type_names, kt_avgs):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "avg_count", "avg_dur_ms"])
        writer.writeheader()
        # kernel_type_names contains only compute families (no collective)
        for ktype in kernel_type_names:
            ac, ad = kt_avgs[ktype]
            writer.writerow({
                "type":       ktype,
                "avg_count":  fmt3(ac),
                "avg_dur_ms": fmt3(ad),
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
    remaining_a = set(avg_triton_a)
    remaining_b = set(avg_triton_b)
    matched = []
    for method in ("exact_name", "code_hash", "code_signature", "normalized_name_tiling", "normalized_name"):
        by_key_a = defaultdict(list)
        by_key_b = defaultdict(list)
        for name in remaining_a:
            for candidate_method, key in _triton_match_candidates(name, avg_triton_a[name]):
                if candidate_method == method:
                    by_key_a[key].append(name)
        for name in remaining_b:
            for candidate_method, key in _triton_match_candidates(name, avg_triton_b[name]):
                if candidate_method == method:
                    by_key_b[key].append(name)
        for key in sorted(set(by_key_a) & set(by_key_b)):
            names_a = sorted(by_key_a[key])
            names_b = sorted(by_key_b[key])
            if len(names_a) != 1 or len(names_b) != 1:
                continue
            name_a = names_a[0]
            name_b = names_b[0]
            if name_a not in remaining_a or name_b not in remaining_b:
                continue
            matched.append((name_a, name_b, method))
            remaining_a.remove(name_a)
            remaining_b.remove(name_b)

    rows = []
    for name_a, name_b, method in matched:
        a = avg_triton_a.get(name_a, zero)
        b = avg_triton_b.get(name_b, zero)
        delta = b["avg_dur_ms"] - a["avg_dur_ms"]
        rows.append({
            "kernel_name":  _triton_display_name(name_a, name_b, a, b),
            "match_method": method,
            "kernel_name_A": name_a,
            "kernel_name_B": name_b,
            "avg_dur_ms_A": fmt3(a["avg_dur_ms"]),
            "avg_dur_ms_B": fmt3(b["avg_dur_ms"]),
            "delta_dur_ms": fmt3(delta),
            "avg_count_A":  fmt3(a["avg_count"]),
            "avg_count_B":  fmt3(b["avg_count"]),
            "avg_io_gb_A":  fmt3(a["avg_io_gb"]),
            "avg_io_gb_B":  fmt3(b["avg_io_gb"]),
            "_sort":        abs(delta),
        })

    for name in sorted(remaining_a):
        a = avg_triton_a.get(name, zero)
        delta = -a["avg_dur_ms"]
        rows.append({
            "kernel_name":  _triton_display_name(name, "", a, zero),
            "match_method": "unmatched",
            "kernel_name_A": name,
            "kernel_name_B": "",
            "avg_dur_ms_A": fmt3(a["avg_dur_ms"]),
            "avg_dur_ms_B": fmt3(0),
            "delta_dur_ms": fmt3(delta),
            "avg_count_A":  fmt3(a["avg_count"]),
            "avg_count_B":  fmt3(0),
            "avg_io_gb_A":  fmt3(a["avg_io_gb"]),
            "avg_io_gb_B":  fmt3(0),
            "_sort":        abs(delta),
        })

    for name in sorted(remaining_b):
        b = avg_triton_b.get(name, zero)
        delta = b["avg_dur_ms"]
        rows.append({
            "kernel_name":  _triton_display_name("", name, zero, b),
            "match_method": "unmatched",
            "kernel_name_A": "",
            "kernel_name_B": name,
            "avg_dur_ms_A": fmt3(0),
            "avg_dur_ms_B": fmt3(b["avg_dur_ms"]),
            "delta_dur_ms": fmt3(delta),
            "avg_count_A":  fmt3(0),
            "avg_count_B":  fmt3(b["avg_count"]),
            "avg_io_gb_A":  fmt3(0),
            "avg_io_gb_B":  fmt3(b["avg_io_gb"]),
            "_sort":        abs(delta),
        })

    rows.sort(key=lambda r: (-r["_sort"], r["kernel_name"]))
    fields = ["kernel_name", "match_method", "kernel_name_A", "kernel_name_B",
              "avg_dur_ms_A", "avg_dur_ms_B", "delta_dur_ms",
              "avg_count_A", "avg_count_B", "avg_io_gb_A", "avg_io_gb_B"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _write_kernel_types_cmp_csv(path, data_a, data_b):
    rows = _kernel_type_cmp_rows(data_a, data_b)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "type", "avg_dur_ms_A", "avg_dur_ms_B",
            "delta_dur_ms", "avg_count_A", "avg_count_B", "delta_count",
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _kernel_type_cmp_rows(data_a, data_b):
    """Build per-kernel-family comparison rows.

    Rows are ordered by the largest absolute duration changes first so the
    remaining type comparison view carries the old Delta tab's value.
    """
    all_types = list(dict.fromkeys(
        [t for t in data_a["KERNEL_TYPES"] if t != "other"] +
        [t for t in data_b["KERNEL_TYPES"] if t != "other"]
    ))
    all_types.append("other")
    rows = []
    for ktype in all_types:
        ac_a, ad_a = data_a["kt_avgs"].get(ktype, (0.0, 0.0))
        ac_b, ad_b = data_b["kt_avgs"].get(ktype, (0.0, 0.0))
        delta_dur = ad_b - ad_a
        delta_count = ac_b - ac_a
        delta_abs = abs(delta_dur)
        rows.append({
            "type":         ktype,
            "avg_dur_ms_A": fmt3(ad_a),
            "avg_dur_ms_B": fmt3(ad_b),
            "delta_dur_ms": fmt3(delta_dur),
            "avg_count_A":  fmt3(ac_a),
            "avg_count_B":  fmt3(ac_b),
            "delta_count":  fmt3(delta_count),
            "_delta_abs":   delta_abs,
        })

    rows.sort(key=lambda row: (-row["_delta_abs"], row["type"] == "other", row["type"]))
    return rows


# ── Top-level write functions ─────────────────────────────────────────────────

def _is_tensorflow_data(data):
    return data.get("framework") == "tensorflow"


def write_single(data, args):
    os.makedirs(args.output_dir, exist_ok=True)
    is_tensorflow = _is_tensorflow_data(data)

    # Per-step triton CSVs + source files
    if args.save_triton_csv or args.save_triton_code:
        triton_fields = ["kernel_name", "dur(ms)", "total io(GB)", "IO efficiency(GB/s)", "tiling config", "triton_code_file"]
        for step in data["all_steps"]:
            kernels = [k for k in data["step_to_triton"][step] if k.get("triton_output_code") is not None]
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
    if not is_tensorflow or data["avg_triton"]:
        _write_triton_avg_csv(os.path.join(args.output_dir, "triton_kernels_avg.csv"), data["avg_triton"])
    if not is_tensorflow or data.get("avg_non_triton_kernel_efficiency"):
        _write_non_triton_kernel_efficiency_avg_csv(
            os.path.join(args.output_dir, "non_triton_kernel_efficiency_avg.csv"),
            data.get("avg_non_triton_kernel_efficiency", {}),
        )
    if not is_tensorflow or data["avg_aten"]:
        write_avg_csv(os.path.join(args.output_dir, "aten_ops_avg.csv"),    data["avg_aten"],    "op_name")
    if data.get("avg_tf_ops"):
        write_avg_csv(os.path.join(args.output_dir, "tf_ops_avg.csv"),      data["avg_tf_ops"],  "op_name")
    _write_kernel_types_csv(os.path.join(args.output_dir, "kernel_types_avg.csv"), data["KERNEL_TYPES"], data["kt_avgs"])


def write_comparison(data_a, data_b, args):
    os.makedirs(args.output_dir, exist_ok=True)
    has_tensorflow = _is_tensorflow_data(data_a) or _is_tensorflow_data(data_b)
    _write_kernels_cmp_csv(os.path.join(args.output_dir, "all_kernels_cmp.csv"), data_a, data_b)
    if not has_tensorflow or data_a["avg_triton"] or data_b["avg_triton"]:
        _write_triton_cmp_csv(os.path.join(args.output_dir, "triton_kernels_cmp.csv"),
                              data_a["avg_triton"], data_b["avg_triton"])
    if not has_tensorflow or data_a["avg_aten"] or data_b["avg_aten"]:
        _write_cmp_avg_csv(os.path.join(args.output_dir, "aten_ops_cmp.csv"),
                           data_a["avg_aten"], data_b["avg_aten"], "op_name")
    if data_a.get("avg_tf_ops") or data_b.get("avg_tf_ops"):
        _write_cmp_avg_csv(os.path.join(args.output_dir, "tf_ops_cmp.csv"),
                           data_a.get("avg_tf_ops", {}), data_b.get("avg_tf_ops", {}), "op_name")
    _write_kernel_types_cmp_csv(os.path.join(args.output_dir, "kernel_types_cmp.csv"), data_a, data_b)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parse PyTorch or TensorFlow Chrome trace JSON(s) and extract kernel/op info. "
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
        "--steps",
        default="",
        help="Analyze only selected steps, e.g. '0', '0,2,5', or '1-4'. For compare, applies to both traces unless --steps-a/--steps-b are set.",
    )
    parser.add_argument("--steps-a", default="", help="Step selector for the first trace in compare mode.")
    parser.add_argument("--steps-b", default="", help="Step selector for the second trace in compare mode.")
    args = parser.parse_args(argv)
    if len(args.trace_files) > 2:
        parser.error("At most two trace files can be provided.")

    if len(args.trace_files) == 1:
        parsed = parse_trace(
            args.trace_files[0],
            keep_triton_code=bool(args.save_triton_csv or args.save_triton_code),
        )
        step_filter = args.steps_a or args.steps
        if step_filter:
            parsed = filter_parsed_steps(parsed, step_filter)
            print(f"Step filter A: {step_filter}")
        data = compute_avgs(parsed)
        print_step_summary(data)
        print_kernel_type_breakdown(data)
        print_top_kernels(data)
        write_single(data, args)
    else:
        label_a = os.path.basename(args.trace_files[0])
        label_b = os.path.basename(args.trace_files[1])
        step_filter_a = args.steps_a or args.steps
        step_filter_b = args.steps_b or args.steps
        keep_triton_code = bool(args.save_triton_csv or args.save_triton_code)
        parsed_a = parse_trace(args.trace_files[0], keep_triton_code=keep_triton_code)
        parsed_b = parse_trace(args.trace_files[1], keep_triton_code=keep_triton_code)
        if step_filter_a:
            parsed_a = filter_parsed_steps(parsed_a, step_filter_a)
            print(f"Step filter A: {step_filter_a}")
        if step_filter_b:
            parsed_b = filter_parsed_steps(parsed_b, step_filter_b)
            print(f"Step filter B: {step_filter_b}")
        data_a = compute_avgs(parsed_a)
        data_b = compute_avgs(parsed_b)
        print_comparison(data_a, data_b, label_a, label_b)
        print_top_kernels(data_a, label=label_a)
        print_top_kernels(data_b, label=label_b)
        write_comparison(data_a, data_b, args)


if __name__ == "__main__":
    main()
