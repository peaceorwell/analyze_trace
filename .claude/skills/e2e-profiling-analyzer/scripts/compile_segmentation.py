#!/usr/bin/env python3
"""
torch.compile segmentation + host-launch-overhead (cpp_wrapper) analysis.

torch.compile 编译区域分段分析，以及 host launch overhead / cpp_wrapper 检查。

The primary host-overhead indicator is the device-stream (queue) gap ratio. The
cpp_wrapper switch is read from converted trace metadata when possible, including
explicit config keys and trace kernel_file evidence. Only fall back to gap-based
interpretation when the trace does not carry a direct signal.
"""

import argparse
import bisect
import collections
import json
import os
import re
import sqlite3
from pathlib import Path


COMPILE_REGION_REGEX = re.compile(
    r"Torch-Compiled Region|CompiledFunction|TorchDynamo|Inductor|"
    r"compiled_fn|AOTAutograd|graph\s*break",
    re.IGNORECASE,
)
RECOMPILE_REGEX = re.compile(r"TorchDynamo Cache Lookup|recompile|guard", re.IGNORECASE)
CPP_WRAPPER_KEY_RE = re.compile(r"cpp\s*[_\-. ]?\s*wrapper|cppwrapper", re.IGNORECASE)
CPP_WRAPPER_VALUE_RE = re.compile(
    r"cpp\s*[_\-. ]?\s*wrapper\w*\s*[:=]\s*['\"]?"
    r"(true|false|1|0|yes|no|on|off|enabled|disabled)",
    re.IGNORECASE,
)
CPP_WRAPPER_ON_EXTENSIONS = {".cc", ".cpp", ".cxx", ".so", ".dylib", ".dll"}
CPP_WRAPPER_OFF_EXTENSIONS = {".py", ".pyc"}
CUSTOM_OP_SYMBOL_RE = re.compile(r"^[A-Za-z_][\w.]*::[A-Za-z_]\w*")
CUSTOM_OP_FRAME_RE = re.compile(
    r"torch/_library/custom_ops\.py|backend_impl|wrapped_fn|torch/_ops\.py\(\d+\): redispatch",
    re.IGNORECASE,
)
CUSTOM_OP_DISPATCH_RE = re.compile(r"torch/_ops\.py\(\d+\): __call__|PyCapsule", re.IGNORECASE)
CUSTOM_OP_USER_PY_RE = re.compile(r"(?<!torch/)[\w./-]+\.py\(\d+\):\s*[A-Za-z_]\w*", re.IGNORECASE)
EXCLUDED_CUSTOM_SYMBOL_PREFIXES = ("aten::", "c10::", "torch::", "torch_mlu::ops::")
SIMPLE_ATEN_RE = re.compile(
    r"^aten::("
    r"add|sub|mul|div|pow|neg|abs|exp|log|sqrt|rsqrt|sigmoid|silu|gelu|relu|"
    r"tanh|erf|clamp|where|masked|eq|ne|lt|le|gt|ge|maximum|minimum|"
    r"empty|empty_like|zeros|zeros_like|ones|ones_like|full|full_like|fill_|zero_|copy_|clone|to|"
    r"cat|stack|cumsum|cumprod|sum|mean|max|min|amax|amin|prod|softmax|_softmax|"
    r"view|reshape|slice|select|unsqueeze|squeeze|transpose|permute|contiguous|"
    r"expand|repeat|flatten|as_strided|index|gather|scatter|nonzero"
    r")(\b|_|\.|$)",
    re.IGNORECASE,
)
CUSTOM_SIMPLE_ATEN_MIN_AVG_PER_CALL = 5
CUSTOM_SIMPLE_ATEN_MIN_TOTAL_COUNT = 50
CUSTOM_SIMPLE_ATEN_MIN_UNIQUE_OPS = 3
CUSTOM_SIMPLE_ATEN_HIGH_AVG_PER_CALL = 10
CUSTOM_SIMPLE_ATEN_HIGH_TOTAL_COUNT = 100


def table_exists(cur, table):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def table_columns(cur, table):
    if not table_exists(cur, table):
        return set()
    return {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}


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


def normalize_metadata_key(key):
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def custom_op_candidate_priority(name):
    lowered = str(name).lower()
    if CUSTOM_OP_SYMBOL_RE.search(name) and not lowered.startswith(EXCLUDED_CUSTOM_SYMBOL_PREFIXES):
        return 0
    if CUSTOM_OP_USER_PY_RE.search(name):
        return 1
    if CUSTOM_OP_FRAME_RE.search(name):
        return 2
    if CUSTOM_OP_DISPATCH_RE.search(name):
        return 3
    return None


def is_simple_aten(name):
    return bool(SIMPLE_ATEN_RE.search(str(name)))


def boolish_cpp_wrapper_state(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled", "enable"}:
        return "on"
    if text in {"0", "false", "no", "off", "disabled", "disable"}:
        return "off"
    match = CPP_WRAPPER_VALUE_RE.search(text)
    if match:
        return boolish_cpp_wrapper_state(match.group(1))
    return None


def parse_json_maybe(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


class CppWrapperDetector:
    def __init__(self):
        self.explicit = []
        self.kernel_file_exts = collections.Counter()

    def observe_kernel_file(self, value, source):
        if not isinstance(value, str) or not value:
            return
        ext = os.path.splitext(value)[1].lower()
        if ext:
            self.kernel_file_exts[ext] += 1

    def observe_mapping(self, value, source, depth=0):
        if not isinstance(value, dict) or depth > 3:
            return
        for key, item in value.items():
            norm_key = normalize_metadata_key(key)
            if norm_key == "kernelfile":
                self.observe_kernel_file(item, f"{source}.{key}")
                continue
            if CPP_WRAPPER_KEY_RE.search(str(key)):
                state = boolish_cpp_wrapper_state(item)
                if state:
                    self.explicit.append(
                        {"source": f"{source}.{key}", "state": state, "value": str(item)[:200]}
                    )
                continue
            if isinstance(item, dict):
                self.observe_mapping(item, f"{source}.{key}", depth + 1)
            elif isinstance(item, str) and "cpp" in item.lower() and "wrapper" in item.lower():
                state = boolish_cpp_wrapper_state(item)
                if state:
                    self.explicit.append(
                        {"source": f"{source}.{key}", "state": state, "value": item[:200]}
                    )

    def observe_json_text(self, value, source):
        obj = parse_json_maybe(value)
        if isinstance(obj, dict):
            self.observe_mapping(obj, source)

    def summary(self):
        explicit_states = {item["state"] for item in self.explicit}
        py_count = sum(self.kernel_file_exts.get(ext, 0) for ext in CPP_WRAPPER_OFF_EXTENSIONS)
        cpp_count = sum(self.kernel_file_exts.get(ext, 0) for ext in CPP_WRAPPER_ON_EXTENSIONS)
        evidence = self.explicit[:6]
        if self.kernel_file_exts:
            evidence.append(
                {
                    "source": "trace.args.kernel_file",
                    "state": "on" if cpp_count and not py_count else "off" if py_count and not cpp_count else "unknown",
                    "value": dict(self.kernel_file_exts.most_common()),
                }
            )
        if len(explicit_states) == 1:
            state, source, confidence = next(iter(explicit_states)), "explicit_trace_metadata", "high"
        elif len(explicit_states) > 1:
            state, source, confidence = "unknown", "conflicting_explicit_trace_metadata", "low"
        elif cpp_count and not py_count:
            state, source, confidence = "on", "kernel_file_extension", "medium"
        elif py_count and not cpp_count:
            state, source, confidence = "off", "kernel_file_extension", "medium"
        elif py_count and cpp_count:
            state, source, confidence = "unknown", "mixed_kernel_file_extension", "low"
        else:
            state, source, confidence = "unknown", "not_found", "unknown"
        return {
            "state": state,
            "source": source,
            "confidence": confidence,
            "kernel_file_extensions": dict(self.kernel_file_exts.most_common()),
            "evidence": evidence,
        }


def load_cpp_wrapper_signal_from_meta(cur):
    detector = CppWrapperDetector()
    if not table_exists(cur, "meta_information"):
        return detector
    for meta_type, value in cur.execute("SELECT type, value FROM meta_information"):
        if meta_type == "torch_compile_cpp_wrapper":
            obj = parse_json_maybe(value)
            if isinstance(obj, dict) and obj.get("state"):
                detector.observe_mapping(obj, f"meta_information.{meta_type}")
                for ext, count in (obj.get("kernel_file_extensions") or {}).items():
                    detector.kernel_file_exts[ext] += count
                state = obj.get("state")
                if state in {"on", "off"} and obj.get("source") == "explicit_trace_metadata":
                    detector.explicit.append(
                        {"source": f"meta_information.{meta_type}", "state": state, "value": obj.get("source")}
                    )
            continue
        if "cpp" in str(value).lower() or "kernel_file" in str(value):
            detector.observe_json_text(value, f"meta_information.{meta_type}")
    return detector


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


def summarize_custom_op_simple_aten(cur, strings, start=None, end=None):
    """Detect custom-op wrappers that still execute many simple aten ops inside."""
    if not table_exists(cur, "Internal_operation_range_data"):
        return {
            "has_issue": False,
            "highlighted_custom_ops": [],
            "candidate_range_count": 0,
            "simple_aten_range_count": 0,
            "note": "Internal_operation_range_data table is absent.",
        }

    where = ""
    params = {}
    if start is not None and end is not None:
        where = " WHERE start < :window_end AND end > :window_start"
        params = {"window_start": start, "window_end": end}

    simple_by_thread = {}
    candidates = []
    for pid, tid, row_start, row_end, name_id in cur.execute(
        "SELECT processId, threadId, start, end, nameId FROM Internal_operation_range_data"
        f"{where} ORDER BY processId, threadId, start",
        params,
    ):
        if row_end <= row_start:
            continue
        clipped_start = max(row_start, start) if start is not None else row_start
        clipped_end = min(row_end, end) if end is not None else row_end
        if clipped_end <= clipped_start:
            continue
        name = name_of(strings, name_id)
        priority = custom_op_candidate_priority(name)
        if priority is not None:
            candidates.append((priority, pid, tid, clipped_start, clipped_end, name))
        if is_simple_aten(name):
            simple_by_thread.setdefault((pid, tid), []).append((clipped_start, clipped_end, name))

    for rows in simple_by_thread.values():
        rows.sort(key=lambda item: item[0])
    starts_by_thread = {key: [item[0] for item in rows] for key, rows in simple_by_thread.items()}

    groups = {}
    nonempty_candidate_count = 0
    for priority, pid, tid, custom_start, custom_end, name in candidates:
        rows = simple_by_thread.get((pid, tid), [])
        if not rows:
            continue
        idx = bisect.bisect_left(starts_by_thread[(pid, tid)], custom_start)
        nested = []
        for aten_start, aten_end, aten_name in rows[idx:]:
            if aten_start >= custom_end:
                break
            if aten_start >= custom_start and aten_end <= custom_end:
                nested.append((aten_name, aten_end - aten_start))
        if not nested:
            continue
        nonempty_candidate_count += 1
        group = groups.setdefault(
            name,
            {
                "custom_op_name": name,
                "priority": priority,
                "range_count": 0,
                "total_host_ns": 0,
                "nested_simple_aten_count": 0,
                "nested_simple_aten_ns": 0,
                "max_nested_simple_aten_per_call": 0,
                "simple_aten_ops": collections.Counter(),
                "simple_aten_ns_by_op": collections.Counter(),
            },
        )
        group["priority"] = min(group["priority"], priority)
        group["range_count"] += 1
        group["total_host_ns"] += max(0, custom_end - custom_start)
        group["nested_simple_aten_count"] += len(nested)
        group["nested_simple_aten_ns"] += sum(duration for _, duration in nested)
        group["max_nested_simple_aten_per_call"] = max(
            group["max_nested_simple_aten_per_call"], len(nested)
        )
        for aten_name, duration in nested:
            group["simple_aten_ops"][aten_name] += 1
            group["simple_aten_ns_by_op"][aten_name] += duration

    if not groups:
        return {
            "has_issue": False,
            "highlighted_custom_ops": [],
            "candidate_range_count": len(candidates),
            "simple_aten_range_count": sum(len(rows) for rows in simple_by_thread.values()),
            "note": "No custom-op range containing simple aten ops was found.",
        }

    best_priority_with_issue = None
    for priority in sorted({group["priority"] for group in groups.values()}):
        priority_groups = [group for group in groups.values() if group["priority"] == priority]
        if any(_custom_simple_aten_is_issue(group) for group in priority_groups):
            best_priority_with_issue = priority
            break
    selected_priority = (
        best_priority_with_issue
        if best_priority_with_issue is not None
        else min(group["priority"] for group in groups.values())
    )

    rows = []
    for group in groups.values():
        if group["priority"] != selected_priority:
            continue
        avg_per_call = (
            group["nested_simple_aten_count"] / group["range_count"] if group["range_count"] else 0.0
        )
        unique_ops = len(group["simple_aten_ops"])
        highlight = _custom_simple_aten_is_issue(group)
        top_ops = []
        for op_name, count in group["simple_aten_ops"].most_common(8):
            top_ops.append(
                {
                    "name": op_name,
                    "count": count,
                    "total_ms": ms(group["simple_aten_ns_by_op"][op_name]),
                }
            )
        report_priority = _custom_simple_aten_report_priority(group)
        rows.append(
            {
                "custom_op_name": group["custom_op_name"],
                "range_count": group["range_count"],
                "total_host_ms": ms(group["total_host_ns"]),
                "nested_simple_aten_count": group["nested_simple_aten_count"],
                "nested_simple_aten_ms": ms(group["nested_simple_aten_ns"]),
                "avg_simple_aten_per_call": avg_per_call,
                "max_nested_simple_aten_per_call": group["max_nested_simple_aten_per_call"],
                "unique_simple_aten_ops": unique_ops,
                "top_simple_aten_ops": top_ops,
                "highlight": highlight,
                "report_priority": report_priority,
                "must_report": report_priority == "high",
                "reason": _custom_simple_aten_reason(avg_per_call, group["nested_simple_aten_count"], unique_ops),
                "recommendation": (
                    "Move repeated simple aten pointwise/view/reduce/copy/allocation work into the custom "
                    "backend kernel, or restructure the custom op wrapper so Inductor can see and fuse it."
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            not row["highlight"],
            -row["nested_simple_aten_count"],
            -row["nested_simple_aten_ms"],
            row["custom_op_name"],
        )
    )

    return {
        "has_issue": any(row["highlight"] for row in rows),
        "must_report": any(row.get("must_report") for row in rows),
        "top_issue": next((row for row in rows if row.get("highlight")), rows[0] if rows else None),
        "selected_priority": selected_priority,
        "candidate_range_count": len(candidates),
        "candidate_ranges_with_simple_aten": nonempty_candidate_count,
        "simple_aten_range_count": sum(len(rows) for rows in simple_by_thread.values()),
        "highlighted_custom_ops": rows[:10],
        "note": (
            "Custom op ranges containing many simple aten calls usually mean the wrapper/backend custom op "
            "still leaves pointwise, view, reduce, copy, or allocation work unfused; consider moving those "
            "ops into the custom kernel or restructuring the op for Inductor fusion."
        ),
    }


def _custom_simple_aten_is_issue(group):
    unique_ops = len(group["simple_aten_ops"])
    avg_per_call = (
        group["nested_simple_aten_count"] / group["range_count"] if group["range_count"] else 0.0
    )
    return (
        unique_ops >= CUSTOM_SIMPLE_ATEN_MIN_UNIQUE_OPS
        and (
            avg_per_call >= CUSTOM_SIMPLE_ATEN_MIN_AVG_PER_CALL
            or group["nested_simple_aten_count"] >= CUSTOM_SIMPLE_ATEN_MIN_TOTAL_COUNT
        )
    )


def _custom_simple_aten_reason(avg_per_call, total_count, unique_ops):
    return (
        f"{total_count:,} nested simple aten ops, avg {avg_per_call:.1f}/call, "
        f"{unique_ops} unique simple op names"
    )


def _custom_simple_aten_report_priority(group):
    if not _custom_simple_aten_is_issue(group):
        return "low"
    avg_per_call = (
        group["nested_simple_aten_count"] / group["range_count"] if group["range_count"] else 0.0
    )
    if (
        avg_per_call >= CUSTOM_SIMPLE_ATEN_HIGH_AVG_PER_CALL
        or group["nested_simple_aten_count"] >= CUSTOM_SIMPLE_ATEN_HIGH_TOTAL_COUNT
    ):
        return "high"
    return "normal"


def analyze_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        strings = load_strings(cur)
        if not table_exists(cur, "device_task_kernel_data"):
            return {"db": db_path, "label": Path(db_path).stem, "error": "missing device_task_kernel_data"}
        cpp_wrapper_detector = load_cpp_wrapper_signal_from_meta(cur)

        # --- Compiled region ranges (host side) ---
        regions_by_thread = {}
        region_inventory = {}
        recompile_count = 0
        has_region_table = table_exists(cur, "Internal_operation_range_data")
        if has_region_table:
            range_cols = table_columns(cur, "Internal_operation_range_data")
            range_select = "processId, threadId, start, end, nameId"
            if "extra" in range_cols:
                range_select += ", extra"
            for row in cur.execute(f"SELECT {range_select} FROM Internal_operation_range_data"):
                if "extra" in range_cols:
                    pid, tid, start, end, name_id, extra = row
                    if extra and ("kernel_file" in extra or "cpp" in extra.lower()):
                        cpp_wrapper_detector.observe_json_text(extra, "Internal_operation_range_data.extra")
                else:
                    pid, tid, start, end, name_id = row
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

        kernel_cols = table_columns(cur, "device_task_kernel_data")
        kernel_select = "correlationId, start, end, nameId"
        if "extra" in kernel_cols:
            kernel_select += ", extra"
        for row in cur.execute(
            f"SELECT {kernel_select} FROM device_task_kernel_data WHERE isComputation = 1"
        ):
            if "extra" in kernel_cols:
                corr, start, end, name_id, extra = row
                if extra and ("kernel_file" in extra or "cpp" in extra.lower()):
                    cpp_wrapper_detector.observe_json_text(extra, "device_task_kernel_data.extra")
            else:
                corr, start, end, name_id = row
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
        cpp_wrapper_signal = cpp_wrapper_detector.summary()
        if cpp_wrapper_signal["state"] == "off":
            cpp_note = (
                "Trace evidence indicates cpp_wrapper is disabled; use host-launch metrics "
                "to judge whether it is the active bottleneck."
            )
        elif cpp_wrapper_signal["state"] == "on":
            cpp_note = (
                "Trace evidence indicates cpp_wrapper is enabled; if the main-stream gap is "
                "still high, investigate graph breaks, synchronization, tiny kernels, or host framework work."
            )
        else:
            cpp_note = (
                "No direct cpp_wrapper trace signal was found; interpret wrapper mode from "
                "device-stream gap and host launch metrics only as a hypothesis."
            )

        region_inv_rows = sorted(
            (
                {"region": nm, "count": v["count"], "host_ms": ms(v["host_ns"])}
                for nm, v in region_inventory.items()
            ),
            key=lambda r: -r["host_ms"],
        )
        custom_op_simple_aten = summarize_custom_op_simple_aten(cur, strings)

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
                "custom_op_simple_aten": custom_op_simple_aten,
            },
            "custom_op_simple_aten": custom_op_simple_aten,
            "top_outside_region_kernels": top_outside,
            "host_launch_overhead": {
                "main_stream_gap_pct": main_stream_gap_pct,
                "launched_kernel_count": launched_kernel_count,
                "avg_launch_self_us": avg_launch_self_us,
                "total_launch_self_ms": ms(total_launch_self_ns),
                "device_compute_ms": ms(device_compute_ns),
                "launch_self_to_compute_ratio": launch_to_compute_ratio,
                "cpp_wrapper_signal": cpp_wrapper_signal,
                "note": cpp_note,
            },
            "cpp_wrapper_signal": cpp_wrapper_signal,
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
        cpp_signal = hlo.get("cpp_wrapper_signal") or profile.get("cpp_wrapper_signal") or {}
        state_text = {
            "on": "enabled / ON",
            "off": "disabled / OFF",
            "unknown": "unconfirmed",
        }.get(cpp_signal.get("state"), "unconfirmed")
        print(
            f"- cpp_wrapper trace signal: **{state_text}** "
            f"(source: {cpp_signal.get('source', 'n/a')}, confidence: {cpp_signal.get('confidence', 'n/a')})"
        )
        if cpp_signal.get("kernel_file_extensions"):
            print(f"- kernel_file extensions: `{cpp_signal['kernel_file_extensions']}`")
        print(f"- main compute stream gap ratio: **{hlo['main_stream_gap_pct']:.2f}%** (key host-overhead indicator)")
        print(
            f"- avg host launch self-time per kernel: {hlo['avg_launch_self_us']:.2f} us "
            f"over {hlo['launched_kernel_count']:,} launches"
        )
        print(
            f"- launch self-time vs device compute: {hlo['total_launch_self_ms']:,.2f} ms / "
            f"{hlo['device_compute_ms']:,.2f} ms (ratio {hlo['launch_self_to_compute_ratio']:.2f})"
        )
        if hlo.get("note"):
            print(f"- interpretation: {hlo['note']}")

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

        custom_simple = profile.get("custom_op_simple_aten") or seg.get("custom_op_simple_aten") or {}
        custom_rows = custom_simple.get("highlighted_custom_ops") or []
        if custom_rows:
            print("\n### Custom Op Simple Aten Nesting")
            print(f"- issue detected: {'yes' if custom_simple.get('has_issue') else 'no'}")
            if custom_simple.get("note"):
                print(f"- interpretation: {custom_simple['note']}")
            print(
                "| Custom op | Priority | Calls | Simple aten | Avg/call | Simple aten ms | Host ms | Top nested ops | Reason |"
            )
            print("|---|---|---:|---:|---:|---:|---:|---|---|")
            for row in custom_rows:
                top_ops = ", ".join(
                    f"{op['name']}({op['count']})" for op in row.get("top_simple_aten_ops", [])[:5]
                )
                print(
                    f"| {row['custom_op_name']} | {row.get('report_priority', 'low')} | "
                    f"{row['range_count']:,} | "
                    f"{row['nested_simple_aten_count']:,} | {row['avg_simple_aten_per_call']:.1f} | "
                    f"{row['nested_simple_aten_ms']:,.3f} | {row['total_host_ms']:,.3f} | "
                    f"{top_ops} | {row.get('reason', '')} |"
                )

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
