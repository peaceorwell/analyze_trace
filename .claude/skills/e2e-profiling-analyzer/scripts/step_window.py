#!/usr/bin/env python3
"""Find iteration boundaries and recommend a steady-state analysis window.

The Measurement Validity Gate needs a defensible window before any steady-state
speed claim. This script derives one from evidence instead of eyeballing the
timeline: it locates step ranges, measures per-step host and device cost, trims
warmup and truncated steps, and reports repeatability.

Usage:
    python3 step_window.py cnperf_data.db
    python3 step_window.py cnperf_data.db --format json
    python3 step_window.py cnperf_data.db --step-regex '^ProfilerStep' --max-steps 10
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys

try:
    from query_common import load_string_map, table_exists
except ImportError:  # allow importing this module from another sys.path
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from query_common import load_string_map, table_exists

# A step shorter than this is noise, not an iteration.
MIN_STEP_NS = 1_000
# Leading steps cost more than this multiple of the remaining median -> warmup.
WARMUP_RATIO = 1.20
# A trailing step this much shorter than the median is a truncated capture tail.
TRUNCATED_RATIO = 0.60
# Repeatability thresholds on the coefficient of variation of retained steps.
CV_PASS = 0.10
CV_PARTIAL = 0.25
# Relative outlier threshold used when the step series is otherwise constant.
OUTLIER_RATIO = 0.15
# Steps needed before "steady state" means anything.
MIN_STEADY_STEPS = 3
# Marker-kernel fallback bounds: enough repeats to be an iteration marker,
# few enough that it is not a per-layer kernel firing thousands of times.
MARKER_MIN_COUNT = 3
MARKER_MAX_COUNT = 2_000
MARKER_MAX_CV = 0.35
DEFAULT_STEP_REGEX = r"(?i)(profilerstep|iteration|train_step|forward_backward)"


def ms(value_ns: float) -> float:
    return round(value_ns / 1e6, 3)


def cv(values: list[float]) -> float | None:
    """Coefficient of variation; None when it is not defined."""
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean <= 0:
        return None
    return round(statistics.stdev(values) / mean, 4)


def _where(process_id, device_id, columns) -> tuple[str, list]:
    clauses, params = [], []
    if process_id is not None and "processId" in columns:
        clauses.append("processId = ?")
        params.append(process_id)
    if device_id is not None and "deviceId" in columns:
        clauses.append("deviceId = ?")
        params.append(device_id)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def _columns(cursor, table: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def load_step_ranges(cursor, string_map, step_regex, process_id):
    """Host step ranges, the strongest available iteration evidence."""
    if not table_exists(cursor, "Internal_operation_range_data"):
        return []
    pattern = re.compile(step_regex)
    matched_ids = [nid for nid, name in string_map.items() if pattern.search(name or "")]
    if not matched_ids:
        return []
    columns = _columns(cursor, "Internal_operation_range_data")
    where, params = _where(process_id, None, columns)
    placeholders = ",".join("?" * len(matched_ids))
    clause = f"nameId IN ({placeholders})"
    where = f"{where} AND {clause}" if where else f" WHERE {clause}"
    cursor.execute(
        f"SELECT start, end, nameId, threadId FROM Internal_operation_range_data{where}",
        params + matched_ids,
    )
    steps = [
        {
            "name": string_map.get(name_id, f"nameId={name_id}"),
            "start_ns": int(start),
            "end_ns": int(end),
            "thread_id": thread_id,
        }
        for start, end, name_id, thread_id in cursor.fetchall()
        if end is not None and start is not None and int(end) - int(start) > MIN_STEP_NS
    ]
    steps.sort(key=lambda step: step["start_ns"])
    return steps


def infer_marker_steps(cursor, string_map, process_id, device_id):
    """Fallback: cut iterations at a regularly repeating device kernel."""
    if not table_exists(cursor, "device_task_kernel_data"):
        return [], None
    columns = _columns(cursor, "device_task_kernel_data")
    where, params = _where(process_id, device_id, columns)
    clause = "isComputation = 1"
    where = f"{where} AND {clause}" if where else f" WHERE {clause}"
    cursor.execute(
        f"SELECT nameId, start FROM device_task_kernel_data{where} ORDER BY start", params
    )
    by_name: dict[int, list[int]] = {}
    for name_id, start in cursor.fetchall():
        by_name.setdefault(name_id, []).append(int(start))

    best = None
    for name_id, starts in by_name.items():
        if not MARKER_MIN_COUNT <= len(starts) <= MARKER_MAX_COUNT:
            continue
        deltas = [b - a for a, b in zip(starts, starts[1:])]
        if not deltas or min(deltas) <= 0:
            continue
        spread = cv([float(delta) for delta in deltas])
        if spread is None or spread > MARKER_MAX_CV:
            continue
        # Prefer the most regular marker, then the one covering the most repeats.
        score = (spread, -len(starts))
        if best is None or score < best[0]:
            best = (score, name_id, starts)
    if best is None:
        return [], None

    _, name_id, starts = best
    marker = string_map.get(name_id, f"nameId={name_id}")
    steps = [
        {"name": f"{marker}#{index}", "start_ns": start, "end_ns": nxt, "thread_id": None}
        for index, (start, nxt) in enumerate(zip(starts, starts[1:]))
    ]
    return steps, marker


def attach_device_metrics(cursor, steps, process_id, device_id):
    """Per-step device compute time and kernel count, to separate host from device variance."""
    if not steps or not table_exists(cursor, "device_task_kernel_data"):
        for step in steps:
            step["device_compute_ms"] = None
            step["kernel_count"] = None
        return
    columns = _columns(cursor, "device_task_kernel_data")
    where, params = _where(process_id, device_id, columns)
    clause = "isComputation = 1"
    where = f"{where} AND {clause}" if where else f" WHERE {clause}"
    cursor.execute(
        f"SELECT start, end FROM device_task_kernel_data{where} ORDER BY start", params
    )
    kernels = [(int(start), int(end)) for start, end in cursor.fetchall() if end is not None]

    index = 0
    for step in steps:
        total = 0
        count = 0
        while index < len(kernels) and kernels[index][0] < step["start_ns"]:
            index += 1
        cursor_index = index
        while cursor_index < len(kernels) and kernels[cursor_index][0] < step["end_ns"]:
            start, end = kernels[cursor_index]
            total += max(0, end - start)
            count += 1
            cursor_index += 1
        step["device_compute_ms"] = ms(total)
        step["kernel_count"] = count


def classify_steps(steps):
    """Split steps into warmup / truncated tail / steady, using durations only."""
    durations = [step["end_ns"] - step["start_ns"] for step in steps]
    for step, duration in zip(steps, durations):
        step["duration_ms"] = ms(duration)
        step["class"] = "steady"

    remaining = list(range(len(steps)))
    # Trim leading warmup: compilation, autotuning and cache warm-up inflate early steps.
    while len(remaining) > MIN_STEADY_STEPS:
        head, rest = remaining[0], remaining[1:]
        rest_median = statistics.median(durations[i] for i in rest)
        if rest_median > 0 and durations[head] > WARMUP_RATIO * rest_median:
            steps[head]["class"] = "warmup"
            remaining = rest
            continue
        break
    # Trim a truncated tail: profiling usually stops mid-iteration.
    while len(remaining) > MIN_STEADY_STEPS:
        tail, rest = remaining[-1], remaining[:-1]
        rest_median = statistics.median(durations[i] for i in rest)
        if rest_median > 0 and durations[tail] < TRUNCATED_RATIO * rest_median:
            steps[tail]["class"] = "truncated"
            remaining = rest
            continue
        break

    # Flag remaining outliers without removing them; they are evidence, not noise.
    if len(remaining) >= MIN_STEADY_STEPS:
        steady = [durations[i] for i in remaining]
        median = statistics.median(steady)
        deviations = [abs(value - median) for value in steady]
        mad = statistics.median(deviations)
        # A near-constant series has mad == 0; fall back to a relative threshold so a
        # single slow step is still flagged instead of hiding behind zero dispersion.
        threshold = 3 * mad if mad > 0 else OUTLIER_RATIO * median
        for index, value in zip(remaining, steady):
            if threshold > 0 and abs(value - median) > threshold:
                steps[index]["class"] = "outlier"
    return remaining


def analyze_db(db_path, step_regex, process_id=None, device_id=None, max_steps=None):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        string_map = load_string_map(cursor)
        limitations = []

        steps = load_step_ranges(cursor, string_map, step_regex, process_id)
        source = "profiler_step_ranges"
        marker = None
        if not steps:
            steps, marker = infer_marker_steps(cursor, string_map, process_id, device_id)
            source = "inferred_marker_kernel" if steps else "none"
            if steps:
                limitations.append(
                    "no host step ranges matched; iterations were inferred from the repeating "
                    f"kernel `{marker}` and are weaker evidence than explicit step annotations"
                )

        if not steps:
            return {
                "db": db_path,
                "source": "none",
                "step_count": 0,
                "steps": [],
                "steady_window": None,
                "repeatability": {
                    "verdict": "fail",
                    "reason": "no step ranges and no regular marker kernel found",
                },
                "limitations": [
                    "steady-state window is unavailable; ask for an explicit host window or a "
                    "capture that records step annotations, and mark steady-state claims blocked"
                ],
            }

        attach_device_metrics(cursor, steps, process_id, device_id)
        retained = classify_steps(steps)
        for index, step in enumerate(steps):
            step["index"] = index

        if max_steps and len(retained) > max_steps:
            retained = retained[-max_steps:]
            limitations.append(f"steady window capped to the last {max_steps} steps")

        steady_durations = [float(steps[i]["end_ns"] - steps[i]["start_ns"]) for i in retained]
        device_ms = [
            steps[i]["device_compute_ms"] for i in retained if steps[i].get("device_compute_ms")
        ]
        duration_cv = cv(steady_durations)
        window = {
            "start_ns": steps[retained[0]]["start_ns"],
            "end_ns": steps[retained[-1]]["end_ns"],
            "step_count": len(retained),
            "mean_ms": ms(statistics.fmean(steady_durations)) if steady_durations else None,
            "median_ms": ms(statistics.median(steady_durations)) if steady_durations else None,
            "cv": duration_cv,
            "basis": f"{source}, warmup and truncated steps removed",
        }

        if len(retained) < MIN_STEADY_STEPS:
            verdict = "fail"
            reason = f"only {len(retained)} usable steps; cannot separate typical from outlier"
        elif duration_cv is not None and duration_cv <= CV_PASS:
            verdict = "pass"
            reason = f"{len(retained)} steps with cv={duration_cv}"
        elif duration_cv is not None and duration_cv <= CV_PARTIAL:
            verdict = "partial"
            reason = f"{len(retained)} steps with cv={duration_cv}; report ranges, not single values"
        else:
            verdict = "fail"
            reason = f"step duration cv={duration_cv} is too high for steady-state claims"
        if source == "inferred_marker_kernel" and verdict == "pass":
            verdict = "partial"
            reason += "; iteration boundaries are inferred, not annotated"

        return {
            "db": db_path,
            "source": source,
            "step_marker": marker or step_regex,
            "step_count": len(steps),
            "steps": steps,
            "warmup_steps": [s["index"] for s in steps if s["class"] == "warmup"],
            "truncated_steps": [s["index"] for s in steps if s["class"] == "truncated"],
            "outlier_steps": [s["index"] for s in steps if s["class"] == "outlier"],
            "steady_window": window,
            "device_compute_cv": cv([float(value) for value in device_ms]),
            "repeatability": {"verdict": verdict, "reason": reason},
            "command_hint": f"--start-ns {window['start_ns']} --end-ns {window['end_ns']}",
            "limitations": limitations,
        }
    finally:
        conn.close()


def print_text(payload) -> None:
    print(f"DB: {payload['db']}")
    print(f"Step source: {payload['source']} ({payload.get('step_marker')})")
    print(f"Steps: {payload['step_count']}")
    window = payload.get("steady_window")
    if not window:
        print(f"Steady window: unavailable - {payload['repeatability']['reason']}")
        for item in payload.get("limitations", []):
            print(f"- {item}")
        return
    print(
        f"Steady window: [{window['start_ns']}, {window['end_ns']}] "
        f"steps={window['step_count']} median={window['median_ms']}ms cv={window['cv']}"
    )
    print(f"Repeatability: {payload['repeatability']['verdict']} - {payload['repeatability']['reason']}")
    print(f"Device compute cv: {payload.get('device_compute_cv')}")
    print(f"Apply with: {payload['command_hint']}")
    print()
    print(f"{'idx':>4} {'class':<10} {'duration_ms':>12} {'device_ms':>11} {'kernels':>8}  name")
    for step in payload["steps"]:
        print(
            f"{step['index']:>4} {step['class']:<10} {step['duration_ms']:>12} "
            f"{str(step.get('device_compute_ms')):>11} {str(step.get('kernel_count')):>8}  {step['name'][:48]}"
        )
    for item in payload.get("limitations", []):
        print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend a steady-state analysis window")
    parser.add_argument("db", nargs="+", help="cnperf SQLite DB path(s)")
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--device-id", type=int)
    parser.add_argument(
        "--step-regex",
        default=DEFAULT_STEP_REGEX,
        help="host range name pattern that marks one iteration (default: %(default)s)",
    )
    parser.add_argument(
        "--max-steps", type=int, help="cap the steady window to the last N steps"
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    payloads = [
        analyze_db(db, args.step_regex, args.process_id, args.device_id, args.max_steps)
        for db in args.db
    ]
    if args.format == "json":
        json.dump({"profiles": payloads}, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        for payload in payloads:
            print_text(payload)
            print()
    raise SystemExit(0 if any(p.get("steady_window") for p in payloads) else 3)


if __name__ == "__main__":
    main()
