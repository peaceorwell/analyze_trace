#!/usr/bin/env python3
"""
Device Timeline Projection, Device-Stream Gap Ratio, and Uncovered Communication.

设备时间线投影、device stream(队列) gap 占比、未掩盖通信开销。

The device-stream queue gap ratio (fraction of a stream's span spent idle between
device tasks) is the key indicator of whether the host is failing to keep the
device fed, i.e. whether host overhead is large.
"""

import argparse
import json
import sqlite3
import sys

try:
    from query_common import (
        add_window_args,
        clip_interval,
        load_string_map,
        window_payload,
        window_sql,
    )
except ImportError:  # allow running from another cwd
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from query_common import (
        add_window_args,
        clip_interval,
        load_string_map,
        window_payload,
        window_sql,
    )


CATEGORY_PRIORITY = (
    "compute_kernel",
    "comm_kernel",
    "memcpy",
    "memset",
    "atomic",
)

# Device-execution task categories. notifier waits are NOT execution; they are
# idle/synchronization and count toward gaps.
EXEC_CATEGORIES = set(CATEGORY_PRIORITY)


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


def uncovered_total(target_intervals, covering_intervals):
    """Total time of target intervals not overlapped by covering intervals."""
    cover = merge_intervals(covering_intervals)
    uncovered = 0
    i = 0
    for start, end in merge_intervals(target_intervals):
        cursor = start
        while i > 0 and cover[i - 1][1] > cursor:
            i -= 1
        while cursor < end:
            if i >= len(cover):
                uncovered += end - cursor
                break
            cover_start, cover_end = cover[i]
            if cover_end <= cursor:
                i += 1
                continue
            if cover_start >= end:
                uncovered += end - cursor
                break
            if cover_start > cursor:
                uncovered += cover_start - cursor
            cursor = max(cursor, cover_end)
            if cover_end <= end:
                i += 1
    return uncovered


def load_events(cursor, process_id=None, device_id=None, start_ns=None, end_ns=None):
    """Return list of (process_id, device_id, queue_id, start, end, category).

    Intervals are clipped to the analysis window so coverage ratios describe the
    window itself instead of the whole capture.
    """
    where = []
    params = []
    if process_id is not None:
        where.append("processId = ?")
        params.append(process_id)
    if device_id is not None:
        where.append("deviceId = ?")
        params.append(device_id)
    window_clauses, window_params = window_sql(start_ns, end_ns, mode="overlap")
    where.extend(window_clauses)
    params.extend(window_params)
    filt = (" WHERE " + " AND ".join(where)) if where else ""

    events = []

    cursor.execute(
        f"SELECT processId, deviceId, queueId, start, end, isComputation "
        f"FROM device_task_kernel_data{filt}",
        params,
    )
    for pid, did, qid, start, end, is_comp in cursor.fetchall():
        clipped = clip_interval(start, end, start_ns, end_ns)
        if clipped is None:
            continue
        cat = "compute_kernel" if is_comp == 1 else "comm_kernel"
        events.append((pid, did, qid, clipped[0], clipped[1], cat))

    for table, cat in (
        ("device_task_memcpy_data", "memcpy"),
        ("device_task_memset_data", "memset"),
        ("device_task_atomic_operation_data", "atomic"),
    ):
        try:
            cursor.execute(
                f"SELECT processId, deviceId, queueId, start, end FROM {table}{filt}",
                params,
            )
        except sqlite3.OperationalError:
            continue
        for pid, did, qid, start, end in cursor.fetchall():
            clipped = clip_interval(start, end, start_ns, end_ns)
            if clipped is None:
                continue
            events.append((pid, did, qid, clipped[0], clipped[1], cat))

    return events


def project_categories(events):
    """Scanline projection: attribute each wall-clock slice to top active category."""
    points = []
    for _, _, _, start, end, cat in events:
        points.append((start, cat, 1))
        points.append((end, cat, -1))
    points.sort(key=lambda x: (x[0], x[2]))

    categories = {}
    current = {}
    prev_time = None
    for time, cat, delta in points:
        if prev_time is not None and time > prev_time and current:
            main_cat = "other"
            for candidate in CATEGORY_PRIORITY:
                if candidate in current:
                    main_cat = candidate
                    break
            categories[main_cat] = categories.get(main_cat, 0) + time - prev_time
        current[cat] = current.get(cat, 0) + delta
        if current[cat] <= 0:
            del current[cat]
        prev_time = time
    return categories


def analyze_group(events):
    starts = [e[3] for e in events]
    ends = [e[4] for e in events]
    span = max(ends) - min(starts)

    categories = project_categories(events)
    busy = sum(categories.values())
    gap = span - busy

    compute_intervals = [(e[3], e[4]) for e in events if e[5] == "compute_kernel"]
    comm_events = [(e[3], e[4], e[5]) for e in events if e[5] == "comm_kernel"]
    comm_total = sum(end - start for start, end, _ in comm_events)
    comm_uncovered = uncovered_total(
        [(s, e) for s, e, _ in comm_events], compute_intervals
    )

    # Per-queue (device stream) gap ratio.
    queues = {}
    for _, _, qid, start, end, cat in events:
        item = queues.setdefault(
            qid, {"intervals": [], "compute": 0, "count": 0}
        )
        item["intervals"].append((start, end))
        item["count"] += 1
        if cat == "compute_kernel":
            item["compute"] += end - start

    queue_rows = []
    for qid, item in queues.items():
        q_starts = [s for s, _ in item["intervals"]]
        q_ends = [e for _, e in item["intervals"]]
        q_span = max(q_ends) - min(q_starts)
        q_busy = interval_total(item["intervals"])
        q_gap = q_span - q_busy
        queue_rows.append(
            {
                "queue_id": qid,
                "span_ms": ms(q_span),
                "busy_ms": ms(q_busy),
                "gap_ms": ms(q_gap),
                "gap_pct": (q_gap / q_span * 100) if q_span > 0 else 0.0,
                "compute_ms": ms(item["compute"]),
                "task_count": item["count"],
            }
        )
    queue_rows.sort(key=lambda r: -r["compute_ms"])
    if queue_rows:
        queue_rows[0]["is_main_compute_stream"] = True
        for row in queue_rows[1:]:
            row["is_main_compute_stream"] = False
    main_stream_gap_pct = queue_rows[0]["gap_pct"] if queue_rows else 0.0

    return {
        "span_ms": ms(span),
        "categories": {cat: ms(val) for cat, val in categories.items()},
        "category_pct": {
            cat: (val / span * 100 if span > 0 else 0.0)
            for cat, val in categories.items()
        },
        "gap_ms": ms(gap),
        "device_gap_pct": (gap / span * 100) if span > 0 else 0.0,
        "main_stream_gap_pct": main_stream_gap_pct,
        "comm_total_ms": ms(comm_total),
        "comm_uncovered_ms": ms(comm_uncovered),
        "comm_uncovered_pct": (comm_uncovered / comm_total * 100)
        if comm_total > 0
        else 0.0,
        "queues": queue_rows,
    }


def comm_by_name(cursor, process_id, device_id, start_ns=None, end_ns=None):
    where = []
    params = []
    if process_id is not None:
        where.append("processId = ?")
        params.append(process_id)
    if device_id is not None:
        where.append("deviceId = ?")
        params.append(device_id)
    where.append("isComputation = 0")
    window_clauses, window_params = window_sql(start_ns, end_ns, mode="overlap")
    where.extend(window_clauses)
    params = params + window_params
    filt = " WHERE " + " AND ".join(where)

    cursor.execute(
        f"SELECT start, end, nameId, processId, deviceId FROM device_task_kernel_data{filt}",
        params,
    )
    rows = []
    for start, end, name_id, pid, did in cursor.fetchall():
        clipped = clip_interval(start, end, start_ns, end_ns)
        if clipped is not None:
            rows.append((clipped[0], clipped[1], name_id, pid, did))

    compute_filt = filt.replace("isComputation = 0", "isComputation = 1")
    cursor.execute(
        f"SELECT start, end FROM device_task_kernel_data{compute_filt}", params
    )
    compute_intervals = []
    for start, end in cursor.fetchall():
        clipped = clip_interval(start, end, start_ns, end_ns)
        if clipped is not None:
            compute_intervals.append(clipped)

    string_map = load_string_map(cursor)
    by_name = {}
    for start, end, name_id, _, _ in rows:
        by_name.setdefault(name_id, []).append((start, end))

    results = []
    for name_id, intervals in by_name.items():
        name = string_map.get(name_id, f"nameId={name_id}")
        total = sum(e - s for s, e in intervals)
        uncovered = uncovered_total(intervals, compute_intervals)
        results.append(
            {
                "kernel_name": name,
                "total_ms": ms(total),
                "uncovered_ms": ms(uncovered),
                "uncovered_pct": (uncovered / total * 100) if total > 0 else 0.0,
            }
        )
    results.sort(key=lambda r: -r["uncovered_ms"])
    return results


def analyze_db(db_path, process_id=None, device_id=None, start_ns=None, end_ns=None):
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        events = load_events(cursor, process_id, device_id, start_ns, end_ns)
        if not events:
            return {"db": db_path, "error": "no device task events"}

        grouped = {}
        for pid, did, qid, start, end, cat in events:
            grouped.setdefault((pid, did), []).append((pid, did, qid, start, end, cat))

        groups = []
        for (pid, did), group_events in sorted(grouped.items()):
            summary = analyze_group(group_events)
            summary["process_id"] = pid
            summary["device_id"] = did
            summary["top_uncovered_comm"] = comm_by_name(
                cursor, pid, did, start_ns, end_ns
            )[:15]
            groups.append(summary)

        return {
            "db": db_path,
            "window": window_payload(start_ns, end_ns),
            "groups": groups,
        }
    finally:
        conn.close()


def print_text(payload):
    if payload.get("error"):
        print(f"Error: {payload['error']}")
        return
    for group in payload["groups"]:
        print("=" * 90)
        print(
            f"process_id={group['process_id']} device_id={group['device_id']}  "
            f"span={group['span_ms']:,.2f} ms"
        )
        print("=" * 90)

        print("\n[1] Device Timeline Projection")
        print(f"{'Category':<20} {'Time (ms)':>14} {'Ratio':>9}")
        print("-" * 45)
        for cat, time_ms in sorted(group["categories"].items(), key=lambda x: -x[1]):
            print(f"{cat:<20} {time_ms:>14,.2f} {group['category_pct'][cat]:>8.2f}%")
        print(f"{'gap':<20} {group['gap_ms']:>14,.2f} {group['device_gap_pct']:>8.2f}%")

        print("\n[2] Device Stream (Queue) Gap Ratio  <-- key host-overhead indicator")
        print(
            f"{'Queue':>8} {'Span(ms)':>12} {'Busy(ms)':>12} {'Gap(ms)':>12} "
            f"{'Gap%':>8} {'Compute(ms)':>13} {'Main':>6}"
        )
        print("-" * 75)
        for row in group["queues"]:
            main = "*" if row.get("is_main_compute_stream") else ""
            print(
                f"{row['queue_id']:>8} {row['span_ms']:>12,.2f} {row['busy_ms']:>12,.2f} "
                f"{row['gap_ms']:>12,.2f} {row['gap_pct']:>7.2f}% "
                f"{row['compute_ms']:>13,.2f} {main:>6}"
            )
        print(
            f"  -> main compute stream gap ratio: {group['main_stream_gap_pct']:.2f}%  "
            f"(device-level gap ratio: {group['device_gap_pct']:.2f}%)"
        )

        if group["top_uncovered_comm"]:
            print("\n[3] Uncovered Communication (by uncovered time)")
            print(f"{'Kernel':<50} {'Total(ms)':>12} {'Uncov(ms)':>12} {'Uncov%':>8}")
            print("-" * 85)
            for row in group["top_uncovered_comm"]:
                print(
                    f"{row['kernel_name'][:50]:<50} {row['total_ms']:>12,.2f} "
                    f"{row['uncovered_ms']:>12,.2f} {row['uncovered_pct']:>7.1f}%"
                )
        print()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", nargs="+", help="cnperf SQLite DB path(s)")
    parser.add_argument("--process-id", type=int, help="filter processId")
    parser.add_argument("--device-id", type=int, help="filter deviceId")
    add_window_args(parser)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def main():
    args = parse_args()
    profiles = [
        analyze_db(db_path, args.process_id, args.device_id, args.start_ns, args.end_ns)
        for db_path in args.db
    ]
    if args.format == "json":
        print(json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2))
    else:
        for payload in profiles:
            print(f"\n##### {payload['db']} #####")
            print_text(payload)


if __name__ == "__main__":
    main()
