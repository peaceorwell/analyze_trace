#!/usr/bin/env python3
"""
Device Timeline Projection and Uncovered Communication Analysis
使用扫描线算法计算 Device 时间线投影，同时分析未掩盖通信开销
"""

import sqlite3
import sys

from query_common import load_string_map


CATEGORY_PRIORITY = (
    'compute_kernel',
    'comm_kernel',
    'memcpy',
    'memset',
    'atomic',
)


def merge_intervals(intervals):
    """Merge overlapping intervals so overlap is counted once."""
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]

def analyze_device_timeline(db_path):
    """使用扫描线算法计算 Device 时间线投影"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    events = []

    cursor.execute("SELECT start, end, nameId, isComputation FROM device_task_kernel_data")
    for start, end, nameId, is_comp in cursor.fetchall():
        cat = 'compute_kernel' if is_comp == 1 else 'comm_kernel'
        events.append((start, end, cat))

    cursor.execute("SELECT start, end FROM device_task_memcpy_data")
    for start, end in cursor.fetchall():
        events.append((start, end, 'memcpy'))

    cursor.execute("SELECT start, end FROM device_task_memset_data")
    for start, end in cursor.fetchall():
        events.append((start, end, 'memset'))

    cursor.execute("SELECT start, end FROM device_task_atomic_operation_data")
    for start, end in cursor.fetchall():
        events.append((start, end, 'atomic'))

    string_map = load_string_map(cursor)

    compute_kernels = [(r[0], r[1]) for r in cursor.execute("SELECT start, end FROM device_task_kernel_data WHERE isComputation = 1").fetchall()]
    comm_kernels_raw = cursor.execute("SELECT start, end, nameId FROM device_task_kernel_data WHERE isComputation = 0").fetchall()

    points = []
    for start, end, cat in events:
        points.append((start, cat, 1))
        points.append((end, cat, -1))

    points.sort(key=lambda x: (x[0], x[2]))

    categories = {}
    current = {}
    prev_time = None

    for time, cat, delta in points:
        if prev_time is not None and time > prev_time and current:
            main_cat = 'other'
            for candidate in CATEGORY_PRIORITY:
                if candidate in current:
                    main_cat = candidate
                    break
            categories[main_cat] = categories.get(main_cat, 0) + time - prev_time

        current[cat] = current.get(cat, 0) + delta
        if current[cat] <= 0:
            del current[cat]
        prev_time = time

    total_span = max(e[1] for e in events) - min(e[0] for e in events)

    def calculate_overlap(comm_start, comm_end, compute_kernels):
        occluded = 0
        overlap_ranges = []
        for c_start, c_end in compute_kernels:
            if c_end <= comm_start or c_start >= comm_end:
                continue
            overlap_start = max(comm_start, c_start)
            overlap_end = min(comm_end, c_end)
            overlap_ranges.append((overlap_start, overlap_end))
        for overlap_start, overlap_end in merge_intervals(overlap_ranges):
            occluded += overlap_end - overlap_start
        total = comm_end - comm_start
        return occluded, total - occluded

    comm_by_name = {}
    for start, end, name_id in comm_kernels_raw:
        if name_id not in comm_by_name:
            comm_by_name[name_id] = []
        comm_by_name[name_id].append((start, end))

    comm_results = []
    for name_id, kernels in comm_by_name.items():
        name = string_map.get(name_id, f"nameId={name_id}")
        total_time = sum(e[1] - e[0] for e in kernels)
        total_occluded = 0
        for start, end in kernels:
            occluded, unoccluded = calculate_overlap(start, end, compute_kernels)
            total_occluded += occluded
        unoccluded = total_time - total_occluded
        unoccluded_pct = unoccluded / total_time * 100 if total_time > 0 else 0
        comm_results.append((name, total_time, total_occluded, unoccluded, unoccluded_pct))

    comm_results.sort(key=lambda x: -x[3])

    conn.close()

    return categories, total_span, comm_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python device_timeline.py <cnperf_db>")
        sys.exit(1)

    db_path = sys.argv[1]
    categories, total_span, comm_results = analyze_device_timeline(db_path)

    print("=" * 80)
    print("1. Device Timeline Projection")
    print("=" * 80)
    print(f"{'Category':<30} {'Time (ms)':<15} {'Ratio':<10}")
    print("-" * 60)
    for cat, time_ns in sorted(categories.items(), key=lambda x: -x[1]):
        time_ms = time_ns / 1e6
        pct = time_ns / total_span * 100
        print(f"{cat:<30} {time_ms:>12,.2f}  {pct:>7.2f}%")

    gap_time = total_span - sum(categories.values())
    print(f"{'gap':<30} {gap_time/1e6:>12,.2f}  {gap_time/total_span*100:>7.2f}%")
    print("-" * 60)

    print()
    print("=" * 80)
    print("2. Uncovered Communication Overhead")
    print("=" * 80)

    total_unoccluded = sum(r[3] for r in comm_results)
    print(f"{'Type':<60} {'Total(ms)':<12} {'Uncovered(ms)':<14} {'Uncovered%':<10} {'Share%':<10}")
    print("-" * 110)

    for name, total, occluded, unoccluded, pct in comm_results:
        share = unoccluded / total_unoccluded * 100 if total_unoccluded > 0 else 0
        print(f"{name:<60} {total/1e6:>10,.2f}  {unoccluded/1e6:>12,.2f}  {pct:>8.1f}%  {share:>8.1f}%")

    print("-" * 110)
    total_comm_time = sum(r[1] for r in comm_results)
    total_uncovered_pct = total_unoccluded / total_comm_time * 100 if total_comm_time > 0 else 0
    print(f"{'TOTAL':<60} {total_comm_time/1e6:>10,.2f}  {total_unoccluded/1e6:>12,.2f}  {total_uncovered_pct:>8.1f}%  {'100.0%':>8}")
