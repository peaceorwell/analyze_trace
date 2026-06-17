#!/usr/bin/env python3
"""Gap summary for compute-kernel gaps."""

import argparse
import json
import sqlite3
import sys

from query_common import (
    get_invoke_info,
    get_name,
    load_string_map,
    simplify_task_name,
)


def build_where_clause(process_id=None, device_id=None, start_time=None, end_time=None):
    where_clauses = ["isComputation = 1"]
    if start_time is not None:
        where_clauses.append(f"start >= {start_time}")
    if end_time is not None:
        where_clauses.append(f"start <= {end_time}")
    if device_id is not None:
        where_clauses.append(f"deviceId = {device_id}")
    if process_id is not None:
        where_clauses.append(f"processId = {process_id}")
    return "WHERE " + " AND ".join(where_clauses)


def find_blocking_event(cursor, process_id, device_id, queue_id, gap_start, gap_end):
    cursor.execute(
        """
        SELECT sync_type, start, end, correlationId, notifierId, extra
        FROM (
            SELECT 'notifier' AS sync_type, start, end, correlationId, notifierId, extra
            FROM device_task_notifier_data
            WHERE processId = :pid AND deviceId = :did AND queueId = :qid
              AND type = 0 AND start >= :gap_start AND start <= :gap_end
            UNION ALL
            SELECT 'atomic' AS sync_type, start, end, correlationId, NULL AS notifierId, extra
            FROM device_task_atomic_operation_data
            WHERE processId = :pid AND deviceId = :did AND queueId = :qid
              AND start >= :gap_start AND start <= :gap_end
            UNION ALL
            SELECT 'memcpy' AS sync_type, start, end, correlationId, NULL AS notifierId, extra
            FROM device_task_memcpy_data
            WHERE processId = :pid AND deviceId = :did AND queueId = :qid
              AND isAsync = 1 AND start >= :gap_start AND start <= :gap_end
        ) combined
        ORDER BY start DESC, end DESC
        LIMIT 1
        """,
        {
            "pid": process_id,
            "did": device_id,
            "qid": queue_id,
            "gap_start": gap_start,
            "gap_end": gap_end,
        },
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "sync_type": row[0],
        "start": row[1],
        "end": row[2],
        "correlation_id": row[3],
        "notifier_id": row[4],
    }


def build_initial_stats():
    return {
        "mini_gap": {"count": 0, "time": 0},
        "host_blocking": {"count": 0, "time": 0, "max": 0},
        "notifier_blocking": {"count": 0, "time": 0, "max": 0},
        "atomicOp_blocking": {"count": 0, "time": 0, "max": 0},
        "memcpy_blocking": {"count": 0, "time": 0, "max": 0},
        "other": {"count": 0, "time": 0, "max": 0},
    }


def record_gap_stat(stats, reason, duration):
    stats[reason]["count"] += 1
    stats[reason]["time"] += duration
    if "max" in stats[reason]:
        stats[reason]["max"] = max(stats[reason]["max"], duration)


def merge_stats(stats_by_group):
    total_stats = build_initial_stats()
    for stats in stats_by_group.values():
        for reason, value in stats.items():
            total_stats[reason]["count"] += value["count"]
            total_stats[reason]["time"] += value["time"]
            if "max" in total_stats[reason]:
                total_stats[reason]["max"] = max(total_stats[reason]["max"], value["max"])
    return total_stats


def format_stats(stats):
    total_count = sum(value["count"] for value in stats.values())
    total_time = sum(value["time"] for value in stats.values())
    output_stats = {
        **stats,
        "total": {"count": total_count, "time": total_time},
    }

    return {
        reason: (
            {
                "count": value["count"],
                "time_ms": value["time"] / 1e6,
                "share_pct": (value["time"] / total_time * 100 if total_time > 0 else 0),
            }
            if reason in ("mini_gap", "total")
            else {
                "count": value["count"],
                "time_ms": value["time"] / 1e6,
                "avg_ms": (value["time"] / value["count"] / 1e6 if value["count"] else 0),
                "max_ms": value["max"] / 1e6,
                "share_pct": (value["time"] / total_time * 100 if total_time > 0 else 0),
            }
        )
        for reason, value in output_stats.items()
    }


def build_summary(stats_by_group):
    total_stats = merge_stats(stats_by_group)

    return {
        "gap_type": "compute_coverage_gap",
        "method": (
            "Merge overlapping compute kernels per process/device before measuring "
            "idle intervals. This avoids treating queue-local stalls hidden by other "
            "compute kernels as exposed compute gaps."
        ),
        "stats": format_stats(total_stats),
        "stats_by_group": [
            {
                "process_id": process_id,
                "device_id": device_id,
                "stats": format_stats(stats),
            }
            for (process_id, device_id), stats in sorted(stats_by_group.items())
        ],
    }


def analyze_device_gaps(
    cursor,
    process_id=None,
    device_id=None,
    start_time=None,
    end_time=None,
    gap_threshold=100000,
    invoke_threshold=100000,
):
    where_clause = build_where_clause(
        process_id=process_id,
        device_id=device_id,
        start_time=start_time,
        end_time=end_time,
    )

    kernel_cursor = cursor.connection.cursor()
    kernel_cursor.execute(
        f"""
        SELECT processId, deviceId, queueId, start, end, correlationId, nameId
        FROM device_task_kernel_data
        {where_clause}
        ORDER BY start
        """
    )

    stats_by_group = {}
    results = []
    coverage_by_group = {}

    for row in kernel_cursor:
        row_process_id, row_device_id, queue_id, start, end, corr_id, name_id = row
        group_key = (row_process_id, row_device_id)
        group_stats = stats_by_group.setdefault(group_key, build_initial_stats())
        coverage = coverage_by_group.get(group_key)

        if coverage is None:
            coverage_by_group[group_key] = {
                "end": end,
                "end_kernel": (queue_id, end, corr_id, name_id),
            }
            continue

        if start <= coverage["end"]:
            if end > coverage["end"]:
                coverage["end"] = end
                coverage["end_kernel"] = (queue_id, end, corr_id, name_id)
            continue

        prev_queue_id, prev_end, prev_corr_id, prev_name_id = coverage["end_kernel"]
        gap = start - prev_end
        coverage["end"] = end
        coverage["end_kernel"] = (queue_id, end, corr_id, name_id)

        if gap < gap_threshold:
            record_gap_stat(group_stats, "mini_gap", gap)
            continue

        item = {
            "gap_type": "compute_coverage_gap",
            "process_id": row_process_id,
            "device_id": row_device_id,
            "reason": "other",
            "start": prev_end,
            "duration": gap,
            "prev": {
                "queue_id": prev_queue_id,
                "corr_id": prev_corr_id,
                "name_id": prev_name_id,
            },
            "curr": {
                "queue_id": queue_id,
                "corr_id": corr_id,
                "name_id": name_id,
            },
        }

        invoke = get_invoke_info(cursor, corr_id)
        dispatch_delay = start - invoke["end"] if invoke else None

        if dispatch_delay is not None and dispatch_delay <= invoke_threshold:
            item["reason"] = "host_blocking"
            results.append(item)
            record_gap_stat(group_stats, item["reason"], gap)
            continue

        blocker = find_blocking_event(
            cursor,
            row_process_id,
            row_device_id,
            queue_id,
            prev_end,
            start,
        )
        if blocker:
            if blocker["sync_type"] == "notifier":
                item["reason"] = "notifier_blocking"
            elif blocker["sync_type"] == "atomic":
                item["reason"] = "atomicOp_blocking"
            elif blocker["sync_type"] == "memcpy":
                item["reason"] = "memcpy_blocking"

        results.append(item)
        record_gap_stat(group_stats, item["reason"], gap)

    results.sort(key=lambda item: -item["duration"])
    summary = build_summary(stats_by_group)
    return summary, results


def fit(value, width):
    text = str(value)
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "~"


def group_results_by_process_device(results):
    groups = {}
    for item in results:
        key = (item["process_id"], item["device_id"])
        groups.setdefault(key, []).append(item)
    return sorted(groups.items())


def ordered_stats_rows(stats):
    rows = [
        item
        for item in sorted(stats.items(), key=lambda item: -item[1]["time_ms"])
        if item[0] != "total"
    ]
    rows.append(("total", stats["total"]))
    return rows


def emit_json_output(db_path, params, summary_payload, results, out):
    payload = dict(summary_payload)
    payload["db_path"] = db_path
    payload["params"] = params
    payload["non_mini_gaps"] = [
        {
            "process_id": process_id,
            "device_id": device_id,
            "gaps": [
                {
                    "reason": item["reason"],
                    "gap_type": item.get("gap_type", "compute_coverage_gap"),
                    "start": item["start"],
                    "duration": item["duration"],
                    "prev": item["prev"],
                    "curr": item["curr"],
                }
                for item in items
            ],
        }
        for (process_id, device_id), items in group_results_by_process_device(results)
    ]
    json.dump(payload, out, indent=2)
    print(file=out)


def emit_text_output(cursor, summary_payload, results, out):
    string_map = load_string_map(cursor)
    print(f"\n{'=' * 80}", file=out)
    print("Compute Coverage Gap Summary", file=out)
    print(f"{'=' * 80}", file=out)
    print(summary_payload.get("method", ""), file=out)
    print(file=out)
    print(
        f"{'pid':<8} {'dev':<5} {'reason':<20} {'count':<8} {'time(ms)':<12} "
        f"{'avg(ms)':<10} {'max(ms)':<10} {'share':<8}",
        file=out,
    )
    print("-" * 98, file=out)
    for group in summary_payload["stats_by_group"]:
        for reason, value in ordered_stats_rows(group["stats"]):
            share = f"{value['share_pct']:.2f}%"
            avg_ms = f"{value['avg_ms']:.4f}" if "avg_ms" in value else "-"
            max_ms = f"{value['max_ms']:.4f}" if "max_ms" in value else "-"
            print(
                f"{group['process_id']:<8} {group['device_id']:<5} {reason:<20} "
                f"{value['count']:<8} {value['time_ms']:<12.2f} "
                f"{avg_ms:<10} {max_ms:<10} {share:<8}",
                file=out,
            )

    print(f"\n{'=' * 120}", file=out)
    print("Non-mini gap list", file=out)
    print(f"{'=' * 120}", file=out)
    columns = [
        ("gap_ms", 10),
        ("start_ms", 10),
        ("reason", 18),
        ("prev_corr", 10),
        ("next_corr", 10),
        ("gap_name", 64),
    ]
    if not results:
        print("(none)", file=out)
        return

    for (process_id, device_id), items in group_results_by_process_device(results):
        print(file=out)
        print(
            f"process_id={process_id} device_id={device_id}: {len(items)} gaps",
            file=out,
        )
        print(" ".join(f"{name:<{width}}" for name, width in columns), file=out)
        print("-" * 110, file=out)
        for item in items:
            prev_name = simplify_task_name(get_name(string_map, item["prev"]["name_id"]))
            next_name = simplify_task_name(get_name(string_map, item["curr"]["name_id"]))
            row = [
                f"{item['duration']/1e6:.4f}",
                f"{item['start']/1e6:.2f}",
                item["reason"],
                item["prev"]["corr_id"],
                item["curr"]["corr_id"],
                f"{prev_name} -> {next_name}",
            ]
            print(
                " ".join(f"{fit(value, width):<{width}}" for value, (_, width) in zip(row, columns)),
                file=out,
            )


def main():
    parser = argparse.ArgumentParser(description="Summary of non-mini compute gaps")
    parser.add_argument("db_path", help="数据库路径")
    parser.add_argument("--process-id", type=int, help="进程号过滤")
    parser.add_argument("--device-id", type=int, help="设备号过滤")
    parser.add_argument("--start-time", type=float, help="开始时间(us)")
    parser.add_argument("--end-time", type=float, help="结束时间(us)")
    parser.add_argument("--gap-threshold", type=float, default=100, help="gap阈值(us), 默认100")
    parser.add_argument("--invoke-threshold", type=float, default=100, help="下发不及时阈值(us), 默认100")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="输出格式")
    args = parser.parse_args()

    start_time_ns = int(args.start_time * 1000) if args.start_time else None
    end_time_ns = int(args.end_time * 1000) if args.end_time else None
    gap_threshold_ns = int(args.gap_threshold * 1000)
    invoke_threshold_ns = int(args.invoke_threshold * 1000)

    conn = sqlite3.connect(args.db_path)
    try:
        cursor = conn.cursor()
        payload, results = analyze_device_gaps(
            cursor,
            process_id=args.process_id,
            device_id=args.device_id,
            start_time=start_time_ns,
            end_time=end_time_ns,
            gap_threshold=gap_threshold_ns,
            invoke_threshold=invoke_threshold_ns,
        )
        output_params = {
            "process_id": args.process_id,
            "device_id": args.device_id,
            "start_time_ns": start_time_ns,
            "end_time_ns": end_time_ns,
            "gap_threshold_ns": gap_threshold_ns,
            "invoke_threshold_ns": invoke_threshold_ns,
        }

        if args.format == "json":
            emit_json_output(args.db_path, output_params, payload, results, sys.stdout)
            return

        emit_text_output(cursor, payload, results, sys.stdout)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
