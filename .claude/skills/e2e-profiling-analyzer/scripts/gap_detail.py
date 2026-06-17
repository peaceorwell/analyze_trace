#!/usr/bin/env python3
"""Detailed dependency-chain analysis for a single non-mini compute gap."""

import argparse
import json
import sqlite3
import sys

from query_common import (
    get_host_context_stack,
    get_invoke_info,
    get_name,
    load_string_map,
    parse_extra,
    simplify_task_name,
)


QUEUE_LOOKBACK_LIMIT = 50


def build_event(kind, start, end, correlation_id, label, extra=None):
    item = {
        "kind": kind,
        "start": start,
        "end": end,
        "correlation_id": correlation_id,
        "label": label,
    }
    if extra:
        item.update(extra)
    return item


def build_kernel_event(kernel, string_map):
    return build_event(
        "compute_kernel",
        kernel["start"],
        kernel["end"],
        kernel["corr_id"],
        simplify_task_name(get_name(string_map, kernel["name_id"])),
        {
            "process_id": kernel["process_id"],
            "device_id": kernel["device_id"],
            "queue_id": kernel["queue_id"],
        },
    )


def get_matching_notifier_place(cursor, event):
    if (
        event["kind"] != "notifier_wait"
        or event.get("notifier_id") is None
        or event.get("unique_val") is None
    ):
        return None

    cursor.execute(
        """
        SELECT queueId, start, end, correlationId
        FROM device_task_notifier_data
        WHERE processId = :pid
          AND deviceId = :did
          AND type = 1
          AND notifierId = :nid
          AND json_extract(extra, '$.unique_val') = :uv
        ORDER BY start DESC
        LIMIT 1
        """,
        {
            "pid": event["process_id"],
            "did": event["device_id"],
            "nid": event["notifier_id"],
            "uv": event["unique_val"],
        },
    )
    row = cursor.fetchone()
    if not row:
        return None

    queue_id, start, end, correlation_id = row
    return build_event(
        "notifier_place",
        start,
        end,
        correlation_id,
        f"notifier place id={event['notifier_id']}",
        {
            "process_id": event["process_id"],
            "device_id": event["device_id"],
            "queue_id": queue_id,
            "notifier_id": event["notifier_id"],
            "unique_val": event.get("unique_val"),
        },
    )


def candidate_order(cursor, candidate, current_event, current_invoke):
    candidate_invoke_end = None
    correlation_id = candidate.get("correlation_id")
    if correlation_id is not None:
        invoke = get_invoke_info(cursor, correlation_id)
        if invoke:
            candidate_invoke_end = invoke["end"]

    if candidate["end"] == current_event["start"] and current_invoke and candidate_invoke_end:
        if candidate_invoke_end > current_invoke["end"]:
            return None

    invoke_order = candidate_invoke_end if candidate_invoke_end is not None else -1
    return (candidate["end"], candidate["start"], invoke_order)


def is_same_event(candidate, current_event):
    current_corr = current_event.get("correlation_id")
    candidate_corr = candidate.get("correlation_id")
    if current_corr is not None and candidate_corr == current_corr:
        return True
    return event_identity(candidate) == event_identity(current_event)


def get_last_queue_event_before(cursor, current_event, string_map):
    candidates = []
    process_id = current_event["process_id"]
    device_id = current_event["device_id"]
    queue_id = current_event["queue_id"]
    timestamp = current_event["start"]
    params = {"pid": process_id, "did": device_id, "qid": queue_id, "ts": timestamp}
    event_scope = {"process_id": process_id, "device_id": device_id, "queue_id": queue_id}

    cursor.execute(
        """
        SELECT start, end, correlationId, nameId, isComputation
        FROM device_task_kernel_data
        WHERE processId = :pid AND deviceId = :did AND queueId = :qid AND end <= :ts
        ORDER BY end DESC
        LIMIT :limit
        """,
        {**params, "limit": QUEUE_LOOKBACK_LIMIT},
    )
    for start, end, correlation_id, name_id, is_computation in cursor.fetchall():
        candidates.append(
            build_event(
                "compute_kernel" if is_computation == 1 else "comm_kernel",
                start,
                end,
                correlation_id,
                simplify_task_name(get_name(string_map, name_id)),
                event_scope,
            )
        )

    cursor.execute(
        """
        SELECT start, end, correlationId, bytes, isAsync
        FROM device_task_memcpy_data
        WHERE processId = :pid AND deviceId = :did AND queueId = :qid AND end <= :ts
        ORDER BY end DESC
        LIMIT :limit
        """,
        {**params, "limit": QUEUE_LOOKBACK_LIMIT},
    )
    for start, end, correlation_id, num_bytes, is_async in cursor.fetchall():
        candidates.append(
            build_event(
                "memcpy",
                start,
                end,
                correlation_id,
                f"{num_bytes} bytes",
                {**event_scope, "is_async": is_async},
            )
        )

    cursor.execute(
        """
        SELECT start, end, correlationId, type, subType, value
        FROM device_task_atomic_operation_data
        WHERE processId = :pid AND deviceId = :did AND queueId = :qid AND end <= :ts
        ORDER BY end DESC
        LIMIT :limit
        """,
        {**params, "limit": QUEUE_LOOKBACK_LIMIT},
    )
    for start, end, correlation_id, op_type, sub_type, value in cursor.fetchall():
        candidates.append(
            build_event(
                "atomic",
                start,
                end,
                correlation_id,
                f"type={op_type}, subType={sub_type}, value={value}",
                event_scope,
            )
        )

    cursor.execute(
        """
        SELECT start, end, correlationId, notifierId, type, extra
        FROM device_task_notifier_data
        WHERE processId = :pid AND deviceId = :did AND queueId = :qid AND end <= :ts
        ORDER BY end DESC
        LIMIT :limit
        """,
        {**params, "limit": QUEUE_LOOKBACK_LIMIT},
    )
    for start, end, correlation_id, notifier_id, event_type, extra_str in cursor.fetchall():
        extra = parse_extra(extra_str)
        label = f"notifier {'wait' if event_type == 0 else 'place'} id={notifier_id}"
        event = build_event(
            "notifier_wait" if event_type == 0 else "notifier_place",
            start,
            end,
            correlation_id,
            label,
            {
                "process_id": process_id,
                "device_id": device_id,
                "notifier_id": notifier_id,
                "type": event_type,
                "unique_val": extra.get("unique_val"),
                "queue_id": queue_id,
            },
        )
        candidates.append(event)

    if not candidates:
        return None

    current_invoke = None
    current_corr = current_event.get("correlation_id")
    if current_corr is not None:
        current_invoke = get_invoke_info(cursor, current_corr)

    ordered_candidates = []
    for candidate in candidates:
        if is_same_event(candidate, current_event):
            continue
        order_key = candidate_order(cursor, candidate, current_event, current_invoke)
        if order_key is None:
            continue
        ordered_candidates.append((order_key, candidate))

    if not ordered_candidates:
        return None
    ordered_candidates.sort(key=lambda item: item[0], reverse=True)
    return ordered_candidates[0][1]


def build_event_invoke_hints(cursor, event):
    hints = []
    correlation_id = event.get("correlation_id")
    if correlation_id is not None:
        invoke = get_invoke_info(cursor, correlation_id)
        if invoke:
            hints.append(invoke)

    return hints


def blocking_reason_from_predecessor(predecessor):
    predecessor_kind = predecessor.get("kind")
    if predecessor_kind in {"notifier_wait", "notifier_place"}:
        return "notifier_blocking"
    if predecessor_kind == "memcpy":
        return "memcpy_blocking"
    if predecessor_kind == "atomic":
        return "atomic_blocking"
    if predecessor_kind in {"compute_kernel", "comm_kernel"}:
        return "kernel_blocking"
    return "pre_task_blocking"


def is_device_execution_event(event):
    return event["kind"] in {"compute_kernel", "comm_kernel", "memcpy", "atomic"}


def get_gap_impact(event, gap_start, gap_end):
    if not is_device_execution_event(event):
        return None
    overlap = max(0, min(event["end"], gap_end) - max(event["start"], gap_start))
    if overlap <= 0:
        return None
    return {
        "duration": overlap,
        "pct": (overlap / (gap_end - gap_start) * 100 if gap_end > gap_start else 0),
    }


def format_event_meta(event, gap_start, gap_end):
    duration_ms = (event["end"] - event["start"]) / 1e6
    meta = f"start={event['start']/1e6:.2f}ms (+{duration_ms:.4f}ms)"
    if event.get("correlation_id") is not None:
        meta += f" corr={event['correlation_id']}"
    if event.get("queue_id") is not None:
        meta += f" q={event['queue_id']}"
    if event.get("tid") is not None:
        meta += f" tid={event['tid']}"
    if event.get("unique_val") is not None:
        meta += f" unique_val={event['unique_val']}"
    impact = get_gap_impact(event, gap_start, gap_end)
    if impact:
        meta += f" gap_impact={impact['duration']/1e6:.4f}ms/{impact['pct']:.2f}%"
    return meta


def format_call_stack_frame(frame):
    correlation = f" corr={frame['correlation_id']}" if frame.get("correlation_id") is not None else ""
    return (
        f"[{frame['source']}] {frame['name']} start={frame['start']/1e6:.2f}ms "
        f"(+{frame['duration']/1e6:.4f}ms){correlation} tid={frame['tid']}"
    )


def build_host_call_stack(cursor, invoke, string_map):
    return [
        {
            "source": frame["source"],
            "name": simplify_task_name(frame.get("name") or get_name(string_map, frame["name_id"])),
            "start": frame["start"],
            "end": frame["end"],
            "duration": frame["end"] - frame["start"],
            "correlation_id": frame["correlation_id"],
            "tid": frame["tid"],
        }
        for frame in get_host_context_stack(cursor, invoke)
    ]


def build_host_event(cursor, invoke, string_map):
    return build_event(
        "host_invoke",
        invoke["start"],
        invoke["end"],
        None,
        simplify_task_name(get_name(string_map, invoke["name_id"])),
        {
            "tid": invoke["tid"],
            "call_stack": build_host_call_stack(cursor, invoke, string_map),
        },
    )


def resolve_event_state(cursor, event, invoke_threshold, string_map):
    event_start = event["start"]
    notifier_place_gap_threshold = 10000

    for invoke in build_event_invoke_hints(cursor, event):
        if invoke["end"] >= event_start - invoke_threshold:
            return {
                "reason": "host_blocking",
                "invoke": invoke,
            }

    predecessor = get_last_queue_event_before(
        cursor,
        event,
        string_map,
    )

    if predecessor:
        if event_start - predecessor["end"] < notifier_place_gap_threshold:
            return {
                "reason": blocking_reason_from_predecessor(predecessor),
                "predecessor": predecessor,
            }

    if event["kind"] == "notifier_wait":
        place_event = get_matching_notifier_place(cursor, event)
        if place_event:
            place_gap = event_start - place_event["end"]
            if place_gap < notifier_place_gap_threshold:
                return {
                    "reason": "notifier_syncing",
                    "predecessor": place_event,
                }

    return {
        "reason": "unknown",
    }


def event_identity(event):
    return (
        event.get("kind"),
        event.get("correlation_id"),
        event.get("queue_id"),
        event.get("start"),
        event.get("end"),
        event.get("notifier_id"),
        event.get("unique_val"),
    )

def trace_event_dependency(cursor, start_event, gap_start, invoke_threshold, string_map):
    chain = []
    current = start_event
    visited = set()

    while True:
        current_identity = event_identity(current)
        if current_identity in visited:
            chain.append(
                {
                    "event": current,
                    "reason": "dependency_cycle_detected",
                }
            )
            break
        visited.add(current_identity)

        step = resolve_event_state(cursor, current, invoke_threshold, string_map)
        predecessor = step.get("predecessor")
        if step["reason"] == "host_blocking":
            predecessor = build_host_event(cursor, step["invoke"], string_map)
        chain.append(
            {
                "event": current,
                "reason": step["reason"],
            }
        )

        if step["reason"] == "host_blocking":
            chain.append(
                {
                    "event": predecessor,
                    "reason": "host_invoke",
                }
            )
            break
        if predecessor is None:
            break
        if predecessor["start"] < gap_start:
            chain.append(
                {
                    "event": predecessor,
                    "reason": "out_of_range",
                }
            )
            break

        current = predecessor

    return chain


def enrich_gap_detail(cursor, next_kernel, gap_start, invoke_threshold, string_map):
    next_kernel_event = build_kernel_event(next_kernel, string_map)
    return trace_event_dependency(
        cursor,
        next_kernel_event,
        gap_start,
        invoke_threshold,
        string_map,
    )


def get_compute_kernel_by_corr(cursor, corr_id):
    cursor.execute(
        """
        SELECT processId, deviceId, queueId, start, end, correlationId, nameId
        FROM device_task_kernel_data
        WHERE correlationId = ? AND isComputation = 1
        LIMIT 1
        """,
        (corr_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "process_id": row[0],
        "device_id": row[1],
        "queue_id": row[2],
        "start": row[3],
        "end": row[4],
        "corr_id": row[5],
        "name_id": row[6],
    }


def print_gap_detail(prev_kernel, next_kernel, gap, main_chain, string_map):
    gap_start = prev_kernel["end"]
    gap_end = next_kernel["start"]
    gap_ms = gap / 1e6
    prev_name = simplify_task_name(get_name(string_map, prev_kernel["name_id"]))
    curr_name = simplify_task_name(get_name(string_map, next_kernel["name_id"]))
    prev_duration_ms = (prev_kernel["end"] - prev_kernel["start"]) / 1e6
    curr_duration_ms = (next_kernel["end"] - next_kernel["start"]) / 1e6

    print(
        f"\n--- Gap: {prev_kernel['end']/1e6:.2f}ms "
        f"(+{gap_ms:.2f}ms), pid={next_kernel['process_id']} dev={next_kernel['device_id']} ---"
    )
    print(
        f"  prev_kernel: {prev_name} start={prev_kernel['start']/1e6:.2f}ms "
        f"(+{prev_duration_ms:.4f}ms) corr={prev_kernel['corr_id']} q={prev_kernel['queue_id']}"
    )
    print(
        f"  next_kernel: {curr_name} start={next_kernel['start']/1e6:.2f}ms "
        f"(+{curr_duration_ms:.4f}ms) corr={next_kernel['corr_id']} q={next_kernel['queue_id']}"
    )

    print("  [main_dependency_chain]")
    for step_index, step in enumerate(main_chain, start=1):
        event = step["event"]
        next_step = main_chain[step_index] if step_index < len(main_chain) else None
        dependency_gap_ms = "-"
        if next_step:
            gap_ms = (event["start"] - next_step["event"]["end"]) / 1e6
            if abs(gap_ms) < 0.00005:
                gap_ms = 0.0
            dependency_gap_ms = f"{gap_ms:.4f}ms"

        meta = format_event_meta(event, gap_start, gap_end)
        event_name = event["label"][:88]
        if len(event_name) <= 40:
            print(f"    {event_name} {meta}")
        else:
            print(f"    {event_name}")
            print(f"      {meta}")
        if event.get("call_stack"):
            print("      call_stack:")
            for frame in event["call_stack"]:
                print(f"        {format_call_stack_frame(frame)}")

        if next_step:
            relation = format_relation_label(step, next_step)
            print(f"    |-> {relation} (+{dependency_gap_ms})")


def build_gap_detail_payload(prev_kernel, next_kernel, gap, main_chain, string_map):
    gap_start = prev_kernel["end"]
    gap_end = next_kernel["start"]
    payload = {
        "process_id": next_kernel["process_id"],
        "device_id": next_kernel["device_id"],
        "prev_corr": prev_kernel["corr_id"],
        "next_corr": next_kernel["corr_id"],
        "prev_kernel": {
            "name": simplify_task_name(get_name(string_map, prev_kernel["name_id"])),
            "start_ms": prev_kernel["start"] / 1e6,
            "duration_ms": (prev_kernel["end"] - prev_kernel["start"]) / 1e6,
            "correlation_id": prev_kernel["corr_id"],
            "queue_id": prev_kernel["queue_id"],
        },
        "next_kernel": {
            "name": simplify_task_name(get_name(string_map, next_kernel["name_id"])),
            "start_ms": next_kernel["start"] / 1e6,
            "duration_ms": (next_kernel["end"] - next_kernel["start"]) / 1e6,
            "correlation_id": next_kernel["corr_id"],
            "queue_id": next_kernel["queue_id"],
        },
        "gap": {
            "start_ms": prev_kernel["end"] / 1e6,
            "duration_ms": gap / 1e6,
        },
        "main_dependency_chain": [],
    }
    for step in main_chain:
        event = step["event"]
        impact = get_gap_impact(event, gap_start, gap_end)
        dependency_gap_ms = None
        step_index = len(payload["main_dependency_chain"])
        next_step = main_chain[step_index + 1] if step_index + 1 < len(main_chain) else None
        item = {
            "name": event["label"],
            "kind": event["kind"],
            "start_ms": event["start"] / 1e6,
            "duration_ms": (event["end"] - event["start"]) / 1e6,
            "correlation_id": event.get("correlation_id"),
            "queue_id": event.get("queue_id"),
            "unique_val": event.get("unique_val"),
            "reason": step["reason"],
        }
        if event.get("call_stack"):
            item["call_stack"] = [
                {
                    "source": frame["source"],
                    "name": frame["name"],
                    "start_ms": frame["start"] / 1e6,
                    "duration_ms": frame["duration"] / 1e6,
                    "correlation_id": frame["correlation_id"],
                    "tid": frame["tid"],
                }
                for frame in event["call_stack"]
            ]
        if impact:
            item["gap_impact_ms"] = impact["duration"] / 1e6
            item["gap_impact_pct"] = impact["pct"]
        if next_step:
            dependency_gap_ms = (event["start"] - next_step["event"]["end"]) / 1e6
            if abs(dependency_gap_ms) < 0.00005:
                dependency_gap_ms = 0.0
        item["dependency_gap_ms"] = dependency_gap_ms
        payload["main_dependency_chain"].append(item)
    return payload


def format_relation_label(step, next_step):
    reason = step["reason"]
    next_kind = next_step["event"].get("kind")
    if reason == "notifier_syncing":
        return "sync with notifier"
    if next_kind == "host_invoke":
        return "block by host"
    if next_kind in {"notifier_wait", "notifier_place"}:
        return "block by notifier"
    if next_kind == "memcpy":
        return "block by memcpy"
    if next_kind == "atomic":
        return "block by atomic"
    if next_kind in {"compute_kernel", "comm_kernel"}:
        return "block by kernel"
    if reason == "notifier_blocking":
        return "block by notifier"
    if reason == "kernel_blocking":
        return "block by kernel"
    if reason == "memcpy_blocking":
        return "block by memcpy"
    if reason == "atomic_blocking":
        return "block by atomic"
    if reason == "host_blocking":
        return "block by host"
    if reason == "host_invoke":
        return "host invoke"
    return "block by pre task"


def main():
    parser = argparse.ArgumentParser(description="Detailed gap dependency-chain analysis for one gap")
    parser.add_argument("db_path", help="数据库路径")
    parser.add_argument("--prev-corr", type=int, required=True, help="前一个compute kernel的correlationId")
    parser.add_argument("--next-corr", type=int, required=True, help="后一个compute kernel的correlationId")
    parser.add_argument("--invoke-threshold", type=float, default=100, help="下发不及时阈值(us), 默认100")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="输出格式; json 更适合AI或脚本消费",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        cursor = conn.cursor()
        string_map = load_string_map(cursor)

        prev_kernel = get_compute_kernel_by_corr(cursor, args.prev_corr)
        if prev_kernel is None:
            parser.error(f"prev compute kernel not found: prev_corr={args.prev_corr}")

        next_kernel = get_compute_kernel_by_corr(cursor, args.next_corr)
        if next_kernel is None:
            parser.error(f"next compute kernel not found: next_corr={args.next_corr}")

        prev_scope = (prev_kernel["process_id"], prev_kernel["device_id"])
        next_scope = (next_kernel["process_id"], next_kernel["device_id"])
        if prev_scope != next_scope:
            parser.error(
                "prev/next kernels are not in the same process/device: "
                f"prev={prev_scope}, next={next_scope}"
            )

        gap = next_kernel["start"] - prev_kernel["end"]
        if gap <= 0:
            parser.error(
                f"invalid gap for prev_corr={args.prev_corr}, next_corr={args.next_corr}: "
                f"gap={gap/1000:.2f}us"
            )
        invoke_threshold = int(args.invoke_threshold * 1000)

        main_chain = enrich_gap_detail(
            cursor,
            next_kernel,
            prev_kernel["end"],
            invoke_threshold,
            string_map,
        )
        if args.format == "json":
            json.dump(
                build_gap_detail_payload(prev_kernel, next_kernel, gap, main_chain, string_map),
                sys.stdout,
                indent=2,
            )
            print()
            return

        print(f"\n{'=' * 150}")
        print(f"Gap 详情: prev_corr={args.prev_corr}, next_corr={args.next_corr}")
        print(f"{'=' * 150}")
        print_gap_detail(prev_kernel, next_kernel, gap, main_chain, string_map)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
