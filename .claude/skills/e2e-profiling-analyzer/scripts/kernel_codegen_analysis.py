#!/usr/bin/env python3
"""Build deterministic Triton and compile/fusion evidence from a cnperf DB."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_TRITON_PATTERN = r"triton"
DEFAULT_FUSION_PATTERN = r"(?:fused|fusion|inductor|torch[_ -]?compiled|compiled[_ -]?region)"


def table_exists(cursor: sqlite3.Cursor, name: str) -> bool:
    cursor.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,))
    return cursor.fetchone() is not None


def table_columns(cursor: sqlite3.Cursor, name: str) -> set[str]:
    cursor.execute(f'PRAGMA table_info("{name}")')
    return {row[1] for row in cursor.fetchall()}


def load_string_map(cursor: sqlite3.Cursor) -> dict[int, str]:
    if not table_exists(cursor, "string_table"):
        return {}
    cursor.execute("SELECT ID,string FROM string_table")
    return {row[0]: row[1] for row in cursor.fetchall()}


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def parse_extra(raw: object) -> dict[str, object]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def kernel_kind(name: str) -> str:
    lowered = name.lower()
    if "persistent" in lowered:
        return "persistent"
    if re.search(r"(?:^|[_:.])(red|reduce|reduction|softmax|norm)(?:[_:.]|$)", lowered):
        return "reduction"
    if re.search(r"(?:^|[_:.])(mm|gemm|matmul|bmm)(?:[_:.]|$)", lowered):
        return "matmul"
    if re.search(r"(?:^|[_:.])(poi|pointwise|elementwise)(?:[_:.]|$)", lowered):
        return "pointwise"
    return "other"


def make_where(
    columns: set[str],
    process_id: int | None,
    device_id: int | None,
    start_ns: int | None,
    end_ns: int | None,
) -> tuple[str, list[int]]:
    clauses = ["isComputation=1"] if "isComputation" in columns else []
    params: list[int] = []
    if process_id is not None and "processId" in columns:
        clauses.append("processId=?")
        params.append(process_id)
    if device_id is not None and "deviceId" in columns:
        clauses.append("deviceId=?")
        params.append(device_id)
    if start_ns is not None:
        clauses.append("end>?")
        params.append(start_ns)
    if end_ns is not None:
        clauses.append("start<?")
        params.append(end_ns)
    return (" WHERE " + " AND ".join(clauses) if clauses else "", params)


def optional_column(columns: set[str], name: str, fallback: str = "NULL") -> str:
    return f'"{name}"' if name in columns else fallback


def load_compute_kernels(
    cursor: sqlite3.Cursor,
    names: dict[int, str],
    process_id: int | None,
    device_id: int | None,
    start_ns: int | None,
    end_ns: int | None,
    triton_pattern: re.Pattern[str],
    fusion_pattern: re.Pattern[str],
    tiny_threshold_ns: int,
) -> tuple[list[dict[str, object]], list[str]]:
    warnings: list[str] = []
    if not table_exists(cursor, "device_task_kernel_data"):
        return [], ["device_task_kernel_data unavailable"]
    columns = table_columns(cursor, "device_task_kernel_data")
    required = {"processId", "deviceId", "queueId", "correlationId", "nameId", "start", "end"}
    missing = sorted(required - columns)
    if missing:
        return [], [f"device_task_kernel_data missing columns: {','.join(missing)}"]

    where, params = make_where(columns, process_id, device_id, start_ns, end_ns)
    extra_column = optional_column(columns, "extra", "'{}'")
    cursor.execute(
        "SELECT processId,deviceId,queueId,correlationId,nameId,start,end,"
        f"{optional_column(columns, 'class')},{optional_column(columns, 'dimX')},"
        f"{optional_column(columns, 'dimY')},{optional_column(columns, 'dimZ')},"
        f"{extra_column} "
        "FROM device_task_kernel_data" + where + " ORDER BY processId,deviceId,queueId,start,end",
        params,
    )
    kernels = []
    for pid, did, queue, corr, name_id, start, end, klass, dim_x, dim_y, dim_z, extra_raw in cursor.fetchall():
        clipped_start = max(start, start_ns) if start_ns is not None else start
        clipped_end = min(end, end_ns) if end_ns is not None else end
        if clipped_end <= clipped_start:
            continue
        name = names.get(name_id, f"nameId={name_id}")
        extra = parse_extra(extra_raw)
        searchable = name + " " + json.dumps(extra, sort_keys=True, ensure_ascii=False)
        explicit_metadata = any("triton" in str(key).lower() or "triton" in str(value).lower() for key, value in extra.items())
        triton_name_signal = bool(triton_pattern.search(name))
        kernels.append(
            {
                "process_id": pid,
                "device_id": did,
                "queue_id": queue,
                "correlation_id": corr,
                "name": name,
                "start_ns": clipped_start,
                "end_ns": clipped_end,
                "duration_ns": clipped_end - clipped_start,
                "class": klass,
                "dim_x": dim_x,
                "dim_y": dim_y,
                "dim_z": dim_z,
                "kind_signal": kernel_kind(name),
                "triton_signal": triton_name_signal or explicit_metadata,
                "triton_attribution": "confirmed_metadata" if explicit_metadata else (
                    "probable_name" if triton_name_signal else "none"
                ),
                "fusion_signal": bool(fusion_pattern.search(searchable)),
                "tiny": clipped_end - clipped_start <= tiny_threshold_ns,
                "metadata_keys": sorted(str(key) for key in extra),
            }
        )
    if not kernels:
        warnings.append("no compute kernels in selected scope")
    return kernels, warnings


def summarize_kernels(kernels: list[dict[str, object]], limit: int | None = None) -> list[dict[str, object]]:
    groups: dict[str, dict[str, object]] = defaultdict(
        lambda: {"durations": [], "configs": set(), "metadata_keys": set(), "kinds": set(), "triton": set()}
    )
    for kernel in kernels:
        item = groups[str(kernel["name"])]
        item["durations"].append(int(kernel["duration_ns"]))
        item["configs"].add((kernel["class"], kernel["dim_x"], kernel["dim_y"], kernel["dim_z"]))
        item["metadata_keys"].update(kernel["metadata_keys"])
        item["kinds"].add(kernel["kind_signal"])
        item["triton"].add(kernel["triton_attribution"])
    rows = []
    for name, item in groups.items():
        durations = item["durations"]
        total = sum(durations)
        rows.append(
            {
                "name": name,
                "count": len(durations),
                "total_ns": total,
                "avg_ns": total / len(durations),
                "p50_ns": percentile(durations, 0.50),
                "p90_ns": percentile(durations, 0.90),
                "max_ns": max(durations),
                "tiny_count": 0,
                "kind_signals": sorted(item["kinds"]),
                "triton_attribution": sorted(item["triton"]),
                "launch_configurations": [
                    {"class": config[0], "dim_x": config[1], "dim_y": config[2], "dim_z": config[3]}
                    for config in sorted(item["configs"], key=lambda value: tuple(-1 if part is None else part for part in value))
                ],
                "metadata_keys": sorted(item["metadata_keys"]),
            }
        )
    rows.sort(key=lambda row: (-row["total_ns"], row["name"]))
    return rows[:limit] if limit is not None else rows


def add_tiny_counts(rows: list[dict[str, object]], kernels: list[dict[str, object]]) -> None:
    tiny_by_name: dict[str, list[int]] = defaultdict(list)
    for kernel in kernels:
        if kernel["tiny"]:
            tiny_by_name[str(kernel["name"])].append(int(kernel["duration_ns"]))
    for row in rows:
        values = tiny_by_name.get(str(row["name"]), [])
        row["tiny_count"] = len(values)
        row["tiny_total_ns"] = sum(values)


def adjacent_pairs(kernels: list[dict[str, object]], limit: int = 30) -> list[dict[str, object]]:
    by_queue: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for kernel in kernels:
        by_queue[(int(kernel["process_id"]), int(kernel["device_id"]), int(kernel["queue_id"]))].append(kernel)
    pairs: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"count": 0, "gap_ns": 0, "overlap_ns": 0, "both_tiny_count": 0}
    )
    for queue_kernels in by_queue.values():
        queue_kernels.sort(key=lambda item: (item["start_ns"], item["end_ns"]))
        for left, right in zip(queue_kernels, queue_kernels[1:]):
            key = (str(left["name"]), str(right["name"]))
            item = pairs[key]
            delta = int(right["start_ns"]) - int(left["end_ns"])
            item["count"] += 1
            item["gap_ns"] += max(0, delta)
            item["overlap_ns"] += max(0, -delta)
            item["both_tiny_count"] += int(bool(left["tiny"] and right["tiny"]))
    rows = [{"left": key[0], "right": key[1], **value} for key, value in pairs.items()]
    rows.sort(key=lambda row: (-row["count"], -row["gap_ns"], row["left"], row["right"]))
    return rows[:limit]


def operator_kernel_mapping(
    cursor: sqlite3.Cursor,
    names: dict[int, str],
    kernels: list[dict[str, object]],
    fusion_pattern: re.Pattern[str],
) -> tuple[dict[str, object], list[str]]:
    warnings: list[str] = []
    if not table_exists(cursor, "Internal_operation_range_data") or not table_exists(
        cursor, "Internal_op_range_relations"
    ):
        return {"coverage_pct": 0, "operators": [], "compiled_region_operators": []}, [
            "operator range or relation table unavailable; op-to-kernel mapping blocked"
        ]
    op_columns = table_columns(cursor, "Internal_operation_range_data")
    relation_columns = table_columns(cursor, "Internal_op_range_relations")
    if not {"processId", "extraId", "nameId"}.issubset(op_columns) or not {
        "externalCorrelationId", "correlationId"
    }.issubset(relation_columns):
        return {"coverage_pct": 0, "operators": [], "compiled_region_operators": []}, [
            "operator range or relation columns unavailable; op-to-kernel mapping blocked"
        ]

    cursor.execute("SELECT externalCorrelationId,correlationId FROM Internal_op_range_relations")
    relations: dict[int, set[int]] = defaultdict(set)
    for external, correlation in cursor.fetchall():
        relations[external].add(correlation)

    kernels_by_corr: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for kernel in kernels:
        kernels_by_corr[(int(kernel["process_id"]), int(kernel["correlation_id"]))].append(kernel)

    cursor.execute(
        "SELECT processId,extraId,nameId,"
        f"{optional_column(op_columns, 'threadId')},{optional_column(op_columns, 'start')},"
        f"{optional_column(op_columns, 'end')} FROM Internal_operation_range_data"
    )
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "occurrences": 0,
            "kernels_per_occurrence": [],
            "kernel_names": set(),
            "triton_launches": 0,
            "fusion_signal_launches": 0,
        }
    )
    linked_kernel_ids: set[tuple[object, ...]] = set()
    for pid, extra_id, name_id, _thread_id, _start, _end in cursor.fetchall():
        matched: dict[tuple[object, ...], dict[str, object]] = {}
        for correlation in relations.get(extra_id, set()):
            for kernel in kernels_by_corr.get((pid, correlation), []):
                identity = (
                    kernel["process_id"], kernel["device_id"], kernel["queue_id"],
                    kernel["correlation_id"], kernel["start_ns"], kernel["end_ns"], kernel["name"],
                )
                matched[identity] = kernel
                linked_kernel_ids.add(identity)
        if not matched:
            continue
        op_name = names.get(name_id, f"nameId={name_id}")
        item = grouped[op_name]
        item["occurrences"] += 1
        item["kernels_per_occurrence"].append(len(matched))
        item["kernel_names"].update(str(kernel["name"]) for kernel in matched.values())
        item["triton_launches"] += sum(bool(kernel["triton_signal"]) for kernel in matched.values())
        item["fusion_signal_launches"] += sum(bool(kernel["fusion_signal"]) for kernel in matched.values())

    rows = []
    for name, item in grouped.items():
        counts = item["kernels_per_occurrence"]
        rows.append(
            {
                "operator_name": name,
                "occurrences_with_mapped_kernels": item["occurrences"],
                "mapped_kernel_launches": sum(counts),
                "avg_kernels_per_occurrence": sum(counts) / len(counts),
                "max_kernels_per_occurrence": max(counts),
                "distinct_kernel_names": sorted(item["kernel_names"]),
                "triton_launches": item["triton_launches"],
                "fusion_signal_launches": item["fusion_signal_launches"],
            }
        )
    rows.sort(key=lambda row: (-row["mapped_kernel_launches"], row["operator_name"]))
    compiled = [row for row in rows if fusion_pattern.search(row["operator_name"])]
    coverage = len(linked_kernel_ids) / len(kernels) * 100 if kernels else 0
    if coverage < 50 and kernels:
        warnings.append(f"low op-to-kernel mapping coverage: {coverage:.2f}%")
    return {"coverage_pct": coverage, "operators": rows, "compiled_region_operators": compiled}, warnings


def analyze_codegen(
    db_path: str,
    process_id: int | None = None,
    device_id: int | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    tiny_threshold_us: float = 20.0,
    triton_regex: str = DEFAULT_TRITON_PATTERN,
    fusion_regex: str = DEFAULT_FUSION_PATTERN,
) -> dict[str, object]:
    triton_pattern = re.compile(triton_regex, re.IGNORECASE)
    fusion_pattern = re.compile(fusion_regex, re.IGNORECASE)
    tiny_threshold_ns = int(tiny_threshold_us * 1000)
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        cursor = connection.cursor()
        names = load_string_map(cursor)
        kernels, warnings = load_compute_kernels(
            cursor, names, process_id, device_id, start_ns, end_ns,
            triton_pattern, fusion_pattern, tiny_threshold_ns,
        )
        triton_kernels = [kernel for kernel in kernels if kernel["triton_signal"]]
        fusion_kernels = [kernel for kernel in kernels if kernel["fusion_signal"]]
        tiny_kernels = [kernel for kernel in kernels if kernel["tiny"]]
        all_rows = summarize_kernels(kernels)
        triton_rows = summarize_kernels(triton_kernels)
        fusion_rows = summarize_kernels(fusion_kernels)
        add_tiny_counts(all_rows, kernels)
        add_tiny_counts(triton_rows, triton_kernels)
        add_tiny_counts(fusion_rows, fusion_kernels)
        mapping, mapping_warnings = operator_kernel_mapping(cursor, names, kernels, fusion_pattern)
        warnings.extend(mapping_warnings)
        starts = [int(kernel["start_ns"]) for kernel in kernels]
        ends = [int(kernel["end_ns"]) for kernel in kernels]
        span_ns = max(ends) - min(starts) if starts and ends else 0
        total_ns = sum(int(kernel["duration_ns"]) for kernel in kernels)
        return {
            "schema_version": "1.0",
            "db_path": str(Path(db_path).resolve()),
            "timestamp_unit": "ns",
            "filters": {
                "process_id": process_id,
                "device_id": device_id,
                "start_ns": start_ns,
                "end_ns": end_ns,
            },
            "heuristics": {
                "triton_regex": triton_regex,
                "fusion_regex": fusion_regex,
                "tiny_threshold_us": tiny_threshold_us,
                "duration_accounting": "aggregate kernel duration; may overlap across queues",
            },
            "totals": {
                "compute_kernel_count": len(kernels),
                "compute_kernel_total_ns": total_ns,
                "kernel_span_ns": span_ns,
                "launches_per_active_ms": len(kernels) / (span_ns / 1e6) if span_ns else 0,
                "triton_signal_count": len(triton_kernels),
                "triton_signal_total_ns": sum(int(kernel["duration_ns"]) for kernel in triton_kernels),
                "triton_count_share_pct": len(triton_kernels) / len(kernels) * 100 if kernels else 0,
                "triton_duration_share_pct": (
                    sum(int(kernel["duration_ns"]) for kernel in triton_kernels) / total_ns * 100 if total_ns else 0
                ),
                "fusion_signal_count": len(fusion_kernels),
                "fusion_signal_total_ns": sum(int(kernel["duration_ns"]) for kernel in fusion_kernels),
                "tiny_kernel_count": len(tiny_kernels),
                "tiny_kernel_total_ns": sum(int(kernel["duration_ns"]) for kernel in tiny_kernels),
                "tiny_count_share_pct": len(tiny_kernels) / len(kernels) * 100 if kernels else 0,
                "tiny_duration_share_pct": (
                    sum(int(kernel["duration_ns"]) for kernel in tiny_kernels) / total_ns * 100 if total_ns else 0
                ),
            },
            "all_compute_kernels": all_rows,
            "triton_kernels": triton_rows,
            "fusion_signal_kernels": fusion_rows,
            "adjacent_kernel_pairs": adjacent_pairs(kernels),
            "operator_kernel_mapping": mapping,
            "warnings": warnings,
        }
    finally:
        connection.close()


def emit_text(payload: dict[str, object]) -> None:
    totals = payload["totals"]
    print(f"DB: {payload['db_path']}")
    print(
        "compute_count={compute_kernel_count} triton_count={triton_signal_count} "
        "fusion_signal_count={fusion_signal_count} tiny_count={tiny_kernel_count}".format(**totals)
    )
    print(
        f"triton_duration_share={totals['triton_duration_share_pct']:.2f}% "
        f"tiny_duration_share={totals['tiny_duration_share_pct']:.2f}%"
    )
    for row in payload["triton_kernels"][:20]:
        print(
            f"TRITON {row['name']} count={row['count']} total_ms={row['total_ns']/1e6:.3f} "
            f"p90_us={row['p90_ns']/1e3:.3f} configs={len(row['launch_configurations'])}"
        )
    for warning in payload["warnings"]:
        print(f"WARNING: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Triton and compile/fusion kernel evidence")
    parser.add_argument("db_path")
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--end-ns", type=int)
    parser.add_argument("--tiny-threshold-us", type=float, default=20.0)
    parser.add_argument("--triton-regex", default=DEFAULT_TRITON_PATTERN)
    parser.add_argument("--fusion-regex", default=DEFAULT_FUSION_PATTERN)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    try:
        payload = analyze_codegen(
            args.db_path,
            args.process_id,
            args.device_id,
            args.start_ns,
            args.end_ns,
            args.tiny_threshold_us,
            args.triton_regex,
            args.fusion_regex,
        )
    except re.error as exc:
        parser.error(f"invalid regex: {exc}")
    if args.format == "json":
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        emit_text(payload)


if __name__ == "__main__":
    main()
