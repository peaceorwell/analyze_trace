#!/usr/bin/env python3
"""Inspect cnperf SQLite compatibility before analysis."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


CORE_TABLES = {
    "string_table": ("ID", "string"),
    "device_task_kernel_data": (
        "processId", "deviceId", "queueId", "correlationId", "nameId", "start", "end", "isComputation"
    ),
    "function_data": ("processId", "threadId", "correlationId", "nameId", "start", "end"),
}

OPTIONAL_TABLES = {
    "device_task_memcpy_data": ("processId", "deviceId", "queueId", "start", "end"),
    "device_task_memset_data": ("processId", "deviceId", "queueId", "start", "end"),
    "device_task_atomic_operation_data": ("processId", "deviceId", "queueId", "start", "end"),
    "device_task_notifier_data": ("processId", "deviceId", "queueId", "start", "end"),
    "Internal_operation_range_data": ("processId", "threadId", "start", "end"),
    "Internal_op_range_relations": ("externalCorrelationId", "correlationId"),
    "device_information": (),
    "meta_information": (),
}

DEVICE_TABLES = (
    "device_task_kernel_data",
    "device_task_memcpy_data",
    "device_task_memset_data",
    "device_task_atomic_operation_data",
    "device_task_notifier_data",
)


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_names(cursor: sqlite3.Cursor) -> set[str]:
    cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    return {row[0] for row in cursor.fetchall()}


def table_columns(cursor: sqlite3.Cursor, table: str) -> list[str]:
    cursor.execute(f"PRAGMA table_info({quote_identifier(table)})")
    return [row[1] for row in cursor.fetchall()]


def table_count(cursor: sqlite3.Cursor, table: str) -> int | None:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}")
        return int(cursor.fetchone()[0])
    except sqlite3.Error:
        return None


def table_range(cursor: sqlite3.Cursor, table: str) -> dict[str, int | None]:
    columns = set(table_columns(cursor, table))
    if not {"start", "end"}.issubset(columns):
        return {"start_ns": None, "end_ns": None, "span_ns": None}
    cursor.execute(f"SELECT MIN(start), MAX(end) FROM {quote_identifier(table)}")
    start, end = cursor.fetchone()
    return {
        "start_ns": start,
        "end_ns": end,
        "span_ns": end - start if start is not None and end is not None else None,
    }


def distinct_pairs(cursor: sqlite3.Cursor, table: str, columns: tuple[str, str]) -> list[dict[str, int]]:
    available = set(table_columns(cursor, table))
    if not set(columns).issubset(available):
        return []
    first, second = columns
    cursor.execute(
        f"SELECT DISTINCT {quote_identifier(first)}, {quote_identifier(second)} "
        f"FROM {quote_identifier(table)} ORDER BY 1, 2"
    )
    return [{first: row[0], second: row[1]} for row in cursor.fetchall()]


def inspect_db(db_path: str) -> dict[str, object]:
    result: dict[str, object] = {
        "db_path": str(Path(db_path).resolve()),
        "exists": Path(db_path).is_file(),
        "read_only": True,
        "timestamp_unit": "ns",
        "errors": [],
        "warnings": [],
    }
    if not result["exists"]:
        result["errors"].append("file_not_found")
        result["compatible"] = False
        return result

    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        result["errors"].append(f"open_failed:{exc}")
        result["compatible"] = False
        return result

    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check(1)")
        result["integrity"] = cursor.fetchone()[0]
        names = table_names(cursor)
        result["table_count"] = len(names)

        schemas = {}
        for table, required in {**CORE_TABLES, **OPTIONAL_TABLES}.items():
            present = table in names
            columns = table_columns(cursor, table) if present else []
            missing = sorted(set(required) - set(columns))
            schemas[table] = {
                "present": present,
                "row_count": table_count(cursor, table) if present else None,
                "columns": columns,
                "missing_required_columns": missing,
                "range": table_range(cursor, table) if present else None,
            }
            if table in CORE_TABLES and not present:
                result["errors"].append(f"missing_core_table:{table}")
            elif table in CORE_TABLES and missing:
                result["errors"].append(f"missing_core_columns:{table}:{','.join(missing)}")
            elif table in OPTIONAL_TABLES and not present:
                result["warnings"].append(f"missing_optional_table:{table}")
        result["schemas"] = schemas

        device_ranges = [schemas[t]["range"] for t in DEVICE_TABLES if schemas.get(t, {}).get("present")]
        starts = [r["start_ns"] for r in device_ranges if r and r["start_ns"] is not None]
        ends = [r["end_ns"] for r in device_ranges if r and r["end_ns"] is not None]
        result["device_range"] = {
            "start_ns": min(starts) if starts else None,
            "end_ns": max(ends) if ends else None,
            "span_ns": max(ends) - min(starts) if starts and ends else None,
        }
        if "function_data" in names:
            result["host_range"] = table_range(cursor, "function_data")

        identity_table = "device_task_kernel_data" if "device_task_kernel_data" in names else None
        result["process_devices"] = (
            distinct_pairs(cursor, identity_table, ("processId", "deviceId")) if identity_table else []
        )
        result["process_threads"] = (
            distinct_pairs(cursor, "function_data", ("processId", "threadId")) if "function_data" in names else []
        )
        result["compatible"] = not result["errors"] and result.get("integrity") == "ok"
        if result["device_range"]["span_ns"] in (None, 0):
            result["warnings"].append("empty_or_zero_device_range")
    except sqlite3.Error as exc:
        result["errors"].append(f"query_failed:{exc}")
        result["compatible"] = False
    finally:
        conn.close()
    return result


def emit_text(payload: dict[str, object]) -> None:
    print(f"DB: {payload['db_path']}")
    print(f"Compatible: {payload.get('compatible', False)}")
    print(f"Integrity: {payload.get('integrity', 'unavailable')}")
    print(f"Timestamp unit: {payload.get('timestamp_unit')}")
    print(f"Process/devices: {payload.get('process_devices', [])}")
    print(f"Device range: {payload.get('device_range')}")
    if payload.get("errors"):
        print("Errors:", *payload["errors"], sep="\n- ")
    if payload.get("warnings"):
        print("Warnings:", *payload["warnings"], sep="\n- ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight a cnperf-compatible SQLite DB")
    parser.add_argument("db_path")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    payload = inspect_db(args.db_path)
    if args.format == "json":
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        emit_text(payload)
    raise SystemExit(0 if payload.get("compatible") else 2)


if __name__ == "__main__":
    main()
