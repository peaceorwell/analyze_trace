#!/usr/bin/env python3
"""Validate the lightweight findings contract without third-party packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STATUSES = {"confirmed", "probable", "possible", "unsupported", "blocked"}
TOP_LEVEL = {"schema_version", "agent", "scope", "findings", "artifacts"}
FINDING_FIELDS = {
    "id", "status", "cause", "summary", "metrics", "evidence", "counter_evidence",
    "affected_scope", "estimated_impact_ms", "confidence", "overlap_group", "follow_up",
}
METRIC_FIELDS = {"name", "value", "unit", "scope", "source"}


def validate(payload):
    errors = []
    missing = TOP_LEVEL - set(payload)
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
    if payload.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")
    if not isinstance(payload.get("agent"), str) or not payload.get("agent"):
        errors.append("agent must be a non-empty string")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        errors.append("scope must be an object")
    else:
        for field in ("dbs", "process_ids", "device_ids", "window", "limitations"):
            if field not in scope:
                errors.append(f"scope missing field: {field}")
        for field in ("dbs", "process_ids", "device_ids", "limitations"):
            if field in scope and not isinstance(scope[field], list):
                errors.append(f"scope.{field} must be an array")
        window = scope.get("window")
        if not isinstance(window, dict):
            errors.append("scope.window must be an object")
        elif not {"start_ns", "end_ns", "basis"}.issubset(window):
            errors.append("scope.window requires start_ns, end_ns, and basis")
    if not isinstance(payload.get("artifacts"), list):
        errors.append("artifacts must be an array")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be an array")
        return errors
    ids = set()
    for index, finding in enumerate(findings):
        label = f"findings[{index}]"
        if not isinstance(finding, dict):
            errors.append(f"{label} must be an object")
            continue
        missing = FINDING_FIELDS - set(finding)
        if missing:
            errors.append(f"{label} missing fields: {sorted(missing)}")
        for field in ("cause", "summary"):
            if not isinstance(finding.get(field), str) or not finding.get(field):
                errors.append(f"{label}.{field} must be a non-empty string")
        finding_id = finding.get("id")
        if not isinstance(finding_id, str) or not finding_id:
            errors.append(f"{label}.id must be a non-empty string")
        elif finding_id in ids:
            errors.append(f"duplicate finding id: {finding_id}")
        ids.add(finding_id)
        if finding.get("status") not in STATUSES:
            errors.append(f"{label}.status must be one of {sorted(STATUSES)}")
        confidence = finding.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"{label}.confidence must be a number in [0,1]")
        impact = finding.get("estimated_impact_ms")
        if impact is not None and (not isinstance(impact, (int, float)) or impact < 0):
            errors.append(f"{label}.estimated_impact_ms must be null or a non-negative number")
        if not isinstance(finding.get("overlap_group"), str) or not finding.get("overlap_group"):
            errors.append(f"{label}.overlap_group must be a non-empty string")
        for field in ("evidence", "counter_evidence", "affected_scope", "follow_up", "metrics"):
            if not isinstance(finding.get(field), list):
                errors.append(f"{label}.{field} must be an array")
        for metric_index, metric in enumerate(finding.get("metrics", []) if isinstance(finding.get("metrics"), list) else []):
            if not isinstance(metric, dict):
                errors.append(f"{label}.metrics[{metric_index}] must be an object")
                continue
            metric_missing = METRIC_FIELDS - set(metric)
            if metric_missing:
                errors.append(f"{label}.metrics[{metric_index}] missing fields: {sorted(metric_missing)}")
            if not metric.get("unit") or not metric.get("source"):
                errors.append(f"{label}.metrics[{metric_index}] requires unit and source")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate E2E profiling findings JSON")
    parser.add_argument("findings_json")
    args = parser.parse_args()
    path = Path(args.findings_json)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        parser.error(f"cannot read JSON: {exc}")
    errors = validate(payload)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(2)
    print(f"Valid findings: {path} ({len(payload['findings'])} findings)")


if __name__ == "__main__":
    main()
