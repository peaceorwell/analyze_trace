# Evidence Contract

Use this contract for every teammate result and the final machine-readable report. The goal is to make claims auditable, comparable, and mergeable.

## Required top-level shape

```json
{
  "schema_version": "1.0",
  "agent": "e2e-triton-kernel-analyst",
  "scope": {
    "dbs": ["/absolute/path/cnperf_data.db"],
    "process_ids": [123],
    "device_ids": [0],
    "window": {"start_ns": 0, "end_ns": 1000000000, "basis": "stable"},
    "limitations": []
  },
  "findings": [],
  "artifacts": ["03_triton_kernel/triton_kernel_efficiency.json"]
}
```

## Required finding shape

```json
{
  "id": "triton-001",
  "status": "probable",
  "cause": "A Triton pointwise kernel has a material long tail",
  "summary": "The explicitly named Triton kernel has stable launch count but a high p90-to-median ratio.",
  "metrics": [
    {
      "name": "kernel_p90_duration",
      "value": 1.42,
      "unit": "ms",
      "scope": "device=0 kernel=observed_name",
      "source": "03_triton_kernel/triton_tables.json#/kernels/0"
    }
  ],
  "evidence": ["raw file and JSON pointer or compact SQL result"],
  "counter_evidence": ["no device performance counters were captured"],
  "affected_scope": ["device=0", "kernel=observed_name"],
  "estimated_impact_ms": 120.0,
  "impact": {
    "observed_cost_ms": 180.0,
    "critical_path_contribution_ms": 120.0,
    "recoverable_upper_bound_ms": 120.0,
    "basis": "stable window, exposed interval union"
  },
  "confidence": 0.78,
  "overlap_group": "triton-fusion-critical-path",
  "follow_up": ["capture generated Triton source and repeat the stable-step window"]
}
```

## Semantics

- `status`: one of `confirmed`, `probable`, `possible`, `unsupported`, `blocked`.
- `confidence`: number in `[0, 1]`; confidence is evidence strength, not impact.
- `estimated_impact_ms`: number or `null`. Use only measured exposed time or skew. Do not invent speedup.
- `impact` is optional for backward compatibility but required in new branch/final outputs. Keep `observed_cost_ms`, `critical_path_contribution_ms`, and `recoverable_upper_bound_ms` distinct; each is a non-negative number or `null`, and `basis` explains the scope. `estimated_impact_ms` mirrors the recoverable upper bound when that bound is available.
- `overlap_group`: non-empty string grouping benefits that may overlap. Use `independent:<id>` only with evidence of independence.
- `metrics`: always include value, unit, scope, and source.
- `evidence`: point to immutable raw or structured artifacts. Do not cite another agent's prose as primary evidence.
- `counter_evidence`: actively list facts that weaken the claim; use an empty array only after an explicit search found none.
- `follow_up`: name the minimum experiment, DB, generated artifact, or query needed to raise confidence.

## Freeform analyst

The freeform analyst may introduce new causal categories and cross-layer hypotheses. It must still use this schema. Novelty never substitutes for evidence.

## Auditor

The auditor may emit findings whose cause is a quality issue, such as `window-mismatch`, `unit-ambiguity`, `double-counting`, or `unsupported-causality`. Reference the finding IDs being challenged in `affected_scope`.

Audit and final findings should preserve `audit_disposition` as one of `supported_primary`, `supported_contributor`, `refuted`, `insufficient`, or `duplicate`. Read `hypothesis_verification.md` for disposition and report-inclusion rules.

## Final merge

- Preserve source finding IDs.
- Merge duplicate causal paths and list contributing agents.
- Do not sum `estimated_impact_ms` within the same `overlap_group`.
- If agents disagree, preserve both claims until the audit resolves or marks them unresolved.
- Put only `supported_primary` and `supported_contributor` in prioritized user-facing conclusions; route `insufficient` to uncertainties and keep `refuted`/`duplicate` in audit artifacts.
