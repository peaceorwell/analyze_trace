# Candidate Hypotheses And Independent Verification

Use this reference after the immutable baseline is ready and before final synthesis. Separate candidate discovery, causal analysis, independent audit, and report inclusion.

## Candidate discovery

Generate candidates from independent anomaly signals, not from a fixed requirement to fill every category. Each candidate must record:

- stable ID and type;
- process/device and selected window;
- observed symptom and source artifact;
- observed cost or delta with units and basis;
- the causal question still to be tested;
- the smallest branch/query that can distinguish it.

Do not state a root cause at discovery time. Keep Device-Gap, memcpy, host, communication exposure, compute hotspots, fusion, and Triton efficiency separate unless evidence already proves one causal chain. Prefer at most 12 material candidates; preserve why lower-priority candidates were not pursued.

## Analysis

Turn one candidate into one falsifiable causal claim. Recompute its key metrics, cite immutable evidence, define affected scope, and actively state counter-evidence and refutation conditions. If available evidence cannot make the candidate testable, mark it `blocked` instead of switching to another hypothesis.

Use a targeted raw timeline query only after the candidate identifies a process/device (or rank) and a step/request or absolute time window. Aggregated evidence comes before raw event lists.

## Independent audit disposition

The auditor tests the claim rather than extending it. Assign one disposition:

- `supported_primary`: evidence supports the mechanism and it explains a major part of the target metric.
- `supported_contributor`: evidence supports the mechanism but it explains only part of the target metric.
- `refuted`: relevant evidence coverage is sufficient and contradicts the claim, or the symptom is not exposed on the target critical path.
- `insufficient`: the discriminating evidence was not captured or is not comparable.
- `duplicate`: the claim counts the same causal path or exposed interval as another supported finding.

Use `refuted` only when coverage is sufficient; otherwise use `insufficient`. Static source or configuration can support a mechanism but cannot replace runtime timing evidence.

Map disposition to the existing evidence status without weakening the distinction: supported findings normally have `confirmed` or `probable`; `refuted` normally has `unsupported`; `insufficient` normally has `blocked` or `possible`. Preserve `audit_disposition` separately in audit/final JSON.

## Impact contract

Keep three quantities separate:

1. `observed_cost_ms`: measured inclusive/aggregate cost or signed A/B delta. It may include overlap and is not automatically recoverable.
2. `critical_path_contribution_ms`: the measured exposed portion attributable to this mechanism on the target window.
3. `recoverable_upper_bound_ms`: the maximum plausible E2E benefit after removing overlap, unavoidable work, and known dependencies.

Each value needs a scope and basis; use `null` when unavailable. Do not substitute one for another. Benefits in the same `overlap_group` are non-additive. Use `independent:<id>` only when evidence shows independent intervals.

## Final inclusion

- Put only `supported_primary` and `supported_contributor` findings in prioritized conclusions and actions.
- Put `refuted` and `duplicate` findings in audit artifacts, not user-facing bottleneck claims.
- Put `insufficient` findings and their minimal evidence request under uncertainty.
- Merge duplicate causal chains before ranking. Rank supported findings by critical-path contribution/recoverable bound, confidence, implementation scope, and validation cost.
- Recommendations must match the audited disposition and include one falsifiable controlled experiment.
