# Agent Team Workflow

## Contents

1. Team lifecycle
2. Shared evidence
3. Task graph
4. Communication protocol
5. Artifact ownership
6. Quality gates
7. Final synthesis

## Team lifecycle

The lead alone creates, expands, shuts down, and cleans up the team. Teammates never create nested teams or edit runtime team configuration.

Use project-scoped teammate definitions from `.claude/agents/`. Spawn the evidence builder first. After baseline completion, spawn material specialists and the freeform analyst. Spawn the auditor only after branch artifacts exist.

If team creation is unavailable, execute the same roles serially and preserve the same artifact layout.

## Shared evidence

Treat these as immutable after baseline completion:

- Input manifest and converted DBs.
- Preflight reports.
- Process/device selection.
- Common time windows and their basis.
- Baseline JSON/log outputs.

If a branch needs a different window, record it as a separate scoped analysis; do not silently replace the common window.

## Task graph

```text
baseline-evidence
  ├─ compute-analysis (conditional)
  ├─ triton-kernel-analysis (conditional)
  ├─ compile-fusion-analysis (conditional)
  ├─ gap-host-analysis (conditional)
  ├─ noncompute-analysis (conditional)
  └─ freeform-analysis (always)

all completed branches
  └─ evidence-audit
      └─ lead final synthesis
```

When the user asks for every angle, create all specialist tasks even if baseline evidence is small. Let the specialist return `unsupported` with evidence instead of skipping.

Branch ownership:

- Compute: `effective-compute-breakdown`.
- Triton kernel: `triton-kernel-efficiency`, generated source, and optional `mlu-triton-optimize` output.
- Compile/fusion: `compile-segmentation` plus `triton-fusion-coverage`.
- Gap/host: `compute-gap-root-cause` and optional `host-window-subphase`.
- Non-compute: `ordinary-non-compute-root-cause`.
- Freeform: categories outside the predefined branches.

There is no communication or multi-rank specialist in this workflow. Keep communication events as opaque baseline categories only.

## Communication protocol

- Send hypotheses, finding IDs, and artifact paths; do not send long copied logs.
- A teammate that discovers a cross-domain hypothesis messages the relevant teammate and the lead.
- The recipient tests the hypothesis independently and records agreement or counter-evidence.
- Only the lead changes shared scope, common windows, or task dependencies.

## Artifact ownership

Use `TEAM_DIR` as the only teammate output root:

- Evidence builder: `01_baseline/`
- Compute: `02_compute/`
- Triton kernel: `03_triton_kernel/`
- Compile/fusion: `04_compile_fusion/`
- Gap/host: `05_gap_host/`
- Non-compute: `06_noncompute/`
- Freeform: `07_freeform/`
- Auditor: `08_audit/`

No two agents write the same file. Only the lead writes root `report.md`, `report.json`, and `evidence_summary.md`.

## Quality gates

Before marking a branch complete:

1. Produce raw structured output and a compact Markdown report.
2. Produce `findings.json` using `evidence_contract.md`.
3. Run `validate_findings.py` successfully.
4. State missing tables, codegen artifacts, stable windows, and conversion limitations.
5. Keep benefit estimates non-additive by assigning overlap groups.

The auditor checks:

- DB and `string_table` identity.
- Units and timestamp basis.
- Window alignment and warmup contamination.
- Process/device/thread mixing.
- Interval overlap and double counting.
- Correlation and notifier identity.
- Triton attribution strength: metadata/source evidence versus name-only signal.
- Fusion granularity claims versus heuristic-only indicators.
- Evidence versus narrative strength.
- Benefit overlap.

## Final synthesis

Use the audit as a gate, not as another vote. Prefer raw evidence over majority agreement. A repeated unsupported claim remains unsupported.

Prioritize actions by measured impact, confidence, implementation scope, and validation cost. Include one first action and one confirming measurement for every high-priority recommendation. Follow the main skill's Final Report Contract; keep detailed branch evidence under `TEAM_DIR`.
