---
name: e2e-profiling-analyzer
description: Analyze single-rank cnperf SQLite databases and torch profiler Chrome trace JSON/JSON.GZ files with a Claude Code Agent Team for end-to-end MLU training or inference bottlenecks. Use when the user asks to inspect `cnperf_data*.db` or `.pt.trace.json`/`.json.gz` traces, diagnose effective compute, Triton kernels, torch.compile graph segmentation and fusion granularity, compute gaps, host/device synchronization, memcpy or ordinary non-compute work, unconstrained hypotheses, adversarial evidence review, and one auditable final report.
---

# E2E Profiling Analyzer

Analyze `cnperf` SQLite DBs from the viewpoint that compute kernels are effective device utilization. If the input is a torch profiler Chrome trace JSON/JSON.GZ file, first convert it to a cnperf-compatible SQLite DB, then analyze the converted DB exactly like native cnperf data.

## Run Checklist

Copy this checklist into the analysis directory as `checklist.md` and keep it updated. Do not drop a
step because the input looks small; record a step as skipped with its reason instead.

- [ ] Mode resolved (`automatic-final` or `interactive-phased`); inputs normalized to DB paths
- [ ] `references/profiling_concepts.md` and `references/pytorch_performance_playbook.md` loaded
- [ ] Measurement Validity Gate applied; capability ledger written from `references/capability_degradation.md`
- [ ] Phase 1 baseline scripts run; `phase1_report.md` and validated `findings.json` written
- [ ] Phase 2 branches selected by measured impact and executed by their owning agents
- [ ] Every teammate `findings.json` passed `scripts/validate_findings.py`
- [ ] Auditor assigned one `audit_disposition` to every candidate
- [ ] `report.md` and `report.json` written, then `scripts/check_report.py` exits 0

## Mode Selection

Use `automatic-final` immediately when any of these are true:

- The prompt says `automatic-final`, `自动最终报告`, `不要向用户追问`, or that this is a Web/server-side/background analysis.
- Environment variables such as `TRACE_AI_JOB_ID`, `TRACE_AI_ANALYSIS_DIR`, `TRACE_AI_REPORT_PATH`, or `TRACE_AI_TRACE_A` are present.
- The user asks for an end-to-end report instead of a phased investigation.

Use `interactive-phased` only when the user explicitly wants to choose branches step by step, or has already named a specific branch or branches to run.

In `automatic-final`, never ask follow-up questions. If required evidence is missing, continue with available evidence and record the missing input under `Open Questions`.

If the prompt is an environment diagnostic or smoke test and asks to reply only `OK`, reply exactly `OK` and do not run tools or load references.

Mode meanings:

- `automatic-final`: build one Phase 1 evidence baseline, run independent specialist agents in parallel when possible, audit their findings, and produce stage reports plus the final synthesis.
- `interactive-phased`: write a report after each stage, stop after Phase 1 for branch selection, then run the branch or branches selected by the user.

Default to `automatic-final` when the user has not specified a mode or a branch. Only use `interactive-phased` when the user explicitly requested step-by-step branch selection, or already named a specific branch or branches to run after Phase 1.

Do not assume the bottleneck is Triton, fusion, a specific kernel family, TCDP, or a known synchronization pattern.

## Agent Team Execution

Treat an explicit request for this Agent Team skill as approval to create the team. If the skill was selected automatically from a generic profiling question, request Agent Team approval once. Use the same roles sequentially when team tools are unavailable.

The lead owns input normalization, task creation, common scope, synthesis, shutdown, and cleanup. Read `references/team_workflow.md` before creating tasks. Use these project agents:

1. `e2e-evidence-builder`
2. `e2e-compute-analyst`
3. `e2e-triton-kernel-analyst`
4. `e2e-compile-fusion-analyst`
5. `e2e-gap-host-analyst`
6. `e2e-noncompute-analyst`
7. `e2e-freeform-analyst`
8. `e2e-evidence-auditor`

Create `TEAM_DIR="$ANALYSIS_DIR/agent_team"` with numbered role directories from `01_baseline` through `08_audit`. Spawn the evidence builder first. After its immutable baseline is complete, start material specialists and always start the freeform analyst. Start the auditor only after branch artifacts exist. The lead alone writes final `report.md` and `report.json`.

```bash
mkdir -p "$TEAM_DIR"/{01_baseline,02_compute,03_triton_kernel,04_compile_fusion,05_gap_host,06_noncompute,07_freeform,08_audit}
```

Require every teammate to validate `findings.json` against `references/evidence_contract.md` with `scripts/validate_findings.py`. Do not accept prose without source artifacts, units, scope, counter-evidence, confidence, an overlap group, and a follow-up test.

## Resources

- `scripts/basic_info.py`: host/device time ranges, device model, device count, per-device kernel usage.
- `scripts/preflight.py`: read-only integrity, schema, identity, timestamp-range, and optional-table checks. Supports `--format json`.
- `scripts/device_timeline.py`: device projection into compute, uncovered communication, projection gap, and per-queue (device stream) gap ratio. The main compute stream gap ratio is the key host-overhead indicator. Supports `--format json` and `--process-id`/`--device-id`.
- `scripts/gap_summary.py`: merged compute-coverage gap summary and non-mini exposed gap list with `prev_corr` / `next_corr`.
- `scripts/gap_detail.py`: dependency chain for one compute gap from `--prev-corr` and `--next-corr`.
- `scripts/compute_breakdown.py`: top compute kernels and per-process/device compute concentration.
- `scripts/kernel_codegen_analysis.py`: Triton attribution signals, duration distributions, launch configurations, tiny kernels, adjacent launch pairs, and operator-to-kernel mapping coverage. Supports `--format json`.
- `scripts/compile_segmentation.py`: torch.compile compiled-region inventory, inside/outside-region (eager) kernel split, recompilation indicators, custom-op ranges that contain many simple `aten::` ops, and the host-launch-overhead / cpp_wrapper check driven by device-stream gap ratio plus trace metadata (`cpp_wrapper` config keys or `kernel_file` evidence). Supports `--format json`.
- `scripts/triton_fusion_coverage.py`: classifies compute kernels into triton-fused / other-triton / non-triton, fusion coverage ratio, Inductor fusion granularity by kernel family (pointwise/reduce/library/etc.), highlighted unfused pointwise/reduce candidates, and top non-fused kernels. Supports `--format json` and `--top`.
- `scripts/triton_kernel_efficiency.py`: triton kernel IO efficiency from `device_task_kernel_data.extra`, treating `io_efficiency` as a folded-bandwidth value (not a 0–1 ratio) compared against device peak bandwidth, plus `output_code` dump (`--dump-dir`). Supports `--format json` and `--top`.
- `../mlu-triton-optimize/scripts/analyze_triton_code.py`: static analysis of dumped Triton `output_code` for MLU optimization candidates such as libdevice math replacement, division lowering, fragmented IO, reduce layout/tiling, grid flattening, and dtype conversion cleanup. Supports `--format json|text`. If that sibling skill is not mounted, record `triton_code_optimization` as unavailable instead of substituting another tool.
- `scripts/query_common.py`: shared helpers and `--host-stack=<function_corr_id>` CLI.
- `scripts/validate_findings.py`: validates teammate and final machine-readable findings.
- `scripts/check_report.py`: deterministic Final Report Contract / Report Readability Gate check for `report.md`. Supports `--format json`, `--analysis-dir`, `--budget`, and `--strict`.
- `scripts/torch_trace_to_cnperf_db.py`: self-contained torch profiler Chrome trace converter. Requires Python module `simdjson` from package `pysimdjson`.
- `references/profiling_concepts.md`: required concepts and causal models. Always load this before starting analysis.
- `references/pytorch_performance_playbook.md`: measurement validity, evidence hierarchy, PyTorch/Inductor diagnosis, and validation rules. Always load this before Phase 1.
- `references/capability_degradation.md`: content-based input probing, capability ledger, branch-scoped degradation, bounded raw-trace fallback, and metric boundaries. Always load this before Phase 1.
- `references/distributed_context.md`: compact topology and local communication extraction, report fields, and single-rank interpretation boundaries. Load before Phase 1 baseline reporting.
- `references/hypothesis_verification.md`: independent candidate discovery, audit dispositions, impact semantics, overlap handling, and final inclusion rules. Load after the baseline and before final synthesis.
- `references/branch_workflows.md`: Phase 2 branch methods, guardrails, commands, and output contracts. Load before executing any selected branch.
- `references/evidence_contract.md`: required structured finding schema, status, confidence, counter-evidence, and benefit-overlap rules.
- `references/team_workflow.md`: Agent Team lifecycle, task graph, file ownership, quality gates, and final synthesis rules.
- `references/db_schema.md`: DB tables, field semantics, notifier wait/place matching, and SQL examples. Load this when writing direct SQL, comparing multiple DBs, or interpreting table fields.

## Core Rules

- Apply the Measurement Validity Gate before bottleneck classification. Unknown capture scope, warm state, workload metadata, profiler overhead, or repeatability lowers confidence; it must not be silently treated as valid steady-state evidence.
- Build the capability ledger from `references/capability_degradation.md` before branch selection. Route from observed content and evidence availability, not filename alone.
- Optional evidence failure is branch-scoped: keep valid baseline evidence and mark only the unsupported capability unavailable. Missing values are never measured zeros.
- Start every analysis by loading `references/profiling_concepts.md`.
- Primary evidence comes from DB tables. Cluster CSVs can validate or label DB-derived findings, but are never required.
- `device_timeline.py` non-compute categories are uncovered/exposed non-effective time. Do not treat them as total task time.
- `gap_summary.py` accounts for exposed intervals after merging overlapping compute kernels per process/device. It is separate from `device_timeline.py`.
- Load `string_table` per DB. Do not mix `nameId` mappings across DBs.
- Report observed kernel names first. Name-based grouping is heuristic.
- Do not infer root cause from one high-level percentage.
- This workflow is single-rank. Keep communication kernels as opaque timeline categories so they are not misclassified as effective compute, but do not start a communication branch or make peer/rank conclusions.
- Extract available distributed topology and local communication exposure into the report using `references/distributed_context.md`. Keep it descriptive and outside prioritized bottleneck findings/actions by default.
- `host_blocking` does not explain itself; trace the host-side blocker before naming a cause.
- Whether host overhead is large is judged primarily by the device-stream (queue) gap ratio — the fraction of the main compute stream's span spent idle between device tasks, from `device_timeline.py`. A high main-stream gap ratio means the host is not keeping the device fed. Do not judge host overhead by host-side wall time alone; a busy host with a well-fed device (low stream gap) is not host-bound.
- Keep different `threadId` timelines separate. Do not merge overlapping threads into one call tree.
- Separate compilation, autotuning, initialization, and cache warm-up from steady-state execution before judging Triton or fusion quality.
- Treat Triton/fusion name matching as a signal. Require metadata, generated artifacts, source linkage, or a controlled experiment for stronger attribution.
- Keep observed cost, critical-path contribution, and recoverable upper bound separate. Never add benefit estimates in the same overlap group.

## Measurement Validity Gate

Before Phase 1, apply the required gate in `references/pytorch_performance_playbook.md`:

1. Identify semantic scope and selected steps/requests.
2. Separate compilation, autotuning, initialization, and cache warm-up from steady state.
3. Record available workload identity: shapes, batch/tokens, dtype, grad/model mode, and rank topology.
4. Record profiler instrumentation that may perturb host timing.
5. Check whether the window contains enough repeated stable work to distinguish typical behavior from outliers.
6. Mark correctness as externally unverified unless the input includes an explicit correctness result.

Write a compact `Measurement Quality` block in `phase1_report.md` with pass/partial/fail for scope, warm state, workload identity, profiler perturbation, repeatability, and correctness. A failed semantic-scope or warm-state check blocks steady-state speed claims, but analysis may continue for cold-start or qualitative diagnosis. Carry unresolved items into `## 不确定性与下一步`.

## Final Report Contract

For both Web/server-side automatic runs and interactive final synthesis, produce one stable,
user-facing final report:

- Write the final user-visible report to `$TRACE_AI_REPORT_PATH` when that environment variable is set.
- Also write the same final report to `report.md` in the current working directory.
- Write `report.json` beside `report.md` using `references/evidence_contract.md`, preserving source finding IDs and overlap groups.
- Print the same final report to stdout. Do not print tool logs, raw command output, prompt text, or progress narration to stdout.
- Save supporting evidence under `TEAM_DIR`, including baseline/branch reports, validated findings, the adversarial audit, `evidence_summary.md`, and script logs.
- If analysis cannot proceed because a trace file, DB table, Python dependency, or tool permission is missing, write a concise failure report instead of a partial or fabricated performance report.
- Prefer Chinese report text when the request is Chinese.

The final `report.md` must use this exact high-level structure:

1. `# AI 性能分析报告`
2. `## 结论概览`
   - 2-4 prioritized findings only; prefer 3 unless the evidence clearly needs more.
   - Use one subsection per finding: `### 发现 N：short title`, followed by separate paragraphs `**结论：** ...`, `**证据：** ...`, and `**建议：** ...`.
   - Keep each `结论` / `证据` / `建议` paragraph to one short sentence. Merge overlapping findings instead of repeating the same cause from multiple script outputs.
   - Treat host gap, launch overhead, and `cpp_wrapper` mode as one causal chain when they describe the same bottleneck; do not split them into separate findings.
   - If `compile_segmentation.json` reports `custom_op_simple_aten.must_report=true`, one finding must cover that custom-op/simple-aten issue even when its direct duration is smaller than host gap or IO-efficiency findings. Use 4 findings if needed instead of dropping it.
   - If `triton_code_optimization.json` reports `has_findings=true`, the final report must include a dedicated top-level `## Triton Kernel 代码优化` section copied from or equivalent to `final_report_guidance.required_table_md`. Do not satisfy this requirement with only a one-line summary, a nested table under another section, or a `产物` entry.
   - Do not output sibling bullets like `- 结论` / `- 证据` / `- 建议`; that renders as a flat wall in the Web UI.
3. `## 关键指标`
   - Compact Markdown table with metric, value, source file/log, and interpretation.
4. `## 分布式与通信概况`
   - Compact Markdown table with available rank/topology metadata, local communication total/uncovered time, observed collective names, source, and evidence boundary.
   - Keep this section descriptive; it does not consume one of the 2-4 prioritized findings and does not introduce communication optimization actions by default.
   - If the capture has no usable distributed/communication evidence, retain the section with one explicit `未捕获` row.
5. `## 优先行动`
   - Prioritized actions with expected benefit, mechanism, implementation cost, risk, correctness guardrail, and validation method.
6. `## Triton Kernel 代码优化` (only when `triton_code_optimization.json.has_findings=true`)
   - Compact table copied from or equivalent to `final_report_guidance.required_table_md`; include all candidates from `triton_code_optimization.json`, not just the top few.
   - Keep wording as static code-level candidates and validation targets unless runtime evidence confirms a speedup.
7. `## 不确定性与下一步`
   - Missing evidence and the next check that would reduce uncertainty.
8. `## 产物`
   - Generated DBs, stage reports, evidence logs, and analysis directory.

Default to a concise Web report. Target no more than 1200 Chinese characters before the `产物`
section. Do not duplicate a full `主要发现` section after `结论概览`; put detailed branch findings,
long evidence, raw tables, stack traces, and script logs into artifacts such as
`phase2_<branch>_report.md` and `evidence_summary.md`, then cite those filenames from the report.
If graph capture, multi-stream execution, or driver/runtime upgrades are only plausible follow-ups
without direct trace evidence, keep them in `不确定性与下一步` instead of promoting them to top
findings or primary actions.

Before writing `report.md`, apply the Report Readability Gate. Run it as a check instead of
recalling it, then fix every `error` and re-run until `ok` is true:

```bash
python3 "$SKILL_DIR/scripts/check_report.py" "$REPORT_MD" \
  --analysis-dir "$ANALYSIS_DIR" --format json > "$TEAM_DIR/report_gate.json"
```

The script enforces one report structure (no `主要发现` / `详细分析` / `执行摘要` parallel summary),
required section set and order, 2-4 `### 发现 N：` blocks each with its own `**结论：**` /
`**证据：**` / `**建议：**` paragraph, no flat `- 结论` / `- 证据` / `- 建议` bullets, unique
top-level headings, Markdown table separator rows, no raw stdout/stderr blocks, a non-empty `产物`
list, the `## Triton Kernel 代码优化` section and its full candidate table when
`triton_code_optimization.json` has findings, and a reserved custom-op/simple-aten finding when
`compile_segmentation.json` sets `must_report`. A `warn` about the length budget or a metadata block
is a rewrite prompt, not a pass.

Keep these judgement rules, which the script cannot check:

- Keep finding titles short and factual. Do not put the full metric, evidence chain, and root
  cause into the title. Use "主要瓶颈" / "证据指向" unless the direct evidence proves a root cause.
- Keep each `**结论：**`, `**证据：**`, and `**建议：**` paragraph to no more than two clauses. Move
  long call chains, raw percentages, and long file lists to stage artifacts.
- Avoid repeating the same number in `结论概览`, `关键指标`, and `不确定性与下一步`; cite it once
  where it is most useful.
- If Triton `output_code` or IO-efficiency metadata is missing, mention it only as a data gap in
  `不确定性与下一步`, not as a main bottleneck or primary action.
- Keep the optional metadata below the H1 compact: either a 2-4 row table or short bullets.

## Setup And Inputs

Create all generated artifacts under one analysis directory.

For Web/server-side automatic analysis, use the existing working directory:

```bash
SKILL_DIR=<absolute-path-to-this-skill>
ANALYSIS_DIR="${TRACE_AI_ANALYSIS_DIR:-$PWD}"
REPORT_MD="${TRACE_AI_REPORT_PATH:-$ANALYSIS_DIR/report.md}"
TEAM_DIR="$ANALYSIS_DIR/agent_team"
```

For interactive local analysis, create a temporary analysis directory in the current working directory:

```bash
SKILL_DIR=<absolute-path-to-this-skill>
ANALYSIS_DIR="e2e_profiling_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir "$ANALYSIS_DIR"
REPORT_MD="$ANALYSIS_DIR/report.md"
TEAM_DIR="$ANALYSIS_DIR/agent_team"
```

For interactive local analysis, use the exact directory name format `e2e_profiling_analysis_YYYYMMDD_HHMMSS`; if it collides, append `_NN`. Put normalized inputs and final reports in this directory and teammate artifacts under `TEAM_DIR`. If trace conversion needs a venv, create it as `<analysis_dir>/.venv-trace-convert`; this is a dependency environment, not an analysis artifact.

Resolve `SKILL_DIR` to this skill directory's absolute path before running any command. This `SKILL.md` lives at `<SKILL_DIR>/SKILL.md`, so derive it from the path you loaded this skill from rather than guessing. If that path is not directly available, locate it once and reuse the result:

```bash
SKILL_DIR=$(dirname "$(find "$HOME/.claude" "$PWD" -type f -path '*/e2e-profiling-analyzer/SKILL.md' 2>/dev/null | head -n1)")
```

Verify `SKILL_DIR/scripts/basic_info.py` exists before proceeding. Do not call scripts from repository-level `tools/`, `.trae/`, or any path outside this skill.

Input normalization:

- Native `*.db`: analyze directly. Do not copy large native DBs into the analysis directory unless the user asks for a self-contained bundle.
- Torch profiler Chrome trace `*.json`, `*.json.gz`, or `*.pt.trace.json.gz`: convert first, then analyze the generated DB.
- Directory input: recursively find native DBs and torch trace JSON/JSON.GZ files. Convert trace files into the analysis directory, then include generated DBs in the analysis set.
- Multiple DBs may represent repeated captures or alternatives, but never infer that they are peer ranks without explicit topology evidence. This single-rank workflow analyzes each DB independently.
- Do not mix raw trace JSON with DB analysis scripts; all later scripts consume SQLite DB paths only.

Dependency setup for trace conversion:

```bash
TRACE_PY=python3
if ! "$TRACE_PY" -c "import simdjson" >/dev/null 2>&1; then
  python3 -m venv "$ANALYSIS_DIR/.venv-trace-convert"
  "$ANALYSIS_DIR/.venv-trace-convert/bin/python" -m pip install pysimdjson \
    > "$ANALYSIS_DIR/venv-trace-convert.install.log" 2>&1
  TRACE_PY="$ANALYSIS_DIR/.venv-trace-convert/bin/python"
fi
```

Do not install `pysimdjson` globally unless the user explicitly asks.

Trace conversion command:

```bash
"$TRACE_PY" "$SKILL_DIR/scripts/torch_trace_to_cnperf_db.py" \
  <trace.json-or-json.gz> \
  --out "$ANALYSIS_DIR/<trace_stem>.db" \
  --report "$ANALYSIS_DIR/<trace_stem>.conversion.json" \
  > "$ANALYSIS_DIR/<trace_stem>.convert.log" 2>&1
```

Use stable, collision-free trace stems. If two input traces have the same basename, include the parent directory name or a short hash.

Converter behavior:

- Uses `simdjson` only.
- Supports gzip-compressed trace files.
- Writes cnperf-compatible core tables for kernels, memcpy, memset, notifier, runtime functions, operation ranges, relations, device information, meta information, and required empty device-task tables.
- Provides `MLU_KERNEL_EVENTS_V1` and `STRING_IDS_V1` compatibility views.
- Normalizes timestamps by subtracting the earliest converted event start.
- `nameId` values are local to the converted DB. Always resolve names through that DB's `string_table`.
- `gpu_user_annotation` events are skipped. CPU-side `user_annotation` and `cpu_op` are converted to `Internal_operation_range_data`.

After conversion, run `basic_info.py` on every generated DB to verify each one opens and its meta/device information is readable. If any converted DB fails this check, record it as blocked and exclude it from the analysis set instead of letting a silently corrupt DB flow into later scripts.

## Output Files

Write these Markdown reports in both modes:

- `TEAM_DIR/01_baseline/phase1_report.md`: Phase 1 baseline analysis and branch recommendation/selection.
- `TEAM_DIR/<branch>/report.md`: detailed report for each completed specialist.
- `TEAM_DIR/08_audit/report.md`: adversarial review and contradiction resolution.
- `report.md`: final synthesis report after automatic branch execution completes, or after the user asks to conclude/declines more branches in `interactive-phased` mode.
- `report.json`: final structured findings and artifact index.
- `TEAM_DIR/report_gate.json`: `check_report.py` result for the final report.

Do not duplicate full per-branch detailed reports inside `report.md`; reference their filenames from `产物`. Keep raw table excerpts in stage reports or `evidence_summary.md`, and cite those artifacts from the final report.

Stage reports and logs should include enough filenames and raw excerpts for later audit.

Report language convention: keep the structural field labels defined in each output contract (`Scope`, `Branch`, `Candidate Causes`, etc.) in English across stage reports so the contracts stay machine-checkable, but write the narrative content (findings, interpretation, conclusions) in the request's language. When the request is Chinese, the narrative in `phase1_report.md` and `phase2_<branch>_report.md` should be Chinese, and the final `report.md` must follow the all-Chinese structure defined in `Final Report Contract`.

## Phase 1: Initial Analysis

Goal: classify why device time is not effective compute, produce branch selection evidence, and write `phase1_report.md`.

Workflow:

1. Apply the Measurement Validity Gate and record capture-quality limitations.
2. Inventory inputs and write a capability ledger using `references/capability_degradation.md`.
   - One DB: analyze it as one process/device first.
   - One torch trace JSON/JSON.GZ: convert it to DB first and analyze the generated DB.
   - Directory: find `cnperf_data_*.db` and analyze each DB independently without cross-rank inference.
3. Run baseline scripts for every analysis DB.

   ```bash
   db_stem=$(basename "<cnperf_db>" .db)
   python3 "$SKILL_DIR/scripts/preflight.py" <cnperf_db> --format json \
     > "$TEAM_DIR/01_baseline/${db_stem}.preflight.json" 2>&1
   python3 "$SKILL_DIR/scripts/basic_info.py" <cnperf_db> \
     > "$TEAM_DIR/01_baseline/${db_stem}.basic_info.log" 2>&1
   python3 "$SKILL_DIR/scripts/device_timeline.py" <cnperf_db> --format json \
     > "$TEAM_DIR/01_baseline/${db_stem}.device_timeline.json" 2>&1
   python3 "$SKILL_DIR/scripts/gap_summary.py" <cnperf_db> --invoke-threshold 100 --format json \
     > "$TEAM_DIR/01_baseline/${db_stem}.gap_summary.json" 2>&1
   python3 "$SKILL_DIR/scripts/kernel_codegen_analysis.py" <cnperf_db> --format json \
     > "$TEAM_DIR/01_baseline/${db_stem}.kernel_codegen.json" 2>&1
   ```

4. Classify the initial situation from the effective-compute perspective.
5. Derive independent candidate signals with stable IDs, scope, observed cost, evidence path, and a falsifiable question; do not force one candidate per category.
6. Recommend or select specialist tasks with the rules below.
7. Write `phase1_report.md` and validated `findings.json` under `TEAM_DIR/01_baseline`.
8. In `interactive-phased`, show the Phase 1 key findings and ask which branch or branches to run next.
9. In `automatic-final`, run the selected Phase 2 branch set.

Initial situation categories:

- `effective-compute-high`: compute kernel time dominates, exposed non-compute time is low, and compute gaps are low.
- `exposed-ordinary-non-compute-high`: uncovered memcpy, memset, atomic, or other ordinary non-compute work is material.
- `compute-gap-high`: `gap_summary.py` reports material compute-kernel gaps or large top gaps.
- `triton-kernel-material`: Triton-attributed kernels have material compute time, long tails, weak IO efficiency, or meaningful codegen metadata.
- `compile-fusion-granularity-material`: compiled-region fragmentation, unfused pointwise/reduce work, custom-op/simple-aten nesting, or tiny/repeated launches are material.

Branch selection:

- Run or recommend `ordinary-non-compute-root-cause` when uncovered memcpy, memset, atomic, or other ordinary non-compute time is material, or top gaps point to memcpy/ordinary device work.
- Run or recommend `compute-gap-root-cause` when `gap_summary.py` shows material total gap time, large individual gaps, or host/notifier/previous-task gap reasons.
- Run or recommend `effective-compute-breakdown` when effective compute dominates total device time or top kernel families are material.
- Run `host-window-subphase` only when the user provided a host time window or explicitly asked for host-window subphase analysis.
- Run or recommend `compile-segmentation` when the workload uses torch.compile/inductor and the DB carries compiled-region annotations (`Torch-Compiled Region`, `CompiledFunction`, `CompiledFunctionBackward`, `TorchDynamo Cache Lookup`, `inductor`, or similar) in `Internal_operation_range_data`, especially when compute gaps or ordinary non-compute work cluster at region boundaries, many kernels run outside compiled regions, or custom/user operators wrap many simple `aten::` pointwise/view/reduce/copy ops.
- Run or recommend `triton-fusion-coverage` when compute is material and a non-trivial share of compute-kernel time comes from non-`triton`/non-fused kernels, indicating ops that fell back to library/eager execution instead of inductor fusion.
- Run or recommend `triton-kernel-efficiency` only when triton kernels carry `output_code` and IO-efficiency metadata in their `extra` JSON. If that metadata is absent, skip this branch and record it under `Skipped Branches` with the missing-metadata reason.

The last three branches are torch.compile/inductor-specific and apply mainly to converted torch profiler traces. The compile/fusion agent owns `compile-segmentation` and `triton-fusion-coverage`; the Triton agent owns `triton-kernel-efficiency` and optional static output-code analysis. If the DB has no compiled-region annotations and no `triton_*` kernel names, record both specialists as unsupported unless the user requested every angle.

In `automatic-final`, limit Phase 2 to branches that can change the final recommendation. If many categories are material, prioritize the largest exposed category first, then add other branches whose measured impact is close enough to affect ranking or whose evidence may explain the largest category. Do not run speculative branches just because they exist.

`phase1_report.md` must include:

- `Measurement Quality`: semantic scope, warm state, workload identity, profiler perturbation, repeatability, and correctness status.
- `Scope`: input DBs, process/device coverage, host/device time range.
- `Effective Compute`: compute kernel time and ratio.
- `Exposed Non-Effective Time`: opaque communication category, ordinary non-compute categories, and projection gap. Do not interpret peer behavior.
- `Distributed Context`: available rank/topology metadata plus local communication total/uncovered time and observed collective names from `references/distributed_context.md`; state missing peer/rank coverage.
- `Device Stream Gap Ratio`: main compute stream gap ratio and device-level gap ratio from `device_timeline.py`. This is the key host-overhead indicator; flag a host-bound situation when the main-stream gap ratio is high.
- `Compute Gap Summary`: total compute-kernel gap, dominant coarse reasons, top relevant gaps.
- `Initial Situation`: one or more categories above, with evidence.
- `Recommended Or Selected Phase 2 Branches`: specialist tasks with evidence and priority.
- `Skipped Branches`: branches not run/recommended, with reasons.
- `Raw Tables`: baseline script output filenames and compact excerpts.
- `Artifacts`: analysis directory, generated DBs, logs, and report paths.

Do not run `gap_detail.py`, host-blocking trace, or host-window subphase analysis inside Phase 1 itself. Run those only inside specialist branches.

## Phase 2: Branch Analysis

Goal: run selected branch analyses, write one detailed `phase2_<branch>_report.md` per completed branch, and save raw script/query outputs in the analysis directory.

Mode behavior:

- `interactive-phased`: run exactly the branch or branches selected by the user, then ask whether to run more branches or proceed to Phase 3.
- `automatic-final`: assign the automatically selected branch set to the matching project agents, always run freeform analysis, then run the evidence auditor before Phase 3.

Parallel execution:

- Run independent specialist agents in parallel when their inputs and outputs do not conflict.
- Branches are parallel-safe when they only read DB inputs and write distinct output files such as `phase2_<branch>_report.md` and branch-specific logs/query outputs.
- Use Phase 1 priority to schedule work, but do not serialize independent branches unnecessarily.
- Do not run `host-window-subphase` in parallel unless the host window is already known and its outputs are isolated.
- If required input is missing, record the branch as blocked. In `interactive-phased`, ask for the missing input or another branch. In `automatic-final`, continue with remaining branches.

Every branch result must include:

- `Branch`: selected branch and why it was selected.
- `Method`: scripts and DB tables used.
- `Findings`: branch-specific metrics and dependency evidence.
- `Candidate Causes`: plausible causes with supporting evidence, counter-evidence, affected process/device scope, estimated impact, confidence, overlap group, and missing evidence.
- `Interpretation`: what the evidence explains and what remains uncertain.
- `Follow-up Suggestions`: optional extra branches or inputs that could reduce uncertainty.
- `Raw Tables`: script logs, JSON/text outputs, or query result files produced for the branch.
- `Artifacts`: branch report path and evidence files.

Cause handling:

- Do not force a single root cause.
- If one cause is clearly supported, mark it as dominant and explain why alternatives are weaker.
- If multiple causes remain plausible, report them with confidence and missing evidence.
- If evidence is insufficient, say unresolved and list the specific missing input needed to disambiguate.

### Branch workflow index

Load `references/branch_workflows.md` before executing any selected Phase 2 branch. It contains the normative workflow, guardrails, commands, and output contract for:

- `effective-compute-breakdown`
- `ordinary-non-compute-root-cause`
- `compute-gap-root-cause`
- `host-window-subphase`
- `compile-segmentation`
- `triton-fusion-coverage`
- `triton-kernel-efficiency`

Agent ownership is fixed: compute owns `effective-compute-breakdown`; compile/fusion owns `compile-segmentation` and `triton-fusion-coverage`; Triton owns `triton-kernel-efficiency`; gap/host owns `compute-gap-root-cause` and optional `host-window-subphase`; noncompute owns `ordinary-non-compute-root-cause`. Load only the assigned branch sections.

## Phase 3: Final Synthesis

Goal: synthesize Phase 1 and completed Phase 2 branches into a concise `report.md`, provide prioritized recommendations, and reference raw evidence artifacts.

Enter Phase 3:

- In `interactive-phased`, after the user asks to conclude or declines more branches.
- In `automatic-final`, after automatic branch execution finishes.

Workflow:

1. List completed inputs: Phase 1 baseline, each validated branch finding file, freeform findings, and the adversarial audit.
2. Reject findings that fail `validate_findings.py` or the auditor's evidence gate.
3. Load `references/hypothesis_verification.md`; merge evidence by causal path, apply the auditor's disposition, and separate supported findings from refuted, duplicate, and insufficient candidates.
4. Read the baseline distributed context and `device_timeline.json`; write the compact `## 分布式与通信概况` section using `references/distributed_context.md`, without promoting it to a bottleneck finding or action by default.
5. Before pruning to the final 2-4 findings, scan `compile_segmentation.json` for `custom_op_simple_aten.must_report=true`. When present, reserve one finding and one action row for the custom-op/simple-aten issue; this is a structural missed-fusion signal and should not be buried because its host-range duration is smaller than other exposed-time metrics.
6. Also scan `triton_code_optimization.json` when present. If `has_findings=true`, read `final_report_guidance` first and include a dedicated top-level `## Triton Kernel 代码优化` section using `final_report_guidance.required_table_md` or an equivalent table. The table must include all candidates from `triton_code_optimization.json`, and name concrete kernels, measured time, BW utilization when available, static estimated throughput, and merged strategy/recommendation. Do not add a separate `证据` column. Place this section after `## 优先行动` and before `## 不确定性与下一步`. Still include a `关键指标` row with scanned file count, candidate kernel count, top strategies, and source file. Keep wording as a validation target unless runtime evidence confirms a speedup.
7. Record observed cost, critical-path contribution, and recoverable upper bound separately. If benefits share an overlap group, state that they are not additive.
8. Prioritize recommendations by expected impact, confidence, and implementation scope.
9. Apply the recommendation contract from `references/pytorch_performance_playbook.md`: include mechanism, correctness guardrail, one controlled experiment, end-to-end success metric, and rollback condition.
10. If custom-op/simple-aten is reserved, phrase the action as moving repeated simple `aten::` pointwise/view/reduce/copy/allocation work into the custom backend kernel, or restructuring the wrapper so Inductor can see and fuse it.
11. Call out missing evidence and which branch or input would close it.
12. Write final `report.json` from the audited evidence contract, preserving source finding IDs, `audit_disposition`, impact fields, overlap groups, and artifact paths.
13. Do not append raw table dumps to `report.md`. Keep audit details in stage reports or `evidence_summary.md`, and reference full output filenames.

Final report writing:

- Follow only the `Final Report Contract` above. Do not restate, reinterpret, or duplicate the
  final structure in Phase 3.
- Apply the Report Readability Gate before writing `report.md`, and run `scripts/check_report.py` on the written file before reporting completion.
- If the draft violates the gate, rewrite the final report once instead of appending a correction.

## Validation And Failure Handling

- Use `python3`, not `python`.
- If a table is missing, state what is unavailable and continue with remaining evidence.
- If `string_table` is missing, report `nameId=...`.
- If multiple processes/devices are present in one DB, call that out and filter when needed.
- Treat all communication events as opaque local timeline categories; never infer missing peer behavior.
- Keep thresholds as triage aids, not hard truth. Prefer measured ratios and dependency evidence.
