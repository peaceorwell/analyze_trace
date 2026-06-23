---
name: e2e-profiling-analyzer
description: Analyze cnperf SQLite databases and torch profiler Chrome trace JSON/JSON.GZ files for end-to-end training or inference bottlenecks on MLU workloads. Use when the user asks to inspect `cnperf_data*.db`, torch profiler `.pt.trace.json`/`.json.gz` traces, identify why effective compute kernels are not dominating device time, or root-cause exposed communication, ordinary non-compute work, compute gaps, host/device synchronization, memcpy, rank imbalance, torch.compile graph segmentation, triton kernel fusion coverage, or triton kernel output-code/IO efficiency.
---

# E2E Profiling Analyzer

Analyze `cnperf` SQLite DBs from the viewpoint that compute kernels are effective device utilization. If the input is a torch profiler Chrome trace JSON/JSON.GZ file, first convert it to a cnperf-compatible SQLite DB, then analyze the converted DB exactly like native cnperf data.

## Mode Selection

Use `automatic-final` immediately when any of these are true:

- The prompt says `automatic-final`, `自动最终报告`, `不要向用户追问`, or that this is a Web/server-side/background analysis.
- Environment variables such as `TRACE_AI_JOB_ID`, `TRACE_AI_ANALYSIS_DIR`, `TRACE_AI_REPORT_PATH`, or `TRACE_AI_TRACE_A` are present.
- The user asks for an end-to-end report instead of a phased investigation.

Use `interactive-phased` only when the user explicitly wants to choose branches step by step, or has already named a specific branch or branches to run.

In `automatic-final`, never ask follow-up questions. If required evidence is missing, continue with available evidence and record the missing input under `Open Questions`.

If the prompt is an environment diagnostic or smoke test and asks to reply only `OK`, reply exactly `OK` and do not run tools or load references.

Mode meanings:

- `automatic-final`: use Phase 1 evidence to choose branch analyses automatically, run independent branches in parallel when possible, and produce stage reports plus the final synthesis.
- `interactive-phased`: write a report after each stage, stop after Phase 1 for branch selection, then run the branch or branches selected by the user.

Default to `automatic-final` when the user has not specified a mode or a branch. Only use `interactive-phased` when the user explicitly requested step-by-step branch selection, or already named a specific branch or branches to run after Phase 1.

Do not assume the bottleneck is communication, a specific kernel family, TCDP, or a known synchronization pattern. `cluster_aggregation/step` CSV files are optional enrichment only.

## Resources

- `scripts/basic_info.py`: host/device time ranges, device model, device count, per-device kernel usage.
- `scripts/device_timeline.py`: device projection into compute, uncovered communication, projection gap, and per-queue (device stream) gap ratio. The main compute stream gap ratio is the key host-overhead indicator. Supports `--format json` and `--process-id`/`--device-id`.
- `scripts/gap_summary.py`: merged compute-coverage gap summary and non-mini exposed gap list with `prev_corr` / `next_corr`.
- `scripts/gap_detail.py`: dependency chain for one compute gap from `--prev-corr` and `--next-corr`.
- `scripts/compute_breakdown.py`: top compute kernels and per-process/device compute skew.
- `scripts/comm_breakdown.py`: communication kernel total/uncovered time, per-process/device exposure, and top long events.
- `scripts/rank_compare.py`: cross-DB process/device span, compute, uncovered communication, and compute-gap skew.
- `scripts/compile_segmentation.py`: torch.compile compiled-region inventory, inside/outside-region (eager) kernel split, recompilation indicators, custom-op ranges that contain many simple `aten::` ops, and the host-launch-overhead / cpp_wrapper check driven by device-stream gap ratio plus trace metadata (`cpp_wrapper` config keys or `kernel_file` evidence). Supports `--format json`.
- `scripts/triton_fusion_coverage.py`: classifies compute kernels into triton-fused / other-triton / non-triton, fusion coverage ratio, Inductor fusion granularity by kernel family (pointwise/reduce/library/etc.), highlighted unfused pointwise/reduce candidates, top non-fused kernels, and per-rank coverage. Supports `--format json` and `--top`.
- `scripts/triton_kernel_efficiency.py`: triton kernel IO efficiency from `device_task_kernel_data.extra`, treating `io_efficiency` as a folded-bandwidth value (not a 0–1 ratio) compared against device peak bandwidth, plus `output_code` dump (`--dump-dir`). Supports `--format json` and `--top`.
- `scripts/query_common.py`: shared helpers and `--host-stack=<function_corr_id>` CLI.
- `scripts/torch_trace_to_cnperf_db.py`: self-contained torch profiler Chrome trace converter. Requires Python module `simdjson` from package `pysimdjson`.
- `references/profiling_concepts.md`: required concepts and causal models. Always load this before starting analysis.
- `references/db_schema.md`: DB tables, field semantics, notifier wait/place matching, and SQL examples. Load this when writing direct SQL, comparing multiple DBs, or interpreting table fields.

## Core Rules

- Start every analysis by loading `references/profiling_concepts.md`.
- Primary evidence comes from DB tables. Cluster CSVs can validate or label DB-derived findings, but are never required.
- `device_timeline.py` non-compute categories are uncovered/exposed non-effective time. Do not treat them as total task time.
- `gap_summary.py` accounts for exposed intervals after merging overlapping compute kernels per process/device. It is separate from `device_timeline.py`.
- Load `string_table` per DB. Do not mix `nameId` mappings across DBs.
- Report observed kernel names first. Name-based grouping is heuristic.
- Do not infer root cause from one high-level percentage.
- High communication time on one rank does not prove that rank is slow; it may be waiting for another rank.
- Single-card communication is not analyzable. When only one card/rank is available (one DB, or one process+device, including a single-card torch trace captured with stack enabled), exposed communication is dominated by waiting for absent peers, not by real communication cost. Treat single-card communication kernels as opaque exposed time. Unless the user explicitly asks for communication analysis, do not run the `communication-root-cause` branch, do not select it in Phase 1, and do not state communication conclusions for single-card input; route the investigation to compute-gap, ordinary-non-compute, or compute-breakdown branches instead.
- `host_blocking` does not explain itself; trace the host-side blocker before naming a cause.
- Whether host overhead is large is judged primarily by the device-stream (queue) gap ratio — the fraction of the main compute stream's span spent idle between device tasks, from `device_timeline.py`. A high main-stream gap ratio means the host is not keeping the device fed. Do not judge host overhead by host-side wall time alone; a busy host with a well-fed device (low stream gap) is not host-bound.
- Keep different `threadId` timelines separate. Do not merge overlapping threads into one call tree.

## Final Report Contract

For both Web/server-side automatic runs and interactive final synthesis, produce one stable,
user-facing final report:

- Write the final user-visible report to `$TRACE_AI_REPORT_PATH` when that environment variable is set.
- Also write the same final report to `report.md` in the current working directory.
- Print the same final report to stdout. Do not print tool logs, raw command output, prompt text, or progress narration to stdout.
- Save supporting evidence as separate files in the analysis directory, such as `phase1_report.md`, `phase2_<branch>_report.md`, `evidence_summary.md`, and script logs.
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
   - Do not output sibling bullets like `- 结论` / `- 证据` / `- 建议`; that renders as a flat wall in the Web UI.
3. `## 关键指标`
   - Compact Markdown table with metric, value, source file/log, and interpretation.
4. `## 优先行动`
   - Prioritized actions with expected benefit, implementation cost, risk, and validation method.
5. `## 不确定性与下一步`
   - Missing evidence and the next check that would reduce uncertainty.
6. `## 产物`
   - Generated DBs, stage reports, evidence logs, and analysis directory.

Default to a concise Web report. Target no more than 1200 Chinese characters before the `产物`
section. Do not duplicate a full `主要发现` section after `结论概览`; put detailed branch findings,
long evidence, raw tables, stack traces, and script logs into artifacts such as
`phase2_<branch>_report.md` and `evidence_summary.md`, then cite those filenames from the report.
If graph capture, multi-stream execution, or driver/runtime upgrades are only plausible follow-ups
without direct trace evidence, keep them in `不确定性与下一步` instead of promoting them to top
findings or primary actions.

## Setup And Inputs

Create all generated artifacts under one analysis directory.

For Web/server-side automatic analysis, use the existing working directory:

```bash
SKILL_DIR=<absolute-path-to-this-skill>
ANALYSIS_DIR="${TRACE_AI_ANALYSIS_DIR:-$PWD}"
REPORT_MD="${TRACE_AI_REPORT_PATH:-$ANALYSIS_DIR/report.md}"
```

For interactive local analysis, create a temporary analysis directory in the current working directory:

```bash
SKILL_DIR=<absolute-path-to-this-skill>
ANALYSIS_DIR="e2e_profiling_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir "$ANALYSIS_DIR"
REPORT_MD="$ANALYSIS_DIR/report.md"
```

For interactive local analysis, use the exact directory name format `e2e_profiling_analysis_YYYYMMDD_HHMMSS`; if it collides, append `_NN`. Put generated DBs, conversion reports, script logs, ad hoc query outputs, stage reports, and final reports directly in this directory. If trace conversion needs a venv, create it as `<analysis_dir>/.venv-trace-convert`; this is a dependency environment, not an analysis artifact.

Resolve `SKILL_DIR` to this skill directory's absolute path before running any command. This `SKILL.md` lives at `<SKILL_DIR>/SKILL.md`, so derive it from the path you loaded this skill from rather than guessing. If that path is not directly available, locate it once and reuse the result:

```bash
SKILL_DIR=$(dirname "$(find "$HOME/.claude" "$PWD" -type f -path '*/e2e-profiling-analyzer/SKILL.md' 2>/dev/null | head -n1)")
```

Verify `SKILL_DIR/scripts/basic_info.py` exists before proceeding. Do not call scripts from repository-level `tools/`, `.trae/`, or any path outside this skill.

Input normalization:

- Native `*.db`: analyze directly. Do not copy large native DBs into the analysis directory unless the user asks for a self-contained bundle.
- Torch profiler Chrome trace `*.json`, `*.json.gz`, or `*.pt.trace.json.gz`: convert first, then analyze the generated DB.
- Directory input: recursively find native DBs and torch trace JSON/JSON.GZ files. Convert trace files into the analysis directory, then include generated DBs in the analysis set.
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

- `phase1_report.md`: Phase 1 baseline analysis and branch recommendation/selection.
- `phase2_<branch>_report.md`: detailed report for each completed branch.
- `report.md`: final synthesis report after automatic branch execution completes, or after the user asks to conclude/declines more branches in `interactive-phased` mode.

Do not duplicate full per-branch detailed reports inside `report.md`; reference their filenames from `产物`. Keep raw table excerpts in stage reports or `evidence_summary.md`, and cite those artifacts from the final report.

Stage reports and logs should include enough filenames and raw excerpts for later audit.

Report language convention: keep the structural field labels defined in each output contract (`Scope`, `Branch`, `Candidate Causes`, etc.) in English across stage reports so the contracts stay machine-checkable, but write the narrative content (findings, interpretation, conclusions) in the request's language. When the request is Chinese, the narrative in `phase1_report.md` and `phase2_<branch>_report.md` should be Chinese, and the final `report.md` must follow the all-Chinese structure defined in `Final Report Contract`.

## Phase 1: Initial Analysis

Goal: classify why device time is not effective compute, produce branch selection evidence, and write `phase1_report.md`.

Workflow:

1. Inventory inputs.
   - One DB: analyze it as one process/device first.
   - One torch trace JSON/JSON.GZ: convert it to DB first and analyze the generated DB.
   - Directory: find `cnperf_data_*.db`, analyze each DB independently, then compare ranks/devices.
   - Optional `cluster_aggregation/step`: note it if present, but keep DB-derived facts primary.
2. Run baseline scripts for every analysis DB.

   ```bash
   db_stem=$(basename "<cnperf_db>" .db)
   python3 "$SKILL_DIR/scripts/basic_info.py" <cnperf_db> \
     > "$ANALYSIS_DIR/${db_stem}.basic_info.log" 2>&1
   python3 "$SKILL_DIR/scripts/device_timeline.py" <cnperf_db> \
     > "$ANALYSIS_DIR/${db_stem}.device_timeline.log" 2>&1
   python3 "$SKILL_DIR/scripts/gap_summary.py" <cnperf_db> --invoke-threshold 100 \
     > "$ANALYSIS_DIR/${db_stem}.gap_summary.log" 2>&1
   ```

3. If optional cluster CSVs are present, use them only to enrich labels and rank/category comparisons.
4. Classify the initial situation from the effective-compute perspective.
5. Recommend or select Phase 2 branches with the rules below.
6. Write `phase1_report.md`.
7. In `interactive-phased`, show the Phase 1 key findings and ask which branch or branches to run next.
8. In `automatic-final`, run the selected Phase 2 branch set.

Initial situation categories:

- `effective-compute-high`: compute kernel time dominates, exposed non-compute time is low, and compute gaps are low.
- `exposed-communication-high`: uncovered communication from `device_timeline.py` is material.
- `exposed-ordinary-non-compute-high`: uncovered memcpy, memset, atomic, or other ordinary non-compute work is material.
- `compute-gap-high`: `gap_summary.py` reports material compute-kernel gaps or large top gaps.

Rank/workload imbalance is not a first-level category. Consider it inside `communication-root-cause` or `compute-gap-root-cause` when multiple DBs show skew.

Branch selection:

- Run or recommend `communication-root-cause` only when multiple rank/card DBs are available and uncovered communication is material, top compute gaps are communication-related, or multi-rank evidence suggests waiting/slow-arriver behavior. For single-card input, skip this branch and record it under `Skipped Branches` with the reason that single-card communication is not analyzable, unless the user explicitly asked for communication analysis.
- Run or recommend `ordinary-non-compute-root-cause` when uncovered memcpy, memset, atomic, or other ordinary non-compute time is material, or top gaps point to memcpy/ordinary device work.
- Run or recommend `compute-gap-root-cause` when `gap_summary.py` shows material total gap time, large individual gaps, or host/notifier/previous-task gap reasons.
- Run or recommend `effective-compute-breakdown` when effective compute dominates total device time or Phase 1 suggests compute imbalance across ranks/devices.
- Run `host-window-subphase` only when the user provided a host time window or explicitly asked for host-window subphase analysis.
- Run or recommend `compile-segmentation` when the workload uses torch.compile/inductor and the DB carries compiled-region annotations (`Torch-Compiled Region`, `CompiledFunction`, `CompiledFunctionBackward`, `TorchDynamo Cache Lookup`, `inductor`, or similar) in `Internal_operation_range_data`, especially when compute gaps or ordinary non-compute work cluster at region boundaries, many kernels run outside compiled regions, or custom/user operators wrap many simple `aten::` pointwise/view/reduce/copy ops.
- Run or recommend `triton-fusion-coverage` when compute is material and a non-trivial share of compute-kernel time comes from non-`triton`/non-fused kernels, indicating ops that fell back to library/eager execution instead of inductor fusion.
- Run or recommend `triton-kernel-efficiency` only when triton kernels carry `output_code` and IO-efficiency metadata in their `extra` JSON. If that metadata is absent, skip this branch and record it under `Skipped Branches` with the missing-metadata reason.

The last three branches are torch.compile/inductor-specific and apply mainly to converted torch profiler traces. If the DB has no compiled-region annotations and no `triton_*` kernel names, the workload likely does not use torch.compile; skip all three and record them under `Skipped Branches`.

In `automatic-final`, limit Phase 2 to branches that can change the final recommendation. If many categories are material, prioritize the largest exposed category first, then add other branches whose measured impact is close enough to affect ranking or whose evidence may explain the largest category. Do not run speculative branches just because they exist.

`phase1_report.md` must include:

- `Scope`: input DBs, process/device/rank coverage, host/device time range.
- `Effective Compute`: compute kernel time and ratio.
- `Exposed Non-Effective Time`: uncovered communication, ordinary non-compute categories, and projection gap.
- `Device Stream Gap Ratio`: main compute stream gap ratio and device-level gap ratio from `device_timeline.py`. This is the key host-overhead indicator; flag a host-bound situation when the main-stream gap ratio is high.
- `Compute Gap Summary`: total compute-kernel gap, dominant coarse reasons, top relevant gaps.
- `Initial Situation`: one or more categories above, with evidence.
- `Recommended Or Selected Phase 2 Branches`: branches with evidence and priority.
- `Skipped Branches`: branches not run/recommended, with reasons.
- `Raw Tables`: baseline script output filenames and compact excerpts.
- `Artifacts`: analysis directory, generated DBs, logs, and report paths.

Do not run `gap_detail.py`, host-blocking trace, rank overlap checks, or host-window subphase analysis inside Phase 1 itself. Run those only inside Phase 2 branches.

## Phase 2: Branch Analysis

Goal: run selected branch analyses, write one detailed `phase2_<branch>_report.md` per completed branch, and save raw script/query outputs in the analysis directory.

Mode behavior:

- `interactive-phased`: run exactly the branch or branches selected by the user, then ask whether to run more branches or proceed to Phase 3.
- `automatic-final`: run the automatically selected branch set, then proceed to Phase 3.

Parallel execution:

- Run independent branches in parallel when their inputs and outputs do not conflict.
- Branches are parallel-safe when they only read DB inputs and write distinct output files such as `phase2_<branch>_report.md` and branch-specific logs/query outputs.
- Use Phase 1 priority to schedule work, but do not serialize independent branches unnecessarily.
- Do not run `host-window-subphase` in parallel unless the host window is already known and its outputs are isolated.
- If required input is missing, record the branch as blocked. In `interactive-phased`, ask for the missing input or another branch. In `automatic-final`, continue with remaining branches.

Every branch result must include:

- `Branch`: selected branch and why it was selected.
- `Method`: scripts and DB tables used.
- `Findings`: branch-specific metrics and dependency evidence.
- `Candidate Causes`: plausible causes with supporting evidence, counter-evidence, affected ranks/devices, estimated impact, confidence, and missing evidence.
- `Interpretation`: what the evidence explains and what remains uncertain.
- `Follow-up Suggestions`: optional extra branches or inputs that could reduce uncertainty.
- `Raw Tables`: script logs, JSON/text outputs, or query result files produced for the branch.
- `Artifacts`: branch report path and evidence files.

Cause handling:

- Do not force a single root cause.
- If one cause is clearly supported, mark it as dominant and explain why alternatives are weaker.
- If multiple causes remain plausible, report them with confidence and missing evidence.
- If evidence is insufficient, say unresolved and list the specific missing input needed to disambiguate.

### Branch: `effective-compute-breakdown`

Question: when effective compute dominates, which compute kernels consume time, and is the issue more work or slower execution?

Workflow:

1. Run `compute_breakdown.py` for the selected DBs.

   ```bash
   python3 "$SKILL_DIR/scripts/compute_breakdown.py" <cnperf_db> [<cnperf_db> ...] \
     --format json > "$ANALYSIS_DIR/compute_breakdown.json"
   python3 "$SKILL_DIR/scripts/compute_breakdown.py" <cnperf_db> [<cnperf_db> ...] \
     --format text > "$ANALYSIS_DIR/compute_breakdown.md"
   ```

2. Aggregate `device_task_kernel_data` rows where `isComputation=1`.
3. Report observed top compute kernel names by count, total time, average, and max duration.
4. If multiple DBs are involved, compare per-rank compute totals and top compute kernel counts/time.
5. If optional cluster computation CSV exists, compare profiler categories, FLOPs, and achieved throughput.
6. Separate "more work" from "slower hardware" by comparing count/FLOPs versus average duration/FLOPS.
7. Report whether compute optimization is likely worthwhile or whether non-effective time remains the bigger target.

Output contract:

- `Scope`: DB files, process/device coverage, optional cluster computation CSV usage.
- `Compute Summary`: total compute time, kernel count, unique compute kernel names, time share versus Phase 1 device time when available.
- `Top Compute Kernels`: `rank`, `kernel_name`, `count`, `total_ms`, `share_of_compute`, `avg_ms`, `max_ms`.
- `Per-Rank Compute`: `db/rank`, `process_id`, `device_id`, `compute_total_ms`, `kernel_count`, `top_kernel`, `top_kernel_total_ms`, skew versus fastest or median rank.
- `Candidate Causes`: more work, slower execution, rank skew, or balanced compute with evidence and confidence.
- `Interpretation` and `Next Step`.

Do not assume compute kernels are GEMM, FA, Conv, or elementwise.

### Branch: `communication-root-cause`

Question: why is uncovered communication exposed instead of hidden by compute?

Apply communication concepts from `references/profiling_concepts.md` before interpreting exposed communication.

Precondition: this branch requires multiple rank/card DBs. If only one card/rank is available (one DB, or one process+device, including a single-card torch trace captured with stack enabled), do not run this branch unless the user explicitly asked for communication analysis. Single-card communication is not analyzable: exposed communication is dominated by waiting for absent peers, so any single-card "communication cost" conclusion would be misleading. When blocked this way, record the branch as skipped, name the missing sibling rank/card DBs, and redirect to compute-gap, ordinary-non-compute, or compute-breakdown branches.

Guardrails (apply before drawing any conclusion):

- Do not conclude "intrinsic communication cost" from high exposed receive time alone.
- Do not rule out a rank as the slow-arriver because its own uncovered communication is low.
- Do not use shorter local device span or similar final device span as proof that a rank is not the bottleneck.
- Do not use PP/EP/Global labels as proof that ranks cannot affect each other.
- Do not rank suspects only by communication total; compare compute time, host-blocking time, launch progress, and overlap with other ranks' exposed communication.

Workflow:

1. Confirm uncovered communication is material from Phase 1.
2. Check input completeness.
   - If only one DB/device is available, this branch is blocked per the precondition above: report that cross-rank attribution is unsupported and stop unless the user explicitly asked to proceed with a single-card local breakdown.
   - Ask for sibling `cnperf_data_*.db` files, a directory containing all rank DBs, or optional `cluster_aggregation/step` CSVs before making cross-rank fast/slow card or slow-arriver conclusions.
3. Build communication breakdown from DB kernel rows by observed communication/non-compute kernel name: count, total/exposed time, average, max, top long events.
   Use `comm_breakdown.py`:

   ```bash
   python3 "$SKILL_DIR/scripts/comm_breakdown.py" <cnperf_db> [<cnperf_db> ...] \
     --format json > "$ANALYSIS_DIR/comm_breakdown.json"
   python3 "$SKILL_DIR/scripts/comm_breakdown.py" <cnperf_db> [<cnperf_db> ...] \
     --format text > "$ANALYSIS_DIR/comm_breakdown.md"
   ```

4. Must perform fast/slow card analysis.
   - For multiple DBs/devices, compare per-rank or per-card compute time, uncovered communication time, compute gap time, host-blocking/gap indicators, device span, launch progress, and top kernels.
   - Use `rank_compare.py` when multiple DBs/devices are available:

     ```bash
     python3 "$SKILL_DIR/scripts/rank_compare.py" <cnperf_db> [<cnperf_db> ...] \
       --format json > "$ANALYSIS_DIR/rank_compare.json"
     python3 "$SKILL_DIR/scripts/rank_compare.py" <cnperf_db> [<cnperf_db> ...] \
       --format text > "$ANALYSIS_DIR/rank_compare.md"
     ```

   - Identify fast cards/ranks and slow cards/ranks by progress and blocking evidence, not by communication total alone.
   - Report whether high-communication ranks are waiting ranks, slow ranks, or both.
   - If only one DB/device is available, explicitly mark fast/slow card analysis as blocked and list the missing sibling rank/card DBs or aligned cluster CSVs needed.
5. If a slow card/rank is possible, further locate the slow-card cause.
   - Compare suspected slow cards against fast cards by compute kernel totals/top kernels, kernel count, avg/max duration, compute gaps, host-blocking indicators, ordinary non-compute work, communication wait/exposure, launch progress, and device span.
   - Always test load imbalance explicitly: a card/rank may be slow because it has more compute work, more kernel launches, heavier top-kernel mix, larger sequence/batch/token work, or otherwise uneven assigned workload.
   - Distinguish "more work" from "same work but slower execution" by comparing compute total, kernel count, top-kernel time/count distribution, average/max duration of matching kernels, and optional cluster FLOPs/throughput when available.
   - Classify the slow-card cause as one or more candidates: load imbalance/more compute work, slower compute execution, host-side blocker, compute gap/notifier dependency, ordinary non-compute work, communication dependency/backpressure, delayed launch/progress, or unresolved.
   - For each candidate, include supporting evidence, counter-evidence, affected ranks/cards, estimated impact, confidence, and missing evidence.
6. Distinguish direct communication participants from E2E dependency suspects using direct operation, timeline, and boundary/backpressure evidence.
7. Test the slow-arriver hypothesis when communication is high on some ranks.
8. Separate intrinsic communication cost evidence from waiting evidence.
9. Use optional cluster communication CSVs as labels and magnitudes only; do not let group names replace timeline evidence.

Output contract:

- `Input Completeness`.
- `Communication Breakdown`.
- `Fast/Slow Card Analysis`: fast and slow rank/card classification, comparison metrics, waiting rank versus slow rank distinction, and blocked status if cross-rank inputs are missing.
- `Slow Card Cause Analysis`: required when a slow card/rank is possible; include candidate causes, evidence, counter-evidence, impact, confidence, and missing evidence.
- `Compute/Progress Comparison`.
- `E2E Dependency Check`.
- `Slow-Arriver Test`.
- `Intrinsic Communication Test`.
- `Candidate Causes`.
- `Interpretation` and `Next Step`.

### Branch: `ordinary-non-compute-root-cause`

Question: why is ordinary non-compute device work exposed?

Workflow:

1. Confirm uncovered memcpy, memset, atomic, or other ordinary non-compute work is material from Phase 1.
2. Aggregate relevant device tables by type, size if available, queue, count, total, average, and max duration.
3. Separate bulk H2D/D2H/D2D copies from host synchronization behavior when host API context is available.
4. Inspect host ranges around material copies or ordinary tasks: `pin_memory`, `copy_`, `to`, `_copy_from`, `__next__`.
5. If multiple DBs are involved, compare per-rank ordinary non-compute totals and skew.
6. Report whether ordinary non-compute work is dominant, hidden by compute, or a symptom of host sync/data pipeline behavior.

Output contract:

- `Scope`: DB files, process/device coverage.
- `Ordinary Non-Compute Breakdown`: `task_type`, `count`, `total_ms`, `share_of_device`, `avg_ms`, `max_ms` aggregated from device tables.
- `Top Ordinary Tasks`: largest individual memcpy/memset/atomic rows by duration with `correlationId` and queue.
- `Host Context`: host ranges and APIs temporally overlapping major ordinary tasks.
- `Per-Rank Ordinary Work`: `db/rank`, `process_id`, `device_id`, `ordinary_total_ms`, skew versus fastest or median rank (when multiple DBs available).
- `Candidate Causes`: bulk data transfer, host sync artifact, data pipeline bottleneck, or other.
- `Interpretation` and `Next Step`.

### Branch: `compute-gap-root-cause`

Question: why did compute kernels fail to start promptly?

Workflow:

1. Start from Phase 1 `gap_summary.py` reason breakdown and top gaps.
2. Select top or representative `prev_corr` / `next_corr` pairs.
3. Run `gap_detail.py` for selected pairs.

   ```bash
   db_stem=$(basename "<cnperf_db>" .db)
   python3 "$SKILL_DIR/scripts/gap_detail.py" <cnperf_db> \
     --prev-corr <prev> --next-corr <next> --invoke-threshold 100 --format text \
     > "$ANALYSIS_DIR/${db_stem}.gap_detail.<prev>.<next>.log" 2>&1
   ```

4. Interpret the dependency chain from the next compute kernel backward.
5. For `host_blocking`, trace the host-side blocker:
   - set `gap_start = prev_kernel.end`, `gap_end = next_kernel.start`
   - find the next kernel's `function_data` row by `correlationId`
   - use that row's `processId/threadId`
   - search same-thread `Internal_operation_range_data` and `function_data` from `gap_start` to `invoke.start`
   - use `Internal_op_range_relations` when a framework range has `extraId`
   - classify from actual observed framework ops and runtime APIs
6. For notifier waits, verify same-queue predecessor before matched notifier place. A wait/place match uses `processId + deviceId + notifierId + extra.unique_val`; `queueId` is not part of notifier identity.
7. If multiple DBs are involved and gap patterns differ by rank, check whether rank progress imbalance explains the gaps.
8. Report the dominant subtype: host-side blocker, notifier dependency, previous kernel, memcpy/atomic, communication source task, out-of-range, or unknown.
9. If the host-side blocker is per-kernel launch overhead (not a queue-sync wait), confirm it with the device-stream gap ratio from `device_timeline.py` (high main compute stream gap %). When the workload also uses torch.compile, hand off to the `compile-segmentation` cpp_wrapper check and apply its recommendation rule: enable `cpp_wrapper` when the trace signal says it is off, verify it when unconfirmed, and look elsewhere when it is already on; do not recommend graph capture alone.

Only attribute a host gap to framework-triggered host synchronization if the DB shows a concrete framework op triggering a synchronization API, such as:

```text
framework op
-> cnrtQueueSync
```

Memcpy/D2H APIs can be supporting evidence, but they are not required for this classification.

### Branch: `host-window-subphase`

Question: within a user-provided host time window, which subphases launched which kernels?

Workflow:

1. Replay `Internal_operation_range_data` for the relevant process, separated by `threadId`.
2. Identify main-thread high-level ranges from actual data, not fixed pattern names.
3. Treat other threads, such as dataloader or `pin_memory`, as parallel context.
4. Attribute launched kernels to subphases through `function_data.correlationId`.
5. Report per-subphase host duration, kernel count, compute/non-compute time, and top observed kernels.
6. Do not run automatic pattern clustering by default.

### torch.compile / triton branches

The next three branches target torch.compile/inductor workloads, mainly converted torch profiler traces. They read only DB inputs and write distinct output files, so they are parallel-safe with each other and with other branches. Their evidence comes from kernel names in `string_table`, compiled-region ranges in `Internal_operation_range_data`, and the per-kernel `args`/metadata preserved in `device_task_kernel_data.extra` (JSON). Always resolve names through this DB's `string_table`, and report observed names/metadata keys first instead of assuming fixed inductor naming.

Shared preconditions:

- If the DB has no compiled-region annotations and no `triton_*` kernel names, the workload likely does not use torch.compile. Skip all three branches and record them under `Skipped Branches`.
- Triton-fused kernels are identified by observed names such as `triton_poi_fused_*`, `triton_red_fused_*`, `triton_per_fused_*`, and `triton_tem_fused_*`. List the actual matched names before grouping; do not hardcode the set.

### Branch: `compile-segmentation`

Question: how does torch.compile partition the model into compiled regions, and do graph breaks or recompilations fragment otherwise-fusable work?

Workflow:

1. Run `compile_segmentation.py` for the selected DBs.

   ```bash
   python3 "$SKILL_DIR/scripts/compile_segmentation.py" <cnperf_db> [<cnperf_db> ...] \
     --format json > "$ANALYSIS_DIR/compile_segmentation.json"
   python3 "$SKILL_DIR/scripts/compile_segmentation.py" <cnperf_db> [<cnperf_db> ...] \
     --format text > "$ANALYSIS_DIR/compile_segmentation.md"
   ```

   The script reports the observed compiled-region inventory (names decoded through `string_table`), inside vs outside-region (eager) compute split, recompilation indicators, custom-op ranges that contain many simple `aten::` ops, the per-queue device-stream gap ratio, and the host-launch-overhead metrics. Report the observed region-name inventory first.
2. Read segmentation: device compute time and kernel count inside compiled regions vs outside (eager/graph-break), and whether work is fragmented across many small regions. The script attributes each kernel by temporal containment of its `function_data` launch within compiled-region ranges.
3. Read recompilation indicators (`TorchDynamo Cache Lookup`/guard ranges) as a sign of re-tracing on dynamic shapes.
4. cpp_wrapper check (trace signal first, device-stream gap second): read `host_launch_overhead.cpp_wrapper_signal` / `cpp_wrapper_signal` before making any inference.
   - `state=off` means the trace indicates Python wrapper / `cpp_wrapper` disabled. This can come from an explicit trace key or from Inductor `kernel_file` evidence such as generated `.py` files.
   - `state=on` means the trace indicates `cpp_wrapper` enabled. This can come from an explicit trace key or generated C++/shared-library style `kernel_file` evidence.
   - `state=unknown` means the trace did not carry a direct signal; only then infer wrapper mode from high main-stream gap ratio, small kernels, high `avg_launch_self_us`, and high `launch_self_to_compute_ratio`.
   - Always report the signal source and confidence. Do not write "无法从 trace 确认 cpp_wrapper" when `cpp_wrapper_signal.source` is `explicit_trace_metadata` or `kernel_file_extension`.
5. Identify the largest outside-region (eager) kernels from `top_outside_region_kernels`.
6. Read `custom_op_simple_aten`. If `has_issue=true`, promote it as an optimization candidate: a custom/user op is present but still executes many simple `aten::` pointwise/view/reduce/copy/allocation ops inside the wrapper, so those ops should be moved into the custom backend kernel or restructured to let Inductor fuse them. If `must_report=true` or the top row has `report_priority=high`, this is a final-report finding, not just branch detail. Cite the concrete `custom_op_name`, call count, nested simple aten count, average nested ops per call, and top nested `aten::` names.
7. Report whether segmentation is material: large compute time or many kernels outside compiled regions, frequent recompilation, custom-op simple-aten nesting, or many small fragmented regions.

Output contract:

- `Scope`: DB files, process/thread coverage, whether compiled-region annotations are present.
- `Compiled Region Inventory`: observed region names, region count, per-region host/device time.
- `Segmentation Summary`: segment count, graph-break count, device compute time inside vs outside compiled regions, kernel count inside vs outside.
- `Recompilation Indicators`: evidence of re-tracing/guards, if any.
- `Custom Op Simple Aten Nesting`: custom/user ops that wrap many simple `aten::` ops; include nested count, average per call, top nested ops, and whether it is a likely missed-fusion/custom-kernel optimization.
- `Host Launch Overhead / cpp_wrapper Check`: main compute stream gap ratio (key indicator), `avg_launch_self_us`, `launch_self_to_compute_ratio`, trace-confirmed or inferred wrapper mode, `cpp_wrapper_signal.source/confidence`, and the device-stream gap evidence.
- `Top Eager / Graph-Break Segments`: largest outside-region kernels (from `top_outside_region_kernels`).
- `Candidate Causes`: Python wrapper host launch overhead only when `cpp_wrapper_signal.state=off` or the mode is unconfirmed and gap metrics support it; custom op wrapping many simple aten ops, graph breaks fragmenting fusion, recompilation overhead, unsupported ops forcing eager fallback, dynamic shapes, or balanced/healthy compilation; each with evidence, counter-evidence, estimated impact, confidence, and missing evidence.
- `Interpretation` and `Next Step`.

Recommendation rule: for a host-bound torch.compile workload with large device kernel bubbles, when `cpp_wrapper_signal.state=off`, recommend enabling `cpp_wrapper` (inductor C++ wrapper codegen) to cut per-launch host overhead. When the state is `unknown`, recommend verifying/enabling it as a hypothesis. When the state is `on`, do not cite disabled `cpp_wrapper` as the root cause; investigate graph breaks, synchronization, tiny kernels, or host framework work. Do not recommend graph capture (CUDA graph / device-graph capture) as the only remedy — it is complementary unless direct trace evidence shows capture is the missing mechanism.

### Branch: `triton-fusion-coverage`

Question: which compute kernels were not fused into triton kernels, and how much device time runs in non-fused/library/eager kernels that inductor could potentially fuse?

Workflow:

1. Run `triton_fusion_coverage.py` for the selected DBs.

   ```bash
   python3 "$SKILL_DIR/scripts/triton_fusion_coverage.py" <cnperf_db> [<cnperf_db> ...] \
     --format json > "$ANALYSIS_DIR/triton_fusion_coverage.json"
   python3 "$SKILL_DIR/scripts/triton_fusion_coverage.py" <cnperf_db> [<cnperf_db> ...] \
     --format text > "$ANALYSIS_DIR/triton_fusion_coverage.md"
   ```

   It classifies compute kernels (`isComputation=1`) by name from `string_table` into triton-fused (`triton_*fused*`), other triton (rare), and non-triton/library/eager. It also groups kernels into Inductor fusion families (`pointwise`, `reduce`, `library_or_gemm`, `communication`, `triton_other`, `other`) and reports fused/unfused time for each family, highlighted unfused pointwise/reduce candidates, top non-fused kernels, and per-process/device fusion coverage.
2. Read the fusion-coverage ratio (triton-fused compute time / total compute time) and the top non-fused kernels as fusion-miss / fallback candidates.
3. Inspect `Inductor Fusion Granularity` first. If `pointwise` has non-zero unfused time, highlight it as the strongest missed-fusion signal; if `reduce` has non-zero unfused time, highlight it as a secondary fusion/reduction candidate. Treat library/GEMM/conv families as likely intended fast paths unless other evidence says otherwise.
4. Cross-reference with `compile-segmentation` when available: are highlighted non-fused pointwise/reduce kernels concentrated in eager/graph-break segments?
5. If multiple DBs are involved, compare the fusion-coverage ratio and pointwise/reduce unfused time across ranks (the script emits `per_process_device`).
6. Report whether raising fusion coverage is a worthwhile target versus other exposed time.

Guardrail: do not assume every non-triton kernel is a fusion defect. Vendor GEMM/conv/library compute primitives are often the intended fast path. Flag fusion misses primarily for elementwise/pointwise/reduction kernels left unfused, not for library compute primitives.

Output contract:

- `Scope`: DB files, process/device coverage.
- `Fusion Coverage Summary`: fused vs non-fused compute time and ratio, kernel counts per class.
- `Inductor Fusion Granularity`: family-level fused/unfused time; explicitly call out unfused `pointwise` and `reduce` time. A non-zero unfused pointwise row must be highlighted.
- `Highlighted Unfused Pointwise/Reduce Candidates`: top kernels whose names look pointwise/reduce-like but did not appear as triton-fused, with impact and the script's reason.
- `Top Non-Fused Kernels`: `kernel_name`, `count`, `total_ms`, `share_of_compute`, `avg_ms`, `max_ms`.
- `Segment Correlation`: whether non-fused kernels cluster in eager/graph-break segments (link to `compile-segmentation` if run).
- `Per-Rank Fusion Coverage`: fusion ratio per `db/rank` and skew (when multiple DBs available).
- `Candidate Causes`: unsupported op/fallback, intentional library primitive, graph break, small-op fusion miss, or already well-fused; with evidence, counter-evidence, impact, confidence, and missing evidence.
- `Interpretation` and `Next Step`.

### Branch: `triton-kernel-efficiency`

Question: for triton kernels that carry generated source (`output_code`) and IO-efficiency metadata, which fused kernels are memory-IO inefficient and why?

Precondition: requires per-kernel `output_code` and IO-efficiency fields inside `device_task_kernel_data.extra` (JSON). This is optional inductor/profiler enrichment and is frequently absent. If neither metadata is present on any triton kernel, skip this branch and record it under `Skipped Branches` with the missing-metadata reason.

`io_efficiency` semantics: this value is NOT a normalized 0–1 ratio or percentage. It is a bandwidth-equivalent value — the kernel's effective/folded bandwidth (a bandwidth quantity, e.g. GB/s). Judge efficiency by comparing it against the device peak bandwidth, not by treating it as a fraction. A low folded bandwidth relative to peak indicates memory-IO inefficiency. Never apply `1 - io_efficiency`.

Workflow:

1. Run `triton_kernel_efficiency.py` for the selected DBs, dumping `output_code` into the analysis directory.

   ```bash
   python3 "$SKILL_DIR/scripts/triton_kernel_efficiency.py" <cnperf_db> [<cnperf_db> ...] \
     --dump-dir "$ANALYSIS_DIR/triton_output_code" --format json \
     > "$ANALYSIS_DIR/triton_kernel_efficiency.json"
   python3 "$SKILL_DIR/scripts/triton_kernel_efficiency.py" <cnperf_db> [<cnperf_db> ...] \
     --dump-dir "$ANALYSIS_DIR/triton_output_code" --format text \
     > "$ANALYSIS_DIR/triton_kernel_efficiency.md"
   ```

   If the script reports `has_io_metadata=false`, skip this branch and record it under `Skipped Branches` with the missing-metadata reason. The script reports the observed metadata keys first, treats `io_efficiency` as folded bandwidth, and uses the MLU-model **theoretical (peak) bandwidth** — MLU590 → 2000, MLU580 → 1200 (GB/s) — falling back to `meta_information` `deviceInfo.m_dev_basic_info.max_bandwidth` only when the model is unknown. It computes `bandwidth_utilization = io_efficiency / peak_bandwidth` when comparable, and ranks by `improvement_target = total_ms * (1 - bandwidth_utilization)` (falling back to lowest folded bandwidth weighted by `total_ms` when utilization is unavailable). Check `peak_bandwidth_source`; if utilization looks impossible (e.g. > 1), treat units as mismatched and rely on the fallback ranking.
2. For the top low-bandwidth kernels, open the dumped `output_code` files (under `triton_output_code/`) and characterize the access pattern: tensor shapes/strides, masking, non-contiguous or gather/scatter access, reduction shape, grid/block configuration, and load/store counts. Do not paste full generated source into the main report; cite the file.
3. Classify the low-bandwidth cause per kernel: memory-bound small kernel, non-coalesced/strided access, redundant recompute, poor tiling/grid, register spill, or already efficient (folded bandwidth near peak).
4. If multiple DBs are involved, compare per-rank folded bandwidth for the same kernel names.

Output contract:

- `Scope`: DB files, whether `output_code` and IO-efficiency metadata are present, and the observed metadata key names.
- `IO Efficiency Summary`: number of triton kernels with metadata, distribution of folded/effective bandwidth (`io_efficiency`), the device peak bandwidth and its units, and bandwidth utilization when computable. State explicitly that `io_efficiency` is a bandwidth value, not a percentage.
- `Top Low-Bandwidth Kernels`: `kernel_name`, `count`, `total_ms`, `io_efficiency` (folded bandwidth with units), `bandwidth_utilization` (`io_efficiency / peak_bandwidth`, when available), and `improvement_target`.
- `Output Code Findings`: per top kernel, the access-pattern characterization with the `output_code` excerpt filename.
- `Candidate Causes`: with evidence, counter-evidence, impact, confidence, and missing evidence.
- `Interpretation` and `Next Step`.

## Phase 3: Final Synthesis

Goal: synthesize Phase 1 and completed Phase 2 branches into a concise `report.md`, provide prioritized recommendations, and reference raw evidence artifacts.

Enter Phase 3:

- In `interactive-phased`, after the user asks to conclude or declines more branches.
- In `automatic-final`, after automatic branch execution finishes.

Workflow:

1. List completed inputs: Phase 1 baseline plus each completed branch.
2. Merge evidence by causal path, not by script output.
3. Separate confirmed findings from hypotheses.
4. Before pruning to the final 2-4 findings, scan `compile_segmentation.json` for `custom_op_simple_aten.must_report=true`. When present, reserve one finding and one action row for the custom-op/simple-aten issue; this is a structural missed-fusion signal and should not be buried because its host-range duration is smaller than other exposed-time metrics.
5. Estimate potential benefit using measured exposed time or skew. If benefits overlap, state that they are not additive.
6. Prioritize recommendations by expected impact, confidence, and implementation scope.
7. If custom-op/simple-aten is reserved, phrase the action as moving repeated simple `aten::` pointwise/view/reduce/copy/allocation work into the custom backend kernel, or restructuring the wrapper so Inductor can see and fuse it.
8. Call out missing evidence and which branch or input would close it.
9. Do not append raw table dumps to `report.md`. Keep audit details in stage reports or `evidence_summary.md`, and reference full output filenames.

Final `report.md` structure:

Use the exact structure defined in `Final Report Contract`:

1. `# AI 性能分析报告`
2. `## 结论概览`
3. `## 关键指标`
4. `## 优先行动`
5. `## 不确定性与下一步`
6. `## 产物`

Output contract:

- `结论概览`: 2-4 prioritized findings, usually 3. Use one `### 发现 N：short title` subsection per finding with separate `**结论：**`, `**证据：**`, and `**建议：**` paragraphs; each paragraph must be one short sentence. Merge host gap / launch overhead / `cpp_wrapper` into one finding when they are the same causal path. If `custom_op_simple_aten.must_report=true`, include a finding titled around "自定义算子内部仍有大量简单 aten 算子" or equivalent.
- `关键指标`: compact table with 4-6 rows: metric, measured value/share, source artifact, and interpretation.
- `优先行动`: 3-5 rows: priority, action, expected benefit, confidence, risk/cost, and validation method. If `custom_op_simple_aten.must_report=true`, include an action for the custom op using the exact custom op name and top nested aten names.
- `不确定性与下一步`: missing data or unresolved hypotheses, plus the branch/input needed to resolve each and one recommended first next step.
- `产物`: analysis directory, report path, stage report paths, generated DB paths, `evidence_summary.md`, and logs used as evidence.

## Validation And Failure Handling

- Use `python3`, not `python`.
- If a table is missing, state what is unavailable and continue with remaining evidence.
- If `string_table` is missing, report `nameId=...`.
- If multiple processes/devices are present in one DB, call that out and filter when needed.
- If cluster CSVs are absent, continue from DB tables and mention that rank-level profiler categories are unavailable.
- Keep thresholds as triage aids, not hard truth. Prefer measured ratios and dependency evidence.
