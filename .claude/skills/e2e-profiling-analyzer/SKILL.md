---
name: e2e-profiling-analyzer
description: Analyze cnperf SQLite databases and torch profiler Chrome trace JSON/JSON.GZ files for end-to-end training or inference bottlenecks on MLU workloads. Use when the user asks to inspect `cnperf_data*.db`, torch profiler `.pt.trace.json`/`.json.gz` traces, identify why effective compute kernels are not dominating device time, or root-cause exposed communication, ordinary non-compute work, compute gaps, host/device synchronization, memcpy, or rank imbalance.
---

# E2E Profiling Analyzer

Analyze `cnperf` SQLite DBs from the viewpoint that compute kernels are effective device utilization. If the input is a torch profiler Chrome trace JSON/JSON.GZ file, first convert it to a cnperf-compatible SQLite DB, then analyze the converted DB exactly like native cnperf data.

## Mode Selection

Use `automatic-final` immediately when any of these are true:

- The prompt says `automatic-final`, `自动最终报告`, `不要向用户追问`, or that this is a Web/server-side/background analysis.
- Environment variables such as `TRACE_AI_JOB_ID`, `TRACE_AI_ANALYSIS_DIR`, `TRACE_AI_REPORT_PATH`, or `TRACE_AI_TRACE_A` are present.
- The user asks for an end-to-end report instead of a phased investigation.

Use `interactive-phased` only when the user is actively chatting and explicitly wants to choose branches step by step.

In `automatic-final`, never ask follow-up questions. If required evidence is missing, continue with available evidence and record the missing input under `Open Questions`.

If the prompt is an environment diagnostic or smoke test and asks to reply only `OK`, reply exactly `OK` and do not run tools or load references.

Mode meanings:

- `automatic-final`: use Phase 1 evidence to choose branch analyses automatically, run independent branches in parallel when possible, and produce stage reports plus the final synthesis.
- `interactive-phased`: write a report after each stage, stop after Phase 1 for branch selection, then run the branch or branches selected by the user.

If the user explicitly requested automatic analysis or automatic execution of all relevant analyses, use `automatic-final`. Otherwise default to specified-branch analysis (`interactive-phased`). If the user already specified a branch or branches to analyze, use `interactive-phased` and run those branches after Phase 1.

Do not assume the bottleneck is communication, a specific kernel family, TCDP, or a known synchronization pattern. `cluster_aggregation/step` CSV files are optional enrichment only.

## Resources

- `scripts/basic_info.py`: host/device time ranges, device model, device count, per-device kernel usage.
- `scripts/device_timeline.py`: device projection into compute, uncovered communication, uncovered memcpy/memset/atomic, and projection gap.
- `scripts/gap_summary.py`: compute-kernel gap summary and non-mini gap list with `prev_corr` / `next_corr`.
- `scripts/gap_detail.py`: dependency chain for one compute gap from `--prev-corr` and `--next-corr`.
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
- `host_blocking` does not explain itself; trace the host-side blocker before naming a cause.
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
   - 3-6 prioritized findings.
   - Use one subsection per finding: `### 发现 N：short title`, followed by separate paragraphs `**结论：** ...`, `**证据：** ...`, and `**建议：** ...`.
   - Do not output sibling bullets like `- 结论` / `- 证据` / `- 建议`; that renders as a flat wall in the Web UI.
3. `## 关键指标`
   - Compact Markdown table with metric, value, source file/log, and interpretation.
4. `## 主要发现`
   - Prioritized findings with evidence, counter-evidence, estimated impact, confidence, and affected ranks/devices.
5. `## 优化建议`
   - Prioritized actions with expected benefit, implementation cost, risk, and validation method.
6. `## 不确定性与下一步`
   - Missing evidence and the next check that would reduce uncertainty.
7. `## 产物`
   - Generated DBs, stage reports, evidence logs, and analysis directory.

Keep the final report concise enough for Web reading. Move large raw tables and long logs into
artifact files such as `evidence_summary.md`, then cite those files from the report.

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

Resolve `SKILL_DIR` to this skill directory's absolute path. Do not call scripts from repository-level `tools/`, `.trae/`, or any path outside this skill.

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

After conversion, run `basic_info.py` on at least one generated DB to verify it opens and meta/device information is readable.

## Output Files

Write these Markdown reports in both modes:

- `phase1_report.md`: Phase 1 baseline analysis and branch recommendation/selection.
- `phase2_<branch>_report.md`: detailed report for each completed branch.
- `report.md`: final synthesis report after automatic branch execution completes, or after the user asks to conclude/declines more branches in `interactive-phased` mode.

Do not duplicate full per-branch detailed reports inside `report.md`; reference their filenames from `产物`. Keep raw table excerpts in stage reports or `evidence_summary.md`, and cite those artifacts from the final report.

Stage reports and logs should include enough filenames and raw excerpts for later audit.

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

- Run or recommend `communication-root-cause` when uncovered communication is material, top compute gaps are communication-related, or multi-rank evidence suggests waiting/slow-arriver behavior.
- Run or recommend `ordinary-non-compute-root-cause` when uncovered memcpy, memset, atomic, or other ordinary non-compute time is material, or top gaps point to memcpy/ordinary device work.
- Run or recommend `compute-gap-root-cause` when `gap_summary.py` shows material total gap time, large individual gaps, or host/notifier/previous-task gap reasons.
- Run or recommend `effective-compute-breakdown` when effective compute dominates total device time or Phase 1 suggests compute imbalance across ranks/devices.
- Run `host-window-subphase` only when the user provided a host time window or explicitly asked for host-window subphase analysis.

In `automatic-final`, limit Phase 2 to branches that can change the final recommendation. If many categories are material, prioritize the largest exposed category first, then add other branches whose measured impact is close enough to affect ranking or whose evidence may explain the largest category. Do not run speculative branches just because they exist.

`phase1_report.md` must include:

- `Scope`: input DBs, process/device/rank coverage, host/device time range.
- `Effective Compute`: compute kernel time and ratio.
- `Exposed Non-Effective Time`: uncovered communication, ordinary non-compute categories, and projection gap.
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

1. Aggregate `device_task_kernel_data` rows where `isComputation=1`.
2. Report observed top compute kernel names by count, total time, average, and max duration.
3. If multiple DBs are involved, compare per-rank compute totals and top compute kernel counts/time.
4. If optional cluster computation CSV exists, compare profiler categories, FLOPs, and achieved throughput.
5. Separate "more work" from "slower hardware" by comparing count/FLOPs versus average duration/FLOPS.
6. Report whether compute optimization is likely worthwhile or whether non-effective time remains the bigger target.

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

Workflow:

1. Confirm uncovered communication is material from Phase 1.
2. Check input completeness.
   - If only one DB/device is available and the workload is multi-rank, report that cross-rank attribution is unsupported.
   - Ask for sibling `cnperf_data_*.db` files, a directory containing all rank DBs, or optional `cluster_aggregation/step` CSVs before making cross-rank fast/slow card or slow-arriver conclusions.
   - Continue with single-rank local communication breakdown only if the user cannot provide more inputs or explicitly asks to proceed.
3. Build communication breakdown from DB kernel rows by observed communication/non-compute kernel name: count, total/exposed time, average, max, top long events.
4. Must perform fast/slow card analysis.
   - For multiple DBs/devices, compare per-rank or per-card compute time, uncovered communication time, compute gap time, host-blocking/gap indicators, device span, launch progress, and top kernels.
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

Guardrails:

- Do not conclude "intrinsic communication cost" from high exposed receive time alone.
- Do not rule out a rank as the slow-arriver because its own uncovered communication is low.
- Do not use shorter local device span or similar final device span as proof that a rank is not the bottleneck.
- Do not use PP/EP/Global labels as proof that ranks cannot affect each other.
- Do not rank suspects only by communication total; compare compute time, host-blocking time, launch progress, and overlap with other ranks' exposed communication.

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

## Phase 3: Final Synthesis

Goal: synthesize Phase 1 and completed Phase 2 branches into `report.md`, provide prioritized recommendations, and append raw table evidence.

Enter Phase 3:

- In `interactive-phased`, after the user asks to conclude or declines more branches.
- In `automatic-final`, after automatic branch execution finishes.

Workflow:

1. List completed inputs: Phase 1 baseline plus each completed branch.
2. Merge evidence by causal path, not by script output.
3. Separate confirmed findings from hypotheses.
4. Estimate potential benefit using measured exposed time or skew. If benefits overlap, state that they are not additive.
5. Prioritize recommendations by expected impact, confidence, and implementation scope.
6. Call out missing evidence and which branch or input would close it.
7. Append raw table information from script outputs and query result files. Include compact raw excerpts sufficient to audit claims, and reference full output filenames.

Final `report.md` structure:

Use the exact structure defined in `Final Report Contract`:

1. `# AI 性能分析报告`
2. `## 结论概览`
3. `## 关键指标`
4. `## 主要发现`
5. `## 优化建议`
6. `## 不确定性与下一步`
7. `## 产物`

Output contract:

- `结论概览`: 3-6 prioritized findings. Use one `### 发现 N：short title` subsection per finding with separate `**结论：**`, `**证据：**`, and `**建议：**` paragraphs.
- `关键指标`: compact table with metric, measured value/share, source artifact, and interpretation.
- `主要发现`: status, cause/hypothesis, evidence, counter-evidence, affected ranks/devices, estimated impact, confidence, and overlap/non-additivity notes.
- `优化建议`: priority, action, expected benefit, confidence, evidence link, risk/cost, and validation method.
- `不确定性与下一步`: missing data or unresolved hypotheses, plus the branch/input needed to resolve each and one recommended first next step.
- `产物`: analysis directory, report path, stage report paths, generated DB paths, `evidence_summary.md`, and logs used as evidence.

## Validation And Failure Handling

- Use `python3`, not `python`.
- If a table is missing, state what is unavailable and continue with remaining evidence.
- If `string_table` is missing, report `nameId=...`.
- If multiple processes/devices are present in one DB, call that out and filter when needed.
- If cluster CSVs are absent, continue from DB tables and mention that rank-level profiler categories are unavailable.
- Keep thresholds as triage aids, not hard truth. Prefer measured ratios and dependency evidence.
