---
name: e2e-profiling-comparator
description: Use when comparing E2E profiling captures from different versions, devices, or configurations to find where the current run is worse than a chosen baseline, using per-file breakdown tables from selected cnperf DB time ranges.
---

# E2E Profiling Comparator

Use this skill to compare profiling captures by collecting the same breakdown tables for a baseline and current run, then reasoning from those tables. The main objective is to find where current is slower, heavier, or less efficient than baseline; current advantages can be noted briefly.

## Web / Automatic Mode

Use non-interactive automatic mode immediately when any of these are true:

- The prompt says `automatic-final`, `自动最终报告`, `不要向用户追问`, or that this is a Web/server-side/background analysis.
- Environment variables such as `TRACE_AI_JOB_ID`, `TRACE_AI_ANALYSIS_DIR`, `TRACE_AI_REPORT_PATH`, `TRACE_AI_TRACE_A`, or `TRACE_AI_TRACE_B` are present.

In this mode, never ask follow-up questions. Treat trace A as baseline, trace B as current, and use `Delta = B - A`. If required evidence is missing, continue with available evidence and record the missing input under `Open Questions`.

Default to this automatic mode when the user has not explicitly asked for an interactive, step-by-step comparison. Only pause for input when the user wants to choose the baseline/current mapping or analysis window interactively.

If the prompt is an environment diagnostic or smoke test and asks to reply only `OK`, reply exactly `OK` and do not run tools or load references.

## Final Report Contract

For both Web/server-side automatic runs and interactive final synthesis, produce one stable,
user-facing final report:

- Write the final user-visible report to `$TRACE_AI_REPORT_PATH` when that environment variable is set.
- Also write the same final report to `report.md` in the current working directory.
- Print the same final report to stdout. Do not print tool logs, raw command output, prompt text, or progress narration to stdout.
- Save supporting evidence as separate files in the analysis directory, such as `baseline.tables.json`, `current.tables.json`, `comparison_evidence.md`, and conversion logs.
- If analysis cannot proceed because a trace file, DB table, Python dependency, or tool permission is missing, write a concise failure report instead of a partial or fabricated comparison.
- Prefer Chinese report text when the request is Chinese. Keep structural field labels (table headers, metric names emitted by the scripts) in English so the evidence stays machine-checkable, but write the narrative (findings, conclusions, recommendations) in the request's language.

The final `report.md` must use this exact high-level structure:

1. `# AI 对比分析报告`
2. `## 结论概览`
   - 2-4 prioritized findings focused on regressions first, then meaningful improvements; prefer 3 unless the evidence clearly needs more.
   - Use one subsection per finding: `### 发现 N：short title`, followed by separate paragraphs `**结论：** ...`, `**证据：** ...`, and `**建议：** ...`.
   - Keep each `结论` / `证据` / `建议` paragraph to one short sentence. Merge overlapping regressions instead of repeating the same delta from multiple tables.
   - Treat device gap, host launch overhead, and `cpp_wrapper` mode as one causal chain when they explain the same A/B regression; do not split them into separate findings.
   - Do not output sibling bullets like `- 结论` / `- 证据` / `- 建议`; that renders as a flat wall in the Web UI.
3. `## 对比口径`
   - Baseline/current files, selected windows, devices, and `Delta = B - A`.
4. `## 关键 Delta`
   - Compact Markdown table with metric, A, B, delta, interpretation, and source.
5. `## 优先行动`
   - Prioritized actions with expected benefit, implementation cost, risk, and validation method.
6. `## 不确定性与下一步`
   - Missing evidence and the next check that would reduce uncertainty.
7. `## 产物`
   - Converted DBs, collected table JSON files, evidence logs, and analysis directory.

Default to a concise Web report. Target no more than 1500 Chinese characters before the `产物`
section. Do not duplicate a full `主要回退与原因假设` section after `结论概览`; put detailed
per-table evidence, raw deltas, long stack traces, and script logs into artifacts, then cite those
filenames from the report.
If graph capture, multi-stream execution, driver/runtime upgrades, or other environment changes are
only plausible follow-ups without direct A/B evidence, keep them in `不确定性与下一步` instead of
promoting them to top findings or primary actions.

## Resources

- `scripts/collect_profile_tables.py`: collect host summaries and device breakdown tables for one cnperf DB over one explicit time range. Also emits the device-stream (queue) gap ratio and torch.compile/triton sections: fusion coverage, Inductor fusion granularity by kernel family (pointwise/reduce/library/etc.), compile segmentation + host-launch-overhead (cpp_wrapper signal), and triton kernel IO efficiency (`io_efficiency` as folded bandwidth vs MLU-model theoretical peak).
- `scripts/compare_profile_tables.py`: compare baseline/current table JSON files and emit A/B/delta evidence, including a direction-aware host-overhead delta (device-stream gap, launch self-time), fusion/segmentation deltas, pointwise/reduce unfused fusion-granularity deltas, and per-kernel IO-efficiency deltas.
- `scripts/torch_trace_to_cnperf_db.py`: convert PyTorch profiler Chrome trace JSON/JSON.GZ to a cnperf-compatible SQLite DB before table collection.
- `references/db_schema.md`: load when writing direct DB queries or interpreting table fields.
- `references/profiling_concepts.md`: load before turning observed differences into causal hypotheses.

## Workflow

1. Create or choose the analysis directory.
   - In Web/server-side automatic mode, use `ANALYSIS_DIR="${TRACE_AI_ANALYSIS_DIR:-$PWD}"` and `REPORT_MD="${TRACE_AI_REPORT_PATH:-$ANALYSIS_DIR/report.md}"`; do not create an extra nested timestamp directory.
   - In interactive local mode, before converting inputs, collecting tables, or writing the comparison report, create one directory under the current working directory.
   - For interactive local mode, use the exact directory name format `e2e_profiling_compare_YYYYMMDD_HHMMSS`; if a collision occurs, append `_NN`.
   - Put all generated artifacts directly in this one directory: converted DBs, conversion report JSON files, table collection outputs, script stdout/stderr logs, ad hoc query outputs, and the final Markdown report.
   - The final report path is `$TRACE_AI_REPORT_PATH` in Web mode, otherwise `<analysis_dir>/report.md`.
2. Define the comparison goal.
   - Version regression: baseline = known-good run; current = run under investigation.
   - Device comparison: baseline = expected-aligned or faster device; current = device under analysis.
   - Configuration experiment: baseline = control configuration; current = experiment configuration.
3. Prepare each capture upstream.
   - Resolve `SKILL_DIR` to this skill directory's absolute path before running any command. Derive it from the path this skill was loaded from (`<SKILL_DIR>/SKILL.md`); if unavailable, locate it once with `SKILL_DIR=$(dirname "$(find "$HOME/.claude" "$PWD" -type f -path '*/e2e-profiling-comparator/SKILL.md' 2>/dev/null | head -n1)")` and verify `SKILL_DIR/scripts/collect_profile_tables.py` exists.
   - Convert JSON/JSON.GZ traces to cnperf-compatible DB in the temporary analysis directory when needed.
   - After conversion, verify each generated DB opens and is readable (e.g. run `collect_profile_tables.py` once, or a quick range query). If a converted DB fails this check, record it as blocked and do not feed a silently corrupt DB into the comparison.
   - Determine raw, preparation, stable, or manually selected analysis windows.
   - Inspect basic information such as device model, card count, driver/cnperf version, host environment, and card usage.
4. Collect breakdown tables independently for each DB and selected time range.
   - Run `collect_profile_tables.py` once per DB/range.
   - Write each collection output to the temporary analysis directory. Prefer JSON for machine-readable comparison and keep text logs when useful for review.
   - Use the same range semantics across files when comparing.
   - The script includes rows whose `start` is in `[start_ms, end_ms)` and clips duration at `end_ms`.
5. Build the delta evidence.
   - Run `compare_profile_tables.py` on the baseline/current JSON outputs.
   - Use the generated JSON or Markdown delta as primary comparison evidence, then inspect raw per-file tables for details.
6. Compare from coarse to fine.
   - Start from upstream E2E windows: raw, preparation, stable.
   - Focus on categories where current is worse than baseline; only briefly record current advantages.
   - Use Device Breakdown Overview to locate current regressions across compute, communication, memcpy, compute gap, pure gap, and other activity.
   - Check the Host Overhead Delta early: if the main compute stream gap ratio increased, the regression is host-bound; follow the cpp_wrapper guidance for torch.compile workloads and report the trace signal source/confidence.
   - When the workload uses torch.compile/triton, use the fusion-coverage, Inductor fusion-granularity, compile-segmentation, and triton kernel IO-efficiency deltas to locate fusion loss, unfused pointwise/reduce kernels, eager fallback, recompilation, or per-kernel bandwidth regressions.
   - Always compare Compute Kernel Summary at name level, because device kernel cost is usually the primary investigation target.
   - Enter other name-level tables for additional regressed categories.
   - Use Host Function Summary and Host Internal Operation Summary as lightweight follow-up signals; inspect sync-like function names there when needed.
7. Generate the final Markdown analysis document.
   - Write the report to `$REPORT_MD` and, in Web/server-side automatic mode, also to `report.md` in the current working directory.
   - Use the exact structure from `Final Report Contract`.
   - Put large raw tables and detailed per-table deltas in supporting artifacts, then cite those artifact filenames from `## 产物`.

## Commands

Create or choose the analysis directory before any analysis.

For Web/server-side automatic analysis:

```bash
SKILL_DIR=<absolute-path-to-this-skill>
ANALYSIS_DIR="${TRACE_AI_ANALYSIS_DIR:-$PWD}"
REPORT_MD="${TRACE_AI_REPORT_PATH:-$ANALYSIS_DIR/report.md}"
```

For interactive local analysis:

```bash
SKILL_DIR=<absolute-path-to-this-skill>
ANALYSIS_DIR="e2e_profiling_compare_$(date +%Y%m%d_%H%M%S)"
mkdir "$ANALYSIS_DIR"
REPORT_MD="$ANALYSIS_DIR/report.md"
```

Convert a PyTorch profiler trace when the input is `.json`, `.json.gz`, or `.pt.trace.json.gz`:

```bash
python3 "$SKILL_DIR/scripts/torch_trace_to_cnperf_db.py" \
  capture.pt.trace.json.gz \
  --out "$ANALYSIS_DIR/baseline.db" \
  --report "$ANALYSIS_DIR/baseline.convert_report.json" \
  > "$ANALYSIS_DIR/baseline.convert.log" 2>&1
```

Useful converter options:

- `--comm-regex <regex>`: override the communication-kernel name heuristic.
- `--process-id <id>`: force the generated process ID.
- `--progress <N>`: print progress every N trace events.
- `--max-events <N>`: limit conversion for smoke testing.

The converter requires Python module `simdjson` from package `pysimdjson`. If the active Python cannot import it, create a local venv and install there:

```bash
python3 -m venv .venv-trace-convert
.venv-trace-convert/bin/python -m pip install pysimdjson
.venv-trace-convert/bin/python "$SKILL_DIR/scripts/torch_trace_to_cnperf_db.py" \
  capture.pt.trace.json.gz \
  --out "$ANALYSIS_DIR/baseline.db" \
  --report "$ANALYSIS_DIR/baseline.convert_report.json" \
  > "$ANALYSIS_DIR/baseline.convert.log" 2>&1
```

Collect breakdown tables from a DB and selected time range:

```bash
python3 "$SKILL_DIR/scripts/collect_profile_tables.py" \
  capture.db --start-ms <start> --end-ms <end> --format json \
  > "$ANALYSIS_DIR/baseline.tables.json" 2> "$ANALYSIS_DIR/baseline.tables.log"
```

`collect_profile_tables.py` only accepts cnperf DB input. JSON conversion, E2E window selection, and basic information collection are upstream steps.

Compare collected baseline/current tables:

```bash
python3 "$SKILL_DIR/scripts/compare_profile_tables.py" \
  "$ANALYSIS_DIR/baseline.tables.json" "$ANALYSIS_DIR/current.tables.json" \
  --format json > "$ANALYSIS_DIR/comparison_delta.json"
python3 "$SKILL_DIR/scripts/compare_profile_tables.py" \
  "$ANALYSIS_DIR/baseline.tables.json" "$ANALYSIS_DIR/current.tables.json" \
  --format text > "$ANALYSIS_DIR/comparison_evidence.md"
```

## Output Tables

### Device Breakdown Overview

One table for the selected range.

| Category | Total ms | Count | Avg ms | Max ms | Range share |
|---|---:|---:|---:|---:|---:|

Required categories:

- `compute kernel`: `device_task_kernel_data` where `isComputation = 1`.
- `communication kernel`: `device_task_kernel_data` where `isComputation = 0`.
- `memcpy`: `device_task_memcpy_data`.
- `compute gap`: exposed positive intervals after merging overlapping compute kernels on the same process/device timeline.
- `pure gap`: time in the selected range not covered by known device-side activity used in this overview.
- `other activity`: known device-side activity outside compute/communication/memcpy, currently notifier, atomic operation, and memset.

`Range share` is `category total ms / selected range duration ms`.

### Compute Kernel Summary

Group compute kernels by observed kernel name.

| Kernel name | Total ms | Count | Avg ms | P90 ms | Max ms | Share |
|---|---:|---:|---:|---:|---:|---:|

Sort by `Share` descending. After cumulative share reaches 95%, merge remaining kernel names into `other`.
`collect_profile_tables.py` already applies this 95% coverage compaction, so use the emitted rows directly during analysis.

### Communication Kernel Summary

Group communication kernels by observed kernel name.

| Kernel name | Total ms | Uncovered ms | Count | Avg ms | P90 ms | Max ms | Share | Uncovered share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

`Uncovered ms` is communication time not overlapped by compute kernels. It estimates communication time more visible to E2E.
`collect_profile_tables.py` already applies 95% share compaction and merges the remainder into `other`; do not compress the emitted rows again.

### Memcpy Summary

Group memcpy by copy direction or type.

| Copy type | Total ms | Uncovered ms | Count | Avg ms | P90 ms | Max ms | Total bytes | Avg bytes | Bandwidth GB/s | Share | Uncovered share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

`bytes` and `bandwidth` are available only when the DB exposes size information.
`collect_profile_tables.py` already applies 95% share compaction and merges the remainder into `other`; do not compress the emitted rows again.

### Device Stream Gap Ratio

Per device stream (queue) within the selected range.

| Queue | Span ms | Busy ms | Gap ms | Gap% | Compute ms | Main |
|---:|---:|---:|---:|---:|---:|:--:|

Plus two scalars: `main compute stream gap %` (the gap ratio of the busiest compute queue) and `device-level gap %`. The **main compute stream gap ratio is the key host-overhead indicator** — a high value means the host is not keeping the device fed. The comparison flags an increase as a regression.

### torch.compile / Triton Tables

Emitted only when the DB carries compiled-region annotations or `triton_*` kernels (mainly converted torch traces). Skip them when absent.

- Fusion Coverage: `fusion coverage %` = triton-fused compute time / total compute time, plus `triton_fused_ms` / `non_triton_ms` and a top non-fused kernel list (fusion-miss / fallback candidates). Lower coverage is worse.
- Inductor Fusion Granularity: family-level fused/unfused time for `pointwise`, `reduce`, `library_or_gemm`, `communication`, `triton_other`, and `other`. Treat increased unfused `pointwise` time as the strongest missed-fusion signal; increased unfused `reduce` time is a secondary fusion/reduction signal. Do not mark vendor GEMM/conv/library rows as fusion defects without graph-break/fallback evidence.
- Compile Segmentation + Host Launch Overhead: compiled region count, inside vs outside-region (eager) compute, recompilation indicators, and the host-launch-overhead metrics — `main_stream_gap_pct`, `avg_launch_self_us`, `launch_self_to_compute_ratio` — plus `cpp_wrapper_signal` (`state`, `source`, `confidence`, and observed `kernel_file` extensions when available).
- Triton Kernel IO Efficiency: per kernel name, `io_efficiency` (folded bandwidth, NOT a 0–1 ratio) and `bandwidth_utilization = io_efficiency / theoretical_peak`. Theoretical peak comes from the MLU model (MLU590 → 2000, MLU580 → 1200). Lower utilization is worse.

### Host Summaries

- Host Function Summary: source `function_data`; include count, selected-range total duration, avg, max, and top names by total. Totals can double count overlapping threads and nested calls.
- Host Internal Operation Summary: source `Internal_operation_range_data`; include count, selected-range total duration, avg, max, and top names by total when the table is present.
- Sync functions are not emitted as a separate table; when relevant, inspect host summaries for names containing `sync`, `synchronize`, or `wait`.

## Metric Rules

- `P90 ms`: 90th percentile of selected per-event duration for the row.
- `Share`: row total divided by the corresponding category total.
- `Uncovered share`: uncovered ms divided by row total ms.
- Compare names as strings resolved through each DB's own `string_table`; never compare `nameId` values across DBs.

## Analysis Guidance

- Treat the chosen baseline/current meaning as part of the analysis contract before comparing tables.
- Prioritize current regressions: larger total time, higher count, worse avg/p90/max, higher uncovered time, lower bandwidth, or higher gap/idle share.
- Write the final deliverable using the exact concise `Final Report Contract` structure.
- Include an `## 产物` section with the temporary analysis directory, report path, converted DB paths, table collection outputs, and delta comparison outputs used as evidence.
- In `## 结论概览`, include only the strongest supporting summary data next to each claim. For each highlighted regression, show the relevant baseline value, current value, delta, and table source when both values are available.
- Prefer compact evidence snippets over separate comparison tables, for example: `stable compute kernel total: baseline 820 ms, current 970 ms, +150 ms (+18.3%), from Device Breakdown Overview`.
- Always include the key Compute Kernel Summary comparison signal in the conclusion, but do not paste a detailed per-kernel table into the main report; cite the artifact instead.
- Use the summary rows exactly as emitted by `collect_profile_tables.py`; they already cover the leading 95% share plus `other`, so do not apply another top-k or 95% compression pass when reading them.
- For compute differences, inspect total/count/avg/p90/max and optionally group kernels by workload semantics such as matmul/gemm, attention, normalization/reduce, elementwise/fusion, Triton, embedding/indexing, or data movement.
- For communication differences, interpret total time together with uncovered time. High total with low uncovered may be hidden by compute overlap.
- For memcpy differences, separate direction/type changes, uncovered time, bytes, and bandwidth.
- For compute gap or pure gap differences, consider scheduling, launch, synchronization, pipeline bubbles, device idle, host submission, or missing activity coverage.
- For host-overhead differences, judge primarily by the device-stream (main compute stream) gap ratio, not host-side wall time. An increased main-stream gap ratio in current means the host is feeding the device less well. Confirm with `avg_launch_self_us`, `launch_self_to_compute_ratio`, and `cpp_wrapper_signal`.
- For `cpp_wrapper`, read the trace signal first. `state=off` with source `explicit_trace_metadata` or `kernel_file_extension` is trace evidence for Python wrapper / disabled cpp_wrapper; `state=on` is trace evidence that cpp_wrapper is enabled; only `state=unknown` should be treated as inferred from host-gap metrics.
- When current is host-bound on a torch.compile workload and `cpp_wrapper_signal.state=off`, recommend enabling `cpp_wrapper` first to cut per-launch host overhead. When state is `unknown`, recommend verifying/enabling it as a hypothesis. When state is `on`, do not cite disabled `cpp_wrapper` as the root cause; investigate graph breaks, synchronization, tiny kernels, or host framework work. Do not recommend graph capture as the only remedy; it is complementary unless the trace directly supports it.
- For fusion-coverage differences, a drop in `fusion coverage %` or a rise in non-triton/eager compute time in current points to lost inductor fusion (graph breaks, fallback ops). Then inspect `Inductor Fusion Granularity Delta`: increased unfused `pointwise` time should be highlighted prominently, increased unfused `reduce` time should be highlighted next, and specific rows from `Highlighted Unfused Pointwise/Reduce Delta` should be cited as evidence. Do not treat vendor GEMM/conv library kernels as fusion defects; flag elementwise/reduction/pointwise fallbacks.
- For triton kernel IO-efficiency differences, treat `io_efficiency` as a folded bandwidth value (not a percentage); compare `bandwidth_utilization` against the same MLU-model theoretical peak on both sides. A utilization drop in current is a memory-IO regression. Never compute `1 - io_efficiency` on the raw value.
- For other activity differences, inspect notifier, atomic operation, memset, or related device task tables.
- When both captures are single-card, treat communication-kernel deltas with caution: single-card communication exposure is dominated by waiting for absent peers, so a comm delta reflects timing/exposure noise more than real communication cost. Do not draw cross-rank communication conclusions from single-card captures unless multi-rank captures are provided.
- End with suggested follow-up branches, not a forced root cause.
