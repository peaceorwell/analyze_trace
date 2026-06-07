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

If the prompt is an environment diagnostic or smoke test and asks to reply only `OK`, reply exactly `OK` and do not run tools or load references.

## Web Output Contract

When running for the Web app, produce a stable, user-facing result:

- Write the final user-visible report to `$TRACE_AI_REPORT_PATH` when that environment variable is set.
- Also write the same final report to `report.md` in the current working directory.
- Print the same final report to stdout. Do not print tool logs, raw command output, prompt text, or progress narration to stdout.
- Save supporting evidence as separate files in the analysis directory, such as `baseline.tables.json`, `current.tables.json`, `comparison_evidence.md`, and conversion logs.
- If analysis cannot proceed because a trace file, DB table, Python dependency, or tool permission is missing, write a concise failure report instead of a partial or fabricated comparison.
- Prefer Chinese report text when the request is Chinese.

The final `report.md` should use this exact high-level structure:

1. `# AI 对比分析报告`
2. `## 结论概览`
   - 3-6 bullets focused on regressions first, then meaningful improvements.
3. `## 对比口径`
   - Baseline/current files, selected windows, devices, and `Delta = B - A`.
4. `## 关键 Delta`
   - Compact Markdown table with metric, A, B, delta, interpretation, and source.
5. `## 主要回退与原因假设`
   - Prioritized findings with evidence, counter-evidence, estimated impact, confidence, and affected ranks/devices.
6. `## 优化建议`
   - Prioritized actions with expected benefit, implementation cost, risk, and validation method.
7. `## 不确定性与下一步`
   - Missing evidence and the next check that would reduce uncertainty.
8. `## 产物`
   - Converted DBs, collected table JSON files, evidence logs, and analysis directory.

Keep the final report concise enough for Web reading. Move large raw tables and long logs into artifact files, then cite those files from the report.

## Resources

- `scripts/collect_profile_tables.py`: collect host summaries and device breakdown tables for one cnperf DB over one explicit time range.
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
   - Resolve the skill directory from the loaded skill path and keep it in `SKILL_DIR`.
   - Convert JSON/JSON.GZ traces to cnperf-compatible DB in the temporary analysis directory when needed.
   - Determine raw, preparation, stable, or manually selected analysis windows.
   - Inspect basic information such as device model, card count, driver/cnperf version, host environment, and card usage.
4. Collect breakdown tables independently for each DB and selected time range.
   - Run `collect_profile_tables.py` once per DB/range.
   - Write each collection output to the temporary analysis directory. Prefer JSON for machine-readable comparison and keep text logs when useful for review.
   - Use the same range semantics across files when comparing.
   - The script includes rows whose `start` is in `[start_ms, end_ms)` and clips duration at `end_ms`.
5. Compare from coarse to fine.
   - Start from upstream E2E windows: raw, preparation, stable.
   - Focus on categories where current is worse than baseline; only briefly record current advantages.
   - Use Device Breakdown Overview to locate current regressions across compute, communication, memcpy, compute gap, pure gap, and other activity.
   - Always compare Compute Kernel Summary at name level, because device kernel cost is usually the primary investigation target.
   - Enter other name-level tables for additional regressed categories.
   - Use Host Function Summary and Host Internal Operation Summary as lightweight follow-up signals; inspect sync-like function names there when needed.
6. Generate a Markdown analysis document.
   - Write the report to `<analysis_dir>/report.md` unless the user requests another path.
   - Put basic information and the final conclusion at the top.
   - Put detailed summaries, analysis evidence, and follow-up suggestions below.

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

## Output Tables

### Device Breakdown Overview

One table for the selected range.

| Category | Total ms | Count | Avg ms | Max ms | Range share |
|---|---:|---:|---:|---:|---:|

Required categories:

- `compute kernel`: `device_task_kernel_data` where `isComputation = 1`.
- `communication kernel`: `device_task_kernel_data` where `isComputation = 0`.
- `memcpy`: `device_task_memcpy_data`.
- `compute gap`: positive intervals between adjacent compute kernels on the same process/device timeline.
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
- Write the final deliverable as a Markdown report with this order:
  1. `Basic Information`: comparison goal, baseline/current files, selected ranges, device/environment notes, and upstream E2E window summary when available.
  2. `Executive Conclusion`: prioritized findings about where current is worse than baseline. Include supporting baseline/current/delta data beside each finding.
  3. `Key Evidence Summary`: compact evidence grouped by Device Breakdown Overview, Compute Kernel Summary, Communication/Memcpy Summary, and Host Summary.
  4. `Detailed Analysis`: compute kernel comparison first, then communication, memcpy, gap/other activity, and host signals as needed.
  5. `Suggestions / Next Checks`: concrete follow-up branches or extra checks.
  6. `Appendix`: raw baseline/current tables or long copied table excerpts when useful.
- Include an `Artifacts` note with the temporary analysis directory, report path, converted DB paths, and table collection outputs used as evidence.
- In `Executive Conclusion`, include the supporting summary data next to each claim. For each highlighted regression, show the relevant baseline value, current value, delta, and table source when both values are available.
- Prefer compact evidence snippets over separate comparison tables, for example: `stable compute kernel total: baseline 820 ms, current 970 ms, +150 ms (+18.3%), from Device Breakdown Overview`.
- Always include a detailed Compute Kernel Summary comparison in the conclusion: compare high-share kernel names by total/count/avg/p90/max/share, and call out whether the regression is from slower kernels, more launches, or long-tail changes.
- Use the summary rows exactly as emitted by `collect_profile_tables.py`; they already cover the leading 95% share plus `other`, so do not apply another top-k or 95% compression pass when reading them.
- For compute differences, inspect total/count/avg/p90/max and optionally group kernels by workload semantics such as matmul/gemm, attention, normalization/reduce, elementwise/fusion, Triton, embedding/indexing, or data movement.
- For communication differences, interpret total time together with uncovered time. High total with low uncovered may be hidden by compute overlap.
- For memcpy differences, separate direction/type changes, uncovered time, bytes, and bandwidth.
- For compute gap or pure gap differences, consider scheduling, launch, synchronization, pipeline bubbles, device idle, host submission, or missing activity coverage.
- For other activity differences, inspect notifier, atomic operation, memset, or related device task tables.
- End with suggested follow-up branches, not a forced root cause.
