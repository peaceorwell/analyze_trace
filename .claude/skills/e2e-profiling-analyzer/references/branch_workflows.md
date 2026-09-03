# Phase 2 Branch Workflows

Load this reference before executing any Phase 2 branch. The main skill owns mode selection, branch selection, artifacts, and final synthesis; this file owns branch-specific methods, guardrails, and output contracts.

Set `BRANCH_DIR` to the assigned exclusive directory under `TEAM_DIR` before running a command. Never write branch artifacts into another agent's directory.

In `compute-gap-root-cause`, pair every gap-ratio claim with host attribution from `host_op_breakdown.py`: name the operators holding host self time, the launch-API count and average per-launch cost, and whether the cost sits in host-only operators or in launches. A gap ratio without a named host owner stays `insufficient`.

Every branch runs inside the baseline steady window. Pass the lead-provided `--start-ns/--end-ns` to every script that accepts them, and state the window in the branch report `Scope`. A branch that must widen the window records it as a separate scoped analysis.

## Contents

- effective-compute-breakdown
- ordinary-non-compute-root-cause
- compute-gap-root-cause
- host-window-subphase
- compile-segmentation
- triton-fusion-coverage
- triton-kernel-efficiency

Apply the evidence hierarchy and measurement-validity rules from the main skill's required performance playbook to every branch. Keep static code findings and name-based classifications below direct timing and dependency evidence.

### Branch: `effective-compute-breakdown`

Question: when effective compute dominates, which compute kernels consume time, and is the issue more work or slower execution?

Workflow:

1. Run `compute_breakdown.py` for the selected DBs.

   ```bash
   python3 "$SKILL_DIR/scripts/compute_breakdown.py" <cnperf_db> [<cnperf_db> ...] \
     --format json > "$BRANCH_DIR/compute_breakdown.json"
   python3 "$SKILL_DIR/scripts/compute_breakdown.py" <cnperf_db> [<cnperf_db> ...] \
     --format text > "$BRANCH_DIR/compute_breakdown.md"
   ```

2. Aggregate `device_task_kernel_data` rows where `isComputation=1`.
3. Report observed top compute kernel names by count, total time, average, and max duration.
4. When comparable captures exist, separate "more work" from "slower execution" using launch count, matching-kernel duration, and optional FLOPs.
5. Report whether compute optimization is likely worthwhile or whether non-effective time remains the bigger target.

Output contract:

- `Scope`: DB files and process/device coverage.
- `Compute Summary`: total compute time, kernel count, unique compute kernel names, time share versus Phase 1 device time when available.
- `Top Compute Kernels`: `kernel_name`, `count`, `total_ms`, `share_of_compute`, `avg_ms`, `max_ms`.
- `Candidate Causes`: more work, slower execution, long-tail behavior, or balanced compute with evidence and confidence.
- `Interpretation` and `Next Step`.

Do not assume compute kernels are GEMM, FA, Conv, or elementwise.

### Branch: `ordinary-non-compute-root-cause`

Question: why is ordinary non-compute device work exposed?

Workflow:

1. Confirm uncovered memcpy, memset, atomic, or other ordinary non-compute work is material from Phase 1.
2. Aggregate relevant device tables by type, size if available, queue, count, total, average, and max duration.
3. Separate bulk H2D/D2H/D2D copies from host synchronization behavior when host API context is available.
4. Inspect host ranges around material copies or ordinary tasks: `pin_memory`, `copy_`, `to`, `_copy_from`, `__next__`.
5. Report whether ordinary non-compute work is dominant, hidden by compute, or a symptom of host sync/data pipeline behavior.

Output contract:

- `Scope`: DB files, process/device coverage.
- `Ordinary Non-Compute Breakdown`: `task_type`, `count`, `total_ms`, `share_of_device`, `avg_ms`, `max_ms` aggregated from device tables.
- `Top Ordinary Tasks`: largest individual memcpy/memset/atomic rows by duration with `correlationId` and queue.
- `Host Context`: host ranges and APIs temporally overlapping major ordinary tasks.
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
     > "$BRANCH_DIR/${db_stem}.gap_detail.<prev>.<next>.log" 2>&1
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
7. Report the dominant subtype: host-side blocker, notifier dependency, previous kernel, memcpy/atomic, fragmented codegen launch, opaque non-compute source task, out-of-range, or unknown.
8. If the host-side blocker is per-kernel launch overhead (not a queue-sync wait), confirm it with the device-stream gap ratio from `device_timeline.py` (high main compute stream gap %). When the workload also uses torch.compile, hand off to the `compile-segmentation` cpp_wrapper check and apply its recommendation rule: enable `cpp_wrapper` when the trace signal says it is off, verify it when unconfirmed, and look elsewhere when it is already on; do not recommend graph capture alone.

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

The next three branches target torch.compile/inductor workloads, mainly converted torch profiler traces. They read only DB inputs and write distinct output files, so they are parallel-safe with each other and with other branches. Their evidence comes from the baseline `kernel_codegen_analysis.json`, kernel names in `string_table`, compiled-region ranges in `Internal_operation_range_data`, and the per-kernel `args`/metadata preserved in `device_task_kernel_data.extra` (JSON). Always resolve names through this DB's `string_table`, and report observed names/metadata keys first instead of assuming fixed inductor naming.

Shared preconditions:

- If the DB has no compiled-region annotations and no `triton_*` kernel names, the workload likely does not use torch.compile. Skip all three branches and record them under `Skipped Branches`.
- Triton-fused kernels are identified by observed names such as `triton_poi_fused_*`, `triton_red_fused_*`, `triton_per_fused_*`, and `triton_tem_fused_*`. List the actual matched names before grouping; do not hardcode the set.

### Branch: `compile-segmentation`

Question: how does torch.compile partition the model into compiled regions, and do graph breaks or recompilations fragment otherwise-fusable work?

Workflow:

1. Run `compile_segmentation.py` for the selected DBs.

   ```bash
   python3 "$SKILL_DIR/scripts/compile_segmentation.py" <cnperf_db> [<cnperf_db> ...] \
     --format json > "$BRANCH_DIR/compile_segmentation.json"
   python3 "$SKILL_DIR/scripts/compile_segmentation.py" <cnperf_db> [<cnperf_db> ...] \
     --format text > "$BRANCH_DIR/compile_segmentation.md"
   ```

   The script reports the observed compiled-region inventory (names decoded through `string_table`), inside vs outside-region (eager) compute split, recompilation indicators, custom-op ranges that contain many simple `aten::` ops, the per-queue device-stream gap ratio, and the host-launch-overhead metrics. Report the observed region-name inventory first.
2. Read segmentation: device compute time and kernel count inside compiled regions vs outside (eager/graph-break), and whether work is fragmented across many small regions. The script attributes each kernel by temporal containment of its `function_data` launch within compiled-region ranges.
3. Cross-check baseline `kernel_codegen_analysis.json` for operator-to-kernel mapping coverage, tiny-kernel share, launch density, and repeated adjacent kernel pairs. Treat low mapping coverage as a limitation.
4. Read recompilation indicators (`TorchDynamo Cache Lookup`/guard ranges) as a sign of re-tracing on dynamic shapes.
5. cpp_wrapper check (trace signal first, device-stream gap second): read `host_launch_overhead.cpp_wrapper_signal` / `cpp_wrapper_signal` before making any inference.
   - `state=off` means the trace indicates Python wrapper / `cpp_wrapper` disabled. This can come from an explicit trace key or from Inductor `kernel_file` evidence such as generated `.py` files.
   - `state=on` means the trace indicates `cpp_wrapper` enabled. This can come from an explicit trace key or generated C++/shared-library style `kernel_file` evidence.
   - `state=unknown` means the trace did not carry a direct signal; only then infer wrapper mode from high main-stream gap ratio, small kernels, high `avg_launch_self_us`, and high `launch_self_to_compute_ratio`.
   - Always report the signal source and confidence. Do not write "无法从 trace 确认 cpp_wrapper" when `cpp_wrapper_signal.source` is `explicit_trace_metadata` or `kernel_file_extension`.
6. Identify the largest outside-region (eager) kernels from `top_outside_region_kernels`.
7. Read `custom_op_simple_aten`. If `has_issue=true`, promote it as an optimization candidate: a custom/user op is present but still executes many simple `aten::` pointwise/view/reduce/copy/allocation ops inside the wrapper, so those ops should be moved into the custom backend kernel or restructured to let Inductor fuse them. If `must_report=true` or the top row has `report_priority=high`, this is a final-report finding, not just branch detail. Cite the concrete `custom_op_name`, call count, nested simple aten count, average nested ops per call, and top nested `aten::` names.
8. Report whether segmentation is material: large compute time or many kernels outside compiled regions, frequent recompilation, custom-op simple-aten nesting, or many small fragmented regions.

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
     --format json > "$BRANCH_DIR/triton_fusion_coverage.json"
   python3 "$SKILL_DIR/scripts/triton_fusion_coverage.py" <cnperf_db> [<cnperf_db> ...] \
     --format text > "$BRANCH_DIR/triton_fusion_coverage.md"
   ```

   It classifies compute kernels (`isComputation=1`) by name from `string_table` into triton-fused (`triton_*fused*`), other triton (rare), and non-triton/library/eager. It also groups kernels into Inductor fusion families (`pointwise`, `reduce`, `library_or_gemm`, `communication`, `triton_other`, `other`) and reports fused/unfused time for each family, highlighted unfused pointwise/reduce candidates, top non-fused kernels, and per-process/device fusion coverage.
2. Read the fusion-coverage ratio (triton-fused compute time / total compute time) and the top non-fused kernels as fusion-miss / fallback candidates.
3. Inspect `Inductor Fusion Granularity` first. If `pointwise` has non-zero unfused time, highlight it as the strongest missed-fusion signal; if `reduce` has non-zero unfused time, highlight it as a secondary fusion/reduction candidate. Treat library/GEMM/conv families as likely intended fast paths unless other evidence says otherwise.
4. Cross-reference with `compile-segmentation` when available: are highlighted non-fused pointwise/reduce kernels concentrated in eager/graph-break segments?
5. Cross-check baseline `kernel_codegen_analysis.json` for tiny/repeated launches and operator mapping that support or weaken the fusion hypothesis.
6. Report whether raising fusion coverage is a worthwhile target versus other exposed time.

Guardrail: do not assume every non-triton kernel is a fusion defect. Vendor GEMM/conv/library compute primitives are often the intended fast path. Flag fusion misses primarily for elementwise/pointwise/reduction kernels left unfused, not for library compute primitives.

Output contract:

- `Scope`: DB files, process/device coverage.
- `Fusion Coverage Summary`: fused vs non-fused compute time and ratio, kernel counts per class.
- `Inductor Fusion Granularity`: family-level fused/unfused time; explicitly call out unfused `pointwise` and `reduce` time. A non-zero unfused pointwise row must be highlighted.
- `Highlighted Unfused Pointwise/Reduce Candidates`: top kernels whose names look pointwise/reduce-like but did not appear as triton-fused, with impact and the script's reason.
- `Top Non-Fused Kernels`: `kernel_name`, `count`, `total_ms`, `share_of_compute`, `avg_ms`, `max_ms`.
- `Segment Correlation`: whether non-fused kernels cluster in eager/graph-break segments (link to `compile-segmentation` if run).
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
     --dump-dir "$BRANCH_DIR/triton_output_code" --format json \
     > "$BRANCH_DIR/triton_kernel_efficiency.json"
   python3 "$SKILL_DIR/scripts/triton_kernel_efficiency.py" <cnperf_db> [<cnperf_db> ...] \
     --dump-dir "$BRANCH_DIR/triton_output_code" --format text \
     > "$BRANCH_DIR/triton_kernel_efficiency.md"
   ```

   If the script reports `has_io_metadata=false`, skip this branch and record it under `Skipped Branches` with the missing-metadata reason. The script reports the observed metadata keys first, treats `io_efficiency` as folded bandwidth, and uses the MLU-model **theoretical (peak) bandwidth** — MLU590 → 2000, MLU580 → 1200 (GB/s) — falling back to `meta_information` `deviceInfo.m_dev_basic_info.max_bandwidth` only when the model is unknown. It computes `bandwidth_utilization = io_efficiency / peak_bandwidth` when comparable, and ranks by `improvement_target = total_ms * (1 - bandwidth_utilization)` (falling back to lowest folded bandwidth weighted by `total_ms` when utilization is unavailable). Check `peak_bandwidth_source`; if utilization looks impossible (e.g. > 1), treat units as mismatched and rely on the fallback ranking.
2. If the sibling `mlu-triton-optimize` skill is available, run its static code analyzer on the dumped `output_code`.

   ```bash
   TRITON_OPT_SKILL_DIR="$(dirname "$SKILL_DIR")/mlu-triton-optimize"
   if [ -f "$TRITON_OPT_SKILL_DIR/scripts/analyze_triton_code.py" ]; then
     python3 "$TRITON_OPT_SKILL_DIR/scripts/analyze_triton_code.py" \
       --input-dir "$BRANCH_DIR/triton_output_code" \
       --efficiency-json "$BRANCH_DIR/triton_kernel_efficiency.json" \
       --format json \
       > "$BRANCH_DIR/triton_code_optimization.json"
     python3 "$TRITON_OPT_SKILL_DIR/scripts/analyze_triton_code.py" \
       --input-dir "$BRANCH_DIR/triton_output_code" \
       --efficiency-json "$BRANCH_DIR/triton_kernel_efficiency.json" \
       --format text \
       > "$BRANCH_DIR/triton_code_optimization.md"
   fi
   ```

   Read `triton_code_optimization.json` when present. Use it as code-level evidence for MLU Triton optimization candidates, not as proof that a rewrite will improve performance.
3. For the top low-bandwidth kernels, combine IO-efficiency metrics, baseline launch-configuration/long-tail data, and `triton_code_optimization` findings. Characterize access pattern, masking, non-contiguous or gather/scatter access, reduction shape, grid/block configuration, load/store counts, static IO/compute throughput estimates, libdevice/division candidates, and dtype conversion chains. Do not paste full generated source into the main report; cite the output-code file and the analyzer artifact.
4. Classify the low-bandwidth cause per kernel: memory-bound small kernel, non-coalesced/strided access, redundant recompute, poor tiling/grid, register spill, expensive math/division, fragmented pseudo-discrete IO, reduce layout/tiling, or already efficient (folded bandwidth near peak).

Output contract:

- `Scope`: DB files, whether `output_code` and IO-efficiency metadata are present, and the observed metadata key names.
- `IO Efficiency Summary`: number of triton kernels with metadata, distribution of folded/effective bandwidth (`io_efficiency`), the device peak bandwidth and its units, and bandwidth utilization when computable. State explicitly that `io_efficiency` is a bandwidth value, not a percentage.
- `Top Low-Bandwidth Kernels`: `kernel_name`, `count`, `total_ms`, `io_efficiency` (folded bandwidth with units), `bandwidth_utilization` (`io_efficiency / peak_bandwidth`, when available), and `improvement_target`.
- `Output Code Findings`: per top kernel, the access-pattern characterization with the `output_code` excerpt filename.
- `Triton Code Optimization Candidates`: when `triton_code_optimization.json` exists, summarize high-priority candidates from the sibling `mlu-triton-optimize` analyzer, including static estimated throughput plus the merged optimization direction and recommendation. Cite `triton_code_optimization.md/json`.
- `Candidate Causes`: with evidence, counter-evidence, impact, confidence, and missing evidence.
- `Interpretation` and `Next Step`.
