# PyTorch Performance Analysis Playbook

Use this playbook to decide whether a profile is trustworthy, turn trace symptoms into causal hypotheses, and define validation experiments. Apply it before selecting bottleneck branches or comparing runs.

## Contents

- Measurement validity gate
- Evidence hierarchy
- Top-down diagnosis
- Kernel and operator interpretation
- torch.compile and Inductor
- Input pipeline and memory
- Recommendation and validation contract
- Primary sources

## Measurement Validity Gate

Record a pass, partial, or fail for each item. A failed item does not always block analysis, but it lowers confidence and must appear under uncertainties.

1. **Semantic scope**
   - Identify whether the trace covers training or inference, forward only or forward/backward/optimizer, and which steps or requests are included.
   - Do not compare windows that contain different semantic work.
2. **Warm state**
   - Separate first-run initialization, compilation, autotuning, allocator warm-up, and data-cache warm-up from steady-state execution.
   - A trace containing both can still diagnose cold start, but must not be used as steady-state evidence without a stable sub-window.
3. **Workload identity**
   - Record batch size, sequence length or token count, input shapes, dtype/precision, grad mode, model mode, and distributed topology when available.
   - Normalize time by an equivalent unit such as step, sample, token, or request when raw work differs.
4. **Profiler perturbation**
   - Record whether stack capture, shape recording, memory profiling, Python tracing, or unusually dense annotations were enabled.
   - Treat host-side microsecond differences cautiously when instrumentation differs or overhead is unknown.
5. **Accelerator asynchrony**
   - Use trace event timestamps or synchronized benchmark measurements. Host enqueue duration is not kernel execution time.
   - Do not infer device speed from unsynchronized Python wall time.
6. **Repeatability**
   - Prefer several stable steps or repeated benchmark measurements. Use median/typical behavior plus a tail metric such as p90/p95.
   - A single maximum-duration event is an outlier signal, not a regression by itself.
7. **Correctness**
   - A faster run is not an optimization if outputs, loss, gradients, executed work, or numerical behavior changed unintentionally.
   - Trace data rarely proves correctness; require an external correctness check in the validation plan.

When metadata is absent, say `unknown`; do not invent configuration values.

## Evidence Hierarchy

Prefer evidence in this order:

1. Aligned end-to-end or step duration on equivalent work.
2. Local critical-path device/host intervals and overlap.
3. Category totals: compute, opaque communication, copies, synchronization, and idle gaps.
4. Operator or kernel aggregate total, count, typical duration, and tail duration.
5. Correlation links, same-thread host context, queue dependencies, compiled-region boundaries, and aligned profile windows.
6. Names, static source patterns, configuration guesses, or generic tuning advice.

Do not promote a lower-level clue over contradictory higher-level evidence. High utilization can coexist with poor throughput when the device performs extra work; low utilization is a symptom until the gap owner is identified.

## Top-Down Diagnosis

Use the following order:

1. Establish the target metric: latency, throughput, time per token/sample, compile latency, memory limit, or scaling efficiency.
2. Establish stable scope and the critical path.
3. Partition time into effective compute, opaque communication, ordinary device work, and gaps.
4. Attribute the dominant exposed interval through queues and host/device correlations.
5. Drill into the largest actionable operator or kernel family.
6. Form at most a few causal hypotheses and define an experiment that can falsify each one.

Keep total work and execution efficiency separate:

- Higher kernel count or FLOPs usually means more work.
- Similar count with higher per-call duration points toward slower execution or different shapes.
- Lower total time accompanied by missing kernels/operators may indicate omitted work.
- Improvements that overlap on the same critical-path interval are not additive.

## Kernel And Operator Interpretation

- Rank primarily by aggregate contribution to the target window, then inspect count, average/median when available, p90/p95, and maximum.
- Group by observed names only after preserving exact names. Name families are heuristics.
- For small kernels, launch count and device gaps may matter more than per-kernel arithmetic.
- For large kernels, distinguish compute-shaped from memory-shaped behavior with achieved throughput, effective bandwidth, arithmetic intensity, and hardware limits when comparable.
- Fusion is valuable when it removes materialization and launch overhead, but it can increase live ranges, on-chip memory, register pressure, or recomputation. Benchmark the fused result.
- Vendor GEMM/conv/library kernels are not fusion failures merely because their names are not Triton.
- When shapes are available, split materially different shapes before averaging.

## torch.compile And Inductor

Analyze compilation and steady-state separately.

- Use compiled-region events and launch correlations to locate graph breaks; do not equate every region boundary with an actionable break.
- Treat guard failures and recompilations as shape/control-flow evidence. Confirm with `TORCH_LOGS=graph_breaks,recompiles,guards,dynamic`, `tlparse`, or equivalent logs when trace metadata is insufficient.
- Check whether varying shapes cause repeated compilation or eventual eager fallback before recommending `dynamic=True`; prefer targeted dynamic-dimension evidence.
- A loss of fusion should be supported by increased eager/outside-region work, unfused pointwise/reduction kernels, or a changed kernel chain.
- Large kernel gaps with many tiny launches support a host-launch hypothesis. Verify wrapper mode and synchronization evidence before recommending wrapper or graph-capture changes.
- Do not mix compilation time, autotuning time, or cache population with steady-state kernel performance.

## Input Pipeline And Memory

Only diagnose a data pipeline problem when device gaps align with host-side input work such as `__next__`, collation, preprocessing, pinning, or copies.

- Tune `num_workers`, prefetching, persistent workers, and pinning as experiments, not generic defaults.
- `non_blocking=True` alone does not prove overlap. The source, stream/queue, dependency, and accelerator support must permit asynchronous transfer.
- Separate transfer volume from transfer exposure; a large copy hidden by compute may not affect end-to-end time.
- Distinguish tensor-allocated memory, allocator-reserved memory, and non-framework allocations.
- Peak allocation, fragmentation, repeated allocation/free, and OOM diagnosis require allocator history or memory snapshots when the trace lacks memory events.
- Activation checkpointing trades compute for memory; evaluate both memory headroom and step-time impact.

CUDA-specific APIs in the primary sources illustrate mechanisms. On MLU, recommend only the corresponding supported runtime/profiler experiment.

## Recommendation And Validation Contract

Every priority recommendation must contain:

1. **Evidence**: measured interval, delta, kernel/operator, process/device, or dependency chain.
2. **Mechanism**: why the proposed change should reduce the target metric.
3. **Scope**: affected shapes, steps, process/devices, or kernels.
4. **Expected bound**: use measured exposed/critical-path time; state when benefits overlap.
5. **Correctness guardrail**: outputs/loss/gradients, tolerances, NaN/Inf behavior, or work-count parity.
6. **Experiment**: one controlled change, warm-up, repeated stable measurements, and identical profiler settings.
7. **Success metric**: end-to-end target first, supporting trace metric second.
8. **Rollback condition**: correctness failure, tail regression, memory regression, or benefit below noise.

Label conclusions as:

- `confirmed`: direct timing and dependency evidence support the mechanism.
- `supported hypothesis`: multiple signals agree but one discriminating input is missing.
- `speculative`: plausible from names or static code only; keep out of priority findings.

## Primary Sources

- PyTorch profiler API and scheduled wait/warmup/active capture: https://docs.pytorch.org/docs/stable/profiler.html
- PyTorch profiler stack-capture overhead warning: https://docs.pytorch.org/tutorials/beginner/profiler.html
- PyTorch benchmark utilities, warm-up, synchronization, and replicates: https://docs.pytorch.org/docs/stable/benchmark_utils.html
- Profiling torch.compile, compiled regions, graph breaks, compile warm-up, and launch gaps: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
- torch.compile programming model, guards, recompilations, and diagnostics: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.html
- Dynamic-shape behavior and targeted annotations: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html
- Data transfer, pinned memory, and overlap conditions: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
- PyTorch allocator snapshots and visibility limits: https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
- Triton fusion and DRAM-traffic reasoning: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- Triton autotune and configuration search: https://triton-lang.org/main/python-api/generated/triton.autotune.html
