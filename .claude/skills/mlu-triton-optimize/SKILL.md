---
name: mlu-triton-optimize
description: Analyze generated Triton output_code for Cambricon MLU optimization opportunities. Use when a trace report needs Triton-kernel code-level optimization candidates, especially for torch.compile/Inductor generated kernels.
disable-model-invocation: false
user-invocable: true
---

# MLU Triton Code Optimization Analyzer

This skill is analysis-only. It does not rewrite Triton code, run benchmark jobs, or require interactive tuning.

It reviews generated Triton `output_code` and produces concise optimization candidates that can be merged into the `e2e-profiling-analyzer` report. It is intended for Web-side AI analysis where stability and repeatability matter more than one-off kernel tuning.

## Inputs

- One or more generated Triton `output_code` files dumped from trace metadata.
- Optional `triton_kernel_efficiency.json` produced by `e2e-profiling-analyzer/scripts/triton_kernel_efficiency.py`.

## Standard Invocation

```bash
python3 "$TRITON_OPT_SKILL_DIR/scripts/analyze_triton_code.py" \
  --input-dir "$ANALYSIS_DIR/triton_output_code" \
  --efficiency-json "$ANALYSIS_DIR/triton_kernel_efficiency.json" \
  --format json \
  > "$ANALYSIS_DIR/triton_code_optimization.json"

python3 "$TRITON_OPT_SKILL_DIR/scripts/analyze_triton_code.py" \
  --input-dir "$ANALYSIS_DIR/triton_output_code" \
  --efficiency-json "$ANALYSIS_DIR/triton_kernel_efficiency.json" \
  --format text \
  > "$ANALYSIS_DIR/triton_code_optimization.md"
```

If no `output_code` files exist, the skill should report `has_findings=false` and explain that Triton source metadata is missing.

## Analysis Scope

Use a source-informed, validation-first flow. The goal is not to prove that a rewrite is faster from source alone, but to identify the most plausible validation targets:

1. **Prioritize by runtime context**: use `total_ms`, bandwidth utilization, and repeated patterns from `triton_kernel_efficiency.json`; ignore tiny kernels unless they repeat enough to matter.
2. **Classify with static Roofline signals**: estimate IO bytes, approximate scalar/vector operation count, arithmetic intensity, IO throughput, and compute throughput from generated code. Treat the result as a direction signal: memory-shaped, compute-shaped, or balanced.
3. **Inspect memory access shape**: look for contiguous bulk IO, pseudo-gather, true gather, block pointer/tensor descriptor usage, mask/other paths, cache hints, and repeated load/store patterns.
4. **Inspect compute shape**: look for libdevice-eligible math, division lowering opportunities, dtype conversion chains, reductions, and expensive scalarized index arithmetic.
5. **Inspect mapping/tuning shape**: look for multi-dimensional `program_id`, `tl.num_programs`, `num_warps`, `num_stages`, missing autotune/config signals, and tile/block sizes that should be swept. Treat tiling as a config space, not a single `BLOCK_*` knob.
6. **Report as validation targets**: recommendations should say what to benchmark or trace next, not claim guaranteed speedup.

Focus on optimization opportunities visible from generated Triton code:

- **libdevice math replacement**: `tl.sigmoid`, `tl.exp`, `tl.log`, `tl.sqrt`, `tl.erf`, `tl.tanh`, `tl.pow`, `x * tl.sigmoid(x)`, and similar compute-heavy patterns that may map better to `tl.extra.mlu.libdevice.fast_*`.
- **Division lowering**: tensor division inside wide tiles, especially division after broadcast. Prefer reducing division count, multiplying by reciprocal, or using libdevice divide/rcp helpers when precision allows.
- **Fragmented IO / pseudo-gather**: many small `tl.load` / `tl.store` operations, even/odd or first-half/second-half access, fixed stride/interleave patterns, and modulo/floor-div indexing that hides contiguous memory. Recommend bulk load/store plus on-chip slice/cat/broadcast reshaping when the mapping is compile-time regular.
- **True gather / repeated lookup**: index/table-like loads that are not compile-time regular. Recommend validating reuse and considering `cache_modifier=".cg"` only when it is a small reused operand.
- **Reduce layout and tiling**: `tl.sum`, `tl.max`, `tl.min`, `tl.reduce`, especially reductions over axis 1 or repeated reductions that may benefit from retiling or transpose-to-pooling-friendly layout.
- **Grid/retiling issues**: multi-dimensional program IDs, complex `tl.num_programs`, or block parameters that suggest poor core mapping. Recommend checking one-dimensional grid flattening and block-size consistency.
- **Block pointer / tensor descriptor shape**: missing `tl.make_block_ptr` on obviously bulk-like IO, or existing block pointers with suspicious `block_shape`, `order`, or stride usage. Recommend descriptor/bulk-IO validation only when supported by the current MLU Triton stack.
- **Autotune/meta-parameter sweep**: when a material kernel has reductions, low bandwidth utilization, multi-axis grid, `tl.dot`, or complex tiling but no visible `@triton.autotune` / `triton.Config`, recommend sweeping `BLOCK_*`, `num_warps`, `num_stages`, grid flattening, loop/range config, indexing strategy, and PID/L2 grouping.
- **Cache hint validation**: for true table/index/gather operands with reuse, consider `cache_modifier` / `eviction_policy` experiments; do not use cache hints as a substitute for regularizing pseudo-gather.
- **dtype conversion chains**: repeated `.to(tl.float32)`, `.to(tl.float16)`, `.to(tl.bfloat16)`, `.to(tl.int*)` conversions around math or stores. Recommend removing redundant conversions or using fast conversion helpers where available.
- **Static IO/compute estimate**: infer domain or tile size from `size_hints`, `block_shape`, or `tl.arange`; count `tl.load` / `tl.store` bytes and scalar `tl.*` arithmetic/comparison/math/reduce operations. Report estimated IO throughput, compute throughput, and arithmetic intensity as heuristic signals, not hardware counters.

## Cambricon Triton 101 Heuristics

Fold these Cambricon Triton 101 rules into the candidate analysis when source evidence supports them:

- **Prefer vectorized block operations**: MLU SIMD execution benefits strongly from `tl.arange`-based vectorized load/compute/store. Flag loops over `range` / `tl.range` / `tl.static_range` that repeatedly load or store one scalar-like element, especially when the loop index appears directly in `tl.load` / `tl.store`.
- **Regularize memory before tuning cache hints**: continuous IO is preferred. Treat even/odd, first/second half, modulo/floor-div offset, fixed-stride, and reshape-like addressing as pseudo-discrete unless proven otherwise. If the logical mapping is regular, recommend contiguous bulk IO followed by on-chip `slice` / `cat` / `broadcast` reshaping. Full discrete gather/scatter is expensive; lowest-dimension contiguous gather-vector may be acceptable when the contiguous dimension is at least about `512B`.
- **Use MLU task mapping intentionally**: on current MLU Triton stacks, `num_warps=1` maps to Block task and `num_warps=4` maps to Union1 task. Other values are unsupported or may silently fall back; report them as validation risks. For SIMD-heavy kernels, start with `1`; for kernels that can use Move/Compute/IO stream overlap or larger per-program work, benchmark `4`.
- **Tune block size for MLU, not GPU defaults**: MLU often benefits from larger non-power-of-two `BLOCK_*` values, bounded by NRAM. Very small blocks can inflate grid count and launch/scheduling overhead. If grid dimensions may exceed `65535`, recommend larger blocks or a persistent-kernel pattern that caps grid by core count and iterates inside the kernel.
- **Use soft pipeline where loops carry IO and compute**: `num_stages=1` means no useful pipelining for the target loop. For persistent or looped kernels with load/compute/store streams, validate `num_stages` around 2-4; higher values can increase resource pressure and must be benchmarked.
- **Group repeated scalar/broadcast reads**: when the kernel repeatedly reads scalar-like on-chip values and broadcasts them, reduce read count or group consecutive scalar reads. `tl.static_range` alone does not guarantee the generated load sequence is latency-friendly.
- **Validate with compiler/profiler evidence**: use MLUIR/Linalg to confirm whether accesses became continuous or `gather.vector`, use `TRITON_PRINT_PIPELINE=true` to inspect software-pipeline decisions, and use cnperf/kernel benchmark to validate any source-level rewrite.

## Helion-Inspired Tiling Config Heuristics

Community Helion treats Triton performance tuning as a bounded configuration search rather than hand-picking one block size. Reuse that mental model when reviewing generated `output_code`:

- **Promote config families, not isolated knobs**: analyze `BLOCK_*` / tile shape, loop order or loop flattening, `tl.range` unroll/stage choices, indexing strategy, `num_warps`, `num_stages`, and PID mapping as one search space. If a material kernel has no `@triton.autotune` / `triton.Config`, recommend a small sweep matrix instead of one magic value.
- **Check tile-shape balance**: very skewed tiles, very small dimensions, or unused-looking `BLOCK_*` values can produce too many programs, poor vector occupancy, or excessive NRAM pressure. Recommend bounded sweeps that include a few MLU-friendly non-power-of-two values, while keeping continuous dimensions large enough for bulk IO.
- **Check PID ordering and L2 reuse**: multi-axis `tl.program_id` without grouping/swizzle hints should be reviewed for program ordering. If neighboring programs reuse the same input tile, suggest PID reorder / L2 grouping validation before changing math.
- **Check indexing strategy as a config**: scalar pointer arithmetic, modulo/floor-div index reconstruction, block pointer/tensor descriptor, and bulk IO + on-chip reshape are alternative implementations of the same logical tile. Recommend comparing them when load/store count is high or address expressions look regular.
- **Check loop/range config**: `tl.range` / `tl.static_range` loops should be reviewed for unroll, `num_stages`, multi-buffering, and flattening. A loop with IO and compute but no effective stage configuration is a pipeline candidate; a loop with scalar-like load/store is a vectorization candidate.
- **Check persistent/grid capping choices**: if grid dimensions can grow too large, use a persistent-kernel style that caps programs by core count and iterates inside the kernel. This is especially useful when larger tile sizes reduce launch/scheduling overhead without exceeding NRAM.
- **Keep the search bounded**: prefer 4-12 targeted configs per kernel, chosen from the static signals above and ranked by observed duration/BW utilization. Every proposed config must include a benchmark or re-trace validation method.

## Reference Heuristics

These heuristics are intentionally lightweight and stable enough for Web-side automation:

- Triton official examples emphasize program ordering, block/tile shape, and autotune knobs (`BLOCK_*`, `num_warps`, `num_stages`) because memory reuse and launch mapping can dominate source-equivalent kernels.
- Helion-style config search is useful for generated kernels because it names the real tuning axes explicitly: block sizes, loop order/flattening, indexing mode, range config, PID mapping, `num_warps`, and `num_stages`.
- Triton `tl.load` supports cache and eviction hints; only suggest them for reused true-gather/table operands after ruling out regular bulk IO.
- Roofline-style reasoning separates memory-shaped and compute-shaped kernels using arithmetic intensity. Use it to choose which optimization family to validate first.
- Vendor Triton optimization guides generally start with profiling context, then inspect IR/source, tune meta-parameters, and only then check lower-level generated code. Keep the final report aligned with that order.
- Recent auto-tuning/agentic Triton work uses static rules plus profiling feedback loops; therefore every recommendation should include a concrete benchmark or re-trace validation method.

## Output Expectations

The JSON output must be machine-readable and include:

- `has_findings`: whether actionable candidates were found.
- `summary`: scanned file count, finding count, and top strategy names.
- `final_report_guidance`: concise Chinese guidance for the parent E2E report, including whether the candidates must be surfaced, whether they should be promoted to a top finding/action, suggested placement, top strategies, candidate summaries, and `required_table_md`, a compact Markdown table with all detected Triton code candidates that can be copied into the final report.
- `kernels`: sorted by priority, each containing `kernel_name`, `file`, optional IO-efficiency metrics, `estimated_profile`, `priority`, `priority_score`, and `findings`.

The Markdown output should be short enough to read in the final AI report:

- A compact summary.
- A top-candidate table with estimated throughput and merged optimization direction/recommendation. Do not use a separate evidence column in the final table.
- Per-kernel findings with evidence lines and recommendations.

## Integration With E2E Reports

When this skill is used by `e2e-profiling-analyzer`, its output should augment the `triton-kernel-efficiency` branch:

- Cite `triton_code_optimization.md` in the branch report.
- Always make `has_findings=true` visible in the final E2E report: copy or faithfully summarize `final_report_guidance.required_table_md` as an independent top-level `## Triton Kernel 代码优化` section, preserving all candidate rows. Place it after `## 优先行动` and before `## 不确定性与下一步`.
- Promote high-priority Triton-code findings to the final report when their kernel time, low bandwidth utilization, or repeated pattern is material.
- Avoid claiming a transformation is definitely profitable. Phrase recommendations as validation targets unless runtime evidence confirms the gain.

## Guardrails

- Do not modify user Triton code.
- Do not run kernels, cnperf, benchmark jobs, or remote workers.
- Do not paste full generated Triton source into final reports.
- Do not treat every complex generated kernel as a defect; use observed duration, IO-efficiency, and repeated patterns to prioritize.
- If evidence is heuristic, state it as a candidate optimization rather than a confirmed bottleneck.
