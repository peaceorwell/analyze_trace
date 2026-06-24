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
5. **Inspect mapping/tuning shape**: look for multi-dimensional `program_id`, `tl.num_programs`, `num_warps`, `num_stages`, missing autotune/config signals, and tile/block sizes that should be swept.
6. **Report as validation targets**: recommendations should say what to benchmark or trace next, not claim guaranteed speedup.

Focus on optimization opportunities visible from generated Triton code:

- **libdevice math replacement**: `tl.sigmoid`, `tl.exp`, `tl.log`, `tl.sqrt`, `tl.erf`, `tl.tanh`, `tl.pow`, `x * tl.sigmoid(x)`, and similar compute-heavy patterns that may map better to `tl.extra.mlu.libdevice.fast_*`.
- **Division lowering**: tensor division inside wide tiles, especially division after broadcast. Prefer reducing division count, multiplying by reciprocal, or using libdevice divide/rcp helpers when precision allows.
- **Fragmented IO / pseudo-gather**: many small `tl.load` / `tl.store` operations, even/odd or first-half/second-half access, fixed stride/interleave patterns, and modulo/floor-div indexing that hides contiguous memory. Recommend bulk load/store plus on-chip slice/cat/broadcast reshaping when the mapping is compile-time regular.
- **True gather / repeated lookup**: index/table-like loads that are not compile-time regular. Recommend validating reuse and considering `cache_modifier=".cg"` only when it is a small reused operand.
- **Reduce layout and tiling**: `tl.sum`, `tl.max`, `tl.min`, `tl.reduce`, especially reductions over axis 1 or repeated reductions that may benefit from retiling or transpose-to-pooling-friendly layout.
- **Grid/retiling issues**: multi-dimensional program IDs, complex `tl.num_programs`, or block parameters that suggest poor core mapping. Recommend checking one-dimensional grid flattening and block-size consistency.
- **Block pointer / tensor descriptor shape**: missing `tl.make_block_ptr` on obviously bulk-like IO, or existing block pointers with suspicious `block_shape`, `order`, or stride usage. Recommend descriptor/bulk-IO validation only when supported by the current MLU Triton stack.
- **Autotune/meta-parameter sweep**: when a material kernel has reductions, low bandwidth utilization, multi-axis grid, `tl.dot`, or complex tiling but no visible `@triton.autotune` / `triton.Config`, recommend sweeping `BLOCK_*`, `num_warps`, `num_stages`, and grid flattening.
- **Cache hint validation**: for true table/index/gather operands with reuse, consider `cache_modifier` / `eviction_policy` experiments; do not use cache hints as a substitute for regularizing pseudo-gather.
- **dtype conversion chains**: repeated `.to(tl.float32)`, `.to(tl.float16)`, `.to(tl.bfloat16)`, `.to(tl.int*)` conversions around math or stores. Recommend removing redundant conversions or using fast conversion helpers where available.
- **Static IO/compute estimate**: infer domain or tile size from `size_hints`, `block_shape`, or `tl.arange`; count `tl.load` / `tl.store` bytes and scalar `tl.*` arithmetic/comparison/math/reduce operations. Report estimated IO throughput, compute throughput, and arithmetic intensity as heuristic signals, not hardware counters.

## Reference Heuristics

These heuristics are intentionally lightweight and stable enough for Web-side automation:

- Triton official examples emphasize program ordering, block/tile shape, and autotune knobs (`BLOCK_*`, `num_warps`, `num_stages`) because memory reuse and launch mapping can dominate source-equivalent kernels.
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
