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

Focus on optimization opportunities visible from generated Triton code:

- **libdevice math replacement**: `tl.sigmoid`, `tl.exp`, `tl.log`, `tl.sqrt`, `tl.erf`, `tl.tanh`, `tl.pow`, `x * tl.sigmoid(x)`, and similar compute-heavy patterns that may map better to `tl.extra.mlu.libdevice.fast_*`.
- **Division lowering**: tensor division inside wide tiles, especially division after broadcast. Prefer reducing division count, multiplying by reciprocal, or using libdevice divide/rcp helpers when precision allows.
- **Fragmented IO / pseudo-gather**: many small `tl.load` / `tl.store` operations, even/odd or first-half/second-half access, fixed stride/interleave patterns, and modulo/floor-div indexing that hides contiguous memory. Recommend bulk load/store plus on-chip slice/cat/broadcast reshaping when the mapping is compile-time regular.
- **True gather / repeated lookup**: index/table-like loads that are not compile-time regular. Recommend validating reuse and considering `cache_modifier=".cg"` only when it is a small reused operand.
- **Reduce layout and tiling**: `tl.sum`, `tl.max`, `tl.min`, `tl.reduce`, especially reductions over axis 1 or repeated reductions that may benefit from retiling or transpose-to-pooling-friendly layout.
- **Grid/retiling issues**: multi-dimensional program IDs, complex `tl.num_programs`, or block parameters that suggest poor core mapping. Recommend checking one-dimensional grid flattening and block-size consistency.
- **dtype conversion chains**: repeated `.to(tl.float32)`, `.to(tl.float16)`, `.to(tl.bfloat16)`, `.to(tl.int*)` conversions around math or stores. Recommend removing redundant conversions or using fast conversion helpers where available.

## Output Expectations

The JSON output must be machine-readable and include:

- `has_findings`: whether actionable candidates were found.
- `summary`: scanned file count, finding count, and top strategy names.
- `final_report_guidance`: concise Chinese guidance for the parent E2E report, including whether the candidates must be surfaced, whether they should be promoted to a top finding/action, suggested placement, top strategies, top candidate summaries, and `required_table_md`, a compact Markdown table that can be copied into the final report.
- `kernels`: sorted by priority, each containing `kernel_name`, `file`, optional IO-efficiency metrics, `priority`, `priority_score`, and `findings`.

The Markdown output should be short enough to read in the final AI report:

- A compact summary.
- A top-candidate table.
- Per-kernel findings with evidence lines and recommendations.

## Integration With E2E Reports

When this skill is used by `e2e-profiling-analyzer`, its output should augment the `triton-kernel-efficiency` branch:

- Cite `triton_code_optimization.md` in the branch report.
- Always make `has_findings=true` visible in the final E2E report: copy or faithfully summarize `final_report_guidance.required_table_md` as a compact `Triton Kernel 代码优化候选` table, either under `优先行动` when `final_report_guidance.promote_to_finding=true`, or under `不确定性与下一步` when larger bottlenecks dominate.
- Promote high-priority Triton-code findings to the final report when their kernel time, low bandwidth utilization, or repeated pattern is material.
- Avoid claiming a transformation is definitely profitable. Phrase recommendations as validation targets unless runtime evidence confirms the gain.

## Guardrails

- Do not modify user Triton code.
- Do not run kernels, cnperf, benchmark jobs, or remote workers.
- Do not paste full generated Triton source into final reports.
- Do not treat every complex generated kernel as a defect; use observed duration, IO-efficiency, and repeated patterns to prioritize.
- If evidence is heuristic, state it as a candidate optimization rather than a confirmed bottleneck.
