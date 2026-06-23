#!/usr/bin/env python3
"""Static optimization-candidate analysis for generated Triton output_code.

This script intentionally does not execute kernels. It scans generated Triton
source and optional IO-efficiency metadata, then emits deterministic JSON or a
compact Markdown report for AI analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CODE_SUFFIXES = {".txt", ".py", ".triton", ".code"}


MATH_PATTERNS = [
    ("silu pattern", re.compile(r"\b\w+\s*\*\s*tl\.sigmoid\s*\("), "fast_silu"),
    ("tl.sigmoid", re.compile(r"\btl\.sigmoid\s*\("), "fast_sigmoid"),
    ("tl.exp", re.compile(r"\btl\.exp(?:2|10)?\s*\("), "fast_expf / fast_exp*"),
    ("tl.log", re.compile(r"\btl\.log(?:2|10)?\s*\("), "fast_log*"),
    ("tl.sqrt", re.compile(r"\btl\.sqrt\s*\("), "fast_sqrt"),
    ("tl.erf", re.compile(r"\btl\.erf\s*\("), "fast_erf"),
    ("tl.tanh", re.compile(r"\btl\.tanh\s*\("), "fast_tanh"),
    ("tl.pow", re.compile(r"\btl\.(?:math\.)?pow\s*\("), "fast_powf / fast_powi"),
]


REDUCE_RE = re.compile(r"\btl\.(?:sum|max|min|reduce)\s*\(")
LOAD_RE = re.compile(r"\btl\.load\s*\(")
STORE_RE = re.compile(r"\btl\.store\s*\(")
PROGRAM_ID_RE = re.compile(r"\btl\.program_id\s*\(\s*(\d+)")
NUM_PROGRAMS_RE = re.compile(r"\btl\.num_programs\s*\(")
CONVERSION_RE = re.compile(r"\.to\s*\(\s*tl\.(?:float32|float16|bfloat16|int(?:8|16|32)|uint(?:8|16|32))")
SINGLE_DIV_RE = re.compile(r"(?<!/)/(?!/)")
FLOOR_DIV_OR_MOD_RE = re.compile(r"(//|%)")
INDEX_WORD_RE = re.compile(r"\b(?:idx|index|indices|offset|offsets|offs|mask|arange)\b", re.I)
GATHER_WORD_RE = re.compile(r"\b(?:index|indices|idx|lookup|table|gather|embedding)\b", re.I)
EVEN_ODD_HALF_RE = re.compile(r"\b(?:even|odd|first|second|half|interleave|strided|stride)\b", re.I)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_comment(line: str) -> str:
    quote = None
    for idx, char in enumerate(line):
        if char in ("'", '"') and (idx == 0 or line[idx - 1] != "\\"):
            quote = None if quote == char else char if quote is None else quote
        elif char == "#" and quote is None:
            return line[:idx]
    return line


def _code_lines(text: str) -> list[str]:
    return [_strip_comment(line).rstrip() for line in text.splitlines()]


def _interesting_lines(lines: list[str], regex: re.Pattern[str], limit: int = 4) -> list[str]:
    out = []
    for line in lines:
        if regex.search(line):
            stripped = line.strip()
            if stripped:
                out.append(stripped)
        if len(out) >= limit:
            break
    return out


def _count(regex: re.Pattern[str], text: str) -> int:
    return len(regex.findall(text))


def _severity(score: float) -> str:
    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def _classify_kernel_name(name: str) -> str:
    lower = name.lower()
    if "triton_poi" in lower or "_poi_" in lower:
        return "pointwise"
    if "triton_red" in lower or "_red_" in lower or "reduce" in lower:
        return "reduce"
    if "triton_per" in lower or "_per_" in lower:
        return "persistent"
    if "triton" in lower:
        return "triton_other"
    return "unknown"


def _load_efficiency_meta(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(_read_text(p))
    meta: dict[str, dict[str, Any]] = {}
    for profile in payload.get("profiles", []):
        for kernel in profile.get("top_low_bandwidth_kernels", []):
            item = dict(kernel)
            item["profile_label"] = profile.get("label") or profile.get("db")
            code_file = kernel.get("output_code_file")
            kernel_name = kernel.get("kernel_name")
            keys = []
            if code_file:
                cp = Path(code_file)
                keys.extend([str(cp), cp.name])
                try:
                    keys.append(str(cp.resolve()))
                except OSError:
                    pass
            if kernel_name:
                keys.append(str(kernel_name))
            for key in keys:
                meta[key] = item
    return meta


def _iter_code_files(input_dir: str | None, meta: dict[str, dict[str, Any]]) -> list[Path]:
    seen = set()
    files: list[Path] = []
    if input_dir:
        root = Path(input_dir)
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file() and (path.suffix in CODE_SUFFIXES or "output_code" in path.name):
                    resolved = str(path.resolve())
                    if resolved not in seen:
                        seen.add(resolved)
                        files.append(path)
    for item in meta.values():
        code_file = item.get("output_code_file")
        if not code_file:
            continue
        path = Path(code_file)
        if path.exists():
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return files


def _meta_for_file(path: Path, meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [str(path), path.name]
    try:
        candidates.append(str(path.resolve()))
    except OSError:
        pass
    for key in candidates:
        if key in meta:
            return dict(meta[key])
    return {}


def _make_finding(
    *,
    category: str,
    strategy: str,
    score: float,
    evidence: str,
    recommendation: str,
    lines: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "strategy": strategy,
        "severity": _severity(score),
        "score": round(score, 3),
        "evidence": evidence,
        "evidence_lines": lines or [],
        "recommendation": recommendation,
    }


def analyze_code_file(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    text = _read_text(path)
    lines = _code_lines(text)
    code = "\n".join(lines)
    kernel_name = meta.get("kernel_name") or path.stem
    total_ms = float(meta.get("total_ms") or 0.0)
    util = meta.get("bandwidth_utilization")
    try:
        bandwidth_utilization = float(util) if util is not None else None
    except (TypeError, ValueError):
        bandwidth_utilization = None
    impact = min(total_ms / 2.0, 8.0)
    if bandwidth_utilization is not None:
        impact += max(0.0, min(1.0, 1.0 - bandwidth_utilization)) * 4.0

    findings: list[dict[str, Any]] = []

    libdevice_hits = []
    if "tl.extra.mlu.libdevice" not in code:
        for label, regex, replacement in MATH_PATTERNS:
            hits = _count(regex, code)
            if hits:
                libdevice_hits.append(f"{label} x{hits} -> {replacement}")
    if libdevice_hits:
        score = impact + min(len(libdevice_hits), 4)
        findings.append(
            _make_finding(
                category="libdevice_math_candidate",
                strategy="libdevice-opt",
                score=score,
                evidence="; ".join(libdevice_hits[:5]),
                lines=_interesting_lines(lines, re.compile(r"tl\.(?:sigmoid|exp|log|sqrt|erf|tanh|(?:math\.)?pow)\s*\(")),
                recommendation="检查这些数学函数是否可替换为 Cambricon libdevice fast_* 实现；优先验证耗时高或低带宽利用率的 kernel。",
            )
        )

    div_lines = [line.strip() for line in lines if SINGLE_DIV_RE.search(line) and "http" not in line and line.strip()]
    tensor_div_lines = [line for line in div_lines if any(token in line for token in ("tl.", "[:,", "[None", "None]", " / "))]
    if tensor_div_lines:
        score = impact + min(len(tensor_div_lines), 4)
        findings.append(
            _make_finding(
                category="tensor_division_candidate",
                strategy="div-to-mul",
                score=score,
                evidence=f"发现 {len(tensor_div_lines)} 行可能的张量除法。",
                lines=tensor_div_lines[:4],
                recommendation="若除数可先在低维空间求倒数，优先改为 reciprocal + multiply，减少 broadcast 后的大量除法；必要时验证 fast_rcp/fast_dividef。",
            )
        )

    index_div_mod_lines = [
        line.strip()
        for line in lines
        if FLOOR_DIV_OR_MOD_RE.search(line) and INDEX_WORD_RE.search(line) and line.strip()
    ]
    if index_div_mod_lines:
        score = impact + min(len(index_div_mod_lines), 4)
        findings.append(
            _make_finding(
                category="index_div_mod_or_boundary_fold",
                strategy="canonicalize / bulk-io-opt",
                score=score,
                evidence=f"索引或 mask 相关表达式中出现 {len(index_div_mod_lines)} 处 // 或 %。",
                lines=index_div_mod_lines[:4],
                recommendation="区分算法语义取模与边界折回；边界折回应改为真实线性地址 + mask/other，固定规则映射可考虑 bulk load/store 后片上重排。",
            )
        )

    load_count = _count(LOAD_RE, code)
    store_count = _count(STORE_RE, code)
    fragmented_lines = _interesting_lines(lines, re.compile(r"tl\.(?:load|store)\s*\(|even|odd|half|interleave|stride", re.I))
    if load_count >= 4 or store_count >= 2 or (EVEN_ODD_HALF_RE.search(code) and load_count >= 2):
        score = impact + min(load_count + store_count, 8) / 2.0
        findings.append(
            _make_finding(
                category="fragmented_or_pseudo_discrete_io",
                strategy="bulk-io-opt",
                score=score,
                evidence=f"tl.load x{load_count}, tl.store x{store_count}，可能存在碎片化或规则离散访存。",
                lines=fragmented_lines[:4],
                recommendation="检查是否为前后半段、奇偶、固定 stride/reshape 等伪离散访存；若地址映射可编译期推导，优先改成连续 bulk IO + 片上 slice/cat/broadcast。",
            )
        )

    gather_lines = [
        line.strip()
        for line in lines
        if "tl.load" in line and GATHER_WORD_RE.search(line) and line.strip()
    ]
    if gather_lines:
        score = impact + min(len(gather_lines), 3)
        findings.append(
            _make_finding(
                category="true_gather_or_reused_lookup",
                strategy="llc-cache-opt",
                score=score,
                evidence=f"发现 {len(gather_lines)} 行 index/table/lookup 相关 load。",
                lines=gather_lines[:4],
                recommendation="如果源表较小且跨 program 复用，验证 `cache_modifier=\".cg\"`；若是固定规则映射，优先走 bulk-io 而不是 cache hint。",
            )
        )

    reduce_count = _count(REDUCE_RE, code)
    if reduce_count:
        axis1 = bool(re.search(r"tl\.(?:sum|max|min|reduce)\s*\([^)]*axis\s*=\s*1", code))
        score = impact + min(reduce_count, 5)
        strategy = "trans-opt / reduce-opt" if axis1 else "reduce-opt / retiling"
        recommendation = (
            "axis=1 或低维 reduce 可尝试转置到 pooling 友好的维度，并复查 reduce 轴分块。"
            if axis1
            else "检查 reduce 轴是否完整分块、是否可合并相邻 reduce 轴，以及是否存在多轮小 reduce。"
        )
        findings.append(
            _make_finding(
                category="reduce_layout_or_tiling_candidate",
                strategy=strategy,
                score=score,
                evidence=f"发现 tl.reduce/sum/max/min x{reduce_count}。",
                lines=_interesting_lines(lines, REDUCE_RE),
                recommendation=recommendation,
            )
        )

    program_axes = sorted({int(axis) for axis in PROGRAM_ID_RE.findall(code)})
    if len(program_axes) >= 2 or NUM_PROGRAMS_RE.search(code):
        score = impact + len(program_axes)
        findings.append(
            _make_finding(
                category="grid_or_retiling_candidate",
                strategy="modify-grid / retiling",
                score=score,
                evidence=f"program_id axes={program_axes or 'unknown'}，存在多维 grid 或动态 num_programs 信号。",
                lines=_interesting_lines(lines, re.compile(r"program_id|num_programs|grid\s*=")),
                recommendation="检查能否展平成一维 grid，并让并行轴分块与核心数/num_warps 匹配；同时复查 BLOCK_* 是否实际参与 offset/mask。",
            )
        )

    conversion_count = _count(CONVERSION_RE, code)
    if conversion_count >= 2:
        score = impact + min(conversion_count, 5) / 2.0
        findings.append(
            _make_finding(
                category="dtype_conversion_chain",
                strategy="canonicalize / libdevice-opt",
                score=score,
                evidence=f"发现 .to(tl.*) 转换 x{conversion_count}。",
                lines=_interesting_lines(lines, CONVERSION_RE),
                recommendation="消除重复 dtype 往返转换；float 到 int8/int16/uint 等路径可验证 fast_float2* 类 helper 是否适用。",
            )
        )

    findings.sort(key=lambda item: item["score"], reverse=True)
    priority_score = round(sum(item["score"] for item in findings[:3]) + impact, 3)
    priority = _severity(priority_score)
    return {
        "kernel_name": kernel_name,
        "file": str(path),
        "profile_label": meta.get("profile_label"),
        "kernel_family": _classify_kernel_name(kernel_name),
        "total_ms": total_ms,
        "avg_io_efficiency": meta.get("avg_io_efficiency"),
        "bandwidth_utilization": bandwidth_utilization,
        "improvement_target": meta.get("improvement_target"),
        "load_count": load_count,
        "store_count": store_count,
        "reduce_count": reduce_count,
        "priority": priority if findings else "none",
        "priority_score": priority_score if findings else 0.0,
        "findings": findings,
    }


def _split_strategies(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"\s*/\s*", value or "") if item.strip()]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _final_report_guidance(kernels: list[dict[str, Any]], scanned_files: int) -> dict[str, Any]:
    if not kernels:
        return {
            "must_surface": False,
            "promote_to_finding": False,
            "suggested_placement": "产物",
            "summary_cn": "未从可用 Triton output_code 中识别到明确代码级优化候选。",
            "candidates": [],
        }

    top_strategies: dict[str, int] = {}
    for kernel in kernels:
        for finding in kernel.get("findings", []):
            for strategy in _split_strategies(finding.get("strategy", "")):
                top_strategies[strategy] = top_strategies.get(strategy, 0) + 1
    sorted_strategies = sorted(top_strategies.items(), key=lambda item: (-item[1], item[0]))
    material = [
        kernel for kernel in kernels
        if kernel.get("priority") == "high"
        and (
            _to_float(kernel.get("total_ms")) >= 1.0
            or _to_float(kernel.get("improvement_target")) >= 1.0
            or (
                kernel.get("bandwidth_utilization") is not None
                and _to_float(kernel.get("bandwidth_utilization"), default=1.0) <= 0.5
            )
        )
    ]
    top_kernel = kernels[0]
    top_strategy_names = [name for name, _ in sorted_strategies[:4]]
    summary_cn = (
        f"扫描 {scanned_files} 个 Triton output_code，"
        f"{len(kernels)} 个 kernel 存在代码级候选，"
        f"主要方向为 {', '.join(top_strategy_names) or '待确认'}。"
    )
    if material:
        summary_cn += (
            f" Top 候选 `{top_kernel.get('kernel_name')}` "
            f"耗时 {_fmt_ms(top_kernel.get('total_ms'))} ms，建议作为验证目标进入优先行动。"
        )
    else:
        summary_cn += " 当前候选绝对耗时较小，建议放在主要瓶颈修复后的下一步验证。"

    candidates = []
    for kernel in kernels[:5]:
        findings = kernel.get("findings", [])
        candidates.append({
            "kernel_name": kernel.get("kernel_name"),
            "file": kernel.get("file"),
            "priority": kernel.get("priority"),
            "total_ms": kernel.get("total_ms"),
            "bandwidth_utilization": kernel.get("bandwidth_utilization"),
            "strategies": sorted({
                strategy
                for finding in findings[:3]
                for strategy in _split_strategies(finding.get("strategy", ""))
            }),
            "evidence": "; ".join(finding.get("evidence", "") for finding in findings[:2] if finding.get("evidence")),
            "recommendation": findings[0].get("recommendation", "") if findings else "",
        })

    return {
        "must_surface": True,
        "promote_to_finding": bool(material),
        "suggested_placement": "结论概览/优先行动" if material else "关键指标/不确定性与下一步",
        "summary_cn": summary_cn,
        "top_strategies": [{"strategy": name, "count": count} for name, count in sorted_strategies[:8]],
        "candidates": candidates,
    }


def analyze(input_dir: str | None, efficiency_json: str | None, top: int = 20) -> dict[str, Any]:
    meta = _load_efficiency_meta(efficiency_json)
    files = _iter_code_files(input_dir, meta)
    kernels = [analyze_code_file(path, _meta_for_file(path, meta)) for path in files]
    kernels = [item for item in kernels if item["findings"]]
    kernels.sort(key=lambda item: item["priority_score"], reverse=True)
    kernels = kernels[:top]
    strategies: dict[str, int] = {}
    for kernel in kernels:
        for finding in kernel["findings"]:
            for strategy in re.split(r"\s*/\s*", finding["strategy"]):
                strategies[strategy] = strategies.get(strategy, 0) + 1
    top_strategies = sorted(strategies.items(), key=lambda item: (-item[1], item[0]))
    return {
        "schema_version": 1,
        "has_findings": bool(kernels),
        "summary": {
            "scanned_files": len(files),
            "kernels_with_findings": len(kernels),
            "finding_count": sum(len(item["findings"]) for item in kernels),
            "top_strategies": [{"strategy": name, "count": count} for name, count in top_strategies[:8]],
        },
        "final_report_guidance": _final_report_guidance(kernels, len(files)),
        "kernels": kernels,
    }


def _fmt_ms(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.2f}"


def _fmt_util(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number * 100:.1f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = ["# Triton Code Optimization Candidates", ""]
    summary = payload["summary"]
    lines.append(f"- Scanned output_code files: {summary['scanned_files']}")
    lines.append(f"- Kernels with findings: {summary['kernels_with_findings']}")
    lines.append(f"- Finding count: {summary['finding_count']}")
    if summary["top_strategies"]:
        joined = ", ".join(f"{item['strategy']} x{item['count']}" for item in summary["top_strategies"])
        lines.append(f"- Top strategies: {joined}")
    guidance = payload.get("final_report_guidance") or {}
    if guidance:
        lines.append(f"- Final report placement: {guidance.get('suggested_placement', '-')}")
        lines.append(f"- Summary for final report: {guidance.get('summary_cn', '-')}")
    if not payload["has_findings"]:
        lines.append("")
        lines.append("No actionable Triton code optimization candidates were detected from available output_code.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "",
            "## Top Candidate Kernels",
            "",
            "| Priority | Kernel | Total ms | BW util | Main strategies | Evidence |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for kernel in payload["kernels"]:
        strategies = []
        for finding in kernel["findings"][:3]:
            strategies.append(finding["strategy"])
        evidence = "; ".join(f"{f['category']}({f['severity']})" for f in kernel["findings"][:3])
        lines.append(
            f"| {kernel['priority']} | `{kernel['kernel_name']}` | {_fmt_ms(kernel['total_ms'])} | "
            f"{_fmt_util(kernel.get('bandwidth_utilization'))} | {', '.join(strategies)} | {evidence} |"
        )

    lines.extend(["", "## Details", ""])
    for kernel in payload["kernels"]:
        lines.append(f"### {kernel['kernel_name']}")
        lines.append(
            f"- File: `{kernel['file']}`; total_ms={_fmt_ms(kernel['total_ms'])}; "
            f"bandwidth_utilization={_fmt_util(kernel.get('bandwidth_utilization'))}; "
            f"loads={kernel['load_count']}; stores={kernel['store_count']}; reduces={kernel['reduce_count']}"
        )
        for finding in kernel["findings"]:
            lines.append(f"- **{finding['severity']} / {finding['strategy']}**: {finding['evidence']}")
            lines.append(f"  - Recommendation: {finding['recommendation']}")
            for evidence_line in finding.get("evidence_lines", [])[:3]:
                lines.append(f"  - Evidence: `{evidence_line}`")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", help="directory containing dumped Triton output_code files")
    parser.add_argument("--efficiency-json", help="optional triton_kernel_efficiency.json")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--format", choices=("json", "text"), default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze(args.input_dir, args.efficiency_json, top=args.top)
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_markdown(payload))


if __name__ == "__main__":
    main()
