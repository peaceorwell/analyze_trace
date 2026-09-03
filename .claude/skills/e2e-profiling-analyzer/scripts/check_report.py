#!/usr/bin/env python3
"""Check a final report against the Final Report Contract and Report Readability Gate.

Deterministic replacement for reading the prose gate line by line: run this before and
after writing `report.md`, fix every ERROR, then re-run until the gate passes.

Usage:
    python3 check_report.py report.md
    python3 check_report.py report.md --format json --analysis-dir .
    python3 check_report.py report.md --budget 1200 --strict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Sections the contract requires, in contract order.
REQUIRED_SECTIONS = [
    "## 结论概览",
    "## 关键指标",
    "## 分布式与通信概况",
    "## 优先行动",
    "## 不确定性与下一步",
    "## 产物",
]
# Parallel summaries that duplicate 结论概览 and make the Web report unreadable.
FORBIDDEN_SECTIONS = ["## 主要发现", "## 详细分析", "## 执行摘要", "## 主要回退", "## 摘要"]
TRITON_SECTION = "## Triton Kernel 代码优化"
# 2-4 prioritized findings; 3 is the preferred default.
MIN_FINDINGS = 2
MAX_FINDINGS = 4
# Web readability budget for the narrative before `## 产物`, in non-whitespace characters.
DEFAULT_BUDGET = 1200
# Fence languages that indicate raw tool output pasted into a user-facing report.
RAW_OUTPUT_LANGUAGES = {"text", "console", "shell", "sh", "bash", "log", "traceback"}
FINDING_RE = re.compile(r"^###\s*发现\s*\d+\s*[:：]")
FLAT_BULLET_RE = re.compile(r"^\s*(?:[-+]|\*(?!\*))\s+\*{0,2}(结论|证据|建议)\*{0,2}\s*[:：]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
BOLD_LINE_RE = re.compile(r"^\*\*[^*]+\*\*")
FAILURE_TITLE_RE = re.compile(r"^#\s.*(失败|Failure|failed)", re.IGNORECASE)


def _load_json(path: Path | None):
    """Read an optional JSON side input; a missing or broken file is simply absent."""
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _strip_code_fences(lines: list[str]) -> tuple[list[str], list[tuple[int, str]]]:
    """Return prose lines plus (line_number, language) for every fenced block."""
    prose: list[str] = []
    fences: list[tuple[int, str]] = []
    language = None
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            if language is None:
                language = stripped[3:].strip().lower()
                fences.append((index, language))
            else:
                language = None
            continue
        if language is None:
            prose.append(line)
    return prose, fences


def _sections(lines: list[str]) -> list[tuple[int, int, str]]:
    """Top-level (##) sections as (level, line_number, title)."""
    found = []
    for index, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            found.append((len(match.group(1)), index, match.group(2)))
    return found


def _body_before_artifacts(text: str) -> str:
    marker = "\n## 产物"
    position = text.find(marker)
    return text if position < 0 else text[:position]


def _table_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.lstrip().startswith("|"):
            current.append(line.strip())
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def check_report(
    report_path: Path,
    budget: int = DEFAULT_BUDGET,
    triton_json: Path | None = None,
    compile_json: Path | None = None,
) -> dict:
    problems: list[dict] = []

    def add(level: str, check: str, message: str, line: int | None = None) -> None:
        problems.append({"level": level, "check": check, "message": message, "line": line})

    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "report": str(report_path),
            "ok": False,
            "is_failure_report": False,
            "problems": [
                {"level": "error", "check": "readable", "message": str(exc), "line": None}
            ],
            "stats": {},
        }

    lines = text.splitlines()
    prose_lines, fences = _strip_code_fences(lines)
    headings = _sections(lines)
    top_level = [(number, title) for level, number, title in headings if level == 2]
    titles = [title for _, title in top_level]
    is_failure_report = any(FAILURE_TITLE_RE.match(line) for line in lines[:5])

    h1 = [title for level, _, title in headings if level == 1]
    if not h1:
        add("error", "h1", "report has no H1 title")
    elif not is_failure_report and h1[0] != "AI 性能分析报告":
        add("error", "h1", f"H1 must be `# AI 性能分析报告`, found `# {h1[0]}`", 1)
    if len(h1) > 1:
        add("error", "h1", f"report has {len(h1)} H1 titles, expected 1")

    if is_failure_report:
        # A failure report intentionally carries raw stdout/stderr and no findings.
        return {
            "report": str(report_path),
            "ok": not any(p["level"] == "error" for p in problems),
            "is_failure_report": True,
            "problems": problems,
            "stats": {"sections": titles},
        }

    for section in REQUIRED_SECTIONS:
        title = section[3:]
        count = titles.count(title)
        if count == 0:
            add("error", "required_section", f"missing required section `{section}`")
        elif count > 1:
            add("error", "duplicate_section", f"section `{section}` appears {count} times")

    for section in FORBIDDEN_SECTIONS:
        title = section[3:]
        if title in titles:
            add(
                "error",
                "forbidden_section",
                f"`{section}` duplicates `## 结论概览`; keep one report structure",
            )

    duplicates = sorted({title for title in titles if titles.count(title) > 1})
    for title in duplicates:
        if f"## {title}" not in REQUIRED_SECTIONS:
            add("error", "duplicate_section", f"top-level section `## {title}` is repeated")

    order = [titles.index(s[3:]) for s in REQUIRED_SECTIONS if s[3:] in titles]
    if order != sorted(order):
        add("error", "section_order", "top-level sections are not in Final Report Contract order")

    findings = [line for line in prose_lines if FINDING_RE.match(line)]
    if len(findings) < MIN_FINDINGS or len(findings) > MAX_FINDINGS:
        add(
            "error",
            "finding_count",
            f"found {len(findings)} `### 发现 N：` blocks, contract requires "
            f"{MIN_FINDINGS}-{MAX_FINDINGS}",
        )
    for label, marker in (("结论", "**结论：**"), ("证据", "**证据：**"), ("建议", "**建议：**")):
        count = text.count(marker)
        if count < len(findings):
            add(
                "error",
                "finding_block",
                f"{count} `{marker}` paragraphs for {len(findings)} findings; "
                f"every finding needs its own {label}",
            )

    for index, line in enumerate(lines, start=1):
        if FLAT_BULLET_RE.match(line):
            add(
                "error",
                "flat_bullets",
                "use `**结论：**` / `**证据：**` / `**建议：**` paragraphs, not sibling bullets",
                index,
            )
            break

    for block in _table_blocks(prose_lines):
        if len(block) < 2 or not re.match(r"^\|[\s:|-]+\|$", block[1]):
            add("error", "table_separator", f"Markdown table missing header separator: {block[0]}")
            break

    for line_number, language in fences:
        if language in RAW_OUTPUT_LANGUAGES:
            add(
                "error",
                "raw_output",
                f"raw ```{language} output block belongs in a stage artifact, not `report.md`",
                line_number,
            )
            break

    previous_bold = False
    for index, line in enumerate(prose_lines, start=1):
        current_bold = bool(BOLD_LINE_RE.match(line.strip())) and not line.strip().startswith("**结论")
        if current_bold and previous_bold:
            add("warn", "metadata_block", "consecutive bold metadata lines need a blank line between them", index)
            break
        previous_bold = current_bold

    body = _body_before_artifacts(text)
    body_chars = len(re.sub(r"\s", "", body))
    if body_chars > budget:
        add(
            "warn",
            "length_budget",
            f"{body_chars} non-whitespace characters before `## 产物` exceed the {budget} budget; "
            "move detail into stage artifacts",
        )

    artifacts_index = titles.index("产物") if "产物" in titles else -1
    if artifacts_index >= 0:
        start = top_level[artifacts_index][0]
        artifact_lines = [line for line in lines[start:] if line.strip().startswith(("-", "*", "|"))]
        if not artifact_lines:
            add("error", "artifacts", "`## 产物` must list generated DBs, stage reports, and logs")

    triton = _load_json(triton_json)
    if isinstance(triton, dict) and triton.get("has_findings"):
        if TRITON_SECTION[3:] not in titles:
            add(
                "error",
                "triton_section",
                f"`triton_code_optimization.json` has findings; add top-level `{TRITON_SECTION}`",
            )
        else:
            position = titles.index(TRITON_SECTION[3:])
            if "优先行动" in titles and position < titles.index("优先行动"):
                add("error", "triton_section", f"`{TRITON_SECTION}` must follow `## 优先行动`")
            if "不确定性与下一步" in titles and position > titles.index("不确定性与下一步"):
                add("error", "triton_section", f"`{TRITON_SECTION}` must precede `## 不确定性与下一步`")
            candidates = triton.get("final_report_guidance", {}).get("candidates")
            if isinstance(candidates, list) and candidates:
                start = top_level[position][0]
                end = top_level[position + 1][0] - 1 if position + 1 < len(top_level) else len(lines)
                rows = [
                    line
                    for line in lines[start:end]
                    if line.lstrip().startswith("|") and not re.match(r"^\|[\s:|-]+\|$", line.strip())
                ]
                # One header row plus one row per candidate.
                if len(rows) < len(candidates) + 1:
                    add(
                        "error",
                        "triton_candidates",
                        f"`{TRITON_SECTION}` lists {max(len(rows) - 1, 0)} of "
                        f"{len(candidates)} candidates; include all of them",
                    )

    compile_payload = _load_json(compile_json)
    if isinstance(compile_payload, dict):
        custom_op = compile_payload.get("custom_op_simple_aten")
        if isinstance(custom_op, dict) and custom_op.get("must_report"):
            summary_start = top_level[titles.index("结论概览")][0] if "结论概览" in titles else 0
            summary_end = (
                top_level[titles.index("关键指标")][0] if "关键指标" in titles else len(lines)
            )
            summary = "\n".join(lines[summary_start:summary_end]).lower()
            if not any(key in summary for key in ("custom", "自定义", "aten")):
                add(
                    "error",
                    "custom_op_finding",
                    "`compile_segmentation.json` sets custom_op_simple_aten.must_report=true; "
                    "reserve one finding for the custom-op / simple-aten issue",
                )

    return {
        "report": str(report_path),
        "ok": not any(problem["level"] == "error" for problem in problems),
        "is_failure_report": False,
        "problems": problems,
        "stats": {
            "lines": len(lines),
            "body_chars_before_artifacts": body_chars,
            "budget": budget,
            "findings": len(findings),
            "sections": titles,
        },
    }


def emit_text(payload: dict) -> None:
    print(f"Report: {payload['report']}")
    stats = payload.get("stats", {})
    if stats:
        print(
            f"Findings: {stats.get('findings')} | "
            f"Body chars before 产物: {stats.get('body_chars_before_artifacts')}"
            f"/{stats.get('budget')}"
        )
    for problem in payload["problems"]:
        location = f" (line {problem['line']})" if problem.get("line") else ""
        print(f"{problem['level'].upper()}: [{problem['check']}] {problem['message']}{location}")
    print("Gate:", "PASS" if payload["ok"] else "FAIL")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check report.md against the Final Report Contract and Readability Gate"
    )
    parser.add_argument("report")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="non-whitespace character budget before `## 产物` (default: %(default)s)",
    )
    parser.add_argument("--triton-json", help="path to triton_code_optimization.json")
    parser.add_argument("--compile-json", help="path to compile_segmentation.json")
    parser.add_argument(
        "--analysis-dir",
        help="directory searched for the JSON side inputs when the flags above are omitted",
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    search_dir = Path(args.analysis_dir) if args.analysis_dir else report_path.parent
    triton_json = Path(args.triton_json) if args.triton_json else None
    compile_json = Path(args.compile_json) if args.compile_json else None
    if triton_json is None:
        found = sorted(search_dir.rglob("triton_code_optimization.json"))
        triton_json = found[0] if found else None
    if compile_json is None:
        found = sorted(search_dir.rglob("*compile_segmentation.json"))
        compile_json = found[0] if found else None

    payload = check_report(report_path, args.budget, triton_json, compile_json)
    if args.format == "json":
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        emit_text(payload)

    failed = not payload["ok"]
    if args.strict and any(problem["level"] == "warn" for problem in payload["problems"]):
        failed = True
    raise SystemExit(2 if failed else 0)


if __name__ == "__main__":
    main()
