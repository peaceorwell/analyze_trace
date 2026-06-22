import importlib.util
import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/triton_fusion_coverage.py"
COLLECT_SCRIPT = ROOT / ".claude/skills/e2e-profiling-comparator/scripts/collect_profile_tables.py"
COMPARE_SCRIPT = ROOT / ".claude/skills/e2e-profiling-comparator/scripts/compare_profile_tables.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_kernel_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE string_table(ID INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        """
        CREATE TABLE device_task_kernel_data(
            processId INTEGER,
            deviceId INTEGER,
            nameId INTEGER,
            start INTEGER,
            end INTEGER,
            isComputation INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO string_table(ID, string) VALUES(?, ?)",
        [
            (1, "triton_poi_fused_add"),
            (2, "aten::add_elementwise_kernel"),
            (3, "reduceKernelAdd"),
            (4, "MLUUnion1BMMEXGEMM"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO device_task_kernel_data(processId, deviceId, nameId, start, end, isComputation)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        [
            (0, 0, 1, 0, 1_000_000, 1),
            (0, 0, 2, 1_000_000, 3_000_000, 1),
            (0, 0, 3, 3_000_000, 6_000_000, 1),
            (0, 0, 4, 6_000_000, 10_000_000, 1),
        ],
    )
    conn.commit()
    conn.close()


def test_analyzer_highlights_unfused_pointwise_and_reduce(tmp_path):
    module = load_module("triton_fusion_coverage", ANALYZER_SCRIPT)
    db_path = tmp_path / "trace.db"
    make_kernel_db(db_path)

    payload = module.analyze_db(str(db_path), top=10)
    granularity = payload["fusion_granularity"]

    assert granularity["highlight_unfused_pointwise"] is True
    assert granularity["unfused_pointwise_ms"] == pytest.approx(2.0)
    assert granularity["unfused_reduce_ms"] == pytest.approx(3.0)
    names = {row["kernel_name"]: row for row in granularity["top_unfused_fusion_sensitive_kernels"]}
    assert names["aten::add_elementwise_kernel"]["fusion_family"] == "pointwise"
    assert names["aten::add_elementwise_kernel"]["highlight_unfused"] is True
    assert names["reduceKernelAdd"]["fusion_family"] == "reduce"


def test_collect_profile_tables_emits_fusion_granularity():
    module = load_module("collect_profile_tables", COLLECT_SCRIPT)
    assert module.classify_fusion_family("MLUBroadcastWhereKernel") == "pointwise"
    assert module.classify_fusion_family("TCDP_RING_ALLREDUCE_SIMPLE_BF16_ADD") == "communication"

    strings = {
        1: "triton_poi_fused_add",
        2: "aten::add_elementwise_kernel",
        3: "reduceKernelAdd",
        4: "MLUUnion1BMMEXGEMM",
    }
    compute_rows = [
        (1, 0, 1_000_000),
        (2, 1_000_000, 3_000_000),
        (3, 3_000_000, 6_000_000),
        (4, 6_000_000, 10_000_000),
    ]

    summary = module.triton_fusion_summary(None, strings, 0, 10_000_000, compute_rows, top=10)
    granularity = summary["fusion_granularity"]
    by_family = {row["family"]: row for row in granularity["families"]}

    assert by_family["pointwise"]["unfused_ms"] == pytest.approx(2.0)
    assert by_family["pointwise"]["highlight"] is True
    assert by_family["reduce"]["unfused_ms"] == pytest.approx(3.0)
    assert by_family["library_or_gemm"]["highlight"] is False
    assert summary["top_non_fused"][0]["fusion_family"] == "library_or_gemm"


def test_compare_profile_tables_reports_unfused_pointwise_delta():
    module = load_module("compare_profile_tables", COMPARE_SCRIPT)
    baseline = {
        "label": "A",
        "db": "a.db",
        "range": {"duration_ms": 10.0},
        "torch_compile": {
            "fusion": {
                "fusion_granularity": {
                    "unfused_pointwise_ms": 0.0,
                    "unfused_reduce_ms": 1.0,
                    "families": [{"name": "pointwise", "family": "pointwise", "unfused_ms": 0.0, "total_ms": 1.0, "count": 1}],
                    "top_unfused_fusion_sensitive": [],
                }
            }
        },
    }
    current = {
        "label": "B",
        "db": "b.db",
        "range": {"duration_ms": 10.0},
        "torch_compile": {
            "fusion": {
                "fusion_granularity": {
                    "unfused_pointwise_ms": 5.0,
                    "unfused_reduce_ms": 1.0,
                    "families": [{"name": "pointwise", "family": "pointwise", "unfused_ms": 5.0, "total_ms": 6.0, "count": 3}],
                    "top_unfused_fusion_sensitive": [
                        {"name": "aten::add_elementwise_kernel", "fusion_family": "pointwise", "total_ms": 5.0, "count": 2}
                    ],
                }
            }
        },
    }

    comparison = module.build_comparison(baseline, current, limit=10)
    scalar_rows = {
        row["metric"]: row for row in comparison["torch_compile_delta"]["fusion_scalar"]
    }
    family_rows = {
        row["name"]: row
        for row in comparison["torch_compile_delta"]["fusion_granularity_families"]
    }
    sensitive_rows = comparison["torch_compile_delta"]["unfused_fusion_sensitive_kernels"]

    assert scalar_rows["unfused pointwise ms"]["delta"] == pytest.approx(5.0)
    assert scalar_rows["unfused pointwise ms"]["status"] == "regression"
    assert family_rows["pointwise"]["unfused_ms"]["delta"] == pytest.approx(5.0)
    assert family_rows["pointwise"]["status"] == "regression"
    assert sensitive_rows[0]["name"] == "aten::add_elementwise_kernel"
    assert sensitive_rows[0]["status"] == "regression"
