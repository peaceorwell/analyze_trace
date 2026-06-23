import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/triton_fusion_coverage.py"
TRITON_EFFICIENCY_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/triton_kernel_efficiency.py"
TRITON_CODE_OPT_SCRIPT = ROOT / ".claude/skills/mlu-triton-optimize/scripts/analyze_triton_code.py"
TRACE_CONVERTER_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/torch_trace_to_cnperf_db.py"
COMPILE_SEGMENTATION_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/compile_segmentation.py"
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


def make_custom_op_simple_aten_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE string_table(ID INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        """
        CREATE TABLE Internal_operation_range_data(
            processId INTEGER,
            threadId INTEGER,
            start INTEGER,
            end INTEGER,
            nameId INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE device_task_kernel_data(
            processId INTEGER,
            deviceId INTEGER,
            queueId INTEGER,
            correlationId INTEGER,
            nameId INTEGER,
            start INTEGER,
            end INTEGER,
            isComputation INTEGER
        )
        """
    )
    strings = [
        (1, "lego_fastop::mlu_xmm_fwd"),
        (2, "aten::mul"),
        (3, "aten::slice"),
        (4, "aten::empty_like"),
        (5, "aten::select"),
        (6, "aten::zero_"),
        (7, "aten::to"),
        (8, "MLUUnion1BMMEXGEMM"),
    ]
    conn.executemany("INSERT INTO string_table(ID, string) VALUES(?, ?)", strings)
    conn.execute(
        "INSERT INTO Internal_operation_range_data VALUES(?, ?, ?, ?, ?)",
        (1, 1, 0, 1_000_000, 1),
    )
    for idx, name_id in enumerate([2, 3, 4, 5, 6, 7] * 2):
        start = 80_000 + idx * 70_000
        conn.execute(
            "INSERT INTO Internal_operation_range_data VALUES(?, ?, ?, ?, ?)",
            (1, 1, start, start + 30_000, name_id),
        )
    conn.execute(
        """
        INSERT INTO device_task_kernel_data(
            processId, deviceId, queueId, correlationId, nameId, start, end, isComputation
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (1, 0, 0, 1, 8, 0, 500_000, 1),
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


def test_trace_converter_preserves_top_level_triton_efficiency_metadata(tmp_path):
    module = load_module("torch_trace_to_cnperf_db", TRACE_CONVERTER_SCRIPT)
    db_path = tmp_path / "trace.db"
    converter = module.Converter(str(db_path))
    try:
        converter.process_event(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "triton_poi_fused_add",
                "ts": 10,
                "dur": 5,
                "args": {
                    "extra": {"dimx": 1, "dimy": 1, "kernel_type": "BLOCK"},
                    "correlation": 7,
                    "stream": 1,
                    "device": 0,
                    "IO efficiency(GB/s)": "350.2 GB/s",
                    "kernel num(GB)": "10.5",
                    "triton output code": "def kernel(): pass",
                },
            }
        )
        converter.conn.commit()
        extra = converter.cur.execute("SELECT extra FROM device_task_kernel_data").fetchone()[0]
    finally:
        converter.close()

    payload = json.loads(extra)

    assert payload["dimx"] == 1
    assert payload["IO efficiency(GB/s)"] == "350.2 GB/s"
    assert payload["io_efficiency"] == "350.2 GB/s"
    assert payload["kernel_num_gb"] == "10.5"
    assert payload["output_code"] == "def kernel(): pass"


def test_trace_converter_records_cpp_wrapper_signal_from_kernel_file(tmp_path):
    module = load_module("torch_trace_to_cnperf_db_cpp_wrapper", TRACE_CONVERTER_SCRIPT)
    db_path = tmp_path / "trace.db"
    converter = module.Converter(str(db_path))
    try:
        converter.process_event(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "triton_poi_fused_add",
                "pid": 1,
                "tid": 2,
                "ts": 10,
                "dur": 5,
                "args": {
                    "kernel_backend": "triton",
                    "kernel_file": "/tmp/.inductor_cache/ab/cabc.py",
                },
            }
        )
        converter.finish({}, "trace.json.gz", "simdjson", None)
        row = converter.cur.execute(
            "SELECT value FROM meta_information WHERE type = 'torch_compile_cpp_wrapper'"
        ).fetchone()
    finally:
        converter.close()

    payload = json.loads(row[0])

    assert payload["state"] == "off"
    assert payload["source"] == "kernel_file_extension"
    assert payload["confidence"] == "medium"
    assert payload["kernel_file_extensions"][".py"] == 1


def test_compile_segmentation_reads_cpp_wrapper_signal_from_converted_db(tmp_path):
    converter_module = load_module("torch_trace_to_cnperf_db_compile", TRACE_CONVERTER_SCRIPT)
    segmentation_module = load_module("compile_segmentation_cpp_wrapper", COMPILE_SEGMENTATION_SCRIPT)
    db_path = tmp_path / "trace.db"
    converter = converter_module.Converter(str(db_path))
    try:
        converter.process_event(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "triton_poi_fused_add",
                "pid": 1,
                "tid": 2,
                "ts": 10,
                "dur": 5,
                "args": {"kernel_file": "/tmp/.inductor_cache/ab/cabc.py"},
            }
        )
        converter.finish({}, "trace.json.gz", "simdjson", None)
    finally:
        converter.close()

    payload = segmentation_module.analyze_db(str(db_path))

    assert payload["cpp_wrapper_signal"]["state"] == "off"
    assert payload["host_launch_overhead"]["cpp_wrapper_signal"]["state"] == "off"
    assert "Trace evidence indicates cpp_wrapper is disabled" in payload["host_launch_overhead"]["note"]


def test_compile_segmentation_highlights_custom_op_with_simple_aten(tmp_path):
    module = load_module("compile_segmentation_custom_simple_aten", COMPILE_SEGMENTATION_SCRIPT)
    db_path = tmp_path / "trace.db"
    make_custom_op_simple_aten_db(db_path)

    payload = module.analyze_db(str(db_path))
    summary = payload["custom_op_simple_aten"]
    row = summary["highlighted_custom_ops"][0]

    assert summary["has_issue"] is True
    assert summary["must_report"] is True
    assert summary["top_issue"]["custom_op_name"] == "lego_fastop::mlu_xmm_fwd"
    assert row["custom_op_name"] == "lego_fastop::mlu_xmm_fwd"
    assert row["nested_simple_aten_count"] == 12
    assert row["avg_simple_aten_per_call"] == pytest.approx(12.0)
    assert row["report_priority"] == "high"
    assert row["must_report"] is True
    assert {item["name"] for item in row["top_simple_aten_ops"]} >= {"aten::mul", "aten::slice"}


def test_triton_efficiency_script_reads_raw_trace_efficiency_keys(tmp_path):
    module = load_module("triton_kernel_efficiency", TRITON_EFFICIENCY_SCRIPT)
    db_path = tmp_path / "trace.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE string_table(ID INTEGER PRIMARY KEY, string TEXT)")
    conn.execute("CREATE TABLE device_information(name TEXT)")
    conn.execute(
        """
        CREATE TABLE device_task_kernel_data(
            nameId INTEGER,
            start INTEGER,
            end INTEGER,
            isComputation INTEGER,
            extra TEXT
        )
        """
    )
    conn.execute("INSERT INTO string_table(ID, string) VALUES(1, 'triton_poi_fused_add')")
    conn.execute("INSERT INTO device_information(name) VALUES('MLU590-M9DK')")
    conn.execute(
        "INSERT INTO device_task_kernel_data VALUES(?, ?, ?, ?, ?)",
        (
            1,
            0,
            10_000_000,
            1,
            json.dumps(
                {
                    "IO efficiency(GB/s)": "350.2 GB/s",
                    "kernel num(GB)": "10.5",
                    "triton output code": "def kernel(): pass",
                }
            ),
        ),
    )
    conn.commit()
    conn.close()

    payload = module.analyze_db(str(db_path), top=10, dump_dir=str(tmp_path / "code"))
    kernels = payload["top_low_bandwidth_kernels"]

    assert payload["has_io_metadata"] is True
    assert "IO efficiency(GB/s)" in payload["observed_metadata_keys"]
    assert kernels[0]["avg_io_efficiency"] == pytest.approx(350.2)
    assert kernels[0]["bandwidth_utilization"] == pytest.approx(350.2 / 2000)
    assert Path(kernels[0]["output_code_file"]).exists()


def test_mlu_triton_code_optimizer_flags_static_candidates(tmp_path):
    module = load_module("mlu_triton_code_opt", TRITON_CODE_OPT_SCRIPT)
    code_dir = tmp_path / "triton_output_code"
    code_dir.mkdir()
    code_file = code_dir / "triton_output_code_00_triton_poi_fused_test.txt"
    code_file.write_text(
        """
import triton
import triton.language as tl

@triton.jit
def triton_poi_fused_test(in_ptr0, in_ptr1, out_ptr, N:tl.constexpr, BLOCK:tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    wrapped = offs % N
    a = tl.load(in_ptr0 + wrapped, mask=offs < N, other=0.0).to(tl.float32)
    b = tl.load(in_ptr1 + wrapped // 2, mask=offs < N, other=0.0).to(tl.float32)
    c = tl.load(in_ptr1 + wrapped + BLOCK, mask=offs + BLOCK < N, other=0.0)
    d = tl.load(in_ptr1 + wrapped + 2 * BLOCK, mask=offs + 2 * BLOCK < N, other=0.0)
    y = a * tl.sigmoid(a) + tl.exp(b) / (tl.sqrt(a + 1.0) + 1.0)
    z = tl.sum(y[:, None] + c[:, None], axis=1)
    tl.store(out_ptr + wrapped, z.to(tl.float32), mask=offs < N)
    tl.store(out_ptr + wrapped + BLOCK, d, mask=offs + BLOCK < N)
""",
        encoding="utf-8",
    )
    efficiency_json = tmp_path / "triton_kernel_efficiency.json"
    efficiency_json.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "label": "trace",
                        "top_low_bandwidth_kernels": [
                            {
                                "kernel_name": "triton_poi_fused_test",
                                "output_code_file": str(code_file),
                                "total_ms": 12.0,
                                "avg_io_efficiency": 180.0,
                                "bandwidth_utilization": 0.09,
                                "improvement_target": 10.9,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = module.analyze(str(code_dir), str(efficiency_json), top=10)
    kernel = payload["kernels"][0]
    categories = {finding["category"] for finding in kernel["findings"]}
    strategies = {strategy for finding in kernel["findings"] for strategy in finding["strategy"].split(" / ")}
    guidance = payload["final_report_guidance"]
    markdown = module.render_markdown(payload)

    assert payload["has_findings"] is True
    assert guidance["must_surface"] is True
    assert guidance["promote_to_finding"] is True
    assert "Triton output_code" in guidance["summary_cn"]
    assert guidance["candidates"][0]["kernel_name"] == "triton_poi_fused_test"
    assert guidance["required_section_title"] == "Triton Kernel 代码优化候选"
    assert "| Kernel | 代码文件 | 耗时 | BW 利用率 |" in guidance["required_table_md"]
    assert "triton_poi_fused_test" in guidance["required_table_md"]
    assert kernel["kernel_name"] == "triton_poi_fused_test"
    assert kernel["bandwidth_utilization"] == pytest.approx(0.09)
    assert "libdevice_math_candidate" in categories
    assert "tensor_division_candidate" in categories
    assert "index_div_mod_or_boundary_fold" in categories
    assert "fragmented_or_pseudo_discrete_io" in categories
    assert "reduce_layout_or_tiling_candidate" in categories
    assert "grid_or_retiling_candidate" in categories
    assert {"libdevice-opt", "div-to-mul", "bulk-io-opt", "reduce-opt", "retiling", "modify-grid"} <= strategies
    assert "Triton Code Optimization Candidates" in markdown
    assert "Final report placement" in markdown
    assert "Required Final Report Snippet" in markdown
    assert "triton_poi_fused_test" in markdown


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


def test_collect_profile_tables_reads_raw_trace_efficiency_keys(tmp_path):
    module = load_module("collect_profile_tables", COLLECT_SCRIPT)
    db_path = tmp_path / "trace.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE device_task_kernel_data(
            nameId INTEGER,
            start INTEGER,
            end INTEGER,
            isComputation INTEGER,
            extra TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO device_task_kernel_data VALUES(?, ?, ?, ?, ?)",
        (
            1,
            0,
            10_000_000,
            1,
            json.dumps({"IO efficiency(GB/s)": "350.2 GB/s"}),
        ),
    )

    summary = module.triton_kernel_efficiency_summary(
        conn.cursor(), {1: "triton_poi_fused_add"}, 0, 20_000_000, peak_bandwidth=2000, top=10
    )
    conn.close()

    assert summary["has_io_metadata"] is True
    assert summary["observed_keys"] == ["IO efficiency(GB/s)"]
    assert summary["kernels"][0]["avg_io_efficiency"] == pytest.approx(350.2)
    assert summary["kernels"][0]["bandwidth_utilization"] == pytest.approx(350.2 / 2000)


def test_collect_profile_tables_compile_summary_keeps_cpp_wrapper_signal(tmp_path):
    module = load_module("collect_profile_tables_cpp_wrapper", COLLECT_SCRIPT)
    db_path = tmp_path / "trace.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE meta_information(type TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO meta_information VALUES(?, ?)",
        (
            "torch_compile_cpp_wrapper",
            json.dumps(
                {
                    "state": "off",
                    "source": "kernel_file_extension",
                    "confidence": "medium",
                    "kernel_file_extensions": {".py": 3},
                }
            ),
        ),
    )
    conn.commit()

    summary = module.compile_segmentation_summary(
        conn.cursor(),
        {},
        0,
        10_000_000,
        [],
        {"main_stream_gap_pct": 12.5},
    )
    conn.close()

    signal = summary["cpp_wrapper_signal"]
    assert signal["state"] == "off"
    assert signal["source"] == "kernel_file_extension"
    assert summary["host_launch_overhead"]["cpp_wrapper_signal"]["kernel_file_extensions"][".py"] == 3


def test_collect_profile_tables_compile_summary_highlights_custom_op_simple_aten(tmp_path):
    module = load_module("collect_profile_tables_custom_simple_aten", COLLECT_SCRIPT)
    db_path = tmp_path / "trace.db"
    make_custom_op_simple_aten_db(db_path)
    conn = sqlite3.connect(db_path)
    strings = module.load_strings(conn.cursor())

    summary = module.compile_segmentation_summary(
        conn.cursor(),
        strings,
        0,
        1_000_000,
        [],
        {"main_stream_gap_pct": 0.0},
    )
    conn.close()
    custom_summary = summary["custom_op_simple_aten"]
    row = custom_summary["highlighted_custom_ops"][0]

    assert custom_summary["has_issue"] is True
    assert custom_summary["must_report"] is True
    assert row["custom_op_name"] == "lego_fastop::mlu_xmm_fwd"
    assert row["nested_simple_aten_count"] == 12
    assert row["avg_simple_aten_per_call"] == pytest.approx(12.0)
    assert row["report_priority"] == "high"
    assert row["must_report"] is True


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
            },
            "segmentation": {
                "custom_op_simple_aten": {
                    "highlighted_custom_ops": []
                }
            },
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
            },
            "segmentation": {
                "custom_op_simple_aten": {
                    "highlighted_custom_ops": [
                        {
                            "name": "lego_fastop::mlu_xmm_fwd",
                            "custom_op_name": "lego_fastop::mlu_xmm_fwd",
                            "range_count": 16,
                            "nested_simple_aten_count": 304,
                            "nested_simple_aten_ms": 1.03,
                            "avg_simple_aten_per_call": 19.0,
                            "unique_simple_aten_ops": 8,
                            "report_priority": "high",
                            "must_report": True,
                        }
                    ]
                }
            },
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
    custom_rows = comparison["torch_compile_delta"]["custom_op_simple_aten"]
    assert custom_rows[0]["name"] == "lego_fastop::mlu_xmm_fwd"
    assert custom_rows[0]["nested_simple_aten_count"]["delta"] == pytest.approx(304)
    assert custom_rows[0]["report_priority_B"] == "high"
    assert custom_rows[0]["must_report"] is True
    assert custom_rows[0]["status"] == "regression"
