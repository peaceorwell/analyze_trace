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
KERNEL_CODEGEN_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/kernel_codegen_analysis.py"
PREFLIGHT_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/preflight.py"
VALIDATE_FINDINGS_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/validate_findings.py"
CHECK_REPORT_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/check_report.py"
STEP_WINDOW_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/step_window.py"
QUERY_COMMON_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/query_common.py"
DEVICE_TIMELINE_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/device_timeline.py"
COMPUTE_BREAKDOWN_SCRIPT = ROOT / ".claude/skills/e2e-profiling-analyzer/scripts/compute_breakdown.py"
COLLECT_SCRIPT = ROOT / ".claude/skills/e2e-profiling-comparator/scripts/collect_profile_tables.py"
COMPARE_SCRIPT = ROOT / ".claude/skills/e2e-profiling-comparator/scripts/compare_profile_tables.py"
E2E_ANALYZER_SKILL = ROOT / ".claude/skills/e2e-profiling-analyzer/SKILL.md"
E2E_COMPARATOR_SKILL = ROOT / ".claude/skills/e2e-profiling-comparator/SKILL.md"
LOG_EVIDENCE_SKILL = ROOT / ".claude/skills/log-evidence-analyzer/SKILL.md"
ANALYZER_BRANCH_WORKFLOWS = ROOT / ".claude/skills/e2e-profiling-analyzer/references/branch_workflows.md"
ANALYZER_PERFORMANCE_PLAYBOOK = ROOT / ".claude/skills/e2e-profiling-analyzer/references/pytorch_performance_playbook.md"
ANALYZER_EVIDENCE_CONTRACT = ROOT / ".claude/skills/e2e-profiling-analyzer/references/evidence_contract.md"
ANALYZER_TEAM_WORKFLOW = ROOT / ".claude/skills/e2e-profiling-analyzer/references/team_workflow.md"
ANALYZER_CAPABILITY_DEGRADATION = ROOT / ".claude/skills/e2e-profiling-analyzer/references/capability_degradation.md"
ANALYZER_DISTRIBUTED_CONTEXT = ROOT / ".claude/skills/e2e-profiling-analyzer/references/distributed_context.md"
ANALYZER_HYPOTHESIS_VERIFICATION = ROOT / ".claude/skills/e2e-profiling-analyzer/references/hypothesis_verification.md"
COMPARATOR_PERFORMANCE_PLAYBOOK = ROOT / ".claude/skills/e2e-profiling-comparator/references/pytorch_performance_playbook.md"
COMPARATOR_DB_SCHEMA = ROOT / ".claude/skills/e2e-profiling-comparator/references/db_schema.md"
COMPARATOR_PROFILING_CONCEPTS = ROOT / ".claude/skills/e2e-profiling-comparator/references/profiling_concepts.md"
PROJECT_AGENTS = ROOT / ".claude/agents"
PROJECT_CLAUDE_SETTINGS = ROOT / ".claude/settings.json"


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


def make_codegen_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE string_table(ID INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        """
        CREATE TABLE device_task_kernel_data(
            processId INTEGER, deviceId INTEGER, queueId INTEGER,
            correlationId INTEGER, nameId INTEGER, start INTEGER, end INTEGER,
            isComputation INTEGER, class INTEGER, dimX INTEGER, dimY INTEGER,
            dimZ INTEGER, extra TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE function_data(
            processId INTEGER, threadId INTEGER, correlationId INTEGER,
            nameId INTEGER, start INTEGER, end INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE Internal_operation_range_data(
            processId INTEGER, threadId INTEGER, start INTEGER, end INTEGER,
            extraId INTEGER, nameId INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE Internal_op_range_relations(
            externalCorrelationId INTEGER, correlationId INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO string_table VALUES(?, ?)",
        [
            (1, "triton_poi_fused_relu_add"),
            (2, "native_conv_kernel"),
            (3, "cnInvokeKernel"),
            (4, "Torch-Compiled Region"),
        ],
    )
    conn.executemany(
        "INSERT INTO device_task_kernel_data VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (1, 0, 0, 101, 1, 0, 100_000, 1, 4, 128, 1, 1, "{}"),
            (1, 0, 0, 102, 2, 120_000, 320_000, 1, 4, 256, 1, 1, "{}"),
        ],
    )
    conn.executemany(
        "INSERT INTO function_data VALUES(?,?,?,?,?,?)",
        [(1, 7, 101, 3, -10_000, 0), (1, 7, 102, 3, 110_000, 120_000)],
    )
    conn.execute(
        "INSERT INTO Internal_operation_range_data VALUES(?,?,?,?,?,?)",
        (1, 7, -20_000, 130_000, 1001, 4),
    )
    conn.executemany(
        "INSERT INTO Internal_op_range_relations VALUES(?,?)",
        [(1001, 101), (1001, 102)],
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
def triton_poi_fused_test(
    in_ptr0,
    in_ptr1,
    out_ptr,
    N:tl.constexpr = 256,
    BLOCK:tl.constexpr = 128,
    BLOCK_M:tl.constexpr = 16,
    BLOCK_N:tl.constexpr = 512,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    wrapped = offs % N
    a = tl.load(in_ptr0 + wrapped, mask=offs < N, other=0.0).to(tl.float32)
    b = tl.load(in_ptr1 + wrapped // 2, mask=offs < N, other=0.0).to(tl.float32)
    c = tl.load(in_ptr1 + wrapped + BLOCK, mask=offs + BLOCK < N, other=0.0)
    d = tl.load(in_ptr1 + wrapped + 2 * BLOCK, mask=offs + 2 * BLOCK < N, other=0.0)
    scalar = 0.0
    for i in tl.range(0, 4):
        scalar += tl.load(in_ptr0 + i)
    y = a * tl.sigmoid(a) + tl.exp(b) / (tl.sqrt(a + 1.0) + 1.0) + scalar
    z = tl.sum(y[:, None] + c[:, None], axis=1)
    tl.store(out_ptr + wrapped, z.to(tl.float32), mask=offs < N)
    tl.store(out_ptr + wrapped + BLOCK, d, mask=offs + BLOCK < N)

launch_config = {"num_warps": 2, "num_stages": 1}
""",
        encoding="utf-8",
    )
    second_code_file = code_dir / "triton_output_code_01_triton_poi_fused_second.txt"
    second_code_file.write_text(
        """
import triton
import triton.language as tl

@triton.jit
def triton_poi_fused_second(in_ptr0, out_ptr, N:tl.constexpr = 256, BLOCK:tl.constexpr = 128):
    pid0 = tl.program_id(0)
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr0 + offs, mask=offs < N, other=0.0).to(tl.float32)
    y = x / 3.0
    tl.store(out_ptr + offs, y, mask=offs < N)
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
                            },
                            {
                                "kernel_name": "triton_poi_fused_second",
                                "output_code_file": str(second_code_file),
                                "total_ms": 1.5,
                                "avg_io_efficiency": 200.0,
                                "bandwidth_utilization": 0.1,
                                "improvement_target": 1.35,
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
    assert payload["validation_protocol"]["correctness"]
    assert payload["validation_protocol"]["measurement"]
    assert payload["validation_protocol"]["success_metrics"]
    assert payload["validation_protocol"]["rollback"]
    assert "## Validation Protocol" in markdown
    assert "end-to-end latency or throughput improvement" in markdown
    assert guidance["must_surface"] is True
    assert guidance["promote_to_finding"] is True
    assert "Triton output_code" in guidance["summary_cn"]
    assert guidance["candidates"][0]["kernel_name"] == "triton_poi_fused_test"
    assert guidance["candidates"][0]["estimated_profile"]["io_bytes"] > 0
    assert guidance["candidates"][0]["estimated_profile"]["compute_ops"] > 0
    assert guidance["candidates"][0]["estimated_profile"]["arithmetic_intensity_ops_per_byte"] > 0
    assert guidance["candidates"][0]["estimated_profile"]["roofline_hint"] == "memory_tilted"
    assert guidance["candidates"][0]["evidence_items"]
    assert guidance["candidates"][0]["recommendation_items"]
    assert len(guidance["candidates"]) == 2
    assert guidance["required_section_title"] == "Triton Kernel 代码优化"
    assert guidance["required_table_md"].startswith("## Triton Kernel 代码优化")
    assert "| Kernel | 代码文件 | 耗时 | BW 利用率 | 计算速率估算 | 优化方向与建议 |" in guidance["required_table_md"]
    assert "| Kernel | 代码文件 | 耗时 | BW 利用率 | 主要方向 | 证据 | 建议 |" not in guidance["required_table_md"]
    assert "计算量 " in guidance["required_table_md"]
    assert "GB/s" not in guidance["required_table_md"]
    assert "AI " not in guidance["required_table_md"]
    assert "方向：" in guidance["required_table_md"]
    assert "Triton 101：" in guidance["required_table_md"]
    assert "Helion：" in guidance["required_table_md"]
    assert "Cambricon Triton 101" in guidance["summary_cn"]
    assert "Helion 配置搜索" in guidance["summary_cn"]
    assert "triton_poi_fused_test" in guidance["required_table_md"]
    assert "triton_poi_fused_second" in guidance["required_table_md"]
    assert "• " in guidance["required_table_md"]
    assert any(item.startswith("Triton 101：") for item in guidance["candidates"][0]["experience_items"])
    assert any(item.startswith("Helion：") for item in guidance["candidates"][0]["experience_items"])
    assert kernel["kernel_name"] == "triton_poi_fused_test"
    assert kernel["bandwidth_utilization"] == pytest.approx(0.09)
    assert kernel["num_warps"] == [2]
    assert kernel["num_stages"] == [1]
    assert kernel["loop_count"] >= 1
    assert kernel["tiling_config"]["program_axes"] == [0, 1]
    assert kernel["tiling_config"]["skew_ratio"] >= 8
    assert kernel["tiling_config"]["has_grouping_hint"] is False
    assert "libdevice_math_candidate" in categories
    assert "tensor_division_candidate" in categories
    assert "index_div_mod_or_boundary_fold" in categories
    assert "fragmented_or_pseudo_discrete_io" in categories
    assert "roofline_memory_tilted" in categories
    assert "block_pointer_or_bulk_io_candidate" in categories
    assert "mlu_num_warps_mapping_candidate" in categories
    assert "vectorization_scalar_loop_candidate" in categories
    assert "pipeline_stage_candidate" in categories
    assert "scalar_broadcast_read_candidate" in categories
    assert "autotune_or_meta_parameter_candidate" in categories
    assert "helion_tiling_config_sweep_candidate" in categories
    assert "pid_grouping_or_l2_swizzle_candidate" in categories
    assert "tile_shape_balance_candidate" in categories
    assert "indexing_strategy_sweep_candidate" in categories
    assert "range_config_sweep_candidate" in categories
    assert "reduce_layout_or_tiling_candidate" in categories
    assert "grid_or_retiling_candidate" in categories
    assert {
        "libdevice-opt",
        "div-to-mul",
        "bulk-io-opt",
        "reduce-opt",
        "retiling",
        "modify-grid",
        "roofline",
        "autotune",
        "helion-style-tiling-sweep",
        "pid-grouping",
        "tile-shape-sweep",
        "block-size-balance",
        "indexing-strategy",
        "block-pointer",
        "range-config",
        "num-warps",
        "mlu-task-mapping",
        "vectorize",
        "scalar-read-opt",
        "soft-pipeline",
        "num-stages",
    } <= strategies
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


def test_single_rank_codegen_baseline_emits_triton_and_fusion_signals(tmp_path):
    module = load_module("kernel_codegen_analysis", KERNEL_CODEGEN_SCRIPT)
    db_path = tmp_path / "single_rank.db"
    make_codegen_db(db_path)

    payload = module.analyze_codegen(str(db_path), process_id=1, device_id=0, tiny_threshold_us=150)

    assert payload["totals"]["compute_kernel_count"] == 2
    assert payload["totals"]["triton_signal_count"] == 1
    assert payload["totals"]["fusion_signal_count"] == 1
    assert payload["totals"]["tiny_kernel_count"] == 1
    assert payload["triton_kernels"][0]["name"] == "triton_poi_fused_relu_add"
    assert payload["operator_kernel_mapping"]["coverage_pct"] == pytest.approx(100.0)
    assert payload["operator_kernel_mapping"]["compiled_region_operators"][0]["mapped_kernel_launches"] == 2


def test_preflight_and_findings_contract(tmp_path):
    preflight = load_module("e2e_preflight", PREFLIGHT_SCRIPT)
    validator = load_module("e2e_validate_findings", VALIDATE_FINDINGS_SCRIPT)
    db_path = tmp_path / "single_rank.db"
    make_codegen_db(db_path)

    inspected = preflight.inspect_db(str(db_path))
    assert inspected["compatible"] is True
    payload = {
        "schema_version": "1.0",
        "agent": "e2e-triton-kernel-analyst",
        "scope": {
            "dbs": [str(db_path)],
            "process_ids": [1],
            "device_ids": [0],
            "window": {"start_ns": 0, "end_ns": 320_000, "basis": "stable"},
            "limitations": [],
        },
        "findings": [
            {
                "id": "triton-001",
                "status": "possible",
                "cause": "tiny Triton launch",
                "summary": "One explicitly named Triton kernel is below the selected threshold.",
                "metrics": [
                    {
                        "name": "duration",
                        "value": 0.1,
                        "unit": "ms",
                        "scope": "device=0",
                        "source": "kernel_codegen.json#/triton_kernels/0",
                    }
                ],
                "evidence": ["kernel_codegen.json#/triton_kernels/0"],
                "counter_evidence": ["no matched before/after capture"],
                "affected_scope": ["device=0"],
                "estimated_impact_ms": 0.1,
                "impact": {
                    "observed_cost_ms": 0.1,
                    "critical_path_contribution_ms": None,
                    "recoverable_upper_bound_ms": 0.1,
                    "basis": "single stable device window",
                },
                "audit_disposition": "supported_contributor",
                "confidence": 0.5,
                "overlap_group": "compile-fusion",
                "follow_up": ["repeat with a steady-state window"],
            }
        ],
        "artifacts": ["kernel_codegen.json"],
    }
    assert validator.validate(payload) == []

    payload["findings"][0]["audit_disposition"] = "upheld"
    assert any("audit_disposition" in error for error in validator.validate(payload))

    payload["findings"][0]["audit_disposition"] = "supported_contributor"
    payload["findings"][0]["impact"]["basis"] = ""
    assert any("impact.basis" in error for error in validator.validate(payload))


def test_project_agent_team_layout_is_single_rank():
    expected = {
        "e2e-evidence-builder",
        "e2e-compute-analyst",
        "e2e-triton-kernel-analyst",
        "e2e-compile-fusion-analyst",
        "e2e-gap-host-analyst",
        "e2e-noncompute-analyst",
        "e2e-freeform-analyst",
        "e2e-evidence-auditor",
    }
    actual = {path.stem for path in PROJECT_AGENTS.glob("e2e-*.md")}
    analyzer_text = E2E_ANALYZER_SKILL.read_text(encoding="utf-8")

    assert actual == expected
    assert "e2e-communication-rank-analyst" not in actual
    assert "communication-root-cause" not in analyzer_text
    assert "## 分布式与通信概况" in analyzer_text
    assert "references/distributed_context.md" in analyzer_text
    assert not (E2E_ANALYZER_SKILL.parent / "scripts/comm_breakdown.py").exists()
    assert not (E2E_ANALYZER_SKILL.parent / "scripts/rank_compare.py").exists()
    assert ANALYZER_EVIDENCE_CONTRACT.is_file()
    assert ANALYZER_TEAM_WORKFLOW.is_file()
    settings = json.loads(PROJECT_CLAUDE_SETTINGS.read_text(encoding="utf-8"))
    assert settings["env"]["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"
    for path in PROJECT_AGENTS.glob("e2e-*.md"):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert f"name: {path.stem}\n" in text


def test_analyzer_and_comparator_share_the_same_trace_converter():
    assert TRACE_CONVERTER_SCRIPT.read_bytes() == (
        E2E_COMPARATOR_SKILL.parent / "scripts/torch_trace_to_cnperf_db.py"
    ).read_bytes()


def test_e2e_analyzer_skill_keeps_one_final_report_contract():
    text = E2E_ANALYZER_SKILL.read_text(encoding="utf-8")

    assert text.count("## Final Report Contract") == 1
    assert text.count("The final `report.md` must use this exact high-level structure:") == 1
    assert text.count("Final `report.md` structure:") == 0
    assert text.count("Output contract:\n\n- `结论概览`") == 0
    assert "Report Readability Gate" in text
    assert "Follow only the `Final Report Contract` above" in text


def test_profiling_skills_keep_progressive_disclosure_and_validity_gates():
    analyzer = E2E_ANALYZER_SKILL.read_text(encoding="utf-8")
    comparator = E2E_COMPARATOR_SKILL.read_text(encoding="utf-8")

    assert len(analyzer.splitlines()) < 500
    assert len(comparator.splitlines()) < 500
    assert "references/branch_workflows.md" in analyzer
    assert "references/pytorch_performance_playbook.md" in analyzer
    assert "references/evidence_contract.md" in analyzer
    assert "references/team_workflow.md" in analyzer
    assert "references/capability_degradation.md" in analyzer
    assert "references/distributed_context.md" in analyzer
    assert "references/hypothesis_verification.md" in analyzer
    assert "Comparison Validity Gate" in comparator
    assert "references/pytorch_performance_playbook.md" in comparator
    assert "capability_degradation.md" in comparator
    assert "hypothesis_verification.md" in comparator
    assert ANALYZER_BRANCH_WORKFLOWS.is_file()
    assert ANALYZER_PERFORMANCE_PLAYBOOK.is_file()
    assert ANALYZER_CAPABILITY_DEGRADATION.is_file()
    assert ANALYZER_DISTRIBUTED_CONTEXT.is_file()
    assert ANALYZER_HYPOTHESIS_VERIFICATION.is_file()
    assert COMPARATOR_PERFORMANCE_PLAYBOOK.is_file()


def test_log_evidence_skill_requires_source_backed_non_profiler_report():
    text = LOG_EVIDENCE_SKILL.read_text(encoding="utf-8")

    assert len(text.splitlines()) < 160
    assert "Treat missing profiler capability as a limitation, not a reason to stop" in text
    assert "filename:line" in text
    assert "已观测事实" in text
    assert "合理推断" in text
    assert "无法确认" in text
    assert "Never fabricate profiler metrics" in text
    assert "## 作业与模型上下文" in text
    assert "## 已观测性能信号" in text


GOOD_REPORT = """# AI 性能分析报告

| 项 | 值 |
| --- | --- |
| job | demo |

## 结论概览

### 发现 1：主机下发不足，主计算流空隙偏高

**结论：** 主计算流 gap ratio 0.32，设备未被喂满。

**证据：** `device_timeline.json` 主流 gap ratio 0.32。

**建议：** 评估 cpp_wrapper 与批量下发。

### 发现 2：custom op 包裹大量简单 aten 运算

**结论：** 自定义算子内部存在重复的简单 aten 序列。

**证据：** `compile_segmentation.json` custom_op_simple_aten。

**建议：** 将重复的简单算子下沉到后端 kernel。

### 发现 3：部分 Triton kernel 折算带宽偏低

**结论：** 3 个 kernel 折算带宽低于峰值三成。

**证据：** `triton_kernel_efficiency.json`。

**建议：** 调整 tiling 与访存布局。

## 关键指标

| 指标 | 值 | 来源 | 解释 |
| --- | --- | --- | --- |
| 主流 gap ratio | 0.32 | device_timeline.json | 主机下发不足 |

## 分布式与通信概况

| 项 | 值 | 来源 | 边界 |
| --- | --- | --- | --- |
| rank 拓扑 | 未捕获 | meta_information | 单 rank 采集 |

## 优先行动

1. 评估 cpp_wrapper 下发路径。

## Triton Kernel 代码优化

| Kernel | 时间 | 策略 |
| --- | --- | --- |
| k1 | 1.2ms | libdevice |
| k2 | 0.8ms | tiling |

## 不确定性与下一步

- 缺少 output_code 时无法确认静态收益。

## 产物

- report.md
"""


def _write_report_bundle(tmp_path, report_text, triton_findings=True):
    report = tmp_path / "report.md"
    report.write_text(report_text, encoding="utf-8")
    (tmp_path / "triton_code_optimization.json").write_text(
        json.dumps(
            {
                "has_findings": triton_findings,
                "final_report_guidance": {"candidates": [{"kernel": "k1"}, {"kernel": "k2"}]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "compile_segmentation.json").write_text(
        json.dumps({"custom_op_simple_aten": {"must_report": True}}), encoding="utf-8"
    )
    return report


def test_check_report_accepts_a_contract_compliant_report(tmp_path):
    checker = load_module("check_report", CHECK_REPORT_SCRIPT)
    report = _write_report_bundle(tmp_path, GOOD_REPORT)

    payload = checker.check_report(
        report,
        triton_json=tmp_path / "triton_code_optimization.json",
        compile_json=tmp_path / "compile_segmentation.json",
    )

    assert payload["ok"] is True
    assert payload["problems"] == []
    assert payload["stats"]["findings"] == 3


def test_check_report_rejects_parallel_summary_and_flat_bullets(tmp_path):
    checker = load_module("check_report", CHECK_REPORT_SCRIPT)
    broken = GOOD_REPORT.replace(
        "## 关键指标", "## 主要发现\n\n- 结论：主机下发不足\n\n## 关键指标", 1
    )
    report = _write_report_bundle(tmp_path, broken)

    payload = checker.check_report(
        report,
        triton_json=tmp_path / "triton_code_optimization.json",
        compile_json=tmp_path / "compile_segmentation.json",
    )
    checks = {problem["check"] for problem in payload["problems"]}

    assert payload["ok"] is False
    assert "forbidden_section" in checks
    assert "flat_bullets" in checks


def test_check_report_requires_triton_section_and_all_candidates(tmp_path):
    checker = load_module("check_report", CHECK_REPORT_SCRIPT)
    without_section = GOOD_REPORT.replace(
        "## Triton Kernel 代码优化\n\n| Kernel | 时间 | 策略 |\n| --- | --- | --- |\n| k1 | 1.2ms | libdevice |\n| k2 | 0.8ms | tiling |\n\n",
        "",
        1,
    )
    report = _write_report_bundle(tmp_path, without_section)
    payload = checker.check_report(
        report,
        triton_json=tmp_path / "triton_code_optimization.json",
        compile_json=tmp_path / "compile_segmentation.json",
    )
    assert payload["ok"] is False
    assert any(problem["check"] == "triton_section" for problem in payload["problems"])

    partial = GOOD_REPORT.replace("| k2 | 0.8ms | tiling |\n", "", 1)
    report = _write_report_bundle(tmp_path, partial)
    payload = checker.check_report(
        report,
        triton_json=tmp_path / "triton_code_optimization.json",
        compile_json=tmp_path / "compile_segmentation.json",
    )
    assert payload["ok"] is False
    assert any(problem["check"] == "triton_candidates" for problem in payload["problems"])


def test_check_report_requires_reserved_custom_op_finding(tmp_path):
    checker = load_module("check_report", CHECK_REPORT_SCRIPT)
    without_custom_op = (
        GOOD_REPORT.replace("### 发现 2：custom op 包裹大量简单 aten 运算", "### 发现 2：拷贝暴露偏高", 1)
        .replace("**结论：** 自定义算子内部存在重复的简单 aten 序列。", "**结论：** 拷贝暴露 12ms。", 1)
        .replace("**证据：** `compile_segmentation.json` custom_op_simple_aten。", "**证据：** `gap_summary.json`。", 1)
        .replace("**建议：** 将重复的简单算子下沉到后端 kernel。", "**建议：** 减少同步拷贝。", 1)
    )
    report = _write_report_bundle(tmp_path, without_custom_op)

    payload = checker.check_report(
        report,
        triton_json=tmp_path / "triton_code_optimization.json",
        compile_json=tmp_path / "compile_segmentation.json",
    )

    assert payload["ok"] is False
    assert any(problem["check"] == "custom_op_finding" for problem in payload["problems"])


def test_check_report_warns_on_length_budget_without_failing(tmp_path):
    checker = load_module("check_report", CHECK_REPORT_SCRIPT)
    report = _write_report_bundle(tmp_path, GOOD_REPORT)

    payload = checker.check_report(
        report,
        budget=50,
        triton_json=tmp_path / "triton_code_optimization.json",
        compile_json=tmp_path / "compile_segmentation.json",
    )

    assert payload["ok"] is True
    assert any(
        problem["check"] == "length_budget" and problem["level"] == "warn"
        for problem in payload["problems"]
    )


def test_check_report_skips_findings_gate_for_failure_reports(tmp_path):
    checker = load_module("check_report", CHECK_REPORT_SCRIPT)
    report = tmp_path / "report.md"
    report.write_text("# AI 分析失败\n\n- 错误: 缺少 device_task_kernel_data\n", encoding="utf-8")

    payload = checker.check_report(report)

    assert payload["is_failure_report"] is True
    assert payload["ok"] is True


def test_analyzer_skill_enforces_the_report_gate_with_a_script():
    text = E2E_ANALYZER_SKILL.read_text(encoding="utf-8")

    assert CHECK_REPORT_SCRIPT.is_file()
    assert "scripts/check_report.py" in text
    assert "## Run Checklist" in text
    assert len(text.splitlines()) < 500


def test_long_reference_files_have_a_contents_index():
    long_references = [
        ANALYZER_BRANCH_WORKFLOWS,
        ANALYZER_PERFORMANCE_PLAYBOOK,
        COMPARATOR_PERFORMANCE_PLAYBOOK,
        COMPARATOR_DB_SCHEMA,
        COMPARATOR_PROFILING_CONCEPTS,
    ]
    for path in long_references:
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) > 100
        assert "## Contents" in text, path


def _make_step_db(path, steps=10):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE string_table(ID INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        """
        CREATE TABLE Internal_operation_range_data(
            processId INTEGER, threadId INTEGER, start INTEGER, end INTEGER,
            extraId INTEGER, nameId INTEGER, extra TEXT, type INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE device_task_kernel_data(
            processId INTEGER, deviceId INTEGER, queueId INTEGER, correlationId INTEGER,
            nameId INTEGER, start INTEGER, end INTEGER, isComputation INTEGER, extra TEXT
        )
        """
    )
    names = {}

    def sid(name):
        if name not in names:
            names[name] = len(names) + 1
            conn.execute("INSERT INTO string_table VALUES(?, ?)", (names[name], name))
        return names[name]

    step_ns = 10_000_000
    cursor_ns = 0
    for index in range(steps):
        duration = step_ns
        if index == 0:
            duration = step_ns * 3   # compilation / warmup
        elif index == 1:
            duration = step_ns * 2   # cache warm-up
        elif index == steps - 1:
            duration = int(step_ns * 0.3)  # truncated capture tail
        conn.execute(
            "INSERT INTO Internal_operation_range_data VALUES(?,?,?,?,?,?,?,?)",
            (1, 7, cursor_ns, cursor_ns + duration, index, sid(f"ProfilerStep#{index}"), "{}", 0),
        )
        kernel_ns = cursor_ns + 1_000
        for slot in range(4):
            conn.execute(
                "INSERT INTO device_task_kernel_data VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    1, 0, 1, index * 100 + slot,
                    sid("triton_poi_fused_add_1" if slot % 2 else "gemm_kernel"),
                    kernel_ns, kernel_ns + int(duration * 0.15), 1, "{}",
                ),
            )
            kernel_ns += int(duration * 0.2)
        cursor_ns += duration + 200_000
    conn.commit()
    conn.close()


def test_step_window_trims_warmup_and_truncated_steps(tmp_path):
    db = tmp_path / "steps.db"
    _make_step_db(db)
    step_window = load_module("step_window", STEP_WINDOW_SCRIPT)

    payload = step_window.analyze_db(str(db), step_window.DEFAULT_STEP_REGEX)

    assert payload["source"] == "profiler_step_ranges"
    assert payload["step_count"] == 10
    assert payload["warmup_steps"] == [0, 1]
    assert payload["truncated_steps"] == [9]
    assert payload["steady_window"]["step_count"] == 7
    assert payload["repeatability"]["verdict"] == "pass"
    assert payload["command_hint"].startswith("--start-ns ")


def test_step_window_reports_unavailable_window_instead_of_guessing(tmp_path):
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE string_table(ID INTEGER PRIMARY KEY, string TEXT)")
    conn.commit()
    conn.close()
    step_window = load_module("step_window", STEP_WINDOW_SCRIPT)

    payload = step_window.analyze_db(str(db), step_window.DEFAULT_STEP_REGEX)

    assert payload["steady_window"] is None
    assert payload["repeatability"]["verdict"] == "fail"
    assert payload["limitations"]


def test_window_sql_modes_keep_stats_and_coverage_separate():
    query_common = load_module("query_common", QUERY_COMMON_SCRIPT)

    start_clauses, start_params = query_common.window_sql(10, 20, mode="start")
    overlap_clauses, overlap_params = query_common.window_sql(10, 20, mode="overlap")

    assert start_clauses == ["start >= ?", "start <= ?"]
    assert start_params == [10, 20]
    assert overlap_clauses == ["start < ?", "end > ?"]
    assert overlap_params == [20, 10]
    assert query_common.window_sql(None, None) == ([], [])
    assert query_common.clip_interval(5, 25, 10, 20) == (10, 20)
    assert query_common.clip_interval(0, 5, 10, 20) is None


def test_windowed_scripts_exclude_warmup_from_baseline_metrics(tmp_path):
    db = tmp_path / "steps.db"
    _make_step_db(db)
    step_window = load_module("step_window", STEP_WINDOW_SCRIPT)
    compute_breakdown = load_module("compute_breakdown", COMPUTE_BREAKDOWN_SCRIPT)
    device_timeline = load_module("device_timeline", DEVICE_TIMELINE_SCRIPT)

    window = step_window.analyze_db(str(db), step_window.DEFAULT_STEP_REGEX)["steady_window"]
    full = compute_breakdown.analyze_db(str(db), 10)
    scoped = compute_breakdown.analyze_db(str(db), 10, window["start_ns"], window["end_ns"])

    assert scoped["compute_summary"]["count"] < full["compute_summary"]["count"]
    assert scoped["compute_summary"]["max_ms"] < full["compute_summary"]["max_ms"]

    timeline_full = device_timeline.analyze_db(str(db))
    timeline_scoped = device_timeline.analyze_db(
        str(db), None, None, window["start_ns"], window["end_ns"]
    )
    assert timeline_scoped["window"]["applied"] is True
    assert timeline_scoped["groups"][0]["span_ms"] < timeline_full["groups"][0]["span_ms"]


def test_baseline_scripts_expose_the_shared_window_flags():
    for path in (DEVICE_TIMELINE_SCRIPT, COMPUTE_BREAKDOWN_SCRIPT, CHECK_REPORT_SCRIPT.parent / "gap_summary.py",
                 CHECK_REPORT_SCRIPT.parent / "triton_fusion_coverage.py",
                 CHECK_REPORT_SCRIPT.parent / "triton_kernel_efficiency.py",
                 CHECK_REPORT_SCRIPT.parent / "compile_segmentation.py"):
        text = path.read_text(encoding="utf-8")
        assert "add_window_args(parser)" in text, path


def test_analyzer_skill_requires_a_steady_window_before_measuring():
    text = E2E_ANALYZER_SKILL.read_text(encoding="utf-8")

    assert STEP_WINDOW_SCRIPT.is_file()
    assert "scripts/step_window.py" in text
    assert "--start-ns/--end-ns" in text
    assert "`Steady Window`" in text
    assert len(text.splitlines()) < 500
