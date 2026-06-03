import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trace_analyzer import (
    fmt3,
    pct,
    classify_kernel,
    safe_float,
    parse_trace,
    compute_avgs,
    write_avg_csv,
    write_comparison,
    write_triton_code_file,
)


class TestFmt3:
    def test_positive_number(self):
        assert fmt3(123.456) == "123"

    def test_small_number(self):
        assert fmt3(0.001234) == "0.00123"

    def test_large_number(self):
        # fmt3 uses Decimal which doesn't use scientific notation for large numbers
        assert fmt3(1234567) == "1230000"

    def test_none(self):
        assert fmt3(None) == ""

    def test_zero(self):
        assert fmt3(0) == "0"


class TestPct:
    def test_increase(self):
        assert pct(100, 120) == "+20.0%"

    def test_decrease(self):
        assert pct(100, 80) == "-20.0%"

    def test_no_change(self):
        assert pct(100, 100) == "+0.0%"

    def test_zero_a(self):
        assert pct(0, 100) == ""


class TestClassifyKernel:
    def test_triton_kernel_family(self):
        result = classify_kernel("triton_poi_fused_add")
        assert result == "triton_pointwise"

    def test_auto_family(self):
        result = classify_kernel("gemm_cuda_kernel")
        assert result == "gemm"

    def test_collective(self):
        result = classify_kernel(
            "some_kernel", {"Collective name": "allreduce"}
        )
        assert result == "collective"

    def test_fallback_family(self):
        result = classify_kernel("random_kernel")
        assert result == "random_kernel"

    def test_case_insensitive(self):
        result = classify_kernel("GEMM_CUDA")
        assert result == "gemm"


class TestSafeFloat:
    def test_plain_number(self):
        assert safe_float("350.2") == 350.2

    def test_number_with_units(self):
        assert safe_float("350.2 GB/s") == 350.2

    def test_malformed_value(self):
        assert safe_float("n/a") is None


class TestParseTrace:
    def test_basic_parsing(self, sample_trace_file):
        result = parse_trace(sample_trace_file)

        assert "step_to_kernels" in result
        assert "step_to_aten" in result
        assert "step_to_cncl" in result
        assert "step_durations" in result

    def test_step_durations(self, sample_trace_file):
        result = parse_trace(sample_trace_file)

        # Step 0: 100000 microseconds = 100 ms
        assert result["step_durations"][0] == 100.0
        assert result["step_durations"][1] == 100.0
        assert result["step_durations"][2] == 100.0

    def test_parse_without_kernel_type_patterns(self, sample_trace_file):
        result = parse_trace(sample_trace_file)
        assert result["step_ranges"][0] == (1000000, 1100000)

    def test_gzip_trace(self, sample_trace_file_gz):
        result = parse_trace(sample_trace_file_gz)
        assert result["step_durations"][0] == 100.0

    def test_tar_gzip_trace(self, sample_trace_file_tar_gz):
        result = parse_trace(sample_trace_file_tar_gz)
        assert result["step_durations"][0] == 100.0

    def test_tgz_trace(self, sample_trace_file_tar_gz, tmp_path):
        tgz_path = tmp_path / "trace.tgz"
        shutil.copyfile(sample_trace_file_tar_gz, tgz_path)

        result = parse_trace(str(tgz_path))

        assert result["step_durations"][0] == 100.0

    def test_zip_trace(self, sample_trace_file_zip):
        result = parse_trace(sample_trace_file_zip)
        assert result["step_durations"][0] == 100.0

    def test_step_underscore_markers(self, tmp_path):
        trace_path = tmp_path / "step_underscore.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "step_0", "cat": "python_function", "ts": 1000, "dur": 1000},
                {"name": "step_0", "cat": "gpu_user_annotation", "ts": 1100, "dur": 1400},
                {"name": "step_1", "cat": "python_function", "ts": 3000, "dur": 1000},
                {"name": "triton_poi_fused_add", "cat": "kernel", "ts": 1200, "dur": 100, "args": {}},
                {"name": "gemm_cuda_kernel", "cat": "kernel", "ts": 3200, "dur": 200, "args": {}},
                {"name": "aten::linear", "cat": "cpu_op", "ts": 1250, "dur": 50, "args": {}},
            ],
        }))

        result = parse_trace(str(trace_path))

        assert result["step_ranges"][0] == (1000, 2500)
        assert result["step_durations"][0] == 1.5
        assert result["step_to_kernels"][0]["triton_poi_fused_add"]["count"] == 1
        assert result["step_to_kernels"][1]["gemm_cuda_kernel"]["count"] == 1
        assert result["step_to_aten"][0]["aten::linear"]["count"] == 1

    def test_run_step_fallback_without_profiler_markers(self, tmp_path):
        trace_path = tmp_path / "run_step_fallback.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "model.py(10): run_step, callsite: 20", "cat": "python_function", "ts": 1000, "dur": 500},
                {"name": "triton_poi_fused_add", "cat": "kernel", "ts": 1100, "dur": 100, "args": {}},
                {"name": "aten::linear", "cat": "cpu_op", "ts": 1200, "dur": 50, "args": {}},
                {"name": "gemm_cuda_kernel", "cat": "kernel", "ts": 1550, "dur": 100, "args": {}},
            ],
        }))

        result = parse_trace(str(trace_path))

        assert result["step_ranges"][0] == (1000.0, 1650.0)
        assert result["step_durations"][0] == 0.65
        assert result["step_to_kernels"][0]["triton_poi_fused_add"]["count"] == 1
        assert result["step_to_kernels"][0]["gemm_cuda_kernel"]["count"] == 1
        assert result["step_to_aten"][0]["aten::linear"]["count"] == 1

    def test_analyzable_range_fallback_without_any_step_markers(self, tmp_path):
        trace_path = tmp_path / "no_step_markers.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "triton_poi_fused_add", "cat": "kernel", "ts": 2000, "dur": 100, "args": {}},
                {"name": "aten::linear", "cat": "cpu_op", "ts": 1800, "dur": 50, "args": {}},
            ],
        }))

        result = parse_trace(str(trace_path))

        assert result["step_ranges"][0] == (1800.0, 2100.0)
        assert result["step_to_kernels"][0]["triton_poi_fused_add"]["count"] == 1
        assert result["step_to_aten"][0]["aten::linear"]["count"] == 1


class TestComputeAvgs:
    def test_compute_avgs(self, sample_trace_file):
        result = parse_trace(sample_trace_file)
        avgs = compute_avgs(result)

        assert "KERNEL_TYPES" in avgs
        assert "avg_kernels" in avgs

    def test_empty_input(self):
        from collections import defaultdict

        empty_data = {
            "step_to_kernels": defaultdict(lambda: defaultdict(dict)),
            "step_to_aten": defaultdict(lambda: defaultdict(dict)),
            "step_to_cncl": defaultdict(lambda: defaultdict(dict)),
            "step_durations": {},
            "step_to_triton": defaultdict(list),
        }
        avgs = compute_avgs(empty_data)
        assert avgs is not None

    def test_triton_io_efficiency_is_call_average(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {
                    "name": "triton_poi_fused_add",
                    "cat": "kernel",
                    "ts": 1100,
                    "dur": 100,
                    "args": {"IO efficiency(GB/s)": "100 GB/s", "kernel num(GB)": "1GB"},
                },
                {
                    "name": "triton_poi_fused_add",
                    "cat": "kernel",
                    "ts": 1200,
                    "dur": 200,
                    "args": {"IO efficiency(GB/s)": "300 GB/s", "kernel num(GB)": "2GB"},
                },
            ],
        }))

        avgs = compute_avgs(parse_trace(str(trace_path)))

        assert avgs["avg_triton"]["triton_poi_fused_add"]["avg_io_eff"] == 200.0

    def test_malformed_triton_args_do_not_fail_analysis(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {
                    "name": "triton_poi_fused_add",
                    "cat": "kernel",
                    "ts": 1100,
                    "dur": 100,
                    "args": {"IO efficiency(GB/s)": "n/a", "kernel num(GB)": ""},
                },
            ],
        }))

        avgs = compute_avgs(parse_trace(str(trace_path)))

        assert avgs["avg_triton"]["triton_poi_fused_add"]["avg_io_eff"] is None

    def test_collective_kernel_is_excluded_from_compute_duration(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 100000},
                {"name": "gemm_cuda_kernel", "cat": "kernel", "ts": 1100, "dur": 30000, "args": {}},
                {"name": "ncclAllReduceKernel", "cat": "kernel", "ts": 50000, "dur": 20000, "args": {}},
            ],
        }))

        avgs = compute_avgs(parse_trace(str(trace_path)))
        _, kernel_count, compute_kernel_dur, _, _, _, _, collective_count, collective_dur = avgs["step_stats"][0]

        assert kernel_count == 2
        assert compute_kernel_dur == 30.0
        assert collective_count == 1
        assert collective_dur == 20.0
        assert "collective" not in avgs["KERNEL_TYPES"]
        assert avgs["kt_avgs"]["collective"] == (1.0, 20.0)


class TestWriteAvgCsv:
    def test_write_csv(self, temp_output_dir):
        data = {
            "kernel_a": {"avg_count": 10.0, "avg_dur_ms": 5.5},
            "kernel_b": {"avg_count": 20.0, "avg_dur_ms": 3.2},
        }
        output_path = os.path.join(temp_output_dir, "test_output.csv")

        write_avg_csv(output_path, data, "kernel_name")

        assert os.path.exists(output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["kernel_name"] == "kernel_a"
        assert rows[0]["avg_count"] == "10"
        assert rows[1]["kernel_name"] == "kernel_b"


class TestWriteTritonCodeFile:
    def test_long_kernel_name_is_truncated(self, temp_output_dir):
        kernel = {
            "kernel_name": "triton_tem_fused_" + ("very_long_segment_" * 40),
            "triton_output_code": "print('ok')\n",
        }

        filename = write_triton_code_file(temp_output_dir, 204, kernel)

        assert len(filename.encode("utf-8")) <= 240
        assert filename.startswith("kernel_204_triton_tem_fused_")
        assert filename.endswith(".py")
        assert os.path.exists(os.path.join(temp_output_dir, filename))


class TestEndToEnd:
    def test_full_analysis(self, sample_trace_file, temp_output_dir):
        """Test full analysis pipeline from parsing to computing averages."""
        result = parse_trace(sample_trace_file)
        avgs = compute_avgs(result)

        assert avgs is not None
        assert len(avgs["KERNEL_TYPES"]) > 0

    def test_comparison_writes_kernel_type_delta_csv(self, temp_output_dir):
        data_a = {
            "KERNEL_TYPES": ["gemm", "attention", "other"],
            "kt_avgs": {"gemm": (10, 20), "attention": (5, 40), "other": (1, 3), "collective": (0, 0)},
            "avg_kernels": {},
            "avg_triton": {},
            "avg_aten": {},
            "avg_cncl": {},
        }
        data_b = {
            "KERNEL_TYPES": ["gemm", "attention", "other"],
            "kt_avgs": {"gemm": (10, 28), "attention": (5, 10), "other": (1, 4), "collective": (0, 0)},
            "avg_kernels": {},
            "avg_triton": {},
            "avg_aten": {},
            "avg_cncl": {},
        }
        args = type("Args", (), {"output_dir": temp_output_dir})

        write_comparison(data_a, data_b, args)

        with open(os.path.join(temp_output_dir, "kernel_types_delta.csv")) as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["type"] == "attention"
        assert rows[0]["delta_dur_ms"] == "-30"
        assert rows[1]["type"] == "gemm"
        assert rows[1]["delta_dur_ms"] == "8"

        with open(os.path.join(temp_output_dir, "kernel_types_cmp.csv")) as f:
            cmp_fields = csv.DictReader(f).fieldnames
        delta_fields = rows[0].keys()

        assert "dur_pct_A" not in cmp_fields
        assert "dur_pct_B" not in cmp_fields
        assert "dur_pct_A" not in delta_fields
        assert "dur_pct_B" not in delta_fields

    def test_comparison_all_kernels_cmp_includes_family(self, temp_output_dir):
        data_a = {
            "KERNEL_TYPES": ["gemm"],
            "kt_avgs": {"gemm": (1, 2), "other": (0, 0), "collective": (0, 0)},
            "avg_kernels": {"gemm_kernel_a": {"avg_count": 1, "avg_dur_ms": 2}},
            "kernel_families": {"gemm_kernel_a": "gemm"},
            "avg_triton": {},
            "avg_aten": {},
            "avg_cncl": {},
        }
        data_b = {
            "KERNEL_TYPES": ["gemm"],
            "kt_avgs": {"gemm": (1, 5), "other": (0, 0), "collective": (0, 0)},
            "avg_kernels": {"gemm_kernel_a": {"avg_count": 1, "avg_dur_ms": 5}},
            "kernel_families": {"gemm_kernel_a": "gemm"},
            "avg_triton": {},
            "avg_aten": {},
            "avg_cncl": {},
        }
        args = type("Args", (), {"output_dir": temp_output_dir})

        write_comparison(data_a, data_b, args)

        with open(os.path.join(temp_output_dir, "all_kernels_cmp.csv")) as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["kernel_name"] == "gemm_kernel_a"
        assert rows[0]["family"] == "gemm"
        assert rows[0]["delta_dur_ms"] == "3"
