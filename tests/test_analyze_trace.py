import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_json.analyze_trace import (
    fmt3,
    pct,
    classify_kernel,
    parse_trace,
    compute_avgs,
    write_avg_csv,
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
    def test_triton_kernel(self):
        result = classify_kernel("triton_matmul", {}, ["gemm"])
        assert result == "triton"

    def test_custom_pattern(self):
        result = classify_kernel("gemm_cuda_kernel", {}, ["gemm", "embedding"])
        assert result == "gemm"

    def test_embedding_pattern(self):
        result = classify_kernel("embedding_lookup", {}, ["embedding"])
        assert result == "embedding"

    def test_collective(self):
        result = classify_kernel(
            "some_kernel", {"Collective name": "allreduce"}, []
        )
        assert result == "collective"

    def test_other(self):
        result = classify_kernel("random_kernel", {}, ["gemm"])
        assert result == "other"

    def test_case_insensitive(self):
        result = classify_kernel("GEMM_CUDA", {}, ["gemm"])
        assert result == "gemm"


class TestParseTrace:
    def test_basic_parsing(self, sample_trace_file):
        result = parse_trace(sample_trace_file, ["gemm", "embedding", "pool"])

        assert "step_to_kernels" in result
        assert "step_to_kernel_types" in result
        assert "step_to_aten" in result
        assert "step_to_cncl" in result
        assert "step_durations" in result

    def test_kernel_classification(self, sample_trace_file):
        result = parse_trace(sample_trace_file, ["gemm", "embedding", "pool"])

        # Step 0 has triton and gemm kernels
        step0_kernels = result["step_to_kernel_types"][0]
        assert "triton" in step0_kernels
        assert "gemm" in step0_kernels

    def test_step_durations(self, sample_trace_file):
        result = parse_trace(sample_trace_file, ["gemm"])

        # Step 0: 100000 microseconds = 100 ms
        assert result["step_durations"][0] == 100.0
        assert result["step_durations"][1] == 100.0
        assert result["step_durations"][2] == 100.0

    def test_missing_kernel_types(self, sample_trace_file):
        # Should not raise an error with empty kernel types
        result = parse_trace(sample_trace_file, [])
        assert "step_to_kernel_types" in result


class TestComputeAvgs:
    def test_compute_avgs(self, sample_trace_file):
        result = parse_trace(sample_trace_file, ["gemm", "embedding", "pool"])
        avgs = compute_avgs(result, ["gemm", "embedding", "pool"])

        assert "KERNEL_TYPES" in avgs
        assert "avg_kernels" in avgs

    def test_empty_input(self):
        from collections import defaultdict

        empty_data = {
            "step_to_kernels": defaultdict(lambda: defaultdict(dict)),
            "step_to_kernel_types": defaultdict(lambda: defaultdict(dict)),
            "step_to_aten": defaultdict(lambda: defaultdict(dict)),
            "step_to_cncl": defaultdict(lambda: defaultdict(dict)),
            "step_durations": {},
            "step_to_triton": defaultdict(list),
        }
        avgs = compute_avgs(empty_data, ["gemm"])
        assert avgs is not None


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


class TestEndToEnd:
    def test_full_analysis(self, sample_trace_file, temp_output_dir):
        """Test full analysis pipeline from parsing to computing averages."""
        result = parse_trace(sample_trace_file, ["gemm", "embedding", "pool"])
        avgs = compute_avgs(result, ["gemm", "embedding", "pool"])

        assert avgs is not None
        assert len(avgs["KERNEL_TYPES"]) > 0