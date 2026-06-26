import csv
import gzip
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trace_analyzer import (
    fmt3,
    pct,
    classify_kernel,
    safe_float,
    parse_trace,
    parse_step_filter,
    filter_parsed_steps,
    compute_avgs,
    write_avg_csv,
    write_single,
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

    def test_gzip_trace_streams_without_json_load(self, sample_trace_file_gz, monkeypatch):
        def fail_json_load(*args, **kwargs):
            raise AssertionError("json.load should not be used for trace parsing")

        monkeypatch.setenv("TRACE_FAST_TRACE_JSON_BYTES", "0")
        monkeypatch.setattr(json, "load", fail_json_load)

        result = parse_trace(sample_trace_file_gz)

        assert result["step_durations"][0] == 100.0

    def test_small_plain_trace_uses_fast_path(self, sample_trace_file, monkeypatch):
        import trace_analyzer.core as core

        monkeypatch.setenv("TRACE_FAST_TRACE_JSON_BYTES", "999999")

        def fail_iter_trace_events(_trace_file):
            raise AssertionError("streaming parser should not be used for small traces")

        monkeypatch.setattr(core, "_iter_trace_events", fail_iter_trace_events)

        result = parse_trace(sample_trace_file)

        assert result["step_durations"][0] == 100.0

    def test_gzip_fast_path_falls_back_when_decompressed_size_exceeds_limit(self, tmp_path, monkeypatch):
        trace_path = tmp_path / "trace.json.gz"
        payload = (
            b'{"traceEvents":['
            + b" " * 512
            + b'{"name":"ProfilerStep#0","cat":"user_annotation","ts":0,"dur":1000},'
            + b'{"name":"gemm_kernel","cat":"kernel","ts":100,"dur":200,"args":{}}'
            + b"]}"
        )
        with gzip.open(trace_path, "wb") as f:
            f.write(payload)

        monkeypatch.setenv("TRACE_FAST_TRACE_JSON_BYTES", "128")

        result = parse_trace(trace_path)

        assert result["step_durations"][0] == 1.0
        assert result["step_to_kernels"][0]["gemm_kernel"]["count"] == 1

    def test_streaming_trace_reads_event_stream_once(self, sample_trace_file_gz, sample_trace_data, monkeypatch):
        import trace_analyzer.core as core

        monkeypatch.setenv("TRACE_FAST_TRACE_JSON_BYTES", "0")
        events = sample_trace_data["traceEvents"]

        calls = 0

        def fake_iter_trace_events(_trace_file):
            nonlocal calls
            calls += 1
            yield from events

        monkeypatch.setattr(core, "_iter_trace_events", fake_iter_trace_events)

        result = parse_trace(sample_trace_file_gz)

        assert result["step_durations"][0] == 100.0
        assert calls == 1

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
    def test_parse_step_filter(self):
        assert parse_step_filter("0, 2-4，ProfilerStep#6;step_8") == (0, 2, 3, 4, 6, 8)

    def test_filter_parsed_steps_limits_averages(self, sample_trace_file):
        parsed = parse_trace(sample_trace_file)
        filtered = filter_parsed_steps(parsed, "2")
        avgs = compute_avgs(filtered)

        assert avgs["all_steps"] == [2]
        assert "triton_elemwise_kernel" in avgs["avg_kernels"]
        assert "triton_matmul_kernel" not in avgs["avg_kernels"]

    def test_filter_parsed_steps_reports_missing_steps(self, sample_trace_file):
        parsed = parse_trace(sample_trace_file)

        with pytest.raises(ValueError, match="Selected step"):
            filter_parsed_steps(parsed, "99")

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

    def test_non_triton_kernel_efficiency_counters_are_collected(self, tmp_path):
        matmul_name = "void MLUFusedMatMulGemmU1Ex<half>(...)"
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {"name": matmul_name, "cat": "kernel", "ts": 1100, "dur": 100, "tid": 7, "args": {}},
                {"name": "Compute Efficiency(%)", "ph": "C", "ts": 1100, "tid": 7, "args": {"utils": 40.0}},
                {"name": "IO Efficiency(%)", "ph": "C", "ts": 1100, "tid": 7, "args": {"utils": 18.0}},
                {"name": "OP Efficiency(%)", "ph": "C", "ts": 1100, "tid": 7, "args": {"utils": 34.0}},
                {"name": "Compute Efficiency(%)", "ph": "C", "ts": 1200, "tid": 7, "args": {"utils": 0.0}},
                {"name": "IO Efficiency(%)", "ph": "C", "ts": 1200, "tid": 7, "args": {"utils": 0.0}},
                {"name": "OP Efficiency(%)", "ph": "C", "ts": 1200, "tid": 7, "args": {"utils": 0.0}},
                {
                    "name": "triton_poi_fused_add",
                    "cat": "kernel",
                    "ts": 1300,
                    "dur": 100,
                    "tid": 8,
                    "args": {"IO efficiency(GB/s)": "100 GB/s"},
                },
                {"name": "Compute Efficiency(%)", "ph": "C", "ts": 1300, "tid": 8, "args": {"utils": 99.0}},
            ],
        }))

        avgs = compute_avgs(parse_trace(str(trace_path)))

        row = avgs["avg_non_triton_kernel_efficiency"][matmul_name]
        assert row["family"] == "gemm"
        assert row["avg_compute_efficiency"] == 40.0
        assert row["avg_io_efficiency"] == 18.0
        assert row["avg_op_efficiency"] == 34.0
        assert "triton_poi_fused_add" not in avgs["avg_non_triton_kernel_efficiency"]

    def test_triton_metadata_keeps_stable_match_fingerprints(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {
                    "name": "triton_poi_fused_add_46",
                    "cat": "kernel",
                    "ts": 1100,
                    "dur": 100,
                    "args": {
                        "kernel kwargs": "M=128, N=256",
                        "triton output code": "def kernel():\n    return 1\n",
                    },
                },
            ],
        }))

        avgs = compute_avgs(parse_trace(str(trace_path)))
        triton = avgs["avg_triton"]["triton_poi_fused_add_46"]

        assert triton["triton_normalized_name"] == "triton_poi_fused_add"
        assert triton["triton_code_hash"]
        assert triton["triton_code_hashes"] == [triton["triton_code_hash"]]
        assert triton["triton_code_signature_hash"]
        assert triton["triton_code_signature_hashes"] == [triton["triton_code_signature_hash"]]
        assert triton["triton_tiling_hash"]
        assert triton["triton_tiling_hashes"] == [triton["triton_tiling_hash"]]

    def test_parse_trace_drops_triton_code_by_default_but_keeps_hashes(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {
                    "name": "triton_poi_fused_add_46",
                    "cat": "kernel",
                    "ts": 1100,
                    "dur": 100,
                    "args": {"triton output code": "def kernel():\n    return 1\n"},
                },
            ],
        }))

        parsed = parse_trace(str(trace_path))
        kernel = parsed["step_to_triton"][0][0]
        avgs = compute_avgs(parsed)

        assert kernel["triton_output_code"] is None
        assert kernel["triton_code_hash"]
        assert avgs["avg_triton"]["triton_poi_fused_add_46"]["triton_code_hash"]

    def test_parse_trace_can_keep_triton_code_for_exports(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        code = "def kernel():\n    return 1\n"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {
                    "name": "triton_poi_fused_add_46",
                    "cat": "kernel",
                    "ts": 1100,
                    "dur": 100,
                    "args": {"triton output code": code},
                },
            ],
        }))

        parsed = parse_trace(str(trace_path), keep_triton_code=True)

        assert parsed["step_to_triton"][0][0]["triton_output_code"] == code

    def test_parse_trace_reports_progress_counts(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {"name": "gemm_cuda_kernel", "cat": "kernel", "ts": 1100, "dur": 100, "args": {}},
            ],
        }))
        progress = []

        parse_trace(str(trace_path), progress_callback=lambda events, records: progress.append((events, records)))

        assert progress[-1] == (2, 1)

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

    def test_comparison_writes_kernel_type_cmp_without_delta_tab_csv(self, temp_output_dir):
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

        with open(os.path.join(temp_output_dir, "kernel_types_cmp.csv")) as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["type"] == "attention"
        assert rows[0]["delta_dur_ms"] == "-30"
        assert rows[1]["type"] == "gemm"
        assert rows[1]["delta_dur_ms"] == "8"

        cmp_fields = rows[0].keys()

        assert "dur_pct_A" not in cmp_fields
        assert "dur_pct_B" not in cmp_fields
        assert not os.path.exists(os.path.join(temp_output_dir, "kernel_types_delta.csv"))

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

    def test_triton_compare_matches_different_suffixes_by_code_hash(self, temp_output_dir):
        data_a = {
            "KERNEL_TYPES": [],
            "kt_avgs": {"other": (0, 0), "collective": (0, 0)},
            "avg_kernels": {},
            "kernel_families": {},
            "avg_triton": {
                "triton_poi_fused_add_46": {
                    "avg_count": 1,
                    "avg_dur_ms": 10,
                    "avg_io_gb": 1,
                    "avg_io_eff": 100,
                    "triton_code_hash": "abc123",
                    "triton_normalized_name": "triton_poi_fused_add",
                },
            },
            "avg_aten": {},
            "avg_cncl": {},
        }
        data_b = {
            "KERNEL_TYPES": [],
            "kt_avgs": {"other": (0, 0), "collective": (0, 0)},
            "avg_kernels": {},
            "kernel_families": {},
            "avg_triton": {
                "triton_poi_fused_add_55": {
                    "avg_count": 1,
                    "avg_dur_ms": 13,
                    "avg_io_gb": 1,
                    "avg_io_eff": 100,
                    "triton_code_hash": "abc123",
                    "triton_normalized_name": "triton_poi_fused_add",
                },
            },
            "avg_aten": {},
            "avg_cncl": {},
        }
        args = type("Args", (), {"output_dir": temp_output_dir})

        write_comparison(data_a, data_b, args)

        with open(os.path.join(temp_output_dir, "triton_kernels_cmp.csv")) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["kernel_name"] == "triton_poi_fused_add"
        assert rows[0]["kernel_name_A"] == "triton_poi_fused_add_46"
        assert rows[0]["kernel_name_B"] == "triton_poi_fused_add_55"
        assert rows[0]["match_method"] == "code_hash"
        assert rows[0]["delta_dur_ms"] == "3"

    def test_triton_compare_matches_by_code_hash_intersection(self, temp_output_dir):
        data_a = {
            "KERNEL_TYPES": [],
            "kt_avgs": {"other": (0, 0), "collective": (0, 0)},
            "avg_kernels": {},
            "kernel_families": {},
            "avg_triton": {
                "triton_poi_fused_add_46": {
                    "avg_count": 1,
                    "avg_dur_ms": 10,
                    "avg_io_gb": 1,
                    "avg_io_eff": 100,
                    "triton_code_hashes": ["a_only", "shared"],
                    "triton_normalized_name": "triton_poi_fused_add",
                },
            },
            "avg_aten": {},
            "avg_cncl": {},
        }
        data_b = {
            "KERNEL_TYPES": [],
            "kt_avgs": {"other": (0, 0), "collective": (0, 0)},
            "avg_kernels": {},
            "kernel_families": {},
            "avg_triton": {
                "triton_poi_fused_add_55": {
                    "avg_count": 1,
                    "avg_dur_ms": 13,
                    "avg_io_gb": 1,
                    "avg_io_eff": 100,
                    "triton_code_hashes": ["b_only", "shared"],
                    "triton_normalized_name": "triton_poi_fused_add",
                },
            },
            "avg_aten": {},
            "avg_cncl": {},
        }
        args = type("Args", (), {"output_dir": temp_output_dir})

        write_comparison(data_a, data_b, args)

        with open(os.path.join(temp_output_dir, "triton_kernels_cmp.csv")) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["match_method"] == "code_hash"
        assert rows[0]["kernel_name_A"] == "triton_poi_fused_add_46"
        assert rows[0]["kernel_name_B"] == "triton_poi_fused_add_55"

    def test_triton_compare_matches_by_code_signature(self, tmp_path, temp_output_dir):
        trace_a = tmp_path / "trace_a.json"
        trace_b = tmp_path / "trace_b.json"
        trace_a.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {
                    "name": "triton_poi_fused_add_46",
                    "cat": "kernel",
                    "ts": 1100,
                    "dur": 10000,
                    "args": {
                        "triton output code": "# generated for A\n"
                                              "def triton_poi_fused_add_46(x):\n"
                                              "    return x + 1\n",
                    },
                },
            ],
        }))
        trace_b.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {
                    "name": "triton_poi_fused_add_55",
                    "cat": "kernel",
                    "ts": 1100,
                    "dur": 13000,
                    "args": {
                        "triton output code": "# generated for B\n"
                                              "def triton_poi_fused_add_55(x):\n"
                                              "    return x + 1\n",
                    },
                },
            ],
        }))
        args = type("Args", (), {"output_dir": temp_output_dir})

        write_comparison(compute_avgs(parse_trace(str(trace_a))), compute_avgs(parse_trace(str(trace_b))), args)

        with open(os.path.join(temp_output_dir, "triton_kernels_cmp.csv")) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["match_method"] == "code_signature"
        assert rows[0]["kernel_name"] == "triton_poi_fused_add"
        assert rows[0]["delta_dur_ms"] == "3"

    def test_triton_compare_matches_orderless_tiling_kwargs(self, tmp_path, temp_output_dir):
        trace_a = tmp_path / "trace_a.json"
        trace_b = tmp_path / "trace_b.json"
        trace_a.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {
                    "name": "triton_poi_fused_add_46",
                    "cat": "kernel",
                    "ts": 1100,
                    "dur": 10000,
                    "args": {"kernel kwargs": "M=128, N=256"},
                },
            ],
        }))
        trace_b.write_text(json.dumps({
            "traceEvents": [
                {"name": "ProfilerStep#0", "cat": "user_annotation", "ts": 1000, "dur": 1000},
                {
                    "name": "triton_poi_fused_add_55",
                    "cat": "kernel",
                    "ts": 1100,
                    "dur": 13000,
                    "args": {"kernel kwargs": "N=256,M=128"},
                },
            ],
        }))
        args = type("Args", (), {"output_dir": temp_output_dir})

        write_comparison(compute_avgs(parse_trace(str(trace_a))), compute_avgs(parse_trace(str(trace_b))), args)

        with open(os.path.join(temp_output_dir, "triton_kernels_cmp.csv")) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["match_method"] == "normalized_name_tiling"
        assert rows[0]["kernel_name"] == "triton_poi_fused_add"

    def test_single_csv_outputs_do_not_include_percentage_columns(self, temp_output_dir):
        data = {
            "all_steps": [],
            "step_to_triton": {},
            "avg_kernels": {
                "gemm_kernel": {"avg_count": 10, "avg_dur_ms": 20},
            },
            "kernel_families": {"gemm_kernel": "gemm"},
            "avg_triton": {
                "triton_poi_kernel": {
                    "avg_count": 5,
                    "avg_dur_ms": 3,
                    "avg_io_gb": 1,
                    "avg_io_eff": 9,
                },
            },
            "avg_non_triton_kernel_efficiency": {
                "matmul_kernel": {
                    "family": "gemm",
                    "avg_count": 4,
                    "avg_dur_ms": 8,
                    "avg_compute_efficiency": 40,
                    "avg_io_efficiency": 18,
                    "avg_op_efficiency": 34,
                    "avg_io_efficiency_gbps": None,
                },
            },
            "avg_aten": {
                "aten::mm": {"avg_count": 2, "avg_dur_ms": 4},
            },
            "avg_cncl": {},
            "KERNEL_TYPES": ["gemm"],
            "kt_avgs": {"gemm": (10, 20), "other": (0, 0), "collective": (0, 0)},
        }
        args = type("Args", (), {
            "output_dir": temp_output_dir,
            "save_triton_csv": False,
            "save_triton_code": False,
        })

        write_single(data, args)

        for name in [
            "all_kernels_avg.csv",
            "triton_kernels_avg.csv",
            "non_triton_kernel_efficiency_avg.csv",
            "aten_ops_avg.csv",
            "kernel_types_avg.csv",
            "cncl_ops_avg.csv",
        ]:
            with open(os.path.join(temp_output_dir, name)) as f:
                fields = next(csv.reader(f))
            assert not any("pct" in field.lower() or "percent" in field.lower() for field in fields)
