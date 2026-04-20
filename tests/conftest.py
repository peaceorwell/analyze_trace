import json
import os
import pytest
import tempfile


@pytest.fixture
def sample_trace_data():
    """Minimal PyTorch Profiler trace JSON structure."""
    return {
        "traceEvents": [
            {
                "name": "ProfilerStep#0",
                "cat": "user_annotation",
                "ts": 1000000,
                "dur": 100000,
            },
            {
                "name": "ProfilerStep#1",
                "cat": "user_annotation",
                "ts": 1100000,
                "dur": 100000,
            },
            {
                "name": "triton_matmul_kernel",
                "cat": "kernel",
                "ts": 1000500,
                "dur": 5000,
                "args": {
                    "kernel num(GB)": "10.5",
                    "IO efficiency(GB/s)": "350.2",
                    "triton output code": "...",
                },
            },
            {
                "name": "gemm_cuda_kernel",
                "cat": "kernel",
                "ts": 1002000,
                "dur": 3000,
                "args": {},
            },
            {
                "name": "aten::linear",
                "cat": "kernel",
                "ts": 1003000,
                "dur": 2000,
                "args": {},
            },
            {
                "name": "cnclAllReduce",
                "cat": "gpu_user_annotation",
                "ts": 1004000,
                "dur": 1000,
                "args": {"Collective name": "allreduce"},
            },
            {
                "name": "ProfilerStep#2",
                "cat": "user_annotation",
                "ts": 1200000,
                "dur": 100000,
            },
            {
                "name": "triton_elemwise_kernel",
                "cat": "kernel",
                "ts": 1200500,
                "dur": 4000,
                "args": {},
            },
        ]
    }


@pytest.fixture
def sample_trace_file(sample_trace_data):
    """Create a temporary trace file and clean up after test."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(sample_trace_data, f)
        temp_path = f.name

    yield temp_path

    os.unlink(temp_path)


@pytest.fixture
def sample_trace_file_gz(sample_trace_data):
    """Create a temporary .json.gz trace file."""
    import gzip

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".json.gz", delete=False
    ) as f:
        f.write(json.dumps(sample_trace_data).encode())
        temp_path = f.name

    yield temp_path

    os.unlink(temp_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory and clean up after test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir