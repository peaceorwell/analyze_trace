import argparse
import csv
import json
import os
from collections import defaultdict
from decimal import Decimal


def fmt3(val):
    """Format a number to 3 significant figures without scientific notation."""
    if val is None:
        return ""
    return format(Decimal(f"{float(val):.3g}"), "f")


def parse_trace(trace_file):
    with open(trace_file) as f:
        trace = json.load(f)

    events = trace["traceEvents"]

    # (pid, tid) -> [(step_num, start, end)]
    step_ranges = defaultdict(list)
    # buffer triton kernel events to process after step ranges are sorted
    triton_events = []

    # single pass: collect ProfilerStep ranges and triton kernel events
    for e in events:
        name = e.get("name", "")
        if name.startswith("ProfilerStep"):
            pid = e.get("pid")
            tid = e.get("tid")
            ts = e.get("ts")
            dur = e.get("dur", 0)
            # name like ProfilerStep#3
            step_num = int(name.split("#")[-1])
            step_ranges[(pid, tid)].append((step_num, ts, ts + dur))
        elif e.get("cat") == "kernel" and "triton output code" in e.get("args", {}):
            triton_events.append(e)

    for k in step_ranges:
        step_ranges[k].sort(key=lambda x: x[1])

    # step_num -> list[kernel_info]
    step_to_triton = defaultdict(list)

    for e in triton_events:
        pid = e.get("pid")
        tid = e.get("tid")
        ts = e.get("ts")
        key = (pid, tid)
        if key not in step_ranges:
            continue
        for step_num, start, end in step_ranges[key]:
            if start <= ts <= end:
                eargs = e["args"]
                raw_dur = e.get("dur")
                info = {
                    "kernel_name": e.get("name"),
                    "dur(ms)": raw_dur / 1000 if raw_dur is not None else None,
                    "total io(GB)": float(eargs["kernel num(GB)"]),
                    "IO efficiency(GB/s)": float(eargs["IO efficiency(GB/s)"]),
                    "tiling config": eargs["kernel kwargs"],
                    "triton_output_code": eargs["triton output code"],
                }
                step_to_triton[step_num].append(info)
                break

    return step_to_triton


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a PyTorch profiler trace JSON and extract Triton kernel info per ProfilerStep."
    )
    parser.add_argument(
        "trace_file",
        help="Path to the profiler trace JSON file (e.g. trace.json)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to write CSV files into (default: current directory)",
    )
    args = parser.parse_args()

    result = parse_trace(args.trace_file)
    sorted_steps = sorted(result.items())

    print(f"{'step':<10} {'kernel_count':<15} {'total_dur(ms)':<20}")
    print("-" * 45)
    for step, kernels in sorted_steps:
        count = len(kernels)
        total_dur = sum(k["dur(ms)"] for k in kernels if k["dur(ms)"] is not None)
        print(f"{step:<10} {count:<15} {total_dur:<20.3f}")

    fields = ["kernel_name", "dur(ms)", "total io(GB)", "IO efficiency(GB/s)", "tiling config", "triton_code_file"]
    for step, kernels in sorted_steps:
        csv_path = os.path.join(args.output_dir, f"step_{step}_kernels.csv")
        code_dir = os.path.join(args.output_dir, f"step_{step}_triton_codes")
        os.makedirs(code_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for idx, kernel in enumerate(kernels):
                safe_name = kernel["kernel_name"].replace("/", "_").replace(" ", "_")
                code_filename = f"kernel_{idx}_{safe_name}.py"
                code_path = os.path.join(code_dir, code_filename)
                with open(code_path, "w") as cf:
                    cf.write(kernel["triton_output_code"])
                writer.writerow({
                    "kernel_name": kernel["kernel_name"],
                    "dur(ms)": fmt3(kernel["dur(ms)"]),
                    "total io(GB)": fmt3(kernel["total io(GB)"]),
                    "IO efficiency(GB/s)": fmt3(kernel["IO efficiency(GB/s)"]),
                    "tiling config": kernel["tiling config"].replace("\n", "\\n").replace("\r", ""),
                    "triton_code_file": os.path.join(f"step_{step}_triton_codes", code_filename),
                })
        print(f"Wrote {csv_path} ({len(kernels)} rows)")






