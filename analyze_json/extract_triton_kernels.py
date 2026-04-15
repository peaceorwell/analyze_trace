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

    # step_num -> list[kernel_info]
    step_to_triton = defaultdict(list)

    # ------------------------
    # pass1: collect ProfilerStep
    # ------------------------
    for e in events:
        name = e.get("name", "")
        if name.startswith("ProfilerStep"):
            pid = e.get("pid")
            tid = e.get("tid")
            ts = e.get("ts")
            dur = e.get("dur", 0)

            end = ts + dur

            # name like ProfilerStep#3
            step_num = int(name.split("#")[-1])

            step_ranges[(pid, tid)].append((step_num, ts, end))

    for k in step_ranges:
        step_ranges[k].sort(key=lambda x: x[1])

    # ------------------------
    # pass2: collect triton kernels
    # ------------------------
    for e in events:
        if e.get("cat") != "kernel":
            continue

        args = e.get("args", {})
        if "triton output code" not in args:
            continue

        pid = e.get("pid")
        tid = e.get("tid")
        ts = e.get("ts")

        key = (pid, tid)
        if key not in step_ranges:
            continue

        for step_num, start, end in step_ranges[key]:
            if start <= ts <= end:
                raw_dur = e.get("dur")
                info = {
                    "kernel_name": e.get("name"),
                    "dur(ms)": raw_dur / 1000 if raw_dur is not None else None,
                    "total io(GB)": float(f"{float(args['kernel num(GB)']):.3g}"),
                    "IO efficiency(GB/s)": float(f"{float(args['IO efficiency(GB/s)']):.3g}"),
                    "tiling config": args["kernel kwargs"],
                    "triton_output_code": args["triton output code"],
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

    # 1. 统计每个 step 的 kernel 数量和 dur 总和
    print(f"{'step':<10} {'kernel_count':<15} {'total_dur(ms)':<20}")
    print("-" * 45)
    for step, kernels in sorted(result.items()):
        count = len(kernels)
        total_dur = sum(k["dur(ms)"] for k in kernels if k["dur(ms)"] is not None)
        print(f"{step:<10} {count:<15} {total_dur:<20.3f}")

    # 2. 每个 step 生成一个 CSV 文件，triton_output_code 单独存文件
    fields = ["kernel_name", "dur(ms)", "total io(GB)", "IO efficiency(GB/s)", "tiling config", "triton_code_file"]
    for step, kernels in sorted(result.items()):
        csv_path = f"{args.output_dir}/step_{step}_kernels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            code_dir = os.path.join(args.output_dir, f"step_{step}_triton_codes")
            os.makedirs(code_dir, exist_ok=True)
            for idx, kernel in enumerate(kernels):
                safe_name = kernel["kernel_name"].replace("/", "_").replace(" ", "_")
                code_filename = f"kernel_{idx}_{safe_name}.py"
                code_path = os.path.join(code_dir, code_filename)
                with open(code_path, "w") as cf:
                    cf.write(kernel["triton_output_code"])

                row = dict(kernel)
                row["dur(ms)"] = fmt3(kernel["dur(ms)"])
                row["total io(GB)"] = fmt3(kernel["total io(GB)"])
                row["IO efficiency(GB/s)"] = fmt3(kernel["IO efficiency(GB/s)"])
                row["tiling config"] = kernel["tiling config"].replace("\n", "\\n").replace("\r", "")
                row["triton_code_file"] = os.path.join(f"step_{step}_triton_codes", code_filename)
                writer.writerow(row)
        print(f"Wrote {csv_path} ({len(kernels)} rows)")






