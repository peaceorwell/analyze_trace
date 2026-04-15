import argparse
import csv
import json
from collections import defaultdict


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
                info = {
                    "kernel_name": e.get("name"),
                    "ts": ts,
                    "dur": e.get("dur"),
                    "triton_output_code": args["triton output code"],
                    "kernel num(GB)": args["kernel num(GB)"],
                    "IO efficiency(GB/s)": args["IO efficiency(GB/s)"],
                    "kernel kwargs": args["kernel kwargs"],
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
    print(f"{'step':<10} {'kernel_count':<15} {'total_dur(us)':<20}")
    print("-" * 45)
    for step, kernels in sorted(result.items()):
        count = len(kernels)
        total_dur = sum(k["dur"] for k in kernels if k["dur"] is not None)
        print(f"{step:<10} {count:<15} {total_dur:<20}")

    # 2. 每个 step 生成一个 CSV 文件
    fields = ["kernel_name", "ts", "dur", "triton_output_code",
              "kernel num(GB)", "IO efficiency(GB/s)", "kernel kwargs"]
    for step, kernels in sorted(result.items()):
        csv_path = f"{args.output_dir}/step_{step}_kernels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kernels)
        print(f"Wrote {csv_path} ({len(kernels)} rows)")






