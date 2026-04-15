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
    result = parse_trace("trace.json")

    dump_first_step = True
    dump_to_file = True
    for step, kernels in result.items():
        for k in kernels:
            print("  ", k["kernel_name"])






