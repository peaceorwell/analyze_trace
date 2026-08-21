# Capability Routing And Graceful Degradation

Use this reference before Phase 1 and before comparing captures. Route by observed content and available evidence, not by filename or an all-or-nothing profiler label.

## Capability ledger

Record every capability as `available`, `partial`, `unavailable`, or `invalid`, with an artifact path and reason:

| Capability | Minimum evidence | What it supports |
|---|---|---|
| `core_db` | readable SQLite core tables plus a local `string_table` | device task, kernel, copy, gap, and host-range analysis |
| `stable_window` | explicit repeated steps/requests after initialization and warmup | steady-state latency and throughput claims |
| `workload_identity` | shapes/batch/tokens/dtype/mode/topology as applicable | workload equivalence and normalization |
| `host_device_correlation` | correlation IDs, relations, notifier or queue identity | causal host/gap attribution |
| `compile_regions` | compiled-region, graph-break, cache or recompilation annotations | compile/fusion segmentation |
| `triton_codegen` | explicit compiler metadata or generated source | strong Triton attribution and static code analysis |
| `triton_io` | comparable IO-efficiency metadata and peak-bandwidth basis | bandwidth-utilization analysis |
| `raw_timeline` | gzip/plain JSON content with a top-level `traceEvents` array | bounded event-window confirmation |
| `precomputed_overlap` | explicit Overlap Analysis annotations in the raw trace | raw-trace compute/communication/non-overlap/idle signals |

The filename is only a hint. Confirm SQLite by opening it read-only. Confirm compressed JSON by gzip magic and confirm a trace from its top-level `traceEvents`; do not treat an arbitrary JSON document as a timeline. Do not recursively combine unrelated capture batches merely because they share a parent directory.

## Routing

1. Run preflight on each native DB. Exclude an invalid DB, but continue with other valid inputs.
2. Convert each raw timeline independently, verify the generated DB, and keep the conversion report and original trace as immutable evidence.
3. Use core DB evidence for the common baseline. Enable optional branches only when their capability is available or partial with a stated boundary.
4. If an optional table, annotation, code artifact, or branch fails, mark only that capability/branch unavailable. Do not discard valid baseline evidence.
5. If conversion cannot produce a valid DB, write a bounded raw-trace capability report before failing. When explicit precomputed overlap annotations exist, they may support a limited raw-trace breakdown; otherwise report only inventory and capture-quality facts. Never manufacture DB-only metrics.

For a bounded raw-trace check, require a known process/device or rank and a step/request or absolute time window. Stream or incrementally parse the selected trace, aggregate names/counts/durations, cap output size, and record truncation. Never `json.load()` an unbounded trace merely to search for a candidate.

## Metric boundaries during degradation

- Missing evidence is `unknown`/`unavailable`, never zero.
- `Device-Gap` or `Device Idle` is device inactivity without a proven owner. It is not Host time, Memcpy time, or a root cause.
- A device-stream gap ratio is a host-feeding signal, not sufficient proof of the responsible host operation. Follow queue/correlation evidence before naming a cause.
- Total communication time and uncovered/non-overlapped communication are not additive. Only exposed communication can bound local E2E benefit.
- Per-name duration sums and per-family interval unions may overlap across streams. Do not add them unless the categories are proven mutually exclusive on the same time basis.
- A FLOPS, bandwidth, shape, or dtype field that cannot be derived is unavailable. A displayed zero caused by missing metadata is not a measured zero.
- A truncated scan can confirm observed matches but cannot prove that an unobserved event is absent.

## Reporting

Include a compact capability ledger in the baseline artifact. The final report should mention only capability gaps that change confidence or the next action. When analysis degrades, state what remains trustworthy, what was lost, and the smallest recapture or artifact needed to restore that capability.
