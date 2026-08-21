# Distributed And Communication Context

Use this reference to extract a compact distributed/communication summary for baseline and final reports. This is descriptive context, not a communication-performance branch.

## Evidence Sources

Use only fields present in the input or generated artifacts, in this order:

1. User/input metadata and `basic_info.py`: global/local rank, world/group size, node/device count, backend/library, and process-group membership.
2. `device_timeline.json`: local `comm_total_ms`, `comm_uncovered_ms`, `comm_uncovered_pct`, and `top_uncovered_comm` names.
3. Kernel `extra`, `meta_information`, and CNPX communication ranges when available: collective name, process-group ranks, group size, and other explicit topology annotations.

Preserve the selected process, device, and time window beside every local timing value. If a field is absent, report `未捕获` or `未知`; never infer it from a filename, process ID, device ID, or the number of DB files.

## Report Content

Write a compact table under `Distributed Context` in Phase 1 and `## 分布式与通信概况` in the final report. Include available rows from:

- topology: rank/local rank, world/group size, node/device count, backend or communication library;
- local communication exposure: total communication event time, uncovered communication time/ratio, and analysis window;
- observed collectives: top local communication kernel/collective names and their measured total/uncovered time;
- evidence boundary: source artifact and missing peer/rank coverage.

When no distributed metadata or communication events exist, keep the section and state that the capture contains no usable distributed/communication evidence.

## Interpretation Boundaries

- Do not create a communication specialist, communication root-cause branch, cross-rank comparison, or communication optimization action by default.
- A local communication event can contain transfer, synchronization, or peer wait. Its duration does not identify which component dominates.
- Total communication event time and uncovered communication time are different views and are not additive. Only uncovered local time can bound local E2E exposure.
- Do not infer bandwidth, collective algorithm, link utilization, straggler rank, or peer behavior without matched multi-rank evidence and the required counters/metadata.
- Keep the section descriptive. Do not consume one of the prioritized bottleneck findings merely because communication events are present.
