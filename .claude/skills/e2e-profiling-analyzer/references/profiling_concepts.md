# Profiling Concepts And Causal Models

Use this reference before interpreting script output, DB queries, Triton/codegen signals, fusion granularity, compute gaps, or final root cause. The goal is to give profiling users a stable mental model before doing causal analysis.

## Contents

- Basic concepts
- Host-device synchronization
- Device-side synchronization
- Triton kernel analysis
- Compilation and fusion granularity
- Single-rank critical-path interpretation
- Evidence and benefit rules

## Basic Concepts

These terms define profiler entities and time scopes. They do not imply root cause by themselves.

### Execution Scope

- Host: CPU-side program, framework, runtime, and driver APIs that prepare work and submit it to the device.
- Device: MLU-side execution target where submitted work runs or waits.

### Time Scope

- Host time: CPU-side elapsed time.
- Device time: MLU-side elapsed time.
- E2E time: one training or inference iteration in the captured single-rank process.
- Critical path: the local chain of host, device, launch, synchronization, and data movement work that determines the captured E2E time.

Host time, device time, and E2E time are related but not interchangeable. For example, host launch duration is not device kernel duration.

### Compute Kernel

Compute kernel is model computation and the default effective device utilization in this skill. Exposed or uncovered time is time not hidden by compute; it is a symptom that needs causal analysis, not a root cause by itself. A long compute kernel can still be the bottleneck if it is on the E2E critical path.

### Communication Kernel

Communication kernel performs send/receive, reduction, or collective-related work. It can represent data movement cost, synchronization wait, or both.

### Ordinary Device Task

Ordinary device task covers non-compute work such as memcpy, memset, or atomic operations. It is only an optimization candidate when it is exposed or blocks useful compute.

### Notifier Task

Notifier task is a device-side synchronization marker or wait. It controls ordering across queues and is usually a mechanism, not the final root cause.

### Queue And Overlap

A device queue is ordered: same-queue tasks are serialized. Different queues may overlap if resources and dependencies allow it, but overlap is not guaranteed.

### Queue Backpressure (Kernel Launch Backpressure)

Queue backpressure occurs when the number of pending (submitted but not yet started) kernels or tasks in a device queue reaches the driver-level maximum capacity. When the queue is full, subsequent host-side launch APIs (e.g., `cnInvokeKernel`) block inside the driver until pending work drains below the limit.

Key properties:

- **Device-side continuity does not rule out backpressure**. The device may be continuously executing kernels while the host is blocked at the driver waiting to submit new ones.
- **Backpressure manifests as long host API self time**. A `cnInvokeKernel` with abnormally large `self` duration (e.g., milliseconds instead of microseconds) is a symptom of driver-level queue blocking.
- **Queue depth estimation**: estimate pending kernel count by correlating host launch times (`function_data` host start) with device start times (`device_task_kernel_data` start). Queue depth = count of kernels where `host_launch < current_time` AND `device_start > current_time`.
- **Plateau detection**: if queue depth stabilizes at a consistent maximum (e.g., ~1024) over a sustained period, the queue has likely hit its capacity limit.
- **Backpressure is a driver/queue mechanism, not a root cause by itself**. Trace why so many kernels were submitted ahead of device execution progress (e.g., host thread burst submission, lack of queue sync pacing, or framework op fusion gaps).

### Launch Latency

Launch latency is the delay between host launch/enqueue completion and device task start. If launch latency is very small, the host likely submitted the task too late, so the device had little queued work ready to run.

## Host-Device Synchronization

Host-device synchronization here mainly means host-side queue sync APIs such as `cnrtQueueSync` or `cnQueueSync`.

These APIs block the host until all tasks previously submitted to the target device queue have completed. While the host is blocked, later kernels may not be launched, which can create host-blocking compute gaps.

Rules:

- A queue sync API explains that host waited for previously submitted queue work to complete. It does not explain which earlier task or dependency made the queue take that long.
- If a compute gap is host-blocking, trace the host stack and related runtime calls before naming the cause.
- Distinguish queue sync blocking from queue backpressure: sync waits for completion, backpressure waits for queue slot availability. Both block the host but have different remedies.

## Device-Side Synchronization

Device-side synchronization controls ordering among device queues and tasks.

Common mechanisms:

- Notifier place: a marker inserted into a queue. It becomes satisfied only after earlier work in that queue reaches the marker.
- Notifier wait: a wait inserted into a queue. Work after it cannot proceed until the matching notifier place is satisfied.
- Same-queue predecessor blocking: any earlier task in the same queue can delay later work, including kernels, communication kernels, memcpy, atomic ops, and notifier waits.

Rules:

- For notifier wait, first check the previous event in the same queue. Only inspect the matching notifier place if the same-queue predecessor does not explain the wait.
- Matching notifier wait/place identifies a cross-queue dependency, not the root cause by itself.
- A notifier place can itself be delayed by earlier work in its own queue.
- Do not stop at the first synchronization primitive name. Trace backward until reaching the task or host operation that consumes the relevant time.

## Triton Kernel Analysis

Treat kernel attribution as evidence with levels:

- Generated Triton source/IR or explicit compiler metadata is strong attribution.
- An observed name containing a stable explicit Triton marker is a probable signal.
- A generic `fused` or codegen-looking name does not prove Triton.

Analyze observed count, total, average, p90, maximum, tiny-launch share, configuration variants, and long tails. Preserve names and metadata before grouping. Kernel names alone cannot establish occupancy, memory bandwidth, register pressure, cache behavior, or generated-code quality; those require counters, artifacts, or controlled experiments.

## Compilation And Fusion Granularity

Compile time and compiled execution time are different scopes. Exclude compilation and warmup before judging steady-state fusion.

Fusion granularity is not equivalent to the number of words in a kernel name. Useful signals include kernels per compiled region or mapped framework operation, tiny-kernel share, repeated launch sequences, launch density, host launch overhead, gaps, and generated graph/IR/code artifacts.

Multiple kernels for one operation can reflect under-fusion, but can also be required by reductions, mutation, aliasing, dynamic shapes, resource limits, or scheduling. A single fused kernel can be slower through recomputation, resource pressure, or reduced parallelism. Prefer matched before/after measurements and bound benefits by exposed critical-path time.

## Single-Rank Critical Path

This workflow explains the captured local rank only. It may report local communication events as timeline categories so they are not misclassified as compute, but it does not infer peer behavior or cross-rank causality.
