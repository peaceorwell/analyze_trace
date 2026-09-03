# Profiling Concepts And Causal Models

Use this reference before interpreting script output, DB queries, rank imbalance, compute gaps, communication exposure, or final root cause. The goal is to give new profiling users a stable mental model before doing causal analysis.

## Contents

- Basic concepts
- Host-device synchronization
- Device-side synchronization
- Communication analysis concepts

## Basic Concepts

These terms define profiler entities and time scopes. They do not imply root cause by themselves.

### Execution Scope

- Host: CPU-side program, framework, runtime, and driver APIs that prepare work and submit it to the device.
- Device: MLU-side execution target where submitted work runs or waits.

### Time Scope

- Host time: CPU-side elapsed time.
- Device time: MLU-side elapsed time.
- E2E time: one training or inference iteration across all participating ranks, governed by the cross-rank critical path.
- Critical path: the chain of host, device, communication, and synchronization work that determines E2E time.

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

## Communication Analysis Concepts

Communication analysis must separate data movement cost from synchronization wait. Exposed communication and communication group labels are symptoms, not final causes.

### How Communication Waits Happen

Communication involves multiple ranks. A rank can enter a communication kernel before its peer, upstream pipeline stage, or collective participants are ready.

Communication time can therefore include:

- Synchronization wait: time spent waiting for another rank, stage, or dependency to arrive.
- Data movement cost: real transfer/reduction work.

The rank showing high exposed communication is often the waiting rank. The rank causing the wait may show low exposed communication because it is busy doing compute or host-side work before reaching the communication point.

Communication groups such as PP, EP, DP, TP, or Global are labels for grouping communication. They are not proof of causality or independence.

Direct communication participation is narrower than E2E dependency. A rank may not appear in the same communication kernel or group, but it can still affect that communication through pipeline progress, step boundary synchronization, backpressure, or another upstream/downstream dependency.

### How To Identify Communication Relationship

Use multiple evidence levels. Do not rely on only one kernel name or one communication group label.

- Direct operation evidence: ranks have matching or aligned communication kernels from the same operation, collective, send/recv pair, or profiler communication group.
- Timeline evidence: one rank's exposed communication interval overlaps another rank's compute, host-blocking, queue dependency, or late progress.
- Boundary evidence: communication occurs near a pipeline boundary, collective boundary, step boundary, or backpressure point where ranks can affect each other indirectly.
- Optional cluster evidence: `communication_statistic.csv` or related cluster files can label PP/EP/DP/TP/Global groups, but labels need timeline validation.

If direct operation evidence is absent, still test timeline and boundary evidence before ruling out a rank.

### Common Situations And Possible Causes

- Single DB or one rank only: cross-rank attribution is not supported. Provide local communication breakdown and ask for other rank DBs or cluster CSVs if the workload is multi-rank.
- High exposed receive/wait on rank A, while rank B has high compute, host-blocking, or late progress: rank A may be waiting for rank B.
- Low exposed communication on rank B does not rule out B as the cause. It may be busy doing useful work while other ranks wait.
- Rank B is not in rank A's communication kernel/group: this only rules out direct participation in that operation. It does not rule out E2E dependency through pipeline or step-level progress.
- Similar high communication exposure on all participating ranks, with no late peer evidence: intrinsic data movement or communication algorithm cost is plausible.
- Few long communication outliers: imbalance, host sync, queue dependency, queue backpressure, or occasional late peer arrival is plausible.
- `100% exposed` all-to-all or receive: the operation was not hidden by compute. It can be intrinsic cost, but it can also be boundary synchronization.
- Similar final device span across ranks: not enough to rule out slow-arriver behavior, because synchronization can align rank completion time.

### Evidence To Prefer

- Cross-rank timeline alignment: waiting interval on one rank versus compute, host-blocking, or queue dependency on another rank.
- Per-rank compute and communication comparison: high communication on one rank plus high compute or late progress on another is waiting evidence.
- E2E dependency check: distinguish direct communication participants from ranks that can indirectly delay the boundary through pipeline, backpressure, or step-level synchronization.
- Operation-level evidence: if no peer-arrival delay is found and all participants show similar long communication, intrinsic communication becomes more plausible.
- Multiple candidates are acceptable. Report plausible causes with confidence and missing evidence instead of forcing one conclusion.
