# cnperf DB Schema Notes

Use this reference when a task needs direct SQLite queries or when script output is insufficient.

## Core Tables

- `string_table`
  - Fields: `ID`, `string`
  - Use: decode `nameId` values.
  - Rule: load per DB. Do not reuse one DB's `nameId` mapping for another DB.

- `device_task_kernel_data`
  - Common fields: `processId`, `deviceId`, `queueId`, `correlationId`, `nameId`, `start`, `end`, `isComputation`, `extra`
  - Use: device kernel timeline and kernel summaries.
  - Rule: `isComputation=1` marks compute kernels. Other kernel rows may be communication or non-compute work.

- `device_task_notifier_data`
  - Common fields: `processId`, `deviceId`, `queueId`, `correlationId`, `notifierId`, `type`, `start`, `end`, `extra`
  - Use: notifier wait/place dependency tracing.
  - Rule: for notifier wait, first analyze same-queue predecessor. Only then inspect matched notifier place if needed.
  - `extra` is JSON. Parse `json_extract(extra, '$.unique_val')` or equivalent JSON parsing to get `unique_val`.
  - Wait/place matching:
    - wait rows use `type = 0`.
    - place rows use `type = 1`.
    - A wait matches a place when `processId`, `deviceId`, `notifierId`, and `extra.unique_val` are equal.
    - `queueId` is not part of the notifier identity; wait and place usually occur on different queues.
    - When multiple matching places exist, choose the place relevant to the wait by time, normally the latest matching place at or before the wait start.
  - Dependency rule:
    - For a notifier wait, first inspect the previous event on the same queue. If that predecessor is close enough to the wait, classify the wait as blocked by the predecessor.
    - Only inspect the matched place if same-queue predecessor analysis does not explain the wait, for example no predecessor exists, the predecessor is outside the gap boundary, or the predecessor-to-wait gap is large.

- `device_task_memcpy_data`, `device_task_atomic_operation_data`, `device_task_memset_data`
  - Use: support events that can block later compute.

- `function_data`
  - Common fields: `correlationId`, `nameId`, `processId`, `threadId`, `start`, `end`, `self`, `extra`
  - Use: host runtime/API calls such as `cnInvokeKernel`, `cnrtQueueSync`, `cnMemcpyDtoH`, `cnclAllReduce`.
  - Rule: connect host and device by `correlationId`.

- `Internal_operation_range_data`
  - Common fields: `processId`, `threadId`, `start`, `end`, `extraId`, `nameId`, `extra`, `type`
  - Use: host framework/operator ranges such as model layers, framework ops, dataloader, or `__next__`.
  - Rule: ranges are nested and overlap totals can double count. Use them as context, not exclusive cost.

- `Internal_op_range_relations`
  - Fields: `externalCorrelationId`, `correlationId`
  - Use: map framework op `extraId` to runtime/API `function_data.correlationId`.
  - Example: a framework op can map to synchronization APIs such as `cnrtQueueSync`; memcpy APIs may appear as supporting context but are not required.

- `cnpx_data`, `INTERNAL_CNPX_FRAMEWORK_RANGES_V1`, `INTERNAL_CNPX_OP_RANGES_V1`, `INTERNAL_CNPX_COMM_RANGES_V1`
  - Use: optional host context and communication annotations.

- `Internal_python_function_trace_data`
  - Use: optional Python context when present.
  - Rule: often absent or sparse; do not depend on it.

## Key Semantics

- `correlationId` is the main linkage between device events and host APIs.
- `queueId` is process-local, not globally unique across DBs.
- `threadId` matters. Do not merge independent host threads into one call tree.
- Device timestamps and host op timestamps can be compared in these profiles, but always verify ranges first with `basic_info.py`.
- Multi-DB analysis must keep each DB's string map and process/device identity separate.

## torch.compile / triton Metadata (converted torch traces)

These sources support the `compile-segmentation`, `triton-fusion-coverage`, and `triton-kernel-efficiency` branches. They are mainly populated in DBs converted from torch profiler traces; native cnperf DBs may or may not carry them.

- Compiled-region ranges: torch.compile emits host ranges such as `Torch-Compiled Region`, `CompiledFunction`, `CompiledFunctionBackward`, `TorchDynamo Cache Lookup`, and `inductor`. They land in `Internal_operation_range_data`; decode `nameId` via `string_table`. Work outside any compiled region is eager/graph-break execution.
- Triton kernel names: inductor-fused device kernels appear in `device_task_kernel_data` with observed names like `triton_poi_fused_*`, `triton_red_fused_*`, `triton_per_fused_*`, `triton_tem_fused_*`. Non-`triton` compute kernels (`isComputation=1`) are library/eager/vendor kernels. Match the actual `string_table` names rather than assuming the full prefix set.
- Per-kernel metadata: the converter stores each kernel event's `args`/`extra` as JSON in `device_task_kernel_data.extra`. Optional inductor/profiler enrichment such as `output_code`, `io_efficiency` (or `io_eff` / `memory_efficiency`), `achieved_bandwidth` / `bandwidth`, and `bytes` appears here when present. Parse with `json_extract(extra, '$.output_code')` etc. Report the observed keys first; these fields are frequently absent.
- `io_efficiency` is a bandwidth-equivalent value (the kernel's effective/folded bandwidth), NOT a normalized 0–1 ratio or percentage. Judge memory-IO efficiency by comparing it against the device **theoretical (peak) bandwidth**; verify units match before dividing; never compute `1 - io_efficiency`. The theoretical bandwidth is taken from the MLU model: **MLU590 → 2000, MLU580 → 1200** (same unit as `io_efficiency`, i.e. GB/s). `triton_kernel_efficiency.py` uses this model-based value, falling back to `meta_information` `deviceInfo.m_dev_basic_info.max_bandwidth` only when the model is unknown.

Example — classify compute kernels into fused vs non-fused by observed name:

```sql
SELECT s.string AS kernel_name,
       COUNT(*) AS count,
       SUM(k.end - k.start) AS total_us,
       AVG(k.end - k.start) AS avg_us,
       MAX(k.end - k.start) AS max_us
FROM device_task_kernel_data k
JOIN string_table s ON s.ID = k.nameId
WHERE k.isComputation = 1
  AND (:process_id IS NULL OR k.processId = :process_id)
GROUP BY k.nameId
ORDER BY total_us DESC;
```

Example — pull IO-efficiency metadata for triton kernels when present:

```sql
SELECT s.string AS kernel_name,
       json_extract(k.extra, '$.io_efficiency') AS io_efficiency,
       json_extract(k.extra, '$.achieved_bandwidth') AS achieved_bandwidth,
       json_extract(k.extra, '$.output_code') AS output_code,
       k.start, k.end, k.correlationId
FROM device_task_kernel_data k
JOIN string_table s ON s.ID = k.nameId
WHERE k.isComputation = 1
  AND s.string LIKE 'triton_%'
ORDER BY (k.end - k.start) DESC;
```

## Common SQL Queries

Load string map:

```sql
SELECT ID, string
FROM string_table;
```

List compute kernels in timeline order:

```sql
SELECT processId, deviceId, queueId, start, end, correlationId, nameId
FROM device_task_kernel_data
WHERE isComputation = 1
  AND (:process_id IS NULL OR processId = :process_id)
  AND (:device_id IS NULL OR deviceId = :device_id)
ORDER BY start;
```

Compute kernel summary by observed name:

```sql
SELECT nameId,
       COUNT(*) AS count,
       SUM(end - start) AS total_us,
       AVG(end - start) AS avg_us,
       MAX(end - start) AS max_us
FROM device_task_kernel_data
WHERE isComputation = 1
  AND (:process_id IS NULL OR processId = :process_id)
  AND (:device_id IS NULL OR deviceId = :device_id)
GROUP BY nameId
ORDER BY total_us DESC;
```

All kernel summary split by `isComputation`:

```sql
SELECT isComputation,
       COUNT(*) AS count,
       SUM(end - start) AS total_us,
       AVG(end - start) AS avg_us,
       MAX(end - start) AS max_us
FROM device_task_kernel_data
WHERE (:process_id IS NULL OR processId = :process_id)
  AND (:device_id IS NULL OR deviceId = :device_id)
GROUP BY isComputation;
```

Find device kernel by correlation id:

```sql
SELECT processId, deviceId, queueId, start, end, correlationId, nameId, isComputation, extra
FROM device_task_kernel_data
WHERE correlationId = :corr_id;
```

Find host API/function by correlation id:

```sql
SELECT processId, threadId, start, end, correlationId, nameId, self, extra
FROM function_data
WHERE correlationId = :corr_id;
```

Find same-thread host ranges overlapping a time window:

```sql
SELECT processId, threadId, start, end, extraId, nameId, extra, type
FROM Internal_operation_range_data
WHERE processId = :process_id
  AND threadId = :thread_id
  AND start < :window_end
  AND end > :window_start
ORDER BY start, end DESC;
```

Map framework range `extraId` to related host API calls:

```sql
SELECT r.externalCorrelationId,
       r.correlationId,
       f.processId,
       f.threadId,
       f.start,
       f.end,
       f.nameId,
       f.self,
       f.extra
FROM Internal_op_range_relations r
JOIN function_data f
  ON f.correlationId = r.correlationId
WHERE r.externalCorrelationId = :extra_id
ORDER BY f.start;
```

Find the last device event on a queue before a timestamp:

```sql
SELECT kind, processId, deviceId, queueId, start, end, correlationId, nameId
FROM (
  SELECT 'kernel' AS kind, processId, deviceId, queueId, start, end, correlationId, nameId
  FROM device_task_kernel_data
  UNION ALL
  SELECT 'notifier' AS kind, processId, deviceId, queueId, start, end, correlationId, NULL AS nameId
  FROM device_task_notifier_data
  UNION ALL
  SELECT 'memcpy' AS kind, processId, deviceId, queueId, start, end, correlationId, NULL AS nameId
  FROM device_task_memcpy_data
  UNION ALL
  SELECT 'atomic' AS kind, processId, deviceId, queueId, start, end, correlationId, NULL AS nameId
  FROM device_task_atomic_operation_data
)
WHERE processId = :process_id
  AND deviceId = :device_id
  AND queueId = :queue_id
  AND end <= :before_time
ORDER BY end DESC, start DESC
LIMIT 1;
```

Find notifier wait rows with decoded `unique_val`:

```sql
SELECT processId,
       deviceId,
       queueId,
       correlationId,
       notifierId,
       json_extract(extra, '$.unique_val') AS unique_val,
       start,
       end,
       extra
FROM device_task_notifier_data
WHERE type = 0
  AND (:process_id IS NULL OR processId = :process_id)
  AND (:device_id IS NULL OR deviceId = :device_id)
ORDER BY start;
```

Find matching notifier place for a wait:

```sql
SELECT processId,
       deviceId,
       queueId,
       correlationId,
       notifierId,
       json_extract(extra, '$.unique_val') AS unique_val,
       start,
       end,
       extra
FROM device_task_notifier_data
WHERE type = 1
  AND processId = :process_id
  AND deviceId = :device_id
  AND notifierId = :notifier_id
  AND json_extract(extra, '$.unique_val') = :unique_val
  AND start <= :wait_start
ORDER BY start DESC
LIMIT 1;
```

Find memcpy events in a window:

```sql
SELECT processId, deviceId, queueId, start, end, correlationId, extra
FROM device_task_memcpy_data
WHERE processId = :process_id
  AND deviceId = :device_id
  AND start < :window_end
  AND end > :window_start
ORDER BY start;
```
