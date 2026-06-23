#!/usr/bin/env python3
"""Convert PyTorch profiler Chrome trace JSON to a cnperf-compatible SQLite DB.

The output is a compatibility subset for the local e2e-profiling-analyzer
scripts. It is not a byte-for-byte reconstruction of a native cnperf DB.
"""

import argparse
import collections
import gzip
import json
import os
import re
import sqlite3
import sys


DEFAULT_COMM_REGEX = (
    r"TCDP|CNCL|NCCL|HCCL|ALLREDUCE|ALL_REDUCE|ALLGATHER|ALL_GATHER|"
    r"REDUCE_SCATTER|BROADCAST|ALLTOALL|ALL_TO_ALL|SEND|RECV|"
    r"ALLCONNECTED|MIXALLTOALL"
)

RUNTIME_CATEGORIES = {
    "privateuse1_runtime",
    "cuda_runtime",
    "cuda_driver",
    "runtime",
    "driver",
}

HOST_RANGE_CATEGORIES = {
    "cpu_op",
    "user_annotation",
}

DEVICE_KERNEL_CATEGORIES = {"kernel"}
DEVICE_MEMCPY_CATEGORIES = {"gpu_memcpy", "memcpy"}
DEVICE_MEMSET_CATEGORIES = {"gpu_memset", "memset"}
DEVICE_NOTIFIER_CATEGORIES = {"mtia_ccp_events"}


def ns_from_us(value):
    return int(round(float(value) * 1000))


def safe_int(value, default=0):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def compact_json(value):
    return json.dumps(value or {}, ensure_ascii=False, separators=(",", ":"))


def to_python(value):
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "as_list"):
        return value.as_list()
    return value


def get_arg(args, *keys, default=None):
    for key in keys:
        if key in args:
            return args[key]
    return default


def event_times(event):
    start = ns_from_us(event.get("ts", 0))
    dur = ns_from_us(event.get("dur", 0))
    return start, start + dur


def event_category_is_supported(event):
    if event.get("ph") != "X":
        return False
    cat = event.get("cat") or ""
    if cat in DEVICE_KERNEL_CATEGORIES:
        return True
    if cat in DEVICE_MEMCPY_CATEGORIES:
        return True
    if cat in DEVICE_MEMSET_CATEGORIES:
        return True
    if cat in DEVICE_NOTIFIER_CATEGORIES and "notifier" in (event.get("name") or "").lower():
        return True
    if cat in RUNTIME_CATEGORIES:
        return True
    return cat in HOST_RANGE_CATEGORIES


def find_trace_time_offset(path, max_events=None):
    min_start = None
    for idx, event in enumerate(iter_trace_events(path, max_events=max_events)):
        event = to_python(event)
        if not event_category_is_supported(event):
            continue
        start, _ = event_times(event)
        if min_start is None or start < min_start:
            min_start = start
    return min_start or 0


def default_device_frequency(device_name):
    if device_name == "MLU590-M9DK":
        return 1850
    return 0


def default_device_bandwidth(device_name):
    if device_name == "MLU590-M9DK":
        return 2304000
    return 0


def kernel_class_from_extra(extra):
    kernel_type = str((extra or {}).get("kernel_type") or "").upper()
    if kernel_type == "BLOCK":
        return 1
    if kernel_type.startswith("U"):
        return safe_int(kernel_type[1:], 0) * 4
    return 0


def normalize_metadata_key(key):
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


KERNEL_METADATA_ALIASES = {
    "ioefficiency": "io_efficiency",
    "ioefficiencygbs": "io_efficiency",
    "ioefficiencygbps": "io_efficiency",
    "ioeff": "io_efficiency",
    "memoryefficiency": "io_efficiency",
    "memefficiency": "io_efficiency",
    "tritonoutputcode": "output_code",
    "outputcode": "output_code",
    "tritoncode": "output_code",
    "sourcecode": "output_code",
    "kernelcode": "output_code",
    "kernelnum": "kernel_num_gb",
    "kernelnumgb": "kernel_num_gb",
    "kernelkwargs": "kernel_kwargs",
    "numstages": "num_stages",
}

CPP_WRAPPER_KEY_RE = re.compile(r"cpp\s*[_\-. ]?\s*wrapper|cppwrapper", re.IGNORECASE)
CPP_WRAPPER_VALUE_RE = re.compile(
    r"cpp\s*[_\-. ]?\s*wrapper\w*\s*[:=]\s*['\"]?"
    r"(true|false|1|0|yes|no|on|off|enabled|disabled)",
    re.IGNORECASE,
)
CPP_WRAPPER_ON_EXTENSIONS = {".cc", ".cpp", ".cxx", ".so", ".dylib", ".dll"}
CPP_WRAPPER_OFF_EXTENSIONS = {".py", ".pyc"}


def boolish_cpp_wrapper_state(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled", "enable"}:
        return "on"
    if text in {"0", "false", "no", "off", "disabled", "disable"}:
        return "off"
    match = CPP_WRAPPER_VALUE_RE.search(text)
    if match:
        return boolish_cpp_wrapper_state(match.group(1))
    return None


class CppWrapperDetector:
    def __init__(self):
        self.explicit = []
        self.kernel_file_exts = collections.Counter()

    def observe_kernel_file(self, value, source):
        if not isinstance(value, str) or not value:
            return
        ext = os.path.splitext(value)[1].lower()
        if not ext:
            return
        self.kernel_file_exts[ext] += 1

    def observe_text(self, value, source):
        if not isinstance(value, str):
            return
        state = boolish_cpp_wrapper_state(value)
        if state:
            self.explicit.append({"source": source, "state": state, "value": value[:200]})

    def observe_mapping(self, value, source, depth=0):
        if not isinstance(value, dict) or depth > 3:
            return
        for key, item in value.items():
            norm_key = normalize_metadata_key(key)
            if norm_key == "kernelfile":
                self.observe_kernel_file(item, f"{source}.{key}")
                continue
            if CPP_WRAPPER_KEY_RE.search(str(key)):
                state = boolish_cpp_wrapper_state(item)
                if state:
                    self.explicit.append(
                        {"source": f"{source}.{key}", "state": state, "value": str(item)[:200]}
                    )
                continue
            if isinstance(item, dict):
                self.observe_mapping(item, f"{source}.{key}", depth + 1)
            elif isinstance(item, str) and "cpp" in item.lower() and "wrapper" in item.lower():
                self.observe_text(item, f"{source}.{key}")

    def observe_event(self, event):
        if not isinstance(event, dict):
            return
        name = event.get("name")
        if isinstance(name, str) and "cpp" in name.lower() and "wrapper" in name.lower():
            self.observe_text(name, "event.name")
        args = event.get("args")
        if isinstance(args, dict):
            self.observe_mapping(args, "event.args")

    def summary(self):
        explicit_states = {item["state"] for item in self.explicit}
        py_count = sum(self.kernel_file_exts.get(ext, 0) for ext in CPP_WRAPPER_OFF_EXTENSIONS)
        cpp_count = sum(self.kernel_file_exts.get(ext, 0) for ext in CPP_WRAPPER_ON_EXTENSIONS)

        evidence = self.explicit[:6]
        if self.kernel_file_exts:
            evidence.append(
                {
                    "source": "event.args.kernel_file",
                    "state": "on" if cpp_count and not py_count else "off" if py_count and not cpp_count else "unknown",
                    "value": dict(self.kernel_file_exts.most_common()),
                }
            )

        if len(explicit_states) == 1:
            state = next(iter(explicit_states))
            source = "explicit_trace_metadata"
            confidence = "high"
        elif len(explicit_states) > 1:
            state = "unknown"
            source = "conflicting_explicit_trace_metadata"
            confidence = "low"
        elif cpp_count and not py_count:
            state = "on"
            source = "kernel_file_extension"
            confidence = "medium"
        elif py_count and not cpp_count:
            state = "off"
            source = "kernel_file_extension"
            confidence = "medium"
        elif py_count and cpp_count:
            state = "unknown"
            source = "mixed_kernel_file_extension"
            confidence = "low"
        else:
            state = "unknown"
            source = "not_found"
            confidence = "unknown"

        return {
            "state": state,
            "source": source,
            "confidence": confidence,
            "kernel_file_extensions": dict(self.kernel_file_exts.most_common()),
            "evidence": evidence,
        }


def is_json_scalar(value):
    return value is None or isinstance(value, (str, int, float, bool))


def kernel_extra_payload(args):
    """Keep device extra plus top-level Triton profiler metadata.

    PyTorch traces put launch geometry in args.extra, but fields such as
    "IO efficiency(GB/s)" and "triton output code" live next to extra. Storing
    only args.extra loses the data used by the AI efficiency scripts.
    """
    raw_extra = args.get("extra") if isinstance(args.get("extra"), dict) else {}
    payload = dict(raw_extra)
    for key, value in args.items():
        if key == "extra" or not is_json_scalar(value):
            continue
        payload.setdefault(key, value)
        alias = KERNEL_METADATA_ALIASES.get(normalize_metadata_key(key))
        if alias:
            payload.setdefault(alias, value)
    return payload


def is_gzip_path(path):
    return path.endswith(".gz")


def open_text_trace(path):
    if is_gzip_path(path):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def load_header_metadata(path, limit_bytes=8 * 1024 * 1024):
    """Load top-level metadata by replacing traceEvents with an empty array."""
    with open_text_trace(path) as f:
        buf = ""
        while len(buf) < limit_bytes:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk
            idx = buf.find('"traceEvents"')
            if idx != -1:
                prefix = buf[:idx]
                candidate = prefix + '"traceEvents":[]}'
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return {}
    return {}


def load_simdjson():
    try:
        import simdjson  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing required module 'simdjson'. Install Python package 'pysimdjson'."
        ) from exc
    return simdjson


def iter_trace_events(path, max_events=None):
    simdjson = load_simdjson()
    parser = simdjson.Parser()
    if is_gzip_path(path):
        with gzip.open(path, "rb") as f:
            data = f.read()
        parsed = parser.parse(data)
    else:
        parsed = parser.load(path)
    events = parsed.get("traceEvents", [])

    for idx, event in enumerate(events):
        if max_events and idx >= max_events:
            break
        yield event


class Converter:
    def __init__(
        self,
        out_path,
        comm_regex=DEFAULT_COMM_REGEX,
        process_id=None,
        batch_size=10000,
        time_offset=0,
    ):
        self.out_path = out_path
        self.comm_pattern = re.compile(comm_regex, re.IGNORECASE)
        self.process_id = process_id
        self.default_process_id = process_id
        self.batch_size = batch_size
        self.time_offset = time_offset
        self.name_to_id = {}
        self.next_name_id = 1
        self.generated_corr = -1
        self.stats = collections.Counter()
        self.table_counts = collections.Counter()
        self.category_counts = collections.Counter()
        self.skipped_counts = collections.Counter()
        self.cpp_wrapper_detector = CppWrapperDetector()

        if os.path.exists(out_path):
            os.remove(out_path)
        self.conn = sqlite3.connect(out_path)
        self.cur = self.conn.cursor()
        self.configure_connection()
        self.create_schema()

    def event_times(self, event):
        start, end = event_times(event)
        return start - self.time_offset, end - self.time_offset

    def configure_connection(self):
        self.cur.executescript(
            """
            PRAGMA synchronous = OFF;
            PRAGMA journal_mode = OFF;
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = -262144;
            """
        )

    def create_schema(self):
        self.cur.executescript(
            """
            CREATE TABLE string_table (ID INTEGER PRIMARY KEY, string TEXT);
            CREATE TABLE device_task_kernel_data (
                processId INTEGER, deviceId INTEGER, contextId INTEGER, queueId INTEGER,
                correlationId INTEGER, nodeId INTEGER, nameId INTEGER, start INTEGER,
                end INTEGER, class INTEGER, dimX INTEGER, dimY INTEGER, dimZ INTEGER,
                isComputation INTEGER, extra TEXT
            );
            CREATE TABLE device_task_memcpy_data (
                processId INTEGER, deviceId INTEGER, contextId INTEGER, queueId INTEGER,
                correlationId INTEGER, nodeId INTEGER, type INTEGER, isAsync INTEGER,
                start INTEGER, end INTEGER, bytes INTEGER, extra TEXT
            );
            CREATE TABLE device_task_memset_data (
                processId INTEGER, deviceId INTEGER, contextId INTEGER, queueId INTEGER,
                correlationId INTEGER, nodeId INTEGER, isAsync INTEGER, start INTEGER,
                end INTEGER, bytes INTEGER, value INTEGER, extra TEXT
            );
            CREATE TABLE device_task_atomic_operation_data (
                processId INTEGER, deviceId INTEGER, contextId INTEGER, queueId INTEGER,
                correlationId INTEGER, start INTEGER, end INTEGER, type INTEGER,
                subType INTEGER, value INTEGER, extra TEXT
            );
            CREATE TABLE device_task_hostfunc_data (
                processId INTEGER, deviceId INTEGER, contextId INTEGER, queueId INTEGER,
                correlationId INTEGER, nodeId INTEGER, start INTEGER, end INTEGER,
                funcAddr INTEGER, extra TEXT
            );
            CREATE TABLE device_task_memory_data (
                processId INTEGER, deviceId INTEGER, contextId INTEGER, queueId INTEGER,
                correlationId INTEGER, nodeId INTEGER, type INTEGER, isAsync INTEGER,
                memoryPoolType INTEGER, start INTEGER, end INTEGER, bytes INTEGER, extra TEXT
            );
            CREATE TABLE device_task_notifier_data (
                processId INTEGER, deviceId INTEGER, contextId INTEGER, queueId INTEGER,
                correlationId INTEGER, start INTEGER, end INTEGER, type INTEGER,
                notifierId INTEGER, sequence INTEGER, extra TEXT, nodeId INTEGER
            );
            CREATE TABLE device_task_memory_pool_data (
                processId INTEGER, deviceId INTEGER, correlationId INTEGER, type INTEGER,
                memoryPoolType INTEGER, start INTEGER, end INTEGER, bytes INTEGER, extra TEXT
            );
            CREATE TABLE device_task_jpu_vpps_data (
                processId INTEGER, deviceId INTEGER, contextId INTEGER, queueId INTEGER,
                correlationId INTEGER, type INTEGER, start INTEGER, end INTEGER, extra TEXT
            );
            CREATE TABLE function_data (
                correlationId INTEGER PRIMARY KEY, nameId INTEGER, processId INTEGER,
                threadId INTEGER, start INTEGER, end INTEGER, self INTEGER, extra TEXT
            );
            CREATE TABLE Internal_operation_range_data (
                processId INTEGER, threadId INTEGER, start INTEGER, end INTEGER,
                extraId INTEGER, nameId INTEGER, extra JSON, type INTEGER
            );
            CREATE TABLE Internal_op_range_relations (
                externalCorrelationId INTEGER, correlationId INTEGER
            );
            CREATE TABLE device_information (
                processId INTEGER, deviceId INTEGER, name TEXT, busId TEXT,
                frequency INTEGER, uuid TEXT
            );
            CREATE TABLE thread_information (
                threadId INTEGER, processId INTEGER, name TEXT
            );
            CREATE TABLE Internal_process_thread_name_data (
                processId INTEGER, threadId INTEGER, nameId INTEGER
            );
            CREATE TABLE notifier_information (
                notifierId INTEGER, processId INTEGER, name TEXT
            );
            CREATE TABLE queue_information (
                queueId INTEGER, processId INTEGER, name TEXT
            );
            CREATE TABLE context_information (
                contextId INTEGER, processId INTEGER, name TEXT
            );
            CREATE TABLE domain_Information (
                domainId INTEGER, processId INTEGER, name TEXT
            );
            CREATE TABLE INTERNAL_CNPX_DOMAIN_INFORMATION_V1 (
                processId INT, domainId INT, type, name, description
            );
            CREATE TABLE meta_information (
                type TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE VIEW STRING_IDS_V1 AS SELECT ID AS id, string FROM string_table;
            CREATE VIEW MLU_KERNEL_EVENTS_V1 AS
                SELECT processId, deviceId, contextId, queueId, correlationId, nodeId,
                       nameId, start, end, class, dimX, dimY, dimZ
                FROM device_task_kernel_data;
            """
        )
        self.conn.commit()

    def create_indexes(self):
        indexes = [
            (
                "kernel_start_index",
                "CREATE INDEX kernel_start_index ON device_task_kernel_data(start)",
            ),
            (
                "kernel_id_index",
                "CREATE INDEX kernel_id_index ON device_task_kernel_data(correlationId)",
            ),
            (
                "kernel_name_index",
                "CREATE INDEX kernel_name_index ON device_task_kernel_data(nameId)",
            ),
            (
                "kernel_deviceid_index",
                "CREATE INDEX kernel_deviceid_index ON device_task_kernel_data(deviceId)",
            ),
            (
                "kernel_processid_deviceid_iscomputation_start_index",
                """
                CREATE INDEX kernel_processid_deviceid_iscomputation_start_index
                ON device_task_kernel_data(processId, deviceId, isComputation, start)
                """,
            ),
            (
                "memcpy_start_index",
                "CREATE INDEX memcpy_start_index ON device_task_memcpy_data(start)",
            ),
            (
                "memcpy_id_index",
                "CREATE INDEX memcpy_id_index ON device_task_memcpy_data(correlationId)",
            ),
            (
                "memset_start_index",
                "CREATE INDEX memset_start_index ON device_task_memset_data(start)",
            ),
            (
                "memset_id_index",
                "CREATE INDEX memset_id_index ON device_task_memset_data(correlationId)",
            ),
            (
                "atomic_op_start_index",
                "CREATE INDEX atomic_op_start_index ON device_task_atomic_operation_data(start)",
            ),
            (
                "atomic_op_id_index",
                "CREATE INDEX atomic_op_id_index ON device_task_atomic_operation_data(correlationId)",
            ),
            (
                "notifier_start_index",
                "CREATE INDEX notifier_start_index ON device_task_notifier_data(start)",
            ),
            (
                "notifier_id_index",
                "CREATE INDEX notifier_id_index ON device_task_notifier_data(correlationId)",
            ),
            (
                "func_start_index",
                "CREATE INDEX func_start_index ON function_data(start)",
            ),
            (
                "func_name_index",
                "CREATE INDEX func_name_index ON function_data(nameId)",
            ),
            (
                "op_range_extraId_index",
                "CREATE INDEX op_range_extraId_index ON Internal_operation_range_data(extraId)",
            ),
            (
                "op_range_relation_index",
                "CREATE INDEX op_range_relation_index ON Internal_op_range_relations(correlationId)",
            ),
            (
                "op_range_relation_external_index",
                """
                CREATE INDEX op_range_relation_external_index
                ON Internal_op_range_relations(externalCorrelationId)
                """,
            ),
        ]

        for index_name, sql in indexes:
            try:
                self.cur.execute(sql)
                self.conn.commit()
            except sqlite3.Error as exc:
                raise RuntimeError(f"failed to create index {index_name}: {exc}") from exc

    def name_id(self, name):
        name = name or "<unknown>"
        existing = self.name_to_id.get(name)
        if existing is not None:
            return existing
        name_id = self.next_name_id
        self.next_name_id += 1
        self.name_to_id[name] = name_id
        self.cur.execute("INSERT INTO string_table (ID, string) VALUES (?, ?)", (name_id, name))
        return name_id

    def correlation_id(self, args):
        corr = get_arg(args, "correlation", "Correlation ID", "correlationId")
        if corr is not None:
            return safe_int(corr)
        ext = get_arg(args, "External id", "external id")
        if ext is not None:
            return safe_int(ext)
        corr = self.generated_corr
        self.generated_corr -= 1
        return corr

    def event_process_id(self, event, device_event=False):
        pid = safe_int(event.get("pid"), 0)
        if not device_event and pid:
            if self.default_process_id is None:
                self.default_process_id = pid
            return pid
        return self.default_process_id if self.default_process_id is not None else pid

    def is_comm_kernel(self, name, args):
        if any(k in args for k in ("Collective name", "Process Group Ranks", "Group size")):
            return True
        return bool(self.comm_pattern.search(name or ""))

    def common_device_fields(self, event):
        args = event.get("args") or {}
        extra = args.get("extra") if isinstance(args.get("extra"), dict) else {}
        start, end = self.event_times(event)
        return {
            "process_id": self.event_process_id(event, device_event=True),
            "device_id": safe_int(get_arg(args, "device", "Device Id"), safe_int(event.get("pid"), 0)),
            "context_id": safe_int(get_arg(args, "context", "contextId"), 0),
            "queue_id": safe_int(get_arg(args, "stream", "queue", "queueId"), safe_int(event.get("tid"), 0)),
            "correlation_id": self.correlation_id(args),
            "node_id": safe_int(get_arg(args, "tasktopo node", "nodeId", "tasktopo"), 0),
            "start": start,
            "end": end,
            "args": args,
            "extra": extra,
        }

    def maybe_commit(self):
        converted = sum(self.table_counts.values())
        if converted and converted % self.batch_size == 0:
            self.conn.commit()

    def insert_kernel(self, event):
        fields = self.common_device_fields(event)
        name = event.get("name") or "<kernel>"
        extra = kernel_extra_payload(fields["args"])
        is_comp = 0 if self.is_comm_kernel(name, fields["args"]) else 1
        self.cur.execute(
            """
            INSERT INTO device_task_kernel_data
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fields["process_id"],
                fields["device_id"],
                fields["context_id"],
                fields["queue_id"],
                fields["correlation_id"],
                fields["node_id"],
                self.name_id(name),
                fields["start"],
                fields["end"],
                kernel_class_from_extra(extra),
                safe_int(extra.get("dimx"), 0),
                safe_int(extra.get("dimy"), 0),
                safe_int(extra.get("dimz"), 0),
                is_comp,
                compact_json(extra if extra else fields["args"]),
            ),
        )
        self.table_counts["device_task_kernel_data"] += 1
        self.stats["compute_kernel" if is_comp else "comm_kernel"] += 1

    def insert_memcpy(self, event):
        fields = self.common_device_fields(event)
        name = event.get("name") or ""
        copy_type = 0
        if "HtoD" in name or "H2D" in name:
            copy_type = 1
        elif "DtoH" in name or "D2H" in name:
            copy_type = 2
        elif "DtoD" in name or "D2D" in name:
            copy_type = 3
        self.cur.execute(
            """
            INSERT INTO device_task_memcpy_data
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fields["process_id"],
                fields["device_id"],
                fields["context_id"],
                fields["queue_id"],
                fields["correlation_id"],
                fields["node_id"],
                copy_type,
                safe_int(get_arg(fields["args"], "is_async"), 0),
                fields["start"],
                fields["end"],
                safe_int(fields["extra"].get("bytes"), 0),
                compact_json(fields["args"]),
            ),
        )
        self.table_counts["device_task_memcpy_data"] += 1

    def insert_memset(self, event):
        fields = self.common_device_fields(event)
        self.cur.execute(
            """
            INSERT INTO device_task_memset_data
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fields["process_id"],
                fields["device_id"],
                fields["context_id"],
                fields["queue_id"],
                fields["correlation_id"],
                fields["node_id"],
                safe_int(get_arg(fields["args"], "is_async"), 0),
                fields["start"],
                fields["end"],
                safe_int(fields["extra"].get("bytes"), 0),
                safe_int(fields["extra"].get("value"), 0),
                compact_json(fields["args"]),
            ),
        )
        self.table_counts["device_task_memset_data"] += 1

    def insert_notifier(self, event):
        fields = self.common_device_fields(event)
        name = event.get("name") or ""
        extra = fields["extra"]
        type_text = str(extra.get("type") or name).upper()
        notifier_type = 1 if "PLACE" in type_text else 0
        notifier_id = safe_int(extra.get("notifier_id"), 0)
        self.cur.execute(
            """
            INSERT INTO device_task_notifier_data
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fields["process_id"],
                fields["device_id"],
                fields["context_id"],
                fields["queue_id"],
                fields["correlation_id"],
                fields["start"],
                fields["end"],
                notifier_type,
                notifier_id,
                0,
                compact_json(extra or fields["args"]),
                fields["node_id"],
            ),
        )
        self.table_counts["device_task_notifier_data"] += 1

    def insert_function(self, event):
        args = event.get("args") or {}
        start, end = self.event_times(event)
        corr = self.correlation_id(args)
        external_id = get_arg(args, "External id", "external id")
        self.cur.execute(
            """
            INSERT OR IGNORE INTO function_data
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                corr,
                self.name_id(event.get("name") or "<runtime>"),
                self.event_process_id(event),
                safe_int(event.get("tid"), 0),
                start,
                end,
                end - start,
                compact_json(args),
            ),
        )
        if self.cur.rowcount:
            self.table_counts["function_data"] += 1
        if external_id is not None:
            self.cur.execute(
                "INSERT INTO Internal_op_range_relations VALUES (?, ?)",
                (safe_int(external_id), corr),
            )
            self.table_counts["Internal_op_range_relations"] += 1

    def insert_host_range(self, event):
        args = event.get("args") or {}
        start, end = self.event_times(event)
        external_id = get_arg(args, "External id", "external id", "Ev Idx")
        self.cur.execute(
            """
            INSERT INTO Internal_operation_range_data
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.event_process_id(event),
                safe_int(event.get("tid"), 0),
                start,
                end,
                safe_int(external_id, 0),
                self.name_id(event.get("name") or "<host_range>"),
                compact_json(args),
                0,
            ),
        )
        self.table_counts["Internal_operation_range_data"] += 1

    def insert_device_information(self, metadata):
        devices = metadata.get("deviceProperties") or []
        for dev in devices:
            device_name = dev.get("name", "Unknown")
            frequency = safe_int(dev.get("frequency"), default_device_frequency(device_name))
            self.cur.execute(
                "INSERT INTO device_information VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.default_process_id if self.default_process_id is not None else self.process_id or 0,
                    safe_int(dev.get("id"), 0),
                    device_name,
                    dev.get("busId", ""),
                    frequency,
                    dev.get("uuid", ""),
                ),
            )
        first_device = devices[0] if devices else {}
        first_device_name = first_device.get("name", "Unknown") if first_device else "Unknown"
        cluster_count = safe_int(first_device.get("clusterCount"), 1)
        mcore_per_cluster = safe_int(first_device.get("McorePerCluster"), 0)
        core_count = safe_int(first_device.get("core_count"), 0)
        if not core_count and cluster_count and mcore_per_cluster:
            core_count = cluster_count * mcore_per_cluster
        if not core_count:
            core_count = cluster_count
        frequency = safe_int(first_device.get("frequency"), default_device_frequency(first_device_name))
        device_info = {
            "m_dev_count": len(devices),
            "m_dev_basic_info": {
                "active_core_count": core_count,
                "cluster_count": cluster_count,
                "core_count": core_count,
                "core_frequency": frequency,
                "dev_name": first_device_name,
                "ipu_frequency": frequency,
                "max_bandwidth": safe_int(
                    first_device.get("max_bandwidth"),
                    default_device_bandwidth(first_device_name),
                ),
                "max_perform": {
                    "bf16Tensor": 0,
                    "bf16Vector": 0,
                    "fp16Tensor": 0,
                    "fp16Vector": 0,
                    "fp32Tensor": 0,
                    "fp32Vector": 0,
                    "int16Tensor": 0,
                    "int16Vector": 0,
                    "int4Tensor": 0,
                    "int8Tensor": 0,
                    "int8Vector": 0,
                },
            },
            "m_real_ids": [safe_int(dev.get("id"), idx) for idx, dev in enumerate(devices)],
            "m_start": 0,
            "m_end": len(devices),
            "m_valid": bool(devices),
        }
        self.cur.execute(
            "INSERT OR REPLACE INTO meta_information VALUES (?, ?)",
            ("deviceInfo", compact_json(device_info)),
        )

    def insert_compatibility_metadata(self):
        ranges = []
        for table in (
            "function_data",
            "device_task_kernel_data",
            "device_task_memcpy_data",
            "device_task_memset_data",
            "device_task_atomic_operation_data",
            "device_task_notifier_data",
            "Internal_operation_range_data",
        ):
            self.cur.execute(f"SELECT MIN(start), MAX(end) FROM {table}")
            start, end = self.cur.fetchone()
            if start is not None and end is not None:
                ranges.append((start, end))

        start_ts = min(start for start, _ in ranges) if ranges else 0
        end_ts = max(end for _, end in ranges) if ranges else 0
        for key, value in (
            ("startTimestamp", start_ts),
            ("endTimestamp", end_ts),
            ("rawStartTimestamp", start_ts),
            ("utcStartTimestamp", start_ts),
        ):
            self.cur.execute(
                "INSERT OR REPLACE INTO meta_information VALUES (?, ?)",
                (key, str(value)),
            )
        self.cur.execute("SELECT COUNT(*) FROM device_information")
        device_count = self.cur.fetchone()[0]
        system_information = {
            "07#cnperf version": "6.6.1 converted",
            "08#device count": device_count,
        }
        self.cur.execute(
            "INSERT OR REPLACE INTO meta_information VALUES (?, ?)",
            ("systemInformation", compact_json(system_information)),
        )

    def insert_compatibility_lookup_rows(self):
        thread_rows = set()
        for table in ("function_data", "Internal_operation_range_data"):
            self.cur.execute(
                f"""
                SELECT DISTINCT processId, threadId
                FROM {table}
                WHERE processId IS NOT NULL AND threadId IS NOT NULL
                """
            )
            thread_rows.update((row[0], row[1]) for row in self.cur.fetchall())

        self.cur.executemany(
            "INSERT INTO thread_information VALUES (?, ?, ?)",
            [(thread_id, process_id, "") for process_id, thread_id in sorted(thread_rows)],
        )
        self.cur.executemany(
            "INSERT INTO Internal_process_thread_name_data VALUES (?, ?, ?)",
            [(process_id, thread_id, 0) for process_id, thread_id in sorted(thread_rows)],
        )

        self.cur.execute("SELECT DISTINCT processId FROM device_information")
        process_ids = [row[0] for row in self.cur.fetchall()]
        if not process_ids and self.default_process_id is not None:
            process_ids = [self.default_process_id]
        for process_id in process_ids:
            self.cur.execute(
                "INSERT INTO domain_Information VALUES (?, ?, ?)",
                (0, process_id, "CNPerfReservedForUnInitializedDomain"),
            )
            self.cur.execute(
                "INSERT INTO INTERNAL_CNPX_DOMAIN_INFORMATION_V1 VALUES (?, ?, ?, ?, ?)",
                (process_id, 0, 0, "CNPerfReservedForUnInitializedDomain", "{}"),
            )

    def insert_meta(self, metadata, source_path, parser, max_events):
        self.cpp_wrapper_detector.observe_mapping(metadata, "trace_header")
        cpp_wrapper_signal = self.cpp_wrapper_detector.summary()
        compact = {
            k: v
            for k, v in metadata.items()
            if k not in {"traceEvents", "deviceProperties"}
        }
        compact["source_trace"] = source_path
        compact["parser"] = parser
        compact["max_events"] = max_events
        compact["cpp_wrapper_signal"] = cpp_wrapper_signal
        self.cur.execute(
            "INSERT OR REPLACE INTO meta_information VALUES (?, ?)",
            ("torch_trace_metadata", compact_json(compact)),
        )
        self.cur.execute(
            "INSERT OR REPLACE INTO meta_information VALUES (?, ?)",
            ("torch_compile_cpp_wrapper", compact_json(cpp_wrapper_signal)),
        )

    def process_event(self, event):
        event = to_python(event)
        self.cpp_wrapper_detector.observe_event(event)
        if event.get("ph") != "X":
            self.skipped_counts[f"ph:{event.get('ph')}"] += 1
            return
        cat = event.get("cat") or ""
        self.category_counts[cat] += 1

        if cat in DEVICE_KERNEL_CATEGORIES:
            self.insert_kernel(event)
        elif cat in DEVICE_MEMCPY_CATEGORIES:
            self.insert_memcpy(event)
        elif cat in DEVICE_MEMSET_CATEGORIES:
            self.insert_memset(event)
        elif cat in DEVICE_NOTIFIER_CATEGORIES and "notifier" in (event.get("name") or "").lower():
            self.insert_notifier(event)
        elif cat in RUNTIME_CATEGORIES:
            self.insert_function(event)
        elif cat in HOST_RANGE_CATEGORIES:
            self.insert_host_range(event)
        else:
            self.skipped_counts[f"cat:{cat}"] += 1
            return
        self.maybe_commit()

    def finish(self, metadata, source_path, parser, max_events):
        self.insert_device_information(metadata)
        self.insert_meta(metadata, source_path, parser, max_events)
        self.insert_compatibility_metadata()
        self.insert_compatibility_lookup_rows()
        self.conn.commit()
        self.create_indexes()
        self.conn.commit()

    def report(self, source_path, parser, max_events):
        return {
            "source_trace": source_path,
            "output_db": self.out_path,
            "parser": parser,
            "max_events": max_events,
            "tables": dict(self.table_counts),
            "kernel_breakdown": {
                "compute_kernel": self.stats["compute_kernel"],
                "comm_kernel": self.stats["comm_kernel"],
            },
            "categories_seen": dict(self.category_counts.most_common()),
            "skipped": dict(self.skipped_counts.most_common(50)),
            "limitations": [
                "This is a cnperf-compatible subset generated from Chrome trace JSON.",
                "isComputation is inferred from kernel name and communication metadata.",
                "Missing notifier/runtime correlation in the input trace will reduce gap_detail fidelity.",
                "CUDA runtime/driver APIs are stored as host function_data; CUDA event APIs are not converted to device_task_notifier_data unless the trace contains explicit device-side notifier events.",
            ],
        }

    def close(self):
        self.conn.close()


def convert(args):
    metadata = load_header_metadata(args.trace_json)
    parser_used = "simdjson"
    time_offset = find_trace_time_offset(args.trace_json, max_events=args.max_events)
    converter = Converter(
        args.out,
        comm_regex=args.comm_regex,
        process_id=args.process_id,
        batch_size=args.batch_size,
        time_offset=time_offset,
    )
    count = 0
    try:
        for event in iter_trace_events(args.trace_json, max_events=args.max_events):
            converter.process_event(event)
            count += 1
            if args.progress and count % args.progress == 0:
                print(f"processed_events={count}", file=sys.stderr)
        converter.finish(metadata, args.trace_json, parser_used, args.max_events)
        report = converter.report(args.trace_json, parser_used, args.max_events)
        report["events_processed"] = count
        report["time_offset_ns"] = time_offset
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        converter.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch profiler Chrome trace JSON to cnperf-compatible SQLite DB."
    )
    parser.add_argument("trace_json", help="PyTorch profiler .pt.trace.json file")
    parser.add_argument("--out", required=True, help="Output SQLite DB path")
    parser.add_argument("--report", help="Optional conversion report JSON path")
    parser.add_argument("--max-events", type=int, help="Stop after N traceEvents, useful for testing")
    parser.add_argument("--process-id", type=int, help="Override processId for generated device rows")
    parser.add_argument("--batch-size", type=int, default=10000, help="SQLite commit batch size")
    parser.add_argument("--progress", type=int, default=0, help="Print progress every N events")
    parser.add_argument(
        "--comm-regex",
        default=DEFAULT_COMM_REGEX,
        help="Regex used to infer communication kernels when metadata is absent",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
