const { createApp, ref, computed, watch, nextTick } = Vue;
const { createRouter, createWebHashHistory } = VueRouter;

// ══════════════════════════════════════════════════════════════════════════════
// Module-level reactive state (shared across all components via closure)
// ══════════════════════════════════════════════════════════════════════════════

let appInitialized = false;

const readStoredJson = (key, fallback) => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (e) {
    return fallback;
  }
};

const readStoredBool = (key, fallback) => {
  const raw = localStorage.getItem(key);
  return raw === null ? fallback : raw === "true";
};

const readStoredNumber = (key, fallback) => {
  const raw = Number(localStorage.getItem(key));
  return Number.isFinite(raw) && raw > 0 ? raw : fallback;
};

// ── Theme ──────────────────────────────────────────────────────────────
const getInitialTheme = () => {
  const saved = localStorage.getItem('tpa-theme');
  if (saved) return saved === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
};
const isDark = ref(getInitialTheme());
const toggleTheme = () => {
  isDark.value = !isDark.value;
  const t = isDark.value ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem('tpa-theme', t);
  if (resultTab.value === 'chart' && selectedJob.value?.status === 'done') {
    nextTick(() => buildChart());
  }
};

// ── Data ────────────────────────────────────────────────────────────────
const projects      = ref([]);
const historyGroups = ref([]);
const historyGroupsTotal = ref(0);
const historyGroupsLimit = ref(100);
const historyGroupsOffset = ref(0);
const historyGroupsLoading = ref(false);
const historyGroupJobsLimit = 50;
const historySearch = ref("");
const filterProject = ref(localStorage.getItem("tpa-filter-project") || "");
const sidebarTab    = ref(localStorage.getItem("tpa-sidebar-tab") || "jobs");
const selectedJobId = ref(null);
const selectedJob   = ref(null);
const jobLoading    = ref(false);
const collapsedGroups = ref(readStoredJson("tpa-expanded-groups", {}));
let preSearchExpandedGroups = null;

// ── Upload form ─────────────────────────────────────────────────────────
const fileA    = ref(null);
const fileAName = ref("");
const quickUploadMode = ref(localStorage.getItem("tpa-upload-mode") || "single");
const quickFileA = ref(null);
const quickFileB = ref(null);
const quickFileAName = ref("");
const quickFileBName = ref("");
const quickCompareStatus = ref("");
const uploadQueue = ref([]);
const submitting = ref(false);
const uploadProgress = ref(0);
const form = ref({
  label: "",
  projectId: "",
  saveTritonCsv: true,
  saveTritonCode: true,
});

// ── Result view ─────────────────────────────────────────────────────────
const resultTab   = ref("console");
const tableSearch = ref("");
const sortCol     = ref("");
const sortAsc     = ref(true);
const tableLimit  = ref(100);
const tableOffset = ref(0);
const tablePageSizeOptions = [50, 100, 200, 500, 1000];
const resultTable = ref({ fields: [], rows: [], total: 0, filtered_total: 0, limit: 100, offset: 0 });
const resultTableFile = ref("");
const resultTableLoading = ref(false);
const resultTableError = ref("");
const preparingResultTab = ref("");
const isReadingMode = ref(false);
const chartTables = ref({});
const colWidths     = ref({});
const colFilters    = ref({});
const colFilterOps  = ref({});
const visibleColumns = ref([]);
const showColumnMenu = ref(false);
const ktChartInst     = ref(null);
const ktChart         = ref(null);
const ktPieChartInst  = ref(null);
const ktPieChart      = ref(null);
const ktPieChartInstB = ref(null);
const ktPieChartB     = ref(null);
const chartSource     = ref("");
const chartMetric     = ref("");
const chartTopN       = ref(10);
const chartLoading    = ref(false);
const chartError      = ref("");
const chartSummaryCards = ref([]);
const chartSlowdowns    = ref([]);
const chartSpeedups     = ref([]);
const chartBarRows      = ref([]);
const chartPieRows      = ref([]);

const allowFileDownload = ref(true);
const allowCodeExecution = ref(false);
const perfettoOpening = ref({});
let activeResultStateJobId = null;

const resultStateKey = jobId => `tpa-result-state:${jobId}`;
const readResultMemory = jobId =>
  jobId ? readStoredJson(resultStateKey(jobId), { lastTab: "console", tabs: {} }) : { lastTab: "console", tabs: {} };
const writeResultMemory = (jobId, memory) => {
  if (!jobId) return;
  localStorage.setItem(resultStateKey(jobId), JSON.stringify(memory));
};
let restoringResultState = false;
let restoreResultStateToken = 0;
let suppressResultTabWatch = false;
let suppressResultTabToken = 0;

const markRestoringResultState = () => {
  restoringResultState = true;
  const token = ++restoreResultStateToken;
  nextTick(() => {
    if (restoreResultStateToken === token) restoringResultState = false;
  });
};

const skipNextResultTabWatch = () => {
  suppressResultTabWatch = true;
  const token = ++suppressResultTabToken;
  nextTick(() => {
    if (suppressResultTabToken === token) suppressResultTabWatch = false;
  });
};

const saveResultViewState = (jobId = selectedJobId.value, tab = resultTab.value) => {
  if (!jobId || !tab) return;
  const memory = readResultMemory(jobId);
  memory.lastTab = tab;
  if (tab.endsWith(".csv")) {
    memory.tabs[tab] = {
      tableSearch: tableSearch.value,
      sortCol: sortCol.value,
      sortAsc: sortAsc.value,
      tableLimit: tableLimit.value,
      tableOffset: tableOffset.value,
      colWidths: colWidths.value,
      colFilters: colFilters.value,
      colFilterOps: colFilterOps.value,
      visibleColumns: visibleColumns.value,
    };
  }
  writeResultMemory(jobId, memory);
};
const rememberResultTabSelection = (jobId, tab) => {
  if (!jobId || !tab) return;
  const memory = readResultMemory(jobId);
  memory.lastTab = tab;
  writeResultMemory(jobId, memory);
};
const defaultResultViewState = () => ({
  tableSearch: "",
  sortCol: "",
  sortAsc: true,
  tableLimit: 100,
  tableOffset: 0,
  colWidths: {},
  colFilters: {},
  colFilterOps: {},
  visibleColumns: [],
});

const resultViewStateFor = (jobId, tab) => {
  const memory = readResultMemory(jobId);
  return { ...defaultResultViewState(), ...(memory.tabs?.[tab] || {}) };
};

const applyResultViewState = state => {
  markRestoringResultState();
  tableSearch.value = state.tableSearch || "";
  sortCol.value = state.sortCol || "";
  sortAsc.value = state.sortAsc ?? true;
  tableLimit.value = state.tableLimit || 100;
  tableOffset.value = state.tableOffset || 0;
  colWidths.value = state.colWidths || {};
  colFilters.value = state.colFilters || {};
  colFilterOps.value = state.colFilterOps || {};
  visibleColumns.value = state.visibleColumns || [];
};
const restoreResultViewState = (jobId, tab) => {
  applyResultViewState(resultViewStateFor(jobId, tab));
};
const rememberedResultTab = jobId => readResultMemory(jobId).lastTab || "console";

const refreshReadingLayout = () => {
  if (resultTab.value === "chart" && selectedJob.value?.status === "done") {
    nextTick(() => buildChart());
  }
};

const toggleReadingMode = () => {
  isReadingMode.value = !isReadingMode.value;
  nextTick(refreshReadingLayout);
};

const exitReadingMode = () => {
  if (!isReadingMode.value) return;
  isReadingMode.value = false;
  nextTick(refreshReadingLayout);
};

window.addEventListener("keydown", event => {
  if (event.key === "Escape" && isReadingMode.value) exitReadingMode();
});

// ── Triton ──────────────────────────────────────────────────────────────
const tritonStatus = ref({});

// ── Error display modal ─────────────────────────────────────────────────
const showErrorModal = ref(false);
const errorModalMsg = ref("");
const errorModalTitle = ref("错误信息");

// ── Layout / Modals ─────────────────────────────────────────────────────
const sidebarWidth     = ref(readStoredNumber("tpa-sidebar-width", 240));
const sidebarCollapsed = ref(readStoredBool("tpa-sidebar-collapsed", false));

const toasts = ref([]);
let toastSeq = 0;
const showToast = (message, kind = "info", duration = 2600) => {
  const id = ++toastSeq;
  toasts.value.push({ id, message, kind });
  setTimeout(() => {
    toasts.value = toasts.value.filter(toast => toast.id !== id);
  }, duration);
};

const showConfirmModal = ref(false);
const confirmModal = ref({
  title: "确认操作",
  message: "",
  confirmText: "确认",
  tone: "primary",
});
let confirmResolver = null;
const askConfirm = (message, options = {}) => new Promise(resolve => {
  confirmModal.value = {
    title: options.title || "确认操作",
    message,
    confirmText: options.confirmText || "确认",
    tone: options.tone || "primary",
  };
  confirmResolver = resolve;
  showConfirmModal.value = true;
});
const resolveConfirm = (value) => {
  showConfirmModal.value = false;
  const resolve = confirmResolver;
  confirmResolver = null;
  if (resolve) resolve(value);
};

const toggleSidebar = () => { sidebarCollapsed.value = !sidebarCollapsed.value; };

const startSidebarResize = (e) => {
  const startX = e.clientX;
  const startW = sidebarWidth.value;
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';
  const onMove = ev => {
    sidebarWidth.value = Math.max(160, Math.min(520, startW + ev.clientX - startX));
  };
  const onUp = () => {
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
  };
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
};

const showNewProject  = ref(false);
const newProjectName  = ref("");
const newProjectDesc  = ref("");

const showRenameProject = ref(false);
const renameProjectId = ref("");
const renameProjectName = ref("");

const showMoveProject = ref(false);
const moveProjectTarget = ref("");

const historyBulkMode = ref(false);
const historySelection = ref([]);
const showBulkMoveProject = ref(false);
const bulkMoveProjectTarget = ref("");

const showRenameJob = ref(false);
const renameJobName = ref("");

const showDeletedProjects = ref(false);
const deletedProjects = ref([]);
const showStorageManager = ref(false);
const storageSummary = ref({ totals: {}, projects: [], jobs: [] });
const storageSelection = ref([]);

const loadDeletedProjects = async () => {
  const r = await fetch("/api/deleted-projects", { credentials: "include" });
  deletedProjects.value = await r.json();
};

const isDeletedOver10Days = (deletedAt) => {
  if (!deletedAt) return false;
  const deletedTime = new Date(deletedAt).getTime();
  const now = Date.now();
  const tenDays = 10 * 24 * 60 * 60 * 1000;
  return now - deletedTime > tenDays;
};

const restoreProject = async (projectId) => {
  if (!await askConfirm("确定恢复该项目？", { confirmText: "恢复" })) return;
  try {
    const r = await fetch(`/api/deleted-projects/${projectId}/restore`, {
      method: "POST",
      credentials: "include",
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      showToast("恢复失败: " + (err.detail || err.message || "未知错误"), "error");
      return;
    }
    await loadDeletedProjects();
    await loadProjects();
    await refreshSidebarData();
    showToast("项目已恢复", "success");
  } catch (e) {
    showToast("恢复出错: " + e.message, "error");
  }
};

const permanentlyDeleteProject = async (projectId) => {
  if (!await askConfirm("确定永久删除？此操作不可恢复。", {
    title: "永久删除项目",
    confirmText: "永久删除",
    tone: "danger",
  })) return;
  const r = await fetch(`/api/deleted-projects/${projectId}`, {
    method: "DELETE",
    credentials: "include",
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("永久删除失败: " + (err.detail || err.message || "未知错误"), "error");
    return;
  }
  await loadDeletedProjects();
  showToast("项目已永久删除", "success");
};

// ── Triton code viewer ──────────────────────────────────────────────────
const showTritonCode = ref(false);
const tritonCodeContent = ref("");
const tritonCodeFilename = ref("");
const tritonCodeEditing = ref(false);
const tritonCodeEditContent = ref("");
const customRunStatus = ref("");
const currentTritonCodePath = ref("");

// ── Guide ───────────────────────────────────────────────────────────────
const showGuide = ref(false);

const compareSelection  = ref([]);
const compareSelectionDetails = ref({});
const compareLabel      = ref("");
const compareProjectId  = ref("");
const compareJobs       = ref([]);
const compareJobsTotal  = ref(0);
const compareJobsLimit  = ref(50);
const compareJobsOffset = ref(0);
const compareJobsLoading = ref(false);
const compareSearch     = ref("");

let pollTimer = null;

// ══════════════════════════════════════════════════════════════════════════════
// Computed properties
// ══════════════════════════════════════════════════════════════════════════════

const groupedJobs = computed(() => historyGroups.value);
const loadedHistoryJobs = computed(() =>
  historyGroups.value.flatMap(group => group.jobs || [])
);
const loadedHistoryJobIds = computed(() =>
  loadedHistoryJobs.value.map(job => job.id)
);
const selectedCompareJobs = computed(() =>
  compareSelection.value
    .map(id => compareSelectionDetails.value[id])
    .filter(Boolean)
);

const availableTabs = computed(() => {
  const res = selectedJob.value?.result_files || selectedJob.value?.results;
  if (!res) return [];
  const tabs = [{ key: "console", label: "控制台" }, { key: "chart", label: "图表" }];
  const csvMap = {
    "all_kernels_avg.csv":      "所有 Kernel",
    "all_kernels_cmp.csv":      "Kernel 对比",
    "triton_kernels_avg.csv":   "Triton",
    "triton_kernels_cmp.csv":   "Triton 对比",
    "aten_ops_avg.csv":         "Aten Ops",
    "aten_ops_cmp.csv":         "Aten 对比",
    "kernel_types_avg.csv":     "Kernel 类型",
    "kernel_types_cmp.csv":     "类型对比",
    "kernel_types_delta.csv":   "类型 Delta",
    "cncl_ops_avg.csv":         "CNCL Ops",
    "cncl_ops_cmp.csv":         "CNCL 对比",
  };
  for (const [file, label] of Object.entries(csvMap)) {
    if (res[file]) tabs.push({ key: file, label });
  }
  const tritonFiles = Object.keys(res).sort()
    .filter(f => f.match(/^step_\d+_triton_kernels\.csv$/))
    .slice(0, 3);
  for (const file of tritonFiles) {
    const stepNum = file.match(/^step_(\d+)_/)[1];
    tabs.push({ key: file, label: `Triton Step ${stepNum}` });
  }
  return tabs;
});

const CHART_SOURCE_CONFIGS = [
  { file: "kernel_types_delta.csv", label: "类型 Delta", mode: "compare", nameField: "type", defaultMetric: "delta_dur_ms" },
  { file: "kernel_types_cmp.csv", label: "类型对比", mode: "compare", nameField: "type", defaultMetric: "delta_dur_ms" },
  { file: "all_kernels_cmp.csv", label: "Kernel Delta", mode: "compare", nameField: "kernel_name", defaultMetric: "delta_dur_ms" },
  { file: "triton_kernels_cmp.csv", label: "Triton Delta", mode: "compare", nameField: "kernel_name", defaultMetric: "delta_dur_ms" },
  { file: "aten_ops_cmp.csv", label: "Aten Delta", mode: "compare", nameField: "op_name", defaultMetric: "delta_dur_ms" },
  { file: "cncl_ops_cmp.csv", label: "CNCL Delta", mode: "compare", nameField: "op_name", defaultMetric: "delta_dur_ms" },
  { file: "kernel_types_avg.csv", label: "Kernel 类型", mode: "single", nameField: "type", defaultMetric: "avg_dur_ms" },
  { file: "all_kernels_avg.csv", label: "所有 Kernel", mode: "single", nameField: "kernel_name", defaultMetric: "avg_dur_ms" },
  { file: "triton_kernels_avg.csv", label: "Triton Kernel", mode: "single", nameField: "kernel_name", defaultMetric: "avg_dur_ms" },
  { file: "aten_ops_avg.csv", label: "Aten Ops", mode: "single", nameField: "op_name", defaultMetric: "avg_dur_ms" },
  { file: "cncl_ops_avg.csv", label: "CNCL Ops", mode: "single", nameField: "op_name", defaultMetric: "avg_dur_ms" },
];

const CHART_METRIC_DEFS = [
  { key: "delta_dur_ms", label: "耗时 Delta (B-A)", unit: "ms", signed: true },
  { key: "delta_count", label: "调用数 Delta", unit: "", signed: true },
  { key: "delta_abs_ms", label: "耗时 Delta 绝对值", unit: "ms" },
  { key: "avg_dur_ms", label: "平均耗时", unit: "ms" },
  { key: "avg_dur_ms_A", label: "A 平均耗时", unit: "ms" },
  { key: "avg_dur_ms_B", label: "B 平均耗时", unit: "ms" },
  { key: "avg_count", label: "平均调用数", unit: "" },
  { key: "avg_count_A", label: "A 平均调用数", unit: "" },
  { key: "avg_count_B", label: "B 平均调用数", unit: "" },
  { key: "avg_us_per_call", label: "单次耗时", unit: "us/call" },
  { key: "avg_io_gb", label: "IO 量", unit: "GB" },
  { key: "avg_io_gb_A", label: "A IO 量", unit: "GB" },
  { key: "avg_io_gb_B", label: "B IO 量", unit: "GB" },
  { key: "avg_io_efficiency", label: "IO 效率", unit: "" },
];

const chartSourceOptions = computed(() => {
  const res = selectedJob.value?.result_files || {};
  const mode = selectedJob.value?.mode === "compare" ? "compare" : "single";
  return CHART_SOURCE_CONFIGS.filter(item => item.mode === mode && res[item.file]);
});

const chartMetricOptions = computed(() => {
  const res = selectedJob.value?.result_files || {};
  const fields = res[chartSource.value]?.fields || [];
  const available = new Set(fields);
  return CHART_METRIC_DEFS.filter(item => available.has(item.key));
});

const isTritonStepTab = computed(() => {
  return resultTab.value && resultTab.value.match(/^step_\d+_triton_kernels\.csv$/);
});

const emptyResultTableForTab = filename => {
  const meta = selectedJob.value?.result_files?.[filename];
  return {
    fields: meta?.fields || [],
    rows: [],
    total: 0,
    filtered_total: 0,
    limit: tableLimit.value,
    offset: tableOffset.value,
  };
};

const currentTable = computed(() => {
  if (!resultTab.value.endsWith(".csv")) return { fields: [], rows: [] };
  const eager = selectedJob.value?.results?.[resultTab.value];
  if (eager) return eager;
  if (resultTableFile.value === resultTab.value) return resultTable.value;
  return emptyResultTableForTab(resultTab.value);
});

const tableTotalRows = computed(() =>
  currentTable.value.filtered_total ?? currentTable.value.total ?? currentTable.value.rows?.length ?? 0
);

const tablePageStart = computed(() =>
  tableTotalRows.value ? (currentTable.value.offset || 0) + 1 : 0
);

const tablePageEnd = computed(() =>
  Math.min((currentTable.value.offset || 0) + (currentTable.value.rows || []).length, tableTotalRows.value)
);

const customTableLimit = computed(() => {
  const limit = Number(tableLimit.value);
  return Number.isFinite(limit) && limit > 0 && !tablePageSizeOptions.includes(limit) ? limit : null;
});

const displayedFields = computed(() => {
  const fields = currentTable.value.fields || [];
  if (!visibleColumns.value.length) return fields;
  const visible = new Set(visibleColumns.value);
  return fields.filter(field => visible.has(field));
});

const hiddenColumnCount = computed(() =>
  Math.max(0, currentTable.value.fields.length - displayedFields.value.length)
);

const storageJobsWithTrace = computed(() =>
  storageSummary.value.jobs.filter(job => job.has_original_trace)
);

const hasColFilters = computed(() =>
  Object.values(colFilters.value).some(v => v)
);

const clearColFilters = () => {
  colFilters.value = {};
  colFilterOps.value = {};
};

const filteredRows = computed(() => {
  let rows = currentTable.value.rows || [];
  if (!selectedJob.value?.results?.[resultTab.value]) return rows;
  if (tableSearch.value) {
    const q = tableSearch.value.toLowerCase();
    rows = rows.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)));
  }
  for (const [field, val] of Object.entries(colFilters.value)) {
    if (!val) continue;
    const op = colFilterOps.value[field] || '~';
    rows = rows.filter(r => {
      const cell = r[field] ?? '';
      if (op === '~' || op === '!~') {
        const terms = val.split('|').map(t => t.toLowerCase()).filter(t => t);
        const hit = terms.some(t => String(cell).toLowerCase().includes(t));
        return op === '~' ? hit : !hit;
      }
      const num = parseFloat(val);
      const cellNum = parseFloat(cell);
      if (isNaN(num) || isNaN(cellNum)) return isNaN(num);
      if (op === '>=') return cellNum >= num;
      if (op === '<=') return cellNum <= num;
      if (op === '>')  return cellNum >  num;
      if (op === '<')  return cellNum <  num;
      if (op === '=')  return cellNum === num;
      return true;
    });
  }
  if (sortCol.value) {
    rows = [...rows].sort((a, b) => {
      const va = parseFloat(a[sortCol.value]) || a[sortCol.value] || "";
      const vb = parseFloat(b[sortCol.value]) || b[sortCol.value] || "";
      if (va < vb) return sortAsc.value ? -1 : 1;
      if (va > vb) return sortAsc.value ? 1 : -1;
      return 0;
    });
  }
  return rows;
});

const colSums = computed(() => {
  const fields = displayedFields.value;
  const rows   = filteredRows.value;
  const result = {};
  for (const f of fields) {
    if (f.toLowerCase().includes('efficiency')) { result[f] = null; continue; }
    if (rows.some(r => String(r[f] ?? '').trim().endsWith('%'))) {
      result[f] = null; continue;
    }
    const nums = rows.map(r => parseFloat(r[f])).filter(n => !isNaN(n));
    result[f] = nums.length > 0 ? nums.reduce((a, b) => a + b, 0) : null;
  }
  return result;
});

const fmtSum = v => {
  if (v === null) return '';
  if (Number.isInteger(v)) return String(v);
  const s = v.toFixed(3);
  return parseFloat(s).toString();
};

const fmtBytes = bytes => {
  const n = Number(bytes || 0);
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = n;
  let unit = "B";
  for (const nextUnit of units) {
    value /= 1024;
    unit = nextUnit;
    if (value < 1024) break;
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${unit}`;
};

const isColumnVisible = field =>
  !visibleColumns.value.length || visibleColumns.value.includes(field);

const resetVisibleColumns = () => {
  visibleColumns.value = [...currentTable.value.fields];
};

const applyCoreColumnPreset = () => {
  const fields = currentTable.value.fields;
  if (!fields.length) return;
  const keep = fields.filter((field, index) =>
    index === 0 || /(name|kernel|op|type|avg|mean|total|time|duration|delta|count|calls|efficiency)/i.test(field)
  );
  visibleColumns.value = keep.length ? keep : fields.slice(0, Math.min(fields.length, 6));
};

const toggleColumnVisibility = field => {
  if (!visibleColumns.value.length) {
    visibleColumns.value = [...currentTable.value.fields];
  }
  if (visibleColumns.value.includes(field)) {
    if (visibleColumns.value.length === 1) return;
    visibleColumns.value = visibleColumns.value.filter(item => item !== field);
    return;
  }
  visibleColumns.value = [...visibleColumns.value, field];
};

// ══════════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════════

const fmtDate = iso => iso ? iso.replace("T", " ").slice(0, 16) : "";

const statusIcon = s => ({ pending: "⏳", running: "⟳", done: "✓", error: "✗" }[s] || s);

const toggleGroup = async label => {
  const opening = !collapsedGroups.value[label];
  collapsedGroups.value[label] = opening;
  if (opening) {
    const group = historyGroups.value.find(item => item.id === label);
    if (group && !group.jobs_loaded) await loadHistoryGroupJobs(label, true);
  }
};

const deltaCellClass = (field, value) => {
  if (!field.includes("delta")) return "";
  const n = parseFloat(value);
  if (isNaN(n)) return "";
  return n > 0 ? "cell-neg" : n < 0 ? "cell-pos" : "";
};

// ══════════════════════════════════════════════════════════════════════════════
// Data fetching
// ══════════════════════════════════════════════════════════════════════════════

const loadConfig = async () => {
  const r = await fetch("/api/config");
  const cfg = await r.json();
  allowFileDownload.value = cfg.allow_file_download ?? true;
  allowCodeExecution.value = cfg.allow_code_execution ?? false;
};

const loadProjects = async () => {
  try {
    const r = await fetch("/api/projects", { credentials: "include" });
    if (!r.ok) throw new Error("加载项目失败: HTTP " + r.status);
    projects.value = await r.json();
    if (
      filterProject.value &&
      filterProject.value !== "__none__" &&
      !projects.value.some(project => project.id === filterProject.value)
    ) {
      filterProject.value = "";
    }
  } catch (e) {
    console.error("loadProjects error:", e);
  }
};

const loadStorageSummary = async () => {
  const r = await fetch("/api/storage/summary", { credentials: "include" });
  if (!r.ok) {
    showToast("加载存储统计失败", "error");
    return;
  }
  storageSummary.value = await r.json();
  const valid = new Set(storageJobsWithTrace.value.map(job => job.id));
  storageSelection.value = storageSelection.value.filter(id => valid.has(id));
};

const openStorageManager = async () => {
  await loadStorageSummary();
  showStorageManager.value = true;
};

const toggleStorageSelection = jobId => {
  const idx = storageSelection.value.indexOf(jobId);
  if (idx >= 0) storageSelection.value.splice(idx, 1);
  else storageSelection.value.push(jobId);
};

const toggleAllStorageSelection = () => {
  const ids = storageJobsWithTrace.value.map(job => job.id);
  const selected = new Set(storageSelection.value);
  if (ids.length && ids.every(id => selected.has(id))) {
    storageSelection.value = [];
    return;
  }
  storageSelection.value = ids;
};

const deleteSelectedStorageFiles = async () => {
  if (!storageSelection.value.length) return;
  const selected = storageSummary.value.jobs.filter(job => storageSelection.value.includes(job.id));
  const affectedCompareCount = selected.reduce((sum, job) => sum + (job.used_by_compare_count || 0), 0);
  const message = affectedCompareCount
    ? `确定删除选中的 ${selected.length} 个任务的原始 trace 文件？其中 ${affectedCompareCount} 个历史对比依赖这些源文件，删除后将无法重新打开对应源 trace。`
    : `确定删除选中的 ${selected.length} 个任务的原始 trace 文件？`;
  if (!await askConfirm(message, {
    title: "删除原始文件",
    confirmText: "删除",
    tone: "danger",
  })) return;

  const ids = [...storageSelection.value];
  const r = await fetch("/api/jobs/bulk/delete-files", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ job_ids: ids }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("批量删除文件失败: " + (err.detail || err.message || "未知错误"), "error");
    return;
  }
  storageSelection.value = [];
  await Promise.all([loadStorageSummary(), refreshSidebarData()]);
  if (selectedJobId.value && ids.includes(selectedJobId.value)) await loadJob(selectedJobId.value);
  showToast(`已清理 ${ids.length} 个任务的原始文件`, "success");
};

let historyGroupsController = null;
let compareJobsController = null;
let resultTableController = null;
const historyGroupControllers = {};

const cancelResultTableRequest = () => {
  if (resultTableController) {
    resultTableController.abort();
    resultTableController = null;
  }
  preparingResultTab.value = "";
  resultTableLoading.value = false;
};

const loadHistoryGroups = async () => {
  if (historyGroupsController) historyGroupsController.abort();
  for (const [groupId, controller] of Object.entries(historyGroupControllers)) {
    controller.abort();
    delete historyGroupControllers[groupId];
  }
  const controller = new AbortController();
  historyGroupsController = controller;
  historyGroupsLoading.value = true;
  const params = new URLSearchParams();
  if (filterProject.value) params.set("project_id", filterProject.value);
  if (historySearch.value.trim()) params.set("q", historySearch.value.trim());
  params.set("limit", String(historyGroupsLimit.value));
  params.set("offset", String(historyGroupsOffset.value));
  try {
    const r = await fetch(`/api/job-groups?${params}`, {
      credentials: "include",
      signal: controller.signal,
    });
    const data = await r.json();
    if (historyGroupsController !== controller) return;
    historyGroups.value = (data.data || []).map(group => ({
      ...group,
      jobs: [],
      jobs_total: 0,
      jobs_offset: 0,
      jobs_loaded: false,
      jobs_loading: false,
    }));
    historyGroupsTotal.value = data.total || 0;

    if (historySearch.value.trim()) {
      collapsedGroups.value = Object.fromEntries(
        historyGroups.value.map(group => [group.id, true])
      );
    }
    const expandedGroups = historyGroups.value.filter(group => collapsedGroups.value[group.id]);
    await Promise.all(expandedGroups.map(group => loadHistoryGroupJobs(group.id, true)));
  } catch (e) {
    if (e.name !== "AbortError") showToast("加载历史记录失败", "error");
  } finally {
    if (historyGroupsController === controller) {
      historyGroupsLoading.value = false;
      historyGroupsController = null;
    }
  }
};

const updateHistoryGroup = (groupId, patch) => {
  historyGroups.value = historyGroups.value.map(group =>
    group.id === groupId ? { ...group, ...patch } : group
  );
};

const loadHistoryGroupJobs = async (groupId, reset = false) => {
  const group = historyGroups.value.find(item => item.id === groupId);
  if (!group) return;

  if (historyGroupControllers[groupId]) historyGroupControllers[groupId].abort();
  const controller = new AbortController();
  historyGroupControllers[groupId] = controller;

  const offset = reset ? 0 : group.jobs_offset;
  updateHistoryGroup(groupId, { jobs_loading: true });
  const params = new URLSearchParams();
  if (historySearch.value.trim()) params.set("q", historySearch.value.trim());
  params.set("limit", String(historyGroupJobsLimit));
  params.set("offset", String(offset));
  try {
    const r = await fetch(`/api/job-groups/${groupId}/jobs?${params}`, {
      credentials: "include",
      signal: controller.signal,
    });
    const data = await r.json();
    if (historyGroupControllers[groupId] !== controller) return;
    const latest = historyGroups.value.find(item => item.id === groupId);
    if (!latest) return;
    const jobs = reset ? (data.data || []) : [...latest.jobs, ...(data.data || [])];
    updateHistoryGroup(groupId, {
      jobs,
      jobs_total: data.total || 0,
      jobs_offset: jobs.length,
      jobs_loaded: true,
      jobs_loading: false,
    });
  } catch (e) {
    if (e.name !== "AbortError") {
      updateHistoryGroup(groupId, { jobs_loading: false });
      showToast("加载项目任务失败", "error");
    }
  } finally {
    if (historyGroupControllers[groupId] === controller) {
      delete historyGroupControllers[groupId];
    }
  }
};

const loadCompareJobs = async () => {
  if (compareJobsController) compareJobsController.abort();
  const controller = new AbortController();
  compareJobsController = controller;
  compareJobsLoading.value = true;
  const params = new URLSearchParams();
  if (filterProject.value) params.set("project_id", filterProject.value);
  if (compareSearch.value.trim()) params.set("q", compareSearch.value.trim());
  params.set("limit", String(compareJobsLimit.value));
  params.set("offset", String(compareJobsOffset.value));
  try {
    const r = await fetch(`/api/compare-candidates?${params}`, {
      credentials: "include",
      signal: controller.signal,
    });
    const data = await r.json();
    if (compareJobsController !== controller) return;
    compareJobs.value = data.data || [];
    compareJobsTotal.value = data.total || 0;

    const details = { ...compareSelectionDetails.value };
    for (const job of compareJobs.value) {
      if (compareSelection.value.includes(job.id)) details[job.id] = job;
    }
    compareSelectionDetails.value = details;
  } catch (e) {
    if (e.name !== "AbortError") showToast("加载对比候选失败", "error");
  } finally {
    if (compareJobsController === controller) {
      compareJobsLoading.value = false;
      compareJobsController = null;
    }
  }
};

const refreshSidebarData = async () => {
  await Promise.all([loadHistoryGroups(), loadCompareJobs()]);
};

const loadJob = async id => {
  const r = await fetch(`/api/jobs/${id}`, { credentials: "include" });
  if (!r.ok) {
    selectedJob.value = null;
    return false;
  }
  selectedJob.value = await r.json();
  resultTable.value = { fields: [], rows: [], total: 0, filtered_total: 0, limit: tableLimit.value, offset: tableOffset.value };
  resultTableFile.value = "";
  chartTables.value = {};
  chartSource.value = "";
  chartMetric.value = "";
  chartError.value = "";
  chartSummaryCards.value = [];
  chartSlowdowns.value = [];
  chartSpeedups.value = [];
  chartBarRows.value = [];
  chartPieRows.value = [];
  return true;
};

const activeColumnFilters = (state = null) => {
  const sourceFilters = state?.colFilters || colFilters.value;
  const sourceOps = state?.colFilterOps || colFilterOps.value;
  const filters = {};
  const ops = {};
  for (const [field, value] of Object.entries(sourceFilters)) {
    if (value === undefined || value === null || value === "") continue;
    filters[field] = value;
    ops[field] = sourceOps[field] || "~";
  }
  return { filters, ops };
};

const buildResultTableParams = (overrides = {}) => {
  const params = new URLSearchParams();
  const useViewState = !overrides.ignoreViewState;
  const state = overrides.viewState || null;
  const searchSource = state ? state.tableSearch || "" : tableSearch.value;
  const sortSource = state ? state.sortCol || "" : sortCol.value;
  const sortAscSource = state ? state.sortAsc ?? true : sortAsc.value;
  const search = overrides.q ?? (useViewState ? String(searchSource).trim() : "");
  const limit = overrides.limit ?? state?.tableLimit ?? tableLimit.value;
  const offset = overrides.offset ?? state?.tableOffset ?? tableOffset.value;
  params.set("limit", String(limit));
  params.set("offset", String(Math.max(0, offset)));
  if (search) params.set("q", search);
  if (useViewState && sortSource) {
    params.set("sort_col", sortSource);
    params.set("sort_dir", sortAscSource ? "asc" : "desc");
  }
  if (useViewState) {
    const { filters, ops } = activeColumnFilters(state);
    if (Object.keys(filters).length) {
      params.set("filters", JSON.stringify(filters));
      params.set("filter_ops", JSON.stringify(ops));
    }
  }
  return params;
};

const fetchResultTable = async (filename, options = {}) => {
  const params = buildResultTableParams(options);
  const r = await fetch(
    `/api/jobs/${selectedJobId.value}/results/${encodeURIComponent(filename)}?${params}`,
    { credentials: "include", signal: options.signal },
  );
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || "加载表格失败");
  }
  return await r.json();
};

const loadResultTable = async ({ resetOffset = false, filename = resultTab.value, viewState = null } = {}) => {
  if (!selectedJobId.value || !filename?.endsWith(".csv")) return;
  if (resetOffset) {
    if (viewState) viewState = { ...viewState, tableOffset: 0 };
    else tableOffset.value = 0;
  }
  if (resultTableController) resultTableController.abort();
  const controller = new AbortController();
  resultTableController = controller;
  resultTableLoading.value = true;
  resultTableError.value = "";
  const shouldClearTable = resultTableFile.value !== filename;
  resultTableFile.value = filename;
  if (shouldClearTable) {
    resultTable.value = emptyResultTableForTab(filename);
  }
  try {
    const data = await fetchResultTable(filename, { signal: controller.signal, viewState });
    if (resultTableController !== controller) return;
    if (resultTableFile.value !== filename) return;
    if (resultTab.value !== filename) return;
    resultTable.value = data;
    tableLimit.value = data.limit || tableLimit.value;
    tableOffset.value = data.offset || 0;
  } catch (e) {
    if (e.name !== "AbortError") resultTableError.value = e.message || "加载表格失败";
  } finally {
    if (resultTableController === controller) {
      resultTableController = null;
      resultTableLoading.value = false;
    }
  }
};

const activateCsvTab = async (filename, { updateRoute = true, savePrevious = true } = {}) => {
  const jobId = selectedJobId.value;
  if (!jobId || !filename?.endsWith(".csv")) return;
  if (savePrevious) saveResultViewState(activeResultStateJobId, resultTab.value);
  if (resultTableController) resultTableController.abort();

  const controller = new AbortController();
  const state = resultViewStateFor(jobId, filename);
  resultTableController = controller;
  preparingResultTab.value = filename;
  resultTableLoading.value = false;
  resultTableError.value = "";

  try {
    const data = await fetchResultTable(filename, { signal: controller.signal, viewState: state });
    if (resultTableController !== controller) return;
    if (selectedJobId.value !== jobId) return;
    applyResultViewState(state);
    resultTableFile.value = filename;
    resultTable.value = data;
    tableLimit.value = data.limit || state.tableLimit || tableLimit.value;
    tableOffset.value = data.offset || state.tableOffset || 0;
    resultTableLoading.value = false;
    resultTableError.value = "";
    showColumnMenu.value = false;
    skipNextResultTabWatch();
    resultTab.value = filename;
    saveResultViewState(jobId, filename);
    if (updateRoute) router.push({ path: `/job/${jobId}/${filename}` });
  } catch (e) {
    if (e.name === "AbortError") return;
    if (selectedJobId.value !== jobId) return;
    applyResultViewState(state);
    resultTableFile.value = filename;
    resultTable.value = emptyResultTableForTab(filename);
    resultTableLoading.value = false;
    resultTableError.value = e.message || "加载表格失败";
    showColumnMenu.value = false;
    skipNextResultTabWatch();
    resultTab.value = filename;
    saveResultViewState(jobId, filename);
    if (updateRoute) router.push({ path: `/job/${jobId}/${filename}` });
  } finally {
    if (resultTableController === controller) {
      resultTableController = null;
      preparingResultTab.value = "";
    }
  }
};

const prevTablePage = () => {
  tableOffset.value = Math.max(0, tableOffset.value - tableLimit.value);
  loadResultTable();
};

const nextTablePage = () => {
  if (tableOffset.value + tableLimit.value >= tableTotalRows.value) return;
  tableOffset.value += tableLimit.value;
  loadResultTable();
};

const changeTableLimit = value => {
  const next = Number(value);
  if (!Number.isFinite(next) || next <= 0 || next === tableLimit.value) return;
  tableLimit.value = Math.floor(next);
  tableOffset.value = 0;
  loadResultTable();
};

const showAllTableRows = () => {
  const total = Number(tableTotalRows.value);
  if (!Number.isFinite(total) || total <= 0) return;
  changeTableLimit(total);
};

const startPoll = () => {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (!selectedJobId.value) return clearInterval(pollTimer);
    try {
      await loadJob(selectedJobId.value);
    } catch (e) {
      return; // network error, retry next tick
    }
    if (selectedJob.value?.status === "done" || selectedJob.value?.status === "error") {
      clearInterval(pollTimer);
      resultTab.value = "console";
      refreshSidebarData();
      return;
    }
  }, 2000);
};

// ══════════════════════════════════════════════════════════════════════════════
// Chart
// ══════════════════════════════════════════════════════════════════════════════

const PIE_COLORS = [
  'rgba(99,102,241,.82)',  'rgba(234,88,12,.82)',   'rgba(16,163,74,.82)',
  'rgba(220,38,38,.82)',   'rgba(168,85,247,.82)',  'rgba(14,165,233,.82)',
  'rgba(245,158,11,.82)',  'rgba(20,184,166,.82)',  'rgba(244,63,94,.82)',
  'rgba(6,182,212,.82)',   'rgba(251,191,36,.82)',  'rgba(52,211,153,.82)',
  'rgba(239,68,68,.82)',   'rgba(139,92,246,.82)',  'rgba(34,197,94,.82)',
  'rgba(249,115,22,.82)',  'rgba(59,130,246,.82)',  'rgba(236,72,153,.82)',
  'rgba(132,204,22,.82)',  'rgba(20,184,166,.82)',
];
const getColors = n => Array.from({ length: n }, (_, i) => PIE_COLORS[i % PIE_COLORS.length]);
const CHART_FETCH_LIMIT = 5000;
const chartTopNOptions = [5, 10, 15, 20, 30];

const chartColors = () => {
  const dark = isDark.value;
  return {
    text:       dark ? '#cbd5e1' : '#475569',
    title:      dark ? '#e2e8f0' : '#1e293b',
    grid:       dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)',
    border:     dark ? '#1e293b' : '#ffffff',
  };
};

const parseChartNumber = value => {
  if (value === null || value === undefined || value === "") return 0;
  const n = parseFloat(String(value).replace(/,/g, "").replace("%", ""));
  return Number.isFinite(n) ? n : 0;
};

const trimNumber = value => {
  if (!Number.isFinite(value)) return "0";
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 1 : abs >= 10 ? 2 : 3;
  return value.toFixed(digits).replace(/\.?0+$/, "");
};

const fmtChartValue = (value, metricDef = {}) => {
  const text = trimNumber(value);
  return metricDef.unit ? `${text} ${metricDef.unit}` : text;
};
const fmtDeltaMs = value => fmtChartValue(value, { unit: "ms" });

const shortChartLabel = (label, max = 44) => {
  const text = String(label || "");
  if (text.length <= max) return text;
  const head = Math.max(12, Math.floor(max * 0.58));
  const tail = Math.max(8, max - head - 3);
  return `${text.slice(0, head)}...${text.slice(-tail)}`;
};

const chartMetricDefFor = key =>
  CHART_METRIC_DEFS.find(item => item.key === key) || { key, label: key, unit: "" };

const resolveChartSource = () => {
  const options = chartSourceOptions.value;
  if (!options.length) return null;
  if (!chartSource.value || !options.some(item => item.file === chartSource.value)) {
    chartSource.value = options[0].file;
  }
  return options.find(item => item.file === chartSource.value) || options[0];
};

const resolveChartMetric = sourceConfig => {
  if (!sourceConfig) return null;
  const options = chartMetricOptions.value;
  if (!options.length) return null;
  if (!chartMetric.value || !options.some(item => item.key === chartMetric.value)) {
    chartMetric.value = options.some(item => item.key === sourceConfig.defaultMetric)
      ? sourceConfig.defaultMetric
      : options[0].key;
  }
  return chartMetricDefFor(chartMetric.value);
};

const inferChartNameField = (fields, sourceConfig) => {
  if (fields.includes(sourceConfig.nameField)) return sourceConfig.nameField;
  return fields.find(field => /(kernel|op|type|name)/i.test(field)) || fields[0] || "";
};

const normalizeChartRows = (rows, fields, sourceConfig, metricDef) => {
  const nameField = inferChartNameField(fields, sourceConfig);
  return (rows || [])
    .map(row => {
      const label = String(row[nameField] ?? "").trim();
      const value = parseChartNumber(row[metricDef.key]);
      const delta = parseChartNumber(row.delta_dur_ms);
      return {
        label,
        shortLabel: shortChartLabel(label),
        value,
        displayValue: metricDef.signed ? Math.abs(value) : value,
        delta,
        countDelta: parseChartNumber(row.delta_count),
        aValue: parseChartNumber(row.avg_dur_ms_A),
        bValue: parseChartNumber(row.avg_dur_ms_B),
        countA: parseChartNumber(row.avg_count_A),
        countB: parseChartNumber(row.avg_count_B),
        source: sourceConfig.file,
        sourceLabel: sourceConfig.label,
        metric: metricDef.key,
        nameField,
        raw: row,
      };
    })
    .filter(row => row.label);
};

const sortChartRows = (rows, metricDef) => {
  const metricValue = row => metricDef.signed ? Math.abs(row.value) : row.value;
  return [...rows]
    .filter(row => metricValue(row) > 0)
    .sort((a, b) => metricValue(b) - metricValue(a));
};

const buildPieRows = (rows, metricDef, topN) => {
  const sorted = sortChartRows(rows, metricDef);
  const top = sorted.slice(0, topN);
  const rest = sorted.slice(topN);
  const otherValue = rest.reduce((sum, row) => sum + row.displayValue, 0);
  const pieRows = top.map(row => ({ ...row }));
  if (otherValue > 0) {
    pieRows.push({
      label: "Other",
      shortLabel: "Other",
      value: otherValue,
      displayValue: otherValue,
      isOther: true,
      metric: metricDef.key,
    });
  }
  return pieRows;
};

const buildChartSummary = (rows, table, sourceConfig) => {
  const totalLabel = table?.total && table.total > rows.length
    ? `前 ${rows.length} / 共 ${table.total} 项`
    : `${rows.length} 项`;
  const makeCard = (label, value, sub = "", tone = "", row = null) => ({ label, value, sub, tone, row });
  if (selectedJob.value?.mode === "compare") {
    const totalA = rows.reduce((sum, row) => sum + row.aValue, 0);
    const totalB = rows.reduce((sum, row) => sum + row.bValue, 0);
    const totalDelta = rows.reduce((sum, row) => sum + row.delta, 0);
    const slowest = rows.filter(row => row.delta > 0).sort((a, b) => b.delta - a.delta)[0] || null;
    const fastest = rows.filter(row => row.delta < 0).sort((a, b) => a.delta - b.delta)[0] || null;
    return [
      makeCard("A 总耗时", fmtChartValue(totalA, { unit: "ms" }), totalLabel),
      makeCard("B 总耗时", fmtChartValue(totalB, { unit: "ms" }), sourceConfig.label),
      makeCard("总 Delta", fmtChartValue(totalDelta, { unit: "ms" }), "B - A", totalDelta > 0 ? "neg" : totalDelta < 0 ? "pos" : ""),
      makeCard("最大回退", slowest ? fmtChartValue(slowest.delta, { unit: "ms" }) : "0", slowest ? shortChartLabel(slowest.label, 28) : "无", "neg", slowest),
      makeCard("最大改善", fastest ? fmtChartValue(fastest.delta, { unit: "ms" }) : "0", fastest ? shortChartLabel(fastest.label, 28) : "无", "pos", fastest),
    ];
  }

  const totalDur = rows.reduce((sum, row) => sum + parseChartNumber(row.raw.avg_dur_ms), 0);
  const totalCount = rows.reduce((sum, row) => sum + parseChartNumber(row.raw.avg_count), 0);
  const hotspot = rows
    .map(row => ({ ...row, hotValue: parseChartNumber(row.raw.avg_dur_ms) }))
    .sort((a, b) => b.hotValue - a.hotValue)[0] || null;
  const topPct = totalDur && hotspot ? hotspot.hotValue / totalDur * 100 : 0;
  return [
    makeCard("总耗时", fmtChartValue(totalDur, { unit: "ms" }), totalLabel),
    makeCard("最大热点", hotspot ? fmtChartValue(hotspot.hotValue, { unit: "ms" }) : "0", hotspot ? shortChartLabel(hotspot.label, 28) : "无", "", hotspot),
    makeCard("总调用数", fmtChartValue(totalCount), "avg_count 合计"),
    makeCard("Top 占比", `${trimNumber(topPct)}%`, hotspot ? shortChartLabel(hotspot.label, 28) : "无"),
  ];
};

const updateDeltaLists = rows => {
  if (selectedJob.value?.mode !== "compare") {
    chartSlowdowns.value = [];
    chartSpeedups.value = [];
    return;
  }
  chartSlowdowns.value = rows
    .filter(row => row.delta > 0)
    .sort((a, b) => b.delta - a.delta)
    .slice(0, 6);
  chartSpeedups.value = rows
    .filter(row => row.delta < 0)
    .sort((a, b) => a.delta - b.delta)
    .slice(0, 6);
};

const destroyChartInstances = () => {
  if (ktChartInst.value)     { ktChartInst.value.destroy();     ktChartInst.value = null; }
  if (ktPieChartInst.value)  { ktPieChartInst.value.destroy();  ktPieChartInst.value = null; }
  if (ktPieChartInstB.value) { ktPieChartInstB.value.destroy(); ktPieChartInstB.value = null; }
};

const buildPie = (canvas, rows, title, metricDef) => {
  const pairs = (rows || []).filter(row => row.displayValue > 0);
  if (!pairs.length || !canvas) return null;
  const total = pairs.reduce((sum, row) => sum + row.displayValue, 0);
  const cc = chartColors();
  const colors = getColors(pairs.length).map((color, index) =>
    pairs[index].isOther ? 'rgba(148,163,184,.72)' : color
  );
  return new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: pairs.map(row => row.shortLabel),
      datasets: [{ data: pairs.map(row => row.displayValue),
        backgroundColor: colors,
        borderWidth: 2, borderColor: cc.border }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: title, font: { size: 13 }, color: cc.title },
        legend: {
          position: 'bottom',
          labels: { font: { size: 11 }, boxWidth: 12, padding: 8, color: cc.text,
                    generateLabels: chart => {
                      const ds = chart.data.datasets[0];
                      return chart.data.labels.map((label, i) => ({
                        text: `${label}  ${(ds.data[i] / total * 100).toFixed(1)}%`,
                        fillStyle: ds.backgroundColor[i],
                        strokeStyle: ds.backgroundColor[i],
                        fontColor: cc.text,
                        hidden: false, index: i,
                      }));
                    }},
        },
        tooltip: { callbacks: { label: ctx => {
          const pct = total ? (ctx.parsed / total * 100).toFixed(1) : 0;
          const row = pairs[ctx.dataIndex];
          const value = row.isOther || metricDef.signed ? row.displayValue : row.value;
          const prefix = metricDef.signed ? "绝对值" : metricDef.label;
          return ` ${row.label}: ${prefix} ${fmtChartValue(value, metricDef)} (${pct}%)`;
        }, title: items => {
          const row = pairs[items[0]?.dataIndex];
          return row?.label || "";
        }}},
      },
    },
  });
};

const drillDownChart = async row => {
  if (!row || row.isOther || !row.source || !selectedJobId.value) return;
  const fields = selectedJob.value?.result_files?.[row.source]?.fields || [];
  if (!fields.length) return;
  const state = {
    ...defaultResultViewState(),
    tableLimit: tableLimit.value || 100,
    tableOffset: 0,
    sortCol: fields.includes(row.metric) ? row.metric : "",
    sortAsc: false,
    colFilters: {},
    colFilterOps: {},
  };
  if (fields.includes(row.nameField)) {
    state.colFilters[row.nameField] = row.label;
    state.colFilterOps[row.nameField] = "~";
  } else {
    state.tableSearch = row.label;
  }
  const memory = readResultMemory(selectedJobId.value);
  memory.tabs = { ...(memory.tabs || {}), [row.source]: state };
  writeResultMemory(selectedJobId.value, memory);
  await activateCsvTab(row.source);
  showToast(`已跳转到 ${row.source} 并筛选: ${shortChartLabel(row.label, 36)}`, "success");
};

const buildChart = async () => {
  await nextTick();
  if (!ktChart.value || !selectedJob.value?.result_files) return;
  const sourceConfig = resolveChartSource();
  if (!sourceConfig) {
    chartError.value = "没有可用的图表数据";
    return;
  }
  const metricDef = resolveChartMetric(sourceConfig);
  if (!metricDef) {
    chartError.value = "当前数据源没有可绘制的指标";
    return;
  }
  chartLoading.value = true;
  chartError.value = "";
  if (!chartTables.value[sourceConfig.file]) {
    try {
      chartTables.value = {
        ...chartTables.value,
        [sourceConfig.file]: await fetchResultTable(sourceConfig.file, {
          limit: CHART_FETCH_LIMIT,
          offset: 0,
          ignoreViewState: true,
        }),
      };
    } catch (e) {
      chartError.value = e.message || "加载图表数据失败";
      chartLoading.value = false;
      return;
    }
  }
  const table = chartTables.value[sourceConfig.file];
  const fields = table?.fields || selectedJob.value.result_files[sourceConfig.file]?.fields || [];
  const rows = normalizeChartRows(table?.rows || [], fields, sourceConfig, metricDef);
  destroyChartInstances();

  chartSummaryCards.value = buildChartSummary(rows, table, sourceConfig);
  updateDeltaLists(rows);
  const topN = Math.max(1, Number(chartTopN.value) || 10);
  chartBarRows.value = sortChartRows(rows, metricDef).slice(0, topN);
  chartPieRows.value = buildPieRows(rows, metricDef, topN);

  if (!chartBarRows.value.length) {
    chartError.value = "当前指标没有可绘制的数据";
    chartLoading.value = false;
    return;
  }

  ktChart.value.parentElement.style.height =
    Math.max(280, Math.min(620, chartBarRows.value.length * 34 + 96)) + 'px';

  const cc = chartColors();
  const barColors = metricDef.signed
    ? chartBarRows.value.map(row => row.value >= 0 ? "rgba(239,68,68,0.76)" : "rgba(34,197,94,0.76)")
    : getColors(chartBarRows.value.length);
  ktChartInst.value = new Chart(ktChart.value, {
    type: "bar",
    data: {
      labels: chartBarRows.value.map(row => row.shortLabel),
      datasets: [{
        label: metricDef.label,
        data: chartBarRows.value.map(row => row.value),
        backgroundColor: barColors,
        borderRadius: 4,
        barThickness: 18,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      onClick: (_, elements) => {
        const row = chartBarRows.value[elements?.[0]?.index];
        if (row) drillDownChart(row);
      },
      onHover: (event, elements) => {
        if (event?.native?.target) event.native.target.style.cursor = elements.length ? "pointer" : "default";
      },
      plugins: {
        legend: { display: false },
        title: { display: true, text: `${sourceConfig.label} · ${metricDef.label}`, font: { size: 13 }, color: cc.title },
        tooltip: { callbacks: {
          title: items => chartBarRows.value[items[0]?.dataIndex]?.label || "",
          label: ctx => ` ${metricDef.label}: ${fmtChartValue(ctx.parsed.x, metricDef)}`,
          afterLabel: ctx => {
            const row = chartBarRows.value[ctx.dataIndex];
            if (!row || selectedJob.value?.mode !== "compare") return "";
            return [
              `A: ${fmtChartValue(row.aValue, { unit: "ms" })}`,
              `B: ${fmtChartValue(row.bValue, { unit: "ms" })}`,
              `count: ${trimNumber(row.countA)} -> ${trimNumber(row.countB)}`,
            ];
          },
        }},
      },
      scales: {
        x: { beginAtZero: true,
          ticks: { font: { size: 11 }, color: cc.text },
          grid:  { color: cc.grid } },
        y: { ticks: { font: { size: 11 }, color: cc.text },
          grid:  { color: cc.grid } },
      },
    },
  });

  ktPieChartInst.value = buildPie(
    ktPieChart.value,
    chartPieRows.value,
    metricDef.signed ? "TopN Delta 绝对值占比" : "TopN 占比",
    metricDef
  );
  chartLoading.value = false;
};

// ══════════════════════════════════════════════════════════════════════════════
// Uploads
// ══════════════════════════════════════════════════════════════════════════════

const setUploadFiles = files => {
  const picked = Array.from(files || []);
  if (!picked.length) return;
  uploadQueue.value = picked.map((file, index) => ({
    id: `${Date.now()}-${index}-${file.name}`,
    file,
    name: file.name,
    status: "ready",
    progress: 0,
    error: "",
    jobId: "",
  }));
  fileA.value = picked[0];
  fileAName.value = picked.length === 1 ? picked[0].name : `${picked.length} 个文件`;
};

const patchUploadQueueItem = (id, patch) => {
  uploadQueue.value = uploadQueue.value.map(item =>
    item.id === id ? { ...item, ...patch } : item
  );
};

const onFileChange = (e) => {
  setUploadFiles(e.target.files);
};

const onDrop = (e) => {
  setUploadFiles(e.dataTransfer.files);
};

const clearFile = () => {
  fileA.value = null;
  fileAName.value = "";
  uploadQueue.value = [];
};

const setQuickUploadMode = mode => {
  if (!["single", "compare"].includes(mode) || submitting.value) return;
  quickUploadMode.value = mode;
  localStorage.setItem("tpa-upload-mode", mode);
};

const setQuickCompareFile = (slot, files) => {
  const file = Array.from(files || [])[0];
  if (!file) return;
  if (slot === "a") {
    quickFileA.value = file;
    quickFileAName.value = file.name;
  } else {
    quickFileB.value = file;
    quickFileBName.value = file.name;
  }
  quickCompareStatus.value = "";
};

const onQuickFileChange = (slot, e) => {
  setQuickCompareFile(slot, e.target.files);
  e.target.value = "";
};

const onQuickDrop = (slot, e) => {
  setQuickCompareFile(slot, e.dataTransfer.files);
};

const clearQuickCompareFile = slot => {
  if (slot === "a") {
    quickFileA.value = null;
    quickFileAName.value = "";
  } else {
    quickFileB.value = null;
    quickFileBName.value = "";
  }
  quickCompareStatus.value = "";
};

const clearQuickCompareFiles = () => {
  quickFileA.value = null;
  quickFileB.value = null;
  quickFileAName.value = "";
  quickFileBName.value = "";
  quickCompareStatus.value = "";
};

// ══════════════════════════════════════════════════════════════════════════════
// Submit job
// ══════════════════════════════════════════════════════════════════════════════

const uploadSingleJob = (queueItem, index, total) => new Promise(resolve => {
  const fd = new FormData();
  const baseLabel = form.value.label.trim();
  const label = total === 1
    ? baseLabel
    : (baseLabel ? `${baseLabel} - ${queueItem.name}` : queueItem.name);
  fd.append("file_a", queueItem.file);
  fd.append("label", label);
  fd.append("project_id", form.value.projectId);
  fd.append("save_triton_csv", form.value.saveTritonCsv);
  fd.append("save_triton_code", form.value.saveTritonCode);

  const xhr = new XMLHttpRequest();
  xhr.upload.onprogress = e => {
    if (!e.lengthComputable) return;
    const progress = Math.round(e.loaded / e.total * 100);
    patchUploadQueueItem(queueItem.id, { status: "uploading", progress });
    uploadProgress.value = Math.round(((index + progress / 100) / total) * 100);
  };
  xhr.onload = () => {
    if (xhr.status < 200 || xhr.status >= 300) {
      let detail = "服务器错误";
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (e) {}
      patchUploadQueueItem(queueItem.id, { status: "error", error: detail });
      resolve(null);
      return;
    }
    const job = JSON.parse(xhr.responseText);
    patchUploadQueueItem(queueItem.id, { status: "submitted", progress: 100, jobId: job.id });
    resolve(job);
  };
  xhr.onerror = () => {
    patchUploadQueueItem(queueItem.id, { status: "error", error: "网络错误" });
    resolve(null);
  };
  xhr.open("POST", "/api/jobs");
  xhr.withCredentials = true;
  xhr.send(fd);
});

const submitJob = async () => {
  if (!uploadQueue.value.length || submitting.value) return;
  submitting.value = true;
  uploadProgress.value = 0;
  const queue = [...uploadQueue.value];
  let lastJob = null;
  let successCount = 0;
  for (let i = 0; i < queue.length; i += 1) {
    patchUploadQueueItem(queue[i].id, { status: "uploading", progress: 0, error: "" });
    const job = await uploadSingleJob(queue[i], i, queue.length);
    if (job) {
      lastJob = job;
      successCount += 1;
    }
  }
  submitting.value = false;
  uploadProgress.value = 0;
  await refreshSidebarData();
  sidebarTab.value = "jobs";
  if (successCount) {
    form.value.label = "";
    showToast(`已提交 ${successCount}/${queue.length} 个任务`, successCount === queue.length ? "success" : "info");
    if (successCount === queue.length) {
      clearFile();
    } else {
      const failedItems = uploadQueue.value.filter(item => item.status === "error");
      uploadQueue.value = failedItems;
      fileA.value = failedItems[0]?.file || null;
      fileAName.value = failedItems.length === 1
        ? failedItems[0].name
        : (failedItems.length ? `${failedItems.length} 个文件` : "");
    }
    if (lastJob) router.push({ path: `/job/${lastJob.id}` });
  } else {
    showToast("提交失败，请检查上传队列", "error");
  }
};

const submitQuickCompare = () => new Promise(resolve => {
  if (!quickFileA.value || !quickFileB.value || submitting.value) {
    resolve(null);
    return;
  }
  submitting.value = true;
  uploadProgress.value = 0;
  quickCompareStatus.value = "uploading";

  const fd = new FormData();
  fd.append("file_a", quickFileA.value);
  fd.append("file_b", quickFileB.value);
  fd.append("label", form.value.label.trim());
  fd.append("project_id", form.value.projectId);
  fd.append("save_triton_csv", false);
  fd.append("save_triton_code", false);

  const xhr = new XMLHttpRequest();
  xhr.upload.onprogress = e => {
    if (!e.lengthComputable) return;
    uploadProgress.value = Math.round(e.loaded / e.total * 100);
  };
  xhr.onload = async () => {
    submitting.value = false;
    uploadProgress.value = 0;
    if (xhr.status < 200 || xhr.status >= 300) {
      let detail = "服务器错误";
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (e) {}
      quickCompareStatus.value = "error";
      showToast("快速对比提交失败: " + detail, "error");
      resolve(null);
      return;
    }
    const job = JSON.parse(xhr.responseText);
    quickCompareStatus.value = "submitted";
    form.value.label = "";
    clearQuickCompareFiles();
    await refreshSidebarData();
    sidebarTab.value = "jobs";
    router.push({ path: `/job/${job.id}` });
    showToast("已提交快速对比任务", "success");
    resolve(job);
  };
  xhr.onerror = () => {
    submitting.value = false;
    uploadProgress.value = 0;
    quickCompareStatus.value = "error";
    showToast("快速对比提交失败: 网络错误", "error");
    resolve(null);
  };
  xhr.open("POST", "/api/jobs");
  xhr.withCredentials = true;
  xhr.send(fd);
});

// ══════════════════════════════════════════════════════════════════════════════
// Job actions
// ══════════════════════════════════════════════════════════════════════════════

const deleteJob = async () => {
  if (!selectedJobId.value) {
    showToast("未选中任务，无法删除", "error");
    return;
  }
  if (!await askConfirm("确定删除该任务及所有关联文件？", {
    title: "删除任务",
    confirmText: "删除",
    tone: "danger",
  })) return;
  try {
    const response = await fetch(`/api/jobs/${selectedJobId.value}`, {
      method: "DELETE", credentials: "include",
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      showToast("删除失败: " + (errorData.detail || errorData.message || "未知错误"), "error");
      return;
    }
    router.push({ path: "/" });
    await refreshSidebarData();
    showToast("任务已删除", "success");
  } catch (error) {
    showToast("删除出错: " + error.message, "error");
  }
};

const deleteFile = async slot => {
  let impact = { count: 0, dependent_compare_jobs: [] };
  try {
    const impactResp = await fetch(`/api/jobs/${selectedJobId.value}/files/${slot}/delete-impact`, {
      credentials: "include",
    });
    if (impactResp.ok) impact = await impactResp.json();
  } catch (e) {}
  const examples = (impact.dependent_compare_jobs || [])
    .slice(0, 3)
    .map(job => job.label || job.id.slice(0, 8))
    .join("、");
  const impactMessage = impact.count
    ? ` 当前有 ${impact.count} 个历史对比依赖它${examples ? `，例如：${examples}` : ""}。`
    : "";
  if (!await askConfirm(`确定删除原始 trace 文件？删除后该文件无法参与历史对比。${impactMessage}`, {
    title: "删除文件",
    confirmText: "删除",
    tone: "danger",
  })) return;
  const r = await fetch(`/api/jobs/${selectedJobId.value}/files/${slot}`, {
    method: "DELETE", credentials: "include",
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("删除文件失败: " + (err.detail || err.message || "未知错误"), "error");
    return;
  }
  await loadJob(selectedJobId.value);
  await refreshSidebarData();
  showToast("文件已删除", "success");
};

const editLabel = () => {
  renameJobName.value = selectedJob.value?.label || "";
  showRenameJob.value = true;
};

const confirmRenameJob = async () => {
  if (!selectedJobId.value) return;
  const r = await fetch(`/api/jobs/${selectedJobId.value}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ label: renameJobName.value }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("重命名失败: " + (err.detail || err.message || "未知错误"), "error");
    return;
  }
  showRenameJob.value = false;
  await loadJob(selectedJobId.value);
  await refreshSidebarData();
  showToast("任务已重命名", "success");
};

const moveProject = () => {
  moveProjectTarget.value = selectedJob.value?.project_id || "";
  showMoveProject.value = true;
};

const confirmMoveProject = async () => {
  if (!selectedJobId.value) {
    showToast("未选中任务", "error");
    return;
  }
  try {
    const r = await fetch(`/api/jobs/${selectedJobId.value}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ project_id: moveProjectTarget.value || null }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      showToast("移动项目失败: " + (err.detail || r.status), "error");
      return;
    }
  } catch (e) {
    showToast("移动项目失败: " + e.message, "error");
    return;
  }
  showMoveProject.value = false;
  await loadJob(selectedJobId.value);
  await refreshSidebarData();
  showToast("任务已移动", "success");
};

const toggleHistoryBulkMode = () => {
  historyBulkMode.value = !historyBulkMode.value;
  if (!historyBulkMode.value) historySelection.value = [];
};

const toggleHistorySelection = job => {
  const idx = historySelection.value.indexOf(job.id);
  if (idx >= 0) historySelection.value.splice(idx, 1);
  else historySelection.value.push(job.id);
};

const handleHistoryJobClick = job => {
  if (historyBulkMode.value) {
    toggleHistorySelection(job);
    return;
  }
  navigateToJob(job.id);
};

const toggleSelectLoadedHistoryJobs = () => {
  const loadedIds = loadedHistoryJobIds.value;
  const selected = new Set(historySelection.value);
  const allLoadedSelected = loadedIds.length > 0 && loadedIds.every(id => selected.has(id));
  if (allLoadedSelected) {
    historySelection.value = historySelection.value.filter(id => !loadedIds.includes(id));
    return;
  }
  historySelection.value = [...new Set([...historySelection.value, ...loadedIds])];
};

const clearHistorySelection = () => {
  historySelection.value = [];
};

const openBulkMoveProject = () => {
  if (!historySelection.value.length) return;
  bulkMoveProjectTarget.value = "";
  showBulkMoveProject.value = true;
};

const confirmBulkMoveProject = async () => {
  if (!historySelection.value.length) return;
  const r = await fetch("/api/jobs/bulk/project", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      job_ids: historySelection.value,
      project_id: bulkMoveProjectTarget.value || null,
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("批量移动失败: " + (err.detail || err.message || "未知错误"), "error");
    return;
  }
  showBulkMoveProject.value = false;
  const ids = [...historySelection.value];
  const moved = ids.length;
  historySelection.value = [];
  await refreshSidebarData();
  if (selectedJobId.value && ids.includes(selectedJobId.value)) await loadJob(selectedJobId.value);
  showToast(`已移动 ${moved} 个任务`, "success");
};

const bulkDeleteJobs = async () => {
  if (!historySelection.value.length) return;
  if (!await askConfirm(`确定删除选中的 ${historySelection.value.length} 个任务及其关联文件？`, {
    title: "批量删除任务",
    confirmText: "删除",
    tone: "danger",
  })) return;
  const ids = [...historySelection.value];
  const r = await fetch("/api/jobs/bulk/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ job_ids: ids }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("批量删除失败: " + (err.detail || err.message || "未知错误"), "error");
    return;
  }
  if (selectedJobId.value && ids.includes(selectedJobId.value)) router.push({ path: "/" });
  historySelection.value = [];
  await refreshSidebarData();
  showToast(`已删除 ${ids.length} 个任务`, "success");
};

const bulkDeleteFiles = async () => {
  if (!historySelection.value.length) return;
  await loadStorageSummary();
  const selected = storageSummary.value.jobs.filter(job => historySelection.value.includes(job.id));
  const affectedCompareCount = selected.reduce((sum, job) => sum + (job.used_by_compare_count || 0), 0);
  const impactMessage = affectedCompareCount
    ? ` 其中 ${affectedCompareCount} 个历史对比依赖这些源文件。`
    : "";
  if (!await askConfirm(`确定删除选中的 ${historySelection.value.length} 个任务的原始 trace 文件？${impactMessage}`, {
    title: "批量删除文件",
    confirmText: "删除",
    tone: "danger",
  })) return;
  const ids = [...historySelection.value];
  const r = await fetch("/api/jobs/bulk/delete-files", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ job_ids: ids }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("批量删除文件失败: " + (err.detail || err.message || "未知错误"), "error");
    return;
  }
  await refreshSidebarData();
  if (selectedJobId.value && ids.includes(selectedJobId.value)) await loadJob(selectedJobId.value);
  showToast(`已处理 ${ids.length} 个任务`, "success");
};

const openRenameModal = (project) => {
  if (!project?.id) return;
  renameProjectId.value = project.id;
  renameProjectName.value = project.name;
  showRenameProject.value = true;
};

const confirmRenameProject = async () => {
  const newName = renameProjectName.value.trim();
  if (!newName) return;
  const pid = renameProjectId.value;
  if (!pid) { showToast("项目ID无效", "error"); return; }
  const proj = projects.value.find(p => p.id === pid);
  if (proj) proj.name = newName;
  try {
    const r = await fetch(`/api/projects/${pid}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ name: newName }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || "更新失败");
    }
  } catch (e) {
    if (proj) await loadProjects();
    showToast("重命名失败: " + e.message, "error");
    return;
  }
  showRenameProject.value = false;
  await loadProjects();
  await refreshSidebarData();
  showToast("项目已重命名", "success");
};

const deleteProject = async (projectId) => {
  if (!await askConfirm("确定删除该项目？项目内的任务将同时被删除。删除后10天内可以找回。", {
    title: "删除项目",
    confirmText: "删除",
    tone: "danger",
  })) return;
  const r = await fetch(`/api/projects/${projectId}`, {
    method: "DELETE",
    credentials: "include",
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("删除失败: " + (err.detail || err.message || `HTTP ${r.status}`), "error");
    return;
  }
  filterProject.value = "";
  router.push({ path: "/" });
  await loadProjects();
  await refreshSidebarData();
  showToast("项目已删除，可在 10 天内找回", "success");
};

const setSort = col => {
  if (sortCol.value === col) sortAsc.value = !sortAsc.value;
  else { sortCol.value = col; sortAsc.value = true; }
};

const startResize = (field, e) => {
  e.preventDefault();
  e.stopPropagation();
  const th = e.target.closest('th');
  const startX = e.clientX;
  const startWidth = th.offsetWidth;
  const onMove = ev => {
    const w = Math.max(60, startWidth + ev.clientX - startX);
    colWidths.value = { ...colWidths.value, [field]: w };
  };
  const onUp = () => {
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
  };
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
};

const downloadCsv = filename => {
  const fields = displayedFields.value;
  const rows   = filteredRows.value;
  if (!fields.length) return;
  const escape = v => {
    const s = String(v ?? '');
    return s.includes(',') || s.includes('"') || s.includes('\n')
      ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const lines = [
    fields.map(escape).join(','),
    ...rows.map(r => fields.map(f => escape(r[f] ?? '')).join(',')),
  ];
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

const runSingleTriton = async (codePath) => {
  if (!allowCodeExecution.value) return;
  if (!selectedJobId.value || !codePath) return;
  tritonStatus.value = { ...tritonStatus.value, [codePath]: { status: 'running' } };
  try {
    const resp = await fetch(`/api/jobs/${selectedJobId.value}/run-triton-single`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code_path: codePath }),
    });
    const data = await resp.json();
    if (resp.ok && data.success) {
      let efficiency = "";
      const m = data.output.match(/([\d.]+)\s*GB\/s/);
      if (m) efficiency = m[1];
      tritonStatus.value = { ...tritonStatus.value, [codePath]: { status: 'success', value: efficiency, output: data.output.trim() } };
      errorModalTitle.value = "执行结果";
      errorModalMsg.value = data.output.trim();
      showErrorModal.value = true;
    } else {
      tritonStatus.value = { ...tritonStatus.value, [codePath]: { status: 'failed' } };
      const errMsg = data.detail || data.message || `HTTP ${resp.status}`;
      errorModalTitle.value = "错误信息";
      errorModalMsg.value = `执行失败: ${errMsg}`;
      showErrorModal.value = true;
    }
  } catch (e) {
    tritonStatus.value = { ...tritonStatus.value, [codePath]: { status: 'failed' } };
    errorModalMsg.value = "执行出错: " + e.message;
    showErrorModal.value = true;
  }
};

const runCustomTriton = async () => {
  if (!allowCodeExecution.value) return;
  if (!selectedJobId.value || !tritonCodeEditContent.value) return;
  customRunStatus.value = "running";
  try {
    const resp = await fetch(`/api/jobs/${selectedJobId.value}/run-triton-custom`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code_content: tritonCodeEditContent.value }),
    });
    const data = await resp.json();
    showTritonCode.value = false;
    if (resp.ok && data.success) {
      customRunStatus.value = "done";
      let efficiency = "";
      const m = data.output.match(/([\d.]+)\s*GB\/s/);
      if (m) efficiency = m[1];
      const codePath = currentTritonCodePath.value;
      if (codePath) {
        tritonStatus.value = { ...tritonStatus.value, [codePath]: { status: 'success', value: efficiency, output: data.output.trim(), custom: true } };
      }
      errorModalTitle.value = "执行结果";
      errorModalMsg.value = data.output.trim();
      showErrorModal.value = true;
    } else {
      customRunStatus.value = "failed";
      const errMsg = data.detail || data.message || `HTTP ${resp.status}`;
      const codePath = currentTritonCodePath.value;
      if (codePath) {
        tritonStatus.value = { ...tritonStatus.value, [codePath]: { status: 'failed' } };
      }
      errorModalTitle.value = "错误信息";
      errorModalMsg.value = `执行失败: ${errMsg}`;
      showErrorModal.value = true;
    }
  } catch (e) {
    showTritonCode.value = false;
    customRunStatus.value = "failed";
    const codePath = currentTritonCodePath.value;
    if (codePath) {
      tritonStatus.value = { ...tritonStatus.value, [codePath]: { status: 'failed' } };
    }
    errorModalMsg.value = "执行出错: " + e.message;
    showErrorModal.value = true;
  }
};

const editTritonCode = () => {
  if (!allowCodeExecution.value) return;
  tritonCodeEditContent.value = tritonCodeContent.value;
  tritonCodeEditing.value = true;
};

const cancelEditTritonCode = () => {
  tritonCodeEditing.value = false;
  tritonCodeEditContent.value = "";
};

const clearInductorCache = async () => {
  if (!allowCodeExecution.value) return;
  if (!selectedJobId.value) return;
  try {
    const resp = await fetch(`/api/jobs/${selectedJobId.value}/clear-inductor-cache`, {
      method: "POST",
      credentials: "include",
    });
    const data = await resp.json();
    if (resp.ok && data.success) {
      const count = data.removed ? data.removed.length : 0;
      errorModalTitle.value = "清除 Cache";
      errorModalMsg.value = `已清除 ${count} 个 torchinductor cache 目录`;
      showErrorModal.value = true;
    } else {
      const errMsg = data.detail || data.message || `HTTP ${resp.status}`;
      errorModalTitle.value = "错误信息";
      errorModalMsg.value = `清除失败: ${errMsg}`;
      showErrorModal.value = true;
    }
  } catch (e) {
    errorModalMsg.value = "清除出错: " + e.message;
    showErrorModal.value = true;
  }
};

const downloadTraceFile = (slot) => {
  if (!selectedJobId.value) return;
  const a = document.createElement('a');
  a.href = `/api/jobs/${selectedJobId.value}/files/${slot}`;
  a.click();
};

const escapeHtml = value => String(value ?? "")
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

const perfettoButtonLabel = (slot) => {
  const prefix = selectedJob.value?.mode === "compare" ? `Perfetto ${slot.toUpperCase()}` : "Perfetto";
  return perfettoOpening.value[slot] ? `${prefix} 打开中...` : `${prefix} ↗`;
};

const buildPerfettoUrl = (slot) => {
  const context = selectedJob.value?.perfetto_context?.[slot];
  if (!context) return 'https://ui.perfetto.dev';
  const params = new URLSearchParams({
    visStart: String(context.vis_start_ns),
    visEnd: String(context.vis_end_ns),
    ts: String(context.ts_ns),
    dur: String(context.dur_ns),
  });
  return `https://ui.perfetto.dev/#!/?${params}`;
};

const showPerfettoError = (message) => {
  errorModalTitle.value = "Perfetto";
  errorModalMsg.value = message;
  showErrorModal.value = true;
};

const renderPerfettoLoadingPage = (win, filename) => {
  try {
    win.document.open();
    win.document.write(`<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Preparing Perfetto</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
      color: #243047;
    }
    .panel {
      width: min(520px, calc(100vw - 48px));
      padding: 28px 30px;
      border-radius: 14px;
      background: #fff;
      box-shadow: 0 18px 60px rgba(40, 52, 92, .18);
      border: 1px solid #e2e8f3;
    }
    .eyebrow {
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .08em;
      color: #6c63ff;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    h1 { font-size: 22px; line-height: 1.25; margin: 0 0 10px; }
    .file {
      margin: 0 0 20px;
      font-size: 13px;
      color: #66728a;
      overflow-wrap: anywhere;
    }
    .status { font-size: 15px; font-weight: 650; margin-bottom: 6px; }
    .detail { min-height: 20px; font-size: 13px; color: #66728a; margin-bottom: 14px; }
    .track {
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #e7ebf4;
    }
    .bar {
      width: 20%;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #6c63ff, #20a4f3);
      transition: width .18s ease;
    }
    .track.indeterminate .bar {
      width: 38%;
      animation: slide 1.1s ease-in-out infinite;
    }
    @keyframes slide {
      0% { transform: translateX(-105%); }
      100% { transform: translateX(265%); }
    }
    @media (prefers-color-scheme: dark) {
      body { background: #111827; color: #e8edf7; }
      .panel { background: #182235; border-color: #2a3750; box-shadow: 0 18px 60px rgba(0, 0, 0, .35); }
      .file, .detail { color: #9aa6bd; }
      .track { background: #2a3750; }
    }
  </style>
</head>
<body>
  <main class="panel">
    <div class="eyebrow">Perfetto</div>
    <h1>正在准备 trace</h1>
    <p class="file">${escapeHtml(filename)}</p>
    <div id="status" class="status">正在启动...</div>
    <div id="detail" class="detail">请保持这个窗口打开</div>
    <div id="track" class="track indeterminate"><div id="bar" class="bar"></div></div>
  </main>
</body>
</html>`);
    win.document.close();
  } catch (e) {
    // The window may already have navigated cross-origin.
  }
};

const updatePerfettoLoadingPage = (win, status, detail = "", progress = null) => {
  try {
    if (!win || win.closed) return;
    const doc = win.document;
    const statusEl = doc.getElementById("status");
    const detailEl = doc.getElementById("detail");
    const trackEl = doc.getElementById("track");
    const barEl = doc.getElementById("bar");
    if (statusEl) statusEl.textContent = status;
    if (detailEl) detailEl.textContent = detail;
    if (trackEl && barEl) {
      if (progress === null) {
        trackEl.classList.add("indeterminate");
        barEl.style.width = "";
      } else {
        trackEl.classList.remove("indeterminate");
        barEl.style.width = `${Math.max(3, Math.min(100, progress))}%`;
      }
    }
  } catch (e) {
    // Ignore once Perfetto has taken over the popup.
  }
};

const readResponseArrayBuffer = async (resp, onProgress) => {
  const total = Number(resp.headers.get("content-length")) || 0;
  if (!resp.body?.getReader) {
    const buffer = await resp.arrayBuffer();
    onProgress(buffer.byteLength, total || buffer.byteLength);
    return buffer;
  }

  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  let lastUpdate = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    const now = performance.now();
    if (now - lastUpdate > 180) {
      onProgress(received, total);
      lastUpdate = now;
    }
  }
  onProgress(received, total || received);

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes.buffer;
};

const openInPerfetto = async (slot) => {
  const job = selectedJob.value;
  if (!job || perfettoOpening.value[slot]) return;
  const fname = (slot === 'a' ? job.file_a_name : job.file_b_name) || `trace_${slot}.json`;
  const PERFETTO = 'https://ui.perfetto.dev';

  perfettoOpening.value[slot] = true;
  const win = window.open("", "_blank");
  if (!win) {
    perfettoOpening.value[slot] = false;
    showPerfettoError('请允许浏览器弹出窗口后重试');
    return;
  }
  renderPerfettoLoadingPage(win, fname);

  const setPerfettoStatus = (status, detail = "", progress = null) => {
    updatePerfettoLoadingPage(win, status, detail, progress);
  };

  setPerfettoStatus("正在准备文件...", "正在解压并读取 trace 数据");

  let resp;
  try {
    resp = await fetch(`/api/jobs/${selectedJobId.value}/files/${slot}?format=json`, {
      credentials: "include",
    });
  } catch (e) {
    perfettoOpening.value[slot] = false;
    win.close();
    showPerfettoError('获取 trace 文件失败');
    return;
  }
  if (!resp.ok) {
    perfettoOpening.value[slot] = false;
    win.close();
    showPerfettoError("获取 trace 文件失败 (" + resp.status + ")");
    return;
  }

  let buffer;
  try {
    buffer = await readResponseArrayBuffer(resp, (loaded, total) => {
      if (total) {
        const pct = Math.round((loaded / total) * 100);
        setPerfettoStatus(`读取中 ${pct}%`, `${fmtBytes(loaded)} / ${fmtBytes(total)}`, pct);
      } else {
        setPerfettoStatus("读取中...", `已读取 ${fmtBytes(loaded)}`);
      }
    });
  } catch (e) {
    perfettoOpening.value[slot] = false;
    win.close();
    showPerfettoError('读取 trace 文件失败');
    return;
  }

  if (win.closed) {
    perfettoOpening.value[slot] = false;
    return;
  }

  setPerfettoStatus("正在打开 Perfetto...", "正在加载 Perfetto UI 并传输 trace", 100);

  let sent = false;
  let pingTimer = null;

  const cleanup = () => {
    window.removeEventListener('message', handler);
    if (pingTimer) clearInterval(pingTimer);
  };

  const handler = (e) => {
    if (e.origin !== PERFETTO || e.data !== 'PONG') return;
    if (sent || win.closed) return;
    sent = true;
    const message = { perfetto: { buffer, title: fname, fileName: fname } };
    try {
      win.postMessage(message, PERFETTO, [buffer]);
    } catch (err) {
      win.postMessage(message, PERFETTO);
    }
    perfettoOpening.value[slot] = false;
    cleanup();
  };

  window.addEventListener('message', handler);
  win.location.href = buildPerfettoUrl(slot);

  const ping = () => {
    if (sent || win.closed) {
      if (win.closed) {
        perfettoOpening.value[slot] = false;
      }
      cleanup();
      return;
    }
    win.postMessage('PING', PERFETTO);
  };

  ping();
  pingTimer = setInterval(ping, 500);
  setTimeout(() => {
    if (!sent) {
      cleanup();
      perfettoOpening.value[slot] = false;
      showPerfettoError('Perfetto 页面未响应，请稍后重试');
    }
  }, 30000);
};

const viewTritonCode = async (codePath) => {
  if (!selectedJobId.value || !codePath) return;
  currentTritonCodePath.value = codePath;
  const resp = await fetch(`/api/jobs/${selectedJobId.value}/triton-code/${codePath}`, { credentials: "include" });
  if (!resp.ok) { showToast("无法加载代码文件", "error"); return; }
  const data = await resp.json();
  tritonCodeContent.value = data.content;
  tritonCodeFilename.value = data.filename;
  showTritonCode.value = true;
  nextTick(() => {
    if (window.hljs) {
      document.querySelectorAll('pre.code-viewer code.language-python').forEach((block) => {
        window.hljs.highlightElement(block);
      });
    }
  });
};

const copyTritonCode = async () => {
  if (!tritonCodeContent.value) return;
  try {
    await navigator.clipboard.writeText(tritonCodeContent.value);
    showToast("已复制到剪贴板", "success");
  } catch (e) {
    const textarea = document.createElement("textarea");
    textarea.value = tritonCodeContent.value;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
    showToast("已复制到剪贴板", "success");
  }
};

const copyErrorModal = async () => {
  if (!errorModalMsg.value) return;
  try {
    await navigator.clipboard.writeText(errorModalMsg.value);
    showToast("已复制到剪贴板", "success");
  } catch (e) {
    const textarea = document.createElement("textarea");
    textarea.value = errorModalMsg.value;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
    showToast("已复制到剪贴板", "success");
  }
};

// ══════════════════════════════════════════════════════════════════════════════
// Compare
// ══════════════════════════════════════════════════════════════════════════════

const toggleCompareSelect = job => {
  if (!job.file_a_exists) return;
  const idx = compareSelection.value.indexOf(job.id);
  const details = { ...compareSelectionDetails.value };
  if (idx >= 0) {
    compareSelection.value.splice(idx, 1);
    delete details[job.id];
  } else if (compareSelection.value.length < 2) {
    compareSelection.value.push(job.id);
    details[job.id] = job;
  } else {
    delete details[compareSelection.value[0]];
    compareSelection.value = [compareSelection.value[1], job.id];
    details[job.id] = job;
  }
  compareSelectionDetails.value = details;
};

const removeCompareSelection = (id) => {
  compareSelection.value = compareSelection.value.filter(selectedId => selectedId !== id);
  const details = { ...compareSelectionDetails.value };
  delete details[id];
  compareSelectionDetails.value = details;
};

const submitCompare = async () => {
  const [a, b] = compareSelection.value;
  const r = await fetch("/api/jobs/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      job_id_a: a, job_id_b: b,
      label: compareLabel.value,
      project_id: compareProjectId.value || null,
    }),
  });
  const job = await r.json();
  if (!r.ok) {
    showToast("对比失败: " + (job.detail || "服务器错误"), "error");
    return;
  }
  compareSelection.value = [];
  compareSelectionDetails.value = {};
  compareLabel.value = "";
  sidebarTab.value = "jobs";
  await refreshSidebarData();
  router.push({ path: `/job/${job.id}` });
};

const openCompareSource = source => {
  if (!source?.id) return;
  router.push({ path: `/job/${source.id}` });
};

const rerunCompareSwapped = async () => {
  const sourceA = selectedJob.value?.compare_sources?.a;
  const sourceB = selectedJob.value?.compare_sources?.b;
  if (!sourceA?.id || !sourceB?.id) return;
  if (!sourceA.file_a_exists || !sourceB.file_a_exists) {
    showToast("源文件已删除，无法重新对比", "error");
    return;
  }
  const r = await fetch("/api/jobs/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      job_id_a: sourceB.id,
      job_id_b: sourceA.id,
      label: `${sourceB.label || sourceB.id.slice(0, 8)} vs ${sourceA.label || sourceA.id.slice(0, 8)}`,
      project_id: selectedJob.value.project_id || null,
    }),
  });
  const job = await r.json();
  if (!r.ok) {
    showToast("重新对比失败: " + (job.detail || "服务器错误"), "error");
    return;
  }
  await refreshSidebarData();
  router.push({ path: `/job/${job.id}` });
};

// ══════════════════════════════════════════════════════════════════════════════
// Projects
// ══════════════════════════════════════════════════════════════════════════════

const createProject = async () => {
  if (!newProjectName.value.trim()) return;
  const r = await fetch("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      name: newProjectName.value,
      description: newProjectDesc.value,
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("创建项目失败: " + (err.detail || err.message || "未知错误"), "error");
    return;
  }
  showNewProject.value = false;
  newProjectName.value = "";
  newProjectDesc.value = "";
  await loadProjects();
  await refreshSidebarData();
  showToast("项目已创建", "success");
};

const prevPage = () => {
  if (historyGroupsOffset.value > 0) {
    historyGroupsOffset.value = Math.max(0, historyGroupsOffset.value - historyGroupsLimit.value);
    loadHistoryGroups();
  }
};

const nextPage = () => {
  if (historyGroupsOffset.value + historyGroupsLimit.value < historyGroupsTotal.value) {
    historyGroupsOffset.value += historyGroupsLimit.value;
    loadHistoryGroups();
  }
};

const prevComparePage = () => {
  if (compareJobsOffset.value > 0) {
    compareJobsOffset.value = Math.max(0, compareJobsOffset.value - compareJobsLimit.value);
    loadCompareJobs();
  }
};

const nextComparePage = () => {
  if (compareJobsOffset.value + compareJobsLimit.value < compareJobsTotal.value) {
    compareJobsOffset.value += compareJobsLimit.value;
    loadCompareJobs();
  }
};

// ══════════════════════════════════════════════════════════════════════════════
// Console formatting
// ══════════════════════════════════════════════════════════════════════════════

const formatConsole = (text) => {
  if (!text) return '';
  return text.split('\n').map(line => {
    const e = line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    if (/^={3,}/.test(line))  return `<span class="co-hdr">${e}</span>`;
    if (/^-{5,}/.test(line))  return `<span class="co-sep">${e}</span>`;
    if (/^Wrote /.test(line)) return `<span class="co-wrote">${e}</span>`;
    if (/^\s*$/.test(line))   return e;
    const highlighted = e.replace(
      /(\b\d+\.?\d*%?|\+[\d.]+|[-][\d.]+)/g,
      '<span class="co-num">$1</span>'
    );
    return `<span class="co-line">${highlighted}</span>`;
  }).join('\n');
};

// ══════════════════════════════════════════════════════════════════════════════
// Navigation helper (used by sidebar click in index.html)
// ══════════════════════════════════════════════════════════════════════════════

const navigateToJob = (id) => router.push({ path: `/job/${id}` });

// ══════════════════════════════════════════════════════════════════════════════
// Route components
// ══════════════════════════════════════════════════════════════════════════════

const Home = {
  template: `
    <!-- Submit form -->
    <section class="card submit-card">
      <div class="submit-head">
        <div class="card-title">提交分析</div>
        <div class="upload-mode-toggle">
          <button :class="['mode-toggle-btn', quickUploadMode==='single'?'active':'']"
                  :disabled="submitting"
                  @click="setQuickUploadMode('single')">单文件/批量</button>
          <button :class="['mode-toggle-btn', quickUploadMode==='compare'?'active':'']"
                  :disabled="submitting"
                  @click="setQuickUploadMode('compare')">快速对比</button>
        </div>
      </div>

      <div v-if="quickUploadMode==='single'" class="submit-cols">
        <div class="upload-box upload-box-sm" @dragover.prevent @drop.prevent="onDrop">
          <input type="file" ref="fileInputA" accept=".json,.json.gz,.gz,.zip,.tar.gz,.tgz" multiple @change="onFileChange" hidden />
          <div @click="$refs.fileInputA.click()" class="upload-inner">
            <div class="upload-icon">📂</div>
            <div class="upload-label">{{ fileAName || '选择文件' }}</div>
          </div>
          <button v-if="fileAName" class="upload-clear" @click.stop="clearFile">✕</button>
        </div>
        <div class="form-row">
          <label>项目</label>
          <select v-model="form.projectId" class="input">
            <option value="">未分组</option>
            <option v-for="p in projects" :key="p.id" :value="p.id">{{ p.name }}</option>
          </select>
        </div>
        <div class="form-row">
          <label>备注</label>
          <input v-model="form.label" class="input" placeholder="可选" />
        </div>
        <button class="btn btn-primary" :disabled="uploadQueue.length===0 || submitting" @click="submitJob">
          {{ submitting ? '提交中 ' + uploadProgress + '%' : (uploadQueue.length > 1 ? '批量提交' : '提交分析') }}
        </button>
      </div>

      <div v-else class="quick-compare-cols">
        <div class="quick-upload-pair">
          <div class="upload-box upload-box-sm quick-upload-box" @dragover.prevent @drop.prevent="onQuickDrop('a', $event)">
            <input type="file" ref="quickFileInputA" accept=".json,.json.gz,.gz,.zip,.tar.gz,.tgz" @change="onQuickFileChange('a', $event)" hidden />
            <div @click="$refs.quickFileInputA.click()" class="upload-inner">
              <span class="trace-slot">A</span>
              <div class="upload-label">{{ quickFileAName || '选择 A trace' }}</div>
            </div>
            <button v-if="quickFileAName" class="upload-clear" @click.stop="clearQuickCompareFile('a')">✕</button>
          </div>
          <div class="upload-box upload-box-sm quick-upload-box" @dragover.prevent @drop.prevent="onQuickDrop('b', $event)">
            <input type="file" ref="quickFileInputB" accept=".json,.json.gz,.gz,.zip,.tar.gz,.tgz" @change="onQuickFileChange('b', $event)" hidden />
            <div @click="$refs.quickFileInputB.click()" class="upload-inner">
              <span class="trace-slot">B</span>
              <div class="upload-label">{{ quickFileBName || '选择 B trace' }}</div>
            </div>
            <button v-if="quickFileBName" class="upload-clear" @click.stop="clearQuickCompareFile('b')">✕</button>
          </div>
        </div>
        <div class="form-row">
          <label>项目</label>
          <select v-model="form.projectId" class="input">
            <option value="">未分组</option>
            <option v-for="p in projects" :key="p.id" :value="p.id">{{ p.name }}</option>
          </select>
        </div>
        <div class="form-row">
          <label>备注</label>
          <input v-model="form.label" class="input" placeholder="默认 A vs B" />
        </div>
        <button class="btn btn-primary" :disabled="!quickFileA || !quickFileB || submitting" @click="submitQuickCompare">
          {{ submitting ? '提交中 ' + uploadProgress + '%' : '提交对比' }}
        </button>
      </div>

      <div v-if="quickUploadMode==='single' && uploadQueue.length" class="upload-queue">
        <div v-for="item in uploadQueue" :key="item.id" class="upload-queue-item">
          <span class="upload-queue-name" :title="item.name">{{ item.name }}</span>
          <span :class="['upload-queue-status', 'queue-' + item.status]">
            <template v-if="item.status==='ready'">待提交</template>
            <template v-else-if="item.status==='uploading'">上传中 {{ item.progress }}%</template>
            <template v-else-if="item.status==='submitted'">已提交</template>
            <template v-else>{{ item.error || '失败' }}</template>
          </span>
        </div>
      </div>
      <div v-if="quickUploadMode==='compare' && (quickFileAName || quickFileBName)" class="quick-compare-summary">
        <span :class="['quick-file-chip', quickFileAName ? 'ready' : '']">A {{ quickFileAName || '未选择' }}</span>
        <span :class="['quick-file-chip', quickFileBName ? 'ready' : '']">B {{ quickFileBName || '未选择' }}</span>
      </div>
      <div v-if="submitting && uploadProgress < 100" class="upload-progress">
        <div class="upload-progress-label">总进度 {{ uploadProgress }}%</div>
        <div class="progress-track">
          <div class="progress-fill" :style="{ width: uploadProgress + '%' }"></div>
        </div>
      </div>
    </section>

    <!-- Empty state -->
    <div v-if="!selectedJob" class="empty-main">
      <div class="empty-main-icon">📊</div>
      <div class="empty-main-title">选择左侧历史记录查看结果，或上传文件开始分析</div>
      <div class="empty-main-tips">
        <div class="empty-tip-item">详细使用指南见右上角「使用指南」</div>
        <div class="empty-tip-item">建议上传的 trace 文件中开启了 triton code 保存功能</div>
      </div>
    </div>
  `,
  setup() {
    const fileInputA = ref(null);
    return {
      fileInputA, fileAName, fileA, quickUploadMode,
      quickFileA, quickFileB, quickFileAName, quickFileBName,
      uploadQueue, submitting, uploadProgress,
      form, projects, selectedJob,
      setQuickUploadMode,
      onDrop, onFileChange, clearFile, submitJob,
      onQuickDrop, onQuickFileChange, clearQuickCompareFile, submitQuickCompare,
    };
  },
};

const JobDetail = {
  template: `
    <!-- Loading state -->
    <div v-if="jobLoading && selectedJobId" class="empty-main">
      <div class="empty-main-icon">⟳</div>
      <div class="empty-main-title">加载任务...</div>
    </div>

    <!-- 404 state -->
    <div v-else-if="!selectedJob && selectedJobId" class="empty-main">
      <div class="empty-main-icon">🔍</div>
      <div class="empty-main-title">任务未找到</div>
    </div>

    <!-- Result panel -->
    <section v-if="!jobLoading && selectedJob" class="card result-card">
      <div class="result-header">
        <div>
          <span class="job-status lg" :class="'status-'+selectedJob.status">
            {{ statusIcon(selectedJob.status) }}
          </span>
          <span class="result-label">{{ selectedJob.label }}</span>
          <span class="job-mode-badge" :class="'mode-'+selectedJob.mode">
            {{ selectedJob.mode==='compare'?'对比':'单文件' }}
          </span>
        </div>
        <div class="result-actions">
          <button class="btn btn-sm btn-outline" @click="editLabel">重命名</button>
          <button class="btn btn-sm btn-outline" @click="moveProject">移动项目</button>
          <button class="btn btn-sm btn-danger" @click="deleteJob">删除任务</button>
        </div>
      </div>

      <!-- File info -->
      <div class="file-info">
        <div v-if="selectedJob.file_a_name" class="trace-file-row">
          <span v-if="selectedJob.mode==='compare'" class="trace-slot">A</span>
          <span class="trace-file-name" :title="selectedJob.file_a_name">📄 {{ selectedJob.file_a_name }}</span>
          <span v-if="!selectedJob.file_a_exists" class="tag-deleted">已删除</span>
          <div v-else class="trace-file-actions">
            <button v-if="allowFileDownload" class="btn btn-xs btn-outline" @click="downloadTraceFile('a')">下载</button>
            <button v-if="allowFileDownload" class="btn btn-xs btn-perfetto"
                    :disabled="perfettoOpening.a"
                    @click="openInPerfetto('a')">{{ perfettoButtonLabel('a') }}</button>
            <button class="btn btn-xs btn-danger" @click="deleteFile('a')">删除文件</button>
          </div>
        </div>
        <div v-if="selectedJob.file_b_name" class="trace-file-row">
          <span v-if="selectedJob.mode==='compare'" class="trace-slot">B</span>
          <span class="trace-file-name" :title="selectedJob.file_b_name">📄 {{ selectedJob.file_b_name }}</span>
          <span v-if="!selectedJob.file_b_exists" class="tag-deleted">已删除</span>
          <div v-else class="trace-file-actions">
            <button v-if="allowFileDownload" class="btn btn-xs btn-outline" @click="downloadTraceFile('b')">下载</button>
            <button v-if="allowFileDownload" class="btn btn-xs btn-perfetto"
                    :disabled="perfettoOpening.b"
                    @click="openInPerfetto('b')">{{ perfettoButtonLabel('b') }}</button>
            <button class="btn btn-xs btn-danger" @click="deleteFile('b')">删除文件</button>
          </div>
        </div>
      </div>

      <div v-if="selectedJob.mode==='compare' && selectedJob.compare_sources" class="compare-source-panel">
        <div class="compare-source-head">
          <span>来源</span>
          <button class="btn btn-xs btn-outline" @click="rerunCompareSwapped">交换 A/B 重新对比</button>
        </div>
        <div class="compare-source-grid">
          <div v-for="slot in ['a','b']" :key="slot" class="compare-source-item">
            <div class="compare-source-slot">{{ slot.toUpperCase() }}</div>
            <div class="compare-source-main">
              <div class="compare-source-title">{{ selectedJob.compare_sources[slot]?.label || '源任务缺失' }}</div>
              <div class="compare-source-meta">
                {{ selectedJob.compare_sources[slot]?.project_name || '未分组' }}
                · {{ fmtDate(selectedJob.compare_sources[slot]?.created_at) }}
              </div>
            </div>
            <span v-if="selectedJob.compare_sources[slot] && !selectedJob.compare_sources[slot].file_a_exists" class="tag-deleted">源文件已删除</span>
            <button v-if="selectedJob.compare_sources[slot]" class="btn btn-xs btn-outline"
                    @click="openCompareSource(selectedJob.compare_sources[slot])">查看源任务</button>
          </div>
        </div>
      </div>

      <div v-if="selectedJob.status==='running' || selectedJob.status==='pending'" class="loading">
        <span class="spinner"></span> 分析中...
      </div>
      <div v-else-if="selectedJob.status==='error'" class="error-box">
        {{ selectedJob.error_msg }}
      </div>

      <!-- Done: tabs -->
      <div v-else-if="selectedJob.status==='done'" :class="['result-body', isReadingMode ? 'reading-mode' : '']">
        <div class="result-tabs">
          <button v-for="t in availableTabs" :key="t.key"
                  :class="['tab', resultTab===t.key?'active':'', preparingResultTab===t.key?'preparing':'']"
                  :disabled="preparingResultTab===t.key"
                  @click="switchTab(t.key)">
            {{ t.label }}
          </button>
          <span class="result-tabs-spacer"></span>
          <button class="tab reading-toggle"
                  :title="isReadingMode ? '退出阅读模式' : '进入阅读模式'"
                  @click="toggleReadingMode">
            {{ isReadingMode ? '退出阅读' : '阅读模式' }}
          </button>
        </div>

        <!-- Console output -->
        <div v-if="resultTab==='console'" class="console-out">
          <div class="console-toolbar">
            <span class="console-dot dot-red"></span>
            <span class="console-dot dot-yellow"></span>
            <span class="console-dot dot-green"></span>
            <span class="console-title">Console Output</span>
          </div>
          <pre v-html="formatConsole(selectedJob.console_out)"></pre>
        </div>

        <!-- Chart tab -->
        <div v-if="resultTab==='chart'" class="chart-wrap">
          <div class="chart-head">
            <div class="chart-controls">
              <label>
                <span>数据源</span>
                <select v-model="chartSource" class="input input-sm" @change="buildChart">
                  <option v-for="source in chartSourceOptions" :key="source.file" :value="source.file">
                    {{ source.label }}
                  </option>
                </select>
              </label>
              <label>
                <span>指标</span>
                <select v-model="chartMetric" class="input input-sm" @change="buildChart">
                  <option v-for="metric in chartMetricOptions" :key="metric.key" :value="metric.key">
                    {{ metric.label }}
                  </option>
                </select>
              </label>
              <label>
                <span>Top</span>
                <select v-model.number="chartTopN" class="input input-sm chart-top-select" @change="buildChart">
                  <option v-for="n in chartTopNOptions" :key="n" :value="n">{{ n }}</option>
                </select>
              </label>
            </div>
            <span v-if="chartLoading" class="chart-loading"><span class="spinner-small"></span> 生成图表...</span>
          </div>

          <div v-if="chartError" class="error-box mb-2">{{ chartError }}</div>

          <div class="chart-summary-grid">
            <div v-for="card in chartSummaryCards" :key="card.label"
                 :class="['chart-summary-card', card.tone ? 'tone-' + card.tone : '', card.row ? 'clickable' : '']"
                 @click="card.row && drillDownChart(card.row)">
              <div class="chart-summary-label">{{ card.label }}</div>
              <div class="chart-summary-value">{{ card.value }}</div>
              <div class="chart-summary-sub">{{ card.sub }}</div>
            </div>
          </div>

          <div v-if="selectedJob.mode==='compare' && (chartSlowdowns.length || chartSpeedups.length)" class="chart-delta-grid">
            <div class="chart-delta-panel">
              <div class="chart-delta-title tone-neg">Top 回退</div>
              <button v-for="row in chartSlowdowns" :key="'slow-'+row.source+'-'+row.label"
                      class="chart-delta-row" @click="drillDownChart(row)">
                <span class="chart-delta-name">{{ row.label }}</span>
                <span class="chart-delta-value tone-neg">+{{ fmtDeltaMs(row.delta) }}</span>
              </button>
            </div>
            <div class="chart-delta-panel">
              <div class="chart-delta-title tone-pos">Top 改善</div>
              <button v-for="row in chartSpeedups" :key="'fast-'+row.source+'-'+row.label"
                      class="chart-delta-row" @click="drillDownChart(row)">
                <span class="chart-delta-name">{{ row.label }}</span>
                <span class="chart-delta-value tone-pos">{{ fmtDeltaMs(row.delta) }}</span>
              </button>
            </div>
          </div>

          <div class="chart-main-grid">
            <div class="chart-panel chart-panel-wide">
              <div class="chart-panel-title">排序排行（点击下钻）</div>
              <div class="chart-bar-area">
                <canvas ref="ktChart"></canvas>
              </div>
            </div>
            <div class="chart-panel">
              <div class="chart-panel-title">TopN + Other 占比</div>
              <div class="chart-pie-area">
                <canvas ref="ktPieChart"></canvas>
              </div>
            </div>
          </div>
        </div>

        <!-- CSV table tabs -->
        <div v-if="resultTab!=='console' && resultTab!=='chart'" class="table-wrap">
          <div class="table-toolbar">
            <input v-model="tableSearch" class="input input-sm" placeholder="全局搜索..." />
            <span v-if="hasColFilters" class="filter-active-tip">
              列筛选已启用
              <button class="btn-clear-filter" @click="clearColFilters()">✕ 清除</button>
            </span>
            <div class="column-menu-wrap">
              <button class="btn btn-sm btn-outline" @click="showColumnMenu=!showColumnMenu">
                列{{ hiddenColumnCount ? ' (' + hiddenColumnCount + ' 已隐藏)' : '' }}
              </button>
              <div v-if="showColumnMenu" class="column-menu">
                <div class="column-menu-actions">
                  <button class="btn btn-xs btn-outline" @click="resetVisibleColumns">全部列</button>
                  <button class="btn btn-xs btn-outline" @click="applyCoreColumnPreset">核心列</button>
                </div>
                <label v-for="f in currentTable.fields" :key="f" class="column-menu-item">
                  <input type="checkbox" :checked="isColumnVisible(f)" @change="toggleColumnVisibility(f)" />
                  <span>{{ f }}</span>
                </label>
              </div>
            </div>
            <button class="btn btn-sm btn-outline" @click="downloadCsv(resultTab)">下载当前页 CSV</button>
            <button v-if="isTritonStepTab && allowCodeExecution" class="btn btn-sm btn-outline" @click="clearInductorCache()">清除 Cache</button>
          </div>
          <div v-if="resultTableError" class="error-box mb-2">{{ resultTableError }}</div>
          <div class="table-scroll">
            <div v-if="resultTableLoading" class="table-loading">
              <span class="spinner-small"></span> 加载表格...
            </div>
            <div class="csv-table-wrap">
            <table class="data-table">
              <colgroup>
                <col v-for="f in displayedFields" :key="f"
                     :style="colWidths[f] ? { width: colWidths[f] + 'px' } : {}" />
              </colgroup>
              <thead>
                <tr>
                  <th v-for="f in displayedFields" :key="f"
                      @click="setSort(f)" class="th-sortable"
                      :style="colWidths[f] ? { width: colWidths[f] + 'px' } : {}">
                    <span class="th-label">{{ f }}</span>
                    <span v-if="sortCol===f" class="th-sort-icon">{{ sortAsc?'↑':'↓' }}</span>
                    <div class="col-resize-handle"
                         @mousedown.stop="startResize(f, $event)"
                         @click.stop></div>
                  </th>
                </tr>
                <tr class="filter-row">
                  <th v-for="f in displayedFields" :key="f"
                      :style="colWidths[f] ? { width: colWidths[f] + 'px' } : {}">
                    <div class="col-filter-wrap">
                      <select v-model="colFilterOps[f]" class="col-filter-op"
                              :class="{ 'op-active': colFilterOps[f] && colFilterOps[f] !== '~' }"
                              @click.stop>
                        <option value="~">包含</option>
                        <option value="!~">不包含</option>
                        <option value=">=">&gt;=</option>
                        <option value="<=">&lt;=</option>
                        <option value=">">&gt;</option>
                        <option value="<">&lt;</option>
                        <option value="=">=</option>
                      </select>
                      <input
                        v-model="colFilters[f]"
                        class="col-filter-input"
                        :class="{ active: colFilters[f] }"
                        :type="(colFilterOps[f] && ['~', '!~'].includes(colFilterOps[f])) ? 'text' : 'number'"
                        placeholder="筛选..."
                        @click.stop />
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row,i) in filteredRows" :key="i">
                  <td v-for="f in displayedFields" :key="f"
                      :class="deltaCellClass(f, row[f])"
                      :title="row[f]">
                    <template v-if="f === 'triton_code_file' && row[f]">
                      <button class="btn btn-xs btn-perfetto mb-1"
                              @click.stop="viewTritonCode(row[f])">
                        查看代码
                      </button>
                      <br />
                      <button v-if="allowCodeExecution && (!tritonStatus[row[f]] || tritonStatus[row[f]].status === 'idle' || tritonStatus[row[f]].status === 'failed')"
                              class="btn btn-xs btn-run"
                              @click.stop="runSingleTriton(row[f])">
                        运行
                      </button>
                      <span v-else-if="allowCodeExecution && tritonStatus[row[f]].status === 'running'"
                            class="status-running">运行中...</span>
                      <template v-else-if="allowCodeExecution && tritonStatus[row[f]].status === 'success'">
                        <span class="eff-value" :class="tritonStatus[row[f]].custom ? 'eff-custom' : 'btn-success'"
                              :title="'点击重新运行\\n' + tritonStatus[row[f]].output"
                              @click.stop="runSingleTriton(row[f])">
                          <span v-if="tritonStatus[row[f]].custom" class="custom-badge">✎</span>
                          {{ tritonStatus[row[f]].value }} GB/s ✓
                        </span>
                        <button class="btn btn-xs btn-outline ml-1"
                                @click.stop="runSingleTriton(row[f])"
                                :title="'点击重新运行'">↻</button>
                      </template>
                    </template>
                    <span v-else>{{ row[f] }}</span>
                  </td>
                </tr>
              </tbody>
              <tfoot v-if="filteredRows.length > 0">
                <tr class="sum-row">
                  <td v-for="(f, i) in displayedFields" :key="f" class="sum-cell">
                    <template v-if="i === 0">Σ 合计</template>
                    <template v-else-if="colSums[f] !== null">{{ fmtSum(colSums[f]) }}</template>
                  </td>
                </tr>
              </tfoot>
            </table>
            </div>
          </div>
          <div class="table-footer table-footer-paged">
            <span>第 {{ tablePageStart }}-{{ tablePageEnd }} 行 / 共 {{ tableTotalRows }} 行</span>
            <div class="table-pagination">
              <span class="page-size-control">
                每页
                <select class="input input-xs table-limit-select"
                        :value="tableLimit"
                        :disabled="resultTableLoading"
                        @change="changeTableLimit($event.target.value)">
                  <option v-for="n in tablePageSizeOptions" :key="n" :value="n">{{ n }}</option>
                  <option v-if="customTableLimit" :value="customTableLimit">全部 {{ customTableLimit }}</option>
                </select>
              </span>
              <button class="btn btn-xs btn-outline" @click="showAllTableRows" :disabled="!tableTotalRows || resultTableLoading || tableLimit >= tableTotalRows">全部</button>
              <button class="btn btn-xs btn-outline" @click="prevTablePage" :disabled="tableOffset===0 || resultTableLoading">上一页</button>
              <button class="btn btn-xs btn-outline" @click="nextTablePage" :disabled="tableOffset + tableLimit >= tableTotalRows || resultTableLoading">下一页</button>
            </div>
          </div>
        </div>
      </div>
    </section>
  `,
  setup() {
    const ktChartRef = ref(null);
    const ktPieChartRef = ref(null);
    const ktPieChartBRef = ref(null);

    // Wire up template refs to module-level refs so buildChart() can use them
    watch(ktChartRef, (el) => { ktChart.value = el; });
    watch(ktPieChartRef, (el) => { ktPieChart.value = el; });
    watch(ktPieChartBRef, (el) => { ktPieChartB.value = el; });

    const switchTab = async (key) => {
      if (key === resultTab.value) {
        if (preparingResultTab.value) cancelResultTableRequest();
        return;
      }
      if (preparingResultTab.value === key) return;
      if (key.endsWith(".csv")) {
        await activateCsvTab(key);
        return;
      }
      cancelResultTableRequest();
      router.push({ path: `/job/${selectedJobId.value}/${key}` });
    };

    return {
      ktChart: ktChartRef, ktPieChart: ktPieChartRef, ktPieChartB: ktPieChartBRef,
      selectedJob, selectedJobId, jobLoading, resultTab, availableTabs, currentTable,
      isReadingMode, toggleReadingMode,
      chartSource, chartMetric, chartTopN, chartTopNOptions, chartSourceOptions,
      chartMetricOptions, chartLoading, chartError, chartSummaryCards,
      chartSlowdowns, chartSpeedups, buildChart, drillDownChart, fmtDeltaMs,
      displayedFields, filteredRows, tableSearch, sortCol, sortAsc, colWidths, colFilters,
      colFilterOps, visibleColumns, showColumnMenu, hiddenColumnCount,
      tableLimit, tableOffset, tableTotalRows, tablePageStart, tablePageEnd,
      tablePageSizeOptions, customTableLimit, changeTableLimit, showAllTableRows,
      resultTableLoading, resultTableError, preparingResultTab, prevTablePage, nextTablePage,
      hasColFilters, colSums,
      isTritonStepTab, tritonStatus, allowFileDownload, allowCodeExecution,
      switchTab,
      statusIcon,
      editLabel, moveProject, deleteJob, deleteFile,
      openCompareSource, rerunCompareSwapped,
      downloadTraceFile, openInPerfetto, perfettoOpening, perfettoButtonLabel,
      setSort, startResize, downloadCsv,
      viewTritonCode, runSingleTriton, clearInductorCache,
      fmtDate, fmtSum, deltaCellClass, clearColFilters,
      isColumnVisible, resetVisibleColumns, applyCoreColumnPreset, toggleColumnVisibility,
      formatConsole,
    };
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Router definition
// ══════════════════════════════════════════════════════════════════════════════

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", component: Home },
    { path: "/job/:id", component: JobDetail },
    { path: "/job/:id/:tab", component: JobDetail },
  ],
});

// ══════════════════════════════════════════════════════════════════════════════
// Navigation guard
// ══════════════════════════════════════════════════════════════════════════════

router.beforeEach(async (to, from) => {
  // Ensure config/data is loaded on first navigation
  if (!appInitialized) {
    await loadConfig();
    await loadProjects();
    await refreshSidebarData();
    appInitialized = true;
  }

  const newJobId = to.params?.id || null;

  if (!newJobId) {
    // Navigated to home -- clean up
    saveResultViewState();
    isReadingMode.value = false;
    if (ktChartInst.value)     { ktChartInst.value.destroy();     ktChartInst.value = null; }
    if (ktPieChartInst.value)  { ktPieChartInst.value.destroy();  ktPieChartInst.value = null; }
    if (ktPieChartInstB.value) { ktPieChartInstB.value.destroy(); ktPieChartInstB.value = null; }
    clearInterval(pollTimer);
    pollTimer = null;
    cancelResultTableRequest();
    selectedJobId.value = null;
    selectedJob.value = null;
    jobLoading.value = false;
    resultTab.value = "console";
    resultTableFile.value = "";
    activeResultStateJobId = null;
    return;
  }

  const tab = to.params?.tab || resultTab.value || "console";

  // Same job, just switch tab
  if (newJobId === selectedJobId.value) {
    const validTabs = availableTabs.value.map(t => t.key);
    const targetTab = validTabs.includes(tab) ? tab : "console";
    if (targetTab !== resultTab.value) {
      if (targetTab.endsWith(".csv")) {
        await activateCsvTab(targetTab, { updateRoute: false });
      } else {
        cancelResultTableRequest();
        resultTab.value = targetTab;
      }
    }
    return;
  }

  // Different job -- full load
  saveResultViewState();
  isReadingMode.value = false;
  if (ktChartInst.value)     { ktChartInst.value.destroy();     ktChartInst.value = null; }
  if (ktPieChartInst.value)  { ktPieChartInst.value.destroy();  ktPieChartInst.value = null; }
  if (ktPieChartInstB.value) { ktPieChartInstB.value.destroy(); ktPieChartInstB.value = null; }
  cancelResultTableRequest();

  selectedJobId.value = newJobId;
  selectedJob.value = null;
  jobLoading.value = true;
  resultTableFile.value = "";
  const loaded = await loadJob(newJobId);

  if (!loaded) {
    selectedJobId.value = null;
    jobLoading.value = false;
    return { path: "/" };
  }

  const requestedTab = to.params?.tab || rememberedResultTab(newJobId);
  const validTabs = availableTabs.value.map(t => t.key);
  const targetTab = validTabs.includes(requestedTab) ? requestedTab : "console";
  activeResultStateJobId = newJobId;
  if (targetTab.endsWith(".csv")) {
    await activateCsvTab(targetTab, { updateRoute: false, savePrevious: false });
  } else {
    skipNextResultTabWatch();
    resultTab.value = targetTab;
    restoreResultViewState(newJobId, targetTab);
    rememberResultTabSelection(newJobId, targetTab);
  }
  if (targetTab === "chart" && selectedJob.value.status === "done") {
    nextTick(() => buildChart());
  }
  if (selectedJob.value.status === "pending" || selectedJob.value.status === "running") {
    startPoll();
  }
  sidebarTab.value = "jobs";
  jobLoading.value = false;
});

// ══════════════════════════════════════════════════════════════════════════════
// Root App component (wraps the #app DOM template in index.html)
// ══════════════════════════════════════════════════════════════════════════════

const App = {
  setup() {
    let historySearchTimer = null;
    let compareSearchTimer = null;
    let resultTableTimer = null;

    // Watchers that need to live at the root level
    watch(resultTab, (v, previousTab) => {
      if (suppressResultTabWatch) return;
      if (previousTab) saveResultViewState(activeResultStateJobId, previousTab);
      restoreResultViewState(selectedJobId.value, v);
      rememberResultTabSelection(selectedJobId.value, v);
      showColumnMenu.value = false;
      if (v?.endsWith(".csv")) loadResultTable();
      if (v === "chart" && selectedJob.value?.status === "done") {
        nextTick(() => buildChart());
      }
    });

    watch(selectedJob, v => {
      if (v?.status === "done") nextTick(() => {
        if (resultTab.value === "chart") buildChart();
      });
    }, { deep: true });

    watch(filterProject, () => {
      historyGroupsOffset.value = 0;
      compareJobsOffset.value = 0;
      historySelection.value = [];
      if (filterProject.value) {
        collapsedGroups.value[filterProject.value] = true;
      }
      localStorage.setItem("tpa-filter-project", filterProject.value);
      refreshSidebarData();
    });

    watch(historySearch, () => {
      clearTimeout(historySearchTimer);
      historySearchTimer = setTimeout(() => {
        const searching = Boolean(historySearch.value.trim());
        if (searching && preSearchExpandedGroups === null) {
          preSearchExpandedGroups = { ...collapsedGroups.value };
        }
        if (!searching && preSearchExpandedGroups !== null) {
          collapsedGroups.value = preSearchExpandedGroups;
          preSearchExpandedGroups = null;
        }
        historyGroupsOffset.value = 0;
        historySelection.value = [];
        loadHistoryGroups();
      }, 250);
    });

    watch(compareSearch, () => {
      clearTimeout(compareSearchTimer);
      compareSearchTimer = setTimeout(() => {
        compareJobsOffset.value = 0;
        loadCompareJobs();
      }, 250);
    });

    watch(compareSelection, () => {
      if (compareSelection.value.length === 2) {
        const [jobA, jobB] = selectedCompareJobs.value;
        if (jobA?.project_id && jobA.project_id === jobB?.project_id) {
          compareProjectId.value = jobA.project_id;
        } else {
          compareProjectId.value = "";
        }
      }
    });

    watch(sidebarWidth, value => localStorage.setItem("tpa-sidebar-width", String(value)));
    watch(sidebarCollapsed, value => localStorage.setItem("tpa-sidebar-collapsed", String(value)));
    watch(sidebarTab, value => localStorage.setItem("tpa-sidebar-tab", value));
    watch(isReadingMode, value => {
      document.body.classList.toggle("result-reading-active", value);
    });
    watch(collapsedGroups, value => {
      if (!historySearch.value.trim()) {
        localStorage.setItem("tpa-expanded-groups", JSON.stringify(value));
      }
    }, { deep: true });

    watch(
      [tableSearch, sortCol, sortAsc, tableLimit, tableOffset, colWidths, colFilters, colFilterOps, visibleColumns],
      () => {
        if (restoringResultState) return;
        saveResultViewState(activeResultStateJobId);
      },
      { deep: true },
    );

    watch([tableSearch, sortCol, sortAsc], () => {
      if (restoringResultState) return;
      if (!resultTab.value.endsWith(".csv")) return;
      clearTimeout(resultTableTimer);
      resultTableTimer = setTimeout(() => loadResultTable({ resetOffset: true }), 250);
    });

    watch([colFilters, colFilterOps], () => {
      if (restoringResultState) return;
      if (!resultTab.value.endsWith(".csv")) return;
      clearTimeout(resultTableTimer);
      resultTableTimer = setTimeout(() => loadResultTable({ resetOffset: true }), 250);
    }, { deep: true });

    watch(() => currentTable.value.fields, fields => {
      if (!fields.length || !visibleColumns.value.length) return;
      const valid = visibleColumns.value.filter(field => fields.includes(field));
      visibleColumns.value = valid.length ? valid : [...fields];
    });

    // Return everything the root template (index.html) needs
    return {
      // Layout/theme
      isDark, toggleTheme, sidebarWidth, sidebarCollapsed,
      toggleSidebar, startSidebarResize,

      // Sidebar data
      projects,
      historyGroupsTotal, historyGroupsLimit, historyGroupsOffset, historyGroupsLoading,
      historySearch, filterProject, sidebarTab, selectedJobId, selectedJob,
      collapsedGroups, groupedJobs, loadedHistoryJobIds,
      prevPage, nextPage, navigateToJob, loadHistoryGroupJobs,
      historyBulkMode, historySelection, toggleHistoryBulkMode,
      toggleSelectLoadedHistoryJobs, clearHistorySelection,
      handleHistoryJobClick, openBulkMoveProject, bulkDeleteFiles, bulkDeleteJobs,

      // Compare
      compareSelection, selectedCompareJobs, compareLabel, compareProjectId,
      compareJobs, compareJobsTotal, compareJobsLimit, compareJobsOffset, compareJobsLoading, compareSearch,
      toggleCompareSelect, removeCompareSelection, submitCompare,
      prevComparePage, nextComparePage,

      // Modals
      showNewProject, newProjectName, newProjectDesc,
      showRenameProject, renameProjectName, openRenameModal,
      confirmRenameProject, deleteProject,
      showMoveProject, moveProjectTarget, confirmMoveProject,
      showBulkMoveProject, bulkMoveProjectTarget, confirmBulkMoveProject,
      showRenameJob, renameJobName, confirmRenameJob,
      showDeletedProjects, deletedProjects, loadDeletedProjects,
      isDeletedOver10Days, restoreProject, permanentlyDeleteProject,
      showStorageManager, storageSummary, storageSelection, storageJobsWithTrace,
      openStorageManager, toggleStorageSelection, toggleAllStorageSelection,
      deleteSelectedStorageFiles, fmtBytes,
      showTritonCode, tritonCodeContent, tritonCodeFilename,
      tritonCodeEditing, tritonCodeEditContent,
      runCustomTriton, editTritonCode, cancelEditTritonCode,
      customRunStatus, allowCodeExecution,
      showGuide, showErrorModal, errorModalMsg, errorModalTitle,
      copyTritonCode, copyErrorModal,
      toasts, showConfirmModal, confirmModal, resolveConfirm,

      // Misc
      fmtDate, statusIcon, toggleGroup, createProject,
    };
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Bootstrap
// ══════════════════════════════════════════════════════════════════════════════

const app = createApp(App);
app.use(router);
app.mount("#app");
