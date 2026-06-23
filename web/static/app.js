const { createApp, ref, computed, watch, nextTick, onBeforeUnmount } = Vue;
const { createRouter, createWebHashHistory } = VueRouter;

// ══════════════════════════════════════════════════════════════════════════════
// Module-level reactive state (shared across all components via closure)
// ══════════════════════════════════════════════════════════════════════════════

let appInitialized = false;
const DEFAULT_RESULT_TAB = "chart";

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
    scheduleBuildChart();
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
const resultTab   = ref(DEFAULT_RESULT_TAB);
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
const consoleSearch = ref("");
const consoleHideWrote = ref(readStoredBool("tpa-console-hide-wrote", true));
const chartTables = ref({});
const colWidths     = ref({});
const colFilters    = ref({});
const colFilterOps  = ref({});
const visibleColumns = ref([]);
const showColumnMenu = ref(false);
const openActionMenu = ref("");
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
const claudeAnalysisEnabled = ref(false);
const appVersion = ref("0.2.103");
const authRequired = ref(false);
const authChecked = ref(false);
const authInitError = ref("");
const currentUser = ref(null);
const currentUserIsAdmin = ref(false);
const LOGIN_USERNAME_KEY = "tpa-login-username";
const LOGIN_REMEMBER_USERNAME_KEY = "tpa-login-remember-username";
const loginRememberUsername = ref(readStoredBool(LOGIN_REMEMBER_USERNAME_KEY, true));
const loginForm = ref({
  username: localStorage.getItem(LOGIN_USERNAME_KEY) || "",
  password: "",
  captcha: "",
});
const loginLoading = ref(false);
const loginError = ref("");
const loginCaptchaRequired = ref(false);
const loginCaptchaImage = ref("");
const perfettoOpening = ref({});
const compareRerunLoading = ref(false);
const singleTraceAnalyzeLoadingSlot = ref("");
const showStepReanalysisModal = ref(false);
const stepReanalysisLoading = ref(false);
const stepReanalysisLabel = ref("");
const stepReanalysisFilterA = ref("");
const stepReanalysisFilterB = ref("");
const aiAnalysisLoading = ref(false);
const aiAnalysisStarting = ref(false);
const aiAnalysisError = ref("");
const aiAnalysisContent = ref("");
const aiAnalysisArtifacts = ref([]);
const aiAnalysisVersions = ref([]);
const aiAnalysisSelectedVersionId = ref("");
const showAiPromptModal = ref(false);
const aiPromptForce = ref(false);
const aiAnalysisPrompt = ref("");
const aiArtifactsExpanded = ref(false);
const showAiCodeViewer = ref(false);
const aiCodeViewerLoading = ref(false);
const aiCodeViewerError = ref("");
const aiCodeViewerPath = ref("");
const aiCodeViewerFilename = ref("");
const aiCodeViewerContent = ref("");
const aiCodeViewerSize = ref(0);
const aiCodeViewerTruncated = ref(false);
const aiDiagnosticsLoading = ref(false);
const aiDiagnosticsError = ref("");
const aiDiagnosticsResult = ref(null);
const uiNow = ref(Date.now());
let activeResultStateJobId = null;
let aiAnalysisPollTimer = null;
let aiCompletionTitleResetTimer = null;
const defaultDocumentTitle = document.title || "torch profiler analyzer";
const isAdmin = computed(() => currentUserIsAdmin.value);

setInterval(() => {
  uiNow.value = Date.now();
}, 1000);

const toggleActionMenu = key => {
  openActionMenu.value = openActionMenu.value === key ? "" : key;
};
const closeActionMenu = () => {
  openActionMenu.value = "";
};
document.addEventListener("click", closeActionMenu);

const resultStateKey = jobId => `tpa-result-state:${jobId}`;
const readResultMemory = jobId =>
  jobId ? readStoredJson(resultStateKey(jobId), { lastTab: DEFAULT_RESULT_TAB, tabs: {} }) : { lastTab: DEFAULT_RESULT_TAB, tabs: {} };
const hasResultMemory = jobId => Boolean(jobId && localStorage.getItem(resultStateKey(jobId)) !== null);
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
const resolveResultTab = (jobId, requestedTab, validTabs) => {
  const fallback = validTabs.includes(DEFAULT_RESULT_TAB)
    ? DEFAULT_RESULT_TAB
    : (validTabs[0] || "console");
  if (requestedTab) return validTabs.includes(requestedTab) ? requestedTab : fallback;
  if (hasResultMemory(jobId)) {
    const remembered = readResultMemory(jobId).lastTab;
    if (validTabs.includes(remembered)) return remembered;
  }
  return fallback;
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
const refreshReadingLayout = () => {
  if (resultTab.value === "chart" && selectedJob.value?.status === "done") {
    scheduleBuildChart();
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
const newProjectShared = ref(false);

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
const showFeedbackBoard = ref(false);
const showFeedbackComposer = ref(false);
const feedbackItems = ref([]);
const feedbackTotal = ref(0);
const feedbackLimit = ref(30);
const feedbackOffset = ref(0);
const feedbackSort = ref("updated");
const feedbackLoading = ref(false);
const feedbackSubmitting = ref(false);
const feedbackForm = ref({ body: "", files: [], previews: [] });
const feedbackPostEditorMode = ref("write");
const feedbackReplies = ref({});
const feedbackEditing = ref({ id: "", body: "", saving: false });
const feedbackMarkdownEditorEnabled = ref(false);
const FEEDBACK_MARKDOWN_EDITOR_CSS = "https://cdn.jsdelivr.net/npm/easymde/dist/easymde.min.css";
const FEEDBACK_MARKDOWN_EDITOR_SCRIPTS = [
  "https://cdn.jsdelivr.net/npm/easymde/dist/easymde.min.js",
  "https://unpkg.com/easymde/dist/easymde.min.js",
];
const selectedFeedbackPostId = ref("");
const selectedFeedbackMessageId = ref("");
const feedbackDetailLoading = ref(false);
const feedbackEmailDiagLoading = ref(false);
const feedbackEmailDiagResult = ref(null);
const feedbackEmojiOptions = Object.freeze([
  "👍", "👎", "😄", "🎉", "🚀", "❤️", "👀", "💡", "✅", "🙏",
  "🔥", "🤔", "😕", "👏", "🙌", "💯", "🧠", "🛠️", "📌", "❓",
  "🙂", "😅", "😭", "✨", "💪", "📝", "🔍", "⚠️", "💬", "🙇",
]);
const feedbackReactionOptions = Object.freeze(["👍", "👎", "😄", "🎉", "🚀", "❤️", "👀", "💡"]);
const feedbackReactionPickerId = ref("");
const feedbackEmojiPickerTarget = ref("");
const feedbackTextTarget = ref({ target: "post", textarea: null });
const feedbackMention = ref({
  visible: false,
  loading: false,
  target: "",
  query: "",
  start: -1,
  end: -1,
  candidates: [],
  activeIndex: 0,
});
let feedbackMentionTimer = null;
let feedbackMentionSeq = 0;
let feedbackMentionTextarea = null;
const FEEDBACK_BODY_LIMIT = 2000;
const feedbackMarkdownEditors = new Map();
let feedbackMarkdownEditorLoadPromise = null;

const selectedFilterProject = computed(() =>
  projects.value.find(project => project.id === filterProject.value) || null
);

const projectOptionLabel = project =>
  `${project.name}${project.is_public ? " · 共享" : ""}`;
const showStorageManager = ref(false);
const storageSummary = ref({ totals: {}, projects: [], jobs: [] });
const storageSelection = ref([]);
const showAdminUsage = ref(false);
const adminUsageLoading = ref(false);
const adminUsageError = ref("");
const adminUsageDays = ref(14);
const adminUsage = ref({
  timezone: "",
  today: {},
  seven_days: {},
  all_time: {},
  daily: [],
  top_users_today: [],
});

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
const batchCompareMode  = ref(false);
const batchBaselineId   = ref("");
const batchCandidateIds = ref([]);
const batchSelectionDetails = ref({});
const batchCompareLabelPrefix = ref("");
const batchCompareLoading = ref(false);
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
const selectedBatchBaseline = computed(() =>
  batchBaselineId.value ? batchSelectionDetails.value[batchBaselineId.value] : null
);
const selectedBatchCandidates = computed(() =>
  batchCandidateIds.value
    .map(id => batchSelectionDetails.value[id])
    .filter(Boolean)
);

const availableTabs = computed(() => {
  const res = selectedJob.value?.result_files || selectedJob.value?.results;
  if (!res) return [];
  const tabs = [
    { key: "chart", label: "性能总览" },
    { key: "console", label: "控制台" },
  ];
  const primaryTypeTabs = {
    "kernel_types_avg.csv": "Kernel 类型",
    "kernel_types_cmp.csv": "类型对比",
  };
  for (const [file, label] of Object.entries(primaryTypeTabs)) {
    if (res[file]) tabs.push({ key: file, label });
  }
  const aiMeta = selectedJob.value?.ai_analysis || {};
  if (claudeAnalysisEnabled.value || aiMeta.report_exists || ["running", "done", "error"].includes(aiMeta.status)) {
    tabs.push({ key: "ai", label: "AI 分析" });
  }
  const csvMap = {
    "all_kernels_avg.csv":      "所有 Kernel",
    "all_kernels_cmp.csv":      "Kernel 对比",
    "triton_kernels_avg.csv":   "Triton",
    "triton_kernels_cmp.csv":   "Triton 对比",
    "aten_ops_avg.csv":         "Aten Ops",
    "aten_ops_cmp.csv":         "Aten 对比",
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

const jobStepFilterLabel = computed(() => {
  const job = selectedJob.value;
  if (!job) return "";
  const a = (job.step_filter_a || "").trim();
  const b = (job.step_filter_b || "").trim();
  if (!a && !b) return "";
  if (job.mode === "compare") return `Step A: ${a || "全部"} / B: ${b || "全部"}`;
  return `Step: ${a}`;
});

const CHART_SOURCE_CONFIGS = [
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

const CHART_COMMUNICATION_SOURCE_FILES = new Set([
  "cncl_ops_avg.csv",
  "cncl_ops_cmp.csv",
]);
const CHART_COMMUNICATION_FAMILIES = new Set([
  "collective",
  "communication",
  "comm",
  "cncl",
  "nccl",
]);
const CHART_COMMUNICATION_NAME_RE =
  /(^|[_:\s./\\-])(tcdp|cncl|nccl|tccl|hccl|mpi)(?=$|[_:\s./\\-])|all[_-]?reduce|all[_-]?gather|all[_-]?to[_-]?all|allconnected|reduce[_-]?scatter|reducescatter|sendrecv|(^|[_:\s./\\-])i?(send|recv)(?=$|[_:\s./\\-])/i;

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
  return CHART_SOURCE_CONFIGS.filter(item =>
    item.mode === mode
      && res[item.file]
      && !CHART_COMMUNICATION_SOURCE_FILES.has(item.file)
  );
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

const isKernelTypeTab = computed(() =>
  ["kernel_types_avg.csv", "kernel_types_cmp.csv"].includes(resultTab.value)
);

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

const adminUsageCards = computed(() => {
  const today = adminUsage.value.today || {};
  const seven = adminUsage.value.seven_days || {};
  return [
    { label: "今日日活", value: today.dau || 0, hint: today.day || "今天" },
    { label: "今日请求", value: today.requests || 0, hint: `时区 ${adminUsage.value.timezone || "-"}` },
    { label: "近 7 日活跃", value: seven.active_users || 0, hint: `${fmtCount(seven.requests || 0)} 次请求` },
    {
      label: "近 7 日任务",
      value: (seven.upload_jobs || 0) + (seven.compare_jobs || 0),
      hint: `AI ${fmtCount(seven.ai_runs || 0)} · 留言 ${fmtCount(seven.feedback_messages || 0)}`,
    },
  ];
});

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

const fmtCount = value => Number(value || 0).toLocaleString("zh-CN");

const traceFormatLabel = filename => {
  const name = String(filename || "").toLowerCase();
  if (name.endsWith(".json.gz")) return "json.gz";
  if (name.endsWith(".json.zip")) return "json.zip";
  if (name.endsWith(".tar.gz")) return "tar.gz";
  if (name.endsWith(".tgz")) return "tgz";
  if (name.endsWith(".zip")) return "zip";
  if (name.endsWith(".gz")) return "gz";
  if (name.endsWith(".json")) return "json";
  return "trace";
};

const uploadFileMeta = file =>
  file ? `${traceFormatLabel(file.name)} · ${fmtBytes(file.size)}` : "";

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
const fmtDateTime = iso => {
  if (!iso) return "";
  const date = new Date(iso);
  if (!Number.isNaN(date.getTime())) {
    const pad = value => String(value).padStart(2, "0");
    return [
      date.getFullYear(),
      pad(date.getMonth() + 1),
      pad(date.getDate()),
    ].join("-") + ` ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  }
  return iso.replace("T", " ").slice(0, 19);
};

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

class ApiRequestError extends Error {
  constructor(message, { status = 0, authExpired = false } = {}) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.authExpired = authExpired;
  }
}

const readJsonResponse = async (response, fallback = {}) => {
  const text = await response.text();
  if (!text) return fallback;
  try {
    return JSON.parse(text);
  } catch {
    throw new ApiRequestError("服务返回了非 JSON 响应，请稍后重试或刷新页面", {
      status: response.status,
    });
  }
};

const apiErrorMessage = (response, payload, fallback) =>
  payload?.detail || payload?.message || `${fallback}: HTTP ${response.status}`;

const fetchJson = async (url, options = {}, fallback = "请求失败") => {
  const response = await fetch(url, options);
  const payload = await readJsonResponse(response, {});
  if (!response.ok) {
    throw new ApiRequestError(apiErrorMessage(response, payload, fallback), {
      status: response.status,
      authExpired: response.status === 401,
    });
  }
  return payload;
};

const handleAuthExpired = () => {
  if (!authRequired.value) return;
  currentUser.value = null;
  currentUserIsAdmin.value = false;
  authChecked.value = true;
  clearInterval(pollTimer);
  pollTimer = null;
  stopAiAnalysisPolling();
  cancelResultTableRequest();
  clearAiDiagnostics();
};

const normalizeApiError = (error, fallback = "请求失败") => {
  if (error?.name === "AbortError") return fallback;
  if (error?.authExpired) {
    handleAuthExpired();
    return "登录已过期，请重新登录";
  }
  return error?.message || fallback;
};

const loadConfig = async () => {
  const cfg = await fetchJson("/api/config", { credentials: "include" }, "加载配置失败");
  appVersion.value = cfg.version || "0.2.103";
  authRequired.value = Boolean(cfg.auth_required);
  allowFileDownload.value = cfg.allow_file_download ?? true;
  allowCodeExecution.value = cfg.allow_code_execution ?? false;
  claudeAnalysisEnabled.value = cfg.claude_analysis_enabled ?? false;
};

const loadMe = async () => {
  const r = await fetch("/api/me", { credentials: "include" });
  const data = await readJsonResponse(r, {});
  if (r.status === 401) {
    currentUser.value = null;
    currentUserIsAdmin.value = false;
    authChecked.value = true;
    return null;
  }
  if (!r.ok) {
    throw new ApiRequestError(apiErrorMessage(r, data, "检查登录状态失败"), {
      status: r.status,
    });
  }
  currentUser.value = data.authenticated ? data.user : null;
  currentUserIsAdmin.value = Boolean(data.is_admin);
  authChecked.value = true;
  return currentUser.value;
};

const applyLoginCaptcha = payload => {
  loginCaptchaRequired.value = Boolean(payload?.captcha_required);
  loginCaptchaImage.value = payload?.captcha_image || "";
  if (loginCaptchaRequired.value) {
    loginForm.value.captcha = "";
  }
};

const refreshLoginCaptcha = async () => {
  if (!authRequired.value) return;
  const username = loginForm.value.username.trim();
  const r = await fetch(`/api/login-captcha?username=${encodeURIComponent(username)}`, {
    credentials: "include",
  });
  const payload = await r.json().catch(() => ({}));
  applyLoginCaptcha(payload);
};

const submitLogin = async () => {
  loginLoading.value = true;
  loginError.value = "";
  try {
    const r = await fetch("/api/login", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(loginForm.value),
    });
    if (!r.ok) {
      const payload = await r.json().catch(() => ({}));
      applyLoginCaptcha(payload);
      throw new Error(payload.detail || "登录失败");
    }
    const data = await r.json();
    const username = loginForm.value.username.trim();
    localStorage.setItem(LOGIN_REMEMBER_USERNAME_KEY, String(loginRememberUsername.value));
    if (loginRememberUsername.value && username) {
      localStorage.setItem(LOGIN_USERNAME_KEY, username);
    } else {
      localStorage.removeItem(LOGIN_USERNAME_KEY);
    }
    currentUser.value = data.user || null;
    currentUserIsAdmin.value = Boolean(data.is_admin);
    window.setTimeout(() => {
      loginForm.value.password = "";
      loginForm.value.captcha = "";
    }, 300);
    loginCaptchaRequired.value = false;
    loginCaptchaImage.value = "";
    appInitialized = true;
    await loadProjects();
    await refreshSidebarData();
    await resumeCurrentRouteAfterLogin();
  } catch (e) {
    loginError.value = e.message || "登录失败";
  } finally {
    loginLoading.value = false;
  }
};

const logout = async () => {
  await fetch("/api/logout", { method: "POST", credentials: "include" }).catch(() => {});
  currentUser.value = null;
  currentUserIsAdmin.value = false;
  appInitialized = false;
  projects.value = [];
  historyGroups.value = [];
  compareJobs.value = [];
  selectedJobId.value = null;
  selectedJob.value = null;
  clearAiDiagnostics();
  router.push({ path: "/" });
};

const loadProjects = async () => {
  try {
    projects.value = await fetchJson("/api/projects", { credentials: "include" }, "加载项目失败");
    if (
      filterProject.value &&
      filterProject.value !== "__none__" &&
      !projects.value.some(project => project.id === filterProject.value)
    ) {
      filterProject.value = "";
    }
  } catch (e) {
    const message = normalizeApiError(e, "加载项目失败");
    console.error("loadProjects error:", e);
    if (e?.authExpired) showToast(message, "error");
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

const loadAdminUsage = async () => {
  adminUsageLoading.value = true;
  adminUsageError.value = "";
  try {
    const days = Math.max(1, Math.min(Number(adminUsageDays.value) || 14, 90));
    adminUsageDays.value = days;
    const r = await fetch(`/api/admin/usage?days=${days}`, { credentials: "include" });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    adminUsage.value = await r.json();
  } catch (e) {
    adminUsageError.value = e.message || "加载使用统计失败";
    showToast(adminUsageError.value, "error");
  } finally {
    adminUsageLoading.value = false;
  }
};

const openAdminUsage = async () => {
  showAdminUsage.value = true;
  await loadAdminUsage();
};

const feedbackHasMore = computed(() => feedbackItems.value.length < feedbackTotal.value);
const selectedFeedbackPost = computed(() =>
  feedbackItems.value.find(item => item.id === selectedFeedbackPostId.value) || null
);
const feedbackSortOptions = [
  { key: "updated", label: "最新更新" },
  { key: "created", label: "发布时间" },
  { key: "hot", label: "热度" },
];

const feedbackPostTitle = item => {
  const text = (item?.body || "").trim();
  if (!text) return (item?.attachments || []).length ? "图片帖子" : "无标题帖子";
  const firstLine = text.split(/\r?\n/).map(line => line.trim()).find(Boolean) || text;
  return firstLine.length > 58 ? `${firstLine.slice(0, 58)}...` : firstLine;
};

const feedbackPostExcerpt = item => {
  const text = (item?.body || "").replace(/\s+/g, " ").trim();
  if (!text) return (item?.attachments || []).length ? "包含图片附件" : "暂无正文";
  return text.length > 110 ? `${text.slice(0, 110)}...` : text;
};

const feedbackPostReplyCount = item => Number(item?.reply_count ?? item?.replies?.length ?? 0);
const feedbackPostActivity = item => item?.last_activity_at || item?.updated_at || item?.created_at || "";
const feedbackMentionTargetKey = target => String(target || "post");
const feedbackUserInitial = value => {
  const text = String(value || "用户").trim();
  const chars = Array.from(text);
  const chineseChars = chars.filter(ch => /[\u4e00-\u9fff]/.test(ch));
  if (chineseChars.length >= 2) return chineseChars.slice(-2).join("");
  if (chineseChars.length === 1) return chineseChars[0];
  const compact = chars.filter(ch => !/\s/.test(ch));
  return compact.slice(-2).join("") || "用户";
};

const currentFeedbackUserToken = computed(() =>
  currentUser.value?.username || (!authRequired.value ? "local" : "")
);

const canEditFeedbackMessage = message => {
  if (!message?.id) return false;
  if (isAdmin.value) return true;
  return Boolean(message.user_token && message.user_token === currentFeedbackUserToken.value);
};

const feedbackEditedText = message => {
  if (!message?.edited_at) return "";
  const count = Number(message.edit_count || 0);
  return count > 1 ? `已编辑 ${count} 次` : "已编辑";
};

const feedbackReactionSummary = message => (message?.reactions || []).filter(item => Number(item.count || 0) > 0);
const feedbackReactionItem = (message, emoji) =>
  (message?.reactions || []).find(item => item.emoji === emoji) || { emoji, count: 0, reacted: false };

const toggleFeedbackReactionPicker = messageId => {
  feedbackReactionPickerId.value = feedbackReactionPickerId.value === messageId ? "" : messageId;
  feedbackEmojiPickerTarget.value = "";
  closeFeedbackMention();
};

const toggleFeedbackEmojiPicker = (target = "post") => {
  const targetKey = feedbackMentionTargetKey(target);
  feedbackEmojiPickerTarget.value = feedbackEmojiPickerTarget.value === targetKey ? "" : targetKey;
  feedbackReactionPickerId.value = "";
  closeFeedbackMention();
};

const updateFeedbackMessageInState = updated => {
  if (!updated?.id) return;
  feedbackItems.value = feedbackItems.value.map(post => {
    if (post.id === updated.id) {
      return {
        ...post,
        ...updated,
        replies: post.replies || updated.replies || [],
        reply_count: post.reply_count ?? updated.reply_count ?? 0,
      };
    }
    const replies = (post.replies || []).map(reply => (
      reply.id === updated.id ? { ...reply, ...updated } : reply
    ));
    const changed = replies.some((reply, index) => reply !== (post.replies || [])[index]);
    if (!changed) return post;
    return {
      ...post,
      replies,
      last_activity_at: updated.updated_at || post.last_activity_at,
    };
  });
};

const setFeedbackMessageReactions = (messageId, reactions) => {
  updateFeedbackMessageInState({ id: messageId, reactions: reactions || [] });
};

const setFeedbackTextTarget = (event, target = "post") => {
  feedbackTextTarget.value = {
    target: feedbackMentionTargetKey(target),
    textarea: event?.target || null,
  };
};

const feedbackTargetForm = targetKey => {
  if (targetKey === "post") return feedbackForm.value;
  if (targetKey.startsWith("edit:")) return feedbackEditing.value;
  return ensureFeedbackReplyForm(targetKey);
};

const feedbackMarkdownEditorElementId = target => {
  const targetKey = feedbackMentionTargetKey(target);
  if (targetKey === "post") return "feedback-compose-post-textarea";
  if (targetKey.startsWith("edit:")) return `feedback-edit-${targetKey.slice(5)}`;
  return `feedback-reply-editor-${targetKey}`;
};

const detectFeedbackMarkdownEditor = () => {
  const ctor = typeof window !== "undefined" && typeof window.EasyMDE === "function"
    ? window.EasyMDE
    : null;
  feedbackMarkdownEditorEnabled.value = Boolean(ctor);
  return ctor;
};

const ensureFeedbackMarkdownEditorCss = () => {
  if (typeof document === "undefined") return;
  const exists = Array.from(document.querySelectorAll("link[rel='stylesheet']"))
    .some(link => link.href === FEEDBACK_MARKDOWN_EDITOR_CSS);
  if (exists) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = FEEDBACK_MARKDOWN_EDITOR_CSS;
  document.head.appendChild(link);
};

const loadFeedbackMarkdownScript = src => new Promise(resolve => {
  if (typeof document === "undefined") {
    resolve(false);
    return;
  }
  const script = document.createElement("script");
  script.src = src;
  script.async = true;
  script.onload = () => resolve(Boolean(detectFeedbackMarkdownEditor()));
  script.onerror = () => resolve(false);
  document.head.appendChild(script);
});

const loadFeedbackMarkdownEditor = () => {
  if (detectFeedbackMarkdownEditor()) return Promise.resolve(true);
  if (feedbackMarkdownEditorLoadPromise) return feedbackMarkdownEditorLoadPromise;
  ensureFeedbackMarkdownEditorCss();
  feedbackMarkdownEditorLoadPromise = (async () => {
    for (const src of FEEDBACK_MARKDOWN_EDITOR_SCRIPTS) {
      const loaded = await loadFeedbackMarkdownScript(src);
      if (loaded || detectFeedbackMarkdownEditor()) return true;
    }
    return Boolean(detectFeedbackMarkdownEditor());
  })().finally(() => {
    if (!feedbackMarkdownEditorEnabled.value) feedbackMarkdownEditorLoadPromise = null;
  });
  return feedbackMarkdownEditorLoadPromise;
};

const destroyFeedbackMarkdownEditor = target => {
  const targetKey = feedbackMentionTargetKey(target);
  const entry = feedbackMarkdownEditors.get(targetKey);
  if (!entry) return;
  try {
    entry.instance?.toTextArea?.();
  } catch (e) {
    console.warn("destroy feedback markdown editor failed:", e);
  }
  feedbackMarkdownEditors.delete(targetKey);
};

const destroyFeedbackMarkdownEditors = () => {
  for (const targetKey of [...feedbackMarkdownEditors.keys()]) {
    destroyFeedbackMarkdownEditor(targetKey);
  }
};

const syncFeedbackMarkdownEditor = (target, body) => {
  const targetKey = feedbackMentionTargetKey(target);
  const entry = feedbackMarkdownEditors.get(targetKey);
  if (!entry?.instance) return;
  const nextBody = String(body || "");
  if (entry.instance.value() === nextBody) return;
  entry.silent = true;
  entry.instance.value(nextBody);
  entry.instance.codemirror?.refresh();
  entry.silent = false;
};

const handleFeedbackMarkdownMention = (target, editor) => {
  const cm = editor?.codemirror;
  if (!cm) return;
  const targetKey = feedbackMentionTargetKey(target);
  const text = editor.value() || "";
  const cursor = cm.indexFromPos(cm.getCursor());
  const detected = detectFeedbackMention(text, cursor);
  if (!detected) {
    if (feedbackMention.value.target === targetKey) closeFeedbackMention();
    return;
  }
  feedbackMentionTextarea = null;
  feedbackTextTarget.value = { target: targetKey, textarea: null };
  feedbackMention.value = {
    visible: true,
    loading: true,
    target: targetKey,
    query: detected.query,
    start: detected.start,
    end: detected.end,
    candidates: [],
    activeIndex: 0,
  };
  scheduleFeedbackMentionFetch(detected.query, targetKey);
};

const initFeedbackMarkdownEditor = target => {
  const MarkdownEditor = detectFeedbackMarkdownEditor();
  if (!MarkdownEditor) return;
  const targetKey = feedbackMentionTargetKey(target);
  const isEditTarget = targetKey.startsWith("edit:");
  const isReplyTarget = targetKey !== "post" && !isEditTarget;
  const editorId = feedbackMarkdownEditorElementId(targetKey);
  const existing = feedbackMarkdownEditors.get(targetKey);
  const existingWrapper = existing?.instance?.codemirror?.getWrapperElement?.();
  if (existing && existingWrapper && document.contains(existingWrapper)) {
    syncFeedbackMarkdownEditor(targetKey, feedbackTargetForm(targetKey).body || "");
    existing.instance.codemirror?.refresh();
    return;
  }
  if (existing) destroyFeedbackMarkdownEditor(targetKey);

  const textarea = document.getElementById(editorId);
  if (!textarea) return;
  const editor = new MarkdownEditor({
    element: textarea,
    initialValue: feedbackTargetForm(targetKey).body || "",
    forceSync: true,
    spellChecker: false,
    status: false,
    minHeight: isReplyTarget ? "104px" : isEditTarget ? "130px" : "210px",
    maxHeight: isReplyTarget ? "150px" : "52vh",
    previewRender: value => renderMarkdown(value || ""),
    placeholder: textarea.getAttribute("placeholder") || "Use Markdown to format your comment.",
    toolbar: [
      "heading", "bold", "italic", "strikethrough", "|",
      "quote", "code", "unordered-list", "ordered-list", "|",
      "link", "table", "horizontal-rule", "|",
      "preview", "side-by-side", "fullscreen", "guide",
    ],
  });
  const entry = { instance: editor, silent: false };
  feedbackMarkdownEditors.set(targetKey, entry);

  editor.codemirror.on("focus", () => {
    feedbackTextTarget.value = { target: targetKey, textarea: null };
  });
  editor.codemirror.on("keydown", (_cm, event) => {
    handleFeedbackMentionKeydown(event, targetKey);
  });
  editor.codemirror.on("cursorActivity", () => {
    handleFeedbackMarkdownMention(targetKey, editor);
  });
  editor.codemirror.on("change", () => {
    if (entry.silent) return;
    let body = editor.value() || "";
    if (body.length > FEEDBACK_BODY_LIMIT) {
      body = body.slice(0, FEEDBACK_BODY_LIMIT);
      syncFeedbackMarkdownEditor(targetKey, body);
      showToast(`留言最多 ${FEEDBACK_BODY_LIMIT} 字`, "error");
    }
    updateFeedbackDraftBody(targetKey, feedbackTargetForm(targetKey), body);
    handleFeedbackMarkdownMention(targetKey, editor);
  });
};

const pruneFeedbackMarkdownEditors = () => {
  for (const targetKey of [...feedbackMarkdownEditors.keys()]) {
    const editorId = feedbackMarkdownEditorElementId(targetKey);
    const entry = feedbackMarkdownEditors.get(targetKey);
    const wrapper = entry?.instance?.codemirror?.getWrapperElement?.();
    if (!document.getElementById(editorId) && (!wrapper || !document.contains(wrapper))) {
      destroyFeedbackMarkdownEditor(targetKey);
    }
  }
};

const refreshFeedbackMarkdownEditors = () => {
  nextTick(() => {
    loadFeedbackMarkdownEditor().then(ready => {
      nextTick(() => {
        if (ready) {
          pruneFeedbackMarkdownEditors();
          if (showFeedbackComposer.value) initFeedbackMarkdownEditor("post");
          if (selectedFeedbackPostId.value && feedbackReplies.value[selectedFeedbackPostId.value]) {
            initFeedbackMarkdownEditor(selectedFeedbackPostId.value);
          }
          if (feedbackEditing.value.id) initFeedbackMarkdownEditor(`edit:${feedbackEditing.value.id}`);
        }
      });
    });
  });
};

const feedbackDraftForTarget = target => {
  const targetKey = feedbackMentionTargetKey(target);
  const form = feedbackTargetForm(targetKey);
  const entry = feedbackMarkdownEditors.get(targetKey);
  const editor = entry?.instance;
  const cm = editor?.codemirror;
  if (cm) {
    const text = editor.value() || "";
    const selection = cm.listSelections()[0];
    const anchor = selection?.anchor || cm.getCursor();
    const head = selection?.head || anchor;
    const anchorIndex = cm.indexFromPos(anchor);
    const headIndex = cm.indexFromPos(head);
    const start = Math.min(anchorIndex, headIndex);
    const end = Math.max(anchorIndex, headIndex);
    return { targetKey, form, text, textarea: null, hasTextarea: false, editor, cm, hasEditor: true, start, end };
  }
  const text = form.body || "";
  const textarea = feedbackTextTarget.value.target === targetKey ? feedbackTextTarget.value.textarea : null;
  const hasTextarea = textarea && document.contains(textarea);
  const start = hasTextarea ? textarea.selectionStart ?? text.length : text.length;
  const end = hasTextarea ? textarea.selectionEnd ?? start : start;
  return { targetKey, form, text, textarea, hasTextarea, editor: null, cm: null, hasEditor: false, start, end };
};

const updateFeedbackDraftBody = (targetKey, form, body) => {
  if (targetKey === "post") {
    feedbackForm.value = { ...feedbackForm.value, body };
    return;
  }
  if (targetKey.startsWith("edit:")) {
    feedbackEditing.value = { ...feedbackEditing.value, body };
    return;
  }
  feedbackReplies.value = {
    ...feedbackReplies.value,
    [targetKey]: { ...(form || ensureFeedbackReplyForm(targetKey)), body },
  };
};

const replaceFeedbackSelection = (target, insertText, options = {}) => {
  const { targetKey, form, text, textarea, hasTextarea, cm, hasEditor, start, end } = feedbackDraftForTarget(target);
  const nextBody = `${text.slice(0, start)}${insertText}${text.slice(end)}`;
  updateFeedbackDraftBody(targetKey, form, nextBody);
  closeFeedbackMention();
  feedbackEmojiPickerTarget.value = "";
  const cursorStart = start + (options.selectStartOffset ?? insertText.length);
  const cursorEnd = start + (options.selectEndOffset ?? options.selectStartOffset ?? insertText.length);
  nextTick(() => {
    if (hasEditor && cm) {
      syncFeedbackMarkdownEditor(targetKey, nextBody);
      cm.focus();
      cm.setSelection(cm.posFromIndex(cursorStart), cm.posFromIndex(cursorEnd));
    } else if (hasTextarea) {
      textarea.focus();
      textarea.setSelectionRange(cursorStart, cursorEnd);
    }
  });
};

const replaceFeedbackRange = (target, rangeStart, rangeEnd, insertText, options = {}) => {
  const { targetKey, form, text, textarea, hasTextarea, cm, hasEditor } = feedbackDraftForTarget(target);
  const start = Math.max(0, Math.min(rangeStart, text.length));
  const end = Math.max(start, Math.min(rangeEnd, text.length));
  const nextBody = `${text.slice(0, start)}${insertText}${text.slice(end)}`;
  updateFeedbackDraftBody(targetKey, form, nextBody);
  closeFeedbackMention();
  feedbackEmojiPickerTarget.value = "";
  const cursorStart = start + (options.selectStartOffset ?? insertText.length);
  const cursorEnd = start + (options.selectEndOffset ?? options.selectStartOffset ?? insertText.length);
  nextTick(() => {
    if (hasEditor && cm) {
      syncFeedbackMarkdownEditor(targetKey, nextBody);
      cm.focus();
      cm.setSelection(cm.posFromIndex(cursorStart), cm.posFromIndex(cursorEnd));
    } else if (hasTextarea) {
      textarea.focus();
      textarea.setSelectionRange(cursorStart, cursorEnd);
    }
  });
};

const feedbackSelectedLineRange = (text, start, end) => {
  const safeStart = Math.max(0, Math.min(start, text.length));
  const safeEnd = Math.max(safeStart, Math.min(end, text.length));
  const effectiveEnd = safeEnd > safeStart && text[safeEnd - 1] === "\n" ? safeEnd - 1 : safeEnd;
  const rangeStart = text.lastIndexOf("\n", Math.max(0, safeStart - 1)) + 1;
  const nextBreak = text.indexOf("\n", effectiveEnd);
  const rangeEnd = nextBreak === -1 ? text.length : nextBreak;
  return { rangeStart, rangeEnd, block: text.slice(rangeStart, rangeEnd) };
};

const insertFeedbackEmoji = (emoji, target = "post") => {
  replaceFeedbackSelection(target, emoji);
};

const insertFeedbackSnippet = (prefix, suffix = "", placeholder = "", target = "post") => {
  const { text, start, end } = feedbackDraftForTarget(target);
  const selected = text.slice(start, end);
  const innerText = selected || placeholder;
  const insertText = `${prefix}${innerText}${suffix}`;
  replaceFeedbackSelection(target, insertText, {
    selectStartOffset: !selected && placeholder ? prefix.length : insertText.length,
    selectEndOffset: !selected && placeholder ? prefix.length + innerText.length : insertText.length,
  });
};

const insertFeedbackList = (ordered = false, target = "post") => {
  const { text, start, end } = feedbackDraftForTarget(target);
  const selected = text.slice(start, end);
  const { rangeStart, rangeEnd, block } = feedbackSelectedLineRange(text, start, end);
  if (!selected && !block.trim()) {
    const template = ordered ? "1. 第一项\n2. 第二项\n3. 第三项" : "- 第一项\n- 第二项\n- 第三项";
    const firstItemStart = ordered ? 3 : 2;
    replaceFeedbackRange(target, rangeStart, rangeEnd, template, {
      selectStartOffset: firstItemStart,
      selectEndOffset: firstItemStart + 3,
    });
    return;
  }

  const markerRe = ordered ? /^(\s*)\d+\.\s+(.*)$/ : /^(\s*)[-*+]\s+(.*)$/;
  const anyListRe = /^(\s*)(?:[-*+]\s+|\d+\.\s+)(.*)$/;
  const lines = block.split("\n");
  const contentLines = lines.filter(line => line.trim());
  const toggleOff = contentLines.length > 0 && contentLines.every(line => markerRe.test(line));
  let itemIndex = 1;
  const insertText = lines.map(line => {
    if (!line.trim()) return line;
    const anyMatch = line.match(anyListRe);
    const indent = anyMatch ? anyMatch[1] : (line.match(/^(\s*)/)?.[1] || "");
    const content = anyMatch ? anyMatch[2] : line.slice(indent.length);
    if (toggleOff) return `${indent}${content}`;
    const marker = ordered ? `${itemIndex++}. ` : "- ";
    return `${indent}${marker}${content}`;
  }).join("\n");
  replaceFeedbackRange(target, rangeStart, rangeEnd, insertText, {
    selectStartOffset: insertText.length,
    selectEndOffset: insertText.length,
  });
};

const insertFeedbackTaskList = (target = "post") => {
  const { text, start, end } = feedbackDraftForTarget(target);
  const selected = text.slice(start, end);
  const { rangeStart, rangeEnd, block } = feedbackSelectedLineRange(text, start, end);
  if (!selected && !block.trim()) {
    const template = "- [ ] 第一项\n- [ ] 第二项\n- [ ] 第三项";
    replaceFeedbackRange(target, rangeStart, rangeEnd, template, {
      selectStartOffset: 6,
      selectEndOffset: 9,
    });
    return;
  }

  const taskRe = /^(\s*)-\s+\[[ xX]\]\s+(.*)$/;
  const listRe = /^(\s*)(?:[-*+]\s+|\d+\.\s+)(.*)$/;
  const lines = block.split("\n");
  const contentLines = lines.filter(line => line.trim());
  const toggleOff = contentLines.length > 0 && contentLines.every(line => taskRe.test(line));
  const insertText = lines.map(line => {
    if (!line.trim()) return line;
    const taskMatch = line.match(taskRe);
    if (toggleOff && taskMatch) return `${taskMatch[1]}${taskMatch[2]}`;
    const listMatch = line.match(listRe);
    const indent = listMatch ? listMatch[1] : (line.match(/^(\s*)/)?.[1] || "");
    const content = taskMatch ? taskMatch[2] : (listMatch ? listMatch[2] : line.slice(indent.length));
    return `${indent}- [ ] ${content}`;
  }).join("\n");
  replaceFeedbackRange(target, rangeStart, rangeEnd, insertText, {
    selectStartOffset: insertText.length,
    selectEndOffset: insertText.length,
  });
};

const insertFeedbackQuote = (target = "post") => {
  const { text, start, end } = feedbackDraftForTarget(target);
  const selected = text.slice(start, end);
  const { rangeStart, rangeEnd, block } = feedbackSelectedLineRange(text, start, end);
  if (!selected && !block.trim()) {
    const template = "> 引用内容";
    replaceFeedbackRange(target, rangeStart, rangeEnd, template, {
      selectStartOffset: 2,
      selectEndOffset: template.length,
    });
    return;
  }

  const quoteRe = /^(\s*)>\s?(.*)$/;
  const lines = block.split("\n");
  const contentLines = lines.filter(line => line.trim());
  const toggleOff = contentLines.length > 0 && contentLines.every(line => quoteRe.test(line));
  const insertText = lines.map(line => {
    if (!line.trim()) return line;
    const quoteMatch = line.match(quoteRe);
    if (toggleOff && quoteMatch) return `${quoteMatch[1]}${quoteMatch[2]}`;
    const indent = line.match(/^(\s*)/)?.[1] || "";
    return `${indent}> ${line.slice(indent.length)}`;
  }).join("\n");
  replaceFeedbackRange(target, rangeStart, rangeEnd, insertText, {
    selectStartOffset: insertText.length,
    selectEndOffset: insertText.length,
  });
};

const insertFeedbackCodeBlock = (target = "post") => {
  const { text, start, end } = feedbackDraftForTarget(target);
  const selected = text.slice(start, end);
  const before = text.slice(0, start);
  const after = text.slice(end);
  const leadingNewline = before && !before.endsWith("\n") ? "\n" : "";
  const trailingNewline = after && !after.startsWith("\n") ? "\n" : "";
  const code = selected || "# 粘贴命令或代码";
  const fence = selected ? "```" : "```bash";
  const insertText = `${leadingNewline}${fence}\n${code}\n\`\`\`${trailingNewline}`;
  const codeStart = leadingNewline.length + fence.length + 1;
  replaceFeedbackSelection(target, insertText, {
    selectStartOffset: selected ? insertText.length : codeStart,
    selectEndOffset: selected ? insertText.length : codeStart + code.length,
  });
};

const insertFeedbackCode = (target = "post") => {
  const { text, start, end } = feedbackDraftForTarget(target);
  const selected = text.slice(start, end);
  if (selected.includes("\n")) {
    insertFeedbackCodeBlock(target);
    return;
  }
  insertFeedbackSnippet("`", "`", "code", target);
};

const closeFeedbackMention = () => {
  if (feedbackMentionTimer) {
    clearTimeout(feedbackMentionTimer);
    feedbackMentionTimer = null;
  }
  feedbackMention.value = {
    ...feedbackMention.value,
    visible: false,
    loading: false,
    candidates: [],
    activeIndex: 0,
  };
};

const closeFeedbackTransientPanels = () => {
  feedbackReactionPickerId.value = "";
  feedbackEmojiPickerTarget.value = "";
  closeFeedbackMention();
};
document.addEventListener("click", closeFeedbackTransientPanels);

const detectFeedbackMention = (text, cursor) => {
  const before = (text || "").slice(0, cursor);
  const match = before.match(/(^|[\s([{"'，。！？；：、])@([A-Za-z0-9_.-]{0,40})$/);
  if (!match) return null;
  const query = match[2] || "";
  if (!query || !/^[A-Za-z][A-Za-z0-9_.-]*$/.test(query)) return null;
  const start = before.lastIndexOf("@");
  return { query, start, end: cursor };
};

const fetchFeedbackMentionCandidates = async (query, target, seq) => {
  const targetKey = feedbackMentionTargetKey(target);
  try {
    const params = new URLSearchParams({ q: query, limit: "8" });
    const r = await fetch(`/api/mention-candidates?${params}`, { credentials: "include" });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(payload.detail || "加载候选失败");
    if (
      seq !== feedbackMentionSeq ||
      !feedbackMention.value.visible ||
      feedbackMention.value.query !== query ||
      feedbackMention.value.target !== targetKey
    ) return;
    feedbackMention.value = {
      ...feedbackMention.value,
      loading: false,
      candidates: payload.data || [],
      activeIndex: 0,
    };
  } catch (e) {
    if (seq !== feedbackMentionSeq) return;
    feedbackMention.value = {
      ...feedbackMention.value,
      loading: false,
      candidates: [],
      activeIndex: 0,
    };
  }
};

const scheduleFeedbackMentionFetch = (query, target) => {
  if (feedbackMentionTimer) clearTimeout(feedbackMentionTimer);
  const seq = ++feedbackMentionSeq;
  feedbackMentionTimer = setTimeout(() => {
    feedbackMentionTimer = null;
    fetchFeedbackMentionCandidates(query, target, seq);
  }, 140);
};

const handleFeedbackMentionInput = (event, target = "post") => {
  const textarea = event?.target;
  if (!textarea) return;
  setFeedbackTextTarget(event, target);
  feedbackMentionTextarea = textarea;
  const detected = detectFeedbackMention(textarea.value, textarea.selectionStart ?? textarea.value.length);
  if (!detected) {
    closeFeedbackMention();
    return;
  }
  const targetKey = feedbackMentionTargetKey(target);
  feedbackMention.value = {
    visible: true,
    loading: true,
    target: targetKey,
    query: detected.query,
    start: detected.start,
    end: detected.end,
    candidates: [],
    activeIndex: 0,
  };
  scheduleFeedbackMentionFetch(detected.query, targetKey);
};

const handleFeedbackMentionKeydown = (event, target = "post") => {
  const state = feedbackMention.value;
  if (!state.visible || state.target !== feedbackMentionTargetKey(target)) return;
  const count = state.candidates.length;
  if (event.key === "Escape") {
    event.preventDefault();
    closeFeedbackMention();
    return;
  }
  if (!count) return;
  if (event.key === "ArrowDown") {
    event.preventDefault();
    feedbackMention.value = { ...state, activeIndex: (state.activeIndex + 1) % count };
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    feedbackMention.value = { ...state, activeIndex: (state.activeIndex - 1 + count) % count };
  } else if (event.key === "Enter" || event.key === "Tab") {
    event.preventDefault();
    selectFeedbackMention(state.candidates[state.activeIndex], target);
  }
};

const selectFeedbackMention = (candidate, target = "post") => {
  if (!candidate?.username) return;
  const state = feedbackMention.value;
  const targetKey = feedbackMentionTargetKey(target || state.target);
  if (state.target && state.target !== targetKey) return;
  const mention = `@${candidate.username} `;
  let nextBody = "";
  if (targetKey === "post") {
    const text = feedbackForm.value.body || "";
    nextBody = `${text.slice(0, state.start)}${mention}${text.slice(state.end)}`;
    feedbackForm.value = { ...feedbackForm.value, body: nextBody };
  } else if (targetKey.startsWith("edit:")) {
    const text = feedbackEditing.value.body || "";
    nextBody = `${text.slice(0, state.start)}${mention}${text.slice(state.end)}`;
    feedbackEditing.value = { ...feedbackEditing.value, body: nextBody };
  } else {
    const form = ensureFeedbackReplyForm(targetKey);
    const text = form.body || "";
    nextBody = `${text.slice(0, state.start)}${mention}${text.slice(state.end)}`;
    feedbackReplies.value = {
      ...feedbackReplies.value,
      [targetKey]: { ...form, body: nextBody },
    };
  }
  const cursor = state.start + mention.length;
  const textarea = feedbackMentionTextarea;
  const editor = feedbackMarkdownEditors.get(targetKey)?.instance;
  const cm = editor?.codemirror;
  closeFeedbackMention();
  nextTick(() => {
    if (cm) {
      syncFeedbackMarkdownEditor(targetKey, nextBody);
      cm.focus();
      cm.setCursor(cm.posFromIndex(cursor));
    } else if (textarea && document.contains(textarea)) {
      textarea.focus();
      textarea.setSelectionRange(cursor, cursor);
    }
  });
};

const mergeFeedbackPost = post => {
  if (!post?.id) return;
  const idx = feedbackItems.value.findIndex(item => item.id === post.id);
  if (idx >= 0) {
    const next = [...feedbackItems.value];
    next[idx] = { ...next[idx], ...post };
    feedbackItems.value = next;
    return;
  }
  feedbackItems.value = [post, ...feedbackItems.value];
  feedbackTotal.value += 1;
  feedbackOffset.value = feedbackItems.value.length;
};

const revokeFeedbackPreviews = previews => {
  for (const preview of previews || []) {
    if (preview.url) URL.revokeObjectURL(preview.url);
  }
};

const feedbackFilePreviews = files =>
  files.map(file => ({
    name: file.name,
    url: URL.createObjectURL(file),
  }));

const setFeedbackFiles = (event, parentId = null) => {
  const selected = Array.from(event.target.files || []).filter(file => file.type.startsWith("image/"));
  if (selected.length !== (event.target.files || []).length) {
    showToast("仅支持图片文件", "error");
  }
  const files = selected.slice(0, 4);
  if (selected.length > 4) showToast("最多选择 4 张图片", "error");
  if (parentId) {
    const form = feedbackReplies.value[parentId] || { open: true, body: "", files: [], previews: [], submitting: false, mode: "write" };
    revokeFeedbackPreviews(form.previews);
    feedbackReplies.value = {
      ...feedbackReplies.value,
      [parentId]: { ...form, files, previews: feedbackFilePreviews(files) },
    };
  } else {
    revokeFeedbackPreviews(feedbackForm.value.previews);
    feedbackForm.value = { ...feedbackForm.value, files, previews: feedbackFilePreviews(files) };
  }
  event.target.value = "";
};

const clearFeedbackForm = (parentId = null) => {
  closeFeedbackMention();
  if (parentId) {
    const form = feedbackReplies.value[parentId];
    revokeFeedbackPreviews(form?.previews || []);
    feedbackReplies.value = {
      ...feedbackReplies.value,
      [parentId]: { open: false, body: "", files: [], previews: [], submitting: false, mode: "write" },
    };
    nextTick(() => syncFeedbackMarkdownEditor(parentId, ""));
    return;
  }
  revokeFeedbackPreviews(feedbackForm.value.previews);
  feedbackForm.value = { body: "", files: [], previews: [] };
  feedbackPostEditorMode.value = "write";
  nextTick(() => syncFeedbackMarkdownEditor("post", ""));
};

const ensureFeedbackReplyForm = id => {
  if (feedbackReplies.value[id]) return feedbackReplies.value[id];
  const form = { open: false, body: "", files: [], previews: [], submitting: false, mode: "write" };
  feedbackReplies.value = { ...feedbackReplies.value, [id]: form };
  return form;
};

const feedbackReplyEditorMode = id => feedbackReplies.value[id]?.mode || "write";

const setFeedbackPostEditorMode = (mode = "write") => {
  feedbackPostEditorMode.value = mode === "preview" ? "preview" : "write";
  closeFeedbackMention();
  feedbackEmojiPickerTarget.value = "";
  if (feedbackPostEditorMode.value === "write") refreshFeedbackMarkdownEditors();
};

const setFeedbackReplyEditorMode = (id, mode = "write") => {
  const form = ensureFeedbackReplyForm(id);
  feedbackReplies.value = {
    ...feedbackReplies.value,
    [id]: { ...form, mode: mode === "preview" ? "preview" : "write" },
  };
  closeFeedbackMention();
  feedbackEmojiPickerTarget.value = "";
  if (feedbackReplies.value[id]?.mode === "write") refreshFeedbackMarkdownEditors();
};

const feedbackPostPreviewHtml = computed(() => renderMarkdown(feedbackForm.value.body || ""));

const feedbackReplyPreviewHtml = id => renderMarkdown(feedbackReplies.value[id]?.body || "");

const feedbackMessageHtml = message => renderMarkdown(message?.body || "");

const focusFeedbackMessage = id => {
  selectedFeedbackMessageId.value = id || "";
  if (!id) return;
  nextTick(() => {
    const el = document.getElementById(`feedback-message-${id}`);
    if (!el) return;
    el.scrollIntoView({ block: "center", behavior: "smooth" });
    window.setTimeout(() => {
      if (selectedFeedbackMessageId.value === id) selectedFeedbackMessageId.value = "";
    }, 6000);
  });
};

const toggleFeedbackReply = id => {
  const form = ensureFeedbackReplyForm(id);
  feedbackReplies.value = {
    ...feedbackReplies.value,
    [id]: { ...form, open: !form.open },
  };
  refreshFeedbackMarkdownEditors();
};

const selectFeedbackPost = async (id, { refresh = true, focusMessageId = "" } = {}) => {
  if (!id) return;
  selectedFeedbackPostId.value = id;
  selectedFeedbackMessageId.value = focusMessageId || "";
  ensureFeedbackReplyForm(id);
  if (!refresh) return;
  feedbackDetailLoading.value = true;
  try {
    const r = await fetch(`/api/feedback/${encodeURIComponent(id)}`, { credentials: "include" });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(payload.detail || "加载帖子失败");
    mergeFeedbackPost(payload);
    selectedFeedbackPostId.value = payload.id || id;
    ensureFeedbackReplyForm(selectedFeedbackPostId.value);
    focusFeedbackMessage(focusMessageId || selectedFeedbackMessageId.value);
  } catch (e) {
    showToast(e.message || "加载帖子失败", "error");
  } finally {
    feedbackDetailLoading.value = false;
    refreshFeedbackMarkdownEditors();
  }
};

const closeFeedbackPost = () => {
  selectedFeedbackPostId.value = "";
  selectedFeedbackMessageId.value = "";
  cancelFeedbackEdit();
  destroyFeedbackMarkdownEditors();
  if (router.currentRoute.value.name === "feedback" && router.currentRoute.value.params?.postId) {
    router.push({ name: "feedback" }).catch(() => {});
  }
};

const openFeedbackComposer = () => {
  clearFeedbackForm();
  showFeedbackComposer.value = true;
  refreshFeedbackMarkdownEditors();
};

const closeFeedbackComposer = () => {
  destroyFeedbackMarkdownEditor("post");
  clearFeedbackForm();
  closeFeedbackMention();
  showFeedbackComposer.value = false;
};

const closeFeedbackBoard = () => {
  closeFeedbackComposer();
  cancelFeedbackEdit();
  closeFeedbackMention();
  selectedFeedbackPostId.value = "";
  selectedFeedbackMessageId.value = "";
  destroyFeedbackMarkdownEditors();
  showFeedbackBoard.value = false;
  if (router.currentRoute.value.name === "feedback") router.push("/").catch(() => {});
};

const showFeedbackSubmitResult = (payload, fallbackMessage) => {
  const notification = payload?.notification;
  if (!notification) {
    showToast(fallbackMessage, "success");
    return;
  }
  if (notification.status === "sent") {
    showToast(`${fallbackMessage}，邮件通知已发送`, "success", 4200);
    return;
  }
  if (notification.status === "queued") {
    showToast(`${fallbackMessage}，邮件通知已提交`, "success", 4200);
    return;
  }
  if (notification.status === "no_recipients" || notification.status === "disabled") {
    showToast(fallbackMessage, "success");
    return;
  }
  const detail = notification.detail || "邮件通知未发送";
  showToast(`${fallbackMessage}，但${detail}`, "error", 8000);
};

const runFeedbackEmailDiagnostics = async () => {
  feedbackEmailDiagLoading.value = true;
  try {
    const r = await fetch("/api/email/diagnostics", { credentials: "include" });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(payload.detail || `HTTP ${r.status}`);
    feedbackEmailDiagResult.value = payload;
    showToast(payload.ok ? "邮件环境诊断通过" : "邮件环境诊断未通过", payload.ok ? "success" : "error", 5000);
  } catch (e) {
    feedbackEmailDiagResult.value = {
      ok: false,
      checks: [{ status: "fail", label: "诊断请求", detail: e.message || "邮件诊断失败" }],
    };
    showToast(e.message || "邮件诊断失败", "error", 7000);
  } finally {
    feedbackEmailDiagLoading.value = false;
  }
};

const setFeedbackSort = async sortKey => {
  if (!sortKey || feedbackSort.value === sortKey) return;
  feedbackSort.value = sortKey;
  selectedFeedbackPostId.value = "";
  await loadFeedback({ reset: true });
};

const loadFeedback = async ({ reset = false, selectId = "" } = {}) => {
  if (feedbackLoading.value) return;
  feedbackLoading.value = true;
  if (reset) {
    feedbackOffset.value = 0;
    feedbackItems.value = [];
  }
  const params = new URLSearchParams({
    limit: String(feedbackLimit.value),
    offset: String(feedbackOffset.value),
    sort: feedbackSort.value,
  });
  try {
    const r = await fetch(`/api/feedback?${params}`, { credentials: "include" });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(payload.detail || "加载留言失败");
    const rows = payload.data || [];
    feedbackItems.value = reset ? rows : [...feedbackItems.value, ...rows];
    feedbackTotal.value = payload.total || 0;
    feedbackOffset.value = feedbackItems.value.length;
    const desiredId = selectId || selectedFeedbackPostId.value;
    if (desiredId && feedbackItems.value.some(item => item.id === desiredId)) {
      selectedFeedbackPostId.value = desiredId;
      ensureFeedbackReplyForm(desiredId);
    } else if (reset) {
      selectedFeedbackPostId.value = "";
    }
  } catch (e) {
    showToast(e.message || "加载留言失败", "error");
  } finally {
    feedbackLoading.value = false;
  }
};

const openFeedbackBoard = async () => {
  if (router.currentRoute.value.name !== "feedback") {
    await router.push({ name: "feedback" }).catch(() => {});
    return;
  }
  await loadMe().catch(() => {});
  showFeedbackBoard.value = true;
  selectedFeedbackPostId.value = "";
  selectedFeedbackMessageId.value = "";
  await loadFeedback({ reset: true });
};

const openFeedbackDeepLink = async ({ postId = "", messageId = "" } = {}) => {
  const targetPostId = postId || messageId;
  await loadMe().catch(() => {});
  showFeedbackBoard.value = true;
  showFeedbackComposer.value = false;
  cancelFeedbackEdit();
  closeFeedbackMention();
  if (!targetPostId) {
    selectedFeedbackPostId.value = "";
    selectedFeedbackMessageId.value = "";
    await loadFeedback({ reset: true });
    return;
  }
  selectedFeedbackMessageId.value = messageId || "";
  await loadFeedback({ reset: true, selectId: targetPostId });
  await selectFeedbackPost(targetPostId, { refresh: true, focusMessageId: selectedFeedbackMessageId.value });
};

const routeString = value => Array.isArray(value) ? (value[0] || "") : (value || "");
const openFeedbackRoute = route => openFeedbackDeepLink({
  postId: routeString(route.params?.postId),
  messageId: routeString(route.query?.message),
});

const refreshFeedbackBoard = async () => {
  await loadMe().catch(() => {});
  if (selectedFeedbackPostId.value) {
    await selectFeedbackPost(selectedFeedbackPostId.value, { refresh: true });
  } else {
    await loadFeedback({ reset: true });
  }
};

const submitFeedback = async (parentId = null) => {
  const isReply = Boolean(parentId);
  const form = isReply ? ensureFeedbackReplyForm(parentId) : feedbackForm.value;
  const body = (form.body || "").trim();
  if (!body && !(form.files || []).length) {
    showToast("请输入文字或选择图片", "error");
    return;
  }
  if (isReply) {
    feedbackReplies.value = {
      ...feedbackReplies.value,
      [parentId]: { ...form, submitting: true },
    };
  } else {
    feedbackSubmitting.value = true;
  }
  const fd = new FormData();
  fd.append("body", body);
  if (parentId) fd.append("parent_id", parentId);
  for (const file of form.files || []) fd.append("images", file);
  try {
    const r = await fetch("/api/feedback", {
      method: "POST",
      credentials: "include",
      body: fd,
    });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(payload.detail || "提交留言失败");
    const targetPostId = isReply ? parentId : payload.id;
    clearFeedbackForm(parentId);
    if (!isReply) {
      destroyFeedbackMarkdownEditor("post");
      showFeedbackComposer.value = false;
    }
    await loadFeedback({ reset: true, selectId: targetPostId });
    if (targetPostId) await selectFeedbackPost(targetPostId, { refresh: true });
    showFeedbackSubmitResult(payload, isReply ? "回复已发布" : "帖子已发布");
  } catch (e) {
    showToast(e.message || "提交留言失败", "error");
  } finally {
    if (isReply) {
      const latest = feedbackReplies.value[parentId] || form;
      feedbackReplies.value = {
        ...feedbackReplies.value,
        [parentId]: { ...latest, submitting: false },
      };
    } else {
      feedbackSubmitting.value = false;
    }
  }
};

const startFeedbackEdit = message => {
  if (!canEditFeedbackMessage(message)) return;
  feedbackEditing.value = {
    id: message.id,
    body: message.body || "",
    saving: false,
  };
  nextTick(() => {
    const el = document.getElementById(`feedback-edit-${message.id}`);
    el?.focus();
    initFeedbackMarkdownEditor(`edit:${message.id}`);
  });
};

const cancelFeedbackEdit = () => {
  if (feedbackEditing.value.id) destroyFeedbackMarkdownEditor(`edit:${feedbackEditing.value.id}`);
  closeFeedbackMention();
  feedbackEditing.value = { id: "", body: "", saving: false };
};

const saveFeedbackEdit = async message => {
  if (!message?.id || feedbackEditing.value.id !== message.id) return;
  const body = (feedbackEditing.value.body || "").trim();
  if (!body && !(message.attachments || []).length) {
    showToast("留言内容不能为空", "error");
    return;
  }
  feedbackEditing.value = { ...feedbackEditing.value, saving: true };
  try {
    const r = await fetch(`/api/feedback/${encodeURIComponent(message.id)}`, {
      method: "PATCH",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body }),
    });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(payload.detail || "编辑失败");
    updateFeedbackMessageInState(payload);
    cancelFeedbackEdit();
    showToast("已保存编辑", "success");
  } catch (e) {
    showToast(e.message || "编辑失败", "error");
    feedbackEditing.value = { ...feedbackEditing.value, saving: false };
  }
};

const toggleFeedbackReaction = async (message, emoji) => {
  if (!message?.id || !emoji) return;
  try {
    const r = await fetch(`/api/feedback/${encodeURIComponent(message.id)}/reactions`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ emoji }),
    });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(payload.detail || "表情操作失败");
    setFeedbackMessageReactions(message.id, payload.reactions || []);
    feedbackReactionPickerId.value = "";
  } catch (e) {
    showToast(e.message || "表情操作失败", "error");
  }
};

const deleteFeedbackMessage = async (message, kind = "post") => {
  if (!isAdmin.value || !message?.id) return;
  const isReply = kind === "reply" || Boolean(message.parent_id);
  const ok = await askConfirm(
    isReply
      ? "确定删除这条交流？图片附件也会一起删除。"
      : "确定删除这个帖子？帖子里的所有交流和图片附件都会一起删除。",
    {
      title: "管理员删除",
      confirmText: "删除",
      tone: "danger",
    },
  );
  if (!ok) return;

  try {
    const r = await fetch(`/api/feedback/${encodeURIComponent(message.id)}`, {
      method: "DELETE",
      credentials: "include",
    });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(payload.detail || "删除失败");

    const deletedIds = new Set(payload.ids || [message.id]);
    if (isReply) {
      feedbackItems.value = feedbackItems.value.map(post => {
        const removedReplies = (post.replies || []).filter(reply => deletedIds.has(reply.id)).length;
        if (!removedReplies) return post;
        return {
          ...post,
          replies: (post.replies || []).filter(reply => !deletedIds.has(reply.id)),
          reply_count: Math.max(0, Number(post.reply_count || 0) - removedReplies),
        };
      });
      if (selectedFeedbackPostId.value) {
        await selectFeedbackPost(selectedFeedbackPostId.value, { refresh: true });
      }
    } else {
      feedbackItems.value = feedbackItems.value.filter(post => !deletedIds.has(post.id));
      feedbackTotal.value = Math.max(0, feedbackTotal.value - 1);
      feedbackOffset.value = feedbackItems.value.length;
      if (deletedIds.has(selectedFeedbackPostId.value)) selectedFeedbackPostId.value = "";
    }
    showToast(isReply ? "交流已删除" : "帖子已删除", "success");
  } catch (e) {
    showToast(e.message || "删除失败", "error");
  }
};

const deleteFeedbackPost = post => deleteFeedbackMessage(post, "post");
const deleteFeedbackReply = reply => deleteFeedbackMessage(reply, "reply");

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
let loadJobRequestSeq = 0;
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
    const data = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, data, "加载历史记录失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
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
    } else if (!historyGroups.value.some(group => collapsedGroups.value[group.id]) && historyGroups.value.length) {
      collapsedGroups.value = {
        ...collapsedGroups.value,
        [historyGroups.value[0].id]: true,
      };
    }
    const expandedGroups = historyGroups.value.filter(group => collapsedGroups.value[group.id]);
    await Promise.all(expandedGroups.map(group => loadHistoryGroupJobs(group.id, true)));
  } catch (e) {
    if (e.name !== "AbortError") showToast(normalizeApiError(e, "加载历史记录失败"), "error");
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
    const data = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, data, "加载项目任务失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
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
      showToast(normalizeApiError(e, "加载项目任务失败"), "error");
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
    const data = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, data, "加载对比候选失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
    if (compareJobsController !== controller) return;
    compareJobs.value = data.data || [];
    compareJobsTotal.value = data.total || 0;

    const details = { ...compareSelectionDetails.value };
    for (const job of compareJobs.value) {
      if (compareSelection.value.includes(job.id)) details[job.id] = job;
    }
    compareSelectionDetails.value = details;

    const batchDetails = { ...batchSelectionDetails.value };
    for (const job of compareJobs.value) {
      if (batchBaselineId.value === job.id || batchCandidateIds.value.includes(job.id)) {
        batchDetails[job.id] = job;
      }
    }
    batchSelectionDetails.value = batchDetails;
  } catch (e) {
    if (e.name !== "AbortError") showToast(normalizeApiError(e, "加载对比候选失败"), "error");
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

const initializeAppData = async () => {
  if (appInitialized && !authInitError.value) {
    return !authRequired.value || Boolean(currentUser.value);
  }
  authInitError.value = "";
  try {
    await loadConfig();
    await loadMe();
    if (authRequired.value && !currentUser.value) {
      appInitialized = true;
      return false;
    }
    await loadProjects();
    await refreshSidebarData();
    appInitialized = true;
    return true;
  } catch (e) {
    appInitialized = false;
    authChecked.value = true;
    authInitError.value = normalizeApiError(e, "初始化失败");
    return false;
  }
};

const loadJob = async id => {
  const jobId = String(id || "");
  const requestSeq = ++loadJobRequestSeq;
  const r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { credentials: "include" });
  const data = await readJsonResponse(r, {});
  if (requestSeq !== loadJobRequestSeq || selectedJobId.value !== jobId) return "stale";
  if (!r.ok) {
    if (r.status === 401) {
      handleAuthExpired();
      return false;
    }
    if (r.status !== 404) {
      throw new ApiRequestError(apiErrorMessage(r, data, "加载任务失败"), {
        status: r.status,
      });
    }
    selectedJob.value = null;
    return false;
  }
  selectedJob.value = data;
  resultTable.value = { fields: [], rows: [], total: 0, filtered_total: 0, limit: tableLimit.value, offset: tableOffset.value };
  resultTableFile.value = "";
  consoleSearch.value = "";
  chartTables.value = {};
  chartSource.value = "";
  chartMetric.value = "";
  chartError.value = "";
  chartSummaryCards.value = [];
  chartSlowdowns.value = [];
  chartSpeedups.value = [];
  chartBarRows.value = [];
  chartPieRows.value = [];
  aiAnalysisError.value = "";
  aiAnalysisContent.value = "";
  aiAnalysisArtifacts.value = [];
  aiAnalysisVersions.value = [];
  aiAnalysisSelectedVersionId.value = "";
  aiAnalysisPrompt.value = "";
  aiArtifactsExpanded.value = false;
  aiDiagnosticsLoading.value = false;
  aiDiagnosticsError.value = "";
  aiDiagnosticsResult.value = null;
  if (isAiAnalysisActive(selectedJob.value?.ai_analysis?.status)) {
    startAiAnalysisPolling();
  } else {
    stopAiAnalysisPolling();
  }
  if (resultTab.value === "ai") refreshAiAnalysis({ silent: true });
  return true;
};

const aiAnalysisMeta = computed(() => selectedJob.value?.ai_analysis || {
  enabled: claudeAnalysisEnabled.value,
  status: "not_started",
  report_exists: false,
});
const aiAnalysisSelectedVersion = computed(() => {
  const versions = aiAnalysisVersions.value || [];
  const selected = versions.find(item => item.id === aiAnalysisSelectedVersionId.value);
  return selected || aiAnalysisMeta.value?.selected_version || null;
});
const aiAnalysisVersionTrigger = computed(() => {
  const version = aiAnalysisSelectedVersion.value || {};
  return version.trigger_user_display || version.trigger_user_token || "未知";
});
const aiAnalysisVersionModel = computed(() => {
  const version = aiAnalysisSelectedVersion.value || {};
  return version.model || aiAnalysisMeta.value?.model || "未知";
});
const aiAnalysisVersionLabel = version => {
  const timeText = fmtDateTime(version.generated_at || version.finished_at) || "未知时间";
  const statusText = aiAnalysisStatusText(version.status || "done");
  const triggerText = version.trigger_user_display || version.trigger_user_token || "未知触发人";
  return `${timeText} · ${statusText} · ${triggerText}`;
};
const aiAnalysisVisibleArtifacts = computed(() => {
  const artifacts = aiAnalysisArtifacts.value || [];
  if (!aiAnalysisContent.value) return artifacts;
  return artifacts.filter(item => item.path !== "ai_analysis.md");
});
const aiArtifactSummary = computed(() =>
  aiAnalysisVisibleArtifacts.value
    .map(item => `${item.path} (${fmtBytes(item.size)})`)
    .join(" · ")
);
const AI_ARTIFACT_DOWNLOAD_EXT_RE = /\.(?:csv|db|json|log|md|py|text|tsv|txt|ya?ml)$/i;
const normalizeAiArtifactDownloadPath = value => {
  const raw = String(value || "").trim().replace(/\\/g, "/");
  if (!raw || raw.startsWith("/") || /^[A-Za-z]:/.test(raw) || raw.includes("://")) return "";
  if (!AI_ARTIFACT_DOWNLOAD_EXT_RE.test(raw)) return "";
  const parts = raw.split("/");
  if (parts.some(part => !part || part === "." || part === ".." || part.startsWith("."))) return "";
  if (parts[0] === "versions") return "";
  return raw;
};
const encodePathSegments = value => String(value || "").split("/").map(encodeURIComponent).join("/");
const resolveAiArtifactPath = artifactOrPath => {
  const normalized = normalizeAiArtifactDownloadPath(
    typeof artifactOrPath === "string" ? artifactOrPath : artifactOrPath?.path,
  );
  if (!normalized) return "";
  const artifacts = aiAnalysisArtifacts.value || [];
  const normalizedLower = normalized.toLowerCase();
  const normalizedName = normalizedLower.split("/").pop();
  const matched = artifacts.find(item => {
    const path = String(item?.path || "").replace(/\\/g, "/");
    const name = String(item?.name || "").toLowerCase();
    const pathLower = path.toLowerCase();
    return pathLower === normalizedLower
      || pathLower.endsWith(`/${normalizedLower}`)
      || name === normalizedName;
  });
  return matched?.path || normalized;
};
const isAiCodeArtifactPath = artifactOrPath => {
  const path = resolveAiArtifactPath(artifactOrPath);
  const lowered = path.toLowerCase();
  const filename = lowered.split("/").pop() || lowered;
  if (filename.endsWith(".py")) return true;
  if (!filename.endsWith(".txt")) return false;
  return lowered.includes("/triton_output_code/")
    || filename.includes("triton_output_code")
    || filename.startsWith("output_code_");
};
const aiArtifactDownloadUrl = artifactOrPath => {
  const path = resolveAiArtifactPath(artifactOrPath);
  if (!selectedJobId.value || !path) return "";
  return `/api/jobs/${encodeURIComponent(selectedJobId.value)}/ai-analysis/artifacts/${encodePathSegments(path)}`;
};
const aiArtifactContentUrl = artifactOrPath => {
  const path = resolveAiArtifactPath(artifactOrPath);
  if (!selectedJobId.value || !path) return "";
  return `/api/jobs/${encodeURIComponent(selectedJobId.value)}/ai-analysis/artifact-content/${encodePathSegments(path)}`;
};
const renderAiArtifactCode = code => {
  const url = aiArtifactDownloadUrl(code);
  if (!url) return "";
  const path = resolveAiArtifactPath(code);
  if (isAiCodeArtifactPath(path)) {
    return `<button type="button" class="ai-artifact-inline-link ai-code-preview-link" data-ai-code-path="${escapeHtml(path)}" title="查看 Python 代码 ${escapeHtml(path)}" aria-label="查看 Python 代码 ${escapeHtml(path)}">查看代码</button>`;
  }
  return `<a class="ai-artifact-inline-link" href="${escapeHtml(url)}" download title="下载 ${escapeHtml(code)}"><code>${escapeHtml(code)}</code></a>`;
};
const aiAnalysisHtml = computed(() => renderMarkdown(aiAnalysisContent.value, {
  codeRenderer: renderAiArtifactCode,
  collapsedSections: ["产物"],
}));
const highlightPythonCodeBlocks = () => nextTick(() => {
  if (!window.hljs) return;
  document.querySelectorAll('pre.code-viewer code.language-python').forEach((block) => {
    window.hljs.highlightElement(block);
  });
});

const aiAnalysisStatusText = status => ({
  not_started: "未开始",
  queued: "排队中",
  running: "分析中",
  done: "已完成",
  error: "失败",
}[status || "not_started"] || status);

const AI_ANALYSIS_ACTIVE_STATUSES = new Set(["queued", "running"]);
const isAiAnalysisActive = status => AI_ANALYSIS_ACTIVE_STATUSES.has(status || "");

const formatDurationMs = ms => {
  const value = Math.max(0, Math.round(Number(ms) || 0));
  const totalSeconds = Math.floor(value / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours) return `${hours}时${String(minutes).padStart(2, "0")}分${String(seconds).padStart(2, "0")}秒`;
  if (minutes) return `${minutes}分${String(seconds).padStart(2, "0")}秒`;
  return `${seconds}秒`;
};

const aiAnalysisElapsedMs = computed(() => {
  const meta = aiAnalysisMeta.value || {};
  if (Number.isFinite(Number(meta.duration_ms))) return Number(meta.duration_ms);
  if (Number.isFinite(Number(meta.elapsed_ms))) return Number(meta.elapsed_ms);
  const started = Date.parse(meta.started_at || "");
  if (isAiAnalysisActive(meta.status) && Number.isFinite(started)) {
    return Math.max(0, uiNow.value - started);
  }
  return 0;
});

const aiAnalysisElapsedText = computed(() => formatDurationMs(aiAnalysisElapsedMs.value));

const updateAiAnalysisState = payload => {
  if (!payload || !selectedJob.value) return;
  const { content, artifacts, diagnostics, versions, selected_version, selected_version_id, ...meta } = payload;
  const nextVersions = Array.isArray(versions) ? versions : [];
  const nextSelectedId = selected_version_id
    || selected_version?.id
    || meta.latest_version_id
    || nextVersions[0]?.id
    || "";
  const nextSelectedVersion = selected_version
    || nextVersions.find(item => item.id === nextSelectedId)
    || null;
  aiAnalysisVersions.value = nextVersions;
  aiAnalysisSelectedVersionId.value = nextSelectedId;
  selectedJob.value = {
    ...selectedJob.value,
    ai_analysis: {
      ...meta,
      diagnostics,
      versions: nextVersions,
      selected_version_id: nextSelectedId,
      selected_version: nextSelectedVersion,
    },
  };
  aiAnalysisContent.value = content || "";
  aiAnalysisArtifacts.value = Array.isArray(artifacts) ? artifacts : [];
  aiDiagnosticsLoading.value = false;
  aiDiagnosticsError.value = "";
  aiDiagnosticsResult.value = diagnostics || null;
};

const stopAiAnalysisPolling = () => {
  if (aiAnalysisPollTimer) {
    clearInterval(aiAnalysisPollTimer);
    aiAnalysisPollTimer = null;
  }
};

const clearAiDiagnostics = () => {
  aiDiagnosticsLoading.value = false;
  aiDiagnosticsError.value = "";
  aiDiagnosticsResult.value = null;
};

const requestAiCompletionNotificationPermission = () => {
  if (!("Notification" in window)) return;
  if (Notification.permission !== "default") return;
  Notification.requestPermission().catch(() => {});
};

const resetAiCompletionTitle = () => {
  if (aiCompletionTitleResetTimer) {
    clearTimeout(aiCompletionTitleResetTimer);
    aiCompletionTitleResetTimer = null;
  }
  if (document.title !== defaultDocumentTitle) document.title = defaultDocumentTitle;
};

const markAiCompletionTitle = title => {
  if (aiCompletionTitleResetTimer) clearTimeout(aiCompletionTitleResetTimer);
  document.title = `${title} - ${defaultDocumentTitle}`;
  aiCompletionTitleResetTimer = setTimeout(resetAiCompletionTitle, 15000);
};

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) resetAiCompletionTitle();
});

const notifyAiAnalysisCompleted = ({ jobId, label, status, error }) => {
  const ok = status === "done";
  const title = ok ? "AI 分析已完成" : "AI 分析失败";
  const kind = ok ? "success" : "error";
  const taskName = label || jobId || "当前任务";
  const body = ok
    ? `任务「${taskName}」的 AI 分析报告已生成。`
    : `任务「${taskName}」AI 分析失败，请返回查看诊断信息。`;
  const pageInForeground = !document.hidden && document.hasFocus();
  if (pageInForeground) {
    showToast(title, kind, 5000);
    return;
  }

  markAiCompletionTitle(title);
  let notified = false;
  if ("Notification" in window && Notification.permission === "granted") {
    try {
      const notification = new Notification(title, {
        body: error ? `${body}\n${error}` : body,
        tag: `tpa-ai-analysis-${jobId || "current"}`,
      });
      notification.onclick = () => {
        window.focus();
        if (jobId) window.location.hash = `#/job/${encodeURIComponent(jobId)}/ai`;
        notification.close();
      };
      notified = true;
    } catch {
      notified = false;
    }
  }
  if (!notified) showToast(title, kind, 10000);
};

const refreshAiAnalysis = async ({ silent = false, versionId } = {}) => {
  if (!selectedJobId.value) return;
  const jobId = selectedJobId.value;
  const previousStatus = selectedJob.value?.ai_analysis?.status || "not_started";
  const jobLabel = selectedJob.value?.label || selectedJob.value?.file_a_name || jobId;
  const requestedVersionId = versionId !== undefined
    ? versionId
    : (isAiAnalysisActive(previousStatus) ? "" : aiAnalysisSelectedVersionId.value);
  const params = new URLSearchParams();
  if (requestedVersionId) params.set("version_id", requestedVersionId);
  const url = `/api/jobs/${jobId}/ai-analysis${params.toString() ? `?${params}` : ""}`;
  if (!silent) aiAnalysisLoading.value = true;
  aiAnalysisError.value = "";
  try {
    const r = await fetch(url, { credentials: "include" });
    const payload = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, payload, "加载 AI 分析失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
    if (selectedJobId.value !== jobId) return;
    updateAiAnalysisState(payload);
    if (isAiAnalysisActive(previousStatus) && payload.status && !isAiAnalysisActive(payload.status)) {
      notifyAiAnalysisCompleted({
        jobId,
        label: jobLabel,
        status: payload.status,
        error: payload.error || "",
      });
    }
    if (isAiAnalysisActive(payload.status)) startAiAnalysisPolling();
    else stopAiAnalysisPolling();
  } catch (e) {
    if (selectedJobId.value === jobId) {
      aiAnalysisError.value = normalizeApiError(e, "加载 AI 分析失败");
    }
  } finally {
    if (selectedJobId.value === jobId) aiAnalysisLoading.value = false;
  }
};

const changeAiAnalysisVersion = () => {
  refreshAiAnalysis({ versionId: aiAnalysisSelectedVersionId.value });
};

const startAiAnalysisPolling = () => {
  if (aiAnalysisPollTimer) return;
  aiAnalysisPollTimer = setInterval(() => {
    if (!selectedJobId.value) {
      stopAiAnalysisPolling();
      return;
    }
    refreshAiAnalysis({ silent: true });
  }, 1000);
};

const openAiPromptModal = (force = false) => {
  if (!selectedJobId.value || aiAnalysisStarting.value || isAiAnalysisActive(aiAnalysisMeta.value.status)) return;
  if (!claudeAnalysisEnabled.value) {
    showToast("AI 分析未启用，请在服务端设置 TRACE_ENABLE_CLAUDE_ANALYSIS=1", "error");
    return;
  }
  aiPromptForce.value = Boolean(force);
  aiAnalysisPrompt.value = "";
  showAiPromptModal.value = true;
  nextTick(() => document.getElementById("ai-analysis-prompt-modal")?.focus());
};

const closeAiPromptModal = () => {
  if (aiAnalysisStarting.value) return;
  showAiPromptModal.value = false;
};

const confirmAiPromptModal = () => {
  const force = aiPromptForce.value;
  const prompt = aiAnalysisPrompt.value;
  showAiPromptModal.value = false;
  startAiAnalysis(force, prompt);
};

const startAiAnalysis = async (force = false, prompt = "") => {
  if (!selectedJobId.value || aiAnalysisStarting.value) return;
  const jobId = selectedJobId.value;
  if (!claudeAnalysisEnabled.value) {
    showToast("AI 分析未启用，请在服务端设置 TRACE_ENABLE_CLAUDE_ANALYSIS=1", "error");
    return;
  }
  requestAiCompletionNotificationPermission();
  aiAnalysisStarting.value = true;
  aiAnalysisError.value = "";
  aiAnalysisSelectedVersionId.value = "";
  aiArtifactsExpanded.value = false;
  aiDiagnosticsLoading.value = false;
  aiDiagnosticsError.value = "";
  aiDiagnosticsResult.value = null;
  try {
    const r = await fetch(`/api/jobs/${jobId}/ai-analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ force, prompt: String(prompt || "").trim() }),
    });
    const payload = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, payload, "提交 AI 分析失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
    if (selectedJobId.value !== jobId) return;
    updateAiAnalysisState(payload);
    if (isAiAnalysisActive(payload.status)) {
      startAiAnalysisPolling();
      showToast(payload.status === "queued" ? "AI 分析已排队" : "AI 分析已开始", "success");
    }
  } catch (e) {
    aiAnalysisError.value = normalizeApiError(e, "提交 AI 分析失败");
    showToast(aiAnalysisError.value, "error");
  } finally {
    aiAnalysisStarting.value = false;
  }
};

const copyAiAnalysisReport = async () => {
  if (!aiAnalysisContent.value) return;
  await copyTextToClipboard(aiAnalysisContent.value);
  showToast("AI 分析报告已复制", "success");
};

const downloadAiAnalysisReport = () => {
  if (!selectedJobId.value || !aiAnalysisContent.value) return;
  const a = document.createElement("a");
  const params = new URLSearchParams();
  if (aiAnalysisSelectedVersionId.value) params.set("version_id", aiAnalysisSelectedVersionId.value);
  a.href = `/api/jobs/${selectedJobId.value}/ai-analysis/report.md${params.toString() ? `?${params}` : ""}`;
  a.click();
};

const downloadAiAnalysisArtifact = artifact => {
  const url = aiArtifactDownloadUrl(artifact);
  if (!url) return;
  const a = document.createElement("a");
  a.href = url;
  a.download = artifact?.name || artifact?.path || "";
  a.click();
};

const openAiCodeViewer = async artifactOrPath => {
  const path = resolveAiArtifactPath(artifactOrPath);
  const url = aiArtifactContentUrl(path);
  if (!url) return;
  showAiCodeViewer.value = true;
  aiCodeViewerLoading.value = true;
  aiCodeViewerError.value = "";
  aiCodeViewerPath.value = path;
  aiCodeViewerFilename.value = path.split("/").pop() || path;
  aiCodeViewerContent.value = "";
  aiCodeViewerSize.value = 0;
  aiCodeViewerTruncated.value = false;
  try {
    const data = await fetchJson(url, { credentials: "include" }, "加载代码文件失败");
    aiCodeViewerPath.value = data.path || path;
    aiCodeViewerFilename.value = data.name || aiCodeViewerFilename.value;
    aiCodeViewerContent.value = data.content || "";
    aiCodeViewerSize.value = data.size || 0;
    aiCodeViewerTruncated.value = Boolean(data.truncated);
    highlightPythonCodeBlocks();
  } catch (e) {
    aiCodeViewerError.value = normalizeApiError(e, "加载代码文件失败");
    showToast(aiCodeViewerError.value, "error");
  } finally {
    aiCodeViewerLoading.value = false;
  }
};

const closeAiCodeViewer = () => {
  showAiCodeViewer.value = false;
  aiCodeViewerError.value = "";
};

const handleAiAnalysisReportClick = event => {
  const trigger = event.target?.closest?.("[data-ai-code-path]");
  if (!trigger) return;
  event.preventDefault();
  event.stopPropagation();
  openAiCodeViewer(trigger.getAttribute("data-ai-code-path") || "");
};

const downloadAiCodeViewer = () => {
  const url = aiArtifactDownloadUrl(aiCodeViewerPath.value);
  if (!url) return;
  const a = document.createElement("a");
  a.href = url;
  a.download = aiCodeViewerFilename.value || aiCodeViewerPath.value;
  a.click();
};

const copyAiAnalysisArtifact = async artifact => {
  if (!artifact?.content) return;
  await copyTextToClipboard(artifact.content);
  showToast(`${artifact.path || "AI 产物"} 已复制`, "success");
};

const aiDiagnosticStatusText = status => ({
  ok: "OK",
  error: "失败",
  skipped: "跳过",
}[status || ""] || status || "");

const formatAiDiagnostics = payload => {
  if (!payload) return "";
  const lines = [
    `AI 环境诊断：${payload.ok ? "通过" : "未通过"}`,
    `skills_dir: ${payload.skills_dir || "-"}`,
    `single_skill: ${payload.single_skill || "-"}`,
    `compare_skill: ${payload.compare_skill || "-"}`,
    `duration_ms: ${payload.duration_ms ?? "-"}`,
    "",
  ];
  for (const check of payload.checks || []) {
    lines.push(`[${aiDiagnosticStatusText(check.status)}] ${check.label || check.name}: ${check.detail || ""}`);
    if (check.command) lines.push(`command: ${check.command}`);
    if (check.stdout_tail) lines.push(`stdout:\n${check.stdout_tail}`);
    if (check.stderr_tail) lines.push(`stderr:\n${check.stderr_tail}`);
    lines.push("");
  }
  return lines.join("\n").trim();
};

const runAiDiagnostics = async () => {
  if (aiDiagnosticsLoading.value) return;
  if (!claudeAnalysisEnabled.value) {
    showToast("AI 分析未启用，请先设置 TRACE_ENABLE_CLAUDE_ANALYSIS=1", "error");
    return;
  }
  aiDiagnosticsLoading.value = true;
  aiDiagnosticsError.value = "";
  aiDiagnosticsResult.value = null;
  try {
    const r = await fetch("/api/ai/diagnostics", {
      method: "POST",
      credentials: "include",
    });
    const payload = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, payload, "AI 环境诊断失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
    aiDiagnosticsResult.value = payload;
    showToast(payload.ok ? "AI 环境诊断通过" : "AI 环境诊断未通过", payload.ok ? "success" : "error");
  } catch (e) {
    aiDiagnosticsError.value = normalizeApiError(e, "AI 环境诊断失败");
    showToast(aiDiagnosticsError.value, "error");
  } finally {
    aiDiagnosticsLoading.value = false;
  }
};

const copyAiDiagnostics = async () => {
  if (!aiDiagnosticsResult.value) return;
  await copyTextToClipboard(formatAiDiagnostics(aiDiagnosticsResult.value));
  showToast("AI 诊断结果已复制", "success");
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
  const jobId = options.jobId || selectedJobId.value;
  if (!jobId) throw new Error("未选择任务");
  const params = buildResultTableParams(options);
  const r = await fetch(
    `/api/jobs/${encodeURIComponent(jobId)}/results/${encodeURIComponent(filename)}?${params}`,
    { credentials: "include", signal: options.signal },
  );
  const data = await readJsonResponse(r, {});
  if (!r.ok) {
    throw new ApiRequestError(apiErrorMessage(r, data, "加载表格失败"), {
      status: r.status,
      authExpired: r.status === 401,
    });
  }
  return data;
};

const loadResultTable = async ({ resetOffset = false, filename = resultTab.value, viewState = null } = {}) => {
  const jobId = selectedJobId.value;
  if (!jobId || !filename?.endsWith(".csv")) return;
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
    const data = await fetchResultTable(filename, { jobId, signal: controller.signal, viewState });
    if (resultTableController !== controller) return;
    if (selectedJobId.value !== jobId) return;
    if (resultTableFile.value !== filename) return;
    if (resultTab.value !== filename) return;
    resultTable.value = data;
    tableLimit.value = data.limit || tableLimit.value;
    tableOffset.value = data.offset || 0;
  } catch (e) {
    if (e.name !== "AbortError" && selectedJobId.value === jobId) {
      resultTableError.value = normalizeApiError(e, "加载表格失败");
    }
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
    const data = await fetchResultTable(filename, { jobId, signal: controller.signal, viewState: state });
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
    resultTableError.value = normalizeApiError(e, "加载表格失败");
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

const canDrillKernelTypeRow = row =>
  isKernelTypeTab.value && Boolean(row?.type);

const drillDownKernelType = async row => {
  if (!canDrillKernelTypeRow(row)) return;
  const type = String(row.type || "").trim();
  const targetFile = selectedJob.value?.mode === "compare"
    ? "all_kernels_cmp.csv"
    : "all_kernels_avg.csv";
  const fields = selectedJob.value?.result_files?.[targetFile]?.fields || [];
  if (!fields.length) {
    showToast(`未找到 ${targetFile}`, "error");
    return;
  }
  if (!fields.includes("family")) {
    showToast("目标 Kernel 表缺少 family 字段，请重新分析/对比后再下钻", "error");
    return;
  }
  const sortField = selectedJob.value?.mode === "compare" && fields.includes("delta_dur_ms")
    ? "delta_dur_ms"
    : (fields.includes("avg_dur_ms") ? "avg_dur_ms" : "");
  const state = {
    ...defaultResultViewState(),
    tableLimit: tableLimit.value || 100,
    tableOffset: 0,
    sortCol: sortField,
    sortAsc: false,
    colFilters: { family: type },
    colFilterOps: { family: "~" },
    visibleColumns: fields.filter(field => field !== "family"),
  };
  const memory = readResultMemory(selectedJobId.value);
  memory.tabs = { ...(memory.tabs || {}), [targetFile]: state };
  writeResultMemory(selectedJobId.value, memory);
  await activateCsvTab(targetFile);
  showToast(`已下钻到 ${type} 相关 Kernel`, "success");
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

const isCommunicationChartRow = (row, sourceConfig) => {
  if (!row) return false;
  if (CHART_COMMUNICATION_SOURCE_FILES.has(sourceConfig?.file)) return true;
  const raw = row.raw || row;
  const family = String(raw.family ?? raw.type ?? "").trim().toLowerCase();
  if (CHART_COMMUNICATION_FAMILIES.has(family)) return true;
  const nameField = row.nameField || sourceConfig?.nameField;
  const label = String(row.label ?? raw[nameField] ?? raw.kernel_name ?? raw.op_name ?? raw.type ?? "").trim();
  return CHART_COMMUNICATION_NAME_RE.test(label);
};

const filterChartCommunicationRows = (rows, sourceConfig) =>
  (rows || []).filter(row => !isCommunicationChartRow(row, sourceConfig));

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

let chartBuildToken = 0;
const scheduleBuildChart = async () => {
  const token = ++chartBuildToken;
  await nextTick();
  await new Promise(resolve => {
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(resolve);
    else setTimeout(resolve, 0);
  });
  if (token !== chartBuildToken) return;
  if (resultTab.value === "chart" && selectedJob.value?.status === "done") {
    await buildChart();
  }
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
  const jobId = selectedJobId.value;
  if (!ktChart.value || !selectedJob.value?.result_files) {
    chartLoading.value = false;
    return;
  }
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
          jobId,
          limit: CHART_FETCH_LIMIT,
          offset: 0,
          ignoreViewState: true,
        }),
      };
      if (selectedJobId.value !== jobId) return;
    } catch (e) {
      chartError.value = normalizeApiError(e, "加载图表数据失败");
      chartLoading.value = false;
      return;
    }
  }
  const table = chartTables.value[sourceConfig.file];
  const fields = table?.fields || selectedJob.value.result_files[sourceConfig.file]?.fields || [];
  const rows = filterChartCommunicationRows(
    normalizeChartRows(table?.rows || [], fields, sourceConfig, metricDef),
    sourceConfig
  );
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
    meta: uploadFileMeta(file),
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
  if (job.is_owner === false) {
    showToast("只能批量操作自己创建的任务", "error");
    return;
  }
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

document.addEventListener("click", event => {
  if (!historyBulkMode.value) return;
  if (event.target?.closest?.(".sidebar")) return;
  historyBulkMode.value = false;
  clearHistorySelection();
});

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

const shareProject = async (project) => {
  if (!project?.id || project.is_public) return;
  if (!await askConfirm("确定将该项目转为共享项目？", {
    title: "转为共享项目",
    confirmText: "转为共享",
  })) return;
  const r = await fetch(`/api/projects/${project.id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      name: project.name,
      description: project.description || "",
      is_public: true,
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("转为共享失败: " + (err.detail || err.message || `HTTP ${r.status}`), "error");
    return;
  }
  await loadProjects();
  await refreshSidebarData();
  showToast("项目已转为共享", "success");
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

const downloadReport = () => {
  if (!selectedJobId.value) return;
  const a = document.createElement("a");
  a.href = `/api/jobs/${selectedJobId.value}/report.md`;
  a.click();
};

const shareJob = async () => {
  if (!selectedJobId.value) return;
  const r = await fetch(`/api/jobs/${selectedJobId.value}/share`, {
    method: "POST",
    credentials: "include",
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("生成分享链接失败: " + (err.detail || err.message || `HTTP ${r.status}`), "error");
    return;
  }
  const data = await r.json();
  const url = data.url || `${window.location.origin}${window.location.pathname}#/job/${selectedJobId.value}`;
  await copyTextToClipboard(url);
  if (data.changed) {
    await loadProjects();
    await refreshSidebarData();
    await loadJob(selectedJobId.value);
  }
  showToast(data.changed ? "已转为共享并复制链接" : "已复制分享链接", "success");
};

const togglePinJob = async () => {
  if (!selectedJobId.value || !selectedJob.value) return;
  const nextPinned = !selectedJob.value.is_pinned;
  const r = await fetch(`/api/jobs/${selectedJobId.value}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ is_pinned: nextPinned }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("置顶失败: " + (err.detail || err.message || `HTTP ${r.status}`), "error");
    return;
  }
  selectedJob.value = await r.json();
  await refreshSidebarData();
  showToast(nextPinned ? "任务已置顶" : "已取消置顶", "success");
};

const escapeHtml = value => String(value ?? "")
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

const safeMarkdownUrl = value => {
  const raw = String(value || "").trim();
  if (!raw) return "";
  if (/^(https?:|mailto:|#)/i.test(raw)) return raw.replace(/"/g, "%22");
  return "";
};

const renderInlineMarkdown = (text, options = {}) => {
  const tokens = [];
  const stash = html => {
    const key = `\u0000MD${tokens.length}\u0000`;
    tokens.push(html);
    return key;
  };
  let value = String(text ?? "");
  value = value.replace(/`([^`]+)`/g, (_, code) => {
    const custom = options.codeRenderer?.(code);
    return stash(custom || `<code>${escapeHtml(code)}</code>`);
  });
  value = value.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => {
    const safeUrl = safeMarkdownUrl(url);
    const safeLabel = escapeHtml(label);
    return safeUrl
      ? stash(`<a href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer">${safeLabel}</a>`)
      : safeLabel;
  });
  value = escapeHtml(value);
  value = value
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[\s([{])_([^_\n]+?)_(?=$|[\s)\]},.!?:;])/g, "$1<em>$2</em>");
  tokens.forEach((html, index) => {
    value = value.replaceAll(`\u0000MD${index}\u0000`, html);
  });
  return value;
};

const parseMarkdownCells = line => {
  let value = line.trim();
  if (value.startsWith("|")) value = value.slice(1);
  if (value.endsWith("|")) value = value.slice(0, -1);
  return value.split("|").map(cell => cell.trim());
};

const isMarkdownTableSep = line => {
  const cells = parseMarkdownCells(line);
  return cells.length > 1 && cells.every(cell => /^:?-{3,}:?$/.test(cell));
};

const normalizeMarkdownHeadingTitle = value => String(value || "")
  .replace(/[`*_~]/g, "")
  .trim();

function renderMarkdown(markdown, options = {}) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let i = 0;
  const collapsedSections = new Set(options.collapsedSections || []);
  const isTableStartAt = index => index + 1 < lines.length && isMarkdownTableSep(lines[index + 1]);
  const isHorizontalRule = line => /^[-*_]\s*[-*_]\s*[-*_][\s\-*_]*$/.test(line.trim());
  const isListStart = line => /^\s*[-*+]\s+/.test(line) || /^\s*\d+\.\s+/.test(line);
  const isBlockStartAt = index => {
    const line = lines[index] || "";
    return !line.trim()
      || /^```/.test(line.trim())
      || /^#{1,6}\s+/.test(line)
      || /^>\s?/.test(line)
      || isHorizontalRule(line)
      || isListStart(line)
      || isTableStartAt(index);
  };
  const renderListItem = parts =>
    parts.map(part => renderInlineMarkdown(part.trim(), options)).join("<br>");

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) {
      i += 1;
      continue;
    }

    const fence = trimmed.match(/^```(\w+)?/);
    if (fence) {
      const lang = fence[1] ? ` data-lang="${escapeHtml(fence[1])}"` : "";
      i += 1;
      const code = [];
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      html.push(`<pre class="md-code"${lang}><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const title = normalizeMarkdownHeadingTitle(heading[2]);
      if (level === 2 && collapsedSections.has(title)) {
        const start = i + 1;
        let end = start;
        while (end < lines.length) {
          const nextHeading = lines[end].match(/^(#{1,6})\s+(.+)$/);
          if (nextHeading && nextHeading[1].length <= level) break;
          end += 1;
        }
        const body = lines.slice(start, end).join("\n").trim();
        const bodyHtml = body
          ? renderMarkdown(body, { ...options, collapsedSections: [] })
          : "<p>暂无内容</p>";
        html.push(
          `<details class="md-collapsible-section md-collapsible-artifacts">`
          + `<summary><span>${renderInlineMarkdown(heading[2], options)}</span><small>点击展开</small></summary>`
          + `<div class="md-collapsible-body">${bodyHtml}</div>`
          + `</details>`
        );
        i = end;
        continue;
      }
      html.push(`<h${level}>${renderInlineMarkdown(heading[2], options)}</h${level}>`);
      i += 1;
      continue;
    }

    if (isHorizontalRule(line)) {
      html.push("<hr>");
      i += 1;
      continue;
    }

    if (i + 1 < lines.length && isMarkdownTableSep(lines[i + 1])) {
      const headers = parseMarkdownCells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        rows.push(parseMarkdownCells(lines[i]));
        i += 1;
      }
      html.push(
        `<div class="md-table-wrap"><table><thead><tr>${headers.map(cell => `<th>${renderInlineMarkdown(cell, options)}</th>`).join("")}</tr></thead>`
        + `<tbody>${rows.map(row => `<tr>${headers.map((_, index) => `<td>${renderInlineMarkdown(row[index] || "", options)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`
      );
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        const parts = [lines[i].replace(/^\s*[-*+]\s+/, "")];
        i += 1;
        while (i < lines.length && !isBlockStartAt(i)) {
          parts.push(lines[i]);
          i += 1;
        }
        items.push(parts);
        while (i < lines.length && !lines[i].trim()) i += 1;
      }
      html.push(`<ul>${items.map(item => `<li>${renderListItem(item)}</li>`).join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        const parts = [lines[i].replace(/^\s*\d+\.\s+/, "")];
        i += 1;
        while (i < lines.length && !isBlockStartAt(i)) {
          parts.push(lines[i]);
          i += 1;
        }
        items.push(parts);
        while (i < lines.length && !lines[i].trim()) i += 1;
      }
      html.push(`<ol>${items.map(item => `<li>${renderListItem(item)}</li>`).join("")}</ol>`);
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quote = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        quote.push(lines[i].replace(/^>\s?/, ""));
        i += 1;
      }
      html.push(`<blockquote>${quote.map(part => `<p>${renderInlineMarkdown(part, options)}</p>`).join("")}</blockquote>`);
      continue;
    }

    const para = [line];
    i += 1;
    while (i < lines.length && !isBlockStartAt(i)) {
      para.push(lines[i]);
      i += 1;
    }
    html.push(`<p>${renderInlineMarkdown(para.join(" "), options)}</p>`);
  }
  return html.join("\n");
}

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
  highlightPythonCodeBlocks();
};

const copyTextToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
  }
};

const copyTritonCode = async () => {
  if (!tritonCodeContent.value) return;
  await copyTextToClipboard(tritonCodeContent.value);
  showToast("已复制到剪贴板", "success");
};

const copyAiCodeViewer = async () => {
  if (!aiCodeViewerContent.value) return;
  await copyTextToClipboard(aiCodeViewerContent.value);
  showToast("代码已复制到剪贴板", "success");
};

const copyErrorModal = async () => {
  if (!errorModalMsg.value) return;
  await copyTextToClipboard(errorModalMsg.value);
  showToast("已复制到剪贴板", "success");
};

// ══════════════════════════════════════════════════════════════════════════════
// Compare
// ══════════════════════════════════════════════════════════════════════════════

const clearBatchCompareSelection = () => {
  batchBaselineId.value = "";
  batchCandidateIds.value = [];
  batchSelectionDetails.value = {};
  batchCompareLabelPrefix.value = "";
};

const setBatchCompareMode = enabled => {
  batchCompareMode.value = enabled;
  if (enabled) {
    compareSelection.value = [];
    compareSelectionDetails.value = {};
    compareLabel.value = "";
  } else {
    clearBatchCompareSelection();
  }
};

const isCompareJobSelected = job => {
  if (!job) return false;
  if (!batchCompareMode.value) return compareSelection.value.includes(job.id);
  return batchBaselineId.value === job.id || batchCandidateIds.value.includes(job.id);
};

const compareJobRoleLabel = job => {
  if (!batchCompareMode.value || !job) return "";
  if (batchBaselineId.value === job.id) return "基线";
  if (batchCandidateIds.value.includes(job.id)) return "候选";
  return "";
};

const toggleBatchCompareSelect = job => {
  const details = { ...batchSelectionDetails.value };
  details[job.id] = job;
  if (!batchBaselineId.value) {
    batchBaselineId.value = job.id;
    batchCandidateIds.value = batchCandidateIds.value.filter(id => id !== job.id);
  } else if (batchBaselineId.value === job.id) {
    batchBaselineId.value = "";
  } else if (batchCandidateIds.value.includes(job.id)) {
    batchCandidateIds.value = batchCandidateIds.value.filter(id => id !== job.id);
  } else {
    batchCandidateIds.value.push(job.id);
  }
  batchSelectionDetails.value = details;
};

const toggleCompareSelect = job => {
  if (!job.file_a_exists) return;
  if (batchCompareMode.value) {
    toggleBatchCompareSelect(job);
    return;
  }
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

const removeBatchBaseline = () => {
  batchBaselineId.value = "";
};

const removeBatchCandidate = id => {
  batchCandidateIds.value = batchCandidateIds.value.filter(selectedId => selectedId !== id);
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

const submitBatchCompare = async () => {
  if (!batchBaselineId.value || !batchCandidateIds.value.length || batchCompareLoading.value) return;
  batchCompareLoading.value = true;
  try {
    const r = await fetch("/api/jobs/batch-compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        baseline_job_id: batchBaselineId.value,
        candidate_job_ids: batchCandidateIds.value,
        label_prefix: batchCompareLabelPrefix.value,
        project_id: compareProjectId.value || null,
      }),
    });
    const payload = await r.json();
    if (!r.ok) {
      showToast("批量对比失败: " + (payload.detail || "服务器错误"), "error");
      return;
    }
    const jobs = payload.data || [];
    showToast(`已创建 ${payload.count || jobs.length} 个对比任务`, "success");
    clearBatchCompareSelection();
    compareProjectId.value = "";
    sidebarTab.value = "jobs";
    await refreshSidebarData();
    if (jobs[0]?.id) router.push({ path: `/job/${jobs[0].id}` });
  } catch (e) {
    showToast("批量对比失败: 网络或服务器错误", "error");
  } finally {
    batchCompareLoading.value = false;
  }
};

const openCompareSource = source => {
  if (!source?.id) return;
  router.push({ path: `/job/${source.id}` });
};

const analyzeCompareTraceSlot = async slot => {
  const normalizedSlot = String(slot || "").trim().toLowerCase();
  if (!["a", "b"].includes(normalizedSlot)) return;
  if (singleTraceAnalyzeLoadingSlot.value) return;
  if (!selectedJobId.value || selectedJob.value?.mode !== "compare") {
    showToast("当前任务不是对比任务", "error");
    return;
  }
  singleTraceAnalyzeLoadingSlot.value = normalizedSlot;
  try {
    const r = await fetch(`/api/jobs/${selectedJobId.value}/analyze-trace-slot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ slot: normalizedSlot }),
    });
    const job = await r.json().catch(() => ({}));
    if (!r.ok) {
      showToast(`单独分析 ${normalizedSlot.toUpperCase()} 失败: ` + (job.detail || "服务器错误"), "error");
      return;
    }
    showToast(`已创建 ${normalizedSlot.toUpperCase()} trace 单独分析任务`, "success");
    await refreshSidebarData();
    router.push({ path: `/job/${job.id}` });
  } catch (e) {
    showToast(`单独分析 ${normalizedSlot.toUpperCase()} 失败: 网络错误`, "error");
  } finally {
    singleTraceAnalyzeLoadingSlot.value = "";
  }
};

const rerunCompareSwapped = async () => {
  if (compareRerunLoading.value) return;
  if (!selectedJobId.value || selectedJob.value?.mode !== "compare") {
    showToast("当前任务不是对比任务", "error");
    return;
  }
  compareRerunLoading.value = true;
  try {
    const r = await fetch(`/api/jobs/${selectedJobId.value}/rerun-swapped`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
    });
    const job = await r.json().catch(() => ({}));
    if (!r.ok) {
      showToast("重新对比失败: " + (job.detail || "服务器错误"), "error");
      return;
    }
    showToast("已提交交换 A/B 对比", "success");
    await refreshSidebarData();
    router.push({ path: `/job/${job.id}` });
  } catch (e) {
    showToast("重新对比失败: 网络错误", "error");
  } finally {
    compareRerunLoading.value = false;
  }
};

const openStepReanalysisModal = () => {
  if (!selectedJobId.value || !selectedJob.value) {
    showToast("未选中任务", "error");
    return;
  }
  if (["pending", "running"].includes(selectedJob.value.status)) {
    showToast("任务仍在分析中，完成后再指定 Step 重分析", "error");
    return;
  }
  stepReanalysisLabel.value = "";
  stepReanalysisFilterA.value = selectedJob.value.step_filter_a || "";
  stepReanalysisFilterB.value = selectedJob.value.mode === "compare" ? (selectedJob.value.step_filter_b || "") : "";
  showStepReanalysisModal.value = true;
};

const closeStepReanalysisModal = () => {
  if (stepReanalysisLoading.value) return;
  showStepReanalysisModal.value = false;
};

const confirmStepReanalysis = async () => {
  if (!selectedJobId.value || !selectedJob.value || stepReanalysisLoading.value) return;
  const filterA = stepReanalysisFilterA.value.trim();
  const filterB = selectedJob.value.mode === "compare" ? stepReanalysisFilterB.value.trim() : "";
  if (selectedJob.value.mode === "single" && !filterA) {
    showToast("请指定要分析的 step", "error");
    return;
  }
  if (selectedJob.value.mode === "compare" && !filterA && !filterB) {
    showToast("请至少指定 A 或 B 的 step；留空的一侧会使用全部 step", "error");
    return;
  }

  stepReanalysisLoading.value = true;
  try {
    const r = await fetch(`/api/jobs/${selectedJobId.value}/reanalyze-steps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        label: stepReanalysisLabel.value.trim(),
        step_filter_a: filterA,
        step_filter_b: filterB,
      }),
    });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) {
      showToast("指定 Step 重分析失败: " + (payload.detail || "服务器错误"), "error");
      return;
    }
    showStepReanalysisModal.value = false;
    showToast("已创建指定 Step 重分析任务", "success");
    await refreshSidebarData();
    router.push({ path: `/job/${payload.id}` });
  } catch (e) {
    showToast("指定 Step 重分析失败: 网络错误", "error");
  } finally {
    stepReanalysisLoading.value = false;
  }
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
      is_public: newProjectShared.value,
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
  newProjectShared.value = false;
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

const escapeRegExp = value => String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const consoleLines = computed(() =>
  String(selectedJob.value?.console_out || "").split("\n")
);

const consoleSectionId = index => `console-section-${selectedJobId.value || "job"}-${index}`;

const cleanConsoleSectionTitle = line =>
  String(line || "")
    .replace(/^\s*=+\s*/, "")
    .replace(/\s*=+\s*$/, "")
    .trim();

const consoleSections = computed(() =>
  consoleLines.value
    .map((line, index) => ({ line, index }))
    .filter(item => /^\s*=+\s*.+?\s*=+\s*$/.test(item.line))
    .map(item => ({
      id: consoleSectionId(item.index),
      title: cleanConsoleSectionTitle(item.line),
    }))
);

const consoleWroteCount = computed(() =>
  consoleLines.value.filter(line => /^Wrote /.test(line)).length
);

const consoleSearchMatchCount = computed(() => {
  const q = consoleSearch.value.trim().toLowerCase();
  if (!q) return 0;
  return consoleLines.value.reduce((count, line) => {
    if (consoleHideWrote.value && /^Wrote /.test(line)) return count;
    const text = line.toLowerCase();
    let offset = 0;
    let next = text.indexOf(q, offset);
    while (next >= 0) {
      count += 1;
      offset = next + q.length;
      next = text.indexOf(q, offset);
    }
    return count;
  }, 0);
});

const decorateConsoleText = (line, search = "") => {
  const q = search.trim();
  const numberPattern = "[+-]\\d+(?:\\.\\d+)?%?|\\b\\d+(?:\\.\\d+)?%?";
  const regex = q
    ? new RegExp(`(${escapeRegExp(q)})|(${numberPattern})`, "gi")
    : new RegExp(`(${numberPattern})`, "g");
  let html = "";
  let last = 0;
  for (const match of line.matchAll(regex)) {
    const text = match[0];
    const start = match.index ?? 0;
    if (start < last) continue;
    html += escapeHtml(line.slice(last, start));
    const isSearchHit = Boolean(q && match[1]);
    if (isSearchHit) {
      html += `<mark class="console-search-hit">${escapeHtml(text)}</mark>`;
    } else if (text.startsWith("+")) {
      html += `<span class="co-delta-up">${escapeHtml(text)}</span>`;
    } else if (text.startsWith("-")) {
      html += `<span class="co-delta-down">${escapeHtml(text)}</span>`;
    } else {
      html += `<span class="co-num">${escapeHtml(text)}</span>`;
    }
    last = start + text.length;
  }
  html += escapeHtml(line.slice(last));
  return html;
};

const formatConsole = (text, options = {}) => {
  if (!text) return "";
  const search = options.search || "";
  const hideWrote = options.hideWrote ?? false;
  let foldedWrote = false;
  return String(text).split("\n").map((line, index) => {
    if (hideWrote && /^Wrote /.test(line)) {
      if (foldedWrote) return null;
      foldedWrote = true;
      return `<span class="co-folded">已折叠 ${consoleWroteCount.value} 条生成文件输出，可在工具栏展开</span>`;
    }
    const decorated = decorateConsoleText(line, search);
    if (/^\s*=+\s*.+?\s*=+\s*$/.test(line)) {
      return `<span id="${consoleSectionId(index)}" class="co-hdr">${decorated}</span>`;
    }
    if (/^-{5,}/.test(line))  return `<span class="co-sep">${escapeHtml(line)}</span>`;
    if (/^Wrote /.test(line)) return `<span class="co-wrote">${decorateConsoleText(line, search)}</span>`;
    if (/^\s*$/.test(line))   return escapeHtml(line);
    return `<span class="co-line">${decorated}</span>`;
  }).filter(line => line !== null).join("\n");
};

const scrollConsoleSection = section => {
  const el = document.getElementById(section.id);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
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
            <div class="upload-label">
              <span>{{ fileAName || '选择文件' }}</span>
              <small v-if="uploadQueue.length===1">{{ uploadQueue[0].meta }}</small>
            </div>
          </div>
          <button v-if="fileAName" class="upload-clear" @click.stop="clearFile">✕</button>
        </div>
        <div class="form-row">
          <label>项目</label>
          <select v-model="form.projectId" class="input">
            <option value="">未分组</option>
            <option v-for="p in projects" :key="p.id" :value="p.id">{{ projectOptionLabel(p) }}</option>
          </select>
        </div>
        <div class="form-row">
          <label>别名</label>
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
              <div class="upload-label">
                <span>{{ quickFileAName || '选择 A trace' }}</span>
                <small v-if="quickFileA">{{ uploadFileMeta(quickFileA) }}</small>
              </div>
            </div>
            <button v-if="quickFileAName" class="upload-clear" @click.stop="clearQuickCompareFile('a')">✕</button>
          </div>
          <div class="upload-box upload-box-sm quick-upload-box" @dragover.prevent @drop.prevent="onQuickDrop('b', $event)">
            <input type="file" ref="quickFileInputB" accept=".json,.json.gz,.gz,.zip,.tar.gz,.tgz" @change="onQuickFileChange('b', $event)" hidden />
            <div @click="$refs.quickFileInputB.click()" class="upload-inner">
              <span class="trace-slot">B</span>
              <div class="upload-label">
                <span>{{ quickFileBName || '选择 B trace' }}</span>
                <small v-if="quickFileB">{{ uploadFileMeta(quickFileB) }}</small>
              </div>
            </div>
            <button v-if="quickFileBName" class="upload-clear" @click.stop="clearQuickCompareFile('b')">✕</button>
          </div>
        </div>
        <div class="form-row">
          <label>项目</label>
          <select v-model="form.projectId" class="input">
            <option value="">未分组</option>
            <option v-for="p in projects" :key="p.id" :value="p.id">{{ projectOptionLabel(p) }}</option>
          </select>
        </div>
        <div class="form-row">
          <label>别名</label>
          <input v-model="form.label" class="input" placeholder="默认 A vs B" />
        </div>
        <button class="btn btn-primary" :disabled="!quickFileA || !quickFileB || submitting" @click="submitQuickCompare">
          {{ submitting ? '提交中 ' + uploadProgress + '%' : '提交对比' }}
        </button>
      </div>

      <div v-if="quickUploadMode==='single' && uploadQueue.length" class="upload-queue">
        <div v-for="item in uploadQueue" :key="item.id" class="upload-queue-item">
          <span class="upload-queue-main">
            <span class="upload-queue-name" :title="item.name">{{ item.name }}</span>
            <span class="upload-queue-meta">{{ item.meta }}</span>
          </span>
          <span :class="['upload-queue-status', 'queue-' + item.status]">
            <template v-if="item.status==='ready'">待提交</template>
            <template v-else-if="item.status==='uploading'">上传中 {{ item.progress }}%</template>
            <template v-else-if="item.status==='submitted'">已提交</template>
            <template v-else>{{ item.error || '失败' }}</template>
          </span>
        </div>
      </div>
      <div v-if="quickUploadMode==='compare' && (quickFileAName || quickFileBName)" class="quick-compare-summary">
        <span :class="['quick-file-chip', quickFileAName ? 'ready' : '']">A {{ quickFileAName || '未选择' }} <small v-if="quickFileA">{{ uploadFileMeta(quickFileA) }}</small></span>
        <span :class="['quick-file-chip', quickFileBName ? 'ready' : '']">B {{ quickFileBName || '未选择' }} <small v-if="quickFileB">{{ uploadFileMeta(quickFileB) }}</small></span>
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
      <div class="empty-main-title">上传 trace 开始分析，或从左侧历史记录继续</div>
      <div class="empty-action-grid">
        <button class="empty-action-card" type="button" @click="openSingleUploadPicker">
          <strong>上传单文件/批量</strong>
          <span>分析一个或多个 trace，自动进入任务队列</span>
        </button>
        <button class="empty-action-card" type="button" @click="setQuickUploadMode('compare')">
          <strong>上传两个 trace 快速对比</strong>
          <span>直接生成 A/B 对比任务</span>
        </button>
        <button class="empty-action-card" type="button" @click="sidebarTab='jobs'">
          <strong>{{ historyGroupsTotal ? '查看历史与共享项目' : '等待历史记录' }}</strong>
          <span>{{ historyGroupsTotal ? '左侧可搜索、置顶和批量管理任务' : '提交成功后会在左侧出现' }}</span>
        </button>
        <button class="empty-action-card" type="button" @click="showGuide=true">
          <strong>打开使用指南</strong>
          <span>查看上传、对比、AI 和社区说明</span>
        </button>
        <button class="empty-action-card" type="button" @click="$router.push('/feedback')">
          <strong>进入灵感社区</strong>
          <span>提建议、看讨论、@ 同事一起完善工具</span>
        </button>
      </div>
      <div class="empty-main-tips">
        <div class="empty-tip-item">支持 .json.gz、.gz、.json.zip、.zip、.json、.tar.gz、.tgz，默认下载为 .json.gz</div>
        <div class="empty-tip-item">建议 trace 中开启 triton code 保存功能，后续可直接查看和运行 Triton kernel</div>
      </div>
    </div>
  `,
  setup() {
    const fileInputA = ref(null);
    const openSingleUploadPicker = async () => {
      setQuickUploadMode("single");
      await nextTick();
      fileInputA.value?.click();
    };
    return {
      fileInputA, fileAName, fileA, quickUploadMode,
      quickFileA, quickFileB, quickFileAName, quickFileBName,
      uploadQueue, submitting, uploadProgress,
      form, projects, projectOptionLabel, selectedJob,
      historyGroupsTotal, sidebarTab, showGuide, uploadFileMeta,
      openSingleUploadPicker,
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
        <div class="result-title-row">
          <span class="job-status lg" :class="'status-'+selectedJob.status">
            {{ statusIcon(selectedJob.status) }}
          </span>
          <span class="result-label">{{ selectedJob.label }}</span>
          <span v-if="selectedJob.is_pinned" class="job-pin-badge">置顶</span>
          <span class="job-mode-badge" :class="'mode-'+selectedJob.mode">
            {{ selectedJob.mode==='compare'?'对比':'单文件' }}
          </span>
          <span v-if="jobStepFilterLabel" class="job-step-badge">{{ jobStepFilterLabel }}</span>
        </div>
        <div class="result-actions action-menu-wrap" @click.stop>
          <button class="action-icon-btn" type="button" title="更多任务操作"
                  aria-label="更多任务操作"
                  @click="toggleActionMenu('job')">...</button>
          <div v-if="openActionMenu==='job'" class="action-menu">
            <button type="button" @click="shareJob(); closeActionMenu()">复制分享链接</button>
            <button type="button" @click="downloadReport(); closeActionMenu()">导出报告</button>
            <button v-if="selectedJob.is_owner !== false" type="button" @click="togglePinJob(); closeActionMenu()">
              {{ selectedJob.is_pinned ? '取消置顶' : '置顶' }}
            </button>
            <button v-if="selectedJob.is_owner !== false" type="button" @click="editLabel(); closeActionMenu()">重命名</button>
            <button v-if="selectedJob.is_owner !== false" type="button" @click="moveProject(); closeActionMenu()">移动项目</button>
            <button type="button" @click="openStepReanalysisModal(); closeActionMenu()">指定 Step 重分析</button>
            <button v-if="selectedJob.is_owner !== false" type="button" class="danger" @click="deleteJob(); closeActionMenu()">删除任务</button>
          </div>
        </div>
      </div>

      <!-- File info -->
      <div class="file-info">
        <div v-if="selectedJob.file_a_name" class="trace-file-row">
          <span v-if="selectedJob.mode==='compare'" class="trace-slot">A</span>
          <span class="trace-file-name" :title="selectedJob.file_a_name">📄 {{ selectedJob.file_a_name }}</span>
          <span v-if="!selectedJob.file_a_exists" class="tag-deleted">已删除</span>
          <div v-else class="trace-file-actions">
            <button v-if="allowFileDownload" class="btn btn-xs btn-perfetto"
                    :disabled="perfettoOpening.a"
                    @click="openInPerfetto('a')">{{ perfettoButtonLabel('a') }}</button>
            <div v-if="allowFileDownload || selectedJob.is_owner !== false"
                 class="action-menu-wrap trace-action-menu" @click.stop>
              <button class="action-icon-btn action-icon-btn-xs" type="button"
                      title="更多文件操作" aria-label="更多文件操作"
                      @click="toggleActionMenu('file-a')">...</button>
              <div v-if="openActionMenu==='file-a'" class="action-menu action-menu-sm">
                <button v-if="allowFileDownload" type="button"
                        @click="downloadTraceFile('a'); closeActionMenu()">下载</button>
                <button v-if="selectedJob.is_owner !== false" type="button" class="danger"
                        @click="deleteFile('a'); closeActionMenu()">删除文件</button>
              </div>
            </div>
          </div>
        </div>
        <div v-if="selectedJob.file_b_name" class="trace-file-row">
          <span v-if="selectedJob.mode==='compare'" class="trace-slot">B</span>
          <span class="trace-file-name" :title="selectedJob.file_b_name">📄 {{ selectedJob.file_b_name }}</span>
          <span v-if="!selectedJob.file_b_exists" class="tag-deleted">已删除</span>
          <div v-else class="trace-file-actions">
            <button v-if="allowFileDownload" class="btn btn-xs btn-perfetto"
                    :disabled="perfettoOpening.b"
                    @click="openInPerfetto('b')">{{ perfettoButtonLabel('b') }}</button>
            <div v-if="allowFileDownload || selectedJob.is_owner !== false"
                 class="action-menu-wrap trace-action-menu" @click.stop>
              <button class="action-icon-btn action-icon-btn-xs" type="button"
                      title="更多文件操作" aria-label="更多文件操作"
                      @click="toggleActionMenu('file-b')">...</button>
              <div v-if="openActionMenu==='file-b'" class="action-menu action-menu-sm">
                <button v-if="allowFileDownload" type="button"
                        @click="downloadTraceFile('b'); closeActionMenu()">下载</button>
                <button v-if="selectedJob.is_owner !== false" type="button" class="danger"
                        @click="deleteFile('b'); closeActionMenu()">删除文件</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-if="selectedJob.mode==='compare' && selectedJob.compare_sources" class="compare-source-panel">
        <div class="compare-source-head">
          <span>来源</span>
          <button class="btn btn-xs btn-outline"
                  :disabled="compareRerunLoading"
                  @click="rerunCompareSwapped">
            {{ compareRerunLoading ? '提交中...' : '交换 A/B 重新对比' }}
          </button>
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
            <div class="compare-source-actions">
              <button class="btn btn-xs btn-outline"
                      :disabled="singleTraceAnalyzeLoadingSlot === slot"
                      @click="analyzeCompareTraceSlot(slot)">
                {{ singleTraceAnalyzeLoadingSlot === slot ? '提交中...' : '单独分析 ' + slot.toUpperCase() }}
              </button>
              <button v-if="selectedJob.compare_sources[slot]" class="btn btn-xs btn-outline"
                      @click="openCompareSource(selectedJob.compare_sources[slot])">查看源任务</button>
            </div>
          </div>
        </div>
      </div>

      <div v-if="selectedJob.status==='running' || selectedJob.status==='pending'" class="loading">
        <span class="spinner"></span> {{ selectedJob.console_out || '分析中...' }}
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
                  :title="isReadingMode ? '退出全屏' : '进入全屏'"
                  @click="toggleReadingMode">
            {{ isReadingMode ? '退出全屏' : '全屏' }}
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
          <div class="console-tools">
            <div class="console-search-wrap">
              <input v-model="consoleSearch" class="console-search-input" placeholder="搜索控制台..." />
              <span class="console-search-count">{{ consoleSearch.trim() ? consoleSearchMatchCount + ' 处' : '搜索' }}</span>
            </div>
            <button v-if="consoleWroteCount" class="console-tool-btn" @click="consoleHideWrote=!consoleHideWrote">
              {{ consoleHideWrote ? '展开生成文件' : '折叠生成文件' }} · {{ consoleWroteCount }}
            </button>
            <div v-if="consoleSections.length" class="console-section-nav">
              <button v-for="section in consoleSections" :key="section.id"
                      class="console-section-btn"
                      :title="section.title"
                      @click="scrollConsoleSection(section)">
                {{ section.title }}
              </button>
            </div>
          </div>
          <pre v-html="formatConsole(selectedJob.console_out, { search: consoleSearch, hideWrote: consoleHideWrote })"></pre>
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
        <div v-if="resultTab==='ai'" class="ai-analysis-wrap">
          <div class="ai-analysis-head">
            <div class="ai-analysis-title-block">
              <div class="ai-analysis-title">Claude Code AI 分析</div>
              <div class="ai-analysis-sub">
                {{ selectedJob.mode === 'compare' ? '使用对比 skill 分析 A/B trace' : '使用单 trace skill 分析当前 trace' }}
              </div>
            </div>
            <div class="ai-analysis-actions">
              <span :class="['ai-status-badge', 'status-' + (aiAnalysisMeta.status || 'not_started')]">
                {{ aiAnalysisStatusText(aiAnalysisMeta.status) }}
              </span>
              <button class="btn btn-sm btn-outline"
                      :disabled="aiAnalysisLoading"
                      @click="refreshAiAnalysis()">
                {{ aiAnalysisLoading ? '刷新中...' : '刷新' }}
              </button>
              <button v-if="aiAnalysisContent" class="btn btn-sm btn-outline" @click="copyAiAnalysisReport">
                复制
              </button>
              <button v-if="aiAnalysisContent" class="btn btn-sm btn-outline" @click="downloadAiAnalysisReport">
                下载 Markdown
              </button>
              <button class="btn btn-sm btn-primary"
                      :disabled="!claudeAnalysisEnabled || aiAnalysisStarting || isAiAnalysisActive(aiAnalysisMeta.status)"
                      @click="openAiPromptModal(aiAnalysisMeta.report_exists || aiAnalysisMeta.status==='done')">
                {{ aiAnalysisStarting ? '提交中...' : (aiAnalysisMeta.report_exists ? '重新分析' : '开始分析') }}
              </button>
            </div>
          </div>

          <div v-if="aiAnalysisSelectedVersion || aiAnalysisMeta.started_at || isAiAnalysisActive(aiAnalysisMeta.status)" class="ai-meta-row">
            <label v-if="aiAnalysisVersions.length > 1" class="ai-version-picker">
              <span>历史版本</span>
              <select v-model="aiAnalysisSelectedVersionId"
                      class="input input-sm ai-version-select"
                      @change="changeAiAnalysisVersion">
                <option v-for="version in aiAnalysisVersions"
                        :key="version.id"
                        :value="version.id">
                  {{ aiAnalysisVersionLabel(version) }}
                </option>
              </select>
            </label>
            <div v-else-if="aiAnalysisSelectedVersion" class="ai-version-static">
              <span>历史版本</span>
              <strong>最新版本</strong>
            </div>
            <div v-if="aiAnalysisSelectedVersion" class="ai-version-info">
              <span>生成时间 <strong>{{ fmtDateTime(aiAnalysisSelectedVersion.generated_at || aiAnalysisSelectedVersion.finished_at) || '-' }}</strong></span>
              <span>触发人 <strong>{{ aiAnalysisVersionTrigger }}</strong></span>
              <span>模型 <strong>{{ aiAnalysisVersionModel }}</strong></span>
            </div>
            <div v-else class="ai-version-info ai-version-info-muted">报告生成后会显示版本信息</div>
            <div v-if="aiAnalysisMeta.started_at || isAiAnalysisActive(aiAnalysisMeta.status)" class="ai-duration-meta">
              <span>{{ isAiAnalysisActive(aiAnalysisMeta.status) ? '已耗时' : '总耗时' }}</span>
              <strong>{{ aiAnalysisElapsedText }}</strong>
            </div>
          </div>
          <div v-if="aiAnalysisSelectedVersion?.user_prompt" class="ai-version-prompt">
            <strong>本版本补充 Prompt</strong>
            <pre :title="aiAnalysisSelectedVersion.user_prompt">{{ aiAnalysisSelectedVersion.user_prompt }}</pre>
          </div>

          <div v-if="!claudeAnalysisEnabled && !aiAnalysisMeta.report_exists" class="info-box">
            AI 分析未启用。服务端设置 TRACE_ENABLE_CLAUDE_ANALYSIS=1 后可使用。
          </div>
          <div v-if="aiDiagnosticsError" class="error-box mb-2">{{ aiDiagnosticsError }}</div>
          <div v-if="aiDiagnosticsResult && !aiDiagnosticsResult.ok" class="ai-diagnostics-panel">
            <div class="ai-diagnostics-head">
              <div>
                <strong>AI 环境诊断</strong>
                <span :class="['ai-diagnostic-overall', aiDiagnosticsResult.ok ? 'ok' : 'error']">
                  {{ aiDiagnosticsResult.ok ? '通过' : '未通过' }}
                </span>
              </div>
              <button class="btn btn-xs btn-outline" @click="copyAiDiagnostics">复制诊断</button>
            </div>
            <div class="ai-diagnostic-meta">
              <span>skills: {{ aiDiagnosticsResult.skills_dir || '-' }}</span>
              <span>耗时 {{ aiDiagnosticsResult.duration_ms }} ms</span>
            </div>
            <div class="ai-diagnostic-checks">
              <div v-for="check in aiDiagnosticsResult.checks || []"
                   :key="check.name"
                   :class="['ai-diagnostic-check', 'status-' + check.status]">
                <div class="ai-diagnostic-check-main">
                  <span class="ai-diagnostic-status">{{ aiDiagnosticStatusText(check.status) }}</span>
                  <strong>{{ check.label || check.name }}</strong>
                  <span>{{ check.detail }}</span>
                </div>
                <pre v-if="check.stdout_tail || check.stderr_tail" class="ai-diagnostic-output">{{ [check.stdout_tail ? 'stdout:\\n' + check.stdout_tail : '', check.stderr_tail ? 'stderr:\\n' + check.stderr_tail : ''].filter(Boolean).join('\\n\\n') }}</pre>
              </div>
            </div>
          </div>
          <div v-if="aiAnalysisError" class="error-box mb-2">{{ aiAnalysisError }}</div>
          <div v-if="isAiAnalysisActive(aiAnalysisMeta.status)" class="ai-analysis-running">
            <span class="spinner-small"></span>
            {{ aiAnalysisMeta.status === 'queued'
              ? 'AI 分析已进入队列，开始后会自动更新状态。'
              : aiAnalysisMeta.phase === 'diagnosing'
              ? '正在进行 AI 环境诊断，诊断通过后会自动开始分析。'
              : 'Claude Code 正在分析 trace，完成后这里会自动刷新。' }}
          </div>
          <div v-if="aiAnalysisContent"
               class="ai-analysis-report markdown-body"
               @click="handleAiAnalysisReportClick"
               v-html="aiAnalysisHtml"></div>
          <div v-if="aiAnalysisVisibleArtifacts.length"
               :class="['ai-artifacts-panel', { collapsed: !aiArtifactsExpanded }]">
            <button type="button"
                    class="ai-artifacts-toggle"
                    :aria-expanded="String(aiArtifactsExpanded)"
                    @click="aiArtifactsExpanded = !aiArtifactsExpanded">
              <div>
                <strong>分析产物</strong>
                <span>{{ aiAnalysisVisibleArtifacts.length }} 个文本文件</span>
              </div>
              <span class="ai-artifacts-summary">{{ aiArtifactSummary }}</span>
              <span class="ai-artifacts-toggle-label">{{ aiArtifactsExpanded ? '收起' : '展开' }}</span>
            </button>
            <div v-if="aiArtifactsExpanded" class="ai-artifact-list">
              <div v-for="artifact in aiAnalysisVisibleArtifacts"
                   :key="artifact.path"
                   class="ai-artifact-card">
                <div class="ai-artifact-head">
                  <div class="ai-artifact-title">{{ artifact.path }}</div>
                  <div class="ai-artifact-meta">
                    <span>{{ fmtBytes(artifact.size) }}</span>
                    <span v-if="artifact.truncated">已截断</span>
                    <button class="btn btn-xs btn-outline"
                            @click="downloadAiAnalysisArtifact(artifact)">
                      下载
                    </button>
                    <button v-if="artifact.content"
                            class="btn btn-xs btn-outline"
                            @click="copyAiAnalysisArtifact(artifact)">
                      复制
                    </button>
                  </div>
                </div>
                <pre class="ai-artifact-content">{{ artifact.content || '(空文件)' }}</pre>
              </div>
            </div>
          </div>
          <div v-if="!aiAnalysisContent && !aiAnalysisVisibleArtifacts.length && !isAiAnalysisActive(aiAnalysisMeta.status)" class="ai-analysis-empty">
            点击“开始分析”后，会调用服务端 Claude Code 和自定义 skill 生成报告。
          </div>
        </div>

        <!-- CSV table tabs -->
        <div v-if="resultTab!=='console' && resultTab!=='chart' && resultTab!=='ai'" class="table-wrap">
          <div class="table-toolbar">
            <div class="table-toolbar-main">
              <input v-model="tableSearch" class="input input-sm table-search-input" placeholder="全局搜索..." />
              <span v-if="hasColFilters" class="filter-active-tip">
                列筛选已启用
                <button class="btn-clear-filter" @click="clearColFilters()">✕ 清除</button>
              </span>
              <span v-if="isKernelTypeTab" class="filter-active-tip">点击类型行下钻到相关 Kernel</span>
            </div>
            <div class="table-toolbar-actions">
              <div class="column-menu-wrap" @click.stop>
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
              <div class="action-menu-wrap table-more-menu" @click.stop>
                <button class="action-icon-btn" type="button" title="更多表格操作"
                        aria-label="更多表格操作"
                        @click="toggleActionMenu('table')">...</button>
                <div v-if="openActionMenu==='table'" class="action-menu">
                  <button type="button" @click="downloadCsv(resultTab); closeActionMenu()">下载当前页 CSV</button>
                  <button v-if="isTritonStepTab && allowCodeExecution" type="button"
                          @click="clearInductorCache(); closeActionMenu()">清除 Cache</button>
                </div>
              </div>
            </div>
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
                <tr v-for="(row,i) in filteredRows" :key="i"
                    :class="{ 'drill-row': canDrillKernelTypeRow(row) }"
                    @click="drillDownKernelType(row)">
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
      jobStepFilterLabel,
      isReadingMode, toggleReadingMode,
      compareRerunLoading, openStepReanalysisModal,
      consoleSearch, consoleHideWrote, consoleSections, consoleWroteCount,
      consoleSearchMatchCount, scrollConsoleSection,
      chartSource, chartMetric, chartTopN, chartTopNOptions, chartSourceOptions,
      chartMetricOptions, chartLoading, chartError, chartSummaryCards,
      chartSlowdowns, chartSpeedups, buildChart, drillDownChart, fmtDeltaMs,
      displayedFields, filteredRows, tableSearch, sortCol, sortAsc, colWidths, colFilters,
      colFilterOps, visibleColumns, showColumnMenu, hiddenColumnCount,
      tableLimit, tableOffset, tableTotalRows, tablePageStart, tablePageEnd,
      tablePageSizeOptions, customTableLimit, changeTableLimit, showAllTableRows,
      resultTableLoading, resultTableError, preparingResultTab, prevTablePage, nextTablePage,
      hasColFilters, colSums, isKernelTypeTab, canDrillKernelTypeRow, drillDownKernelType,
      isTritonStepTab, tritonStatus, allowFileDownload, allowCodeExecution,
      claudeAnalysisEnabled, aiAnalysisMeta, aiAnalysisLoading, aiAnalysisStarting,
      aiAnalysisError, aiAnalysisContent, aiAnalysisArtifacts, aiAnalysisVisibleArtifacts,
      aiAnalysisVersions, aiAnalysisSelectedVersionId, aiAnalysisSelectedVersion,
      showAiPromptModal, aiAnalysisPrompt, aiPromptForce,
      aiAnalysisVersionTrigger, aiAnalysisVersionModel, aiAnalysisVersionLabel,
      aiArtifactsExpanded, aiArtifactSummary, aiAnalysisHtml, aiAnalysisStatusText,
      isAiAnalysisActive, aiAnalysisElapsedText,
      aiDiagnosticsLoading, aiDiagnosticsError, aiDiagnosticsResult, aiDiagnosticStatusText,
      refreshAiAnalysis, startAiAnalysis, openAiPromptModal, closeAiPromptModal, confirmAiPromptModal,
      copyAiAnalysisReport,
      downloadAiAnalysisReport, changeAiAnalysisVersion, copyAiAnalysisArtifact, downloadAiAnalysisArtifact,
      handleAiAnalysisReportClick,
      runAiDiagnostics, copyAiDiagnostics,
      openActionMenu, toggleActionMenu, closeActionMenu,
      switchTab,
      statusIcon,
      shareJob, togglePinJob, editLabel, moveProject, deleteJob, deleteFile,
      openCompareSource, rerunCompareSwapped, singleTraceAnalyzeLoadingSlot, analyzeCompareTraceSlot,
      downloadTraceFile, downloadReport, openInPerfetto, perfettoOpening, perfettoButtonLabel,
      setSort, startResize, downloadCsv,
      viewTritonCode, runSingleTriton, clearInductorCache,
      fmtDate, fmtDateTime, fmtSum, fmtBytes, deltaCellClass, clearColFilters,
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
    { path: "/feedback/:postId?", name: "feedback", component: Home },
    { path: "/job/:id", component: JobDetail },
    { path: "/job/:id/:tab", component: JobDetail },
  ],
});

const resetJobRuntimeState = () => {
  isReadingMode.value = false;
  if (ktChartInst.value)     { ktChartInst.value.destroy();     ktChartInst.value = null; }
  if (ktPieChartInst.value)  { ktPieChartInst.value.destroy();  ktPieChartInst.value = null; }
  if (ktPieChartInstB.value) { ktPieChartInstB.value.destroy(); ktPieChartInstB.value = null; }
  stopAiAnalysisPolling();
  cancelResultTableRequest();
  clearAiDiagnostics();
};

const clearSelectedJobRoute = () => {
  saveResultViewState();
  resetJobRuntimeState();
  clearInterval(pollTimer);
  pollTimer = null;
  selectedJobId.value = null;
  selectedJob.value = null;
  jobLoading.value = false;
  resultTab.value = DEFAULT_RESULT_TAB;
  resultTableFile.value = "";
  activeResultStateJobId = null;
};

const loadJobRoute = async to => {
  const newJobId = to.params?.id || null;
  if (!newJobId) return true;

  saveResultViewState();
  resetJobRuntimeState();

  selectedJobId.value = newJobId;
  selectedJob.value = null;
  jobLoading.value = true;
  resultTableFile.value = "";

  let loaded;
  try {
    loaded = await loadJob(newJobId);
  } catch (e) {
    jobLoading.value = false;
    const message = normalizeApiError(e, "加载任务失败");
    if (!e?.authExpired) showToast(message, "error");
    return false;
  }

  if (loaded === "stale") {
    jobLoading.value = false;
    return false;
  }
  if (!loaded) {
    selectedJobId.value = null;
    clearAiDiagnostics();
    jobLoading.value = false;
    return { path: "/" };
  }

  const requestedTab = to.params?.tab || "";
  const validTabs = availableTabs.value.map(t => t.key);
  const targetTab = resolveResultTab(newJobId, requestedTab, validTabs);
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
    scheduleBuildChart();
  }
  if (targetTab === "ai" && selectedJob.value.status === "done") {
    refreshAiAnalysis({ silent: true });
  }
  if (selectedJob.value.status === "pending" || selectedJob.value.status === "running") {
    startPoll();
  }
  sidebarTab.value = "jobs";
  jobLoading.value = false;
  return true;
};

const resumeCurrentRouteAfterLogin = async () => {
  const route = router.currentRoute.value;
  if (route.name === "feedback") {
    await openFeedbackRoute(route);
    return;
  }
  if (route.params?.id) {
    const result = await loadJobRoute(route);
    if (result && result !== true) await router.replace(result);
  }
};

const retryInitializeApp = async () => {
  authInitError.value = "";
  authChecked.value = false;
  appInitialized = false;
  const ready = await initializeAppData();
  if (ready) await resumeCurrentRouteAfterLogin();
};

// ══════════════════════════════════════════════════════════════════════════════
// Navigation guard
// ══════════════════════════════════════════════════════════════════════════════

router.beforeEach(async (to, from) => {
  // Ensure config/data is loaded on first navigation
  if (!appInitialized) {
    const ready = await initializeAppData();
    if (!ready) {
      if (authInitError.value) return false;
      return;
    }
  }

  if (authRequired.value && !currentUser.value) return;

  if (to.name === "feedback") {
    await openFeedbackRoute(to);
    return;
  }
  if (showFeedbackBoard.value) {
    showFeedbackBoard.value = false;
    selectedFeedbackPostId.value = "";
    selectedFeedbackMessageId.value = "";
    cancelFeedbackEdit();
    closeFeedbackMention();
    destroyFeedbackMarkdownEditors();
  }

  const newJobId = to.params?.id || null;

  if (!newJobId) {
    // Navigated to home -- clean up
    clearSelectedJobRoute();
    return;
  }

  const requestedTabForSameJob = to.params?.tab || "";

  // Same job, just switch tab
  if (newJobId === selectedJobId.value) {
    const validTabs = availableTabs.value.map(t => t.key);
    const targetTab = resolveResultTab(newJobId, requestedTabForSameJob, validTabs);
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
  const result = await loadJobRoute(to);
  if (result !== true) return result;
});

// ══════════════════════════════════════════════════════════════════════════════
// Root App component (wraps the #app DOM template in index.html)
// ══════════════════════════════════════════════════════════════════════════════

const App = {
  setup() {
    let historySearchTimer = null;
    let compareSearchTimer = null;
    let resultTableTimer = null;
    const isFeedbackRoute = computed(() => router.currentRoute.value.name === "feedback");

    // Watchers that need to live at the root level
    watch(resultTab, (v, previousTab) => {
      if (suppressResultTabWatch) return;
      if (previousTab) saveResultViewState(activeResultStateJobId, previousTab);
      restoreResultViewState(selectedJobId.value, v);
      rememberResultTabSelection(selectedJobId.value, v);
      showColumnMenu.value = false;
      if (v?.endsWith(".csv")) loadResultTable();
      if (v === "chart" && selectedJob.value?.status === "done") {
        scheduleBuildChart();
      }
      if (v === "ai" && selectedJob.value?.status === "done") {
        refreshAiAnalysis({ silent: true });
      }
    });

    watch(selectedJob, v => {
      if (v?.status === "done" && resultTab.value === "chart") scheduleBuildChart();
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

    watch([batchBaselineId, batchCandidateIds], () => {
      if (!batchCompareMode.value || !batchBaselineId.value || !batchCandidateIds.value.length) return;
      const jobs = [selectedBatchBaseline.value, ...selectedBatchCandidates.value].filter(Boolean);
      const firstProject = jobs[0]?.project_id || "";
      compareProjectId.value = jobs.length && jobs.every(job => (job.project_id || "") === firstProject)
        ? firstProject
        : "";
    }, { deep: true });

    watch(sidebarWidth, value => localStorage.setItem("tpa-sidebar-width", String(value)));
    watch(sidebarCollapsed, value => localStorage.setItem("tpa-sidebar-collapsed", String(value)));
    watch(sidebarTab, value => localStorage.setItem("tpa-sidebar-tab", value));
    watch(consoleHideWrote, value => localStorage.setItem("tpa-console-hide-wrote", String(value)));
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

    watch([showFeedbackComposer, selectedFeedbackPostId], () => {
      refreshFeedbackMarkdownEditors();
    });
    watch(() => feedbackEditing.value.id, () => {
      refreshFeedbackMarkdownEditors();
    });
    onBeforeUnmount(() => {
      destroyFeedbackMarkdownEditors();
    });

    // Return everything the root template (index.html) needs
    return {
      // Layout/theme
      isDark, toggleTheme, sidebarWidth, sidebarCollapsed, appVersion, isFeedbackRoute,
      toggleSidebar, startSidebarResize,
      authRequired, authChecked, authInitError, currentUser, isAdmin, loginForm, loginRememberUsername, loginLoading, loginError,
      loginCaptchaRequired, loginCaptchaImage,
      retryInitializeApp, submitLogin, refreshLoginCaptcha, logout,

      // Sidebar data
      projects,
      selectedFilterProject, projectOptionLabel,
      historyGroupsTotal, historyGroupsLimit, historyGroupsOffset, historyGroupsLoading,
      historySearch, filterProject, sidebarTab, selectedJobId, selectedJob,
      collapsedGroups, groupedJobs, loadedHistoryJobIds,
      prevPage, nextPage, navigateToJob, loadHistoryGroupJobs,
      historyBulkMode, historySelection, toggleHistoryBulkMode,
      toggleSelectLoadedHistoryJobs, clearHistorySelection,
      handleHistoryJobClick, openBulkMoveProject, bulkDeleteFiles, bulkDeleteJobs,

      // Compare
      compareSelection, selectedCompareJobs, compareLabel, compareProjectId,
      batchCompareMode, batchBaselineId, selectedBatchBaseline, selectedBatchCandidates,
      batchCandidateIds, batchCompareLabelPrefix, batchCompareLoading,
      compareJobs, compareJobsTotal, compareJobsLimit, compareJobsOffset, compareJobsLoading, compareSearch,
      setBatchCompareMode, isCompareJobSelected, compareJobRoleLabel,
      toggleCompareSelect, removeCompareSelection, removeBatchBaseline, removeBatchCandidate,
      submitCompare, submitBatchCompare,
      prevComparePage, nextComparePage,

      // Modals
      showNewProject, newProjectName, newProjectDesc, newProjectShared,
      showRenameProject, renameProjectName, openRenameModal,
      confirmRenameProject, deleteProject, shareProject,
      showMoveProject, moveProjectTarget, confirmMoveProject,
      showBulkMoveProject, bulkMoveProjectTarget, confirmBulkMoveProject,
      showRenameJob, renameJobName, confirmRenameJob,
      showDeletedProjects, deletedProjects, loadDeletedProjects,
      isDeletedOver10Days, restoreProject, permanentlyDeleteProject,
      showFeedbackBoard, showFeedbackComposer, feedbackItems, feedbackTotal, feedbackLoading,
      feedbackSort, feedbackSortOptions,
      feedbackSubmitting, feedbackForm, feedbackPostEditorMode, feedbackReplies, feedbackEditing,
      feedbackMarkdownEditorEnabled, feedbackHasMore,
      feedbackEmailDiagLoading, feedbackEmailDiagResult, runFeedbackEmailDiagnostics,
      feedbackEmojiOptions, feedbackReactionOptions, feedbackReactionPickerId, feedbackEmojiPickerTarget,
      feedbackUserInitial, setFeedbackTextTarget, insertFeedbackEmoji, insertFeedbackSnippet,
      insertFeedbackList, insertFeedbackTaskList, insertFeedbackQuote, insertFeedbackCodeBlock, insertFeedbackCode,
      feedbackMention, handleFeedbackMentionInput, handleFeedbackMentionKeydown, selectFeedbackMention,
      selectedFeedbackPostId, selectedFeedbackMessageId, selectedFeedbackPost, feedbackDetailLoading,
      feedbackPostTitle, feedbackPostExcerpt, feedbackPostReplyCount, feedbackPostActivity,
      feedbackEditedText, feedbackReactionSummary, feedbackReactionItem, canEditFeedbackMessage,
      feedbackMessageHtml,
      openFeedbackBoard, refreshFeedbackBoard, loadFeedback, setFeedbackSort, setFeedbackFiles, clearFeedbackForm,
      toggleFeedbackReply, feedbackReplyEditorMode, setFeedbackPostEditorMode,
      setFeedbackReplyEditorMode, feedbackPostPreviewHtml, feedbackReplyPreviewHtml,
      selectFeedbackPost, closeFeedbackPost,
      openFeedbackComposer, closeFeedbackComposer, closeFeedbackBoard, submitFeedback,
      startFeedbackEdit, cancelFeedbackEdit, saveFeedbackEdit, toggleFeedbackReaction, toggleFeedbackReactionPicker,
      toggleFeedbackEmojiPicker,
      deleteFeedbackPost, deleteFeedbackReply,
      showStorageManager, storageSummary, storageSelection, storageJobsWithTrace,
      openStorageManager, toggleStorageSelection, toggleAllStorageSelection,
      deleteSelectedStorageFiles, fmtBytes,
      showAdminUsage, adminUsageLoading, adminUsageError, adminUsageDays,
      adminUsage, adminUsageCards, openAdminUsage, loadAdminUsage,
      showTritonCode, tritonCodeContent, tritonCodeFilename,
      tritonCodeEditing, tritonCodeEditContent,
      runCustomTriton, editTritonCode, cancelEditTritonCode,
      customRunStatus, allowCodeExecution,
      showAiCodeViewer, aiCodeViewerLoading, aiCodeViewerError,
      aiCodeViewerPath, aiCodeViewerFilename, aiCodeViewerContent,
      aiCodeViewerSize, aiCodeViewerTruncated,
      closeAiCodeViewer, copyAiCodeViewer, downloadAiCodeViewer,
      showGuide, showErrorModal, errorModalMsg, errorModalTitle,
      copyTritonCode, copyErrorModal,
      showAiPromptModal, aiAnalysisPrompt, aiPromptForce,
      openAiPromptModal, closeAiPromptModal, confirmAiPromptModal,
      showStepReanalysisModal, stepReanalysisLoading, stepReanalysisLabel,
      stepReanalysisFilterA, stepReanalysisFilterB,
      openStepReanalysisModal, closeStepReanalysisModal, confirmStepReanalysis,
      toasts, showConfirmModal, confirmModal, resolveConfirm,
      openActionMenu, toggleActionMenu, closeActionMenu,

      // Misc
      fmtDate, fmtDateTime, fmtCount, statusIcon, toggleGroup, createProject,
    };
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Bootstrap
// ══════════════════════════════════════════════════════════════════════════════

const app = createApp(App);
app.use(router);
app.mount("#app");
