const { createApp, ref, shallowRef, reactive, computed, watch, nextTick, onBeforeUnmount } = Vue;
const { createRouter, createWebHashHistory } = VueRouter;

// ══════════════════════════════════════════════════════════════════════════════
// Module-level reactive state (shared across all components via closure)
// ══════════════════════════════════════════════════════════════════════════════

let appInitialized = false;
const DEFAULT_RESULT_TAB = "chart";
const CLIENT_APP_VERSION = "0.5.31";
const APP_VERSION_CHECK_INTERVAL_MS = 60_000;
let appVersionCheckTimer = null;

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

const SETTINGS_KEY = "tpa-settings";
const AVATAR_COLORS = ["slate", "blue", "indigo", "violet", "rose", "amber", "emerald", "teal"];
const DEFAULT_USER_PREFS = {
  tableDensity: "default",
  sidebarDefaultCollapsed: false,
};
const sanitizeUserPrefs = prefs => ({
  tableDensity: prefs?.tableDensity === "compact" ? "compact" : "default",
  sidebarDefaultCollapsed: Boolean(prefs?.sidebarDefaultCollapsed),
});
const userPrefs = reactive({
  ...DEFAULT_USER_PREFS,
  ...sanitizeUserPrefs(readStoredJson(SETTINGS_KEY, DEFAULT_USER_PREFS)),
});
const applyUserPrefs = () => {
  document.documentElement.setAttribute("data-density", userPrefs.tableDensity);
};
const saveUserPrefs = () => {
  Object.assign(userPrefs, sanitizeUserPrefs(userPrefs));
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({
    tableDensity: userPrefs.tableDensity,
    sidebarDefaultCollapsed: userPrefs.sidebarDefaultCollapsed,
  }));
  applyUserPrefs();
};
applyUserPrefs();

// ── Theme ──────────────────────────────────────────────────────────────
const THEME_STORAGE_KEY = "tpa-theme";
const systemPrefersDark = () =>
  window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
const storedThemePreference = () => {
  const saved = localStorage.getItem(THEME_STORAGE_KEY);
  return ["light", "dark", "system"].includes(saved) ? saved : "system";
};
const resolveTheme = preference =>
  preference === "system" ? (systemPrefersDark() ? "dark" : "light") : preference;
const themePreference = ref(storedThemePreference());
const isDark = ref(resolveTheme(themePreference.value) === "dark");
const applyThemePreference = (preference = themePreference.value, persist = true) => {
  themePreference.value = ["light", "dark", "system"].includes(preference) ? preference : "system";
  const t = resolveTheme(themePreference.value);
  isDark.value = t === "dark";
  document.documentElement.setAttribute("data-theme", t);
  if (persist) localStorage.setItem(THEME_STORAGE_KEY, themePreference.value);
  if (resultTab.value === 'chart' && selectedJob.value?.status === 'done') {
    scheduleBuildChart();
  }
};
const setThemePreference = preference => applyThemePreference(preference);
const toggleTheme = () => {
  setThemePreference(isDark.value ? "light" : "dark");
};
window.matchMedia?.("(prefers-color-scheme: dark)")?.addEventListener?.("change", () => {
  if (themePreference.value === "system") applyThemePreference("system", false);
});

// ── Data ────────────────────────────────────────────────────────────────
const projects      = ref([]);
const historyGroups = ref([]);
const historyGroupsTotal = ref(0);
const historyGroupsLimit = ref(200);
const historyGroupsOffset = ref(0);
const historyGroupsLoading = ref(false);
const historyGroupJobsLimit = 50;
const historyJobs = ref([]);
const historyJobsTotal = ref(0);
const historyJobsLimit = ref(100);
const historyJobsOffset = ref(0);
const historyJobsLoading = ref(false);
const showTaskCenter = ref(false);
const taskCenterJobs = ref([]);
const taskCenterLoading = ref(false);
const taskCenterError = ref("");
const homeRecentJobs = ref([]);
const recentViewedJobs = ref([]);
const homeJobView = ref("completed");
const homeProjectView = ref("recent");
const homeDashboardLoading = ref(false);
const homeDashboardError = ref("");
const historySearch = ref("");
const filterProject = ref(localStorage.getItem("tpa-filter-project") || "");
const historyProjectView = ref(localStorage.getItem("tpa-history-project-view") || "all");
const storedSidebarTab = localStorage.getItem("tpa-sidebar-tab") || "jobs";
const sidebarTab    = ref(storedSidebarTab === "compare" ? "jobs" : storedSidebarTab);
const recentViewedProjects = ref([]);
const selectedJobId = ref(null);
const selectedJobHandle = ref(null);
const selectedJob   = ref(null);
const jobLoading    = ref(false);
const collapsedGroups = ref({});
let preSearchExpandedGroups = null;
const draggingProjectId = ref("");
const dragOverProjectId = ref("");
const projectOrderSaving = ref(false);

// ── Upload form ─────────────────────────────────────────────────────────
const fileA    = ref(null);
const fileAName = ref("");
const storedUploadMode = localStorage.getItem("tpa-upload-mode") || "single";
const quickUploadMode = ref(["single", "compare", "multi"].includes(storedUploadMode) ? storedUploadMode : "single");
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
const kernelHostOpDetail = ref({
  kernelName: "",
  rows: [],
  total: 0,
  loading: false,
  error: "",
});
let kernelHostOpDetailController = null;
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
const ktChartInst     = shallowRef(null);
const ktChart         = ref(null);
const callTimeScatterInst = shallowRef(null);
const callTimeScatter = ref(null);
const chartSource     = ref("");
const chartMetric     = ref("");
const chartTopN       = ref(10);
const chartCompareDirection = ref("all");
const chartMinAbsDelta = ref(0);
const chartMinDeltaPct = ref(0);
const chartMinBaselineMs = ref(0);
const chartExcludeCommunication = ref(true);
const chartFilterResultText = ref("");
const chartLoading    = ref(false);
const chartError      = ref("");
const chartSummaryCards = ref([]);
const chartSlowdowns    = ref([]);
const chartSpeedups     = ref([]);
const chartAddedRows    = ref([]);
const chartRemovedRows  = ref([]);
const chartBarRows      = ref([]);
const chartCanvasReady  = ref(false);
const chartScatterRows  = ref([]);
const chartDumbbellActive = ref(false);

const allowFileDownload = ref(true);
const allowCodeExecution = ref(false);
const claudeAnalysisEnabled = ref(false);
const appVersion = ref(CLIENT_APP_VERSION);
const authRequired = ref(false);
const authChecked = ref(false);
const authInitError = ref("");
const currentUser = ref(null);
const currentUserIsAdmin = ref(false);
const PROJECT_EXPANSION_STORAGE_PREFIX = "tpa-expanded-groups-v2";
const RECENT_VIEWED_PROJECTS_STORAGE_PREFIX = "tpa-recent-viewed-projects-v1";
const RECENT_VIEWED_JOBS_STORAGE_PREFIX = "tpa-recent-viewed-jobs-v1";
const RECENT_VIEWED_PROJECT_LIMIT = 6;
const RECENT_VIEWED_PROJECT_STORE_LIMIT = 12;
const RECENT_VIEWED_JOB_LIMIT = 6;
const RECENT_VIEWED_JOB_STORE_LIMIT = 12;
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

const projectExpansionUserKey = () => {
  const userKey =
    currentUser.value?.username ||
    currentUser.value?.email ||
    currentUser.value?.display_name ||
    "";
  const normalized = String(userKey).trim();
  if (normalized) return encodeURIComponent(normalized);
  return authRequired.value ? "anonymous" : "local";
};

const projectExpansionStorageKey = () =>
  `${PROJECT_EXPANSION_STORAGE_PREFIX}:${projectExpansionUserKey()}`;

const restoreProjectExpansionState = () => {
  collapsedGroups.value = readStoredJson(projectExpansionStorageKey(), {});
};

const recentViewedProjectsStorageKey = () =>
  `${RECENT_VIEWED_PROJECTS_STORAGE_PREFIX}:${projectExpansionUserKey()}`;
const recentViewedJobsStorageKey = () =>
  `${RECENT_VIEWED_JOBS_STORAGE_PREFIX}:${projectExpansionUserKey()}`;
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

const projectMenuKey = projectId => `project:${projectId}`;
const toggleProjectMenu = projectId => {
  if (!projectId || projectId === "__none__") return;
  toggleActionMenu(projectMenuKey(projectId));
};

const openReleaseNotes = () => {
  showReleaseNotes.value = true;
  closeActionMenu();
};

const resultStateKey = jobId => `tpa-result-state:${jobId}`;
const jobRouteHandle = jobOrId => {
  if (jobOrId && typeof jobOrId === "object") {
    const seq = jobOrId.seq;
    if (seq !== null && seq !== undefined && seq !== "") return String(seq);
    return String(jobOrId.id || "");
  }
  return String(jobOrId || "");
};
const jobRoutePath = (jobOrId, tab = "") => {
  const handle = jobRouteHandle(jobOrId);
  if (!handle) return "";
  const suffix = tab ? `/${encodeURIComponent(tab)}` : "";
  return `/job/${encodeURIComponent(handle)}${suffix}`;
};
const currentJobRouteHandle = () =>
  jobRouteHandle(selectedJob.value) || selectedJobHandle.value || selectedJobId.value || "";
const readResultMemory = jobId =>
  jobId ? readStoredJson(resultStateKey(jobId), { lastTab: DEFAULT_RESULT_TAB, tabs: {} }) : { lastTab: DEFAULT_RESULT_TAB, tabs: {} };
const hasResultMemory = jobId => Boolean(jobId && localStorage.getItem(resultStateKey(jobId)) !== null);
const writeResultMemory = (jobId, memory) => {
  if (!jobId) return;
  try {
    localStorage.setItem(resultStateKey(jobId), JSON.stringify(memory));
  } catch (e) {
    console.warn("Failed to persist result view state", e);
  }
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
      colWidths: sanitizeTableColumnWidths(colWidths.value),
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
  colWidths.value = sanitizeTableColumnWidths(state.colWidths);
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

// ── Triton ──────────────────────────────────────────────────────────────
const tritonStatus = ref({});

// ── Error display modal ─────────────────────────────────────────────────
const showErrorModal = ref(false);
const errorModalMsg = ref("");
const errorModalTitle = ref("错误信息");

// ── Layout / Modals ─────────────────────────────────────────────────────
const COMPACT_SIDEBAR_QUERY = "(max-width: 720px)";
const isCompactSidebarViewport = () =>
  window.matchMedia?.(COMPACT_SIDEBAR_QUERY).matches ?? false;
const sidebarWidth     = ref(readStoredNumber("tpa-sidebar-width", 240));
const sidebarCollapsed = ref(
  isCompactSidebarViewport()
    ? true
    : readStoredBool("tpa-sidebar-collapsed", userPrefs.sidebarDefaultCollapsed)
);
let sidebarWasCompact = isCompactSidebarViewport();
window.addEventListener("resize", () => {
  const compact = isCompactSidebarViewport();
  if (compact && !sidebarWasCompact) sidebarCollapsed.value = true;
  sidebarWasCompact = compact;
});

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
  if (confirmResolver) confirmResolver(false);
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
const newProjectLoading = ref(false);

const showRenameProject = ref(false);
const renameProjectId = ref("");
const renameProjectName = ref("");
const renameProjectLoading = ref(false);
const projectMutationLoading = ref("");

const showMoveProject = ref(false);
const moveProjectTarget = ref("");

const historyBulkMode = ref(false);
const historySelection = ref([]);
const showBulkMoveProject = ref(false);
const bulkMoveProjectTarget = ref("");
const showProjectBulkModal = ref(false);
const projectBulkId = ref("");
const projectBulkName = ref("");
const projectBulkJobs = ref([]);
const projectBulkJobsTotal = ref(0);
const projectBulkJobsOffset = ref(0);
const projectBulkJobsLimit = ref(100);
const projectBulkJobsLoading = ref(false);
const projectBulkActionLoading = ref("");
const projectBulkSearch = ref("");
const projectBulkSelectionDetails = ref({});

const showRenameJob = ref(false);
const renameJobName = ref("");

const showDeletedProjects = ref(false);
const deletedProjects = ref([]);
const deletedProjectActionLoading = ref("");
const showFeedbackBoard = ref(false);
const showFeedbackComposer = ref(false);
const feedbackItems = ref([]);
const feedbackTotal = ref(0);
const feedbackLimit = ref(30);
const feedbackOffset = ref(0);
const feedbackSort = ref("updated");
const feedbackLoading = ref(false);
const feedbackSubmitting = ref(false);
const feedbackForm = ref({ body: "", files: [], imageRefs: [], previews: [] });
const feedbackPostEditorMode = ref("write");
const feedbackReplies = ref({});
const feedbackEditing = ref({ id: "", body: "", files: [], imageRefs: [], previews: [], saving: false });
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
let feedbackImageRefSeq = 0;
const FEEDBACK_BODY_LIMIT = 2000;
const FEEDBACK_MAX_IMAGES = 4;
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
  range: {},
  daily: [],
  top_users_today: [],
});
const adminUsageTrendMetric = ref("requests");

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
  if (deletedProjectActionLoading.value) return;
  if (!await askConfirm("确定恢复该项目？", { confirmText: "恢复" })) return;
  deletedProjectActionLoading.value = `restore:${projectId}`;
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
  } finally {
    deletedProjectActionLoading.value = "";
  }
};

const permanentlyDeleteProject = async (projectId) => {
  if (deletedProjectActionLoading.value) return;
  if (!await askConfirm("确定永久删除？此操作不可恢复。", {
    title: "永久删除项目",
    confirmText: "永久删除",
    tone: "danger",
  })) return;
  deletedProjectActionLoading.value = `delete:${projectId}`;
  try {
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
  } catch (e) {
    showToast("永久删除出错: " + e.message, "error");
  } finally {
    deletedProjectActionLoading.value = "";
  }
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
const showSettings = ref(false);
const profileForm = reactive({
  username: "",
  email: "",
  source: "",
  display_name_ldap: "",
  display_name: "",
  display_name_effective: "",
  avatar_color: null,
  avatar_color_effective: "",
  avatar_colors: [...AVATAR_COLORS],
  loading: false,
  saving: false,
  error: "",
});
const showReleaseNotes = ref(false);
const releaseNotes = Object.freeze([
  {
    version: "0.5.31",
    date: "2026-07-20",
    title: "Host Op 展示与明细性能优化",
    items: [
      "Kernel 汇总和对比结果合并重复的主要 Host/Aten Op 列，仅保留信息更完整的主要 Host Op。",
      "Host/Aten 多对多关系仍在 Host Op 关联数中按需展开，诊断信息保持完整。",
      "大规模关系明细改为精确索引式读取并增加前端缓存，显著降低展开等待时间。",
    ],
  },
  {
    version: "0.5.30",
    date: "2026-07-20",
    title: "筛选占比与版本同步修复",
    items: [
      "耗时占比与累计占比改为按当前搜索和列筛选结果重新计算。",
      "筛选结果下载与页面保持同一占比口径，累计值最终准确到达 100%。",
      "页面定时检测服务版本并自动刷新，减少升级后旧资源和旧页签残留。",
    ],
  },
  {
    version: "0.5.29",
    date: "2026-07-20",
    title: "Host Op 关系展示精简",
    items: [
      "移除重复的 Kernel ↔ Host Op 一级页签，将异常关系收进所有 Kernel 按需展开。",
      "明细展示 Host/Aten Op、调用数、耗时占比和匹配方式，并保留完整 CSV 导出。",
      "补充同一 Host 对应多个 Aten 的计数，并修复明细渲染和横向滚动布局。",
    ],
  },
  {
    version: "0.5.28",
    date: "2026-07-19",
    title: "Kernel 与 Host Op 关联",
    items: [
      "Kernel 汇总和对比结果新增主要 Host Op、主要 Aten Op 与映射覆盖率。",
      "新增 Kernel ↔ Host Op 关系表，支持查看多对多关联、耗时占比和匹配方式。",
      "修复百分比字段合计和性能总览图表悬停联动异常。",
    ],
  },
  {
    version: "0.5.25",
    date: "2026-07-17",
    title: "对比散点图简化",
    items: [
      "对比散点图改为直接展示 B 相比 A 多或少的实际耗时。",
      "中间横线表示无变化，上方 B 更慢、下方 B 更快，降低读图成本。",
      "Tooltip 简化为 A 到 B 的耗时变化，以及变慢或变快的毫秒数。",
    ],
  },
  {
    version: "0.5.24",
    date: "2026-07-17",
    title: "对比散点图重设计",
    items: [
      "对比散点图改为 A/B 实际耗时双对数坐标，以等值线直观区分回退和改善。",
      "新增与消失项不再挤压散点图范围，继续通过独立排行展示。",
      "精简坐标刻度和热点标签，并保留 A/B 耗时、差值与变化率 Tooltip。",
    ],
  },
  {
    version: "0.5.23",
    date: "2026-07-17",
    title: "对比表格指标精简",
    items: [
      "对比表格移除变化率、绝对变化和回退贡献三列。",
      "默认展示 A/B 两侧 Kernel 数量及数量差值，并支持排序、筛选与导出。",
      "补齐 Triton 对比结果和历史任务中的 Kernel 数量差值。",
    ],
  },
  {
    version: "0.5.21",
    date: "2026-07-17",
    title: "散点图视觉升级",
    items: [
      "对比散点图增加回退/改善语义分区、0% 基线和 Top 影响项标签。",
      "点大小统一表达实际耗时影响，并增加颜色与大小图例。",
      "优化画布层次、坐标刻度、悬停反馈和 Tooltip 显示效果。",
    ],
  },
  {
    version: "0.5.20",
    date: "2026-07-17",
    title: "新增与消失项识别",
    items: [
      "对比结果识别新增、消失、精确匹配、推断匹配和低可信度匹配，并在表格中显示状态标签。",
      "性能总览新增新增/消失数量、耗时和匹配质量摘要。",
      "新增项与消失项提供独立 Top 列表，可直接点击下钻到对应 CSV。",
    ],
  },
  {
    version: "0.5.19",
    date: "2026-07-17",
    title: "对比表格重构",
    items: [
      "对比 CSV 新增变化率、绝对变化和回退贡献，支持排序、筛选与导出。",
      "新增精简对比、调用数和完整字段三种列预设，首次打开默认按绝对影响排序。",
      "A 基线、B 当前和变化列使用分组标签与独立色彩，回退贡献使用进度背景展示。",
    ],
  },
  {
    version: "0.5.18",
    date: "2026-07-17",
    title: "对比变化筛选器",
    items: [
      "性能总览支持按回退/改善方向、绝对变化、变化率和 A 基线耗时筛选。",
      "筛选在完整 CSV 扫描阶段执行，并统一作用于贡献排行、排序图和散点图。",
      "支持切换是否排除通信项，并显示筛选后数据量。",
    ],
  },
  {
    version: "0.5.17",
    date: "2026-07-17",
    title: "对比任务展示升级",
    items: [
      "对比摘要新增整体变化率、回退/改善结论及全量变化项数量。",
      "回退排行新增单项贡献和累计贡献，回退与改善分别按完整数据统计。",
      "新增 A 耗时 × 变化率散点图，通过颜色和点大小突出高影响回退与改善项。",
    ],
  },
  {
    version: "0.5.16",
    date: "2026-07-17",
    title: "修复最大热点摘要显示",
    items: [
      "修复性能总览中最大热点耗时错误显示为 0 的问题。",
      "最大热点耗时和占比统一读取全量汇总结果，与下方排行保持一致。",
    ],
  },
  {
    version: "0.5.15",
    date: "2026-07-17",
    title: "性能总览与首页体验升级",
    items: [
      "性能总览改为基于完整 CSV 汇总，新增耗时占比、累计占比和调用数 × 单次耗时散点图。",
      "耗时统计默认剔除通信类 Kernel/Op，排行、筛选和导出使用一致的全量口径。",
      "首页工作台支持在最近访问/最近完成任务、最近/收藏项目之间快速切换。",
    ],
  },
  {
    version: "0.5.14",
    date: "2026-07-16",
    title: "修复表格固定列滚动显示",
    items: [
      "修复结果表格纵向滚动时，固定首列内容覆盖左上角表头的问题。",
      "统一表头、底部合计行和固定首列的显示层级，横向与纵向滚动时内容保持清晰。",
    ],
  },
  {
    version: "0.5.13",
    date: "2026-07-16",
    title: "项目选择更快更清晰",
    items: [
      "上传 Trace 时可直接搜索项目，项目较多时无需在超长列表中逐项查找。",
      "项目列表限制显示高度，并将收藏、我的项目和共享项目分组展示。",
      "项目选择支持方向键、Enter 和 Esc 操作，长名称可悬浮查看完整内容。",
    ],
  },
  {
    version: "0.5.12",
    date: "2026-07-16",
    title: "完善键盘与无障碍体验",
    items: [
      "上传、侧边栏任务、对比选择和表格排序均可使用键盘完成，并补齐屏幕阅读器所需的名称与状态。",
      "弹窗增加 Tab 焦点约束和关闭后的焦点恢复，动态通知使用可感知的状态语义。",
      "系统开启减少动态效果时，会自动关闭非必要动画和过渡。",
    ],
  },
  {
    version: "0.5.11",
    date: "2026-07-16",
    title: "首页升级为工作台",
    items: [
      "首页优先展示进行中的任务、最近完成结果和最近项目，回访时可直接继续工作。",
      "新用户保留清晰的首个 Trace 引导，使用指南和灵感社区收敛为辅助入口。",
    ],
  },
  {
    version: "0.5.10",
    date: "2026-07-16",
    title: "提升大表格体验",
    items: [
      "常用字段增加中文名称、单位和解释，横向滚动时固定首列，定位 Kernel 更轻松。",
      "页面一次最多渲染 2,000 行，避免超大 CSV 卡住浏览器；全部筛选结果可直接导出。",
      "表格加载失败时可原地重试，分页与“全部”按钮会明确提示安全展示上限。",
    ],
  },
  {
    version: "0.5.9",
    date: "2026-07-16",
    title: "新增全局任务中心",
    items: [
      "页头集中显示排队中、运行中和最近失败的任务，不必停留在任务详情页等待。",
      "任务状态会在页面可见时自动刷新，完成或失败后给出提示；失败详情支持一键复制。",
    ],
  },
  {
    version: "0.5.8",
    date: "2026-07-16",
    title: "明确性能统计口径",
    items: [
      "性能总览明确标注设备计算耗时的统计范围，并提示该指标不等同于端到端耗时。",
      "对比视图中的耗时和 Delta 统一标记为计算口径，调用数说明不再暴露内部字段名。",
    ],
  },
  {
    version: "0.5.7",
    date: "2026-07-16",
    title: "修复 CSV 文本筛选",
    items: [
      "CSV 表格首次使用列筛选时，会按默认的“包含”操作显示文本输入框，kernel 名称等文本列可直接输入关键词。",
    ],
  },
  {
    version: "0.5.6",
    date: "2026-07-16",
    title: "优化页面刷新体验",
    items: [
      "快速刷新时不再闪现整屏品牌初始化页，只有加载较慢时才显示轻量提示。",
      "增加 Vue 首屏挂载保护，避免原始模板在脚本初始化前短暂露出。",
    ],
  },
  {
    version: "0.5.5",
    date: "2026-07-16",
    title: "修复 Web 交互稳定性",
    items: [
      "分享任务前会明确提示整个项目的公开范围，删除操作不再因页面切换误作用到其他任务。",
      "社区发帖草稿关闭前会确认，叠加弹窗中的确认框也能正常显示。",
      "项目管理与弹窗补齐键盘访问、Esc 关闭、回车提交和焦点恢复。",
    ],
  },
  {
    version: "0.5.4",
    date: "2026-07-16",
    title: "修复项目管理操作",
    items: [
      "项目重命名会正确回显名称，各项变更操作增加提交中状态，避免连续点击重复执行。",
      "删除和恢复项目会保留实验树、节点布局与任务数字短链，并阻止删除仍有活动任务的项目。",
      "找回项目弹窗支持清理超过 10 天的过期项目。",
    ],
  },
  {
    version: "0.5.2",
    date: "2026-07-16",
    title: "保留 Step 重分析效率 CSV",
    items: [
      "指定 step 重分析会保留逐 step Triton kernel 效率 CSV，以及 Triton 与非 Triton 效率汇总 CSV。",
    ],
  },
  {
    version: "0.5.1",
    date: "2026-07-10",
    title: "补充 Triton 启动维度",
    items: [
      "逐 step Triton CSV 新增 launch_dims 列，汇总 kernel trace 中的 dimx、dimy 和 dimz。",
    ],
  },
  {
    version: "0.5.0",
    date: "2026-07-05",
    title: "新增用户设置",
    items: [
      "更多菜单新增用户设置，可调整主题、表格密度、默认落地页和侧栏默认状态。",
      "账号资料支持自定义昵称和头像色板，顶栏改为头像展示，灵感社区会保存发帖时头像色快照。",
    ],
  },
  {
    version: "0.4.33",
    date: "2026-07-03",
    title: "修复性能总览空白兜底",
    items: [
      "性能总览图表未能及时创建时，会使用同一批排行数据渲染兜底条形图，避免出现空白面板。",
    ],
  },
  {
    version: "0.4.32",
    date: "2026-07-01",
    title: "优化灵感社区图片预览",
    items: [
      "点击帖子图片时弹出当前页预览窗口，再次点击预览图或背景关闭。",
    ],
  },
  {
    version: "0.4.31",
    date: "2026-07-01",
    title: "优化灵感社区图片交互",
    items: [
      "点击帖子图片时在当前页面切换放大/还原，不再打开独立图片页面。",
    ],
  },
  {
    version: "0.4.30",
    date: "2026-07-01",
    title: "优化灵感社区滚动与图片展示",
    items: [
      "图文混排图片超过评论区时缩放到评论区宽度，未超过时按原始尺寸的 50% 展示。",
      "反馈页左右空白区域也支持滚轮上下滚动。",
    ],
  },
  {
    version: "0.4.24",
    date: "2026-07-01",
    title: "修复 CSV 列宽随排序跳动",
    items: [
      "自动列宽改为按任务/页签/数据总量冻结快照，排序、筛选、翻页不再触发列宽重新计算和跳动。",
    ],
  },
  {
    version: "0.4.23",
    date: "2026-07-01",
    title: "移除 CNCL/NCCL 通信耗时统计",
    items: [
      "移除独立的 CNCL Ops 页签、CSV 输出和实验树 Comm 指标，修复通信耗时被重复计入的问题。",
      "删除 trace 文件前会检查是否有对比任务依赖该文件，避免误删后对比任务失效。",
    ],
  },
  {
    version: "0.4.22",
    date: "2026-07-01",
    title: "CSV 数值列居中",
    items: [
      "CSV 表格数值列改为居中对齐，并支持按列内容识别数字列。",
      "CSV 表头字段统一居中，筛选输入和效率徽标对齐同步优化。",
    ],
  },
  {
    version: "0.4.21",
    date: "2026-07-01",
    title: "CSV 默认列宽收紧",
    items: [
      "收紧 avg_dur_ms、avg_us_per_call 等短数值列的默认宽度，减少空白列占位。",
      "保留手动拖拽扩宽能力，已调整过的列仍可按需拉宽。",
    ],
  },
  {
    version: "0.4.20",
    date: "2026-07-01",
    title: "CSV 分类与侧边栏顺序修复",
    items: [
      "未知 Triton family 统一显示为 triton_other，并兼容合并旧 CSV 中的 triton / triton_other 行。",
      "类型对比 CSV 补充调用数 Delta，统计展示更完整。",
      "侧边栏打开任务时只高亮当前条目，不再改变项目内原有顺序。",
    ],
  },
  {
    version: "0.4.19",
    date: "2026-06-30",
    title: "CSV 列宽按内容自适应",
    items: [
      "CSV 列宽按内容自适应：数字等列贴合最宽值，列少或内容短时不再强行铺满整页。",
      "kernel name 等长文本列限制最大宽度并截断显示。",
    ],
  },
  {
    version: "0.4.18",
    date: "2026-06-30",
    title: "CSV 列宽拖拽手感优化",
    items: [
      "列宽拖拽改为基于抓取位置的增量调整，消除抓住分割线时的跳动，边缘跟手更自然。",
      "首次调整列宽时其余列不再被压缩跳动，保留自动布局的宽度。",
      "移除列宽上限，标题列等可以手动拖得更宽。",
      "CSV 单元格之间增加浅色细分割线。",
    ],
  },
  {
    version: "0.4.17",
    date: "2026-06-30",
    title: "CSV 列宽调整修复",
    items: [
      "修复 CSV 表格列宽拖拽方向和视觉宽度不一致的问题。",
      "拖拽列宽时不再误触发表头排序，并为列边界增加浅色分割线。",
    ],
  },
  {
    version: "0.4.16",
    date: "2026-06-30",
    title: "版本更新",
    items: [
      "版本号更新到 0.4.16。",
    ],
  },
  {
    version: "0.4.15",
    date: "2026-06-30",
    title: "任务数字短链",
    items: [
      "新任务 URL 改为 8 位数字短 ID，同时保留旧 UUID 链接兼容。",
      "分享链接、AI 分析通知和实验树跳转统一优先使用短链。",
    ],
  },
  {
    version: "0.4.14",
    date: "2026-06-30",
    title: "最近查看与 CSV 筛选优化",
    items: [
      "侧边栏新增最近查看项目入口，打开任务、实验树或展开项目后可快速回到上下文。",
      "优化 CSV 表格列筛选行，筛选输入框铺满单元格剩余宽度。",
    ],
  },
  {
    version: "0.4.13",
    date: "2026-06-29",
    title: "CSV 表格对齐与列宽优化",
    items: [
      "统一 CSV 表格数值列、效率列、长文本列的识别规则，修复带括号单位字段的对齐问题。",
      "为不同字段类型设置默认列宽，并让宽表按列宽横向滚动，提升多表扫描稳定性。",
    ],
  },
  {
    version: "0.4.12",
    date: "2026-06-29",
    title: "实验树布局修复与折线图视觉优化",
    items: [
      "修复旧节点尺寸和单节点缩放状态残留导致的画布自动整理重叠问题。",
      "美化折线图线条、点、图例、网格和目标线标签，并让坐标轴配色跟随当前主题。",
    ],
  },
  {
    version: "0.4.11",
    date: "2026-06-29",
    title: "实验树提示与指标对齐优化",
    items: [
      "收紧折线图悬浮提示宽度，减少指标名和数值之间的空白。",
      "画布节点多行指标的右侧 delta 统一右对齐，提升卡片扫描效率。",
    ],
  },
  {
    version: "0.4.10",
    date: "2026-06-29",
    title: "实验树折线图与画布交互优化",
    items: [
      "新增实验树折线图视图，支持按主指标查看代际趋势、设置指标目标线，并与画布详情交互保持一致。",
      "收敛画布和折线图工具栏位置，移除折线图缩放入口，仅保留拖拽平移。",
      "优化节点详情、悬浮提示和画布节点尺寸，取消节点单独缩放，避免数值截断和点击回退后的伪手动布局重叠。",
    ],
  },
  {
    version: "0.4.9",
    date: "2026-06-28",
    title: "实验树自动尺寸修正",
    items: [
      "自动整理时按 Compute time 主指标、右侧 delta chip 和次级指标共同估算节点宽度。",
      "节点主指标行的 delta chip 保持完整显示，同时减少自动尺寸带来的多余空白。",
    ],
  },
  {
    version: "0.4.8",
    date: "2026-06-28",
    title: "实验树主指标文案精简",
    items: [
      "节点主指标行保留 Compute time 作为指标名，右侧 delta chip 只显示箭头和百分比。",
      "移除同一行内重复出现的 compute 文案，让节点卡片更简洁。",
    ],
  },
  {
    version: "0.4.7",
    date: "2026-06-28",
    title: "实验树节点主指标调整",
    items: [
      "实验树节点卡片的主指标改为 Compute time，不再默认突出 E2E。",
      "节点卡片下方改为展示 E2E 与 Kernel，避免 Compute time 重复出现。",
    ],
  },
  {
    version: "0.4.6",
    date: "2026-06-28",
    title: "实验树悬浮详情层级修复",
    items: [
      "节点悬浮详情改为页面级浮层，不再被右侧详情面板遮挡。",
      "悬浮详情会随画布缩放和平移换算屏幕坐标，并自动限制在当前窗口内。",
    ],
  },
  {
    version: "0.4.5",
    date: "2026-06-28",
    title: "实验树验收修复",
    items: [
      "关系 delta 文案、颜色和优化判定统一优先使用 Compute time，避免 E2E 与 Compute 口径打架。",
      "节点不再显示基线/最优文字标签，改用卡片底色、边框和文字颜色区分性能最优、正优化与负优化。",
      "移除侧栏项目列表头的任务数字，降低项目入口噪音。",
    ],
  },
  {
    version: "0.4.4",
    date: "2026-06-28",
    title: "实验树节点缩放修复",
    items: [
      "修复节点缩放后可能覆盖其他实验节点、形成视觉叠层的问题。",
      "节点缩放时会为相邻节点保留安全间距，并将受影响的布局一并保存。",
      "加强节点卡片内部内容收敛，避免指标 chip 在窄宽度下溢出。",
    ],
  },
  {
    version: "0.4.3",
    date: "2026-06-28",
    title: "实验树前端视觉重构",
    items: [
      "实验树节点改为中性卡片，把正优化和负优化集中到边的 delta 芯片表达。",
      "节点卡片突出 E2E 主指标，Compute 和 Kernel 降为次要信息，降低数值截断和视觉噪音。",
      "未连接任务主操作改为设为优化结果，画布补充拖拽/缩放提示，并在刷新和自动整理后自动居中。",
    ],
  },
  {
    version: "0.4.2",
    date: "2026-06-28",
    title: "实验树可信度与可读性修复",
    items: [
      "实验树性能摘要新增 step 口径信息，step 数或单步耗时口径不一致时会在关系详情中提示。",
      "实验树加载指标文件时移入线程池执行，减少打开大项目时阻塞其他请求的风险。",
      "优化节点默认尺寸、指标行换行和关系告警展示，自动整理后更容易完整看清 E2E、Kernel、Compute 与 delta。",
      "标题栏中部标语仅在首页/空状态显示，进入具体任务后自动隐藏。",
    ],
  },
  {
    version: "0.4.1",
    date: "2026-06-28",
    title: "实验树工作流升级",
    items: [
      "新增实验树关系建模、节点拖拽缩放、关系标签编辑、侧边栏快速打开实验树等能力。",
      "优化实验树节点、连线、右侧详情面板和标题栏的视觉一致性，使用 compute time 作为优化判定主指标。",
      "补充实验树接口、持久化字段和 Web API 覆盖测试。",
    ],
  },
  {
    version: "0.4.0",
    date: "2026-06-27",
    title: "阶段版本升级",
    items: [
      "将 Web 应用版本升级到 0.4.0，标记当前侧边栏、首页、AI 分析和 Trace 解析能力的阶段性稳定版本。",
    ],
  },
  {
    version: "0.3.30",
    date: "2026-06-27",
    title: "标题栏与侧边栏视觉整理",
    items: [
      "优化标题栏标语字体和关键词配色，让口号更自然地融入背景。",
      "精简侧边栏项目树拖拽入口和保存状态提示，降低项目列表视觉噪音。",
    ],
  },
  {
    version: "0.3.29",
    date: "2026-06-27",
    title: "Perfetto 打开稳定性优化",
    items: [
      "增强 Perfetto 页面握手逻辑，降低页面加载慢或消息时序导致的偶发无法打开。",
      "打开失败时恢复按钮状态，并给出更明确的重试提示。",
    ],
  },
  {
    version: "0.3.28",
    date: "2026-06-27",
    title: "非 Triton 效率表列整理",
    items: [
      "非 Triton 效率 CSV 移除 operator_details 列，保留 operator 与输入 shape 相关字段。",
    ],
  },
  {
    version: "0.3.27",
    date: "2026-06-27",
    title: "静态样式细节整理",
    items: [
      "补齐页面交互主色变量，避免局部按钮或状态样式引用缺失。",
      "微调禁用按钮和表格表头的对比度，让静态视觉状态更清晰。",
    ],
  },
  {
    version: "0.3.26",
    date: "2026-06-27",
    title: "项目拖拽排序",
    items: [
      "侧边栏项目支持拖拽调整顺序，并按当前用户单独保存。",
      "未分组任务保持固定入口，不参与项目排序，避免误操作。",
    ],
  },
  {
    version: "0.3.25",
    date: "2026-06-27",
    title: "侧边栏项目展开记忆",
    items: [
      "切换“全部项目 / 收藏 / 我创建的 / 共享给我的”视图时，项目列表默认保持折叠，减少侧边栏初始噪音。",
      "项目展开/收起状态按当前用户单独记住，避免不同用户之间互相污染侧边栏浏览状态。",
    ],
  },
  {
    version: "0.3.24",
    date: "2026-06-27",
    title: "整体配色细节整理",
    items: [
      "微调标题栏、侧边栏和主内容区的浅色/深色主题背景，让整体视觉更统一。",
      "优化首页标语拆分和侧边栏快捷视图的状态配色，降低界面单调感。",
    ],
  },
  {
    version: "0.3.23",
    date: "2026-06-27",
    title: "项目批量管理弹窗",
    items: [
      "项目菜单中的“多选任务”改为弹窗式批量管理，支持项目内搜索、勾选、批量移动、删文件和删任务。",
      "保留侧边栏项目树的浏览状态，避免进入多选模式后打断当前查看上下文。",
    ],
  },
  {
    version: "0.3.22",
    date: "2026-06-27",
    title: "标题栏视觉优化",
    items: [
      "为标题栏增加更有层次的背景色，让中间标语自然融入整体氛围。",
      "优化浅色和深色主题下标题栏按钮、标语和背景的协调性。",
    ],
  },
  {
    version: "0.3.21",
    date: "2026-06-27",
    title: "项目菜单入口收敛",
    items: [
      "将新建对比入口移入项目右侧菜单，打开后固定当前项目，只需选择 A/B 任务。",
      "将多选入口移入项目菜单，并移除低价值的“只看该项目”操作。",
    ],
  },
  {
    version: "0.3.20",
    date: "2026-06-27",
    title: "侧边栏顶部精简",
    items: [
      "移除侧边栏顶部“工作区 / Trace workspace”标题块。",
      "移除侧边栏右上角历史总数徽标，让搜索和项目入口更靠前。",
    ],
  },
  {
    version: "0.3.19",
    date: "2026-06-27",
    title: "项目管理菜单",
    items: [
      "侧边栏项目管理收敛到项目行右侧项目管理菜单，减少独立操作条占位。",
      "项目菜单支持只看、收藏、重命名、转共享、转个人和删除等常用操作。",
    ],
  },
  {
    version: "0.3.18",
    date: "2026-06-27",
    title: "更多菜单交互修复",
    items: [
      "提高标题栏更多菜单层级，并改为不透明背景，避免主页面内容透出。",
      "修复“版本更新”菜单项点击不稳定的问题。",
    ],
  },
  {
    version: "0.3.17",
    date: "2026-06-27",
    title: "首页和导航视觉优化",
    items: [
      "首页移除“当前视图”模块，只保留常用入口和必要提示。",
      "优化标题栏和侧边栏配色，去掉标题栏底部硬边线，改为更柔和的层次过渡。",
    ],
  },
  {
    version: "0.3.16",
    date: "2026-06-27",
    title: "版本更新入口",
    items: [
      "标题栏更多菜单新增“版本更新”，可直接查看每个版本的主要改动。",
      "补充前端版本记录弹窗和 CHANGELOG，便于部署和使用时追踪功能变化。",
    ],
  },
  {
    version: "0.3.15",
    date: "2026-06-27",
    title: "首页精简",
    items: [
      "移除首页偏重的引导大卡片，让上传区和常用入口更聚焦。",
      "保留最近任务、当前视图和常用入口，减少空项目首页的视觉噪音。",
    ],
  },
  {
    version: "0.3.14",
    date: "2026-06-27",
    title: "视觉分区优化",
    items: [
      "优化标题栏、侧边栏和主内容区的配色层次，减少整体颜色过于单一的问题。",
      "同步调整浅色和深色主题下的边框、背景与按钮状态。",
    ],
  },
  {
    version: "0.3.12",
    date: "2026-06-27",
    title: "首页工作台",
    items: [
      "首页增加当前视图统计、最近任务和常用入口。",
      "空状态下提供上传、对比、使用指南和灵感社区的快捷路径。",
    ],
  },
  {
    version: "0.3.11",
    date: "2026-06-26",
    title: "上传模式拆分",
    items: [
      "上传 trace 拆分为单个、两个、多个三种模式。",
      "两个 trace 直接生成 A/B 对比；多个 trace 会逐个分析并生成独立任务。",
    ],
  },
  {
    version: "0.3.10",
    date: "2026-06-26",
    title: "侧边栏工作区",
    items: [
      "侧边栏新增全部项目、收藏、我创建的、共享给我的快捷视图。",
      "项目与任务按树形结构展示，提升历史任务检索效率。",
    ],
  },
  {
    version: "0.3.9",
    date: "2026-06-26",
    title: "对比流程优化",
    items: [
      "新建对比改为弹窗流程，先选择项目，再选择 A/B 条目。",
      "对比结果默认归属到所选项目下，减少历史列表混乱。",
    ],
  },
  {
    version: "0.3.7",
    date: "2026-06-26",
    title: "TensorFlow Trace 体验",
    items: [
      "TensorFlow trace 结果页隐藏 PyTorch 专属表格，避免无效空表干扰。",
      "修复 TensorFlow trace 上传后的类型识别与展示入口。",
    ],
  },
  {
    version: "0.3.5",
    date: "2026-06-26",
    title: "TensorFlow Trace 兼容",
    items: [
      "新增独立的 TensorFlow Chrome Trace 基础分析流程。",
      "尽量避免 TensorFlow 与 PyTorch 处理流程耦合，降低回归风险。",
    ],
  },
  {
    version: "0.3.3",
    date: "2026-06-26",
    title: "Kernel 效率分析",
    items: [
      "新增非 Triton kernel 效率 CSV，用于展示 CNNL、matmul 等 kernel 的效率指标。",
      "保留原有 Triton CSV，并改进 Compute / IO / OP Efficiency 字段解析。",
    ],
  },
  {
    version: "0.3.0",
    date: "2026-06-26",
    title: "0.3 功能线",
    items: [
      "首页和标题栏引入“让热点显形，让细节说话”的性能分析标语。",
      "持续完善 AI 分析报告、Triton code 优化展示、下钻和代码查看体验。",
    ],
  },
  {
    version: "0.2.0",
    date: "2026-06-08",
    title: "团队化使用",
    items: [
      "支持 LDAP 登录、用户隔离、共享项目、管理员能力和使用统计。",
      "新增 Claude Code AI 分析、灵感社区、邮件通知、日志、备份和监控能力。",
    ],
  },
  {
    version: "0.1.0",
    date: "2024-04-20",
    title: "初始版本",
    items: [
      "提供 PyTorch Profiler trace 上传、基础分析、CSV 导出和 Web 查看能力。",
      "支持项目分组、历史记录、Perfetto 跳转和 A/B 对比基础流程。",
    ],
  },
]);

const compareSelection  = ref([]);
const compareSelectionDetails = ref({});
const compareLabel      = ref("");
const compareProjectId  = ref("");
const showCompareModal  = ref(false);
const batchCompareMode  = ref(false);
const batchBaselineId   = ref("");
const batchCandidateIds = ref([]);
const batchSelectionDetails = ref({});
const batchCompareLabelPrefix = ref("");
const batchCompareLoading = ref(false);
const compareSubmitLoading = ref(false);
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
const historyProjectGroups = computed(() => historyGroups.value);
const loadedHistoryJobs = computed(() =>
  historyProjectGroups.value.flatMap(group => group.jobs || [])
);
const loadedHistoryJobIds = computed(() =>
  loadedHistoryJobs.value.map(job => job.id)
);
const historyAllJobCount = computed(() =>
  historyProjectGroups.value.reduce((sum, group) => sum + Number(group.job_count || 0), 0)
);
const sidebarProjectIdForJob = job => job ? (job.project_id || "__none__") : "";
const currentSidebarProjectId = computed(() => sidebarProjectIdForJob(selectedJob.value));
const isSidebarProjectActive = projectOrGroup =>
  Boolean(projectOrGroup?.id && (
    filterProject.value === projectOrGroup.id ||
    currentSidebarProjectId.value === projectOrGroup.id
  ));
const isRecentProjectActive = project =>
  Boolean(project?.id && project.id === (filterProject.value || currentSidebarProjectId.value));
const projectViewStats = computed(() => {
  const allProjects = projects.value.filter(project => project.id);
  return {
    all: allProjects.length,
    favorite: allProjects.filter(project => project.is_favorite).length,
    mine: allProjects.filter(project => project.is_owner).length,
    shared: allProjects.filter(project => project.is_public && !project.is_owner).length,
  };
});
const projectQuickViews = computed(() => [
  {
    id: "all",
    label: "全部项目",
    hint: "所有可访问内容",
    icon: "⌘",
    count: projectViewStats.value.all,
  },
  {
    id: "favorite",
    label: "收藏",
    hint: "常用项目入口",
    icon: "★",
    count: projectViewStats.value.favorite,
  },
  {
    id: "mine",
    label: "我创建的",
    hint: "个人负责项目",
    icon: "◇",
    count: projectViewStats.value.mine,
  },
  {
    id: "shared",
    label: "共享给我的",
    hint: "团队公开项目",
    icon: "↗",
    count: projectViewStats.value.shared,
  },
]);
const activeProjectView = computed(() =>
  projectQuickViews.value.find(view => view.id === historyProjectView.value) || projectQuickViews.value[0]
);
const activeHistoryProject = computed(() => {
  if (!filterProject.value) return null;
  if (filterProject.value === "__none__") return { id: "__none__", label: "未分组", job_count: historyJobsTotal.value };
  const group = historyProjectGroups.value.find(item => item.id === filterProject.value);
  const project = projects.value.find(item => item.id === filterProject.value);
  return group || (project ? { id: project.id, label: project.name, job_count: 0 } : null);
});
const historyListTitle = computed(() =>
  activeHistoryProject.value?.label || activeProjectView.value?.label || "全部任务"
);
const historyListSubtitle = computed(() => {
  const q = historySearch.value.trim();
  if (q && filterProject.value) return `当前项目内搜索 "${q}"`;
  if (q) return `${activeProjectView.value?.label || "全部项目"}内搜索 "${q}"`;
  if (filterProject.value) return "当前项目";
  if (historyProjectView.value === "all") return "最近更新";
  return activeProjectView.value?.hint || "快捷视图";
});
const selectedCompareJobs = computed(() =>
  compareSelection.value
    .map(id => compareSelectionDetails.value[id])
    .filter(Boolean)
);
const compareProjectLabel = computed(() => {
  if (!compareProjectId.value) return "";
  if (compareProjectId.value === "__none__") return "未分组";
  const project = projects.value.find(item => item.id === compareProjectId.value);
  const group = historyProjectGroups.value.find(item => item.id === compareProjectId.value);
  return project?.name || group?.label || "当前项目";
});
const selectedBatchBaseline = computed(() =>
  batchBaselineId.value ? batchSelectionDetails.value[batchBaselineId.value] : null
);
const selectedBatchCandidates = computed(() =>
  batchCandidateIds.value
    .map(id => batchSelectionDetails.value[id])
    .filter(Boolean)
);
const projectBulkSelectedJobs = computed(() =>
  historySelection.value
    .map(id => projectBulkSelectionDetails.value[id])
    .filter(Boolean)
);
const projectBulkLoadedOwnerIds = computed(() =>
  projectBulkJobs.value
    .filter(job => job.is_owner !== false)
    .map(job => job.id)
);
const projectBulkLoadedAllSelected = computed(() => {
  const ids = projectBulkLoadedOwnerIds.value;
  return ids.length > 0 && ids.every(id => historySelection.value.includes(id));
});

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
    "non_triton_kernel_efficiency_avg.csv": "非 Triton 效率",
    "aten_ops_avg.csv":         "Aten Ops",
    "aten_ops_cmp.csv":         "Aten 对比",
    "tf_ops_avg.csv":           "TF Ops",
    "tf_ops_cmp.csv":           "TF 对比",
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
  { file: "tf_ops_cmp.csv", label: "TF Ops Delta", mode: "compare", nameField: "op_name", defaultMetric: "delta_dur_ms" },
  { file: "kernel_types_avg.csv", label: "Kernel 类型", mode: "single", nameField: "type", defaultMetric: "avg_dur_ms" },
  { file: "all_kernels_avg.csv", label: "所有 Kernel", mode: "single", nameField: "kernel_name", defaultMetric: "avg_dur_ms" },
  { file: "triton_kernels_avg.csv", label: "Triton Kernel", mode: "single", nameField: "kernel_name", defaultMetric: "avg_dur_ms" },
  { file: "non_triton_kernel_efficiency_avg.csv", label: "非 Triton 效率", mode: "single", nameField: "kernel_name", defaultMetric: "avg_dur_ms" },
  { file: "aten_ops_avg.csv", label: "Aten Ops", mode: "single", nameField: "op_name", defaultMetric: "avg_dur_ms" },
  { file: "tf_ops_avg.csv", label: "TF Ops", mode: "single", nameField: "op_name", defaultMetric: "avg_dur_ms" },
];

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
  { key: "avg_compute_efficiency", label: "Compute 效率", unit: "%" },
  { key: "avg_op_efficiency", label: "OP 效率", unit: "%" },
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
const activeChartSourceConfig = computed(() =>
  chartSourceOptions.value.find(item => item.file === chartSource.value) || chartSourceOptions.value[0] || null
);
const activeChartMetricDef = computed(() => chartMetricDefFor(chartMetric.value));
const chartScatterMode = computed(() => selectedJob.value?.mode === "compare" ? "compare" : "single");
const chartScatterAvailable = computed(() => {
  const source = activeChartSourceConfig.value;
  const fields = selectedJob.value?.result_files?.[source?.file]?.fields || [];
  if (chartScatterMode.value === "compare") {
    return fields.includes("avg_dur_ms_A")
      && fields.includes("avg_dur_ms_B")
      && fields.includes("delta_dur_ms");
  }
  return fields.includes("avg_count")
    && (fields.includes("avg_us_per_call") || fields.includes("avg_dur_ms"));
});
const chartScatterTitle = computed(() => chartScatterMode.value === "compare"
  ? "A 耗时与耗时增减（点击下钻）"
  : "调用数 × 单次耗时（点击下钻）");
const chartScatterNote = computed(() => chartScatterMode.value === "compare"
  ? "横轴按 A 耗时量级从短到长排列；中间横线表示没变化，上方 B 更慢，下方 B 更快。新增和消失项在上方独立展示。"
  : "展示全量数据中的 Top 100 耗时热点；横纵轴均为对数坐标。");
const chartScatterEmptyText = computed(() => chartScatterMode.value === "compare"
  ? "暂无可绘制的 A/B 耗时数据"
  : "暂无可绘制的调用数与单次耗时数据");
const activeChartTitle = computed(() => {
  const source = activeChartSourceConfig.value;
  const metric = activeChartMetricDef.value;
  return source ? `${source.label} · ${metric.label}` : "";
});
const chartFallbackMaxValue = computed(() =>
  Math.max(1, ...chartBarRows.value.map(row => Number(row.displayValue) || Math.abs(Number(row.value) || 0)))
);

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

const hasKernelHostOpRelations = computed(() => Boolean(
  selectedJob.value?.result_files?.["kernel_host_ops.csv"]
  || selectedJob.value?.results?.["kernel_host_ops.csv"]
));

const isKernelTypeTab = computed(() =>
  ["kernel_types_avg.csv", "kernel_types_cmp.csv"].includes(resultTab.value)
);
const EFFICIENCY_TABLE_FILES = new Set(["non_triton_kernel_efficiency_avg.csv"]);
const EFFICIENCY_COLUMN_PRESET = [
  "kernel_name",
  "family",
  "operator",
  "avg_count",
  "avg_dur_ms",
  "avg_us_per_call",
  "avg_compute_efficiency",
  "avg_io_efficiency",
  "avg_op_efficiency",
];
const LONG_TABLE_FIELD_RE = /(^|_)(input|stride|concrete|shape|dims|types|details|tiling|config|code_file)(_|$)/i;
const NUMERIC_TABLE_FIELD_RE = /(^|_)(count|dur|duration|time|us|ms|gb|efficiency|delta|avg|total|pct|percent|call|calls)(_|$)/i;
const NUMERIC_TABLE_VALUE_RE = /^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:e[+-]?\d+)?%?$/i;
const EFFICIENCY_FIELD_RE = /(^|_)efficiency(_|$)/i;
const DURATION_SHARE_FIELDS = new Set(["duration_pct", "cumulative_pct"]);
const SOURCE_PERCENT_FIELD_RE = /^(host_op_coverage_pct(?:_[ab])?|share_pct)$/i;
const LEGACY_COMPARISON_TABLE_FIELDS = new Set(["delta_pct", "delta_abs_ms", "regression_contribution"]);
const KERNEL_NAME_FIELD_RE = /^kernel_name(?:_[ab])?$/i;
const TEXT_NAME_FIELD_RE = /(^|_)(name|operator|op_name|host_op|aten_op)(_|$)/i;
const SHORT_TEXT_FIELD_RE = /^(type|family|match_method)$/i;
const COMPACT_NUMERIC_COLUMN_MAX = {
  avg_count: 104,
  avg_count_a: 112,
  avg_count_b: 112,
  delta_count: 112,
  avg_dur_ms: 118,
  avg_dur_ms_a: 126,
  avg_dur_ms_b: 126,
  delta_dur_ms: 126,
  avg_us_per_call: 126,
  host_op_count: 112,
  host_op_coverage_pct: 132,
  share_pct: 112,
  avg_io_gb: 112,
  avg_io_gb_a: 118,
  avg_io_gb_b: 118,
  duration_pct: 112,
  cumulative_pct: 118,
};
const TABLE_COLUMN_MIN_WIDTH = 60;
// Single generous ceiling for both manual dragging and width preservation. It's
// only a sanity guard against corrupt/absurd values — columns (incl. the sticky
// title column) can be dragged as wide as this.
const TABLE_COLUMN_SAFETY_MAX = 2000;
// Content-based auto sizing: columns fit their widest value (header + cells)
// instead of stretching to fill the page. Long text columns (kernel name, etc.)
// are capped so they truncate; short/numeric columns shrink to content.
const TABLE_AUTO_MIN_WIDTH = 96;       // floor so the filter-row op + input stay usable
const TABLE_WIDE_COL_MAX = 440;        // truncation cap for long text columns
const TABLE_CELL_PADDING_X = 9;        // matches td/th horizontal padding
const TABLE_AUTO_MEASURE_SAMPLE = 400; // rows sampled when measuring width
const TABLE_HEADER_FONT = "600 10.5px 'JetBrains Mono', monospace";
const TABLE_MONO_VALUE_FONT = "11px 'JetBrains Mono', monospace";
const TABLE_TEXT_VALUE_FONT = "11px 'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
const TABLE_MAX_RENDER_ROWS = 2000;
const TABLE_FIELD_META = Object.freeze({
  kernel_name: { label: "Kernel 名称", hint: "设备 Kernel 的完整名称" },
  op_name: { label: "算子名称", hint: "框架算子名称" },
  operator: { label: "算子 / 实现", hint: "识别到的算子或实现类型" },
  host_op: { label: "Host Op", hint: "通过关联标识精确匹配到的 Host 侧算子" },
  aten_op: { label: "Aten Op", hint: "同一关联标识下识别到的 Aten 算子" },
  primary_host_op: { label: "主要 Host Op", hint: "关联 Kernel 耗时占比最高的 Host 侧算子" },
  type: { label: "Kernel 类型", hint: "按执行特征归类的 Kernel 类型" },
  family: { label: "Kernel 家族", hint: "Kernel 所属计算或通信家族" },
  match_method: { label: "匹配方式", hint: "当前行使用的算子关联或对比匹配依据" },
  avg_count: { label: "平均调用数", hint: "所选 step 中的平均调用次数" },
  delta_count: { label: "调用数差值 (B-A)", hint: "B 相对 A 的平均调用次数变化" },
  avg_dur_ms: { label: "平均耗时 (ms)", hint: "所选 step 中累计设备耗时的平均值" },
  delta_dur_ms: { label: "耗时差值 (B-A, ms)", hint: "B 相对 A 的设备耗时变化，正值表示变慢" },
  avg_us_per_call: { label: "平均单次耗时 (μs)", hint: "每次调用的平均设备执行时间" },
  host_op_count: { label: "Host Op 关联数", hint: "该 Kernel 关联到的不同 Host/Aten 关系数量" },
  host_op_coverage_pct: { label: "Host Op 覆盖率 (%)", hint: "成功关联 Host Op 的 Kernel 设备耗时占比" },
  share_pct: { label: "Kernel 内占比 (%)", hint: "该 Host Op 关系在此 Kernel 设备耗时中的占比" },
  avg_io_gb: { label: "平均 IO 量 (GB)", hint: "估算的平均读写数据量" },
  avg_compute_efficiency: { label: "计算效率 (%)", hint: "相对理论计算峰值的利用率" },
  avg_io_efficiency: { label: "IO 效率", hint: "相对理论 IO 峰值的利用率" },
  avg_op_efficiency: { label: "综合效率 (%)", hint: "计算与 IO 瓶颈下的综合效率" },
  duration_pct: { label: "耗时占比 (%)", hint: "该行耗时占当前筛选结果中计算总耗时的比例，通信项不参与分母" },
  cumulative_pct: { label: "累计占比 (%)", hint: "按耗时从高到低累计，占当前筛选结果中计算总耗时的比例" },
  compare_status: { label: "对比状态", hint: "新增、消失或 A/B 行的匹配可信度" },
  triton_code_file: { label: "Triton 代码", hint: "生成的 Triton 源码文件" },
});

const tableFieldMeta = field => {
  const raw = String(field || "");
  const key = raw.toLowerCase();
  const kernelCountTable = /^(all_kernels|triton_kernels|kernel_types)_/.test(resultTab.value || "");
  const contextualMeta = baseKey => {
    if (kernelCountTable && baseKey === "avg_count") {
      return { label: "Kernel 数量", hint: "所选 step 中每步平均出现的 Kernel 数量" };
    }
    if (kernelCountTable && baseKey === "delta_count") {
      return { label: "Kernel 数量差值 (B-A)", hint: "B 相对 A 的平均 Kernel 数量变化" };
    }
    return TABLE_FIELD_META[baseKey];
  };
  if (TABLE_FIELD_META[key]) return contextualMeta(key);
  const sideMatch = key.match(/^(.*)_([ab])$/);
  if (sideMatch && TABLE_FIELD_META[sideMatch[1]]) {
    const base = contextualMeta(sideMatch[1]);
    return { label: `${base.label} ${sideMatch[2].toUpperCase()}`, hint: `${sideMatch[2].toUpperCase()} 侧：${base.hint}` };
  }
  return { label: raw, hint: "" };
};
const tableFieldLabel = field => tableFieldMeta(field).label;
const tableFieldTitle = field => {
  const meta = tableFieldMeta(field);
  return [meta.hint, `原字段：${field}`].filter(Boolean).join("\n");
};

const isEfficiencyTable = computed(() =>
  EFFICIENCY_TABLE_FILES.has(resultTab.value)
  || ["avg_compute_efficiency", "avg_io_efficiency", "avg_op_efficiency"].every(field =>
    (currentTable.value.fields || []).includes(field)
  )
);

const tableTotalRows = computed(() =>
  currentTable.value.filtered_total ?? currentTable.value.total ?? currentTable.value.rows?.length ?? 0
);

const tableAllRowLimit = computed(() => Math.min(Number(tableTotalRows.value) || 0, TABLE_MAX_RENDER_ROWS));
const tableAllRowsCapped = computed(() => Number(tableTotalRows.value) > TABLE_MAX_RENDER_ROWS);
const tableAllRowsLabel = computed(() =>
  tableAllRowsCapped.value ? `前 ${fmtCount(TABLE_MAX_RENDER_ROWS)}` : "全部"
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
const isComparisonTable = computed(() => resultTab.value?.endsWith("_cmp.csv"));

const hiddenColumnCount = computed(() =>
  Math.max(0, currentTable.value.fields.length - displayedFields.value.length)
);

const storageJobsWithTrace = computed(() =>
  storageSummary.value.jobs.filter(job => job.has_original_trace)
);

const adminUsageRange = computed(() => adminUsage.value.range || {});

const adminUsageCards = computed(() => {
  const today = adminUsage.value.today || {};
  const range = adminUsageRange.value;
  const rangeDays = range.days || adminUsage.value.days || adminUsageDays.value;
  const rangeLabel = `近 ${rangeDays} 天`;
  return [
    { label: "范围活跃", value: range.active_users || 0, hint: `${rangeLabel} · 今日 ${fmtCount(today.dau || 0)}` },
    { label: "范围请求", value: range.requests || 0, hint: `今日 ${fmtCount(today.requests || 0)} · ${adminUsage.value.timezone || "-"}` },
    {
      label: "范围任务",
      value: (range.upload_jobs || 0) + (range.compare_jobs || 0),
      hint: `上传 ${fmtCount(range.upload_jobs || 0)} · 对比 ${fmtCount(range.compare_jobs || 0)}`,
    },
    {
      label: "AI / 留言",
      value: (range.ai_runs || 0) + (range.feedback_messages || 0),
      hint: `AI ${fmtCount(range.ai_runs || 0)} · 留言 ${fmtCount(range.feedback_messages || 0)}`,
    },
  ];
});

const adminUsageMetricOptions = [
  { key: "dau", label: "日活" },
  { key: "requests", label: "请求" },
  { key: "tasks", label: "任务" },
  { key: "ai_runs", label: "AI" },
];

const adminUsageTrendOption = computed(() =>
  adminUsageMetricOptions.find(item => item.key === adminUsageTrendMetric.value)
  || adminUsageMetricOptions[1]
);

const adminUsageTrendRows = computed(() => [...(adminUsage.value.daily || [])].reverse());

const adminUsageTrendValue = row => {
  if (adminUsageTrendMetric.value === "tasks") {
    return Number(row?.upload_jobs || 0) + Number(row?.compare_jobs || 0);
  }
  return Number(row?.[adminUsageTrendMetric.value] || 0);
};

const fmtUsageDay = day => String(day || "").slice(5).replace("-", "/");

const adminUsageTrendValues = computed(() =>
  adminUsageTrendRows.value.map(row => adminUsageTrendValue(row))
);

const adminUsageTrendMaxValue = computed(() =>
  Math.max(0, ...adminUsageTrendValues.value)
);

const adminUsageTrendPoints = computed(() => {
  const rows = adminUsageTrendRows.value;
  const values = adminUsageTrendValues.value;
  if (!rows.length) return [];
  const chartWidth = 100;
  const chartHeight = 42;
  const paddingY = 4;
  const scaleMax = Math.max(1, adminUsageTrendMaxValue.value);
  const usableHeight = chartHeight - paddingY * 2;
  return rows.map((row, index) => {
    const value = values[index] || 0;
    const x = rows.length === 1 ? chartWidth / 2 : (index / (rows.length - 1)) * chartWidth;
    const y = chartHeight - paddingY - (value / scaleMax) * usableHeight;
    return {
      x: Number(x.toFixed(2)),
      y: Number(y.toFixed(2)),
      value,
      label: row.day || "",
    };
  });
});

const adminUsageTrendPolyline = computed(() =>
  adminUsageTrendPoints.value.map(point => `${point.x},${point.y}`).join(" ")
);

const adminUsageTrendArea = computed(() => {
  const points = adminUsageTrendPoints.value;
  if (!points.length) return "";
  const line = points.map(point => `L ${point.x} ${point.y}`).join(" ");
  return `M ${points[0].x} 42 ${line} L ${points[points.length - 1].x} 42 Z`;
});

const adminUsageTrendFirstDay = computed(() =>
  fmtUsageDay(adminUsageTrendRows.value[0]?.day)
);

const adminUsageTrendLastDay = computed(() =>
  fmtUsageDay(adminUsageTrendRows.value[adminUsageTrendRows.value.length - 1]?.day)
);

const adminUsageUserLastPath = user => {
  if (!user?.last_path) return "";
  const status = user.last_status ? `${user.last_status}` : "";
  return [user.last_method, status, user.last_path].filter(Boolean).join(" ");
};

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
    if (DURATION_SHARE_FIELDS.has(f)) { result[f] = null; continue; }
    if (isSourcePercentField(f)) { result[f] = null; continue; }
    if (f.toLowerCase().includes('efficiency')) { result[f] = null; continue; }
    if (rows.some(r => String(r[f] ?? '').trim().endsWith('%'))) {
      result[f] = null; continue;
    }
    const nums = rows.map(r => parseFloat(r[f])).filter(n => !isNaN(n));
    result[f] = nums.length > 0 ? nums.reduce((a, b) => a + b, 0) : null;
  }
  return result;
});

// Keep the large table subtree stable while the Host Op drawer changes state.
// The drawer is rendered outside the table and does not need to invalidate
// thousands of otherwise unchanged cells on open, load, or close.
const tableRenderMemo = computed(() => [
  displayedFields.value,
  filteredRows.value,
  colSums.value,
  sortCol.value,
  sortAsc.value,
  colWidths.value,
  colFilters.value,
  colFilterOps.value,
  resultTab.value,
  tableStyle.value,
  tritonStatus.value,
  allowCodeExecution.value,
  hasKernelHostOpRelations.value,
]);

const fmtSum = v => {
  if (v === null || v === undefined || v === '') return '';
  const number = Number(v);
  if (!Number.isFinite(number)) return String(v);
  if (Number.isInteger(number)) return String(number);
  const s = number.toFixed(3);
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

const efficiencyPresetColumns = (fields = currentTable.value.fields || []) => {
  const available = new Set(fields);
  const keep = EFFICIENCY_COLUMN_PRESET.filter(field => available.has(field));
  return keep.length ? keep : fields.slice(0, Math.min(fields.length, 8));
};

const applyEfficiencyColumnPreset = () => {
  visibleColumns.value = efficiencyPresetColumns();
};

const comparisonPresetColumns = (mode = "compact", fields = currentTable.value.fields || []) => {
  if (mode === "full") return [...fields];
  const available = new Set(fields);
  const identity = fields.filter(field => /^(kernel_name|op_name|operator|type|family|compare_status|match_method)$/i.test(field));
  const metrics = mode === "calls"
    ? ["avg_count_A", "avg_count_B", "delta_count"]
    : ["avg_dur_ms_A", "avg_dur_ms_B", "delta_dur_ms", "avg_count_A", "avg_count_B", "delta_count"];
  return [...new Set([...identity, ...metrics.filter(field => available.has(field))])];
};

const applyComparisonColumnPreset = mode => {
  visibleColumns.value = comparisonPresetColumns(mode);
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

const normalizedTableField = field =>
  String(field || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");

const isNumericTableField = field => NUMERIC_TABLE_FIELD_RE.test(normalizedTableField(field));
const isNumericTableValue = value => {
  if (typeof value === "number") return Number.isFinite(value);
  const text = String(value ?? "").trim();
  if (!text) return false;
  return NUMERIC_TABLE_VALUE_RE.test(text.replace(/,/g, ""));
};
const isNumericTableColumn = (field, rows = currentTable.value.rows || []) => {
  if (isNumericTableField(field)) return true;
  let seen = 0;
  const sample = rows.length > TABLE_AUTO_MEASURE_SAMPLE ? rows.slice(0, TABLE_AUTO_MEASURE_SAMPLE) : rows;
  for (const row of sample) {
    const value = row?.[field];
    if (value === undefined || value === null || value === "") continue;
    if (!isNumericTableValue(value)) return false;
    seen += 1;
  }
  return seen > 0;
};
const isEfficiencyField = field => EFFICIENCY_FIELD_RE.test(normalizedTableField(field));
const isDurationShareField = field => DURATION_SHARE_FIELDS.has(normalizedTableField(field));
const isSourcePercentField = field => SOURCE_PERCENT_FIELD_RE.test(normalizedTableField(field));
const isLongTableField = field => LONG_TABLE_FIELD_RE.test(normalizedTableField(field));
// Long text columns whose content should truncate rather than dictate width.
const isWideTextField = field => {
  const key = normalizedTableField(field);
  return KERNEL_NAME_FIELD_RE.test(key) || TEXT_NAME_FIELD_RE.test(key) || isLongTableField(field);
};
const compactNumericColumnMax = field => COMPACT_NUMERIC_COLUMN_MAX[normalizedTableField(field)] || null;
const clampTableColumnWidth = (width, { max = TABLE_COLUMN_SAFETY_MAX } = {}) => {
  const value = Number(width);
  if (!Number.isFinite(value) || value <= 0) return null;
  return Math.min(max, Math.max(TABLE_COLUMN_MIN_WIDTH, value));
};

let _tableMeasureCtx = null;
const measureTextWidth = (text, font) => {
  if (typeof document === "undefined") return 0;
  if (!_tableMeasureCtx) _tableMeasureCtx = document.createElement("canvas").getContext("2d");
  if (!_tableMeasureCtx) return 0;
  _tableMeasureCtx.font = font;
  return _tableMeasureCtx.measureText(String(text ?? "")).width;
};
// Width a column needs to fit its widest value; long text columns are capped so
// they ellipsize, everything else snaps to content.
const measureColumnContentWidth = (field, rows) => {
  // triton_code_file renders action buttons, not the path text — fixed width.
  if (field === "triton_code_file") return 110;
  const numeric = isNumericTableColumn(field, rows);
  const valueFont = numeric ? TABLE_MONO_VALUE_FONT : TABLE_TEXT_VALUE_FONT;
  // Header reserves room for the sort icon (~16px).
  let max = measureTextWidth(tableFieldLabel(field), TABLE_HEADER_FONT) + 16;
  // Badge/chip cells carry extra horizontal padding around the value.
  const badgeExtra = isEfficiencyField(field) || normalizedTableField(field) === "family" ? 16 : 0;
  const sample = rows.length > TABLE_AUTO_MEASURE_SAMPLE ? rows.slice(0, TABLE_AUTO_MEASURE_SAMPLE) : rows;
  for (const row of sample) {
    const value = row?.[field];
    if (value === undefined || value === null || value === "") continue;
    const w = measureTextWidth(value, valueFont) + badgeExtra;
    if (w > max) max = w;
  }
  let width = Math.ceil(max) + TABLE_CELL_PADDING_X * 2 + 6;
  if (isWideTextField(field)) width = Math.min(width, TABLE_WIDE_COL_MAX);
  const compactMax = compactNumericColumnMax(field);
  if (compactMax) width = Math.min(width, compactMax);
  return Math.max(TABLE_AUTO_MIN_WIDTH, Math.min(TABLE_COLUMN_SAFETY_MAX, width));
};
// Auto widths are frozen per (job, tab, unfiltered row count) and only
// recomputed when that combination changes — not on sort/filter/pagination,
// which re-fetch a differently ordered/sliced page of the same dataset and
// would otherwise make columns visibly resize on every click.
const autoColWidthsSnapshot = ref({});
const autoColWidthsKey = ref("");
const autoColWidthsSourceKey = computed(() => {
  const table = currentTable.value;
  return `${selectedJobId.value || ""}::${resultTab.value || ""}::${table.total ?? ""}`;
});
const refreshAutoColWidthsIfNeeded = () => {
  const key = autoColWidthsSourceKey.value;
  if (key === autoColWidthsKey.value) return;
  autoColWidthsKey.value = key;
  const table = currentTable.value;
  const fields = table.fields || [];
  const rows = table.rows || [];
  const widths = {};
  for (const field of fields) widths[field] = measureColumnContentWidth(field, rows);
  autoColWidthsSnapshot.value = widths;
};
const autoColWidths = computed(() => autoColWidthsSnapshot.value);
const tableColumnWidth = field => {
  if (colWidths.value[field] !== undefined) return clampTableColumnWidth(colWidths.value[field]);
  const auto = autoColWidths.value[field];
  return auto ? clampTableColumnWidth(auto) : null;
};
const tableColumnStyle = field => {
  const width = tableColumnWidth(field);
  return width ? { width: `${width}px` } : {};
};
// Table width = sum of column widths, so a few short columns stay narrow instead
// of stretching across the page; wide content scrolls inside .table-scroll.
const tableStyle = computed(() => {
  const total = displayedFields.value.reduce(
    (sum, field) => sum + (tableColumnWidth(field) || TABLE_AUTO_MIN_WIDTH),
    0,
  );
  const width = `${Math.round(total)}px`;
  return { tableLayout: "fixed", width, minWidth: width };
});
const numericTableColumns = computed(() => {
  const table = currentTable.value;
  const rows = table.rows || [];
  return new Set((table.fields || []).filter(field => isNumericTableColumn(field, rows)));
});
const sanitizeTableColumnWidths = widths => {
  if (!widths || typeof widths !== "object") return {};
  const result = {};
  for (const [field, width] of Object.entries(widths)) {
    const clamped = clampTableColumnWidth(width);
    if (clamped) result[field] = clamped;
  }
  return result;
};
const tableHeaderClass = field => ({
  "num-col": numericTableColumns.value.has(field),
  "long-col": isLongTableField(field),
  "eff-col": isEfficiencyField(field),
  "share-col": isDurationShareField(field),
  "compare-a-col": isComparisonTable.value && /_A$/i.test(field),
  "compare-b-col": isComparisonTable.value && /_B$/i.test(field),
  "compare-change-col": isComparisonTable.value && /^delta_/i.test(field),
});
const tableCellClass = (field, value) => ({
  ...tableHeaderClass(field),
  [`eff-${efficiencyTone(value)}`]: isEfficiencyField(field),
});
const tableRowClass = row => {
  if (!isEfficiencyTable.value) return {};
  const op = parseFloat(row?.avg_op_efficiency);
  if (!Number.isFinite(op)) return {};
  return {
    "eff-row-bad": op < 30,
    "eff-row-warn": op >= 30 && op < 45,
  };
};
const familyChipClass = value => {
  const key = String(value || "").toLowerCase();
  if (key.includes("gemm") || key.includes("matmul")) return "fam-gemm";
  if (key.includes("triton")) return "fam-triton";
  if (key.includes("reduce")) return "fam-reduce";
  if (key.includes("comm") || key.includes("cncl") || key.includes("nccl")) return "fam-comm";
  return "";
};
const efficiencyTone = value => {
  const number = parseFloat(value);
  if (!Number.isFinite(number)) return "neutral";
  if (number >= 60) return "good";
  if (number >= 35) return "warn";
  return "bad";
};
const durationShareStyle = value => ({
  "--duration-share": `${Math.max(0, Math.min(100, Number.parseFloat(value) || 0))}%`,
});
const formatDurationShare = value => {
  const number = Number.parseFloat(value);
  if (!Number.isFinite(number)) return "—";
  const digits = number >= 10 ? 1 : 2;
  return `${Number(number.toFixed(digits))}%`;
};
const comparisonColumnGroup = field => {
  if (/_A$/i.test(field)) return "A 基线";
  if (/_B$/i.test(field)) return "B 当前";
  if (/^delta_/i.test(field)) return "变化";
  return "";
};
const comparisonStatusLabel = value => ({
  added: "新增",
  removed: "消失",
  exact: "精确匹配",
  inferred: "推断匹配",
  low_confidence: "低可信度",
})[value] || value || "—";
const comparisonStatusClass = value => `compare-status-${String(value || "unknown").replace(/_/g, "-")}`;
const shouldUseEfficiencyPreset = (filename, fields, state = null) => {
  if (!EFFICIENCY_TABLE_FILES.has(filename)) return false;
  if (!fields?.length) return false;
  const saved = state?.visibleColumns || [];
  return !saved.length;
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
const statusText = s => ({ pending: "排队中", running: "分析中", done: "已完成", error: "失败" }[s] || s);

const findProjectMeta = projectId => {
  if (!projectId || projectId === "__none__") return null;
  return projects.value.find(project => project.id === projectId)
    || historyProjectGroups.value.find(group => group.id === projectId)
    || null;
};

const normalizeRecentViewedJob = (job, viewedAt) => {
  if (!job?.id || job.status !== "done") return null;
  const project = findProjectMeta(job.project_id);
  return {
    id: job.id,
    seq: job.seq,
    label: job.label || "",
    file_a_name: job.file_a_name || "",
    project_id: job.project_id || "",
    project_name: job.project_name || project?.name || project?.label || "",
    created_at: job.created_at || "",
    viewed_at: viewedAt || job.viewed_at || new Date().toISOString(),
    status: "done",
    mode: job.mode || "single",
  };
};

const writeRecentViewedJobs = () => {
  try {
    localStorage.setItem(
      recentViewedJobsStorageKey(),
      JSON.stringify(recentViewedJobs.value.slice(0, RECENT_VIEWED_JOB_STORE_LIMIT)),
    );
  } catch (e) {}
};

const restoreRecentViewedJobs = () => {
  const stored = readStoredJson(recentViewedJobsStorageKey(), []);
  recentViewedJobs.value = (Array.isArray(stored) ? stored : [])
    .map(job => normalizeRecentViewedJob(job, job?.viewed_at))
    .filter(Boolean)
    .slice(0, RECENT_VIEWED_JOB_STORE_LIMIT);
  homeJobView.value = recentViewedJobs.value.length ? "visited" : "completed";
};

const rememberRecentViewedJob = job => {
  const record = normalizeRecentViewedJob(job);
  if (!record) return;
  recentViewedJobs.value = [
    record,
    ...recentViewedJobs.value.filter(item => item.id !== record.id),
  ].slice(0, RECENT_VIEWED_JOB_STORE_LIMIT);
  writeRecentViewedJobs();
};

const forgetRecentViewedJob = jobId => {
  const next = recentViewedJobs.value.filter(job => job.id !== jobId);
  if (next.length === recentViewedJobs.value.length) return;
  recentViewedJobs.value = next;
  writeRecentViewedJobs();
};

const normalizeRecentViewedProject = (projectOrId, viewedAt) => {
  const id = typeof projectOrId === "string" ? projectOrId : projectOrId?.id;
  if (!id || id === "__none__") return null;
  const source = typeof projectOrId === "string"
    ? findProjectMeta(projectOrId)
    : { ...(findProjectMeta(id) || {}), ...(projectOrId || {}) };
  const label = String(source?.name || source?.label || "").trim();
  return {
    id,
    label: label || `项目 ${String(id).slice(0, 8)}`,
    is_public: source?.is_public ? 1 : 0,
    is_owner: source?.is_owner !== false,
    is_favorite: source?.is_favorite ? 1 : 0,
    has_experiment_tree: source?.has_experiment_tree ? 1 : 0,
    viewed_at: viewedAt !== undefined ? viewedAt : (source?.viewed_at || new Date().toISOString()),
  };
};

const writeRecentViewedProjects = () => {
  try {
    localStorage.setItem(
      recentViewedProjectsStorageKey(),
      JSON.stringify(recentViewedProjects.value.slice(0, RECENT_VIEWED_PROJECT_STORE_LIMIT)),
    );
  } catch (e) {}
};

const restoreRecentViewedProjects = () => {
  const stored = readStoredJson(recentViewedProjectsStorageKey(), []);
  recentViewedProjects.value = (Array.isArray(stored) ? stored : [])
    .map(item => normalizeRecentViewedProject(item, item?.viewed_at))
    .filter(Boolean)
    .slice(0, RECENT_VIEWED_PROJECT_STORE_LIMIT);
};

const syncRecentViewedProjectsWithProjects = () => {
  const accessibleProjects = new Set(projects.value.map(project => project.id));
  const next = recentViewedProjects.value
    .filter(item => accessibleProjects.has(item.id))
    .map(item => normalizeRecentViewedProject(item, item.viewed_at))
    .filter(Boolean)
    .slice(0, RECENT_VIEWED_PROJECT_STORE_LIMIT);
  recentViewedProjects.value = next;
  writeRecentViewedProjects();
};

const rememberRecentProject = projectOrId => {
  const record = normalizeRecentViewedProject(projectOrId);
  if (!record) return;
  recentViewedProjects.value = [
    record,
    ...recentViewedProjects.value.filter(item => item.id !== record.id),
  ].slice(0, RECENT_VIEWED_PROJECT_STORE_LIMIT);
  writeRecentViewedProjects();
};

const recentViewedProjectItems = computed(() =>
  recentViewedProjects.value
    .map(item => normalizeRecentViewedProject(item, item.viewed_at))
    .filter(Boolean)
    .slice(0, RECENT_VIEWED_PROJECT_LIMIT)
);

const homeRecentProjects = computed(() => {
  if (recentViewedProjectItems.value.length) return recentViewedProjectItems.value.slice(0, 4);
  return [...projects.value]
    .sort((a, b) => Number(Boolean(b.is_favorite)) - Number(Boolean(a.is_favorite)))
    .slice(0, 4)
    .map(project => ({
      id: project.id,
      label: project.name || project.label || `项目 ${String(project.id).slice(0, 8)}`,
      is_public: project.is_public ? 1 : 0,
      is_owner: project.is_owner !== false,
      is_favorite: project.is_favorite ? 1 : 0,
      has_experiment_tree: project.has_experiment_tree ? 1 : 0,
      viewed_at: project.updated_at || project.created_at || "",
    }));
});
const homeFavoriteProjects = computed(() =>
  projects.value
    .filter(project => project.is_favorite)
    .slice(0, 4)
    .map(project => normalizeRecentViewedProject(project, ""))
    .filter(Boolean)
);
const homeDisplayedProjects = computed(() =>
  homeProjectView.value === "favorite" ? homeFavoriteProjects.value : homeRecentProjects.value
);
const homeVisitedJobs = computed(() => recentViewedJobs.value.slice(0, RECENT_VIEWED_JOB_LIMIT));
const homeDisplayedJobs = computed(() =>
  homeJobView.value === "visited" ? homeVisitedJobs.value : homeRecentJobs.value
);
const homeHasActivity = computed(() =>
  Boolean(
    taskCenterJobs.value.some(job => job.status === "pending" || job.status === "running")
    || homeRecentJobs.value.length
    || homeVisitedJobs.value.length
    || homeRecentProjects.value.length
    || homeFavoriteProjects.value.length
  )
);

const homeJobMeta = job => {
  const parts = [];
  if (job?.project_name) parts.push(job.project_name);
  const timestamp = homeJobView.value === "visited" ? job?.viewed_at : job?.created_at;
  if (timestamp) parts.push(`${homeJobView.value === "visited" ? "访问" : "完成"} ${fmtDate(timestamp)}`);
  return parts.join(" · ") || "未分组任务";
};

const recentProjectSubtitle = project => {
  const visibility = project.is_public
    ? (project.is_owner ? "我创建 · 已共享" : "共享给我")
    : "我创建";
  const viewedAt = fmtDate(project.viewed_at);
  return viewedAt ? `${visibility} · ${viewedAt}` : visibility;
};

const clearRecentViewedProjects = () => {
  recentViewedProjects.value = [];
  writeRecentViewedProjects();
};

const openRecentProject = project => {
  if (!project?.id) return;
  rememberRecentProject(project);
  historyProjectView.value = "all";
  filterProject.value = project.id;
  collapsedGroups.value = { ...collapsedGroups.value, [project.id]: true };
  loadHistoryGroupJobs(project.id, true);
};

const openRecentProjectTree = project => {
  if (!project?.id) return;
  rememberRecentProject(project);
  router.push({ path: `/project/${project.id}/tree` });
};

const toggleGroup = async label => {
  const opening = !collapsedGroups.value[label];
  collapsedGroups.value[label] = opening;
  if (opening) {
    const group = historyGroups.value.find(item => item.id === label);
    rememberRecentProject(group);
    if (group && !group.jobs_loaded) await loadHistoryGroupJobs(label, true);
  }
};

const orderedProjectIds = groups =>
  groups
    .filter(group => group.id && group.id !== "__none__")
    .map(group => group.id);

const persistProjectOrder = async projectIds => {
  if (!projectIds.length) return true;
  projectOrderSaving.value = true;
  try {
    const r = await fetch("/api/projects/order", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ project_ids: projectIds }),
    });
    const data = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, data, "保存项目排序失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
    return true;
  } catch (e) {
    showToast(normalizeApiError(e, "保存项目排序失败"), "error");
    return false;
  } finally {
    projectOrderSaving.value = false;
  }
};

const startProjectDrag = (group, event) => {
  if (!group?.id || group.id === "__none__") return;
  draggingProjectId.value = group.id;
  dragOverProjectId.value = "";
  if (event?.dataTransfer) {
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", group.id);
  }
};

const dragProjectOver = (group, event) => {
  if (!draggingProjectId.value || !group?.id || group.id === "__none__" || group.id === draggingProjectId.value) return;
  dragOverProjectId.value = group.id;
  if (event?.dataTransfer) event.dataTransfer.dropEffect = "move";
};

const endProjectDrag = () => {
  draggingProjectId.value = "";
  dragOverProjectId.value = "";
};

const dropProject = async group => {
  const sourceId = draggingProjectId.value;
  const targetId = group?.id;
  endProjectDrag();
  if (!sourceId || !targetId || targetId === "__none__" || sourceId === targetId) return;

  const fromIndex = historyGroups.value.findIndex(item => item.id === sourceId);
  const toIndex = historyGroups.value.findIndex(item => item.id === targetId);
  if (fromIndex < 0 || toIndex < 0) return;

  const before = [...historyGroups.value];
  const next = [...historyGroups.value];
  const [moved] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, moved);
  historyGroups.value = next;
  const saved = await persistProjectOrder(orderedProjectIds(next));
  if (!saved) {
    historyGroups.value = before;
    await loadHistoryGroups();
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

const defaultAvatarColor = identity => {
  const key = String(identity || "local");
  let total = 0;
  for (const ch of Array.from(key)) {
    total += ch.codePointAt(0) || 0;
  }
  return AVATAR_COLORS[total % AVATAR_COLORS.length];
};

const avatarColorFor = (identity, color) =>
  AVATAR_COLORS.includes(color) ? color : defaultAvatarColor(identity);

const currentUserAvatarColor = computed(() =>
  avatarColorFor(
    currentUser.value?.display_name_ldap || currentUser.value?.display_name || currentUser.value?.username || "local",
    currentUser.value?.avatar_color,
  )
);

const profileAvatarColor = computed(() =>
  avatarColorFor(
    profileForm.display_name_ldap || profileForm.display_name_effective || profileForm.username || "local",
    profileForm.avatar_color || profileForm.avatar_color_effective,
  )
);

const profileSourceLabel = computed(() => profileForm.source === "ldap" ? "LDAP" : "本地模式");

const applyProfileForm = profile => {
  profileForm.username = profile.username || "";
  profileForm.email = profile.email || "";
  profileForm.source = profile.source || "";
  profileForm.display_name_ldap = profile.display_name_ldap || "";
  profileForm.display_name = profile.display_name_override || "";
  profileForm.display_name_effective = profile.display_name_effective || profile.display_name_ldap || profile.username || "";
  profileForm.avatar_color = profile.avatar_color || null;
  profileForm.avatar_color_effective = profile.avatar_color_effective || avatarColorFor(profile.display_name_ldap || profile.username || "local", profile.avatar_color);
  profileForm.avatar_colors = (profile.avatar_colors || AVATAR_COLORS).filter(color => AVATAR_COLORS.includes(color));
  if (!profileForm.avatar_colors.length) profileForm.avatar_colors = [...AVATAR_COLORS];
};

const loadUserProfile = async () => {
  profileForm.loading = true;
  profileForm.error = "";
  try {
    const profile = await fetchJson("/api/user/profile", { credentials: "include" }, "加载用户设置失败");
    applyProfileForm(profile);
    if (profile.username) {
      currentUser.value = {
        ...(currentUser.value || {}),
        username: profile.username,
        display_name: profile.display_name_effective,
        display_name_ldap: profile.display_name_ldap,
        email: profile.email || "",
        avatar_color: profile.avatar_color_effective,
      };
    }
  } catch (e) {
    profileForm.error = normalizeApiError(e, "加载用户设置失败");
  } finally {
    profileForm.loading = false;
  }
};

const openSettings = async () => {
  showSettings.value = true;
  closeActionMenu();
  await loadUserProfile();
};

const closeSettings = () => {
  showSettings.value = false;
  profileForm.error = "";
};

const saveProfile = async () => {
  profileForm.saving = true;
  profileForm.error = "";
  try {
    const profile = await fetchJson("/api/user/profile", {
      method: "PUT",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        display_name: profileForm.display_name,
        avatar_color: profileForm.avatar_color || null,
      }),
    }, "保存用户设置失败");
    applyProfileForm(profile);
    currentUser.value = {
      ...(currentUser.value || {}),
      username: profile.username,
      display_name: profile.display_name_effective,
      display_name_ldap: profile.display_name_ldap,
      email: profile.email || "",
      avatar_color: profile.avatar_color_effective,
    };
    closeSettings();
    showToast("用户设置已保存", "success");
  } catch (e) {
    profileForm.error = normalizeApiError(e, "保存用户设置失败");
  } finally {
    profileForm.saving = false;
  }
};

const saveUserPrefsFromSettings = () => {
  saveUserPrefs();
};

const applySidebarDefaultFromSettings = () => {
  sidebarCollapsed.value = Boolean(userPrefs.sidebarDefaultCollapsed);
  localStorage.setItem("tpa-sidebar-collapsed", String(sidebarCollapsed.value));
  saveUserPrefs();
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

const reloadForAppVersionMismatch = serverVersion => {
  const normalized = String(serverVersion || "").trim();
  if (!normalized || normalized === CLIENT_APP_VERSION) return false;
  const reloadKey = `tpa-version-reload:${CLIENT_APP_VERSION}:${normalized}`;
  try {
    if (sessionStorage.getItem(reloadKey) === "1") return false;
    sessionStorage.setItem(reloadKey, "1");
  } catch {
    // Continue with the reload when session storage is unavailable.
  }
  const url = new URL(window.location.href);
  url.searchParams.set("_app_version", normalized);
  window.location.replace(url.toString());
  return true;
};

const checkAppVersion = async () => {
  try {
    const cfg = await fetchJson(
      "/api/config",
      { credentials: "include", cache: "no-store" },
      "检查版本失败",
    );
    reloadForAppVersionMismatch(cfg.version);
  } catch {
    // A transient version-check failure must not interrupt the active page.
  }
};

const startAppVersionChecks = () => {
  if (appVersionCheckTimer) return;
  appVersionCheckTimer = setInterval(checkAppVersion, APP_VERSION_CHECK_INTERVAL_MS);
};

const loadConfig = async () => {
  const cfg = await fetchJson(
    "/api/config",
    { credentials: "include", cache: "no-store" },
    "加载配置失败",
  );
  if (reloadForAppVersionMismatch(cfg.version)) return false;
  appVersion.value = cfg.version || CLIENT_APP_VERSION;
  authRequired.value = Boolean(cfg.auth_required);
  allowFileDownload.value = cfg.allow_file_download ?? true;
  allowCodeExecution.value = cfg.allow_code_execution ?? false;
  claudeAnalysisEnabled.value = cfg.claude_analysis_enabled ?? false;
  startAppVersionChecks();
  return true;
};

const loadMe = async () => {
  const r = await fetch("/api/me", { credentials: "include" });
  const data = await readJsonResponse(r, {});
  if (r.status === 401) {
    currentUser.value = null;
    currentUserIsAdmin.value = false;
    authChecked.value = true;
    restoreProjectExpansionState();
    restoreRecentViewedProjects();
    restoreRecentViewedJobs();
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
  restoreProjectExpansionState();
  restoreRecentViewedProjects();
  restoreRecentViewedJobs();
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
    restoreProjectExpansionState();
    restoreRecentViewedProjects();
    restoreRecentViewedJobs();
    window.setTimeout(() => {
      loginForm.value.password = "";
      loginForm.value.captcha = "";
    }, 300);
    loginCaptchaRequired.value = false;
    loginCaptchaImage.value = "";
    appInitialized = true;
    await loadProjects();
    await refreshSidebarData();
    await Promise.all([
      loadTaskCenter({ silent: true }),
      loadHomeDashboard({ silent: true }),
    ]);
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
  historyJobs.value = [];
  taskCenterJobs.value = [];
  homeRecentJobs.value = [];
  homeDashboardError.value = "";
  homeDashboardController?.abort();
  homeDashboardController = null;
  taskCenterInitialized = false;
  closeTaskCenter();
  recentViewedProjects.value = [];
  recentViewedJobs.value = [];
  homeJobView.value = "completed";
  homeProjectView.value = "recent";
  collapsedGroups.value = {};
  compareJobs.value = [];
  selectedJobId.value = null;
  selectedJobHandle.value = null;
  selectedJob.value = null;
  clearAiDiagnostics();
  router.push({ path: "/" });
};

const loadProjects = async () => {
  try {
    projects.value = await fetchJson("/api/projects", { credentials: "include" }, "加载项目失败");
    if (!["all", "favorite", "mine", "shared"].includes(historyProjectView.value)) {
      historyProjectView.value = "all";
    }
    if (
      filterProject.value &&
      filterProject.value !== "__none__" &&
      !projects.value.some(project => project.id === filterProject.value)
    ) {
      filterProject.value = "";
    }
    syncRecentViewedProjectsWithProjects();
  } catch (e) {
    const message = normalizeApiError(e, "加载项目失败");
    console.error("loadProjects error:", e);
    if (e?.authExpired) showToast(message, "error");
  }
};

const clearProjectFilterIfJobIsHidden = job => {
  if (!job) return;
  const jobProjectId = job.project_id || "__none__";
  if (filterProject.value && filterProject.value !== jobProjectId) {
    filterProject.value = "";
  }
  if (historyProjectView.value !== "all") {
    historyProjectView.value = "all";
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
const emptyFeedbackForm = () => ({ body: "", files: [], imageRefs: [], previews: [] });
const emptyFeedbackReplyForm = (open = false) => ({ open, body: "", files: [], imageRefs: [], previews: [], submitting: false, mode: "write" });
const emptyFeedbackEditing = () => ({ id: "", body: "", files: [], imageRefs: [], previews: [], saving: false });
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

const feedbackMessageById = id => {
  const messageId = String(id || "");
  const post = selectedFeedbackPost.value;
  if (!messageId || !post) return null;
  if (post.id === messageId) return post;
  return (post.replies || []).find(reply => reply.id === messageId) || null;
};

const feedbackExistingImageCount = target => {
  const targetKey = feedbackMentionTargetKey(target);
  if (!targetKey.startsWith("edit:")) return 0;
  return (feedbackMessageById(targetKey.slice(5))?.attachments || []).length;
};

const feedbackReferencedAttachmentUrls = message => {
  const body = String(message?.body || "");
  const urls = new Set();
  for (const match of body.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)) {
    urls.add((match[1] || "").trim());
  }
  return urls;
};

const feedbackVisibleAttachments = message => {
  const refs = feedbackReferencedAttachmentUrls(message);
  return (message?.attachments || []).filter(item => !refs.has(item.url));
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
  editor.codemirror.on("paste", (_cm, event) => {
    handleFeedbackPaste(event, targetKey);
  });
  editor.codemirror.on("drop", (_cm, event) => {
    handleFeedbackDrop(event, targetKey);
  });
  editor.codemirror.on("dragover", (_cm, event) => {
    handleFeedbackDragOver(event);
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

const feedbackFilePreviews = (files, refs = []) =>
  files.map((file, index) => ({
    name: file.name,
    ref: refs[index] || "",
    url: URL.createObjectURL(file),
  }));

const nextFeedbackImageRef = () => {
  feedbackImageRefSeq += 1;
  return `img-${Date.now().toString(36)}-${feedbackImageRefSeq.toString(36)}`;
};

const feedbackImageFileExt = file => {
  const type = String(file?.type || "").toLowerCase();
  if (type.includes("jpeg") || type.includes("jpg")) return ".jpg";
  if (type.includes("gif")) return ".gif";
  if (type.includes("webp")) return ".webp";
  return ".png";
};

const normalizeFeedbackImageFile = (file, index = 0) => {
  if (!file || !file.type?.startsWith("image/")) return null;
  if (file.name) return file;
  const filename = `pasted-image-${Date.now()}-${index + 1}${feedbackImageFileExt(file)}`;
  return new File([file], filename, { type: file.type || "image/png", lastModified: file.lastModified || Date.now() });
};

const patchFeedbackDraftFiles = (target, patch) => {
  const targetKey = feedbackMentionTargetKey(target);
  if (targetKey === "post") {
    feedbackForm.value = { ...feedbackForm.value, ...patch };
    return;
  }
  if (targetKey.startsWith("edit:")) {
    feedbackEditing.value = { ...feedbackEditing.value, ...patch };
    return;
  }
  const form = ensureFeedbackReplyForm(targetKey);
  feedbackReplies.value = {
    ...feedbackReplies.value,
    [targetKey]: { ...form, ...patch },
  };
};

const feedbackClipboardImageFiles = event => {
  const items = Array.from(event?.clipboardData?.items || []);
  const files = items
    .filter(item => item.kind === "file" && item.type?.startsWith("image/"))
    .map(item => item.getAsFile())
    .filter(Boolean);
  if (files.length) return files;
  return Array.from(event?.clipboardData?.files || []).filter(file => file.type?.startsWith("image/"));
};

const feedbackDroppedImageFiles = event =>
  Array.from(event?.dataTransfer?.files || []).filter(file => file.type?.startsWith("image/"));

const feedbackImageMarkdown = (file, ref) => {
  const alt = String(file?.name || "image").replace(/[\[\]\n\r]/g, " ").trim() || "image";
  return `![${alt}](feedback-image:${ref})`;
};

const insertFeedbackImageMarkdown = (target, files, refs) => {
  if (!files.length || !refs.length) return;
  const { text, start, end } = feedbackDraftForTarget(target);
  const before = text.slice(0, start);
  const after = text.slice(end);
  const leading = before && !before.endsWith("\n") ? "\n" : "";
  const trailing = after && !after.startsWith("\n") ? "\n" : "";
  const insertText = `${leading}${files.map((file, index) => feedbackImageMarkdown(file, refs[index])).join("\n")}${trailing}`;
  replaceFeedbackSelection(target, insertText);
};

const appendFeedbackImageFiles = (target, rawFiles, { source = "paste" } = {}) => {
  const targetKey = feedbackMentionTargetKey(target);
  const form = feedbackTargetForm(targetKey);
  const rawIncoming = Array.from(rawFiles || []);
  const incoming = rawIncoming
    .map((file, index) => normalizeFeedbackImageFile(file, index))
    .filter(Boolean);
  if (incoming.length !== rawIncoming.length) {
    showToast("仅支持图片文件", "error");
  }
  if (!incoming.length) return 0;
  const existingFiles = form?.files || [];
  const remaining = FEEDBACK_MAX_IMAGES - feedbackExistingImageCount(targetKey) - existingFiles.length;
  if (remaining <= 0) {
    showToast(`最多保留 ${FEEDBACK_MAX_IMAGES} 张图片`, "error");
    return 0;
  }
  const accepted = incoming.slice(0, remaining);
  if (incoming.length > accepted.length) showToast(`最多保留 ${FEEDBACK_MAX_IMAGES} 张图片`, "error");
  const acceptedRefs = accepted.map(() => nextFeedbackImageRef());
  patchFeedbackDraftFiles(targetKey, {
    files: [...existingFiles, ...accepted],
    imageRefs: [...(form?.imageRefs || []), ...acceptedRefs],
    previews: [...(form?.previews || []), ...feedbackFilePreviews(accepted, acceptedRefs)],
  });
  insertFeedbackImageMarkdown(targetKey, accepted, acceptedRefs);
  const sourceText = source === "drop" ? "拖入" : source === "pick" ? "选择" : "粘贴";
  showToast(`已添加 ${accepted.length} 张${sourceText}图片`, "success");
  return accepted.length;
};

const setFeedbackFiles = (event, target = "post") => {
  appendFeedbackImageFiles(target || "post", event?.target?.files || [], { source: "pick" });
  if (event?.target) event.target.value = "";
};

const handleFeedbackPaste = (event, target = "post") => {
  const files = feedbackClipboardImageFiles(event);
  if (!files.length) return;
  event.preventDefault();
  appendFeedbackImageFiles(target, files, { source: "paste" });
};

const handleFeedbackDrop = (event, target = "post") => {
  const files = feedbackDroppedImageFiles(event);
  if (!files.length) return;
  event.preventDefault();
  appendFeedbackImageFiles(target, files, { source: "drop" });
};

const handleFeedbackDragOver = event => {
  const hasImage = Array.from(event?.dataTransfer?.items || [])
    .some(item => item.kind === "file" && item.type?.startsWith("image/"));
  if (hasImage) {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }
};

const clearFeedbackForm = (parentId = null) => {
  closeFeedbackMention();
  if (parentId) {
    const form = feedbackReplies.value[parentId];
    revokeFeedbackPreviews(form?.previews || []);
    feedbackReplies.value = {
      ...feedbackReplies.value,
      [parentId]: emptyFeedbackReplyForm(false),
    };
    nextTick(() => syncFeedbackMarkdownEditor(parentId, ""));
    return;
  }
  revokeFeedbackPreviews(feedbackForm.value.previews);
  feedbackForm.value = emptyFeedbackForm();
  feedbackPostEditorMode.value = "write";
  nextTick(() => syncFeedbackMarkdownEditor("post", ""));
};

const ensureFeedbackReplyForm = id => {
  if (feedbackReplies.value[id]) return feedbackReplies.value[id];
  const form = emptyFeedbackReplyForm(false);
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

const FEEDBACK_MARKDOWN_OPTIONS = Object.freeze({ feedbackImages: true });

const feedbackPostPreviewHtml = computed(() => renderMarkdown(feedbackForm.value.body || "", FEEDBACK_MARKDOWN_OPTIONS));

const feedbackReplyPreviewHtml = id => renderMarkdown(feedbackReplies.value[id]?.body || "", FEEDBACK_MARKDOWN_OPTIONS);

const feedbackMessageHtml = message => renderMarkdown(message?.body || "", FEEDBACK_MARKDOWN_OPTIONS);

const FEEDBACK_INLINE_IMAGE_SCALE = 0.5;
let feedbackImageResizeRaf = 0;

const feedbackImageAvailableWidth = img => {
  const container = img?.closest?.(".md-image-scroll") || img?.closest?.(".markdown-body");
  if (!container) return 0;
  return Math.floor(container.clientWidth || container.getBoundingClientRect?.().width || 0);
};

const resizeFeedbackMarkdownImage = img => {
  if (!img) return;
  if (!img.complete || !img.naturalWidth) {
    if (!img.dataset.feedbackImageSizeBound) {
      img.dataset.feedbackImageSizeBound = "1";
      img.addEventListener("load", scheduleFeedbackMarkdownImageResize, { once: true });
    }
    return;
  }
  const available = feedbackImageAvailableWidth(img);
  if (!available) return;
  const targetWidth = img.naturalWidth > available
    ? available
    : Math.max(1, Math.round(img.naturalWidth * FEEDBACK_INLINE_IMAGE_SCALE));
  img.style.width = `${targetWidth}px`;
  img.style.height = "auto";
};

const resizeFeedbackMarkdownImages = () => {
  if (typeof document === "undefined") return;
  document.querySelectorAll(".markdown-body .md-feedback-image").forEach(resizeFeedbackMarkdownImage);
};

function scheduleFeedbackMarkdownImageResize() {
  if (typeof window === "undefined") return;
  if (feedbackImageResizeRaf) window.cancelAnimationFrame(feedbackImageResizeRaf);
  feedbackImageResizeRaf = window.requestAnimationFrame(() => {
    feedbackImageResizeRaf = 0;
    resizeFeedbackMarkdownImages();
  });
}

const feedbackImageViewer = ref({ open: false, url: "", alt: "" });

const openFeedbackImageViewer = (url, alt = "") => {
  const imageUrl = String(url || "").trim();
  if (!imageUrl) return;
  feedbackImageViewer.value = {
    open: true,
    url: imageUrl,
    alt: String(alt || "图片预览"),
  };
};

const closeFeedbackImageViewer = () => {
  feedbackImageViewer.value = { open: false, url: "", alt: "" };
};

const toggleFeedbackMarkdownImage = event => {
  const trigger = event?.target?.closest?.(".md-image-toggle");
  if (!trigger) return false;
  const image = trigger.querySelector(".md-feedback-image");
  if (!image) return false;
  event.preventDefault();
  event.stopPropagation();
  openFeedbackImageViewer(trigger.dataset.imageUrl || image.currentSrc || image.src, trigger.dataset.imageAlt || image.alt);
  return true;
};

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

const feedbackComposerHasDraft = () => Boolean(
  String(feedbackForm.value.body || "").trim()
  || (feedbackForm.value.files || []).length
);

const closeFeedbackComposer = async ({ force = false } = {}) => {
  if (feedbackSubmitting.value) return false;
  if (!force && feedbackComposerHasDraft()) {
    const discard = await askConfirm("当前帖子还没有发布，关闭后草稿将被清空。确定放弃草稿？", {
      title: "放弃发帖草稿",
      confirmText: "放弃草稿",
      tone: "danger",
    });
    if (!discard) return false;
  }
  destroyFeedbackMarkdownEditor("post");
  clearFeedbackForm();
  closeFeedbackMention();
  showFeedbackComposer.value = false;
  return true;
};

const closeFeedbackBoard = async () => {
  if (!await closeFeedbackComposer()) return;
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
  const imageRefs = form.imageRefs || [];
  for (const [index, file] of (form.files || []).entries()) {
    fd.append("images", file);
    fd.append("image_refs", imageRefs[index] || "");
  }
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
    files: [],
    imageRefs: [],
    previews: [],
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
  revokeFeedbackPreviews(feedbackEditing.value.previews || []);
  closeFeedbackMention();
  feedbackEditing.value = emptyFeedbackEditing();
};

const saveFeedbackEdit = async message => {
  if (!message?.id || feedbackEditing.value.id !== message.id) return;
  const body = (feedbackEditing.value.body || "").trim();
  const files = feedbackEditing.value.files || [];
  if (!body && !(message.attachments || []).length && !files.length) {
    showToast("留言内容不能为空", "error");
    return;
  }
  feedbackEditing.value = { ...feedbackEditing.value, saving: true };
  const fd = new FormData();
  fd.append("body", body);
  const imageRefs = feedbackEditing.value.imageRefs || [];
  for (const [index, file] of files.entries()) {
    fd.append("images", file);
    fd.append("image_refs", imageRefs[index] || "");
  }
  try {
    const r = await fetch(`/api/feedback/${encodeURIComponent(message.id)}`, {
      method: "PATCH",
      credentials: "include",
      body: fd,
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
    body: JSON.stringify({ job_ids: ids, force: true }),
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
let historyJobsController = null;
let taskCenterController = null;
let taskCenterPollTimer = null;
let taskCenterInitialized = false;
let homeDashboardController = null;
let compareJobsController = null;
let projectBulkJobsController = null;
let resultTableController = null;
let loadJobRequestSeq = 0;
let suppressSidebarAutoRefresh = false;
let suppressSidebarAutoRefreshToken = 0;
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
  else if (historyProjectView.value !== "all") params.set("project_view", historyProjectView.value);
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
    const expandedGroupIds = historyGroups.value
      .filter(group => collapsedGroups.value[group.id] || (filterProject.value && group.id === filterProject.value))
      .map(group => group.id);
    for (const groupId of expandedGroupIds) {
      collapsedGroups.value[groupId] = true;
      loadHistoryGroupJobs(groupId, true);
    }

  } catch (e) {
    if (e.name !== "AbortError") showToast(normalizeApiError(e, "加载历史记录失败"), "error");
  } finally {
    if (historyGroupsController === controller) {
      historyGroupsLoading.value = false;
      historyGroupsController = null;
    }
  }
};

const loadHistoryJobs = async () => {
  if (historyJobsController) historyJobsController.abort();
  const controller = new AbortController();
  historyJobsController = controller;
  historyJobsLoading.value = true;
  const params = new URLSearchParams();
  if (filterProject.value) params.set("project_id", filterProject.value);
  else if (historyProjectView.value !== "all") params.set("project_view", historyProjectView.value);
  if (historySearch.value.trim()) params.set("q", historySearch.value.trim());
  params.set("limit", String(historyJobsLimit.value));
  params.set("offset", String(historyJobsOffset.value));
  try {
    const r = await fetch(`/api/jobs?${params}`, {
      credentials: "include",
      signal: controller.signal,
    });
    const data = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, data, "加载历史任务失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
    if (historyJobsController !== controller) return;
    historyJobs.value = data.data || [];
    historyJobsTotal.value = data.total || 0;
  } catch (e) {
    if (e.name !== "AbortError") showToast(normalizeApiError(e, "加载历史任务失败"), "error");
  } finally {
    if (historyJobsController === controller) {
      historyJobsLoading.value = false;
      historyJobsController = null;
    }
  }
};

const taskCenterActiveJobs = computed(() =>
  taskCenterJobs.value.filter(job => job.status === "pending" || job.status === "running")
);
const taskCenterFailedJobs = computed(() =>
  taskCenterJobs.value.filter(job => job.status === "error").slice(0, 8)
);
const taskCenterActiveCount = computed(() => taskCenterActiveJobs.value.length);

const taskCenterJobLabel = job => job?.label || job?.file_a_name || String(job?.id || "").slice(0, 8);
const taskCenterJobMeta = job => {
  const parts = [];
  if (job?.project_name) parts.push(job.project_name);
  if (job?.created_at) parts.push(fmtDate(job.created_at));
  return parts.join(" · ") || "未分组任务";
};

const loadTaskCenter = async ({ silent = false } = {}) => {
  if (taskCenterController) {
    if (silent) return;
    taskCenterController.abort();
  }
  const controller = new AbortController();
  taskCenterController = controller;
  if (!silent) taskCenterLoading.value = true;
  const previousActive = new Map(taskCenterActiveJobs.value.map(job => [job.id, job]));
  try {
    const params = new URLSearchParams({
      statuses: "pending,running,error",
      limit: "40",
      offset: "0",
    });
    const r = await fetch(`/api/jobs?${params}`, {
      credentials: "include",
      signal: controller.signal,
    });
    const data = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, data, "加载任务中心失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
    if (taskCenterController !== controller) return;

    const jobs = data.data || [];
    taskCenterJobs.value = jobs;
    taskCenterError.value = "";
    if (taskCenterInitialized && previousActive.size) {
      const currentById = new Map(jobs.map(job => [job.id, job]));
      let shouldRefreshSidebar = false;
      for (const [jobId, previous] of previousActive) {
        const current = currentById.get(jobId);
        if (current?.status === "error") {
          showToast(`任务“${taskCenterJobLabel(current)}”分析失败`, "error", 4200);
          shouldRefreshSidebar = true;
          continue;
        }
        if (current?.status === "pending" || current?.status === "running") continue;
        try {
          const detail = await fetchJson(`/api/jobs/${encodeURIComponent(jobId)}`, { credentials: "include" }, "读取任务状态失败");
          if (detail?.status === "done") {
            showToast(`任务“${taskCenterJobLabel(detail) || taskCenterJobLabel(previous)}”分析完成`, "success", 4200);
            shouldRefreshSidebar = true;
          }
        } catch (_error) {
          // The task may have been deleted while the center was refreshing.
        }
      }
      if (shouldRefreshSidebar) {
        refreshSidebarData();
        loadHomeDashboard({ silent: true });
      }
    }
    taskCenterInitialized = true;
  } catch (e) {
    if (e.name !== "AbortError") {
      taskCenterError.value = normalizeApiError(e, "加载任务中心失败");
      if (!silent && !e?.authExpired) showToast(taskCenterError.value, "error");
    }
  } finally {
    if (taskCenterController === controller) {
      taskCenterLoading.value = false;
      taskCenterController = null;
    }
  }
};

let taskCenterReturnFocus = null;
const closeTaskCenter = () => {
  const panel = document.getElementById("task-center-panel");
  const shouldRestoreFocus = Boolean(panel?.contains(document.activeElement));
  showTaskCenter.value = false;
  if (shouldRestoreFocus) {
    nextTick(() => taskCenterReturnFocus?.focus?.({ preventScroll: true }));
  }
};
const toggleTaskCenter = event => {
  if (!showTaskCenter.value) taskCenterReturnFocus = document.activeElement;
  showTaskCenter.value = !showTaskCenter.value;
  closeActionMenu();
  if (showTaskCenter.value) {
    loadTaskCenter();
    if (!event?.currentTarget?.classList?.contains("task-center-trigger")) {
      nextTick(() => document.querySelector("#task-center-panel button:not([disabled])")?.focus?.({ preventScroll: true }));
    }
  }
};
const openTaskCenterJob = job => {
  closeTaskCenter();
  navigateToJob(job);
};
const copyTaskCenterError = async job => {
  const message = String(job?.error_msg || "").trim();
  if (!message) return showToast("该任务没有可复制的错误详情", "info");
  await copyTextToClipboard(message);
};

const loadHomeDashboard = async ({ silent = false } = {}) => {
  if (homeDashboardController) {
    if (silent) return;
    homeDashboardController.abort();
  }
  const controller = new AbortController();
  homeDashboardController = controller;
  if (!silent) homeDashboardLoading.value = true;
  try {
    const params = new URLSearchParams({ statuses: "done", limit: "6", offset: "0" });
    const r = await fetch(`/api/jobs?${params}`, {
      credentials: "include",
      signal: controller.signal,
    });
    const data = await readJsonResponse(r, {});
    if (!r.ok) {
      throw new ApiRequestError(apiErrorMessage(r, data, "加载首页任务失败"), {
        status: r.status,
        authExpired: r.status === 401,
      });
    }
    if (homeDashboardController !== controller) return;
    homeRecentJobs.value = data.data || [];
    homeDashboardError.value = "";
  } catch (e) {
    if (e.name !== "AbortError") {
      homeDashboardError.value = normalizeApiError(e, "加载首页任务失败");
      if (!silent && !e?.authExpired) showToast(homeDashboardError.value, "error");
    }
  } finally {
    if (homeDashboardController === controller) {
      homeDashboardController = null;
      homeDashboardLoading.value = false;
    }
  }
};

const updateHistoryGroup = (groupId, patch) => {
  historyGroups.value = historyGroups.value.map(group =>
    group.id === groupId ? { ...group, ...patch } : group
  );
};

const mergeHistoryJobs = (existing = [], incoming = []) => {
  const seen = new Set();
  const merged = [];
  for (const job of [...existing, ...incoming]) {
    if (!job?.id || seen.has(job.id)) continue;
    seen.add(job.id);
    merged.push(job);
  }
  return merged;
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
    const incomingJobs = data.data || [];
    const jobs = reset ? incomingJobs : mergeHistoryJobs(latest.jobs, incomingJobs);
    updateHistoryGroup(groupId, {
      jobs,
      jobs_total: data.total || 0,
      jobs_offset: reset ? incomingJobs.length : offset + incomingJobs.length,
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

const sidebarJobSnapshot = job => ({
  id: job.id,
  seq: job.seq,
  label: job.label,
  status: job.status,
  mode: job.mode,
  created_at: job.created_at,
  is_pinned: job.is_pinned,
  is_owner: job.is_owner,
  project_id: job.project_id,
});

const upsertHistoryGroupJob = (groupId, job) => {
  if (!groupId || !job?.id) return;
  const group = historyGroups.value.find(item => item.id === groupId);
  if (!group) return;
  const jobs = group.jobs || [];
  const index = jobs.findIndex(item => item.id === job.id);
  if (index < 0) return;
  const nextJobs = [...jobs];
  nextJobs.splice(index, 1, { ...jobs[index], ...sidebarJobSnapshot(job) });
  updateHistoryGroup(groupId, {
    jobs: nextJobs,
    jobs_loaded: true,
    jobs_total: Math.max(Number(group.jobs_total || 0), Number(group.job_count || 0), jobs.length),
  });
};

const setSidebarFiltersSilently = patch => {
  const token = ++suppressSidebarAutoRefreshToken;
  suppressSidebarAutoRefresh = true;
  if (Object.prototype.hasOwnProperty.call(patch, "historySearch")) historySearch.value = patch.historySearch;
  if (Object.prototype.hasOwnProperty.call(patch, "historyProjectView")) historyProjectView.value = patch.historyProjectView;
  if (Object.prototype.hasOwnProperty.call(patch, "filterProject")) filterProject.value = patch.filterProject;
  historyGroupsOffset.value = 0;
  historyJobsOffset.value = 0;
  compareJobsOffset.value = 0;
  historySelection.value = [];
  localStorage.setItem("tpa-filter-project", filterProject.value);
  localStorage.setItem("tpa-history-project-view", historyProjectView.value);
  nextTick(() => {
    if (suppressSidebarAutoRefreshToken === token) suppressSidebarAutoRefresh = false;
  });
};

const focusCurrentJobInSidebar = async job => {
  if (!job?.id) return;
  const focusJobId = job.id;
  const groupId = sidebarProjectIdForJob(job);
  if (!groupId) return;
  sidebarTab.value = "jobs";
  collapsedGroups.value = { ...collapsedGroups.value, [groupId]: true };

  try {
    let needsReload = false;
    const needsProjectFocus =
      filterProject.value !== groupId ||
      historyProjectView.value !== "all" ||
      Boolean(historySearch.value.trim());
    if (needsProjectFocus) {
      setSidebarFiltersSilently({
        historySearch: "",
        historyProjectView: "all",
        filterProject: groupId,
      });
      needsReload = true;
    }
    if (!historyGroups.value.some(group => group.id === groupId)) {
      needsReload = true;
    }
    if (needsReload) {
      if (needsProjectFocus) await refreshSidebarData();
      else await loadHistoryGroups();
    }

    if (!historyGroups.value.some(group => group.id === groupId)) {
      setSidebarFiltersSilently({
        historySearch: "",
        historyProjectView: "all",
        filterProject: groupId,
      });
      await refreshSidebarData();
    }

    const group = historyGroups.value.find(item => item.id === groupId);
    if (!group || selectedJobId.value !== focusJobId) return;
    collapsedGroups.value = { ...collapsedGroups.value, [groupId]: true };
    if (!group.jobs_loaded || !(group.jobs || []).some(item => item.id === focusJobId)) {
      await loadHistoryGroupJobs(groupId, true);
    }
    if (selectedJobId.value === focusJobId) upsertHistoryGroupJob(groupId, job);
  } catch (e) {
    if (e.name !== "AbortError") console.warn("focusCurrentJobInSidebar failed", e);
  }
};

const compareProjectForSubmit = () => {
  if (compareProjectId.value === "__none__") return null;
  return compareProjectId.value || null;
};

const loadCompareJobs = async () => {
  if (compareJobsController) compareJobsController.abort();
  if (!showCompareModal.value || !compareProjectId.value) {
    compareJobs.value = [];
    compareJobsTotal.value = 0;
    compareJobsLoading.value = false;
    compareJobsController = null;
    return;
  }
  const controller = new AbortController();
  compareJobsController = controller;
  compareJobsLoading.value = true;
  const params = new URLSearchParams();
  params.set("project_id", compareProjectId.value);
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
  const loaders = [loadHistoryGroups(), loadHistoryJobs()];
  if (showCompareModal.value) loaders.push(loadCompareJobs());
  if (showProjectBulkModal.value) loaders.push(loadProjectBulkJobs(true));
  await Promise.all(loaders);
};

const initializeAppData = async () => {
  if (appInitialized && !authInitError.value) {
    return !authRequired.value || Boolean(currentUser.value);
  }
  authInitError.value = "";
  try {
    if (!await loadConfig()) return false;
    await loadMe();
    if (authRequired.value && !currentUser.value) {
      appInitialized = true;
      return false;
    }
    await loadProjects();
    await refreshSidebarData();
    await Promise.all([
      loadTaskCenter({ silent: true }),
      loadHomeDashboard({ silent: true }),
    ]);
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
  const jobHandle = String(id || "");
  const requestSeq = ++loadJobRequestSeq;
  const r = await fetch(`/api/jobs/${encodeURIComponent(jobHandle)}`, { credentials: "include" });
  const data = await readJsonResponse(r, {});
  if (requestSeq !== loadJobRequestSeq) return "stale";
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
    forgetRecentViewedJob(jobHandle);
    return false;
  }
  selectedJob.value = data;
  selectedJobId.value = data.id;
  clearProjectFilterIfJobIsHidden(data);
  rememberRecentProject(data.project_id);
  rememberRecentViewedJob(data);
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
  chartAddedRows.value = [];
  chartRemovedRows.value = [];
  chartBarRows.value = [];
  chartCanvasReady.value = false;
  chartScatterRows.value = [];
  chartDumbbellActive.value = false;
  chartCompareDirection.value = "all";
  chartMinAbsDelta.value = 0;
  chartMinDeltaPct.value = 0;
  chartMinBaselineMs.value = 0;
  chartExcludeCommunication.value = true;
  chartFilterResultText.value = "";
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
const isTritonCodeOptimizationFileCell = context =>
  normalizeMarkdownHeadingTitle(context?.sectionTitle) === "Triton Kernel 代码优化"
  && normalizeMarkdownHeadingTitle(context?.columnHeader) === "代码文件";

const renderAiArtifactCode = (code, context = {}) => {
  const url = aiArtifactDownloadUrl(code);
  if (!url) return "";
  const path = resolveAiArtifactPath(code);
  if (isAiCodeArtifactPath(path) && isTritonCodeOptimizationFileCell(context)) {
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
        if (jobId) {
          const handle = selectedJobId.value === jobId ? currentJobRouteHandle() : jobId;
          window.location.hash = `#${jobRoutePath(handle, "ai")}`;
        }
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
  const requestedLimit = overrides.limit ?? state?.tableLimit ?? tableLimit.value;
  const limit = overrides.limit === undefined
    ? Math.min(Math.max(1, Number(requestedLimit) || 100), TABLE_MAX_RENDER_ROWS)
    : requestedLimit;
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

const KERNEL_HOST_OP_DETAIL_LIMIT = 50;
const KERNEL_HOST_OP_DETAIL_CACHE_LIMIT = 100;
const kernelHostOpDetailCache = new Map();
const kernelHostOpNumber = value => {
  const number = Number.parseFloat(String(value ?? "").replace("%", ""));
  return Number.isFinite(number) ? number : null;
};
const canExpandKernelHostOpRow = row => {
  if (resultTab.value !== "all_kernels_avg.csv" || !hasKernelHostOpRelations.value || !row?.kernel_name) {
    return false;
  }
  const count = kernelHostOpNumber(row.host_op_count);
  const coverage = kernelHostOpNumber(row.host_op_coverage_pct);
  if (count === null || coverage === null) return false;
  return count !== 1 || coverage < 99.999;
};
const isKernelHostOpDetailExpanded = row =>
  Boolean(row?.kernel_name && kernelHostOpDetail.value.kernelName === row.kernel_name);
const kernelHostOpDetailReason = row => {
  const count = kernelHostOpNumber(row?.host_op_count) || 0;
  const coverage = kernelHostOpNumber(row?.host_op_coverage_pct) || 0;
  if (count === 0 || coverage <= 0) return "未匹配";
  if (coverage < 99.999) return "部分匹配";
  return "多关联";
};
const kernelHostOpMatchLabel = method => ({
  external_id: "External id",
  tf_op: "tf_op",
  unmatched: "未匹配",
}[method] || method || "未匹配");
const closeKernelHostOpDetail = () => {
  kernelHostOpDetailController?.abort();
  kernelHostOpDetailController = null;
  kernelHostOpDetail.value = { kernelName: "", rows: [], total: 0, loading: false, error: "" };
};
const loadKernelHostOpDetail = async row => {
  const jobId = selectedJobId.value;
  const kernelName = String(row?.kernel_name || "").trim();
  if (!jobId || !kernelName || !canExpandKernelHostOpRow(row)) return;
  kernelHostOpDetailController?.abort();
  kernelHostOpDetailController = null;
  const cacheKey = `${jobId}\n${kernelName}`;
  const cached = kernelHostOpDetailCache.get(cacheKey);
  if (cached) {
    kernelHostOpDetailCache.delete(cacheKey);
    kernelHostOpDetailCache.set(cacheKey, cached);
    kernelHostOpDetail.value = { kernelName, ...cached, loading: false, error: "" };
    return;
  }
  const controller = new AbortController();
  kernelHostOpDetailController = controller;
  kernelHostOpDetail.value = { kernelName, rows: [], total: 0, loading: true, error: "" };
  try {
    const params = new URLSearchParams({
      kernel_name: kernelName,
      limit: String(KERNEL_HOST_OP_DETAIL_LIMIT),
    });
    const response = await fetch(
      `/api/jobs/${encodeURIComponent(jobId)}/kernel-host-ops?${params}`,
      { credentials: "include", signal: controller.signal },
    );
    const data = await readJsonResponse(response, {});
    if (!response.ok) {
      throw new ApiRequestError(apiErrorMessage(response, data, "加载 Host Op 关联失败"), {
        status: response.status,
        authExpired: response.status === 401,
      });
    }
    if (kernelHostOpDetailController !== controller) return;
    if (selectedJobId.value !== jobId || kernelHostOpDetail.value.kernelName !== kernelName) return;
    const detail = {
      rows: data.rows || [],
      total: data.filtered_total ?? data.total ?? 0,
      loading: false,
      error: "",
    };
    kernelHostOpDetailCache.set(cacheKey, detail);
    if (kernelHostOpDetailCache.size > KERNEL_HOST_OP_DETAIL_CACHE_LIMIT) {
      kernelHostOpDetailCache.delete(kernelHostOpDetailCache.keys().next().value);
    }
    kernelHostOpDetail.value = { kernelName, ...detail };
  } catch (error) {
    if (error.name === "AbortError" || kernelHostOpDetailController !== controller) return;
    kernelHostOpDetail.value = {
      kernelName,
      rows: [],
      total: 0,
      loading: false,
      error: normalizeApiError(error, "加载 Host Op 关联失败"),
    };
  } finally {
    if (kernelHostOpDetailController === controller) kernelHostOpDetailController = null;
  }
};
const toggleKernelHostOpDetail = row => {
  if (isKernelHostOpDetailExpanded(row)) {
    closeKernelHostOpDetail();
    return;
  }
  loadKernelHostOpDetail(row);
};
const retryKernelHostOpDetail = () => {
  const row = filteredRows.value.find(item => item.kernel_name === kernelHostOpDetail.value.kernelName);
  if (row) loadKernelHostOpDetail(row);
};

const loadResultTable = async ({ resetOffset = false, filename = resultTab.value, viewState = null } = {}) => {
  const jobId = selectedJobId.value;
  if (!jobId || !filename?.endsWith(".csv")) return;
  if (kernelHostOpDetail.value.kernelName) closeKernelHostOpDetail();
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
  if (filename !== "all_kernels_avg.csv" && kernelHostOpDetail.value.kernelName) {
    closeKernelHostOpDetail();
  }
  if (savePrevious) saveResultViewState(activeResultStateJobId, resultTab.value);
  if (resultTableController) resultTableController.abort();

  const controller = new AbortController();
  const state = resultViewStateFor(jobId, filename);
  const hasSavedView = Boolean(readResultMemory(jobId).tabs?.[filename]);
  if (filename.endsWith("_cmp.csv")) {
    const fields = selectedJob.value?.result_files?.[filename]?.fields || [];
    const hasLegacyColumns = (state.visibleColumns || []).some(field => LEGACY_COMPARISON_TABLE_FIELDS.has(field));
    if (!hasSavedView || hasLegacyColumns) {
      state.visibleColumns = comparisonPresetColumns("compact", fields);
    }
    if (LEGACY_COMPARISON_TABLE_FIELDS.has(state.sortCol)) {
      state.sortCol = "";
      state.sortAsc = true;
    }
  }
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
    if (shouldUseEfficiencyPreset(filename, data.fields || [], state)) {
      visibleColumns.value = efficiencyPresetColumns(data.fields || []);
    }
    resultTableLoading.value = false;
    resultTableError.value = "";
    showColumnMenu.value = false;
    skipNextResultTabWatch();
    resultTab.value = filename;
    saveResultViewState(jobId, filename);
    if (updateRoute) router.push({ path: jobRoutePath(selectedJob.value || jobId, filename) });
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
    if (updateRoute) router.push({ path: jobRoutePath(selectedJob.value || jobId, filename) });
  } finally {
    if (resultTableController === controller) {
      resultTableController = null;
      preparingResultTab.value = "";
    }
  }
};

const canDrillKernelTypeRow = row =>
  isKernelTypeTab.value && Boolean(row?.type);

const isKernelTypeDrillCell = (field, row) =>
  field === "type" && canDrillKernelTypeRow(row);

const drillDownKernelType = async row => {
  if (!canDrillKernelTypeRow(row)) return;
  const type = String(row.type || "").trim();
  if (!type) return;
  const targetFile = selectedJob.value?.mode === "compare"
    ? "all_kernels_cmp.csv"
    : "all_kernels_avg.csv";
  const fields = selectedJob.value?.result_files?.[targetFile]?.fields || [];
  if (!fields.length) {
    showToast(`未找到 ${targetFile}`, "error");
    return;
  }
  const canFilterFamily = fields.includes("family");
  const sortField = selectedJob.value?.mode === "compare" && fields.includes("delta_dur_ms")
    ? "delta_dur_ms"
    : (fields.includes("avg_dur_ms") ? "avg_dur_ms" : "");
  const state = {
    ...defaultResultViewState(),
    tableLimit: tableLimit.value || 100,
    tableOffset: 0,
    sortCol: sortField,
    sortAsc: false,
    tableSearch: canFilterFamily ? "" : type,
    colFilters: canFilterFamily ? { family: type } : {},
    colFilterOps: canFilterFamily ? { family: "==" } : {},
    visibleColumns: canFilterFamily ? fields.filter(field => field !== "family") : fields,
  };
  const memory = readResultMemory(selectedJobId.value);
  memory.tabs = { ...(memory.tabs || {}), [targetFile]: state };
  writeResultMemory(selectedJobId.value, memory);
  await activateCsvTab(targetFile);
  showToast(
    canFilterFamily ? `已下钻到 ${type} 相关 Kernel` : `已用搜索下钻到 ${type} 相关 Kernel`,
    "success",
  );
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
  tableLimit.value = Math.min(Math.floor(next), TABLE_MAX_RENDER_ROWS);
  tableOffset.value = 0;
  loadResultTable();
};

const showAllTableRows = () => {
  const total = Number(tableTotalRows.value);
  if (!Number.isFinite(total) || total <= 0) return;
  if (total > TABLE_MAX_RENDER_ROWS) {
    showToast(`为保证页面流畅，最多展示前 ${fmtCount(TABLE_MAX_RENDER_ROWS)} 行；可导出全部筛选结果`, "info", 4200);
  }
  changeTableLimit(Math.min(total, TABLE_MAX_RENDER_ROWS));
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
const chartTopNOptions = [5, 10, 15, 20, 30];
const CHART_FETCH_TOP_LIMIT = 100;

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

const fmtChartPercent = value =>
  Number.isFinite(value) ? `${value.toFixed(1)}%` : "0.0%";

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
        compareStatus: String(row.compare_status || "").trim(),
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
  const raw = row.raw || row;
  const family = String(raw.family ?? raw.type ?? "").trim().toLowerCase();
  if (CHART_COMMUNICATION_FAMILIES.has(family)) return true;
  const nameField = row.nameField || sourceConfig?.nameField;
  const label = String(row.label ?? raw[nameField] ?? raw.kernel_name ?? raw.op_name ?? raw.type ?? "").trim();
  return CHART_COMMUNICATION_NAME_RE.test(label);
};

const filterChartCommunicationRows = (rows, sourceConfig) =>
  (rows || []).filter(row => !isCommunicationChartRow(row, sourceConfig));
const applyChartCommunicationFilter = (rows, sourceConfig) =>
  chartExcludeCommunication.value ? filterChartCommunicationRows(rows, sourceConfig) : (rows || []);

const chartFilterParams = () => {
  const params = {
    exclude_communication: String(chartExcludeCommunication.value),
  };
  if (chartScatterMode.value === "compare") {
    params.chart_direction = chartCompareDirection.value;
    params.min_abs_delta = String(Math.max(0, Number(chartMinAbsDelta.value) || 0));
    params.min_delta_pct = String(Math.max(0, Number(chartMinDeltaPct.value) || 0));
    params.min_baseline_ms = String(Math.max(0, Number(chartMinBaselineMs.value) || 0));
  }
  return params;
};
const chartFilterCacheKey = () => new URLSearchParams(chartFilterParams()).toString();
const resetChartCompareFilters = () => {
  chartCompareDirection.value = "all";
  chartMinAbsDelta.value = 0;
  chartMinDeltaPct.value = 0;
  chartMinBaselineMs.value = 0;
  chartExcludeCommunication.value = true;
  buildChart();
};

const sortChartRows = (rows, metricDef) => {
  const metricValue = row => metricDef.signed ? Math.abs(row.value) : row.value;
  return [...rows]
    .filter(row => metricValue(row) > 0)
    .sort((a, b) => metricValue(b) - metricValue(a));
};

const buildChartBarRows = (rows, metricDef, topN, fullMetricTotal = null) => {
  const sorted = sortChartRows(rows, metricDef);
  const total = Number.isFinite(Number(fullMetricTotal))
    ? Number(fullMetricTotal)
    : sorted.reduce((sum, row) => sum + row.displayValue, 0);
  return sorted.slice(0, topN).map(row => {
    const sharePct = total ? row.displayValue / total * 100 : 0;
    return { ...row, sharePct, shareLabel: fmtChartPercent(sharePct) };
  });
};

const buildCallTimeScatterRows = rows => {
  const prepared = (rows || []).map(row => {
    const calls = parseChartNumber(row.raw?.avg_count);
    const totalMs = parseChartNumber(row.raw?.avg_dur_ms);
    const rawUsPerCall = parseChartNumber(row.raw?.avg_us_per_call);
    const usPerCall = rawUsPerCall > 0 ? rawUsPerCall : (calls > 0 ? totalMs * 1000 / calls : 0);
    return { ...row, calls, totalMs, usPerCall };
  })
    .filter(row => row.calls > 0 && row.usPerCall > 0)
    .sort((a, b) => b.totalMs - a.totalMs)
    .slice(0, CHART_FETCH_TOP_LIMIT);
  const maxDuration = Math.max(1, ...prepared.map(row => row.totalMs));
  return prepared.map(row => ({
    ...row,
    radius: 4 + Math.sqrt(row.totalMs / maxDuration) * 8,
  }));
};

const buildCompareScatterRows = rows => {
  const prepared = (rows || [])
    .map(row => {
      const baselineMs = parseChartNumber(row.raw?.avg_dur_ms_A);
      const currentMs = parseChartNumber(row.raw?.avg_dur_ms_B);
      const delta = parseChartNumber(row.raw?.delta_dur_ms);
      const changePct = baselineMs > 0 ? delta / baselineMs * 100 : null;
      const inferredMatch = ["inferred", "low_confidence"].includes(row.compareStatus);
      return { ...row, baselineMs, currentMs, delta, changePct, absDelta: Math.abs(delta), inferredMatch };
    })
    .filter(row => row.baselineMs > 0 && row.currentMs > 0 && Number.isFinite(row.changePct) && row.absDelta > 0)
    .sort((a, b) => b.absDelta - a.absDelta)
    .slice(0, CHART_FETCH_TOP_LIMIT)
    .sort((a, b) => a.absDelta - b.absDelta);
  const maxDelta = Math.max(1, ...prepared.map(row => row.absDelta));
  return prepared.map(row => ({
    ...row,
    radius: 3 + Math.sqrt(row.absDelta / maxDelta) * 7,
  }));
};

const logScaleTickLabel = value => {
  const number = Number(value);
  if (!(number > 0)) return "";
  const exponent = Math.floor(Math.log10(number));
  const mantissa = number / (10 ** exponent);
  if (![1, 5].some(mark => Math.abs(mantissa - mark) < 1e-8)) return "";
  return String(Number(number.toPrecision(3)));
};

const chartShareHeader = metricDef => metricDef.signed ? "绝对值占比" : "占比";
const chartFallbackBarWidth = row => {
  const value = Number(row?.displayValue) || Math.abs(Number(row?.value) || 0);
  return `${Math.max(2, Math.min(100, value / chartFallbackMaxValue.value * 100))}%`;
};
const chartFallbackBarTone = row =>
  activeChartMetricDef.value.signed
    ? (Number(row?.value) >= 0 ? "neg" : "pos")
    : "";
const chartFallbackValueLabel = row => fmtChartValue(Number(row?.value) || 0, activeChartMetricDef.value);

const chartValueLabelsPlugin = {
  id: "chartValueLabels",
  afterDatasetsDraw(chart, _args, options = {}) {
    const rows = options.rows || [];
    const metricDef = options.metricDef || {};
    const meta = chart.getDatasetMeta(0);
    const area = chart.chartArea;
    if (!rows.length || !meta?.data?.length || !area) return;

    const fontFamily = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    const ctx = chart.ctx;
    const shareColumnWidth = options.shareColumnWidth || 56;
    const maxRight = chart.width - shareColumnWidth - 12;
    ctx.save();
    ctx.textBaseline = "middle";
    ctx.fillStyle = options.color || "#1e293b";
    ctx.font = `700 11px ${fontFamily}`;
    meta.data.forEach((element, index) => {
      const row = rows[index];
      if (!row) return;
      const label = fmtChartValue(row.value, metricDef);
      const width = ctx.measureText(label).width;
      const isNegative = Number(row.value) < 0;
      const y = element.y;
      let x = isNegative ? element.x - 6 : element.x + 6;
      let align = isNegative ? "right" : "left";

      if (!isNegative && x + width > maxRight) {
        x = Math.min(element.x - 6, maxRight);
        align = "right";
      } else if (isNegative && x - width < area.left) {
        x = Math.max(element.x + 6, area.left + 2);
        align = "left";
      }

      if (align === "left" && x + width > maxRight) x = maxRight - width;
      if (align === "right" && x - width < area.left) x = area.left + width + 2;
      ctx.textAlign = align;
      ctx.fillText(label, x, y);
    });
    ctx.restore();
  },
};

const chartShareLabelsPlugin = {
  id: "chartShareLabels",
  afterDraw(chart, _args, options = {}) {
    const rows = options.rows || [];
    const meta = chart.getDatasetMeta(0);
    const area = chart.chartArea;
    if (!rows.length || !meta?.data?.length || !area) return;

    const x = chart.width - 10;
    const fontFamily = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    const ctx = chart.ctx;
    ctx.save();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillStyle = options.headerColor || options.color || "#475569";
    ctx.font = `700 11px ${fontFamily}`;
    if (options.header) {
      ctx.fillText(options.header, x, Math.max(12, area.top - 12));
    }
    ctx.fillStyle = options.color || "#475569";
    ctx.font = `700 11px ${fontFamily}`;
    meta.data.forEach((element, index) => {
      const row = rows[index];
      if (row) ctx.fillText(row.shareLabel || fmtChartPercent(row.sharePct), x, element.y);
    });
    ctx.restore();
  },
};

const chartScatterGuidesPlugin = {
  id: "chartScatterGuides",
  beforeDatasetsDraw(chart, _args, options = {}) {
    if (!options.compare || !chart.chartArea || !chart.scales?.x || !chart.scales?.y) return;
    const { ctx, chartArea } = chart;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    const zeroY = Math.max(chartArea.top, Math.min(chartArea.bottom, chart.scales.y.getPixelForValue(0)));
    ctx.save();
    ctx.fillStyle = options.regressionFill || "rgba(239,68,68,.045)";
    ctx.fillRect(chartArea.left, chartArea.top, chartArea.width, Math.max(0, zeroY - chartArea.top));
    ctx.fillStyle = options.improvementFill || "rgba(34,197,94,.045)";
    ctx.fillRect(chartArea.left, zeroY, chartArea.width, Math.max(0, chartArea.bottom - zeroY));
    ctx.strokeStyle = options.equalityColor || "rgba(100,116,139,.72)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(chartArea.left, zeroY);
    ctx.lineTo(chartArea.right, zeroY);
    ctx.stroke();

    const guideRate = Number(options.relativeGuideRate) || 0;
    if (guideRate > 0 && xScale.min > 0 && xScale.max > xScale.min) {
      const minLog = Math.log10(xScale.min);
      const maxLog = Math.log10(xScale.max);
      ctx.beginPath();
      ctx.rect(chartArea.left, chartArea.top, chartArea.width, chartArea.height);
      ctx.clip();
      ctx.strokeStyle = options.relativeGuideColor || "rgba(100,116,139,.54)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      for (const rate of [guideRate, -guideRate]) {
        ctx.beginPath();
        let labelPoint = null;
        for (let index = 0; index <= 64; index += 1) {
          const xValue = 10 ** (minLog + (maxLog - minLog) * index / 64);
          const yValue = xValue * rate;
          const x = xScale.getPixelForValue(xValue);
          const y = yScale.getPixelForValue(yValue);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          if (y >= chartArea.top + 8 && y <= chartArea.bottom - 8) labelPoint = { x, y };
        }
        ctx.stroke();
        if (labelPoint) {
          ctx.setLineDash([]);
          ctx.fillStyle = options.relativeGuideText || options.relativeGuideColor || "rgba(100,116,139,.72)";
          ctx.font = "700 10px system-ui, sans-serif";
          ctx.textAlign = "right";
          ctx.textBaseline = rate > 0 ? "bottom" : "top";
          ctx.fillText(`${rate > 0 ? "+" : "−"}${trimNumber(guideRate * 100)}%`, labelPoint.x - 4, labelPoint.y + (rate > 0 ? -2 : 2));
          ctx.setLineDash([3, 4]);
        }
      }
    }
    ctx.restore();
  },
  afterDraw(chart, _args, options = {}) {
    if (!options.compare || !chart.chartArea) return;
    const { ctx, chartArea } = chart;
    ctx.save();
    ctx.font = "700 10px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillStyle = options.regressionText || "rgba(220,38,38,.78)";
    ctx.fillText("B 更慢（耗时增加）", chartArea.left + 8, chartArea.top + 15);
    ctx.textAlign = "right";
    ctx.fillStyle = options.improvementText || "rgba(22,163,74,.78)";
    ctx.fillText("B 更快（耗时减少）", chartArea.right - 8, chartArea.bottom - 8);
    ctx.restore();
  },
};

const chartCompareDumbbellPlugin = {
  id: "chartCompareDumbbell",
  beforeDatasetsDraw(chart, _args, options = {}) {
    const rows = options.rows || [];
    const xScale = chart.scales?.x;
    const yScale = chart.scales?.y;
    if (!rows.length || !xScale || !yScale) return;
    const ctx = chart.ctx;
    ctx.save();
    rows.forEach(row => {
      const xA = xScale.getPixelForValue(row.aValue);
      const xB = xScale.getPixelForValue(row.bValue);
      const y = yScale.getPixelForValue(row.label);
      if (![xA, xB, y].every(Number.isFinite)) return;
      const linked = chart.$linkedLabel === row.label;
      ctx.strokeStyle = row.delta >= 0 ? options.regressionColor : options.improvementColor;
      ctx.globalAlpha = linked ? .88 : .48;
      ctx.lineWidth = linked ? 5 : 3;
      ctx.beginPath();
      ctx.moveTo(xA, y);
      ctx.lineTo(xB, y);
      ctx.stroke();
    });
    ctx.restore();
  },
  afterDatasetsDraw(chart, _args, options = {}) {
    const rows = options.rows || [];
    const yScale = chart.scales?.y;
    if (!rows.length || !yScale) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.fillStyle = options.labelColor || "#475569";
    ctx.font = "700 11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    rows.forEach(row => {
      const y = yScale.getPixelForValue(row.label);
      if (!Number.isFinite(y)) return;
      ctx.fillText(`${trimNumber(row.aValue)} → ${trimNumber(row.bValue)}`, chart.width - 10, y);
    });
    ctx.restore();
  },
};

const chartScatterHotspotLabelsPlugin = {
  id: "chartScatterHotspotLabels",
  afterDatasetsDraw(chart, _args, options = {}) {
    const rows = options.rows || [];
    const meta = chart.getDatasetMeta(0);
    const area = chart.chartArea;
    if (!rows.length || !meta?.data?.length || !area) return;
    const ranked = rows
      .map((row, index) => ({ row, index, impact: row.absDelta ?? row.totalMs ?? 0 }))
      .sort((a, b) => b.impact - a.impact)
      .slice(0, options.maxLabels || 5);
    const boxes = [];
    const ctx = chart.ctx;
    ctx.save();
    ctx.font = "700 10px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (const item of ranked) {
      const point = meta.data[item.index];
      if (!point) continue;
      const label = shortChartLabel(item.row.label, 22);
      const width = ctx.measureText(label).width + 10;
      const height = 18;
      const candidates = [
        { x: point.x + 8, y: point.y - height - 5 },
        { x: point.x - width - 8, y: point.y - height - 5 },
        { x: point.x + 8, y: point.y + 5 },
      ];
      const box = candidates.find(candidate =>
        candidate.x >= area.left && candidate.x + width <= area.right
        && candidate.y >= area.top + (options.topInset || 0)
        && candidate.y + height <= area.bottom - (options.bottomInset || 0)
        && boxes.every(used => candidate.x + width < used.x || candidate.x > used.x + used.width
          || candidate.y + height < used.y || candidate.y > used.y + used.height)
      );
      if (!box) continue;
      boxes.push({ ...box, width, height });
      ctx.fillStyle = options.background || "rgba(255,255,255,.88)";
      ctx.strokeStyle = options.border || "rgba(148,163,184,.42)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") ctx.roundRect(box.x, box.y, width, height, 4);
      else ctx.rect(box.x, box.y, width, height);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = options.color || "#334155";
      ctx.textAlign = "left";
      ctx.fillText(label, box.x + 5, box.y + height / 2);
    }
    ctx.restore();
  },
};

const buildChartSummary = (rows, table, sourceConfig) => {
  const totalLabel = table?.aggregated
    ? `全量 ${table.total || 0} 项`
    : table?.total && table.total > rows.length
    ? `前 ${rows.length} / 共 ${table.total} 项`
    : `${rows.length} 项`;
  const makeCard = (label, value, sub = "", tone = "", row = null) => ({ label, value, sub, tone, row });
  const highlightRow = key => normalizeChartRows(
    table?.highlights?.[key] ? [table.highlights[key]] : [],
    table?.fields || [],
    sourceConfig,
    activeChartMetricDef.value,
  )[0] || null;
  if (selectedJob.value?.mode === "compare") {
    const totalA = parseChartNumber(table?.sums?.avg_dur_ms_A ?? rows.reduce((sum, row) => sum + row.aValue, 0));
    const totalB = parseChartNumber(table?.sums?.avg_dur_ms_B ?? rows.reduce((sum, row) => sum + row.bValue, 0));
    const totalDelta = parseChartNumber(table?.sums?.delta_dur_ms ?? rows.reduce((sum, row) => sum + row.delta, 0));
    const deltaPct = totalA ? totalDelta / totalA * 100 : null;
    const conclusion = deltaPct === null
      ? "暂无基线"
      : Math.abs(deltaPct) < 1 ? "基本持平" : deltaPct > 0 ? "回退" : "改善";
    const conclusionTone = deltaPct === null || Math.abs(deltaPct) < 1 ? "" : deltaPct > 0 ? "neg" : "pos";
    const signedDelta = `${totalDelta > 0 ? "+" : ""}${fmtChartValue(totalDelta, { unit: "ms" })}`;
    const signedPct = deltaPct === null ? "变化率不可计算" : `${deltaPct > 0 ? "+" : ""}${trimNumber(deltaPct)}%`;
    const directionStats = table?.direction_stats || {};
    const statusStats = table?.status_stats || {};
    const slowdownCount = directionStats.slowdown_count ?? rows.filter(row => row.delta > 0).length;
    const speedupCount = directionStats.speedup_count ?? rows.filter(row => row.delta < 0).length;
    const slowest = highlightRow("slowdown") || rows.filter(row => row.delta > 0).sort((a, b) => b.delta - a.delta)[0] || null;
    const fastest = highlightRow("speedup") || rows.filter(row => row.delta < 0).sort((a, b) => a.delta - b.delta)[0] || null;
    const cards = [
      makeCard("A 计算耗时", fmtChartValue(totalA, { unit: "ms" }), `已排除通信 · ${totalLabel}`),
      makeCard("B 计算耗时", fmtChartValue(totalB, { unit: "ms" }), `已排除通信 · ${sourceConfig.label} · ${totalLabel}`),
      makeCard("整体结论", deltaPct === null ? conclusion : `${conclusion} ${signedPct}`, `B - A: ${signedDelta}`, conclusionTone),
      makeCard("回退 / 改善项", `${slowdownCount} / ${speedupCount}`, `净变化 ${signedDelta}`, conclusionTone),
      makeCard("最大回退", slowest ? fmtChartValue(slowest.delta, { unit: "ms" }) : "0", slowest ? shortChartLabel(slowest.label, 28) : "无", "neg", slowest),
      makeCard("最大改善", fastest ? fmtChartValue(fastest.delta, { unit: "ms" }) : "0", fastest ? shortChartLabel(fastest.label, 28) : "无", "pos", fastest),
    ];
    cards.push(
      makeCard(
        "新增 / 消失",
        `${statusStats.added_count || 0} / ${statusStats.removed_count || 0}`,
        `新增 +${trimNumber(parseChartNumber(statusStats.added_duration))} ms · 消失 -${trimNumber(parseChartNumber(statusStats.removed_duration))} ms`,
      ),
      makeCard(
        "匹配质量",
        `${statusStats.exact_count || 0} 精确 · ${statusStats.inferred_count || 0} 推断`,
        `低可信度 ${statusStats.low_confidence_count || 0} 项`,
        statusStats.low_confidence_count ? "neg" : "",
      ),
    );
    return cards;
  }

  const totalDur = parseChartNumber(table?.sums?.avg_dur_ms ?? rows.reduce((sum, row) => sum + parseChartNumber(row.raw.avg_dur_ms), 0));
  const totalCount = parseChartNumber(table?.sums?.avg_count ?? rows.reduce((sum, row) => sum + parseChartNumber(row.raw.avg_count), 0));
  const hotspot = highlightRow("hotspot") || rows
    .map(row => ({ ...row, hotValue: parseChartNumber(row.raw.avg_dur_ms) }))
    .sort((a, b) => b.hotValue - a.hotValue)[0] || null;
  const hotspotDuration = hotspot
    ? parseChartNumber(hotspot.raw?.avg_dur_ms ?? hotspot.hotValue ?? hotspot.value)
    : 0;
  const topPct = totalDur ? hotspotDuration / totalDur * 100 : 0;
  return [
    makeCard("设备计算耗时", fmtChartValue(totalDur, { unit: "ms" }), `已排除通信 · ${totalLabel}`),
    makeCard("最大热点", hotspot ? fmtChartValue(hotspotDuration, { unit: "ms" }) : "0", hotspot ? shortChartLabel(hotspot.label, 28) : "无", "", hotspot),
    makeCard("Kernel 调用数", fmtChartValue(totalCount), table?.aggregated ? "全量数据合计" : "当前已载入数据合计"),
    makeCard("最大热点占比", `${trimNumber(topPct)}%`, hotspot ? `占设备计算耗时 · ${shortChartLabel(hotspot.label, 28)}` : "无"),
  ];
};

const updateDeltaLists = (rows, table, sourceConfig) => {
  if (selectedJob.value?.mode !== "compare") {
    chartSlowdowns.value = [];
    chartSpeedups.value = [];
    return;
  }
  const fields = table?.fields || [];
  const metricDef = chartMetricDefFor("delta_dur_ms");
  const normalizeDirectionRows = key => applyChartCommunicationFilter(
    normalizeChartRows(table?.direction_rows?.[key] || [], fields, sourceConfig, metricDef),
    sourceConfig,
  );
  const slowdowns = normalizeDirectionRows("slowdowns");
  const speedups = normalizeDirectionRows("speedups");
  const fallbackSlowdowns = rows.filter(row => row.delta > 0).sort((a, b) => b.delta - a.delta);
  const fallbackSpeedups = rows.filter(row => row.delta < 0).sort((a, b) => a.delta - b.delta);
  const slowdownRows = slowdowns.length ? slowdowns : fallbackSlowdowns;
  const speedupRows = speedups.length ? speedups : fallbackSpeedups;
  const slowdownTotal = parseChartNumber(table?.direction_stats?.slowdown_total)
    || slowdownRows.reduce((sum, row) => sum + row.delta, 0);
  const speedupTotal = parseChartNumber(table?.direction_stats?.speedup_total)
    || speedupRows.reduce((sum, row) => sum + Math.abs(row.delta), 0);
  let slowdownCumulative = 0;
  chartSlowdowns.value = slowdownRows.slice(0, 8).map(row => {
    const contributionPct = slowdownTotal ? row.delta / slowdownTotal * 100 : 0;
    slowdownCumulative += contributionPct;
    return { ...row, contributionPct, cumulativePct: slowdownCumulative };
  });
  chartSpeedups.value = speedupRows.slice(0, 8).map(row => ({
    ...row,
    contributionPct: speedupTotal ? Math.abs(row.delta) / speedupTotal * 100 : 0,
  }));
};

const updateComparisonStatusLists = (table, sourceConfig) => {
  if (selectedJob.value?.mode !== "compare") {
    chartAddedRows.value = [];
    chartRemovedRows.value = [];
    return;
  }
  const fields = table?.fields || [];
  const metricDef = chartMetricDefFor("delta_dur_ms");
  const normalizeStatusRows = key => applyChartCommunicationFilter(
    normalizeChartRows(table?.status_rows?.[key] || [], fields, sourceConfig, metricDef),
    sourceConfig,
  );
  chartAddedRows.value = normalizeStatusRows("added");
  chartRemovedRows.value = normalizeStatusRows("removed");
};

const destroyChartInstances = () => {
  chartCanvasReady.value = false;
  if (ktChartInst.value)     { ktChartInst.value.destroy();     ktChartInst.value = null; }
  if (callTimeScatterInst.value) { callTimeScatterInst.value.destroy(); callTimeScatterInst.value = null; }
};

let chartHighlightSyncing = false;
const syncChartLinkedHighlight = label => {
  if (chartHighlightSyncing) return;
  const nextLabel = label || "";
  chartHighlightSyncing = true;
  try {
    for (const chart of [ktChartInst.value, callTimeScatterInst.value]) {
      if (!chart || chart.$linkedLabel === nextLabel) continue;
      chart.$linkedLabel = nextLabel;
      chart.update("none");
    }
  } finally {
    chartHighlightSyncing = false;
  }
};

const waitChartFrame = () => new Promise(resolve => {
  let settled = false;
  const finish = () => {
    if (settled) return;
    settled = true;
    resolve();
  };
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(finish);
  setTimeout(finish, 32);
});

let chartBuildToken = 0;
const scheduleBuildChart = async () => {
  const token = ++chartBuildToken;
  await nextTick();
  await waitChartFrame();
  await waitChartFrame();
  if (token !== chartBuildToken) return;
  if (resultTab.value === "chart" && selectedJob.value?.status === "done") {
    await buildChart();
  }
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
  try {
    const filterCacheKey = chartFilterCacheKey();
    const chartCacheKey = `${sourceConfig.file}:${metricDef.key}:${filterCacheKey}`;
    if (!chartTables.value[chartCacheKey]) {
      try {
        const params = new URLSearchParams({
          chart_metric: metricDef.key,
          chart_limit: String(CHART_FETCH_TOP_LIMIT),
          ...chartFilterParams(),
        });
        chartTables.value = {
          ...chartTables.value,
          [chartCacheKey]: await fetchJson(
            `/api/jobs/${encodeURIComponent(jobId)}/results/${encodeURIComponent(sourceConfig.file)}?${params}`,
            { credentials: "include" },
            "加载性能总览失败",
          ),
        };
        if (selectedJobId.value !== jobId) {
          chartLoading.value = false;
          return;
        }
      } catch (e) {
        chartError.value = normalizeApiError(e, "加载图表数据失败");
        chartLoading.value = false;
        return;
      }
    }
    const table = chartTables.value[chartCacheKey];
    chartFilterResultText.value = chartScatterMode.value === "compare"
      ? `筛选后 ${table?.filtered_total ?? table?.total ?? 0} / ${table?.total ?? 0} 项`
      : "";
    const fields = table?.fields || selectedJob.value.result_files[sourceConfig.file]?.fields || [];
    const rows = applyChartCommunicationFilter(
      normalizeChartRows(table?.rows || [], fields, sourceConfig, metricDef),
      sourceConfig
    );
    let scatterTable = null;
    if (chartScatterAvailable.value) {
      const scatterMetric = chartScatterMode.value === "compare" ? "delta_dur_ms" : "avg_dur_ms";
      const scatterCacheKey = `${sourceConfig.file}:${scatterMetric}:${filterCacheKey}`;
      if (!chartTables.value[scatterCacheKey]) {
        const scatterParams = new URLSearchParams({
          chart_metric: scatterMetric,
          chart_limit: String(CHART_FETCH_TOP_LIMIT),
          ...chartFilterParams(),
        });
        chartTables.value = {
          ...chartTables.value,
          [scatterCacheKey]: await fetchJson(
            `/api/jobs/${encodeURIComponent(jobId)}/results/${encodeURIComponent(sourceConfig.file)}?${scatterParams}`,
            { credentials: "include" },
            "加载调用耗时散点图失败",
          ),
        };
      }
      scatterTable = chartTables.value[scatterCacheKey];
    }
    destroyChartInstances();

    chartSummaryCards.value = buildChartSummary(rows, table, sourceConfig);
    updateDeltaLists(rows, table, sourceConfig);
    updateComparisonStatusLists(table, sourceConfig);
    const topN = Math.max(1, Number(chartTopN.value) || 10);
    const dumbbellRows = rows.filter(row => row.aValue > 0 && row.bValue > 0);
    chartDumbbellActive.value = selectedJob.value?.mode === "compare"
      && metricDef.key === "delta_dur_ms"
      && dumbbellRows.length > 0;
    chartBarRows.value = buildChartBarRows(
      chartDumbbellActive.value ? dumbbellRows : rows,
      metricDef,
      topN,
      table?.metric_total,
    );
    if (scatterTable) {
      const scatterMetric = chartScatterMode.value === "compare" ? "delta_dur_ms" : "avg_dur_ms";
      const scatterSourceRows = applyChartCommunicationFilter(
        normalizeChartRows(
          scatterTable.rows || [],
          scatterTable.fields || fields,
          sourceConfig,
          chartMetricDefFor(scatterMetric),
        ),
        sourceConfig,
      );
      chartScatterRows.value = chartScatterMode.value === "compare"
        ? buildCompareScatterRows(scatterSourceRows)
        : buildCallTimeScatterRows(scatterSourceRows);
    } else {
      chartScatterRows.value = [];
    }

    if (!chartBarRows.value.length) {
      chartError.value = "当前指标没有可绘制的数据";
      chartLoading.value = false;
      return;
    }

    ktChart.value.parentElement.style.height =
      Math.max(280, Math.min(620, chartBarRows.value.length * 34 + 96)) + 'px';
    await waitChartFrame();
    if (selectedJobId.value !== jobId || resultTab.value !== "chart") {
      chartLoading.value = false;
      return;
    }
    const chartRect = ktChart.value.parentElement.getBoundingClientRect();
    if (chartRect.width <= 0 || chartRect.height <= 0) {
      setTimeout(() => {
        if (selectedJobId.value === jobId && resultTab.value === "chart") scheduleBuildChart();
      }, 50);
      return;
    }
    if (typeof Chart === "undefined") {
      chartCanvasReady.value = false;
      return;
    }

    const cc = chartColors();
    let chart;
    if (chartDumbbellActive.value) {
      const pointFill = isDark.value ? "rgba(15,23,42,.96)" : "rgba(255,255,255,.98)";
      const regressionColor = "rgba(239,68,68,.82)";
      const improvementColor = "rgba(34,197,94,.82)";
      const linkedRadius = (context, base) => {
        const row = chartBarRows.value[context.raw?.rowIndex];
        return base + (row && context.chart.$linkedLabel === row.label ? 2 : 0);
      };
      chart = new Chart(ktChart.value, {
        type: "scatter",
        plugins: [chartCompareDumbbellPlugin],
        data: {
          labels: chartBarRows.value.map(row => row.label),
          datasets: [
            {
              label: "A",
              data: chartBarRows.value.map((row, index) => ({ x: row.aValue, y: row.label, rowIndex: index })),
              backgroundColor: pointFill,
              borderColor: cc.text,
              borderWidth: 2,
              pointRadius: context => linkedRadius(context, 5),
              pointHoverRadius: 7,
            },
            {
              label: "B",
              data: chartBarRows.value.map((row, index) => ({ x: row.bValue, y: row.label, rowIndex: index })),
              backgroundColor: context => {
                const row = chartBarRows.value[context.raw?.rowIndex];
                if (!row || row.inferredMatch || ["inferred", "low_confidence"].includes(row.compareStatus)) return pointFill;
                return row.delta >= 0 ? regressionColor : improvementColor;
              },
              borderColor: context => {
                const row = chartBarRows.value[context.raw?.rowIndex];
                return row?.delta >= 0 ? regressionColor : improvementColor;
              },
              borderWidth: context => {
                const row = chartBarRows.value[context.raw?.rowIndex];
                return row && ["inferred", "low_confidence"].includes(row.compareStatus) ? 2.5 : 1.5;
              },
              pointRadius: context => linkedRadius(context, 6),
              pointHoverRadius: 8,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 280 },
          layout: { padding: { top: 4, right: 108 } },
          interaction: { mode: "nearest", intersect: true },
          onClick: (_, elements) => {
            const rowIndex = elements?.[0]?.index;
            const row = chartBarRows.value[rowIndex];
            if (row) drillDownChart(row);
          },
          onHover: (event, elements) => {
            const rowIndex = elements?.[0]?.index;
            const row = chartBarRows.value[rowIndex];
            syncChartLinkedHighlight(row?.label || "");
            if (event?.native?.target) event.native.target.style.cursor = row ? "pointer" : "default";
          },
          plugins: {
            legend: { display: false },
            chartCompareDumbbell: {
              rows: chartBarRows.value,
              regressionColor,
              improvementColor,
              labelColor: cc.text,
            },
            tooltip: {
              displayColors: false,
              callbacks: {
                title: items => {
                  const rowIndex = items[0]?.raw?.rowIndex;
                  return chartBarRows.value[rowIndex]?.label || "";
                },
                label: context => {
                  const row = chartBarRows.value[context.raw?.rowIndex];
                  if (!row) return "";
                  const signedPct = row.aValue ? row.delta / row.aValue * 100 : null;
                  return [
                    ` A → B: ${trimNumber(row.aValue)} → ${trimNumber(row.bValue)} ms`,
                    ` B - A: ${row.delta > 0 ? "+" : ""}${trimNumber(row.delta)} ms${signedPct === null ? "" : `（${signedPct > 0 ? "+" : ""}${trimNumber(signedPct)}%）`}`,
                    ` 匹配: ${comparisonStatusLabel(row.compareStatus)}`,
                  ];
                },
              },
            },
          },
          scales: {
            x: {
              type: "logarithmic",
              title: { display: true, text: "A / B 平均耗时（ms，对数）", color: cc.text },
              ticks: { color: cc.text, callback: logScaleTickLabel, maxRotation: 0, minRotation: 0 },
              grid: { color: context => logScaleTickLabel(context.tick?.value) ? cc.grid : "transparent" },
              border: { color: cc.grid },
            },
            y: {
              type: "category",
              offset: true,
              ticks: {
                color: cc.text,
                callback: (_value, index) => chartBarRows.value[index]?.shortLabel || "",
              },
              grid: { display: false },
              border: { display: false },
            },
          },
        },
      });
    } else {
      const barColors = metricDef.signed
        ? chartBarRows.value.map(row => row.value >= 0 ? "rgba(239,68,68,0.76)" : "rgba(34,197,94,0.76)")
        : getColors(chartBarRows.value.length);
      chart = new Chart(ktChart.value, {
        type: "bar",
        plugins: [chartValueLabelsPlugin, chartShareLabelsPlugin],
        data: {
          labels: chartBarRows.value.map(row => row.shortLabel),
          datasets: [{
            label: metricDef.label,
            data: chartBarRows.value.map(row => row.value),
            backgroundColor: barColors,
            borderColor: cc.title,
            borderWidth: context => context.chart.$linkedLabel === chartBarRows.value[context.dataIndex]?.label ? 2 : 0,
            borderRadius: 4,
            barThickness: 18,
          }],
        },
        options: {
          indexAxis: 'y',
          responsive: true, maintainAspectRatio: false,
          layout: { padding: { right: metricDef.signed ? 154 : 132 } },
          onClick: (_, elements) => {
            const row = chartBarRows.value[elements?.[0]?.index];
            if (row) drillDownChart(row);
          },
          onHover: (event, elements) => {
            const row = chartBarRows.value[elements?.[0]?.index];
            syncChartLinkedHighlight(row?.label || "");
            if (event?.native?.target) event.native.target.style.cursor = row ? "pointer" : "default";
          },
          plugins: {
            legend: { display: false },
            title: { display: true, text: `${sourceConfig.label} · ${metricDef.label}`, font: { size: 13 }, color: cc.title },
            chartValueLabels: {
              rows: chartBarRows.value,
              metricDef,
              color: cc.title,
              shareColumnWidth: metricDef.signed ? 78 : 56,
            },
            chartShareLabels: {
              rows: chartBarRows.value,
              header: chartShareHeader(metricDef),
              color: cc.text,
              headerColor: cc.title,
            },
            tooltip: { callbacks: {
              title: items => chartBarRows.value[items[0]?.dataIndex]?.label || "",
              label: ctx => ` ${metricDef.label}: ${fmtChartValue(ctx.parsed.x, metricDef)}`,
              afterLabel: ctx => {
                const row = chartBarRows.value[ctx.dataIndex];
                if (!row) return "";
                const lines = [`${chartShareHeader(metricDef)}: ${row.shareLabel}`];
                if (selectedJob.value?.mode === "compare") {
                  lines.push(
                    `A: ${fmtChartValue(row.aValue, { unit: "ms" })}`,
                    `B: ${fmtChartValue(row.bValue, { unit: "ms" })}`,
                    `count: ${trimNumber(row.countA)} -> ${trimNumber(row.countB)}`
                  );
                }
                return lines;
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
    }
    ktChartInst.value = chart;
    chartCanvasReady.value = true;

    if (callTimeScatter.value && chartScatterRows.value.length) {
      const isCompareScatter = chartScatterMode.value === "compare";
      const scatterPointFill = isDark.value ? "rgba(15,23,42,.96)" : "rgba(255,255,255,.98)";
      const scatter = new Chart(callTimeScatter.value, {
        type: "scatter",
        plugins: [chartScatterGuidesPlugin, chartScatterHotspotLabelsPlugin],
        data: {
          datasets: [{
            label: "Kernel / Op",
            data: chartScatterRows.value.map(row => isCompareScatter
              ? {
                  x: row.baselineMs,
                  y: row.delta,
                  radius: row.radius,
                  label: row.label,
                  regression: row.delta > 0,
                  inferredMatch: row.inferredMatch,
                }
              : { x: row.calls, y: row.usPerCall, radius: row.radius, label: row.label }),
            backgroundColor: context => isCompareScatter
              ? (context.raw?.inferredMatch
                  ? scatterPointFill
                  : context.raw?.regression ? "rgba(239,68,68,0.68)" : "rgba(34,197,94,0.68)")
              : `rgba(99,102,241,${Math.min(.82, .42 + (context.raw?.radius || 4) / 30)})`,
            borderColor: context => isCompareScatter
              ? (context.raw?.regression ? "rgba(220,38,38,0.92)" : "rgba(22,163,74,0.92)")
              : "rgba(79,70,229,0.92)",
            borderWidth: context => {
              if (context.chart.$linkedLabel === context.raw?.label) return 3;
              return context.raw?.inferredMatch ? 2.5 : 1.5;
            },
            pointStyle: context => isCompareScatter && !context.raw?.regression ? "rectRot" : "circle",
            pointRadius: context => (context.raw?.radius || 4)
              + (context.chart.$linkedLabel === context.raw?.label ? 2 : 0),
            pointHoverRadius: context => (context.raw?.radius || 4) + 3,
            pointHoverBorderWidth: 2.5,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 280 },
          layout: { padding: { top: 8, right: 8, bottom: 2, left: 4 } },
          interaction: { mode: "nearest", intersect: true },
          onClick: (_, elements) => {
            const row = chartScatterRows.value[elements?.[0]?.index];
            if (row) drillDownChart(row);
          },
          onHover: (event, elements) => {
            const row = chartScatterRows.value[elements?.[0]?.index];
            syncChartLinkedHighlight(row?.label || "");
            if (event?.native?.target) event.native.target.style.cursor = row ? "pointer" : "default";
          },
          plugins: {
            legend: { display: false },
            chartScatterGuides: {
              compare: isCompareScatter,
              equalityColor: isDark.value ? "rgba(148,163,184,.68)" : "rgba(100,116,139,.62)",
              relativeGuideRate: .2,
              relativeGuideColor: isDark.value ? "rgba(148,163,184,.48)" : "rgba(100,116,139,.46)",
              relativeGuideText: cc.text,
            },
            chartScatterHotspotLabels: {
              rows: chartScatterRows.value,
              maxLabels: 4,
              topInset: isCompareScatter ? 20 : 0,
              bottomInset: isCompareScatter ? 18 : 0,
              color: cc.title,
              background: isDark.value ? "rgba(15,23,42,.9)" : "rgba(255,255,255,.9)",
              border: isDark.value ? "rgba(148,163,184,.34)" : "rgba(100,116,139,.26)",
            },
            tooltip: {
              backgroundColor: isDark.value ? "rgba(15,23,42,.96)" : "rgba(255,255,255,.98)",
              titleColor: cc.title,
              bodyColor: cc.text,
              borderColor: isDark.value ? "rgba(148,163,184,.35)" : "rgba(100,116,139,.24)",
              borderWidth: 1,
              padding: 10,
              displayColors: false,
              callbacks: {
              title: items => chartScatterRows.value[items[0]?.dataIndex]?.label || "",
              label: ctx => {
                const row = chartScatterRows.value[ctx.dataIndex];
                if (!row) return "";
                if (isCompareScatter) {
                  const direction = row.delta > 0 ? "变慢" : "变快";
                  const signedPct = `${row.changePct > 0 ? "+" : ""}${trimNumber(row.changePct)}%`;
                  return [
                    ` A → B: ${trimNumber(row.baselineMs)} → ${trimNumber(row.currentMs)} ms`,
                    ` B 比 A ${direction}: ${trimNumber(Math.abs(row.delta))} ms（${signedPct}）`,
                    ` 匹配: ${comparisonStatusLabel(row.compareStatus)}`,
                  ];
                }
                return [
                  ` 调用数: ${trimNumber(row.calls)}`,
                  ` 单次耗时: ${trimNumber(row.usPerCall)} us`,
                  ` 总耗时: ${trimNumber(row.totalMs)} ms`,
                ];
              },
              },
            },
          },
          scales: {
            x: {
              type: "logarithmic",
              title: {
                display: true,
                text: isCompareScatter ? "A 平均耗时（ms，从短到长）" : "平均调用数（对数）",
                color: cc.text,
              },
              ticks: isCompareScatter
                ? { color: cc.text, callback: logScaleTickLabel, maxRotation: 0, minRotation: 0 }
                : { color: cc.text },
              grid: {
                color: context => !isCompareScatter || logScaleTickLabel(context.tick?.value)
                  ? cc.grid
                  : "transparent",
              },
              border: { color: cc.grid },
            },
            y: {
              type: isCompareScatter ? "linear" : "logarithmic",
              beginAtZero: isCompareScatter,
              grace: isCompareScatter ? "10%" : 0,
              title: {
                display: true,
                text: isCompareScatter ? "B 比 A 多 / 少的耗时（ms）" : "平均单次耗时（μs，对数）",
                color: cc.text,
              },
              ticks: isCompareScatter
                ? { color: cc.text, callback: value => `${Number(value) > 0 ? "+" : ""}${trimNumber(Number(value))}` }
                : { color: cc.text },
              grid: {
                color: context => isCompareScatter && Number(context.tick?.value) === 0
                  ? (isDark.value ? "rgba(148,163,184,.42)" : "rgba(100,116,139,.34)")
                  : cc.grid,
              },
              border: { color: cc.grid },
            },
          },
        },
      });
      callTimeScatterInst.value = scatter;
    }
  } catch (e) {
    chartCanvasReady.value = false;
    chartError.value = normalizeApiError(e, "生成图表失败");
  } finally {
    if (selectedJobId.value === jobId && resultTab.value === "chart") {
      chartLoading.value = false;
    }
  }
};

// ══════════════════════════════════════════════════════════════════════════════
// Uploads
// ══════════════════════════════════════════════════════════════════════════════

const setUploadFiles = files => {
  const picked = Array.from(files || []);
  if (!picked.length) return;
  const nextFiles = quickUploadMode.value === "multi" ? picked : picked.slice(0, 1);
  if (quickUploadMode.value === "single" && picked.length > 1) {
    showToast("单个模式只会使用第一个文件；如需逐个分析多个 trace，请切到“多个”。", "info");
  }
  uploadQueue.value = nextFiles.map((file, index) => ({
    id: `${Date.now()}-${index}-${file.name}`,
    file,
    name: file.name,
    meta: uploadFileMeta(file),
    status: "ready",
    progress: 0,
    error: "",
    jobId: "",
  }));
  fileA.value = nextFiles[0];
  fileAName.value = nextFiles.length === 1 ? nextFiles[0].name : `${nextFiles.length} 个文件`;
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
  if (!["single", "compare", "multi"].includes(mode) || submitting.value) return;
  quickUploadMode.value = mode;
  localStorage.setItem("tpa-upload-mode", mode);
  if (mode === "single" && uploadQueue.value.length > 1) {
    uploadQueue.value = uploadQueue.value.slice(0, 1);
    fileA.value = uploadQueue.value[0]?.file || null;
    fileAName.value = uploadQueue.value[0]?.name || "";
  }
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
    if (lastJob) router.push({ path: jobRoutePath(lastJob) });
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
      showToast("两文件对比提交失败: " + detail, "error");
      resolve(null);
      return;
    }
    const job = JSON.parse(xhr.responseText);
    quickCompareStatus.value = "submitted";
    form.value.label = "";
    clearQuickCompareFiles();
    await refreshSidebarData();
    sidebarTab.value = "jobs";
    router.push({ path: jobRoutePath(job) });
    showToast("已提交两文件对比任务", "success");
    resolve(job);
  };
  xhr.onerror = () => {
    submitting.value = false;
    uploadProgress.value = 0;
    quickCompareStatus.value = "error";
    showToast("两文件对比提交失败: 网络错误", "error");
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
  const jobId = selectedJobId.value;
  const jobLabel = selectedJob.value?.label || selectedJob.value?.file_a_name || jobId;
  if (!jobId) {
    showToast("未选中任务，无法删除", "error");
    return;
  }
  if (!await askConfirm(`确定删除任务「${jobLabel}」及所有关联文件？`, {
    title: "删除任务",
    confirmText: "删除",
    tone: "danger",
  })) return;
  try {
    const response = await fetch(`/api/jobs/${jobId}`, {
      method: "DELETE", credentials: "include",
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      showToast("删除失败: " + (errorData.detail || errorData.message || "未知错误"), "error");
      return;
    }
    forgetRecentViewedJob(jobId);
    if (selectedJobId.value === jobId) router.push({ path: "/" });
    await refreshSidebarData();
    showToast("任务已删除", "success");
  } catch (error) {
    showToast("删除出错: " + error.message, "error");
  }
};

const deleteFile = async slot => {
  const jobId = selectedJobId.value;
  const jobLabel = selectedJob.value?.label || selectedJob.value?.file_a_name || jobId;
  if (!jobId) {
    showToast("未选中任务，无法删除文件", "error");
    return;
  }
  let impact = { count: 0, dependent_compare_jobs: [] };
  try {
    const impactResp = await fetch(`/api/jobs/${jobId}/files/${slot}/delete-impact`, {
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
  if (!await askConfirm(`确定删除任务「${jobLabel}」的 ${slot.toUpperCase()} 原始 trace 文件？删除后该文件无法参与历史对比。${impactMessage}`, {
    title: "删除文件",
    confirmText: "删除",
    tone: "danger",
  })) return;
  const r = await fetch(`/api/jobs/${jobId}/files/${slot}?force=true`, {
    method: "DELETE", credentials: "include",
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("删除文件失败: " + (err.detail || err.message || "未知错误"), "error");
    return;
  }
  if (selectedJobId.value === jobId) await loadJob(jobId);
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

const resolveProjectMeta = projectOrGroup => {
  if (!projectOrGroup?.id || projectOrGroup.id === "__none__") return projectOrGroup || null;
  return projects.value.find(project => project.id === projectOrGroup.id)
    || historyProjectGroups.value.find(group => group.id === projectOrGroup.id)
    || projectOrGroup;
};

const selectHistoryProjectView = viewId => {
  const nextView = ["all", "favorite", "mine", "shared"].includes(viewId) ? viewId : "all";
  historyProjectView.value = nextView;
  filterProject.value = "";
};

const selectHistoryProject = projectId => {
  historyProjectView.value = "all";
  filterProject.value = projectId || "";
};

const startProjectBulkMode = projectOrGroup => {
  const project = resolveProjectMeta(projectOrGroup);
  if (!project?.id || project.id === "__none__") return;
  historySelection.value = [];
  historyBulkMode.value = true;
  collapsedGroups.value = { ...collapsedGroups.value, [project.id]: true };
  selectHistoryProject(project.id);
};

const resetProjectBulkSelection = () => {
  historySelection.value = [];
  projectBulkSelectionDetails.value = {};
};

const closeProjectBulkModal = () => {
  if (projectBulkActionLoading.value) return;
  showProjectBulkModal.value = false;
  projectBulkSearch.value = "";
  projectBulkJobs.value = [];
  projectBulkJobsTotal.value = 0;
  projectBulkJobsOffset.value = 0;
  resetProjectBulkSelection();
  if (projectBulkJobsController) {
    projectBulkJobsController.abort();
    projectBulkJobsController = null;
  }
};

const loadProjectBulkJobs = async (reset = false) => {
  if (!showProjectBulkModal.value || !projectBulkId.value) return;
  if (projectBulkJobsController) projectBulkJobsController.abort();
  const controller = new AbortController();
  projectBulkJobsController = controller;
  projectBulkJobsLoading.value = true;
  const offset = reset ? 0 : projectBulkJobsOffset.value;
  const params = new URLSearchParams();
  const q = projectBulkSearch.value.trim();
  if (q) params.set("q", q);
  params.set("limit", String(projectBulkJobsLimit.value));
  params.set("offset", String(offset));
  try {
    const r = await fetch(`/api/job-groups/${encodeURIComponent(projectBulkId.value)}/jobs?${params}`, {
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
    if (projectBulkJobsController !== controller) return;
    const nextJobs = reset ? (data.data || []) : [...projectBulkJobs.value, ...(data.data || [])];
    projectBulkJobs.value = nextJobs;
    projectBulkJobsTotal.value = data.total || 0;
    projectBulkJobsOffset.value = nextJobs.length;

    const details = { ...projectBulkSelectionDetails.value };
    for (const job of nextJobs) {
      if (historySelection.value.includes(job.id)) details[job.id] = job;
    }
    projectBulkSelectionDetails.value = details;
  } catch (e) {
    if (e.name !== "AbortError") showToast(normalizeApiError(e, "加载项目任务失败"), "error");
  } finally {
    if (projectBulkJobsController === controller) {
      projectBulkJobsLoading.value = false;
      projectBulkJobsController = null;
    }
  }
};

const openProjectBulkModal = async projectOrGroup => {
  const project = resolveProjectMeta(projectOrGroup);
  if (!project?.id || project.id === "__none__") return;
  historyBulkMode.value = false;
  resetProjectBulkSelection();
  projectBulkId.value = project.id;
  projectBulkName.value = project.name || project.label || "当前项目";
  projectBulkSearch.value = "";
  projectBulkJobs.value = [];
  projectBulkJobsTotal.value = 0;
  projectBulkJobsOffset.value = 0;
  showProjectBulkModal.value = true;
  await loadProjectBulkJobs(true);
};

const toggleProjectFavorite = async (projectOrGroup, event) => {
  event?.preventDefault?.();
  event?.stopPropagation?.();
  const project = resolveProjectMeta(projectOrGroup);
  if (!project?.id || project.id === "__none__" || projectMutationLoading.value) return;
  const nextFavorite = project.is_favorite ? 0 : 1;
  projectMutationLoading.value = `favorite:${project.id}`;
  try {
    const r = await fetch(`/api/projects/${encodeURIComponent(project.id)}/favorite`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ is_favorite: Boolean(nextFavorite) }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    projects.value = projects.value.map(item => item.id === data.id ? { ...item, ...data } : item);
    historyGroups.value = historyGroups.value.map(group =>
      group.id === data.id ? { ...group, ...data, label: data.name || group.label } : group
    );
    if (historyProjectView.value === "favorite" && !nextFavorite && !filterProject.value) {
      await refreshSidebarData();
    }
  } catch (e) {
    showToast("更新收藏失败: " + (e.message || "未知错误"), "error");
  } finally {
    projectMutationLoading.value = "";
  }
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

const toggleProjectBulkSelection = job => {
  if (job.is_owner === false) {
    showToast("只能批量操作自己创建的任务", "error");
    return;
  }
  const selected = new Set(historySelection.value);
  const details = { ...projectBulkSelectionDetails.value };
  if (selected.has(job.id)) {
    selected.delete(job.id);
    delete details[job.id];
  } else {
    selected.add(job.id);
    details[job.id] = job;
  }
  historySelection.value = [...selected];
  projectBulkSelectionDetails.value = details;
};

const toggleLoadedProjectBulkJobs = () => {
  const ids = projectBulkLoadedOwnerIds.value;
  if (!ids.length) return;
  const selected = new Set(historySelection.value);
  const details = { ...projectBulkSelectionDetails.value };
  if (projectBulkLoadedAllSelected.value) {
    for (const id of ids) {
      selected.delete(id);
      delete details[id];
    }
  } else {
    for (const job of projectBulkJobs.value) {
      if (job.is_owner === false) continue;
      selected.add(job.id);
      details[job.id] = job;
    }
  }
  historySelection.value = [...selected];
  projectBulkSelectionDetails.value = details;
};

const clearProjectBulkSelection = () => resetProjectBulkSelection();

const handleHistoryJobClick = job => {
  if (historyBulkMode.value) {
    toggleHistorySelection(job);
    return;
  }
  navigateToJob(job);
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
  if (!historySelection.value.length || projectBulkActionLoading.value) return;
  projectBulkActionLoading.value = "move";
  try {
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
    projectBulkSelectionDetails.value = {};
    await refreshSidebarData();
    if (selectedJobId.value && ids.includes(selectedJobId.value)) await loadJob(selectedJobId.value);
    showToast(`已移动 ${moved} 个任务`, "success");
  } catch (e) {
    showToast("批量移动失败: " + (e.message || "网络错误"), "error");
  } finally {
    projectBulkActionLoading.value = "";
  }
};

const bulkDeleteJobs = async () => {
  if (!historySelection.value.length || projectBulkActionLoading.value) return;
  projectBulkActionLoading.value = "delete-jobs";
  try {
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
    projectBulkSelectionDetails.value = {};
    await refreshSidebarData();
    showToast(`已删除 ${ids.length} 个任务`, "success");
  } catch (e) {
    showToast("批量删除失败: " + (e.message || "网络错误"), "error");
  } finally {
    projectBulkActionLoading.value = "";
  }
};

const bulkDeleteFiles = async () => {
  if (!historySelection.value.length || projectBulkActionLoading.value) return;
  projectBulkActionLoading.value = "delete-files";
  try {
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
      body: JSON.stringify({ job_ids: ids, force: true }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      showToast("批量删除文件失败: " + (err.detail || err.message || "未知错误"), "error");
      return;
    }
    await refreshSidebarData();
    if (selectedJobId.value && ids.includes(selectedJobId.value)) await loadJob(selectedJobId.value);
    showToast(`已处理 ${ids.length} 个任务`, "success");
  } catch (e) {
    showToast("批量删除文件失败: " + (e.message || "网络错误"), "error");
  } finally {
    projectBulkActionLoading.value = "";
  }
};

const openRenameModal = (projectOrGroup) => {
  const project = resolveProjectMeta(projectOrGroup);
  if (!project?.id) return;
  renameProjectId.value = project.id;
  renameProjectName.value = project.name || project.label || "";
  showRenameProject.value = true;
};

const confirmRenameProject = async () => {
  if (renameProjectLoading.value) return;
  const newName = String(renameProjectName.value || "").trim();
  if (!newName) return;
  const pid = renameProjectId.value;
  if (!pid) { showToast("项目ID无效", "error"); return; }
  renameProjectLoading.value = true;
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
    showRenameProject.value = false;
    await loadProjects();
    await refreshSidebarData();
    showToast("项目已重命名", "success");
  } catch (e) {
    showToast("重命名失败: " + (e.message || "网络错误"), "error");
  } finally {
    renameProjectLoading.value = false;
  }
};

const deleteProject = async (projectId) => {
  if (projectMutationLoading.value) return;
  if (!await askConfirm("确定删除该项目？项目内的任务将同时被删除。删除后10天内可以找回。", {
    title: "删除项目",
    confirmText: "删除",
    tone: "danger",
  })) return;
  projectMutationLoading.value = `delete:${projectId}`;
  try {
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
  } catch (e) {
    showToast("删除失败: " + (e.message || "网络错误"), "error");
  } finally {
    projectMutationLoading.value = "";
  }
};

const shareProject = async (project) => {
  if (!project?.id || project.is_public || projectMutationLoading.value) return;
  if (!await askConfirm("确定将该项目转为共享项目？", {
    title: "转为共享项目",
    confirmText: "转为共享",
  })) return;
  projectMutationLoading.value = `share:${project.id}`;
  try {
    const r = await fetch(`/api/projects/${project.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ is_public: true }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      showToast("转为共享失败: " + (err.detail || err.message || `HTTP ${r.status}`), "error");
      return;
    }
    await loadProjects();
    await refreshSidebarData();
    showToast("项目已转为共享", "success");
  } catch (e) {
    showToast("转为共享失败: " + (e.message || "网络错误"), "error");
  } finally {
    projectMutationLoading.value = "";
  }
};

const unshareProject = async (project) => {
  if (!project?.id || !project.is_public || projectMutationLoading.value) return;
  if (!await askConfirm("确定将该共享项目转为个人项目？其他用户将不再看到该项目。", {
    title: "转为个人项目",
    confirmText: "转为个人",
  })) return;
  projectMutationLoading.value = `unshare:${project.id}`;
  try {
    const r = await fetch(`/api/projects/${project.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ is_public: false }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      showToast("转为个人失败: " + (err.detail || err.message || `HTTP ${r.status}`), "error");
      return;
    }
    await loadProjects();
    await refreshSidebarData();
    showToast("项目已转为个人", "success");
  } catch (e) {
    showToast("转为个人失败: " + (e.message || "网络错误"), "error");
  } finally {
    projectMutationLoading.value = "";
  }
};

let tableResizeSortGuard = null;
let tableResizeSortGuardTimer = null;
const armTableResizeSortGuard = field => {
  if (tableResizeSortGuardTimer) clearTimeout(tableResizeSortGuardTimer);
  tableResizeSortGuard = { field, until: Date.now() + 450 };
  tableResizeSortGuardTimer = setTimeout(() => {
    if (tableResizeSortGuard && Date.now() >= tableResizeSortGuard.until) {
      tableResizeSortGuard = null;
    }
    tableResizeSortGuardTimer = null;
  }, 500);
};
const consumeTableResizeSortGuard = field => {
  if (!tableResizeSortGuard) return false;
  if (Date.now() > tableResizeSortGuard.until) {
    tableResizeSortGuard = null;
    return false;
  }
  const matched = tableResizeSortGuard.field === field;
  if (matched) {
    tableResizeSortGuard = null;
    if (tableResizeSortGuardTimer) {
      clearTimeout(tableResizeSortGuardTimer);
      tableResizeSortGuardTimer = null;
    }
  }
  return matched;
};

const setSort = (col, e) => {
  if (consumeTableResizeSortGuard(col)) {
    e?.preventDefault?.();
    e?.stopPropagation?.();
    return;
  }
  if (sortCol.value === col) sortAsc.value = !sortAsc.value;
  else { sortCol.value = col; sortAsc.value = true; }
};

let activeTableResizeCleanup = null;
const startResize = (field, e) => {
  e?.preventDefault?.();
  e?.stopPropagation?.();
  const source = e?.currentTarget || e?.target;
  const th = source?.closest?.("th");
  const startX = Number(e?.clientX);
  if (!th || !Number.isFinite(startX)) return;
  if (activeTableResizeCleanup) activeTableResizeCleanup();
  armTableResizeSortGuard(field);
  const startRect = th.getBoundingClientRect();
  // Resize relative to the column's width at grab time (delta-based) rather than
  // snapping the edge to the cursor. The handle is 8px wide, so an absolute
  // `clientX - left` mapping would jump the column by the grab offset on the
  // first mousemove. Delta keeps the edge tracking the pointer 1:1.
  const startWidth = clampTableColumnWidth(startRect.width) || startRect.width;
  const isRtl = getComputedStyle(th).direction === "rtl";
  const previousCursor = document.body.style.cursor;
  const previousUserSelect = document.body.style.userSelect;
  document.body.style.cursor = "col-resize";
  document.body.style.userSelect = "none";
  const resizeTo = clientX => {
    const delta = clientX - startX;
    const w = clampTableColumnWidth(startWidth + (isRtl ? -delta : delta));
    if (!w) return;
    colWidths.value = { ...colWidths.value, [field]: w };
  };
  const onMove = ev => {
    const clientX = Number(ev?.clientX);
    if (!Number.isFinite(clientX)) return;
    resizeTo(clientX);
  };
  const cleanup = () => {
    armTableResizeSortGuard(field);
    document.body.style.cursor = previousCursor;
    document.body.style.userSelect = previousUserSelect;
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", cleanup);
    window.removeEventListener("blur", cleanup);
    activeTableResizeCleanup = null;
  };
  activeTableResizeCleanup = cleanup;
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", cleanup);
  window.addEventListener("blur", cleanup);
};

const downloadCsv = filename => {
  const fields = currentTable.value.fields || [];
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

const downloadFilteredCsv = filename => {
  const jobId = selectedJobId.value;
  if (!jobId || !filename?.endsWith(".csv")) return;
  const params = buildResultTableParams({ limit: 1, offset: 0 });
  params.delete("limit");
  params.delete("offset");
  params.set("download", "true");
  const a = document.createElement("a");
  a.href = `/api/jobs/${encodeURIComponent(jobId)}/results/${encodeURIComponent(filename)}?${params}`;
  a.download = filename.replace(/\.csv$/i, "_filtered.csv");
  document.body.appendChild(a);
  a.click();
  a.remove();
  showToast(`已开始导出 ${fmtCount(tableTotalRows.value)} 行筛选结果`, "success");
};

const downloadKernelHostOpsCsv = () => {
  const jobId = selectedJobId.value;
  if (!jobId || !hasKernelHostOpRelations.value) return;
  const params = new URLSearchParams({ download: "true" });
  const a = document.createElement("a");
  a.href = `/api/jobs/${encodeURIComponent(jobId)}/results/kernel_host_ops.csv?${params}`;
  a.download = "kernel_host_ops.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  showToast("已开始导出完整 Host Op 关系 CSV", "success");
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
  const jobId = selectedJobId.value;
  if (!jobId) return;
  if (authRequired.value) {
    const projectId = selectedJob.value?.project_id || "";
    const project = projects.value.find(item => item.id === projectId);
    if (!projectId) {
      if (!await askConfirm("当前任务未分组。复制分享链接会自动创建共享项目并将该任务移入，其他用户将可访问该任务。是否继续？", {
        title: "确认共享任务",
        confirmText: "创建共享项目并复制",
      })) return;
    } else if (!project?.is_public) {
      const projectName = project?.name || selectedJob.value?.project_name || "当前项目";
      if (!await askConfirm(`复制分享链接会将私有项目「${projectName}」转为共享项目，项目内所有任务都将对其他用户可见。是否继续？`, {
        title: "确认公开整个项目",
        confirmText: "转为共享并复制",
      })) return;
    }
  }
  const r = await fetch(`/api/jobs/${jobId}/share`, {
    method: "POST",
    credentials: "include",
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast("生成分享链接失败: " + (err.detail || err.message || `HTTP ${r.status}`), "error");
    return;
  }
  const data = await r.json();
  const url = data.url || `${window.location.origin}${window.location.pathname}#${jobRoutePath(currentJobRouteHandle())}`;
  await copyTextToClipboard(url);
  if (data.changed) {
    await loadProjects();
    await refreshSidebarData();
    if (selectedJobId.value === jobId) await loadJob(jobId);
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

const safeMarkdownImageUrl = value => {
  const raw = String(value || "").trim();
  if (!raw) return "";
  if (/^https?:/i.test(raw) || raw.startsWith("/api/feedback/images/")) {
    return raw.replace(/"/g, "%22");
  }
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
    const custom = options.codeRenderer?.(code, options.inlineContext || {});
    return stash(custom || `<code>${escapeHtml(code)}</code>`);
  });
  value = value.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, url) => {
    const safeUrl = safeMarkdownImageUrl(url);
    const safeAlt = escapeHtml(alt || "image");
    if (!safeUrl) return safeAlt;
    const escapedUrl = escapeHtml(safeUrl);
    if (options.feedbackImages) {
      return stash(`<span class="md-image-scroll"><button type="button" class="md-image-link md-image-toggle" data-image-url="${escapedUrl}" data-image-alt="${safeAlt}" aria-label="点击预览图片" title="点击预览图片"><img src="${escapedUrl}" alt="${safeAlt}" class="md-image md-feedback-image" loading="lazy"></button></span>`);
    }
    return stash(`<span class="md-image-scroll"><a href="${escapedUrl}" target="_blank" rel="noopener noreferrer" class="md-image-link"><img src="${escapedUrl}" alt="${safeAlt}" class="md-image" loading="lazy"></a></span>`);
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

const splitMarkdownTableCellItems = cell => {
  const raw = String(cell ?? "").trim();
  if (!raw) return [];
  const normalized = raw
    .replace(/&lt;br\s*\/?&gt;/gi, "<br>")
    .replace(/<br\s*\/?>/gi, "\n");
  const parts = normalized
    .split(/\n+/)
    .map(part => part.trim())
    .filter(Boolean);
  if (parts.length <= 1) return [];
  return parts
    .map(part => part.replace(/^\s*(?:[-*+]|\d+\.|•)\s+/, "").trim())
    .filter(Boolean);
};

const renderMarkdownTableCell = (cell, options = {}, context = {}) => {
  const items = splitMarkdownTableCellItems(cell);
  if (items.length > 1) {
    const cellOptions = { ...options, inlineContext: context };
    return `<ul class="md-cell-list">${items.map(item => `<li>${renderInlineMarkdown(item, cellOptions)}</li>`).join("")}</ul>`;
  }
  return renderInlineMarkdown(cell, { ...options, inlineContext: context });
};

const normalizeMarkdownHeadingTitle = value => String(value || "")
  .replace(/[`*_~]/g, "")
  .trim();

function renderMarkdown(markdown, options = {}) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let i = 0;
  const collapsedSections = new Set(options.collapsedSections || []);
  let currentSectionTitle = options.currentSectionTitle || "";
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
      if (level === 2) currentSectionTitle = title;
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
        + `<tbody>${rows.map(row => `<tr>${headers.map((header, index) => `<td>${renderMarkdownTableCell(row[index] || "", options, {
          sectionTitle: currentSectionTitle,
          columnHeader: normalizeMarkdownHeadingTitle(header),
          columnIndex: index,
          tableHeaders: headers,
        })}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`
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
  const PING_INTERVAL_MS = 500;
  const OPEN_TIMEOUT_MS = 90000;

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
  let timeoutTimer = null;
  let lastPingError = null;

  const cleanup = () => {
    window.removeEventListener('message', handler);
    if (pingTimer) clearInterval(pingTimer);
    if (timeoutTimer) clearTimeout(timeoutTimer);
  };

  const failPerfettoOpen = (message) => {
    cleanup();
    perfettoOpening.value[slot] = false;
    showPerfettoError(message);
  };

  const handler = (e) => {
    if (e.source !== win || e.origin !== PERFETTO || e.data !== 'PONG') return;
    if (sent || win.closed) return;
    sent = true;
    const message = { perfetto: { buffer, title: fname, fileName: fname } };
    try {
      win.postMessage(message, PERFETTO, [buffer]);
    } catch (err) {
      try {
        win.postMessage(message, PERFETTO);
      } catch (fallbackErr) {
        failPerfettoOpen(`Perfetto 已响应，但 trace 传输失败：${fallbackErr?.message || err?.message || '未知错误'}`);
        return;
      }
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
    try {
      win.postMessage('PING', PERFETTO);
    } catch (err) {
      lastPingError = err;
    }
  };

  ping();
  pingTimer = setInterval(ping, PING_INTERVAL_MS);
  timeoutTimer = setTimeout(() => {
    if (!sent) {
      const extra = lastPingError?.message ? `浏览器返回：${lastPingError.message}` : "";
      failPerfettoOpen(`Perfetto 页面未响应。可能是 Perfetto 页面加载较慢、网络不可达或弹窗被浏览器限制，请稍后重试。${extra}`);
    }
  }, OPEN_TIMEOUT_MS);
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

const resetCompareSelections = () => {
  compareSelection.value = [];
  compareSelectionDetails.value = {};
  compareLabel.value = "";
  clearBatchCompareSelection();
};

const inferCompareProjectId = () => {
  if (filterProject.value) return filterProject.value;
  if (selectedJob.value?.project_id) return selectedJob.value.project_id;
  const projectGroup = historyProjectGroups.value.find(group => group.id && group.id !== "__none__");
  if (projectGroup) return projectGroup.id;
  return historyProjectGroups.value.some(group => group.id === "__none__") ? "__none__" : "";
};

const openCompareModal = projectOrId => {
  resetCompareSelections();
  const fixedProjectId = typeof projectOrId === "string" ? projectOrId : projectOrId?.id;
  compareProjectId.value = fixedProjectId || inferCompareProjectId();
  compareSearch.value = "";
  compareJobsOffset.value = 0;
  showCompareModal.value = true;
  loadCompareJobs();
};

const openProjectCompareModal = projectOrGroup => {
  const project = resolveProjectMeta(projectOrGroup);
  if (!project?.id || project.id === "__none__") return;
  openCompareModal(project.id);
};

const closeCompareModal = () => {
  if (compareSubmitLoading.value || batchCompareLoading.value) return;
  showCompareModal.value = false;
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
  if (compareSubmitLoading.value) return;
  if (!compareProjectId.value) {
    showToast("请先选择项目", "error");
    return;
  }
  if (compareSelection.value.length !== 2) {
    showToast("请选择两个任务进行对比", "error");
    return;
  }
  const [a, b] = compareSelection.value;
  compareSubmitLoading.value = true;
  try {
    const r = await fetch("/api/jobs/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        job_id_a: a, job_id_b: b,
        label: compareLabel.value,
        project_id: compareProjectForSubmit(),
      }),
    });
    const job = await r.json().catch(() => ({}));
    if (!r.ok) {
      showToast("对比失败: " + (job.detail || "服务器错误"), "error");
      return;
    }
    compareSelection.value = [];
    compareSelectionDetails.value = {};
    compareLabel.value = "";
    showCompareModal.value = false;
    sidebarTab.value = "jobs";
    await refreshSidebarData();
    router.push({ path: jobRoutePath(job) });
  } catch (e) {
    showToast("对比失败: " + (e.message || "网络错误"), "error");
  } finally {
    compareSubmitLoading.value = false;
  }
};

const submitBatchCompare = async () => {
  if (!batchBaselineId.value || !batchCandidateIds.value.length || batchCompareLoading.value) return;
  if (!compareProjectId.value) {
    showToast("请先选择项目", "error");
    return;
  }
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
        project_id: compareProjectForSubmit(),
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
    showCompareModal.value = false;
    sidebarTab.value = "jobs";
    await refreshSidebarData();
    if (jobs[0]?.id) router.push({ path: jobRoutePath(jobs[0]) });
  } catch (e) {
    showToast("批量对比失败: 网络或服务器错误", "error");
  } finally {
    batchCompareLoading.value = false;
  }
};

const openCompareSource = source => {
  if (!source?.id) return;
  router.push({ path: jobRoutePath(source) });
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
    router.push({ path: jobRoutePath(job) });
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
    router.push({ path: jobRoutePath(job) });
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
    router.push({ path: jobRoutePath(payload) });
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
  if (newProjectLoading.value) return;
  const name = String(newProjectName.value || "").trim();
  if (!name) return;
  newProjectLoading.value = true;
  try {
    const r = await fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        name,
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
  } catch (e) {
    showToast("创建项目失败: " + (e.message || "网络错误"), "error");
  } finally {
    newProjectLoading.value = false;
  }
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

const navigateToJob = jobOrId => {
  const path = jobRoutePath(jobOrId);
  if (path) router.push({ path });
};

const openHomeProject = project => {
  if (!project?.id) return;
  openRecentProject(project);
  sidebarCollapsed.value = false;
};

// ══════════════════════════════════════════════════════════════════════════════
// Route components
// ══════════════════════════════════════════════════════════════════════════════

let projectPickerSequence = 0;
const ProjectPicker = {
  props: {
    modelValue: { type: String, default: "" },
    projects: { type: Array, default: () => [] },
  },
  emits: ["update:modelValue"],
  template: `
    <div class="project-picker" @focusout="onFocusOut" @keydown.esc.stop="closePicker">
      <button
        class="project-picker-trigger input"
        type="button"
        role="combobox"
        :aria-expanded="String(open)"
        :aria-controls="listboxId"
        :title="selectedProject?.name || '未分组'"
        @click="togglePicker"
        @keydown.down.prevent="openAndMove(1)"
        @keydown.up.prevent="openAndMove(-1)"
      >
        <span :class="['project-picker-mark', selectedProject?.is_public ? 'shared' : '']" aria-hidden="true">
          {{ selectedProject ? (selectedProject.is_public ? '共' : '项') : '—' }}
        </span>
        <span class="project-picker-value">{{ selectedProject?.name || '未分组' }}</span>
        <span v-if="selectedProject?.is_favorite" class="project-picker-favorite" aria-label="已收藏">★</span>
        <span class="project-picker-chevron" aria-hidden="true">⌄</span>
      </button>

      <div v-if="open" class="project-picker-popover">
        <div class="project-picker-search-wrap">
          <span aria-hidden="true">⌕</span>
          <input
            ref="searchInput"
            v-model="query"
            class="project-picker-search"
            placeholder="快速搜索项目..."
            aria-label="快速搜索项目"
            @keydown.down.prevent="moveActive(1)"
            @keydown.up.prevent="moveActive(-1)"
            @keydown.enter.prevent="selectActive"
          />
          <span class="project-picker-count">{{ filteredProjects.length }}</span>
        </div>
        <div :id="listboxId" class="project-picker-list" role="listbox" aria-label="项目">
          <template v-for="group in groupedProjects" :key="group.key">
            <div v-if="group.label" class="project-picker-group">{{ group.label }}</div>
            <button
              v-for="project in group.items"
              :key="project.id || '__none__'"
              :class="['project-picker-option', optionIndex(project) === activeIndex ? 'active' : '', project.id === modelValue ? 'selected' : '']"
              type="button"
              role="option"
              :aria-selected="String(project.id === modelValue)"
              :title="project.name"
              @mouseenter="activeIndex = optionIndex(project)"
              @mousedown.prevent="choose(project.id)"
            >
              <span :class="['project-picker-mark', project.is_public ? 'shared' : '']" aria-hidden="true">
                {{ project.id ? (project.is_public ? '共' : '项') : '—' }}
              </span>
              <span class="project-picker-option-main">
                <strong>{{ project.name }}</strong>
                <small>{{ projectMeta(project) }}</small>
              </span>
              <span v-if="project.is_favorite" class="project-picker-favorite" aria-label="已收藏">★</span>
              <span v-if="project.id === modelValue" class="project-picker-check" aria-hidden="true">✓</span>
            </button>
          </template>
          <div v-if="!flatProjects.length" class="project-picker-empty">
            没有匹配“{{ query.trim() }}”的项目
          </div>
        </div>
        <div class="project-picker-hint">↑↓ 选择 · Enter 确认 · Esc 关闭</div>
      </div>
    </div>
  `,
  setup(props, { emit }) {
    const open = ref(false);
    const query = ref("");
    const activeIndex = ref(0);
    const searchInput = ref(null);
    const listboxId = `project-picker-${++projectPickerSequence}`;
    const ungroupedProject = Object.freeze({ id: "", name: "未分组", is_public: false, is_favorite: false });
    const selectedProject = computed(() =>
      props.projects.find(project => project.id === props.modelValue) || null
    );
    const filteredProjects = computed(() => {
      const keyword = query.value.trim().toLocaleLowerCase();
      if (!keyword) return props.projects;
      return props.projects.filter(project =>
        String(project.name || "").toLocaleLowerCase().includes(keyword)
      );
    });
    const groupedProjects = computed(() => {
      const matches = filteredProjects.value;
      if (query.value.trim()) {
        return [{ key: "results", label: "搜索结果", items: matches }];
      }
      const favoriteIds = new Set(matches.filter(project => project.is_favorite).map(project => project.id));
      return [
        { key: "default", label: "", items: [ungroupedProject] },
        { key: "favorites", label: "收藏", items: matches.filter(project => project.is_favorite) },
        {
          key: "mine",
          label: "我的项目",
          items: matches.filter(project => !favoriteIds.has(project.id) && !(project.is_public && project.is_owner === false)),
        },
        {
          key: "shared",
          label: "共享给我",
          items: matches.filter(project => !favoriteIds.has(project.id) && project.is_public && project.is_owner === false),
        },
      ].filter(group => group.items.length);
    });
    const flatProjects = computed(() => groupedProjects.value.flatMap(group => group.items));
    const projectMeta = project => {
      if (!project.id) return "不归入任何项目";
      if (!project.is_public) return "个人项目";
      return project.is_owner === false ? "共享给我" : "我创建 · 已共享";
    };
    const optionIndex = project => flatProjects.value.findIndex(item => item.id === project.id);
    const focusSelected = () => {
      const selectedIndex = flatProjects.value.findIndex(project => project.id === props.modelValue);
      activeIndex.value = selectedIndex >= 0 ? selectedIndex : 0;
    };
    const openPicker = () => {
      open.value = true;
      query.value = "";
      focusSelected();
      nextTick(() => searchInput.value?.focus());
    };
    const closePicker = () => {
      open.value = false;
      query.value = "";
    };
    const togglePicker = () => open.value ? closePicker() : openPicker();
    const choose = projectId => {
      emit("update:modelValue", projectId);
      closePicker();
    };
    const moveActive = direction => {
      if (!flatProjects.value.length) return;
      activeIndex.value = (activeIndex.value + direction + flatProjects.value.length) % flatProjects.value.length;
    };
    const openAndMove = direction => {
      if (!open.value) openPicker();
      nextTick(() => moveActive(direction));
    };
    const selectActive = () => {
      const project = flatProjects.value[activeIndex.value];
      if (project) choose(project.id);
    };
    const onFocusOut = event => {
      if (!event.currentTarget.contains(event.relatedTarget)) closePicker();
    };
    watch(query, () => { activeIndex.value = 0; });
    return {
      open, query, activeIndex, searchInput, listboxId,
      selectedProject, filteredProjects, groupedProjects, flatProjects,
      projectMeta, optionIndex, togglePicker, closePicker, choose,
      moveActive, openAndMove, selectActive, onFocusOut,
    };
  },
};

const Home = {
  components: { ProjectPicker },
  template: `
    <!-- Submit form -->
    <section class="card submit-card">
      <div class="submit-head">
        <div class="card-title">提交分析</div>
        <div class="upload-mode-toggle">
          <button :class="['mode-toggle-btn', quickUploadMode==='single'?'active':'']"
                  type="button"
                  :aria-pressed="String(quickUploadMode==='single')"
                  :disabled="submitting"
                  @click="setQuickUploadMode('single')">单个</button>
          <button :class="['mode-toggle-btn', quickUploadMode==='compare'?'active':'']"
                  type="button"
                  :aria-pressed="String(quickUploadMode==='compare')"
                  :disabled="submitting"
                  @click="setQuickUploadMode('compare')">两个</button>
          <button :class="['mode-toggle-btn', quickUploadMode==='multi'?'active':'']"
                  type="button"
                  :aria-pressed="String(quickUploadMode==='multi')"
                  :disabled="submitting"
                  @click="setQuickUploadMode('multi')">多个</button>
        </div>
      </div>

      <div v-if="quickUploadMode==='single' || quickUploadMode==='multi'" class="submit-cols">
        <div class="upload-box upload-box-sm" @dragover.prevent @drop.prevent="onDrop">
          <input type="file" ref="fileInputA" accept=".json,.json.gz,.gz,.zip,.tar.gz,.tgz" :multiple="quickUploadMode==='multi'" @change="onFileChange" hidden />
          <button type="button" @click="$refs.fileInputA.click()" class="upload-inner"
                  :aria-label="quickUploadMode==='multi' ? '选择多个 Trace 文件' : '选择单个 Trace 文件'">
            <div class="upload-icon">📂</div>
            <div class="upload-label">
              <span>{{ fileAName || (quickUploadMode==='multi' ? '选择多个文件' : '选择单个文件') }}</span>
              <small v-if="uploadQueue.length===1">{{ uploadQueue[0].meta }}</small>
              <small v-else-if="quickUploadMode==='multi' && uploadQueue.length">将逐个生成分析任务</small>
            </div>
          </button>
          <button v-if="fileAName" class="upload-clear" @click.stop="clearFile">✕</button>
        </div>
        <div class="form-row">
          <label>项目</label>
          <project-picker v-model="form.projectId" :projects="projects"></project-picker>
        </div>
        <div class="form-row">
          <label>别名</label>
          <input v-model="form.label" class="input" :placeholder="quickUploadMode==='multi' ? '可选，将自动追加文件名' : '可选'" />
        </div>
        <button class="btn btn-primary" :disabled="uploadQueue.length===0 || submitting" @click="submitJob">
          {{ submitting ? '提交中 ' + uploadProgress + '%' : (quickUploadMode==='multi' ? '逐个分析' : '提交分析') }}
        </button>
      </div>

      <div v-else class="quick-compare-cols">
        <div class="quick-upload-pair">
          <div class="upload-box upload-box-sm quick-upload-box" @dragover.prevent @drop.prevent="onQuickDrop('a', $event)">
            <input type="file" ref="quickFileInputA" accept=".json,.json.gz,.gz,.zip,.tar.gz,.tgz" @change="onQuickFileChange('a', $event)" hidden />
            <button type="button" @click="$refs.quickFileInputA.click()" class="upload-inner" aria-label="选择 A Trace 文件">
              <span class="trace-slot">A</span>
              <div class="upload-label">
                <span>{{ quickFileAName || '选择 A trace' }}</span>
                <small v-if="quickFileA">{{ uploadFileMeta(quickFileA) }}</small>
              </div>
            </button>
            <button v-if="quickFileAName" class="upload-clear" @click.stop="clearQuickCompareFile('a')">✕</button>
          </div>
          <div class="upload-box upload-box-sm quick-upload-box" @dragover.prevent @drop.prevent="onQuickDrop('b', $event)">
            <input type="file" ref="quickFileInputB" accept=".json,.json.gz,.gz,.zip,.tar.gz,.tgz" @change="onQuickFileChange('b', $event)" hidden />
            <button type="button" @click="$refs.quickFileInputB.click()" class="upload-inner" aria-label="选择 B Trace 文件">
              <span class="trace-slot">B</span>
              <div class="upload-label">
                <span>{{ quickFileBName || '选择 B trace' }}</span>
                <small v-if="quickFileB">{{ uploadFileMeta(quickFileB) }}</small>
              </div>
            </button>
            <button v-if="quickFileBName" class="upload-clear" @click.stop="clearQuickCompareFile('b')">✕</button>
          </div>
        </div>
        <div class="form-row">
          <label>项目</label>
          <project-picker v-model="form.projectId" :projects="projects"></project-picker>
        </div>
        <div class="form-row">
          <label>别名</label>
          <input v-model="form.label" class="input" placeholder="默认 A vs B" />
        </div>
        <button class="btn btn-primary" :disabled="!quickFileA || !quickFileB || submitting" @click="submitQuickCompare">
          {{ submitting ? '提交中 ' + uploadProgress + '%' : '提交对比' }}
        </button>
      </div>

      <div v-if="(quickUploadMode==='single' || quickUploadMode==='multi') && uploadQueue.length" class="upload-queue">
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

    <!-- Returning-user workbench -->
    <section v-if="!selectedJob" class="home-workbench">
      <header class="home-workbench-head">
        <div>
          <span class="home-eyebrow">Performance Workbench</span>
          <h2>继续你的性能分析</h2>
          <p>跟进进行中的任务，或回到最近的结果和项目。</p>
        </div>
        <div class="home-stat-row">
          <span><strong>{{ taskCenterActiveJobs.length }}</strong> 进行中</span>
          <span><strong>{{ homeRecentJobs.length }}</strong> 最近结果</span>
          <span><strong>{{ projects.length }}</strong> 项目</span>
        </div>
      </header>

      <section v-if="taskCenterActiveJobs.length" class="home-active-panel">
        <div class="home-panel-head">
          <div>
            <strong>正在分析</strong>
            <small>任务完成后会自动更新并提醒你</small>
          </div>
          <button class="link-btn" type="button" @click.stop="toggleTaskCenter">查看任务中心</button>
        </div>
        <div class="home-active-grid">
          <button v-for="job in taskCenterActiveJobs" :key="job.id" class="home-active-job" type="button" @click="navigateToJob(job)">
            <span :class="['home-job-status', 'status-'+job.status]">{{ statusIcon(job.status) }}</span>
            <span class="home-job-main">
              <strong>{{ taskCenterJobLabel(job) }}</strong>
              <small>{{ taskCenterJobMeta(job) }}</small>
            </span>
            <span>{{ statusText(job.status) }}</span>
          </button>
        </div>
      </section>

      <div v-if="homeHasActivity" class="home-dashboard-grid">
        <section class="home-dashboard-panel">
          <div class="home-panel-head">
            <div>
              <strong>最近结果</strong>
              <small>{{ homeJobView==='visited' ? '回到最近查看过的结果' : '查看最近完成的分析' }}</small>
            </div>
            <span class="home-panel-actions">
              <span class="home-view-toggle" aria-label="最近结果视图">
                <button type="button" :class="{ active: homeJobView==='visited' }"
                        :aria-pressed="String(homeJobView==='visited')" @click="homeJobView='visited'">最近访问</button>
                <button type="button" :class="{ active: homeJobView==='completed' }"
                        :aria-pressed="String(homeJobView==='completed')" @click="homeJobView='completed'">最近完成</button>
              </span>
              <button v-if="homeJobView==='completed'" class="link-btn" type="button"
                      :disabled="homeDashboardLoading" @click="loadHomeDashboard()">
                {{ homeDashboardLoading ? '刷新中' : '刷新' }}
              </button>
            </span>
          </div>
          <div v-if="homeDashboardError && homeJobView==='completed'" class="home-panel-error">
            <span>{{ homeDashboardError }}</span>
            <button type="button" @click="loadHomeDashboard()">重试</button>
          </div>
          <div v-if="homeDisplayedJobs.length" class="home-list">
            <button v-for="job in homeDisplayedJobs" :key="job.id" class="home-list-row" type="button" @click="navigateToJob(job)">
              <span class="home-job-status status-done">{{ statusIcon(job.status) }}</span>
              <span class="home-job-main">
                <strong>{{ taskCenterJobLabel(job) }}</strong>
                <small>{{ homeJobMeta(job) }}</small>
              </span>
              <span :class="['job-mode-badge', 'mode-'+job.mode]">{{ job.mode==='compare' ? '对比' : '单文件' }}</span>
            </button>
          </div>
          <div v-else-if="!homeDashboardLoading" class="home-panel-empty">
            {{ homeJobView==='visited' ? '尚无最近访问记录，打开一个已完成任务后会显示在这里。' : '尚无已完成任务，提交首个 trace 后会显示在这里。' }}
          </div>
        </section>

        <section class="home-dashboard-panel">
          <div class="home-panel-head">
            <div>
              <strong>项目</strong>
              <small>{{ homeProjectView==='favorite' ? '快速进入收藏项目' : '继续最近项目内的实验和对比' }}</small>
            </div>
            <span class="home-panel-actions">
              <span class="home-view-toggle" aria-label="项目视图">
                <button type="button" :class="{ active: homeProjectView==='recent' }"
                        :aria-pressed="String(homeProjectView==='recent')" @click="homeProjectView='recent'">最近</button>
                <button type="button" :class="{ active: homeProjectView==='favorite' }"
                        :aria-pressed="String(homeProjectView==='favorite')" @click="homeProjectView='favorite'">收藏</button>
              </span>
              <span class="home-panel-count">{{ homeDisplayedProjects.length }}</span>
            </span>
          </div>
          <div v-if="homeDisplayedProjects.length" class="home-project-list">
            <div v-for="project in homeDisplayedProjects" :key="project.id" class="home-project-row">
              <button class="home-project-main" type="button" @click="openHomeProject(project)">
                <span :class="['home-project-icon', project.is_public ? 'shared' : '']">{{ project.is_public ? '共' : '项' }}</span>
                <span>
                  <strong>{{ project.label }}</strong>
                  <small>{{ recentProjectSubtitle(project) }}</small>
                </span>
              </button>
              <button v-if="project.has_experiment_tree" class="home-project-tree" type="button" @click="openRecentProjectTree(project)">实验树</button>
            </div>
          </div>
          <div v-else class="home-panel-empty">
            {{ homeProjectView==='favorite' ? '尚未收藏项目，可在左侧项目管理中添加收藏。' : '创建或打开项目后，可在这里快速返回。' }}
          </div>
        </section>
      </div>

      <section v-else class="home-welcome-panel">
        <strong>从第一个 Trace 开始</strong>
        <span>上传单个 trace 查看热点，或上传两个 trace 直接比较性能变化。</span>
      </section>

      <div class="home-shortcuts">
        <button class="home-shortcut" type="button" @click="showGuide=true">
          <strong>使用指南</strong>
          <span>了解上传、对比、AI 分析和结果解读</span>
        </button>
        <button class="home-shortcut" type="button" @click="$router.push('/feedback')">
          <strong>灵感社区</strong>
          <span>提建议、看讨论，与同事一起完善分析方法</span>
        </button>
      </div>
      <div class="home-format-tip">支持 .json.gz、.gz、.json.zip、.zip、.json、.tar.gz、.tgz</div>
    </section>
  `,
  setup() {
    const fileInputA = ref(null);
    return {
      fileInputA, fileAName, fileA, quickUploadMode,
      quickFileA, quickFileB, quickFileAName, quickFileBName,
      uploadQueue, submitting, uploadProgress,
      form, projects, projectOptionLabel, selectedJob,
      historyGroupsTotal, sidebarTab, showGuide, uploadFileMeta,
      taskCenterActiveJobs, taskCenterJobLabel, taskCenterJobMeta, toggleTaskCenter,
      homeRecentJobs, homeRecentProjects, homeDisplayedJobs, homeDisplayedProjects,
      homeJobView, homeProjectView, homeJobMeta,
      homeHasActivity, homeDashboardLoading, homeDashboardError,
      loadHomeDashboard, navigateToJob, openHomeProject, openRecentProjectTree, recentProjectSubtitle,
      setQuickUploadMode,
      onDrop, onFileChange, clearFile, submitJob,
      onQuickDrop, onQuickFileChange, clearQuickCompareFile, submitQuickCompare,
      fmtCount, fmtDate, statusIcon, statusText,
    };
  },
};

const JobDetail = {
  template: `
    <!-- Loading state -->
    <div v-if="jobLoading && (selectedJobId || selectedJobHandle)" class="empty-main">
      <div class="empty-main-icon">⟳</div>
      <div class="empty-main-title">加载任务...</div>
    </div>

    <!-- 404 state -->
    <div v-else-if="!selectedJob && (selectedJobId || selectedJobHandle)" class="empty-main">
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
          <button class="action-icon-btn action-text-btn" type="button" title="任务管理"
                  aria-label="任务管理"
                  @click="toggleActionMenu('job')">任务管理</button>
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
              <button class="action-icon-btn action-icon-btn-xs action-text-btn" type="button"
                      title="A 文件管理" aria-label="A 文件管理"
                      @click="toggleActionMenu('file-a')">文件管理</button>
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
              <button class="action-icon-btn action-icon-btn-xs action-text-btn" type="button"
                      title="B 文件管理" aria-label="B 文件管理"
                      @click="toggleActionMenu('file-b')">文件管理</button>
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

          <div v-if="selectedJob.mode==='compare'" class="chart-compare-filters">
            <div class="chart-direction-switch" aria-label="变化方向筛选">
              <button v-for="option in [{key:'all',label:'全部'},{key:'slowdown',label:'仅回退'},{key:'speedup',label:'仅改善'}]"
                      :key="option.key" type="button"
                      :class="['chart-filter-button', chartCompareDirection===option.key ? 'active' : '']"
                      @click="chartCompareDirection=option.key; buildChart()">
                {{ option.label }}
              </button>
            </div>
            <label><span>绝对变化 ≥</span><input v-model.number="chartMinAbsDelta" type="number" min="0" step="0.1" class="input input-sm" @change="buildChart" /><em>ms</em></label>
            <label><span>变化率 ≥</span><input v-model.number="chartMinDeltaPct" type="number" min="0" step="1" class="input input-sm" @change="buildChart" /><em>%</em></label>
            <label><span>A 耗时 ≥</span><input v-model.number="chartMinBaselineMs" type="number" min="0" step="0.1" class="input input-sm" @change="buildChart" /><em>ms</em></label>
            <label class="chart-filter-check"><input v-model="chartExcludeCommunication" type="checkbox" @change="buildChart" /> 排除通信</label>
            <span class="chart-filter-result">{{ chartFilterResultText }}</span>
            <button type="button" class="btn btn-xs btn-outline" @click="resetChartCompareFilters">重置</button>
          </div>

          <div class="chart-scope-note">
            <span class="chart-scope-badge">统计口径</span>
            汇总当前数据源的完整设备计算数据，默认剔除通信类 Kernel/Op；不等同于模型端到端耗时。
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
              <div class="chart-delta-title tone-neg">回退贡献 Top</div>
              <button v-for="row in chartSlowdowns" :key="'slow-'+row.source+'-'+row.label"
                      class="chart-delta-row contribution-row"
                      :style="{ '--contribution': Math.min(100, row.contributionPct) + '%' }"
                      @click="drillDownChart(row)">
                <span class="chart-delta-name">{{ row.label }}</span>
                <span class="chart-delta-metrics">
                  <span class="chart-delta-value tone-neg">+{{ fmtDeltaMs(row.delta) }}</span>
                  <span class="chart-delta-share">贡献 {{ fmtChartPercent(row.contributionPct) }} · 累计 {{ fmtChartPercent(row.cumulativePct) }}</span>
                </span>
              </button>
            </div>
            <div class="chart-delta-panel">
              <div class="chart-delta-title tone-pos">Top 改善</div>
              <button v-for="row in chartSpeedups" :key="'fast-'+row.source+'-'+row.label"
                      class="chart-delta-row" @click="drillDownChart(row)">
                <span class="chart-delta-name">{{ row.label }}</span>
                <span class="chart-delta-metrics">
                  <span class="chart-delta-value tone-pos">{{ fmtDeltaMs(row.delta) }}</span>
                  <span class="chart-delta-share">改善贡献 {{ fmtChartPercent(row.contributionPct) }}</span>
                </span>
              </button>
            </div>
          </div>

          <div v-if="selectedJob.mode==='compare' && (chartAddedRows.length || chartRemovedRows.length)" class="chart-status-grid">
            <div v-if="chartAddedRows.length" class="chart-delta-panel">
              <div class="chart-delta-title tone-neg">新增项 Top</div>
              <button v-for="row in chartAddedRows" :key="'added-'+row.source+'-'+row.label"
                      class="chart-delta-row" @click="drillDownChart(row)">
                <span class="chart-delta-name">{{ row.label }}</span>
                <span class="chart-delta-value tone-neg">+{{ fmtChartValue(row.bValue, { unit: 'ms' }) }}</span>
              </button>
            </div>
            <div v-if="chartRemovedRows.length" class="chart-delta-panel">
              <div class="chart-delta-title tone-pos">消失项 Top</div>
              <button v-for="row in chartRemovedRows" :key="'removed-'+row.source+'-'+row.label"
                      class="chart-delta-row" @click="drillDownChart(row)">
                <span class="chart-delta-name">{{ row.label }}</span>
                <span class="chart-delta-value tone-pos">-{{ fmtChartValue(row.aValue, { unit: 'ms' }) }}</span>
              </button>
            </div>
          </div>

          <div class="chart-main-grid">
            <div class="chart-panel chart-panel-wide">
              <div class="chart-panel-title">
                {{ chartDumbbellActive && chartCanvasReady ? 'Top 项 A / B 精确对照（点击下钻）' : '排序排行（点击下钻）' }}
              </div>
              <div v-if="chartDumbbellActive && chartCanvasReady" class="chart-panel-note">
                空心点为 A，实心点为 B；连线颜色表示回退或改善，空心 B 表示推断或低可信度匹配。悬停会与散点图联动。
              </div>
              <div class="chart-bar-area">
                <div v-if="chartBarRows.length && !chartCanvasReady" class="chart-fallback-bars">
                  <div class="chart-fallback-heading">
                    <strong>{{ activeChartTitle }}</strong>
                    <span>{{ chartShareHeader(activeChartMetricDef) }}</span>
                  </div>
                  <button
                    v-for="row in chartBarRows"
                    :key="'chart-fallback-'+row.source+'-'+row.label"
                    class="chart-fallback-row"
                    type="button"
                    @click="drillDownChart(row)"
                  >
                    <span class="chart-fallback-name" :title="row.label">{{ row.shortLabel }}</span>
                    <span class="chart-fallback-track" :style="{ '--bar-width': chartFallbackBarWidth(row) }">
                      <span
                        :class="['chart-fallback-bar', chartFallbackBarTone(row)]"
                        :style="{ width: 'var(--bar-width)' }"
                      ></span>
                      <span class="chart-fallback-value">{{ chartFallbackValueLabel(row) }}</span>
                    </span>
                    <span class="chart-fallback-share">{{ row.shareLabel }}</span>
                  </button>
                </div>
                <canvas
                  ref="ktChart"
                  :class="{ 'chart-canvas-pending': chartBarRows.length && !chartCanvasReady }"
                  role="img"
                  :aria-label="chartDumbbellActive ? 'Top 变化项 A 与 B 耗时精确对照图' : '性能指标排序排行图'"
                ></canvas>
              </div>
            </div>
            <div v-if="chartScatterAvailable" class="chart-panel chart-panel-wide chart-scatter-panel">
              <div class="chart-panel-title">{{ chartScatterTitle }}</div>
              <div class="chart-panel-note">{{ chartScatterNote }}</div>
              <div class="chart-scatter-legend">
                <template v-if="selectedJob.mode==='compare'">
                  <span><i class="scatter-legend-dot regression"></i>红色：B 更慢</span>
                  <span><i class="scatter-legend-dot improvement"></i>绿色：B 更快</span>
                  <span><i class="scatter-legend-line"></i>横线：没有变化</span>
                  <span><i class="scatter-legend-line rate"></i>斜线：±20% 变化率</span>
                  <span><i class="scatter-legend-dot inferred"></i>空心：推断 / 低可信度</span>
                  <span><i class="scatter-legend-size small"></i><i class="scatter-legend-size large"></i>点越大：耗时变化越大</span>
                </template>
                <template v-else>
                  <span><i class="scatter-legend-dot hotspot"></i>Kernel / Op</span>
                  <span><i class="scatter-legend-size small"></i><i class="scatter-legend-size large"></i>点越大，总耗时越高</span>
                </template>
              </div>
              <div class="chart-scatter-area">
                <div v-if="!chartScatterRows.length" class="chart-scatter-empty">{{ chartScatterEmptyText }}</div>
                <canvas v-show="chartScatterRows.length" ref="callTimeScatter" role="img" :aria-label="chartScatterTitle"></canvas>
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
              <input v-model="tableSearch" class="input input-sm table-search-input" placeholder="全局搜索..." aria-label="搜索当前结果表格" />
              <span v-if="hasColFilters" class="filter-active-tip">
                列筛选已启用
                <button class="btn-clear-filter" @click="clearColFilters()">✕ 清除</button>
              </span>
              <span v-if="isKernelTypeTab" class="filter-active-tip">点击类型行下钻到相关 Kernel</span>
              <span v-if="resultTab==='all_kernels_avg.csv' && hasKernelHostOpRelations" class="filter-active-tip">
                多关联或未匹配项可在 Host Op 关联数列展开
              </span>
            </div>
            <div class="table-toolbar-actions">
              <div v-if="isComparisonTable" class="compare-column-presets" aria-label="对比表格列预设">
                <button class="btn btn-sm btn-outline" type="button" @click="applyComparisonColumnPreset('compact')">精简对比</button>
                <button class="btn btn-sm btn-outline" type="button" @click="applyComparisonColumnPreset('calls')">调用数</button>
                <button class="btn btn-sm btn-outline" type="button" @click="applyComparisonColumnPreset('full')">完整字段</button>
              </div>
              <button
                v-if="isEfficiencyTable"
                class="btn btn-sm btn-outline"
                type="button"
                @click="applyEfficiencyColumnPreset"
              >效率视图</button>
              <div class="column-menu-wrap" @click.stop>
                <button class="btn btn-sm btn-outline" type="button"
                        :aria-expanded="String(showColumnMenu)" aria-haspopup="menu"
                        @click="showColumnMenu=!showColumnMenu">
                  列{{ hiddenColumnCount ? ' (' + hiddenColumnCount + ' 已隐藏)' : '' }}
                </button>
                <div v-if="showColumnMenu" class="column-menu">
                  <div class="column-menu-actions">
                    <button class="btn btn-xs btn-outline" @click="resetVisibleColumns">全部列</button>
                    <button class="btn btn-xs btn-outline" @click="applyCoreColumnPreset">核心列</button>
                  </div>
                  <label v-for="f in currentTable.fields" :key="f" class="column-menu-item">
                    <input type="checkbox" :checked="isColumnVisible(f)" @change="toggleColumnVisibility(f)" />
                    <span class="column-menu-label">
                      <strong>{{ tableFieldLabel(f) }}</strong>
                      <small>{{ f }}</small>
                    </span>
                  </label>
                </div>
              </div>
              <div class="action-menu-wrap table-more-menu" @click.stop>
                <button class="action-icon-btn action-text-btn" type="button" title="表格管理"
                        aria-label="表格管理"
                        @click="toggleActionMenu('table')">表格管理</button>
                <div v-if="openActionMenu==='table'" class="action-menu">
                  <button type="button" @click="downloadFilteredCsv(resultTab); closeActionMenu()">导出全部筛选结果</button>
                  <button type="button" @click="downloadCsv(resultTab); closeActionMenu()">下载当前页 CSV</button>
                  <button v-if="resultTab==='all_kernels_avg.csv' && hasKernelHostOpRelations"
                          type="button" @click="downloadKernelHostOpsCsv(); closeActionMenu()">导出 Host Op 关系 CSV</button>
                  <button v-if="isTritonStepTab && allowCodeExecution" type="button"
                          @click="clearInductorCache(); closeActionMenu()">清除 Cache</button>
                </div>
              </div>
            </div>
          </div>
          <div v-if="resultTableError" class="error-box table-error-box mb-2">
            <span>{{ resultTableError }}</span>
            <button class="btn btn-xs btn-outline" type="button" @click="loadResultTable()">重试</button>
          </div>
          <div class="table-scroll">
            <div v-if="resultTableLoading" class="table-loading">
              <span class="spinner-small"></span> 加载表格...
            </div>
            <div class="csv-table-wrap">
            <table v-memo="tableRenderMemo" class="data-table" :style="tableStyle" aria-label="性能分析结果表格">
              <colgroup>
                <col v-for="f in displayedFields" :key="f"
                     :style="tableColumnStyle(f)" />
              </colgroup>
              <thead>
                <tr>
                  <th v-for="f in displayedFields" :key="f"
                      @click="setSort(f, $event)"
                      @keydown.enter.prevent="setSort(f, $event)"
                      @keydown.space.prevent="setSort(f, $event)"
                      :class="['th-sortable', tableHeaderClass(f)]"
                      tabindex="0"
                      :aria-sort="sortCol===f ? (sortAsc ? 'ascending' : 'descending') : 'none'"
                      :aria-label="tableFieldLabel(f) + '，点击排序'"
                      :title="tableFieldTitle(f)"
                      :style="tableColumnStyle(f)">
                    <span v-if="comparisonColumnGroup(f)" class="compare-column-group">{{ comparisonColumnGroup(f) }}</span>
                    <span class="th-label">{{ tableFieldLabel(f) }}</span>
                    <span v-if="sortCol===f" class="th-sort-icon">{{ sortAsc?'↑':'↓' }}</span>
                    <div class="col-resize-handle"
                         aria-hidden="true"
                         @mousedown.stop.prevent="startResize(f, $event)"
                         @click.stop.prevent></div>
                  </th>
                </tr>
                <tr class="filter-row">
                  <th v-for="f in displayedFields" :key="f"
                      :class="tableHeaderClass(f)"
                      :style="tableColumnStyle(f)">
                    <div class="col-filter-wrap">
                      <select v-model="colFilterOps[f]" class="col-filter-op"
                              :class="{ 'op-active': colFilterOps[f] && colFilterOps[f] !== '~' }"
                              :aria-label="tableFieldLabel(f) + ' 筛选条件'"
                              @click.stop>
                        <option value="~">包含</option>
                        <option value="!~">不包含</option>
                        <option value="==">等于</option>
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
                        :type="(!colFilterOps[f] || ['~', '!~', '=='].includes(colFilterOps[f])) ? 'text' : 'number'"
                        :aria-label="'筛选 ' + tableFieldLabel(f)"
                        placeholder="筛选..."
                        @click.stop />
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
                <template v-for="(row,i) in filteredRows" :key="row.kernel_name || i">
                  <tr :class="[{ 'drill-row': canDrillKernelTypeRow(row) }, tableRowClass(row)]"
                      @click="drillDownKernelType(row)">
                    <td v-for="f in displayedFields" :key="f"
                        :class="[deltaCellClass(f, row[f]), tableCellClass(f, row[f])]"
                        :title="row[f]">
                      <template v-if="isKernelTypeDrillCell(f, row)">
                        <button type="button"
                                class="table-cell-link kernel-type-drill-link"
                                :title="'下钻到 ' + row[f] + ' 相关 Kernel'"
                                @click.stop="drillDownKernelType(row)">
                          {{ row[f] }}
                        </button>
                      </template>
                      <template v-else-if="f === 'host_op_count' && canExpandKernelHostOpRow(row)">
                        <button type="button"
                                class="table-cell-link kernel-host-op-toggle"
                                aria-haspopup="dialog"
                                :title="kernelHostOpDetailReason(row) + '，点击查看 Host Op 关联明细'"
                                @click.stop="toggleKernelHostOpDetail(row)">
                          <span>{{ row[f] || 0 }}</span>
                          <span class="kernel-host-op-reason">{{ kernelHostOpDetailReason(row) }}</span>
                          <span class="kernel-host-op-caret" aria-hidden="true">⌄</span>
                        </button>
                      </template>
                      <template v-else-if="f === 'triton_code_file' && row[f]">
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
                      <template v-else-if="isDurationShareField(f)">
                        <span class="duration-share-cell" :style="durationShareStyle(row[f])">
                          <span>{{ formatDurationShare(row[f]) }}</span>
                        </span>
                      </template>
                      <template v-else-if="isSourcePercentField(f)">
                        <span>{{ formatDurationShare(row[f]) }}</span>
                      </template>
                      <template v-else-if="f === 'compare_status'">
                        <span :class="['compare-status-chip', comparisonStatusClass(row[f])]">{{ comparisonStatusLabel(row[f]) }}</span>
                      </template>
                      <template v-else-if="f === 'family'">
                        <span :class="['family-chip', familyChipClass(row[f])]">{{ row[f] || '-' }}</span>
                      </template>
                      <template v-else-if="isEfficiencyField(f)">
                        <span :class="['eff-badge', 'eff-' + efficiencyTone(row[f])]">{{ row[f] || '-' }}</span>
                      </template>
                      <span v-else>{{ row[f] }}</span>
                    </td>
                  </tr>
                </template>
              </tbody>
              <tfoot v-if="filteredRows.length > 0">
                <tr class="sum-row">
                  <td v-for="(f, i) in displayedFields" :key="f" :class="['sum-cell', tableHeaderClass(f)]">
                    <template v-if="i === 0">Σ 合计</template>
                    <template v-else-if="colSums[f] !== null">{{ fmtSum(colSums[f]) }}</template>
                  </td>
                </tr>
              </tfoot>
            </table>
            </div>
          </div>
          <div v-if="kernelHostOpDetail.kernelName"
               class="kernel-host-op-detail-drawer"
               role="dialog"
               aria-label="Host Op 关联明细">
            <div class="kernel-host-op-detail-panel">
              <div class="kernel-host-op-detail-header">
                <div>
                  <strong>Host Op 关联明细</strong>
                  <span :title="kernelHostOpDetail.kernelName">{{ kernelHostOpDetail.kernelName }}</span>
                </div>
                <div>
                  <span v-if="!kernelHostOpDetail.loading && !kernelHostOpDetail.error">
                    {{ kernelHostOpDetail.total }} 条关系<span v-if="kernelHostOpDetail.total > kernelHostOpDetail.rows.length">，显示前 {{ kernelHostOpDetail.rows.length }} 条</span>
                  </span>
                  <button type="button" class="btn btn-xs btn-outline" @click.stop="closeKernelHostOpDetail">收起</button>
                </div>
              </div>
              <div v-if="kernelHostOpDetail.loading" class="kernel-host-op-detail-state">
                <span class="spinner-small"></span> 加载关联明细...
              </div>
              <div v-else-if="kernelHostOpDetail.error" class="kernel-host-op-detail-state error">
                <span>{{ kernelHostOpDetail.error }}</span>
                <button type="button" class="btn btn-xs btn-outline" @click.stop="retryKernelHostOpDetail">重试</button>
              </div>
              <div v-else-if="!kernelHostOpDetail.rows.length" class="kernel-host-op-detail-state">暂无关联明细</div>
              <div v-else class="kernel-host-op-detail-list">
                <div v-for="(relation, relationIndex) in kernelHostOpDetail.rows"
                     :key="relation.host_op + '|' + relation.aten_op + '|' + relationIndex"
                     :class="['kernel-host-op-detail-item', { unmatched: !relation.host_op }]">
                  <div class="kernel-host-op-detail-name">
                    <strong :title="relation.host_op || '未匹配 Host Op'">{{ relation.host_op || '未匹配 Host Op' }}</strong>
                    <span v-if="relation.aten_op && relation.aten_op !== relation.host_op" :title="relation.aten_op">Aten · {{ relation.aten_op }}</span>
                  </div>
                  <span><small>平均调用</small>{{ fmtSum(relation.avg_count) }}</span>
                  <span><small>平均耗时</small>{{ fmtSum(relation.avg_dur_ms) }} ms</span>
                  <span><small>Kernel 内占比</small>{{ formatDurationShare(relation.share_pct) }}</span>
                  <span class="kernel-host-op-match">{{ kernelHostOpMatchLabel(relation.match_method) }}</span>
                </div>
              </div>
            </div>
          </div>
          <div class="table-footer table-footer-paged">
            <div class="table-footer-summary">
              <span>第 {{ tablePageStart }}-{{ tablePageEnd }} 行 / 共 {{ tableTotalRows }} 行</span>
              <span v-if="tableAllRowsCapped && tableLimit >= tableAllRowLimit" class="table-render-cap">
                页面最多展示 {{ fmtCount(tableAllRowLimit) }} 行，完整数据请导出
              </span>
            </div>
            <div class="table-pagination">
              <span class="page-size-control">
                每页
                <select class="input input-xs table-limit-select"
                        :value="tableLimit"
                        :disabled="resultTableLoading"
                        @change="changeTableLimit($event.target.value)">
                  <option v-for="n in tablePageSizeOptions" :key="n" :value="n">{{ n }}</option>
                  <option v-if="customTableLimit" :value="customTableLimit">{{ customTableLimit >= tableTotalRows ? '全部' : '每页' }} {{ customTableLimit }}</option>
                </select>
              </span>
              <button class="btn btn-xs btn-outline" @click="showAllTableRows" :disabled="!tableTotalRows || resultTableLoading || tableLimit >= tableAllRowLimit">{{ tableAllRowsLabel }}</button>
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
    const callTimeScatterRef = ref(null);

    // Wire up template refs to module-level refs so buildChart() can use them
    watch(ktChartRef, (el) => {
      ktChart.value = el;
      if (el && resultTab.value === "chart" && selectedJob.value?.status === "done") {
        scheduleBuildChart();
      }
    });
    watch(callTimeScatterRef, (el) => {
      callTimeScatter.value = el;
    });

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
      router.push({ path: jobRoutePath(selectedJob.value || selectedJobId.value, key) });
    };

    return {
      ktChart: ktChartRef,
      callTimeScatter: callTimeScatterRef,
      selectedJob, selectedJobId, selectedJobHandle, jobLoading, resultTab, availableTabs, currentTable,
      jobStepFilterLabel,
      isReadingMode, toggleReadingMode,
      compareRerunLoading, openStepReanalysisModal,
      consoleSearch, consoleHideWrote, consoleSections, consoleWroteCount,
      consoleSearchMatchCount, scrollConsoleSection,
      chartSource, chartMetric, chartTopN, chartTopNOptions, chartSourceOptions,
      chartCompareDirection, chartMinAbsDelta, chartMinDeltaPct, chartMinBaselineMs,
      chartExcludeCommunication, chartFilterResultText, resetChartCompareFilters,
      chartMetricOptions, chartLoading, chartError, chartSummaryCards,
      chartSlowdowns, chartSpeedups, chartAddedRows, chartRemovedRows, chartBarRows, chartCanvasReady,
      chartDumbbellActive,
      chartScatterRows, chartScatterAvailable, chartScatterTitle, chartScatterNote, chartScatterEmptyText,
      activeChartMetricDef, activeChartTitle, chartShareHeader,
      chartFallbackBarWidth, chartFallbackBarTone, chartFallbackValueLabel,
      buildChart, drillDownChart, fmtDeltaMs, fmtChartPercent, fmtChartValue,
      displayedFields, filteredRows, tableRenderMemo, tableSearch, sortCol, sortAsc, colWidths, colFilters,
      colFilterOps, visibleColumns, showColumnMenu, hiddenColumnCount,
      tableLimit, tableOffset, tableTotalRows, tablePageStart, tablePageEnd,
      tablePageSizeOptions, customTableLimit, changeTableLimit, showAllTableRows,
      tableAllRowLimit, tableAllRowsCapped, tableAllRowsLabel,
      resultTableLoading, resultTableError, preparingResultTab, prevTablePage, nextTablePage, loadResultTable,
      hasColFilters, colSums, isKernelTypeTab, canDrillKernelTypeRow,
      hasKernelHostOpRelations, kernelHostOpDetail,
      canExpandKernelHostOpRow, kernelHostOpDetailReason,
      kernelHostOpMatchLabel, toggleKernelHostOpDetail,
      retryKernelHostOpDetail, closeKernelHostOpDetail,
      isComparisonTable, applyComparisonColumnPreset, comparisonColumnGroup,
      comparisonStatusLabel, comparisonStatusClass,
      isKernelTypeDrillCell, drillDownKernelType,
      isEfficiencyTable, isEfficiencyField, applyEfficiencyColumnPreset,
      tableStyle, tableColumnStyle, tableHeaderClass, tableCellClass, tableRowClass, familyChipClass, efficiencyTone,
      tableFieldLabel, tableFieldTitle, isDurationShareField, isSourcePercentField,
      durationShareStyle, formatDurationShare,
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
      setSort, startResize, downloadCsv, downloadFilteredCsv, downloadKernelHostOpsCsv,
      viewTritonCode, runSingleTriton, clearInductorCache,
      fmtDate, fmtDateTime, fmtSum, fmtBytes, deltaCellClass, clearColFilters,
      isColumnVisible, resetVisibleColumns, applyCoreColumnPreset, toggleColumnVisibility,
      formatConsole,
    };
  },
};

const ExperimentTree = {
  template: `
    <section class="exp-page">
      <header class="exp-toolbar">
        <div class="exp-title-block">
          <button class="btn btn-sm btn-outline" type="button" @click="$router.push('/')">←</button>
          <div>
            <div class="exp-title">实验树</div>
            <div class="exp-subtitle">{{ projectName }}</div>
          </div>
        </div>
        <div class="exp-view-toggle exp-toolbar-view-toggle" role="tablist" aria-label="实验树视图">
          <button
            type="button"
            :class="['exp-view-tab', viewMode === 'canvas' ? 'active' : '']"
            :aria-selected="viewMode === 'canvas'"
            @click="viewMode='canvas'"
          >画布</button>
          <button
            type="button"
            :class="['exp-view-tab', viewMode === 'chart' ? 'active' : '']"
            :aria-selected="viewMode === 'chart'"
            @click="viewMode='chart'"
          >折线图</button>
          <button
            type="button"
            :class="['exp-view-tab', viewMode === 'roi' ? 'active' : '']"
            :aria-selected="viewMode === 'roi'"
            @click="viewMode='roi'"
          >变量收益</button>
        </div>
        <div class="exp-toolbar-actions">
          <button class="btn btn-sm btn-primary" type="button" @click="openAddEdge()">标记优化关系</button>
          <button
            v-if="viewMode === 'canvas'"
            class="btn btn-sm btn-outline exp-action-btn"
            type="button"
            title="自动整理"
            aria-label="自动整理"
            @click="resetLayout"
            :disabled="saving"
          ><span aria-hidden="true">⤢</span>自动整理</button>
          <button
            class="btn btn-sm btn-outline exp-action-btn"
            type="button"
            title="刷新"
            aria-label="刷新"
            @click="loadGraph"
            :disabled="loading"
          ><span aria-hidden="true">⟳</span>刷新</button>
        </div>
      </header>

      <datalist id="exp-variable-name-options">
        <option v-for="name in variableNameOptions" :key="name" :value="name"></option>
      </datalist>

      <div :class="['exp-body', panelCollapsed ? 'panel-collapsed' : '']">
        <div
          v-if="viewMode === 'canvas'"
          ref="viewportRef"
          class="exp-canvas"
          @mousedown="startPan"
          @wheel.prevent="onWheel"
        >
          <div v-if="loading" class="exp-loading">加载中...</div>
          <div
            v-else
            class="exp-layer"
            :style="{
              width: canvasSize.width + 'px',
              height: canvasSize.height + 'px',
              transform: 'translate(' + view.tx + 'px,' + view.ty + 'px) scale(' + view.scale + ')'
            }"
          >
            <svg class="exp-edge-svg" :width="canvasSize.width" :height="canvasSize.height">
              <path
                v-for="item in edgePaths"
                :key="item.edge.id"
                :d="item.d"
                :class="['exp-edge-path', edgeOptimizationClass(item.edge), isEdgeHighlighted(item.edge) ? 'active' : '', hasSelection && !isEdgeHighlighted(item.edge) ? 'muted' : '']"
              ></path>
              <path
                v-for="item in edgePaths"
                :key="'arrow-' + item.edge.id"
                :d="item.arrowD"
                :class="['exp-edge-arrow', edgeOptimizationClass(item.edge), isEdgeHighlighted(item.edge) ? 'active' : '', hasSelection && !isEdgeHighlighted(item.edge) ? 'muted' : '']"
              ></path>
              <template v-for="item in edgePaths" :key="'connector-' + item.edge.id">
                <path
                  v-if="item.connectorD"
                  :d="item.connectorD"
                  :class="['exp-edge-label-link', edgeOptimizationClass(item.edge), isEdgeHighlighted(item.edge) ? 'active' : '', hasSelection && !isEdgeHighlighted(item.edge) ? 'muted' : '']"
                ></path>
              </template>
            </svg>

            <div
              v-for="item in edgePaths"
              :key="'label-' + item.edge.id"
              :class="['exp-edge-label', edgeOptimizationClass(item.edge), isEdgeHighlighted(item.edge) ? 'active' : '', hasSelection && !isEdgeHighlighted(item.edge) ? 'muted' : '']"
              :style="edgeLabelStyle(item)"
              role="button"
              tabindex="0"
              @mousedown.stop.prevent="startEdgeLabelDrag(item.edge, item, $event)"
              @click.stop="selectEdge(item.edge)"
              @keydown.enter.prevent="selectEdge(item.edge)"
              @keydown.space.prevent="selectEdge(item.edge)"
              @mouseenter="hoverEdgeId = item.edge.id"
              @mouseleave="hoverEdgeId = ''"
            >
              <div class="exp-edge-label-main">
                <div class="exp-edge-label-content">
                  <span
                    v-if="edgeLabelDescription(item.edge)"
                    class="exp-edge-label-text"
                    :title="edgeLabelDescription(item.edge)"
                  >{{ edgeLabelDescription(item.edge) }}</span>
                  <div v-if="edgeLabelVariables(item.edge).length" class="exp-edge-chip-row">
                    <span
                      v-for="(variable, index) in edgeLabelVariables(item.edge)"
                      :key="index + '-' + variable.name + '-' + variable.from + '-' + variable.to"
                      class="exp-edge-var-chip"
                      :title="variableDisplayLabel(variable)"
                    >{{ variableDisplayLabel(variable) }}</span>
                  </div>
                  <span v-if="!edgeLabelDescription(item.edge) && !edgeLabelVariables(item.edge).length" class="exp-edge-label-text">优化关系</span>
                </div>
                <em :class="deltaClass(item.edge)">{{ edgeDeltaChipText(item.edge) }}</em>
              </div>
              <button
                class="exp-edge-resize"
                type="button"
                title="缩放关系框"
                aria-label="缩放关系框"
                @mousedown.stop.prevent="startEdgeLabelResize(item.edge, item, $event)"
              >↘</button>
            </div>

            <article
              v-for="node in displayNodes"
              :key="node.id"
              :class="[
                'exp-node',
                node.status,
                nodeOptimizationClass(node),
                isBaselineNode(node) ? 'baseline-node' : '',
                selectedNodeId === node.id ? 'selected' : '',
                hasSelection && !isNodeHighlighted(node) ? 'muted' : ''
              ]"
              :style="nodeStyle(node)"
              @mousedown.stop="startNodeDrag(node, $event)"
              @click.stop="selectNode(node)"
              @dblclick.stop="openJob(node)"
            >
              <div class="exp-node-head">
                <span :class="['exp-status-dot', node.status]"></span>
                <strong :title="nodeTitle(node)">{{ nodeTitle(node) }}</strong>
                <span v-if="node.status !== 'done'" class="exp-node-chip status">{{ statusText(node.status) }}</span>
              </div>
              <div class="exp-node-primary">
                <b>{{ formatNodeMetricNumber(node, 'compute_ms', 'ms') }}</b>
                <span>ms · compute time</span>
                <em v-if="nodePrimaryDeltaText(node)" :class="nodeMetricChipClass(node, 'compute_ms')">{{ nodePrimaryDeltaText(node) }}</em>
              </div>
              <div class="exp-node-secondary">
                <div>
                  <span>E2E</span>
                  <b>{{ formatNodeMetricValue(node, 'e2e_ms', 'ms') }}</b>
                  <em v-if="nodeMetricChipText(node, 'e2e_ms', 'e2e')" :class="nodeMetricChipClass(node, 'e2e_ms')">{{ nodeMetricChipText(node, 'e2e_ms', 'e2e') }}</em>
                </div>
                <div>
                  <span>Kernel</span>
                  <b>{{ formatNodeMetricValue(node, 'kernel_count', 'count') }}</b>
                  <em v-if="nodeMetricChipText(node, 'kernel_count', 'kernel', 'count')" :class="nodeMetricChipClass(node, 'kernel_count')">{{ nodeMetricChipText(node, 'kernel_count', 'kernel', 'count') }}</em>
                </div>
              </div>
            </article>

          </div>
          <button
            v-if="panelCollapsed"
            class="exp-panel-expand-btn"
            type="button"
            title="展开右侧面板"
            aria-label="展开右侧面板"
            @click.stop="panelCollapsed=false"
          >‹</button>
          <div class="exp-canvas-hint">拖拽排布 · 滚轮缩放 · 拖空白平移</div>
        </div>

        <div
          v-else-if="viewMode === 'chart'"
          class="exp-chart-view"
        >
          <div v-if="loading" class="exp-loading">加载中...</div>
          <template v-else>
            <div class="exp-chart-toolbar exp-subtoolbar">
              <div class="exp-chart-controls">
                <label class="exp-chart-field">
                  <span>主指标</span>
                  <select v-model="lineageMetricKey" class="input exp-chart-select">
                    <option v-for="metric in lineageMetricOptions" :key="metric.key" :value="metric.key">{{ metric.label }}</option>
                  </select>
                </label>
                <label class="exp-chart-field exp-chart-target-field">
                  <span>{{ currentMetricTargetTitle }}</span>
                  <input
                    v-model="projectMetricTargetDraft"
                    class="input exp-chart-target-input"
                    type="text"
                    inputmode="decimal"
                    placeholder="未设置"
                    @keydown.enter.prevent="saveProjectMetricTarget"
                  />
                </label>
                <button
                  class="btn btn-sm btn-primary"
                  type="button"
                  @click="saveProjectMetricTarget"
                  :disabled="projectMetricTargetSaving || !projectMetricTargetDirty"
                >{{ projectMetricTargetSaving ? '保存中' : '保存目标' }}</button>
                <button
                  v-if="currentMetricTargetValue !== null || projectMetricTargetDraft"
                  class="btn btn-sm btn-outline"
                  type="button"
                  @click="clearProjectMetricTarget"
                  :disabled="projectMetricTargetSaving"
                >清除</button>
              </div>
              <div v-if="lineageChartWarning" class="exp-chart-note">{{ lineageChartWarning }}</div>
            </div>
            <div v-if="lineageChartEmptyText" class="exp-chart-empty">{{ lineageChartEmptyText }}</div>
            <div v-else class="exp-chart-canvas-wrap">
              <canvas ref="lineageChartCanvas"></canvas>
            </div>
          </template>
          <button
            v-if="panelCollapsed"
            class="exp-panel-expand-btn"
            type="button"
            title="展开右侧面板"
            aria-label="展开右侧面板"
            @click.stop="panelCollapsed=false"
          >‹</button>
          <div class="exp-canvas-hint">拖空白平移</div>
        </div>

        <div
          v-else-if="viewMode === 'roi'"
          class="exp-roi-view"
        >
          <div v-if="loading" class="exp-loading">加载中...</div>
          <template v-else>
            <div class="exp-roi-toolbar exp-subtoolbar">
              <div class="exp-chart-controls">
                <label class="exp-chart-field">
                  <span>主指标</span>
                  <select v-model="roiMetricKey" class="input exp-chart-select">
                    <option v-for="metric in roiMetricOptions" :key="metric.key" :value="metric.key">{{ metric.label }}</option>
                  </select>
                </label>
                <div class="exp-segmented" role="group" aria-label="收益聚合口径">
                  <button
                    type="button"
                    :class="['exp-segmented-btn', roiGroupMode === 'variable' ? 'active' : '']"
                    @click="roiGroupMode='variable'"
                  >按变量</button>
                  <button
                    type="button"
                    :class="['exp-segmented-btn', roiGroupMode === 'change' ? 'active' : '']"
                    @click="roiGroupMode='change'"
                  >按具体变更</button>
                </div>
                <div class="exp-segmented" role="group" aria-label="条形图指标">
                  <button
                    type="button"
                    :class="['exp-segmented-btn', roiBarMode === 'total' ? 'active' : '']"
                    @click="roiBarMode='total'"
                  >总收益</button>
                  <button
                    type="button"
                    :class="['exp-segmented-btn', roiBarMode === 'average' ? 'active' : '']"
                    @click="roiBarMode='average'"
                  >平均收益</button>
                </div>
              </div>
              <div v-if="roiSkippedEdgeCount" class="exp-chart-note">有 {{ roiSkippedEdgeCount }} 条带变量关系缺少 {{ currentRoiMetricLabel }} delta，已计入无法评估。</div>
            </div>

            <div v-if="roiEmptyText" class="exp-chart-empty">{{ roiEmptyText }}</div>
            <template v-else>
              <div class="exp-roi-summary">
                <div><span>变量种类</span><b>{{ roiSummary.variableCount }}</b></div>
                <div><span>可评估关系</span><b>{{ roiSummary.evaluableEdgeCount }}</b></div>
                <div><span>收益最高</span><b :class="roiGainClass(roiSummary.best?.totalGainMs)">{{ roiSummary.best ? roiSummary.best.label : '—' }}</b></div>
                <div><span>退化最多</span><b :class="roiGainClass(roiSummary.worst?.totalGainMs)">{{ roiSummary.worst ? roiSummary.worst.label : '—' }}</b></div>
              </div>

              <div v-if="roiBarItems.length" class="exp-roi-bars">
                <button
                  v-for="item in roiBarItems"
                  :key="item.key"
                  class="exp-roi-bar-row"
                  type="button"
                  :title="item.distributionText"
                  @click="focusRoiGroup(item)"
                >
                  <span class="exp-roi-bar-label">{{ item.label }}</span>
                  <span class="exp-roi-bar-track">
                    <i :class="roiGainClass(item.barValue)" :style="{ width: item.barWidth + '%' }"></i>
                  </span>
                  <b :class="roiGainClass(item.barValue)">{{ formatRoiGainMs(item.barValue) }}</b>
                </button>
              </div>
              <div v-else class="exp-roi-isolated-empty">当前只有组合变更，暂无法单独归因；做单变量实验后这里会显示各变量收益</div>

              <div class="exp-roi-table-wrap">
                <table class="exp-roi-table">
                  <thead>
                    <tr>
                      <th><button type="button" :class="{ active: roiSort.key === 'label' }" @click="setRoiSort('label')">变量<span class="exp-sort-mark">{{ roiSortMark('label') || '↕' }}</span></button></th>
                      <th><button type="button" :class="{ active: roiSort.key === 'isolatedCount' }" @click="setRoiSort('isolatedCount')">单独(组合)<span class="exp-sort-mark">{{ roiSortMark('isolatedCount') || '↕' }}</span></button></th>
                      <th><button type="button" :class="{ active: roiSort.key === 'averageGainMs' }" @click="setRoiSort('averageGainMs')">平均收益<span class="exp-sort-mark">{{ roiSortMark('averageGainMs') || '↕' }}</span></button></th>
                      <th><button type="button" :class="{ active: roiSort.key === 'totalGainMs' }" @click="setRoiSort('totalGainMs')">总收益<span class="exp-sort-mark">{{ roiSortMark('totalGainMs') || '↕' }}</span></button></th>
                      <th><button type="button" :class="{ active: roiSort.key === 'hitRate' }" @click="setRoiSort('hitRate')">命中率<span class="exp-sort-mark">{{ roiSortMark('hitRate') || '↕' }}</span></button></th>
                      <th><button type="button" :class="{ active: roiSort.key === 'bestGainMs' }" @click="setRoiSort('bestGainMs')">最佳一次<span class="exp-sort-mark">{{ roiSortMark('bestGainMs') || '↕' }}</span></button></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="row in sortedRoiRows"
                      :key="row.key"
                      :title="row.distributionText"
                      @click="focusRoiGroup(row)"
                    >
                      <td>
                        <strong>{{ row.label }}</strong>
                        <span v-if="row.groupMode === 'change'">{{ row.variableName }}</span>
                      </td>
                      <td><b>{{ row.isolatedCount }}</b><span>({{ row.combinedCount }})</span></td>
                      <td :class="roiGainClass(row.averageGainMs)">
                        <b>{{ formatRoiGainMs(row.averageGainMs) }}</b>
                        <span v-if="isFiniteRoiValue(row.averageGainPct)">{{ formatRoiGainPct(row.averageGainPct) }}</span>
                      </td>
                      <td :class="roiGainClass(row.totalGainMs)">
                        <b>{{ formatRoiGainMs(row.totalGainMs) }}</b>
                        <span v-if="isFiniteRoiValue(row.totalGainPct)">{{ formatRoiGainPct(row.totalGainPct) }}</span>
                      </td>
                      <td>{{ formatHitRate(row.hitRate) }}</td>
                      <td>
                        <button
                          v-if="row.bestEdgeId"
                          class="btn btn-xs btn-outline exp-roi-best-link"
                          type="button"
                          @click.stop="openRoiBestEdge(row)"
                        >{{ formatRoiGainMs(row.bestGainMs) }}</button>
                        <span v-else>—</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <section class="exp-roi-section">
                <div class="exp-roi-section-head">
                  <strong>组合变更</strong>
                  <span>组合效果，未拆分到单变量</span>
                </div>
                <div v-if="roiCombinedEdges.length" class="exp-roi-combined-list">
                  <button
                    v-for="item in roiCombinedEdges"
                    :key="item.edge.id"
                    class="exp-roi-combined-item"
                    type="button"
                    @click="focusEdge(item.edge)"
                  >
                    <span>{{ item.variablesText }}</span>
                    <b :class="roiGainClass(item.gainMs)">{{ formatRoiGainMs(item.gainMs) }} <em>{{ formatRoiGainPct(item.gainPct) }}</em></b>
                  </button>
                </div>
                <div v-else class="exp-empty-small">暂无组合变更关系</div>
              </section>

              <section class="exp-roi-section">
                <div class="exp-roi-section-head">
                  <strong>多版本指标</strong>
                  <span>所有节点关键指标</span>
                </div>
                <div class="exp-roi-table-wrap">
                  <table class="exp-roi-table exp-node-metric-table">
                    <thead>
                      <tr>
                        <th><button type="button" :class="{ active: nodeMetricSort.key === 'label' }" @click="setNodeMetricSort('label')">节点<span class="exp-sort-mark">{{ nodeMetricSortMark('label') || '↕' }}</span></button></th>
                        <th><button type="button" :class="{ active: nodeMetricSort.key === 'e2e_ms' }" @click="setNodeMetricSort('e2e_ms')">E2E<span class="exp-sort-mark">{{ nodeMetricSortMark('e2e_ms') || '↕' }}</span></button></th>
                        <th><button type="button" :class="{ active: nodeMetricSort.key === 'compute_ms' }" @click="setNodeMetricSort('compute_ms')">Compute<span class="exp-sort-mark">{{ nodeMetricSortMark('compute_ms') || '↕' }}</span></button></th>
                        <th><button type="button" :class="{ active: nodeMetricSort.key === 'kernel_count' }" @click="setNodeMetricSort('kernel_count')">Kernel<span class="exp-sort-mark">{{ nodeMetricSortMark('kernel_count') || '↕' }}</span></button></th>
                        <th>操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="node in sortedNodeMetricRows" :key="node.id" @click="selectNodeFromRoi(node)">
                        <td><strong>{{ nodeTitle(node) }}</strong><span>{{ statusText(node.status) }}</span></td>
                        <td>{{ formatMs(node.e2e_ms) }}</td>
                        <td>{{ formatMs(node.compute_ms) }}</td>
                        <td>{{ formatCount(node.kernel_count) }}</td>
                        <td><button class="btn btn-xs btn-outline" type="button" @click.stop="openJob(node)">详情</button></td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </section>
            </template>
          </template>
          <button
            v-if="panelCollapsed"
            class="exp-panel-expand-btn"
            type="button"
            title="展开右侧面板"
            aria-label="展开右侧面板"
            @click.stop="panelCollapsed=false"
          >‹</button>
        </div>

        <aside v-if="!panelCollapsed" class="exp-panel">
          <button
            class="exp-panel-fold-btn"
            type="button"
            title="折叠右侧面板"
            aria-label="折叠右侧面板"
            @click="panelCollapsed=true"
          >›</button>

          <template v-if="selectedEdge">
            <div class="exp-edge-editor-tools">
              <button class="btn btn-sm btn-outline" type="button" @click="selectedEdgeId=''">关闭</button>
            </div>

            <div class="exp-editor-card">
              <label class="exp-field">
                <span>连线显示正文</span>
                <textarea v-model="edgeDraft.title" class="input exp-display-textarea" placeholder="写在连线框里的正文"></textarea>
              </label>
            </div>

            <div class="exp-editor-card exp-variable-card">
              <div class="exp-field-title">
                <span>变更项</span>
                <button
                  v-if="edgeDraft.variablesText"
                  class="btn btn-sm btn-outline"
                  type="button"
                  @click="edgeDraft.variablesText=''"
                >清空</button>
              </div>
              <div class="exp-variable-quick">
                <input
                  v-model="edgeVariableQuick.name"
                  class="input"
                  list="exp-variable-name-options"
                  placeholder="变量名"
                  @keydown.enter.prevent="appendQuickVariable('edge')"
                />
                <input v-model="edgeVariableQuick.from" class="input" placeholder="from" @keydown.enter.prevent="appendQuickVariable('edge')" />
                <input v-model="edgeVariableQuick.to" class="input" placeholder="to" @keydown.enter.prevent="appendQuickVariable('edge')" />
                <button class="btn btn-sm btn-outline" type="button" @click="appendQuickVariable('edge')">加入</button>
              </div>
              <textarea
                v-model="edgeDraft.variablesText"
                class="input exp-variable-textarea"
                placeholder="gemm: 1 -> 2&#10;triton: 3 -> 4&#10;或者直接粘贴一段变更说明"
              ></textarea>
              <div class="exp-field-hint">每行一条；能识别“名称: 前 -> 后”，其他内容会作为说明保留。</div>
            </div>

            <div v-if="draftVariableInsertItems.length" class="exp-field exp-display-insert">
              <div class="exp-field-title">
                <span>变量加入显示</span>
                <button class="btn btn-sm btn-outline" type="button" @click="appendAllDraftVariablesToTitle">全部加入</button>
              </div>
              <div class="exp-insert-chip-list">
                <button
                  v-for="item in draftVariableInsertItems"
                  :key="item.key"
                  class="exp-insert-chip"
                  type="button"
                  @click="appendDraftVariableToTitle(item.text)"
                >+ {{ item.label }}</button>
              </div>
            </div>

            <div v-if="selectedEdgePerfNote" class="exp-inline-note">
              <span>{{ selectedEdgePerfNote }}</span>
              <button v-if="selectedEdge.perf?.incomplete" class="btn btn-sm btn-outline" type="button" @click="refreshPerf" :disabled="saving">重新计算</button>
            </div>

            <div class="exp-panel-actions">
              <button class="btn btn-primary" type="button" @click="saveEdge" :disabled="saving">保存</button>
              <button v-if="selectedEdge.compare_job_id" class="btn btn-outline" type="button" @click="openJob(selectedEdge.compare_job_id)">查看对比详情 ↗</button>
              <button v-else class="btn btn-outline" type="button" @click="createCompare" :disabled="saving">生成详细对比</button>
              <button class="btn btn-danger" type="button" @click="deleteSelectedEdge" :disabled="saving">删除关系</button>
            </div>
          </template>

          <template v-else-if="selectedNode">
            <div class="exp-panel-head">
              <div class="exp-node-name-editor">
                <small>节点</small>
                <input
                  v-model="nodeNameDraft"
                  class="input exp-node-name-input"
                  :placeholder="nodeTitle(selectedNode)"
                  @keydown.enter.prevent="saveNodeName"
                />
                <div class="exp-node-name-actions">
                  <button
                    class="btn btn-sm btn-primary"
                    type="button"
                    @click="saveNodeName"
                    :disabled="nodeNameSaving || !nodeNameDirty"
                  >{{ nodeNameSaving ? '保存中' : '保存名称' }}</button>
                  <button
                    class="btn btn-sm btn-outline"
                    type="button"
                    @click="resetNodeNameDraft"
                    :disabled="nodeNameSaving || !nodeNameDirty"
                  >还原</button>
                </div>
              </div>
              <button class="btn btn-sm btn-outline" type="button" @click="selectedNodeId=''">关闭</button>
            </div>
            <div class="exp-node-detail">
              <div
                v-for="row in selectedNodeDetailRows"
                :key="row.key"
                :class="['exp-node-detail-row', 'tone-' + row.tone]"
              >
                <span :title="row.title || row.label">{{ row.label }}</span>
                <b>
                  <i>{{ row.value }}</i>
                  <em v-if="row.deltaText" :class="row.deltaClass">{{ row.deltaText }}</em>
                </b>
              </div>
            </div>
            <div class="exp-node-compare">
              <label class="exp-field">
                <span>与节点对比</span>
                <select v-model="nodeCompareTargetId" class="input">
                  <option value="">选择另一个节点</option>
                  <option v-for="job in nodeCompareTargetOptions" :key="job.id" :value="job.id">{{ jobOptionLabel(job) }}</option>
                </select>
              </label>
              <button class="btn btn-sm btn-outline" type="button" @click="compareSelectedNode" :disabled="nodeCompareSaving || !nodeCompareTargetId">
                {{ nodeCompareSaving ? '生成中...' : '打开对比' }}
              </button>
            </div>
            <div class="exp-node-attachments">
              <div class="exp-node-attachments-head">
                <div>
                  <strong>节点附件</strong>
                  <span>{{ selectedNodeAttachments.length ? selectedNodeAttachments.length + ' 个附件' : '暂无附件' }}</span>
                </div>
                <label class="btn btn-sm btn-outline exp-node-upload-btn" :class="{ disabled: nodeAttachmentUploading }">
                  <input type="file" @change="uploadNodeAttachment" :disabled="nodeAttachmentUploading" />
                  {{ nodeAttachmentUploading ? '上传中...' : '上传附件' }}
                </label>
              </div>
              <div v-if="selectedNodeAttachments.length" class="exp-node-attachment-list">
                <div
                  v-for="attachment in selectedNodeAttachments"
                  :key="attachment.id"
                  class="exp-node-attachment-item"
                >
                  <div>
                    <strong :title="attachment.filename">{{ attachment.filename || '未命名附件' }}</strong>
                    <span>
                      {{ fmtBytes(attachment.size_bytes) }}
                      <template v-if="attachment.uploaded_by"> · {{ attachment.uploaded_by }}</template>
                      <template v-if="attachment.uploaded_at"> · {{ fmtDateTime(attachment.uploaded_at) }}</template>
                    </span>
                  </div>
                  <div class="exp-node-attachment-actions">
                    <button class="btn btn-xs btn-outline" type="button" @click="downloadNodeAttachment(attachment)">下载</button>
                    <button
                      class="btn btn-xs btn-outline"
                      type="button"
                      @click="deleteNodeAttachment(attachment)"
                      :disabled="nodeAttachmentDeletingId === attachment.id"
                    >{{ nodeAttachmentDeletingId === attachment.id ? '删除中' : '删除' }}</button>
                  </div>
                </div>
              </div>
              <div v-else class="exp-node-attachment-empty">支持上传当前节点相关文件，单个附件最大 500MB。</div>
            </div>
            <div class="exp-panel-actions">
              <button class="btn btn-primary" type="button" @click="openJob(selectedNode)">打开完整分析</button>
            </div>
          </template>

          <template v-else>
            <div class="exp-panel-head">
              <div>
                <small>项目</small>
                <strong>{{ projectName }}</strong>
              </div>
            </div>
            <div class="exp-summary-grid">
              <div><span>节点</span><b>{{ nodes.length }}</b></div>
              <div><span>关系</span><b>{{ edges.length }}</b></div>
              <div><span>未连接</span><b>{{ unconnected.length }}</b></div>
              <div><span>{{ currentMetricTargetTitle }}</span><b>{{ currentMetricTargetValueLabel }}</b></div>
            </div>
            <div class="exp-panel-subtitle">未连接任务</div>
            <div class="exp-unconnected-list">
              <div v-for="job in unconnected" :key="job.id" class="exp-unconnected-item">
                <div>
                  <strong>{{ nodeTitle(job) }}</strong>
                  <span>E2E {{ formatMs(job.e2e_ms) }} · Kernel {{ formatCount(job.kernel_count) }} · Compute {{ formatMs(job.compute_ms) }}</span>
                </div>
                <div class="exp-unconnected-actions">
                  <button class="btn btn-xs btn-primary" type="button" @click="openAddEdge('', job.id)">设为优化结果</button>
                  <button class="btn btn-xs btn-outline" type="button" @click="openJob(job)">详情</button>
                </div>
              </div>
              <div v-if="!unconnected.length" class="exp-empty-small">暂无未连接任务</div>
            </div>
          </template>
        </aside>
      </div>

      <div v-if="showAddEdge" class="modal-mask modal-mask-front" @click.self="closeAddEdge" @keydown.esc.prevent.stop="closeAddEdge">
        <div class="modal exp-edge-modal" role="dialog" aria-modal="true" aria-label="标记优化关系">
          <div class="modal-title">
            <span>标记优化关系</span>
            <button class="btn btn-sm btn-outline" type="button" @click="closeAddEdge">关闭</button>
          </div>
          <div class="exp-edge-form">
            <label class="exp-field">
              <span>父节点</span>
              <select v-model="addForm.parent_job_id" class="input">
                <option value="">选择父节点</option>
                <option v-for="job in candidateOptions" :key="job.id" :value="job.id">{{ jobOptionLabel(job) }}</option>
              </select>
            </label>
            <label class="exp-field">
              <span>子节点</span>
              <select v-model="addForm.child_job_id" class="input">
                <option value="">选择子节点</option>
                <option v-for="job in candidateOptions" :key="job.id" :value="job.id">{{ jobOptionLabel(job) }}</option>
              </select>
            </label>
            <label class="exp-field">
              <span>名称</span>
              <input v-model="addForm.title" class="input" />
            </label>
            <label class="exp-field">
              <span>描述</span>
              <textarea v-model="addForm.description" class="input exp-textarea"></textarea>
            </label>
            <div class="exp-field">
              <div class="exp-field-title">
                <span>变更项</span>
              </div>
              <div class="exp-variable-quick">
                <input
                  v-model="addVariableQuick.name"
                  class="input"
                  list="exp-variable-name-options"
                  placeholder="变量名"
                  @keydown.enter.prevent="appendQuickVariable('add')"
                />
                <input v-model="addVariableQuick.from" class="input" placeholder="from" @keydown.enter.prevent="appendQuickVariable('add')" />
                <input v-model="addVariableQuick.to" class="input" placeholder="to" @keydown.enter.prevent="appendQuickVariable('add')" />
                <button class="btn btn-sm btn-outline" type="button" @click="appendQuickVariable('add')">加入</button>
              </div>
              <textarea
                v-model="addForm.variablesText"
                class="input exp-variable-textarea exp-variable-textarea-compact"
                placeholder="gemm: 1 -> 2&#10;triton: 3 -> 4&#10;或者直接粘贴一段变更说明"
              ></textarea>
              <div class="exp-field-hint">每行一条；能识别“名称: 前 -> 后”。</div>
            </div>
          </div>
          <div class="modal-actions">
            <button class="btn btn-outline" type="button" @click="closeAddEdge">取消</button>
            <button class="btn btn-primary" type="button" @click="submitAddEdge" :disabled="saving">创建</button>
          </div>
        </div>
      </div>
    </section>
  `,
  setup() {
    const route = VueRouter.useRoute();
    const viewportRef = ref(null);
    const lineageChartCanvas = ref(null);
    const loading = ref(false);
    const saving = ref(false);
    const nodes = ref([]);
    const unconnected = ref([]);
    const edges = ref([]);
    const projectMeta = ref(null);
    const projectMetricTargetDraft = ref("");
    const projectMetricTargetSaving = ref(false);
    const candidateJobs = ref([]);
    const selectedNodeId = ref("");
    const selectedEdgeId = ref("");
    const hoverEdgeId = ref("");
    const showAddEdge = ref(false);
    const panelCollapsed = ref(false);
    const viewMode = ref("canvas");
    const lineageMetricKey = ref("compute_ms");
    const roiMetricKey = ref("e2e_ms");
    const roiGroupMode = ref("variable");
    const roiBarMode = ref("total");
    const roiSort = ref({ key: "totalGainMs", dir: "desc" });
    const nodeMetricSort = ref({ key: "created_at", dir: "asc" });
    const roiHighlightEdgeIds = ref([]);
    const nodeCompareTargetId = ref("");
    const nodeCompareSaving = ref(false);
    const nodeNameDraft = ref("");
    const nodeNameOriginal = ref("");
    const nodeNameSaving = ref(false);
    const nodeAttachmentUploading = ref(false);
    const nodeAttachmentDeletingId = ref("");
    const view = ref({ scale: 1, tx: 36, ty: 36 });
    const addForm = ref({
      parent_job_id: "",
      child_job_id: "",
      title: "",
      description: "",
      variablesText: "",
    });
    const edgeDraft = ref({ title: "", description: "", variables: [], variablesText: "" });
    const emptyVariableQuick = () => ({ name: "", from: "", to: "" });
    const edgeVariableQuick = ref(emptyVariableQuick());
    const addVariableQuick = ref(emptyVariableQuick());
    let panState = null;
    let dragState = null;
    let edgeDragState = null;
    let edgeResizeState = null;
    let lineageChartInst = null;
    let lineageChartTooltipEl = null;
    let lineageChartBuildToken = 0;

    const NODE_W = 224;
    const NODE_H = 136;
    const NODE_MIN_W = 200;
    const NODE_MIN_H = 118;
    const NODE_MAX_W = 900;
    const NODE_MAX_H = 640;
    const SIBLING_GAP = 88;
    const EDGE_LABEL_GAP = 30;
    const LAYER_GAP = 60;
    const EDGE_LABEL_W = 320;
    const EDGE_LABEL_H = 58;
    const EDGE_LABEL_MIN_W = 210;
    const EDGE_LABEL_MIN_H = 58;
    const EDGE_LABEL_MAX_W = 900;
    const EDGE_LABEL_MAX_H = 420;
    const EDGE_LABEL_AUTO_MAX_W = 520;
    const NODE_COLLISION_GAP_X = 28;
    const NODE_COLLISION_GAP_Y = 32;
    const NODE_EDGE_GAP_Y = Math.max(EDGE_LABEL_H + EDGE_LABEL_GAP * 2 + 22, 120);
    const MAX_LINEAGE_BRANCHES = 24;
    const LINEAGE_BRANCH_COLORS = [
      "#6366f1", "#10b981", "#f59e0b", "#0ea5e9",
      "#ef4444", "#8b5cf6", "#14b8a6", "#f97316",
    ];
    const LINEAGE_METRIC_DEFS = [
      { key: "compute_ms", label: "计算耗时", unit: "ms", beginAtZero: false },
      { key: "e2e_ms", label: "端到端耗时", unit: "ms", beginAtZero: false },
      { key: "kernel_count", label: "Kernel 数", unit: "count", beginAtZero: true },
      { key: "aten_ops_ms", label: "ATen 耗时", unit: "ms", beginAtZero: false },
      { key: "aten_ops_count", label: "ATen 操作数", unit: "count", beginAtZero: true },
      { key: "step_dur_ms", label: "Step 耗时", unit: "ms", beginAtZero: false },
    ];
    const ROI_METRIC_DEFS = [
      { key: "e2e_ms", label: "端到端耗时" },
      { key: "compute_ms", label: "计算耗时" },
    ];
    const roundLayout = value => Math.round(Number(value || 0) * 10) / 10;
    const hasLayoutNumber = value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
    const charWidth = char => /[\u2E80-\u9FFF]/.test(char) ? 13 : /[A-Z0-9]/.test(char) ? 8 : /[a-z]/.test(char) ? 7 : 6.5;
    const textWidth = text => Array.from(String(text || "")).reduce((sum, char) => sum + charWidth(char), 0);
    const looksLikeFileNameText = value => /\.(json|gz|zip|tgz|tar|trace|pt)(\.|$)/i.test(String(value || ""));
    const shortNodeIdText = id => String(id || "").slice(0, 8);
    const nodeTitleText = node => {
      const label = String(node?.label || "").trim();
      const fileName = String(node?.file_a_name || "").trim();
      if (label && label !== fileName && !looksLikeFileNameText(label)) return label;
      return `Job ${shortNodeIdText(node?.id)}`;
    };
    const compactMsText = value => Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)} ms` : "-";
    const compactCountText = value => {
      if (!Number.isFinite(Number(value))) return "-";
      const number = Number(value);
      return Number.isInteger(number) ? String(number) : number.toFixed(1);
    };
    const compactSignedMsText = value => {
      if (!Number.isFinite(Number(value))) return "-";
      const number = Number(value);
      return `${number > 0 ? "+" : ""}${number.toFixed(2)} ms`;
    };
    const compactSignedCountText = value => {
      if (!Number.isFinite(Number(value))) return "-";
      const number = Number(value);
      const text = Number.isInteger(number) ? String(number) : number.toFixed(1);
      return `${number > 0 ? "+" : ""}${text}`;
    };
    const compactPctDeltaText = value => {
      const number = Number(value);
      if (!Number.isFinite(number)) return "";
      if (number === 0) return "0.0%";
      return `${number < 0 ? "▼" : "▲"}${Math.abs(number).toFixed(1)}%`;
    };
    const autoTextBoxSize = (text, options = {}) => {
      const {
        minWidth = 160,
        maxWidth = 520,
        minHeight = 44,
        maxHeight = 420,
        paddingX = 12,
        paddingY = 10,
        lineHeight = 16,
        extraWidth = 0,
        extraHeight = 0,
      } = options;
      const lines = String(text || "").split("\n");
      const longest = Math.max(0, ...lines.map(line => textWidth(line)));
      const width = Math.max(minWidth, Math.min(maxWidth, roundLayout(longest + paddingX * 2 + extraWidth)));
      const contentWidth = Math.max(40, width - paddingX * 2 - extraWidth);
      const visualLines = lines.reduce((sum, line) => sum + Math.max(1, Math.ceil(textWidth(line) / contentWidth)), 0);
      const height = Math.max(minHeight, Math.min(maxHeight, roundLayout(visualLines * lineHeight + paddingY * 2 + extraHeight)));
      return { width, height };
    };
    const clampNodeWidth = value => Math.max(NODE_MIN_W, Math.min(NODE_MAX_W, roundLayout(hasLayoutNumber(value) ? value : NODE_W)));
    const clampNodeHeight = value => Math.max(NODE_MIN_H, Math.min(NODE_MAX_H, roundLayout(hasLayoutNumber(value) ? value : NODE_H)));
    const nodeAutoSize = node => {
      const nodePaddingX = 14;
      const rowGap = 10;
      const titleWidth = 10 + 8 + textWidth(nodeTitleText(node)) * 1.15 + (node?.status !== "done" ? 64 : 0);
      const parentEdge = edges.value.find(edge => edge.child_job_id === node?.id);
      const parent = parentEdge ? nodes.value.find(item => item.id === parentEdge.parent_job_id) : null;
      const metricValueText = (key, kind = "ms") => (kind === "count" ? compactCountText(node?.[key]) : compactMsText(node?.[key]));
      const metricDelta = key => {
        const childValue = Number(node?.[key]);
        const parentValue = Number(parent?.[key]);
        if (!Number.isFinite(childValue) || !Number.isFinite(parentValue)) return null;
        return childValue - parentValue;
      };
      const metricPctDeltaText = key => {
        const delta = metricDelta(key);
        const parentValue = Number(parent?.[key]);
        if (delta === null || !Number.isFinite(parentValue) || parentValue === 0) return "";
        return compactPctDeltaText(Math.round((delta / Math.abs(parentValue)) * 1000) / 10);
      };
      const metricChipText = (key, label, kind = "ms") => {
        if (node?.status !== "done") return "";
        const delta = metricDelta(key);
        if (delta === null) return "";
        if (kind === "count") {
          const value = Math.abs(delta).toFixed(delta % 1 ? 1 : 0);
          return delta === 0 ? `${label} 0` : `${label} ${delta < 0 ? "▼" : "▲"}${value}`;
        }
        const deltaText = metricPctDeltaText(key);
        return deltaText ? `${label} ${deltaText}` : "";
      };
      const chipWidth = text => text ? Math.max(44, textWidth(text) + 24) : 0;
      const primaryNumber = Number.isFinite(Number(node?.compute_ms)) ? Number(node.compute_ms).toFixed(2) : "-";
      const primaryWidth = textWidth(primaryNumber) * 1.45
        + textWidth("ms · compute time")
        + chipWidth(metricPctDeltaText("compute_ms"))
        + 18;
      const secondaryRowWidth = (label, key, kind = "ms") => {
        const labelWidth = Math.max(48, textWidth(label) + 4);
        const valueWidth = textWidth(metricValueText(key, kind)) + 8;
        return labelWidth + valueWidth + chipWidth(metricChipText(key, label.toLowerCase(), kind)) + 16;
      };
      const secondaryWidth = Math.max(
        secondaryRowWidth("E2E", "e2e_ms", "ms"),
        secondaryRowWidth("Kernel", "kernel_count", "count"),
      );
      const statsWidth = nodePaddingX * 2 + Math.max(titleWidth, primaryWidth, secondaryWidth, 164);
      const statsHeight = 76;
      const contentHeight = 15 + 18 + rowGap + statsHeight + 18;
      return {
        width: clampNodeWidth(Math.max(NODE_W, titleWidth, statsWidth)),
        height: clampNodeHeight(Math.max(NODE_H, contentHeight)),
      };
    };
    const nodeWidth = node => nodeAutoSize(node).width;
    const nodeHeight = node => nodeAutoSize(node).height;
    const clampEdgeLabelWidth = value => Math.max(EDGE_LABEL_MIN_W, Math.min(EDGE_LABEL_MAX_W, roundLayout(value || EDGE_LABEL_W)));
    const clampEdgeLabelHeight = value => Math.max(EDGE_LABEL_MIN_H, Math.min(EDGE_LABEL_MAX_H, roundLayout(value || EDGE_LABEL_H)));
    const edgeLabelAutoSize = edge => autoTextBoxSize(edgeLabelSizingText(edge), {
      minWidth: EDGE_LABEL_MIN_W,
      maxWidth: EDGE_LABEL_AUTO_MAX_W,
      minHeight: EDGE_LABEL_MIN_H,
      maxHeight: EDGE_LABEL_MAX_H,
      paddingX: 12,
      paddingY: 10,
      lineHeight: 16.2,
      extraWidth: 96,
      extraHeight: 8,
    });
    const edgeLabelWidth = edge => clampEdgeLabelWidth(hasLayoutNumber(edge?.label_width) ? edge.label_width : edgeLabelAutoSize(edge).width);
    const edgeLabelHeight = edge => clampEdgeLabelHeight(hasLayoutNumber(edge?.label_height) ? edge.label_height : edgeLabelAutoSize(edge).height);
    const edgeArrowPath = (tipX, tipY, fromX, fromY) => {
      const dx = tipX - fromX;
      const dy = tipY - fromY;
      const length = Math.hypot(dx, dy) || 1;
      const ux = dx / length;
      const uy = dy / length;
      const px = -uy;
      const py = ux;
      const headLength = 11;
      const headWidth = 5.5;
      const baseX = tipX - ux * headLength;
      const baseY = tipY - uy * headLength;
      const leftX = baseX + px * headWidth;
      const leftY = baseY + py * headWidth;
      const rightX = baseX - px * headWidth;
      const rightY = baseY - py * headWidth;
      const curveX = tipX - ux * (headLength * 0.72);
      const curveY = tipY - uy * (headLength * 0.72);
      const point = (x, y) => `${roundLayout(x)} ${roundLayout(y)}`;
      return `M ${point(tipX, tipY)} L ${point(leftX, leftY)} Q ${point(curveX, curveY)} ${point(rightX, rightY)} Z`;
    };

    const projectId = computed(() => String(route.params.pid || ""));
    const projectName = computed(() => {
      if (projectMeta.value?.name) return projectMeta.value.name;
      const found = projects.value.find(project => project.id === projectId.value);
      return found?.name || activeHistoryProject.value?.label || projectId.value || "项目";
    });
    const normalizeProjectMetricTarget = value => {
      if (value === null || value === undefined || value === "") return null;
      const number = Number(String(value).replace(/,/g, "").trim());
      return Number.isFinite(number) && number > 0 ? number : null;
    };
    const formatProjectMetricTargetDraft = value => {
      const number = normalizeProjectMetricTarget(value);
      return number === null ? "" : String(Math.round(number * 100) / 100);
    };
    const projectMetricTargets = computed(() => {
      const localTargets = projectMeta.value?.metric_targets;
      const project = projects.value.find(item => item.id === projectId.value);
      const rawTargets = localTargets && typeof localTargets === "object" ? localTargets : (project?.metric_targets || {});
      const targets = {};
      Object.entries(rawTargets || {}).forEach(([key, value]) => {
        const target = normalizeProjectMetricTarget(value);
        if (target !== null) targets[key] = target;
      });
      const legacyComputeTarget = normalizeProjectMetricTarget(
        projectMeta.value && Object.prototype.hasOwnProperty.call(projectMeta.value, "compute_target_ms")
          ? projectMeta.value.compute_target_ms
          : project?.compute_target_ms,
      );
      if (legacyComputeTarget !== null && targets.compute_ms === undefined) {
        targets.compute_ms = legacyComputeTarget;
      }
      return targets;
    });

    const lineageMetricValue = (node, key) => {
      if (node?.[key] === null || node?.[key] === undefined || node?.[key] === "") return null;
      const number = Number(node?.[key]);
      return Number.isFinite(number) ? number : null;
    };
    const lineageMetricOptions = computed(() => {
      const available = LINEAGE_METRIC_DEFS.filter(def =>
        def.key === "compute_ms" || nodes.value.some(node => lineageMetricValue(node, def.key) !== null)
      );
      return available.length ? available : [LINEAGE_METRIC_DEFS[0]];
    });
    const currentLineageMetricDef = computed(() =>
      lineageMetricOptions.value.find(item => item.key === lineageMetricKey.value)
      || LINEAGE_METRIC_DEFS.find(item => item.key === lineageMetricKey.value)
      || LINEAGE_METRIC_DEFS[0]
    );
    const currentMetricTargetValue = computed(() =>
      projectMetricTargets.value[currentLineageMetricDef.value.key] ?? null
    );
    const currentMetricTargetTitle = computed(() => `${currentLineageMetricDef.value.label}目标`);
    const currentMetricTargetValueLabel = computed(() =>
      currentMetricTargetValue.value === null ? "未设置" : formatLineageMetricValue(currentMetricTargetValue.value, currentLineageMetricDef.value)
    );
    const projectMetricTargetDirty = computed(() => {
      const draft = String(projectMetricTargetDraft.value || "").trim();
      if (!draft) return currentMetricTargetValue.value !== null;
      const draftValue = normalizeProjectMetricTarget(draft);
      if (draftValue === null) return true;
      return currentMetricTargetValue.value === null || Math.abs(draftValue - currentMetricTargetValue.value) > 1e-6;
    });
    const lineageMetricTargetLineValue = computed(() => currentMetricTargetValue.value);
    const refreshProjectMetricTargetDraft = () => {
      projectMetricTargetDraft.value = formatProjectMetricTargetDraft(currentMetricTargetValue.value);
    };
    const sortLineageNodeIds = (ids, nodeMap) => ids.slice().sort((a, b) => {
      const left = nodeMap.get(a) || {};
      const right = nodeMap.get(b) || {};
      return String(left.created_at || "").localeCompare(String(right.created_at || ""))
        || nodeTitleText(left).localeCompare(nodeTitleText(right))
        || String(a).localeCompare(String(b));
    });
    const lineageTopology = computed(() => {
      const nodeMap = new Map();
      nodes.value.forEach(node => {
        if (node?.id) nodeMap.set(String(node.id), node);
      });
      const ids = sortLineageNodeIds(Array.from(nodeMap.keys()), nodeMap);
      const childrenOf = new Map(ids.map(id => [id, []]));
      const parentsOf = new Map(ids.map(id => [id, []]));
      edges.value.forEach(edge => {
        const parentId = String(edge?.parent_job_id || "");
        const childId = String(edge?.child_job_id || "");
        if (!nodeMap.has(parentId) || !nodeMap.has(childId)) return;
        childrenOf.get(parentId).push(childId);
        parentsOf.get(childId).push(parentId);
      });
      ids.forEach(id => {
        childrenOf.set(id, sortLineageNodeIds(childrenOf.get(id) || [], nodeMap));
        parentsOf.set(id, sortLineageNodeIds(parentsOf.get(id) || [], nodeMap));
      });
      let roots = ids.filter(id => !(parentsOf.get(id) || []).length);
      if (!roots.length) roots = ids;
      roots = sortLineageNodeIds(roots, nodeMap);
      const indegree = new Map(ids.map(id => [id, (parentsOf.get(id) || []).length]));
      const generation = new Map(ids.map(id => [id, 0]));
      const queue = [...roots];
      const processed = new Set();
      while (queue.length) {
        const id = queue.shift();
        processed.add(id);
        for (const childId of childrenOf.get(id) || []) {
          generation.set(childId, Math.max(generation.get(childId) || 0, (generation.get(id) || 0) + 1));
          const nextInDegree = (indegree.get(childId) || 0) - 1;
          indegree.set(childId, nextInDegree);
          if (nextInDegree === 0) queue.push(childId);
        }
      }
      ids.forEach(id => {
        if (!processed.has(id)) generation.set(id, generation.get(id) || 0);
      });
      return { nodeMap, ids, childrenOf, parentsOf, roots, generation };
    });
    const lineageBranches = computed(() => {
      const graph = lineageTopology.value;
      const branches = [];
      let truncated = false;
      const addBranch = path => {
        if (branches.length >= MAX_LINEAGE_BRANCHES) {
          truncated = true;
          return;
        }
        branches.push(path);
      };
      const visit = (id, path, seen) => {
        if (truncated) return;
        const nextPath = [...path, id];
        const children = (graph.childrenOf.get(id) || []).filter(childId => !seen.has(childId));
        if (!children.length) {
          addBranch(nextPath);
          return;
        }
        for (const childId of children) {
          seen.add(childId);
          visit(childId, nextPath, seen);
          seen.delete(childId);
          if (truncated) break;
        }
      };
      for (const rootId of graph.roots) {
        visit(rootId, [], new Set([rootId]));
        if (truncated) break;
      }
      return { branches, truncated };
    });
    const formatLineageMetricValue = (value, metricDef = currentLineageMetricDef.value) => {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
      const number = Number(value);
      if (metricDef.unit === "count") return Number.isInteger(number) ? String(number) : number.toFixed(1);
      return `${number.toFixed(2)}${metricDef.unit ? ` ${metricDef.unit}` : ""}`;
    };
    const formatLineageSignedValue = (value, metricDef = currentLineageMetricDef.value) => {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
      const number = Number(value);
      const sign = number > 0 ? "+" : "";
      if (metricDef.unit === "count") {
        const text = Number.isInteger(number) ? String(number) : number.toFixed(1);
        return `${sign}${text}`;
      }
      return `${sign}${number.toFixed(2)}${metricDef.unit ? ` ${metricDef.unit}` : ""}`;
    };
    const formatLineagePct = value => {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "";
      const number = Number(value);
      return `${number > 0 ? "+" : ""}${number.toFixed(1)}%`;
    };
    const formatLineageAxisTick = (value, metricDef = currentLineageMetricDef.value) => {
      const number = Number(value);
      if (!Number.isFinite(number)) return value;
      const trimZeros = text => {
        const trimmed = text.replace(/\.?0+$/, "");
        return trimmed && trimmed !== "-" ? trimmed : "0";
      };
      if (metricDef.unit === "count") {
        return Number.isInteger(number) ? String(number) : number.toFixed(1).replace(/\.0$/, "");
      }
      const abs = Math.abs(number);
      const decimals = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
      return trimZeros(number.toFixed(decimals));
    };
    const lineageChartModel = computed(() => {
      const graph = lineageTopology.value;
      const metricDef = currentLineageMetricDef.value;
      const branchNameCounts = new Map();
      const datasets = lineageBranches.value.branches.map((path, index) => {
        const leaf = graph.nodeMap.get(path[path.length - 1]);
        const baseName = shortChartLabel(nodeTitleText(leaf) || `分支 ${index + 1}`, 34);
        const branchNameCount = (branchNameCounts.get(baseName) || 0) + 1;
        branchNameCounts.set(baseName, branchNameCount);
        const label = branchNameCount === 1 ? baseName : `${baseName} · ${branchNameCount}`;
        const color = LINEAGE_BRANCH_COLORS[index % LINEAGE_BRANCH_COLORS.length];
        const data = path.map((nodeId, pointIndex) => {
          const node = graph.nodeMap.get(nodeId);
          const parentId = path[pointIndex - 1] || "";
          const parentNode = parentId ? graph.nodeMap.get(parentId) : null;
          const rawValue = lineageMetricValue(node, metricDef.key);
          const parentValue = parentNode ? lineageMetricValue(parentNode, metricDef.key) : null;
          const delta = rawValue !== null && parentValue !== null ? rawValue - parentValue : null;
          const deltaPct = delta !== null && parentValue !== null && parentValue !== 0
            ? Math.round((delta / Math.abs(parentValue)) * 1000) / 10
            : null;
          return {
            x: graph.generation.get(nodeId) || 0,
            y: rawValue,
            nodeId,
            label: nodeTitleText(node),
            status: node?.status || "",
            fileName: String(node?.file_a_name || ""),
            rawValue,
            parentNodeId: parentId,
            parentLabel: parentNode ? nodeTitleText(parentNode) : "",
            delta,
            deltaPct,
          };
        });
        if (!data.some(point => point.rawValue !== null)) return null;
        return {
          label,
          data,
          borderColor: color,
          backgroundColor: chartAlphaColor(color, 0.12),
          borderWidth: 2.6,
          hoverBorderWidth: 3.2,
          borderCapStyle: "round",
          borderJoinStyle: "round",
          cubicInterpolationMode: "monotone",
          pointBackgroundColor: color,
          pointHoverBackgroundColor: color,
          pointBorderColor: "#ffffff",
          pointHoverBorderColor: color,
          pointBorderWidth: 2,
          pointHoverBorderWidth: 3,
          pointRadius: ctx => ctx.raw?.rawValue === null ? 0 : 4.5,
          pointHoverRadius: ctx => ctx.raw?.rawValue === null ? 0 : 6.5,
          pointHitRadius: 10,
          tension: 0.25,
          spanGaps: true,
        };
      }).filter(Boolean);
      return { datasets, truncated: lineageBranches.value.truncated };
    });
    const lineageMetricHasData = computed(() =>
      nodes.value.some(node => lineageMetricValue(node, currentLineageMetricDef.value.key) !== null)
    );
    const lineageChartEmptyText = computed(() => {
      if (!edges.value.length) return "先建立优化关系后查看代际趋势";
      if (!nodes.value.length) return "暂无实验节点";
      if (!lineageMetricHasData.value || !lineageChartModel.value.datasets.length) return "该指标暂无数据，建议换指标";
      return "";
    });
    const lineageChartWarning = computed(() =>
      lineageChartModel.value.truncated
        ? `分支过多，仅显示前 ${MAX_LINEAGE_BRANCHES} 条；可在画布视图查看全图`
        : ""
    );
    const lineageChartDataPoints = computed(() =>
      lineageChartModel.value.datasets
        .flatMap(dataset => dataset.data || [])
        .filter(point => point?.rawValue !== null && point?.rawValue !== undefined)
    );
    const lineageChartAxisBounds = computed(() => {
      const points = lineageChartDataPoints.value;
      if (!points.length) return {};
      const metricDef = currentLineageMetricDef.value;
      const xValues = points.map(point => Number(point.x)).filter(Number.isFinite);
      const yValues = points.map(point => Number(point.rawValue)).filter(Number.isFinite);
      const targetValue = lineageMetricTargetLineValue.value;
      const yDomainValues = targetValue === null ? yValues : [...yValues, targetValue];
      const boundsFor = (values, options = {}) => {
        if (!values.length) return {};
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const center = (minValue + maxValue) / 2;
        const rawSpan = Math.max(maxValue - minValue, options.fallbackSpan || 1);
        const paddedSpan = rawSpan * (options.paddingFactor || 1.25);
        const span = paddedSpan;
        let min = center - span / 2;
        let max = center + span / 2;
        if (options.clampMinZero && minValue >= 0 && min < 0) {
          min = 0;
          max = Math.max(max, span);
          if (max <= min) max = min + rawSpan;
        }
        return { min, max };
      };
      const yBoundsFor = values => {
        if (!values.length) return {};
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        let min = minValue >= 0 ? minValue * 0.8 : minValue * 1.25;
        let max = maxValue >= 0 ? maxValue * 1.25 : maxValue * 0.8;
        if (max <= min) {
          const fallbackSpan = metricDef.unit === "count" ? 2 : Math.max(1, Math.abs(maxValue || minValue || 1) * 0.25);
          max = min + fallbackSpan;
        }
        const span = Math.max(1e-9, max - min);
        const roughStep = span / 5;
        const exponent = 10 ** Math.floor(Math.log10(roughStep));
        const fraction = roughStep / exponent;
        const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
        const step = Math.max(metricDef.unit === "count" ? 1 : 0, niceFraction * exponent);
        min = Math.floor(min / step) * step;
        max = Math.ceil(max / step) * step;
        const decimals = step >= 1 ? 0 : Math.min(6, Math.ceil(Math.abs(Math.log10(step))) + 1);
        min = Number(min.toFixed(decimals));
        max = Number(max.toFixed(decimals));
        if (max <= min) max = Number((min + step).toFixed(decimals));
        return { min, max };
      };
      return {
        x: boundsFor(xValues, { fallbackSpan: 2, paddingFactor: 1.25, clampMinZero: true }),
        y: yBoundsFor(yDomainValues),
      };
    });

    const sortedGraphNodes = computed(() => {
      const byId = new Map(nodes.value.map(node => [node.id, { ...node }]));
      const graphNodes = Array.from(byId.values());
      const indegree = {};
      const children = {};
      graphNodes.forEach(node => {
        indegree[node.id] = 0;
        children[node.id] = [];
      });
      edges.value.forEach(edge => {
        if (!(edge.parent_job_id in indegree) || !(edge.child_job_id in indegree)) return;
        indegree[edge.child_job_id] += 1;
        children[edge.parent_job_id].push(edge.child_job_id);
      });
      const queue = graphNodes
        .filter(node => indegree[node.id] === 0)
        .sort((a, b) => String(a.created_at || "").localeCompare(String(b.created_at || "")))
        .map(node => node.id);
      const layer = {};
      queue.forEach(id => { layer[id] = 0; });
      while (queue.length) {
        const id = queue.shift();
        for (const childId of children[id] || []) {
          layer[childId] = Math.max(layer[childId] || 0, (layer[id] || 0) + 1);
          indegree[childId] -= 1;
          if (indegree[childId] === 0) queue.push(childId);
        }
      }
      graphNodes.forEach(node => {
        if (layer[node.id] === undefined) layer[node.id] = 0;
      });
      const byLayer = {};
      graphNodes.forEach(node => {
        const key = layer[node.id] || 0;
        if (!byLayer[key]) byLayer[key] = [];
        byLayer[key].push(node);
      });
      const layerHeights = {};
      Object.entries(byLayer).forEach(([key, items]) => {
        layerHeights[key] = Math.max(NODE_H, ...items.map(node => nodeHeight(node)));
      });
      const edgeLabelHeightsByLayer = {};
      edges.value.forEach(edge => {
        const parentLayer = layer[edge.parent_job_id];
        const childLayer = layer[edge.child_job_id];
        if (!Number.isFinite(Number(parentLayer)) || !Number.isFinite(Number(childLayer))) return;
        if (Number(childLayer) !== Number(parentLayer) + 1) return;
        edgeLabelHeightsByLayer[parentLayer] = Math.max(
          edgeLabelHeightsByLayer[parentLayer] || 0,
          edgeLabelHeight(edge),
        );
      });
      const layerTops = {};
      const layerKeys = Object.keys(byLayer).map(Number).sort((a, b) => a - b);
      layerKeys.forEach((key, index) => {
        if (index === 0) {
          layerTops[key] = 0;
          return;
        }
        const prevKey = layerKeys[index - 1];
        const labelSpace = Math.max(EDGE_LABEL_H, edgeLabelHeightsByLayer[prevKey] || 0);
        layerTops[key] = layerTops[prevKey] + (layerHeights[prevKey] || NODE_H) + labelSpace + EDGE_LABEL_GAP * 2 + LAYER_GAP;
      });
      Object.entries(byLayer).forEach(([key, items]) => {
        items.sort((a, b) => String(a.created_at || "").localeCompare(String(b.created_at || "")));
        let cursorX = 0;
        items.forEach((node, index) => {
          const pinned = Number(node.pinned) === 1 && Number.isFinite(Number(node.x)) && Number.isFinite(Number(node.y));
          node.x = pinned ? Number(node.x) : cursorX;
          const layerTop = layerTops[Number(key)] || 0;
          node.y = pinned ? Number(node.y) : layerTop;
          node.scale = 1;
          node.pinned = pinned ? 1 : 0;
          cursorX = Math.max(cursorX, node.x + nodeWidth(node) + SIBLING_GAP);
        });
      });
      return graphNodes;
    });

    const displayNodes = computed(() => sortedGraphNodes.value);
    const nodeById = computed(() => Object.fromEntries(displayNodes.value.map(node => [node.id, node])));
    const canvasSize = computed(() => {
      const maxX = Math.max(820, ...displayNodes.value.map(node => node.x + nodeWidth(node) + 120), 820);
      const maxY = Math.max(520, ...displayNodes.value.map(node => node.y + nodeHeight(node) + 120), 520);
      return { width: maxX, height: maxY };
    });
    const bestNodeId = computed(() => {
      const valueFor = node => {
        const compute = Number(node.compute_ms);
        if (Number.isFinite(compute)) return compute;
        const e2e = Number(node.e2e_ms);
        return Number.isFinite(e2e) ? e2e : null;
      };
      const doneNodes = displayNodes.value
        .filter(node => node.status === "done" && valueFor(node) !== null)
        .sort((a, b) => valueFor(a) - valueFor(b));
      return doneNodes[0]?.id || "";
    });
    const selectedEdge = computed(() => edges.value.find(edge => edge.id === selectedEdgeId.value) || null);
    const selectedEdgePerfNote = computed(() => String(selectedEdge.value?.perf?.notes || "").trim());
    const selectedNode = computed(() => nodeById.value[selectedNodeId.value] || null);
    const nodeTopKernels = node => (node?.top_kernels || []).slice(0, 5);
    const nodeDetailRows = node => {
      if (!node) return [];
      const rows = [
        nodeMetricDetailRow(node, "e2e_ms", "e2e", "ms", "time"),
        nodeMetricDetailRow(node, "kernel_count", "Kernel", "count", "count"),
        nodeMetricDetailRow(node, "compute_ms", "Compute", "ms", "time"),
        nodeMetricDetailRow(node, "aten_ops_count", "aten_ops", "count", "count"),
      ];
      return rows.concat(nodeTopKernels(node).map((kernel, index) => hotKernelDetailRow(node, kernel, index)));
    };
    const selectedNodeTopKernels = computed(() => nodeTopKernels(selectedNode.value));
    const selectedNodeDetailRows = computed(() => nodeDetailRows(selectedNode.value));
    const selectedNodeAttachments = computed(() =>
      Array.isArray(selectedNode.value?.attachments) ? selectedNode.value.attachments : []
    );
    const lineageHoverMetricDefs = () => [
      { ...LINEAGE_METRIC_DEFS.find(item => item.key === "compute_ms"), label: "compute time" },
      { ...LINEAGE_METRIC_DEFS.find(item => item.key === "e2e_ms"), label: "e2e time" },
      { ...LINEAGE_METRIC_DEFS.find(item => item.key === "kernel_count"), label: "kernel num" },
      { ...LINEAGE_METRIC_DEFS.find(item => item.key === "aten_ops_count"), label: "aten ops" },
    ].filter(Boolean);
    const lineageHoverMetricRows = node => {
      if (!node) return [];
      return lineageHoverMetricDefs().map(metricDef => ({
        key: metricDef.key,
        label: metricDef.label,
        tone: metricDef.unit === "count" ? "count" : "time",
        value: formatLineageMetricValue(lineageMetricValue(node, metricDef.key), metricDef),
        deltaText: "",
        deltaClass: "neutral",
      }));
    };
    const hasSelection = computed(() => Boolean(selectedNodeId.value || selectedEdgeId.value || roiHighlightEdgeIds.value.length));
    const lineage = computed(() => {
      const nodeIds = new Set();
      const edgeIds = new Set();
      if (roiHighlightEdgeIds.value.length) {
        const highlighted = new Set(roiHighlightEdgeIds.value);
        edges.value.forEach(edge => {
          if (!highlighted.has(edge.id)) return;
          edgeIds.add(edge.id);
          nodeIds.add(edge.parent_job_id);
          nodeIds.add(edge.child_job_id);
        });
        return { nodeIds, edgeIds };
      }
      if (selectedEdge.value) {
        edgeIds.add(selectedEdge.value.id);
        nodeIds.add(selectedEdge.value.parent_job_id);
        nodeIds.add(selectedEdge.value.child_job_id);
        return { nodeIds, edgeIds };
      }
      const root = selectedNodeId.value;
      if (!root) return { nodeIds, edgeIds };
      nodeIds.add(root);
      const walk = (startIds, direction) => {
        const queue = [...startIds];
        const seen = new Set(queue);
        while (queue.length) {
          const id = queue.shift();
          edges.value.forEach(edge => {
            const match = direction === "down" ? edge.parent_job_id === id : edge.child_job_id === id;
            if (!match) return;
            const nextId = direction === "down" ? edge.child_job_id : edge.parent_job_id;
            edgeIds.add(edge.id);
            nodeIds.add(nextId);
            if (!seen.has(nextId)) {
              seen.add(nextId);
              queue.push(nextId);
            }
          });
        }
      };
      walk([root], "down");
      walk([root], "up");
      return { nodeIds, edgeIds };
    });
    const edgePaths = computed(() => edges.value
      .map(edge => {
        const parent = nodeById.value[edge.parent_job_id];
        const child = nodeById.value[edge.child_job_id];
        if (!parent || !child) return null;
        const sx = parent.x + nodeWidth(parent) / 2;
        const sy = parent.y + nodeHeight(parent);
        const tx = child.x + nodeWidth(child) / 2;
        const ty = child.y;
        const midY = sy + (ty - sy) / 2;
        const labelWidth = edgeLabelWidth(edge);
        const labelHeight = edgeLabelHeight(edge);
        const defaultLabelX = (sx + tx) / 2 - labelWidth / 2;
        const defaultLabelY = (sy + ty) / 2 - labelHeight / 2;
        const labelX = hasLayoutNumber(edge.label_x) ? Number(edge.label_x) : defaultLabelX;
        const labelY = hasLayoutNumber(edge.label_y) ? Number(edge.label_y) : defaultLabelY;
        const anchorX = (sx + tx) / 2;
        const anchorY = (sy + ty) / 2;
        const labelCenterX = labelX + labelWidth / 2;
        const labelCenterY = labelY + labelHeight / 2;
        const dx = anchorX - labelCenterX;
        const dy = anchorY - labelCenterY;
        let connectorD = "";
        if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
          let endX = labelCenterX;
          let endY = labelCenterY;
          if (Math.abs(dx) * labelHeight > Math.abs(dy) * labelWidth && Math.abs(dx) > 0) {
            endX = labelCenterX + Math.sign(dx) * labelWidth / 2;
            endY = labelCenterY + dy * (labelWidth / 2) / Math.abs(dx);
          } else if (Math.abs(dy) > 0) {
            endY = labelCenterY + Math.sign(dy) * labelHeight / 2;
            endX = labelCenterX + dx * (labelHeight / 2) / Math.abs(dy);
          }
          connectorD = `M ${anchorX} ${anchorY} L ${endX} ${endY}`;
        }
        return {
          edge,
          d: `M ${sx} ${sy} C ${sx} ${midY} ${tx} ${midY} ${tx} ${ty}`,
          arrowD: edgeArrowPath(tx, ty, tx, midY),
          connectorD,
          labelX,
          labelY,
          labelWidth,
          labelHeight,
        };
      })
      .filter(Boolean));
    const candidateOptions = computed(() => {
      const map = new Map();
      [...candidateJobs.value, ...nodes.value, ...unconnected.value].forEach(job => {
        if (job?.id && job.mode === "single") map.set(job.id, job);
      });
      return Array.from(map.values());
    });
    const graphBounds = () => {
      const boxes = displayNodes.value.map(node => ({
        minX: node.x,
        minY: node.y,
        maxX: node.x + nodeWidth(node),
        maxY: node.y + nodeHeight(node),
      }));
      edgePaths.value.forEach(item => {
        boxes.push({
          minX: item.labelX,
          minY: item.labelY,
          maxX: item.labelX + item.labelWidth,
          maxY: item.labelY + item.labelHeight,
        });
      });
      if (!boxes.length) return null;
      return {
        minX: Math.min(...boxes.map(box => box.minX)),
        minY: Math.min(...boxes.map(box => box.minY)),
        maxX: Math.max(...boxes.map(box => box.maxX)),
        maxY: Math.max(...boxes.map(box => box.maxY)),
      };
    };
    const fitGraphToViewport = () => {
      const bounds = graphBounds();
      const viewport = viewportRef.value;
      if (!bounds || !viewport) {
        view.value = { scale: 1, tx: 36, ty: 36 };
        return;
      }
      const viewportWidth = viewport.clientWidth || 900;
      const viewportHeight = viewport.clientHeight || 620;
      const paddingX = 96;
      const paddingY = 86;
      const graphWidth = Math.max(1, bounds.maxX - bounds.minX);
      const graphHeight = Math.max(1, bounds.maxY - bounds.minY);
      const fitScale = Math.min(
        1.12,
        Math.max(0.48, (viewportWidth - paddingX * 2) / graphWidth, 0.48),
        Math.max(0.48, (viewportHeight - paddingY * 2) / graphHeight, 0.48),
      );
      view.value = {
        scale: Math.round(fitScale * 100) / 100,
        tx: roundLayout((viewportWidth - graphWidth * fitScale) / 2 - bounds.minX * fitScale),
        ty: roundLayout(Math.max(24, (viewportHeight - graphHeight * fitScale) / 2 - bounds.minY * fitScale)),
      };
    };
    const fitGraphAfterPaint = async () => {
      await nextTick();
      fitGraphToViewport();
    };

    const cleanVariables = variables => (variables || [])
      .map(item => ({
        name: String(item.name || "").trim(),
        from: String(item.from ?? "").trim(),
        to: String(item.to ?? "").trim(),
      }))
      .filter(item => item.name);
    const normalizeVariableNameKey = value => String(value || "").trim().toLocaleLowerCase();
    const normalizeVariableValue = value => String(value ?? "").trim();
    const incrementCount = (map, key, inc = 1) => {
      if (!key) return;
      map.set(key, (map.get(key) || 0) + inc);
    };
    const mostCommonText = counts => {
      let best = "";
      let bestCount = -1;
      counts.forEach((count, text) => {
        if (count > bestCount || (count === bestCount && String(text).localeCompare(best) < 0)) {
          best = text;
          bestCount = count;
        }
      });
      return best;
    };
    const variableNameOptions = computed(() => {
      const byKey = new Map();
      edges.value.forEach(edge => {
        cleanVariables(edge?.variables || []).forEach(variable => {
          const key = normalizeVariableNameKey(variable.name);
          if (!key) return;
          if (!byKey.has(key)) byKey.set(key, new Map());
          incrementCount(byKey.get(key), variable.name);
        });
      });
      return Array.from(byKey.values())
        .map(mostCommonText)
        .filter(Boolean)
        .sort((a, b) => a.localeCompare(b, "zh-Hans-CN"));
    });
    const variableDisplayLabel = variable => {
      const from = String(variable?.from || "").trim();
      const to = String(variable?.to || "").trim();
      return from || to ? `${variable.name}: ${from || "-"} → ${to || "-"}` : variable.name;
    };
    const variableListLine = variable => `- ${variableDisplayLabel(variable)}`;
    const variableListText = variables => cleanVariables(variables).map(variableListLine).join("\n");
    const variablesTextFromVariables = variables => cleanVariables(variables).map(variableDisplayLabel).join("\n");
    const parseVariableLine = value => {
      const line = String(value || "").trim().replace(/^[-*]\s+/, "");
      if (!line) return null;
      const structured = line.match(/^(.+?)\s*[:：=]\s*(.*?)\s*(?:->|→|=>|⇒)\s*(.*)$/);
      if (structured) {
        return {
          name: structured[1].trim(),
          from: structured[2].trim(),
          to: structured[3].trim(),
        };
      }
      const arrowOnly = line.match(/^(.+?)\s*(?:->|→|=>|⇒)\s*(.*)$/);
      if (arrowOnly) {
        return {
          name: arrowOnly[1].trim(),
          from: "",
          to: arrowOnly[2].trim(),
        };
      }
      return { name: line, from: "", to: "" };
    };
    const parseVariablesText = text => String(text || "")
      .split(/\r?\n/)
      .map(parseVariableLine)
      .filter(item => item?.name);
    const quickVariableLine = quick => {
      const name = String(quick?.name || "").trim();
      const from = normalizeVariableValue(quick?.from);
      const to = normalizeVariableValue(quick?.to);
      if (!name) return "";
      return from || to ? `${name}: ${from} -> ${to}` : name;
    };
    const appendVariableLine = (target, line) => {
      if (!line) return;
      const current = String(target.variablesText || "").replace(/\s+$/, "");
      target.variablesText = current ? `${current}\n${line}` : line;
    };
    const appendQuickVariable = targetName => {
      const isAdd = targetName === "add";
      const quickRef = isAdd ? addVariableQuick : edgeVariableQuick;
      const line = quickVariableLine(quickRef.value);
      if (!line) {
        showToast("请先填写变量名", "error");
        return;
      }
      appendVariableLine(isAdd ? addForm.value : edgeDraft.value, line);
      quickRef.value = emptyVariableQuick();
    };
    const draftVariables = () => parseVariablesText(edgeDraft.value.variablesText);
    const hydrateEdgeDraft = edge => {
      const variables = cleanVariables(edge?.variables || []);
      edgeDraft.value = {
        title: edge?.title || "",
        description: edge?.description || "",
        variables: variables.map(item => ({ ...item })),
        variablesText: variablesTextFromVariables(variables),
      };
    };
    watch(selectedEdge, edge => {
      if (edge) hydrateEdgeDraft(edge);
    });

    const roiMetricOptions = computed(() => ROI_METRIC_DEFS);
    const currentRoiMetricLabel = computed(() =>
      (roiMetricOptions.value.find(item => item.key === roiMetricKey.value) || roiMetricOptions.value[0]).label
    );
    const metricNumber = value => {
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    };
    const roiMetricForEdge = edge => {
      if (!edge?.perf || edge.perf.incomplete) return null;
      const metric = edge.perf.metrics?.[roiMetricKey.value] || null;
      const deltaPct = metricNumber(metric?.delta_pct);
      if (deltaPct === null) return null;
      const parent = metricNumber(metric?.parent);
      const child = metricNumber(metric?.child);
      return {
        gainMs: parent !== null && child !== null ? parent - child : null,
        gainPct: -deltaPct,
      };
    };
    const changeValueKey = value => normalizeVariableValue(value).toLocaleLowerCase();
    const createRoiGroup = (key, label, variableName, groupMode) => ({
      key,
      label,
      variableName,
      groupMode,
      isolatedCount: 0,
      combinedCount: 0,
      evaluableIsolatedCount: 0,
      totalGainMsRaw: 0,
      totalGainPctRaw: 0,
      hitCount: 0,
      bestGainMs: null,
      bestGainPct: null,
      bestEdge: null,
      bestEdgeId: "",
      edgeIds: new Set(),
      nameCounts: new Map(),
      distributionCounts: new Map(),
    });
    const roiGroupForVariable = (groups, variable) => {
      const nameKey = normalizeVariableNameKey(variable.name);
      if (!nameKey) return null;
      const from = normalizeVariableValue(variable.from);
      const to = normalizeVariableValue(variable.to);
      const mode = roiGroupMode.value;
      if (mode === "change" && (!from || !to)) return null;
      const key = mode === "change"
        ? `${nameKey}\u0000${changeValueKey(from)}\u0000${changeValueKey(to)}`
        : nameKey;
      if (!groups.has(key)) {
        const label = mode === "change" ? `${variable.name}: ${from} → ${to}` : variable.name;
        groups.set(key, createRoiGroup(key, label, variable.name, mode));
      }
      const group = groups.get(key);
      incrementCount(group.nameCounts, variable.name);
      incrementCount(group.distributionCounts, `${from || "未填"} → ${to || "未填"}`);
      if (mode === "variable") group.label = mostCommonText(group.nameCounts) || group.label;
      return group;
    };
    const finalizeRoiGroup = group => {
      const evaluable = group.evaluableIsolatedCount;
      const distribution = Array.from(group.distributionCounts.entries())
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "zh-Hans-CN"))
        .map(([label, count]) => `${label} x${count}`)
        .join("；");
      return {
        ...group,
        edgeIds: Array.from(group.edgeIds),
        totalGainMs: evaluable ? group.totalGainMsRaw : null,
        totalGainPct: evaluable ? group.totalGainPctRaw : null,
        averageGainMs: evaluable ? group.totalGainMsRaw / evaluable : null,
        averageGainPct: evaluable ? group.totalGainPctRaw / evaluable : null,
        hitRate: evaluable ? group.hitCount / evaluable : null,
        distributionText: distribution ? `取值分布：${distribution}` : "暂无取值分布",
      };
    };
    const roiAggregation = computed(() => {
      const groups = new Map();
      const variableNameKeys = new Set();
      const evaluableEdgeIds = new Set();
      const combined = [];
      let skippedEdgeCount = 0;
      edges.value.forEach(edge => {
        const variables = cleanVariables(edge?.variables || []);
        if (!variables.length) return;
        variables.forEach(variable => variableNameKeys.add(normalizeVariableNameKey(variable.name)));
        const metric = roiMetricForEdge(edge);
        if (!metric) skippedEdgeCount += 1;
        else evaluableEdgeIds.add(edge.id);
        const isolated = variables.length === 1;
        if (!isolated) {
          combined.push({
            edge,
            variablesText: variables.map(variableDisplayLabel).join("；"),
            gainMs: metric?.gainMs ?? null,
            gainPct: metric?.gainPct ?? null,
          });
        }
        variables.forEach(variable => {
          const group = roiGroupForVariable(groups, variable);
          if (!group) return;
          group.edgeIds.add(edge.id);
          if (isolated) {
            group.isolatedCount += 1;
            if (metric && metric.gainMs !== null) {
              group.evaluableIsolatedCount += 1;
              group.totalGainMsRaw += metric.gainMs;
              group.totalGainPctRaw += metric.gainPct;
              if (metric.gainMs > 0) group.hitCount += 1;
              if (group.bestGainMs === null || metric.gainMs > group.bestGainMs) {
                group.bestGainMs = metric.gainMs;
                group.bestGainPct = metric.gainPct;
                group.bestEdge = edge;
                group.bestEdgeId = edge.id;
              }
            }
          } else {
            group.combinedCount += 1;
          }
        });
      });
      const rows = Array.from(groups.values()).map(finalizeRoiGroup);
      combined.sort((a, b) => {
        const av = Number.isFinite(Number(a.gainMs)) ? Math.abs(Number(a.gainMs)) : -1;
        const bv = Number.isFinite(Number(b.gainMs)) ? Math.abs(Number(b.gainMs)) : -1;
        return bv - av || String(a.edge?.created_at || "").localeCompare(String(b.edge?.created_at || ""));
      });
      return {
        rows,
        combined,
        skippedEdgeCount,
        evaluableEdgeCount: evaluableEdgeIds.size,
        variableCount: Array.from(variableNameKeys).filter(Boolean).length,
        edgeWithVariablesCount: edges.value.filter(edge => cleanVariables(edge?.variables || []).length).length,
      };
    });
    const roiRows = computed(() => roiAggregation.value.rows);
    const roiCombinedEdges = computed(() => roiAggregation.value.combined);
    const roiSkippedEdgeCount = computed(() => roiAggregation.value.skippedEdgeCount);
    const roiEmptyText = computed(() => {
      if (!roiAggregation.value.edgeWithVariablesCount) return "在标记优化关系时填写『变更项』，这里会自动汇总各变量收益";
      if (!roiRows.value.length && roiGroupMode.value === "change") return "具体变更聚合需要同时填写 from 和 to";
      if (!roiRows.value.length) return "暂无可聚合的变量收益";
      return "";
    });
    const isFiniteRoiValue = value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
    const roiGainClass = value => {
      if (!isFiniteRoiValue(value)) return "neutral";
      const number = Number(value);
      if (number === 0) return "neutral";
      return number > 0 ? "good" : "bad";
    };
    const formatRoiGainMs = value => {
      if (!isFiniteRoiValue(value)) return "—";
      const number = Number(value);
      return `${number > 0 ? "+" : ""}${number.toFixed(2)} ms`;
    };
    const formatRoiGainPct = value => {
      if (!isFiniteRoiValue(value)) return "—";
      const number = Number(value);
      return `${number > 0 ? "+" : ""}${number.toFixed(1)}%`;
    };
    const formatHitRate = value => {
      if (!isFiniteRoiValue(value)) return "—";
      const number = Number(value);
      return `${Math.round(number * 100)}%`;
    };
    const roiSortValue = (row, key) => {
      if (key === "label") return String(row.label || "");
      if (key === "bestGainMs") return row.bestGainMs;
      return row[key];
    };
    const sortRows = (items, sortRef, valueFor) => {
      const { key, dir } = sortRef.value;
      const direction = dir === "asc" ? 1 : -1;
      return items.slice().sort((a, b) => {
        const av = valueFor(a, key);
        const bv = valueFor(b, key);
        const aMissing = av === null || av === undefined || av === "";
        const bMissing = bv === null || bv === undefined || bv === "";
        if (aMissing || bMissing) {
          if (aMissing && bMissing) return 0;
          return aMissing ? 1 : -1;
        }
        const an = Number(av);
        const bn = Number(bv);
        const aNumeric = Number.isFinite(an);
        const bNumeric = Number.isFinite(bn);
        if (aNumeric || bNumeric) {
          if (!aNumeric && !bNumeric) return 0;
          if (!aNumeric) return 1;
          if (!bNumeric) return -1;
          return (an - bn) * direction;
        }
        return String(av || "").localeCompare(String(bv || ""), "zh-Hans-CN") * direction;
      });
    };
    const sortedRoiRows = computed(() => sortRows(roiRows.value, roiSort, roiSortValue));
    const setSort = (sortRef, key, defaultDir = "desc") => {
      const current = sortRef.value;
      sortRef.value = current.key === key
        ? { key, dir: current.dir === "asc" ? "desc" : "asc" }
        : { key, dir: defaultDir };
    };
    const sortMark = (sortRef, key) => sortRef.value.key === key ? (sortRef.value.dir === "asc" ? "↑" : "↓") : "";
    const setRoiSort = key => setSort(roiSort, key, key === "label" ? "asc" : "desc");
    const roiSortMark = key => sortMark(roiSort, key);
    const roiBarItems = computed(() => {
      const items = sortedRoiRows.value
        .map(row => ({
          ...row,
          barValue: roiBarMode.value === "average" ? row.averageGainMs : row.totalGainMs,
        }))
        .filter(row => isFiniteRoiValue(row.barValue))
        .slice(0, 12);
      const maxAbs = Math.max(1, ...items.map(row => Math.abs(Number(row.barValue))));
      return items.map(row => ({
        ...row,
        barWidth: Math.max(4, Math.round((Math.abs(Number(row.barValue)) / maxAbs) * 1000) / 10),
      }));
    });
    const roiSummary = computed(() => {
      const evaluableRows = roiRows.value.filter(row => isFiniteRoiValue(row.totalGainMs));
      const byGainDesc = evaluableRows.slice().sort((a, b) => Number(b.totalGainMs) - Number(a.totalGainMs));
      const degradedRows = evaluableRows
        .filter(row => Number(row.totalGainMs) < 0)
        .sort((a, b) => Number(a.totalGainMs) - Number(b.totalGainMs));
      return {
        variableCount: roiAggregation.value.variableCount,
        evaluableEdgeCount: roiAggregation.value.evaluableEdgeCount,
        best: byGainDesc[0] || null,
        worst: degradedRows[0] || null,
      };
    });
    const focusRoiGroup = row => {
      const edgeIds = Array.from(row?.edgeIds || []);
      if (!edgeIds.length) return;
      roiHighlightEdgeIds.value = edgeIds;
      selectedNodeId.value = "";
      selectedEdgeId.value = "";
      hoverEdgeId.value = "";
      viewMode.value = "canvas";
      nextTick().then(fitGraphToViewport);
    };
    const focusEdge = edge => {
      if (!edge) return;
      roiHighlightEdgeIds.value = [];
      viewMode.value = "canvas";
      nextTick().then(() => selectEdge(edge));
    };
    const openRoiBestEdge = row => {
      if (row?.bestEdge?.compare_job_id) {
        openJob(row.bestEdge.compare_job_id);
        return;
      }
      focusEdge(row?.bestEdge);
    };
    const allMetricNodes = computed(() => {
      const byId = new Map();
      [...nodes.value, ...unconnected.value].forEach(node => {
        if (node?.id) byId.set(node.id, node);
      });
      return Array.from(byId.values());
    });
    const nodeMetricSortValue = (node, key) => {
      if (key === "label") return nodeTitleText(node);
      if (key === "created_at") return String(node?.created_at || "");
      return metricNumber(node?.[key]);
    };
    const sortedNodeMetricRows = computed(() => sortRows(allMetricNodes.value, nodeMetricSort, nodeMetricSortValue));
    const setNodeMetricSort = key => setSort(nodeMetricSort, key, key === "label" ? "asc" : "desc");
    const nodeMetricSortMark = key => sortMark(nodeMetricSort, key);
    const selectNodeFromRoi = node => {
      if (!node?.id) return;
      roiHighlightEdgeIds.value = [];
      viewMode.value = "canvas";
      nextTick().then(() => selectNode(node));
    };
    const nodeCompareTargetOptions = computed(() =>
      candidateOptions.value.filter(job => job.id !== selectedNode.value?.id && job.mode === "single")
    );
    const compareSelectedNode = async () => {
      if (!selectedNode.value?.id || !nodeCompareTargetId.value || nodeCompareSaving.value) return;
      nodeCompareSaving.value = true;
      try {
        const payload = await fetchJson(
          `/api/projects/${encodeURIComponent(projectId.value)}/experiments/compare`,
          {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              job_id_a: selectedNode.value.id,
              job_id_b: nodeCompareTargetId.value,
            }),
          },
          "生成节点对比失败",
        );
        if (payload.compare_job_id) {
          showToast(payload.existing ? "已打开已有对比" : "节点对比已创建", "success");
          router.push({ path: jobRoutePath({ id: payload.compare_job_id, seq: payload.compare_job_seq }) });
        }
      } catch (e) {
        showToast(normalizeApiError(e, "生成节点对比失败"), "error");
      } finally {
        nodeCompareSaving.value = false;
      }
    };

    const loadCandidates = async () => {
      if (!projectId.value) return;
      try {
        const payload = await fetchJson(
          `/api/jobs?project_id=${encodeURIComponent(projectId.value)}&limit=500`,
          { credentials: "include" },
          "加载候选任务失败",
        );
        candidateJobs.value = (payload.data || []).filter(job => job.mode === "single");
      } catch (e) {
        candidateJobs.value = [...nodes.value, ...unconnected.value];
      }
    };
    const loadGraph = async () => {
      if (!projectId.value) return;
      loading.value = true;
      try {
        const payload = await fetchJson(
          `/api/projects/${encodeURIComponent(projectId.value)}/experiments`,
          { credentials: "include" },
          "加载实验树失败",
        );
        if (payload.project) {
          projectMeta.value = payload.project;
          const projectIndex = projects.value.findIndex(project => project.id === payload.project.id);
          if (projectIndex >= 0) projects.value.splice(projectIndex, 1, { ...projects.value[projectIndex], ...payload.project });
          rememberRecentProject(payload.project);
          refreshProjectMetricTargetDraft();
        }
        nodes.value = payload.nodes || [];
        unconnected.value = payload.unconnected || [];
        edges.value = payload.edges || [];
        if (selectedEdgeId.value && !edges.value.some(edge => edge.id === selectedEdgeId.value)) selectedEdgeId.value = "";
        if (selectedNodeId.value && !nodes.value.some(node => node.id === selectedNodeId.value)) selectedNodeId.value = "";
        await loadCandidates();
        await fitGraphAfterPaint();
      } catch (e) {
        showToast(normalizeApiError(e, "加载实验树失败"), "error");
      } finally {
        loading.value = false;
      }
    };
    const syncProjectMeta = project => {
      if (!project?.id) return;
      projectMeta.value = { ...(projectMeta.value || {}), ...project };
      const projectIndex = projects.value.findIndex(item => item.id === project.id);
      if (projectIndex >= 0) projects.value.splice(projectIndex, 1, { ...projects.value[projectIndex], ...project });
    };
    const saveProjectMetricTarget = async () => {
      if (projectMetricTargetSaving.value || !projectId.value) return;
      const metricDef = currentLineageMetricDef.value;
      const draft = String(projectMetricTargetDraft.value || "").trim();
      const nextTarget = draft ? normalizeProjectMetricTarget(draft) : null;
      if (draft && nextTarget === null) {
        showToast(`${metricDef.label}目标必须是正数`, "error");
        return;
      }
      const nextTargets = { ...projectMetricTargets.value };
      if (nextTarget === null) delete nextTargets[metricDef.key];
      else nextTargets[metricDef.key] = nextTarget;
      projectMetricTargetSaving.value = true;
      try {
        const updated = await fetchJson(
          `/api/projects/${encodeURIComponent(projectId.value)}`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            credentials: "include",
            body: JSON.stringify({ metric_targets: nextTargets }),
          },
          `保存${metricDef.label}目标失败`,
        );
        syncProjectMeta(updated);
        refreshProjectMetricTargetDraft();
        showToast(nextTarget === null ? `${metricDef.label}目标已清除` : `${metricDef.label}目标已保存`, "success");
      } catch (e) {
        showToast(normalizeApiError(e, `保存${metricDef.label}目标失败`), "error");
      } finally {
        projectMetricTargetSaving.value = false;
      }
    };
    const clearProjectMetricTarget = () => {
      projectMetricTargetDraft.value = "";
      if (currentMetricTargetValue.value !== null) saveProjectMetricTarget();
    };

    const replaceEdge = edge => {
      const index = edges.value.findIndex(item => item.id === edge.id);
      if (index >= 0) edges.value.splice(index, 1, edge);
      else edges.value.push(edge);
    };
    const selectNode = node => {
      roiHighlightEdgeIds.value = [];
      selectedNodeId.value = node.id;
      selectedEdgeId.value = "";
    };
    const selectNodeById = nodeId => {
      const node = nodeById.value[nodeId] || nodes.value.find(item => item.id === nodeId);
      if (node) selectNode(node);
    };
    const selectEdge = edge => {
      roiHighlightEdgeIds.value = [];
      selectedEdgeId.value = edge.id;
      selectedNodeId.value = "";
      hydrateEdgeDraft(edge);
    };
    const openJob = jobOrId => {
      const path = jobRoutePath(jobOrId);
      if (path) router.push({ path });
    };
    const chartAlphaColor = (color, alpha) => {
      const text = String(color || "").trim();
      if (/^#[0-9a-f]{6}$/i.test(text)) {
        const value = parseInt(text.slice(1), 16);
        return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
      }
      if (/^rgb\(/i.test(text)) return text.replace(/^rgb\((.+)\)$/i, `rgba($1, ${alpha})`);
      return text;
    };
    const lineageChartThemeColors = () => {
      if (typeof window === "undefined") {
        return {
          text: "#64748b",
          title: "#1e293b",
          grid: "rgba(148,163,184,.16)",
          gridStrong: "rgba(148,163,184,.28)",
          axis: "rgba(100,116,139,.36)",
          target: "#6366f1",
          targetSoft: "rgba(99,102,241,.10)",
          pointBorder: "#ffffff",
          pointLabelBg: "rgba(255,255,255,.92)",
          pointLabelBorder: "rgba(148,163,184,.32)",
        };
      }
      const style = getComputedStyle(document.documentElement);
      const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
      const border = read("--border", "#cbd5e1");
      return {
        text: read("--text2", "#64748b"),
        title: read("--text", "#1e293b"),
        grid: chartAlphaColor(border, 0.46),
        gridStrong: chartAlphaColor(border, 0.74),
        axis: chartAlphaColor(border, 0.95),
        target: read("--purple-l", "#6366f1"),
        targetSoft: read("--purple-bg", "rgba(99,102,241,.10)"),
        pointBorder: read("--exp-surface", "#ffffff"),
        pointLabelBg: read("--exp-surface", "#ffffff"),
        pointLabelBorder: chartAlphaColor(border, 0.88),
      };
    };
    const ensureLineageChartTooltip = () => {
      if (typeof document === "undefined") return null;
      if (lineageChartTooltipEl) return lineageChartTooltipEl;
      lineageChartTooltipEl = document.createElement("div");
      lineageChartTooltipEl.className = "exp-chart-tooltip";
      document.body.appendChild(lineageChartTooltipEl);
      return lineageChartTooltipEl;
    };
    const hideLineageChartTooltip = () => {
      if (lineageChartTooltipEl) lineageChartTooltipEl.style.opacity = "0";
    };
    const destroyLineageChartTooltip = () => {
      if (!lineageChartTooltipEl) return;
      lineageChartTooltipEl.remove();
      lineageChartTooltipEl = null;
    };
    const renderLineageChartTooltip = context => {
      const tooltip = context?.tooltip;
      const chart = context?.chart;
      if (!tooltip || tooltip.opacity === 0 || !chart?.canvas) {
        hideLineageChartTooltip();
        return;
      }
      const point = tooltip.dataPoints?.[0]?.raw || {};
      const node = nodeById.value[point.nodeId] || nodes.value.find(item => item.id === point.nodeId) || point;
      const tooltipEl = ensureLineageChartTooltip();
      if (!tooltipEl) return;

      tooltipEl.replaceChildren();
      const title = document.createElement("strong");
      title.className = "exp-chart-tooltip-title";
      title.textContent = point.label || "节点";
      tooltipEl.appendChild(title);

      const rowsEl = document.createElement("div");
      rowsEl.className = "exp-chart-tooltip-rows";
      lineageHoverMetricRows(node).forEach(row => {
        const rowEl = document.createElement("div");
        rowEl.className = "exp-chart-tooltip-row";
        const labelEl = document.createElement("span");
        labelEl.textContent = row.label;
        const valueEl = document.createElement("b");
        valueEl.textContent = row.value;
        rowEl.append(labelEl, valueEl);
        rowsEl.appendChild(rowEl);
      });
      tooltipEl.appendChild(rowsEl);

      const rect = chart.canvas.getBoundingClientRect();
      const margin = 10;
      tooltipEl.style.opacity = "1";
      tooltipEl.style.left = "0px";
      tooltipEl.style.top = "0px";
      const width = tooltipEl.offsetWidth || 240;
      const height = tooltipEl.offsetHeight || 150;
      const rawLeft = rect.left + tooltip.caretX + 12;
      const rawTop = rect.top + tooltip.caretY - height / 2;
      tooltipEl.style.left = `${Math.min(Math.max(rawLeft, margin), window.innerWidth - width - margin)}px`;
      tooltipEl.style.top = `${Math.min(Math.max(rawTop, margin), window.innerHeight - height - margin)}px`;
    };
    const destroyLineageChart = () => {
      destroyLineageChartTooltip();
      if (!lineageChartInst) return;
      lineageChartInst.destroy();
      lineageChartInst = null;
    };
    const chartPointFromHit = (chart, hit) => {
      if (!chart || !hit) return null;
      return chart.data.datasets?.[hit.datasetIndex]?.data?.[hit.index] || null;
    };
    const chartEventPoint = (chart, event) => {
      const directX = Number(event?.x);
      const directY = Number(event?.y);
      if (Number.isFinite(directX) && Number.isFinite(directY)) return { x: directX, y: directY };
      const native = event?.native || event;
      const rect = chart?.canvas?.getBoundingClientRect?.();
      if (!rect || !Number.isFinite(Number(native?.clientX)) || !Number.isFinite(Number(native?.clientY))) return null;
      return { x: Number(native.clientX) - rect.left, y: Number(native.clientY) - rect.top };
    };
    const pointSegmentDistance = (point, start, end) => {
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const lengthSquared = dx * dx + dy * dy;
      if (!lengthSquared) return Math.hypot(point.x - start.x, point.y - start.y);
      const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
      return Math.hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
    };
    const edgeBetweenNodes = (parentId, childId) =>
      edges.value.find(edge => edge.parent_job_id === parentId && edge.child_job_id === childId) || null;
    const chartSegmentFromEvent = (chart, event) => {
      const eventPoint = chartEventPoint(chart, event);
      if (!eventPoint || !chart?.scales?.x || !chart?.scales?.y) return null;
      let best = null;
      const hitDistance = 12;
      chart.data.datasets.forEach((dataset, datasetIndex) => {
        if (typeof chart.isDatasetVisible === "function" && !chart.isDatasetVisible(datasetIndex)) return;
        const data = dataset.data || [];
        for (let index = 1; index < data.length; index += 1) {
          const startPoint = data[index - 1];
          const endPoint = data[index];
          if (startPoint?.rawValue === null || endPoint?.rawValue === null) continue;
          const edge = edgeBetweenNodes(endPoint.parentNodeId, endPoint.nodeId);
          if (!edge) continue;
          const start = {
            x: chart.scales.x.getPixelForValue(startPoint.x),
            y: chart.scales.y.getPixelForValue(startPoint.rawValue),
          };
          const end = {
            x: chart.scales.x.getPixelForValue(endPoint.x),
            y: chart.scales.y.getPixelForValue(endPoint.rawValue),
          };
          if (![start.x, start.y, end.x, end.y].every(Number.isFinite)) continue;
          const distance = pointSegmentDistance(eventPoint, start, end);
          if (distance <= hitDistance && (!best || distance < best.distance)) {
            best = { edge, distance };
          }
        }
      });
      return best;
    };
    const buildLineageChart = () => {
      if (typeof Chart === "undefined" || !lineageChartCanvas.value || viewMode.value !== "chart" || lineageChartEmptyText.value) {
        destroyLineageChart();
        return;
      }
      destroyLineageChart();
      const metricDef = currentLineageMetricDef.value;
      const colors = lineageChartThemeColors();
      const axisTitle = `${metricDef.label}${metricDef.unit ? ` (${metricDef.unit})` : ""}`;
      const axisBounds = lineageChartAxisBounds.value;
      const targetLineValue = lineageMetricTargetLineValue.value;
      const targetLinePlugin = {
        id: "lineageMetricTargetLine",
        afterDatasetsDraw(chart) {
          if (targetLineValue === null || !Number.isFinite(Number(targetLineValue))) return;
          const { ctx, chartArea, scales } = chart;
          const y = scales.y.getPixelForValue(targetLineValue);
          if (!Number.isFinite(y) || y < chartArea.top - 1 || y > chartArea.bottom + 1) return;
          const label = `目标 ${formatLineageMetricValue(targetLineValue, metricDef)}`;
          ctx.save();
          ctx.setLineDash([7, 5]);
          ctx.strokeStyle = colors.target;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(chartArea.left, y);
          ctx.lineTo(chartArea.right, y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.font = "700 11px Inter, system-ui, sans-serif";
          const textWidth = ctx.measureText(label).width;
          const boxW = textWidth + 14;
          const boxH = 22;
          const boxX = Math.max(chartArea.left + 8, chartArea.right - boxW - 8);
          const boxY = Math.max(chartArea.top + 6, y - boxH - 6);
          ctx.fillStyle = colors.targetSoft;
          ctx.strokeStyle = chartAlphaColor(colors.target, 0.22);
          ctx.lineWidth = 1;
          ctx.beginPath();
          if (typeof ctx.roundRect === "function") ctx.roundRect(boxX, boxY, boxW, boxH, 6);
          else ctx.rect(boxX, boxY, boxW, boxH);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = colors.target;
          ctx.fillText(label, boxX + 7, boxY + 14);
          ctx.restore();
        },
      };
      const pointValueLabelPlugin = {
        id: "lineageMetricPointValueLabels",
        afterDatasetsDraw(chart) {
          const { ctx, chartArea } = chart;
          const drawn = new Set();
          ctx.save();
          ctx.font = "750 10px Inter, system-ui, sans-serif";
          ctx.textBaseline = "middle";
          chart.data.datasets.forEach((dataset, datasetIndex) => {
            if (typeof chart.isDatasetVisible === "function" && !chart.isDatasetVisible(datasetIndex)) return;
            const meta = chart.getDatasetMeta(datasetIndex);
            (dataset.data || []).forEach((point, pointIndex) => {
              if (point?.rawValue === null || point?.rawValue === undefined) return;
              const element = meta.data?.[pointIndex];
              const x = Number(element?.x);
              const y = Number(element?.y);
              if (!Number.isFinite(x) || !Number.isFinite(y)) return;
              const key = `${point.nodeId}:${point.rawValue}`;
              if (drawn.has(key)) return;
              drawn.add(key);
              const label = formatLineageMetricValue(point.rawValue, metricDef);
              const textWidth = ctx.measureText(label).width;
              const boxW = textWidth + 10;
              const boxH = 18;
              const boxX = Math.min(Math.max(x - boxW / 2, chartArea.left + 4), chartArea.right - boxW - 4);
              let boxY = y - boxH - 10;
              if (boxY < chartArea.top + 4) boxY = y + 10;
              if (boxY + boxH > chartArea.bottom - 2) boxY = chartArea.bottom - boxH - 2;
              ctx.fillStyle = colors.pointLabelBg;
              ctx.strokeStyle = colors.pointLabelBorder;
              ctx.lineWidth = 1;
              ctx.beginPath();
              if (typeof ctx.roundRect === "function") ctx.roundRect(boxX, boxY, boxW, boxH, 6);
              else ctx.rect(boxX, boxY, boxW, boxH);
              ctx.fill();
              ctx.stroke();
              ctx.fillStyle = colors.title;
              ctx.textAlign = "center";
              ctx.fillText(label, boxX + boxW / 2, boxY + boxH / 2 + 0.5);
            });
          });
          ctx.restore();
        },
      };
      const datasets = lineageChartModel.value.datasets.map(dataset => ({
        ...dataset,
        pointBorderColor: colors.pointBorder,
        pointHoverBorderColor: colors.pointBorder,
        data: dataset.data.map(point => ({ ...point })),
      }));
      lineageChartInst = new Chart(lineageChartCanvas.value, {
        type: "line",
        data: {
          datasets,
        },
        plugins: targetLineValue === null ? [pointValueLabelPlugin] : [targetLinePlugin, pointValueLabelPlugin],
        options: {
          responsive: true,
          maintainAspectRatio: false,
          parsing: false,
          normalized: true,
          layout: {
            padding: { top: 16, right: 16, bottom: 8, left: 8 },
          },
          elements: {
            line: {
              borderCapStyle: "round",
              borderJoinStyle: "round",
            },
            point: {
              hoverBorderWidth: 3,
            },
          },
          interaction: { mode: "nearest", intersect: true },
          onClick: (event, _elements, chart) => {
            const hits = chart.getElementsAtEventForMode(event, "nearest", { intersect: true }, false);
            const point = chartPointFromHit(chart, hits?.[0]);
            if (point?.nodeId) {
              selectNodeById(point.nodeId);
              return;
            }
            const segment = chartSegmentFromEvent(chart, event);
            if (segment?.edge) selectEdge(segment.edge);
          },
          onHover: (event, elements) => {
            const target = event?.native?.target || lineageChartCanvas.value;
            if (target) target.style.cursor = elements?.length || chartSegmentFromEvent(event.chart || lineageChartInst, event) ? "pointer" : "default";
          },
          plugins: {
            legend: {
              position: "top",
              align: "start",
              labels: {
                color: colors.text,
                boxWidth: 10,
                boxHeight: 10,
                padding: 14,
                font: { size: 11, weight: "650" },
                usePointStyle: true,
                pointStyle: "circle",
              },
            },
            tooltip: {
              enabled: false,
              external: renderLineageChartTooltip,
            },
          },
          scales: {
            x: {
              type: "linear",
              min: axisBounds.x?.min ?? 0,
              max: axisBounds.x?.max,
              title: {
                display: true,
                text: "子孙代数",
                color: colors.title,
                padding: { top: 10 },
                font: { size: 13, weight: "800" },
              },
              ticks: {
                stepSize: 1,
                precision: 0,
                color: colors.title,
                padding: 10,
                font: { size: 12, weight: "800" },
                callback: value => Number.isInteger(Number(value)) ? value : "",
              },
              grid: {
                color: context => Number(context.tick?.value) === 0 ? colors.gridStrong : colors.grid,
                lineWidth: context => Number(context.tick?.value) === 0 ? 1.4 : 0.8,
                tickLength: 4,
                tickColor: colors.axis,
              },
              border: { color: colors.axis, width: 2 },
            },
            y: {
              beginAtZero: false,
              min: axisBounds.y?.min,
              max: axisBounds.y?.max,
              title: {
                display: true,
                text: axisTitle,
                color: colors.title,
                padding: { bottom: 10 },
                font: { size: 13, weight: "800" },
              },
              ticks: {
                color: colors.title,
                padding: 10,
                font: { size: 12, weight: "800" },
                callback: value => formatLineageAxisTick(value, metricDef),
              },
              grid: {
                color: colors.grid,
                lineWidth: 0.8,
                tickLength: 4,
                tickColor: colors.axis,
              },
              border: { color: colors.axis, width: 2 },
            },
          },
        },
      });
    };
    const scheduleLineageChart = async () => {
      const token = ++lineageChartBuildToken;
      await nextTick();
      await new Promise(resolve => {
        if (typeof requestAnimationFrame === "function") requestAnimationFrame(resolve);
        else setTimeout(resolve, 0);
      });
      if (token !== lineageChartBuildToken) return;
      buildLineageChart();
    };
    const openAddEdge = (parentId = "", childId = "") => {
      addForm.value = {
        parent_job_id: parentId,
        child_job_id: childId,
        title: "",
        description: "",
        variablesText: "",
      };
      showAddEdge.value = true;
      loadCandidates();
      nextTick(() => document.querySelector(".exp-edge-modal select")?.focus());
    };
    const closeAddEdge = () => {
      if (saving.value) return;
      showAddEdge.value = false;
    };
    const submitAddEdge = async () => {
      if (!addForm.value.parent_job_id || !addForm.value.child_job_id) {
        showToast("请选择父节点和子节点", "error");
        return;
      }
      saving.value = true;
      try {
        const { variablesText, ...payload } = addForm.value;
        const edge = await fetchJson(
          `/api/projects/${encodeURIComponent(projectId.value)}/experiments/edges`,
          {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...payload,
              variables: parseVariablesText(variablesText),
            }),
          },
          "创建优化关系失败",
        );
        replaceEdge(edge);
        showAddEdge.value = false;
        selectedEdgeId.value = edge.id;
        await loadGraph();
        showToast("优化关系已创建", "success");
      } catch (e) {
        showToast(normalizeApiError(e, "创建优化关系失败"), "error");
      } finally {
        saving.value = false;
      }
    };
    const saveEdge = async () => {
      if (!selectedEdge.value) return;
      saving.value = true;
      try {
        const edge = await fetchJson(
          `/api/experiments/edges/${encodeURIComponent(selectedEdge.value.id)}`,
          {
            method: "PATCH",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              title: edgeDraft.value.title,
              description: edgeDraft.value.description,
              variables: draftVariables(),
            }),
          },
          "保存优化关系失败",
        );
        replaceEdge(edge);
        showToast("已保存", "success");
      } catch (e) {
        showToast(normalizeApiError(e, "保存优化关系失败"), "error");
      } finally {
        saving.value = false;
      }
    };
    const deleteSelectedEdge = async () => {
      if (!selectedEdge.value || !window.confirm("删除这条优化关系？")) return;
      saving.value = true;
      try {
        await fetch(`/api/experiments/edges/${encodeURIComponent(selectedEdge.value.id)}`, {
          method: "DELETE",
          credentials: "include",
        }).then(async response => {
          if (!response.ok) throw new ApiRequestError((await readJsonResponse(response, {}))?.detail || "删除优化关系失败", { status: response.status });
        });
        selectedEdgeId.value = "";
        await loadGraph();
        showToast("关系已删除", "success");
      } catch (e) {
        showToast(normalizeApiError(e, "删除优化关系失败"), "error");
      } finally {
        saving.value = false;
      }
    };
    const refreshPerf = async () => {
      if (!selectedEdge.value) return;
      saving.value = true;
      try {
        const edge = await fetchJson(
          `/api/experiments/edges/${encodeURIComponent(selectedEdge.value.id)}/refresh-perf`,
          { method: "POST", credentials: "include" },
          "重新计算失败",
        );
        replaceEdge(edge);
        if (edge.skipped) {
          showToast("性能摘要为手动编辑，未覆盖", "info");
        } else if (edge.perf?.incomplete) {
          showToast("已重新计算，仍缺少 e2e 或 compute 数据", "info");
        } else {
          showToast("已重新计算，性能摘要已完整", "success");
        }
      } catch (e) {
        showToast(normalizeApiError(e, "重新计算失败"), "error");
      } finally {
        saving.value = false;
      }
    };
    const createCompare = async () => {
      if (!selectedEdge.value) return;
      saving.value = true;
      try {
        const payload = await fetchJson(
          `/api/experiments/edges/${encodeURIComponent(selectedEdge.value.id)}/compare`,
          { method: "POST", credentials: "include" },
          "生成详细对比失败",
        );
        if (payload.compare_job_id) {
          selectedEdge.value.compare_job_id = payload.compare_job_id;
          showToast("详细对比已创建", "success");
          router.push({ path: jobRoutePath({ id: payload.compare_job_id, seq: payload.compare_job_seq }) });
        }
      } catch (e) {
        showToast(normalizeApiError(e, "生成详细对比失败"), "error");
      } finally {
        saving.value = false;
      }
    };

    const saveLayout = async positions => {
      if (!positions.length) return;
      try {
        await fetchJson(
          `/api/projects/${encodeURIComponent(projectId.value)}/experiments/layout`,
          {
            method: "PUT",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ positions }),
          },
          "保存布局失败",
        );
      } catch (e) {
        showToast(normalizeApiError(e, "保存布局失败"), "error");
      }
    };
    const saveEdgeLabelLayout = async layout => {
      if (!layout?.id) return;
      try {
        const edge = await fetchJson(
          `/api/experiments/edges/${encodeURIComponent(layout.id)}`,
          {
            method: "PATCH",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              label_layout: {
                label_x: layout.label_x,
                label_y: layout.label_y,
                label_width: layout.label_width,
                label_height: layout.label_height,
              },
            }),
          },
          "保存关系框布局失败",
        );
        replaceEdge(edge);
      } catch (e) {
        showToast(normalizeApiError(e, "保存关系框布局失败"), "error");
      }
    };
    const resetLayout = async () => {
      saving.value = true;
      try {
        await fetch(`/api/projects/${encodeURIComponent(projectId.value)}/experiments/layout`, {
          method: "DELETE",
          credentials: "include",
        }).then(async response => {
          if (!response.ok) throw new ApiRequestError((await readJsonResponse(response, {}))?.detail || "重置布局失败", { status: response.status });
        });
        await loadGraph();
      } catch (e) {
        showToast(normalizeApiError(e, "重置布局失败"), "error");
      } finally {
        saving.value = false;
      }
    };

    const startPan = event => {
      if (event.target.closest(".exp-node, .exp-edge-label, .exp-panel, button, input, textarea, select")) return;
      panState = {
        x: event.clientX,
        y: event.clientY,
        tx: view.value.tx,
        ty: view.value.ty,
      };
      window.addEventListener("mousemove", movePan);
      window.addEventListener("mouseup", stopPan);
    };
    const movePan = event => {
      if (!panState) return;
      view.value.tx = panState.tx + event.clientX - panState.x;
      view.value.ty = panState.ty + event.clientY - panState.y;
    };
    const stopPan = () => {
      panState = null;
      window.removeEventListener("mousemove", movePan);
      window.removeEventListener("mouseup", stopPan);
    };
    const finiteLayoutNumber = (value, fallback = 0) => {
      const number = Number(value);
      return Number.isFinite(number) ? number : fallback;
    };
    const layoutNodeById = id => nodes.value.find(node => node.id === id) || nodeById.value[id] || null;
    const nodeLayoutRect = node => {
      const x = finiteLayoutNumber(node?.x, 0);
      const y = finiteLayoutNumber(node?.y, 0);
      const width = nodeWidth(node);
      const height = nodeHeight(node);
      return {
        minX: x,
        minY: y,
        maxX: x + width,
        maxY: y + height,
        width,
        height,
        centerX: x + width / 2,
        centerY: y + height / 2,
      };
    };
    const moveLayoutNode = (id, x, y) => {
      const index = nodes.value.findIndex(node => node.id === id);
      if (index < 0) return false;
      const current = nodes.value[index];
      const nextX = roundLayout(x);
      const nextY = roundLayout(y);
      if (roundLayout(current.x) === nextX && roundLayout(current.y) === nextY && Number(current.pinned) === 1) return false;
      nodes.value.splice(index, 1, { ...current, x: nextX, y: nextY, pinned: 1 });
      return true;
    };
    const updateNodePosition = (id, x, y) => moveLayoutNode(id, x, y);
    const hasDirectEdge = (sourceId, targetId) => edges.value.some(edge => edge.parent_job_id === sourceId && edge.child_job_id === targetId);
    const rectsTooClose = (source, target, gapX, gapY) => (
      target.minX < source.maxX + gapX
      && target.maxX > source.minX - gapX
      && target.minY < source.maxY + gapY
      && target.maxY > source.minY - gapY
    );
    const pushNodeAway = (sourceId, targetId) => {
      const sourceNode = layoutNodeById(sourceId);
      const targetNode = layoutNodeById(targetId);
      if (!sourceNode || !targetNode) return false;
      const source = nodeLayoutRect(sourceNode);
      const target = nodeLayoutRect(targetNode);
      const directDownstream = hasDirectEdge(sourceId, targetId);
      const gapY = directDownstream ? NODE_EDGE_GAP_Y : NODE_COLLISION_GAP_Y;
      if (!rectsTooClose(source, target, NODE_COLLISION_GAP_X, gapY)) return false;

      const pushRight = target.centerX >= source.centerX
        ? Math.max(0, source.maxX + NODE_COLLISION_GAP_X - target.minX)
        : Infinity;
      const pushDown = target.centerY >= source.centerY - 8
        ? Math.max(0, source.maxY + gapY - target.minY)
        : Infinity;
      if (!Number.isFinite(pushRight) && !Number.isFinite(pushDown)) {
        return moveLayoutNode(targetId, target.minX, source.maxY + gapY);
      }
      if (directDownstream || pushDown <= pushRight * 1.35) {
        return moveLayoutNode(targetId, target.minX, target.minY + pushDown);
      }
      return moveLayoutNode(targetId, target.minX + pushRight, target.minY);
    };
    const resolveNodeOverlaps = anchorId => {
      const movedIds = new Set();
      let frontier = [anchorId];
      for (let pass = 0; pass < 12 && frontier.length; pass += 1) {
        const next = new Set();
        const ids = displayNodes.value.map(node => node.id);
        frontier.forEach(sourceId => {
          ids.forEach(targetId => {
            if (!targetId || targetId === sourceId || targetId === anchorId) return;
            if (pushNodeAway(sourceId, targetId)) {
              movedIds.add(targetId);
              next.add(targetId);
            }
          });
        });
        frontier = [...next];
      }
      return movedIds;
    };
    const saveableNodeLayout = (id, fallback = {}) => {
      const raw = nodes.value.find(item => item.id === id);
      const displayed = nodeById.value[id] || raw;
      if (!displayed && !raw) return null;
      const x = finiteLayoutNumber(raw?.x, finiteLayoutNumber(displayed?.x, finiteLayoutNumber(fallback.x, 0)));
      const y = finiteLayoutNumber(raw?.y, finiteLayoutNumber(displayed?.y, finiteLayoutNumber(fallback.y, 0)));
      return { job_id: id, x, y, scale: 1, width: null, height: null, pinned: 1 };
    };
    const nodeStyle = node => ({
      left: `${node.x}px`,
      top: `${node.y}px`,
      width: `${nodeWidth(node)}px`,
      height: `${nodeHeight(node)}px`,
    });
    const startNodeDrag = (node, event) => {
      dragState = {
        id: node.id,
        x: event.clientX,
        y: event.clientY,
        nodeX: node.x,
        nodeY: node.y,
        moved: false,
      };
      window.addEventListener("mousemove", moveNode);
      window.addEventListener("mouseup", stopNodeDrag);
    };
    const moveNode = event => {
      if (!dragState) return;
      const screenDx = event.clientX - dragState.x;
      const screenDy = event.clientY - dragState.y;
      if (!dragState.moved && Math.abs(screenDx) <= 3 && Math.abs(screenDy) <= 3) return;
      dragState.moved = true;
      const dx = screenDx / view.value.scale;
      const dy = screenDy / view.value.scale;
      updateNodePosition(dragState.id, Math.round((dragState.nodeX + dx) * 10) / 10, Math.round((dragState.nodeY + dy) * 10) / 10);
    };
    const stopNodeDrag = async () => {
      if (!dragState) return;
      const state = dragState;
      dragState = null;
      window.removeEventListener("mousemove", moveNode);
      window.removeEventListener("mouseup", stopNodeDrag);
      if (!state.moved) return;
      const layout = saveableNodeLayout(state.id, { x: state.nodeX, y: state.nodeY });
      if (layout) await saveLayout([layout]);
    };
    const updateEdgeLabelLayout = (id, patch) => {
      const index = edges.value.findIndex(edge => edge.id === id);
      if (index < 0) return;
      edges.value.splice(index, 1, { ...edges.value[index], ...patch });
    };
    const currentEdgeLabelLayout = (id, fallback = {}) => {
      const item = edgePaths.value.find(path => path.edge.id === id);
      const edge = edges.value.find(candidate => candidate.id === id) || item?.edge || {};
      return {
        id,
        label_x: roundLayout(hasLayoutNumber(edge.label_x) ? edge.label_x : item?.labelX ?? fallback.label_x ?? 0),
        label_y: roundLayout(hasLayoutNumber(edge.label_y) ? edge.label_y : item?.labelY ?? fallback.label_y ?? 0),
        label_width: clampEdgeLabelWidth(edge.label_width || item?.labelWidth || fallback.label_width || EDGE_LABEL_W),
        label_height: clampEdgeLabelHeight(edge.label_height || item?.labelHeight || fallback.label_height || EDGE_LABEL_H),
      };
    };
    const edgeLabelStyle = item => ({
      left: `${item.labelX}px`,
      top: `${item.labelY}px`,
      width: `${item.labelWidth}px`,
      height: `${item.labelHeight}px`,
    });
    const startEdgeLabelDrag = (edge, item, event) => {
      if (edgeResizeState) return;
      edgeDragState = {
        id: edge.id,
        x: event.clientX,
        y: event.clientY,
        labelX: item.labelX,
        labelY: item.labelY,
        labelWidth: item.labelWidth,
        labelHeight: item.labelHeight,
        moved: false,
      };
      window.addEventListener("mousemove", moveEdgeLabel);
      window.addEventListener("mouseup", stopEdgeLabelDrag);
    };
    const moveEdgeLabel = event => {
      if (!edgeDragState) return;
      const dx = (event.clientX - edgeDragState.x) / view.value.scale;
      const dy = (event.clientY - edgeDragState.y) / view.value.scale;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) edgeDragState.moved = true;
      updateEdgeLabelLayout(edgeDragState.id, {
        label_x: roundLayout(edgeDragState.labelX + dx),
        label_y: roundLayout(edgeDragState.labelY + dy),
        label_width: edgeDragState.labelWidth,
        label_height: edgeDragState.labelHeight,
      });
    };
    const stopEdgeLabelDrag = async () => {
      if (!edgeDragState) return;
      const state = edgeDragState;
      edgeDragState = null;
      window.removeEventListener("mousemove", moveEdgeLabel);
      window.removeEventListener("mouseup", stopEdgeLabelDrag);
      if (state.moved) await saveEdgeLabelLayout(currentEdgeLabelLayout(state.id, state));
    };
    const startEdgeLabelResize = (edge, item, event) => {
      edgeResizeState = {
        id: edge.id,
        x: event.clientX,
        y: event.clientY,
        labelX: item.labelX,
        labelY: item.labelY,
        labelWidth: item.labelWidth,
        labelHeight: item.labelHeight,
      };
      window.addEventListener("mousemove", moveEdgeLabelResize);
      window.addEventListener("mouseup", stopEdgeLabelResize);
    };
    const moveEdgeLabelResize = event => {
      if (!edgeResizeState) return;
      const dx = (event.clientX - edgeResizeState.x) / view.value.scale;
      const dy = (event.clientY - edgeResizeState.y) / view.value.scale;
      updateEdgeLabelLayout(edgeResizeState.id, {
        label_x: edgeResizeState.labelX,
        label_y: edgeResizeState.labelY,
        label_width: clampEdgeLabelWidth(edgeResizeState.labelWidth + dx),
        label_height: clampEdgeLabelHeight(edgeResizeState.labelHeight + dy),
      });
    };
    const stopEdgeLabelResize = async () => {
      if (!edgeResizeState) return;
      const state = edgeResizeState;
      edgeResizeState = null;
      window.removeEventListener("mousemove", moveEdgeLabelResize);
      window.removeEventListener("mouseup", stopEdgeLabelResize);
      await saveEdgeLabelLayout(currentEdgeLabelLayout(state.id, state));
    };
    const zoomBy = delta => {
      const next = Math.max(0.4, Math.min(1.6, Math.round((view.value.scale + delta) * 100) / 100));
      view.value.scale = next;
    };
    const onWheel = event => {
      zoomBy(event.deltaY > 0 ? -0.02 : 0.02);
    };
    onBeforeUnmount(() => {
      destroyLineageChart();
      stopPan();
      window.removeEventListener("mousemove", moveNode);
      window.removeEventListener("mouseup", stopNodeDrag);
      window.removeEventListener("mousemove", moveEdgeLabel);
      window.removeEventListener("mouseup", stopEdgeLabelDrag);
      window.removeEventListener("mousemove", moveEdgeLabelResize);
      window.removeEventListener("mouseup", stopEdgeLabelResize);
    });

    const metricRows = edge => {
      const metrics = edge?.perf?.metrics || {};
      const defs = [
        ["e2e_ms", "端到端"],
        ["compute_ms", "计算"],
      ];
      return defs.map(([key, label]) => ({ key, label, ...(metrics[key] || {}) }));
    };
    const formatMs = value => Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)} ms` : "-";
    const formatCount = value => {
      if (!Number.isFinite(Number(value))) return "-";
      const number = Number(value);
      return Number.isInteger(number) ? String(number) : number.toFixed(1);
    };
    const formatSignedMs = value => {
      if (!Number.isFinite(Number(value))) return "-";
      const number = Number(value);
      return `${number > 0 ? "+" : ""}${number.toFixed(2)} ms`;
    };
    const formatSignedCount = value => {
      if (!Number.isFinite(Number(value))) return "-";
      const number = Number(value);
      const text = Number.isInteger(number) ? String(number) : number.toFixed(1);
      return `${number > 0 ? "+" : ""}${text}`;
    };
    const formatNodeMetricNumber = (node, key, kind = "ms") => {
      if (node?.status !== "done") return "-";
      if (kind === "count") return formatCount(node?.[key]);
      const number = Number(node?.[key]);
      return Number.isFinite(number) ? number.toFixed(2) : "-";
    };
    const formatNodeMetricValue = (node, key, kind = "ms") => {
      if (node?.status !== "done") return "-";
      return kind === "count" ? formatCount(node?.[key]) : formatMs(node?.[key]);
    };
    const numericMetric = value => {
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    };
    const incomingEdgeFor = node => edges.value.find(edge => edge.child_job_id === node?.id) || null;
    const isBaselineNode = node => Boolean(node?.id) && !incomingEdgeFor(node);
    const isBestNode = node => node?.status === "done" && bestNodeId.value === node?.id;
    const parentNodeFor = node => {
      if (!node?.id) return null;
      const parentEdge = incomingEdgeFor(node);
      return parentEdge ? nodeById.value[parentEdge.parent_job_id] || null : null;
    };
    const nodeMetricDelta = (node, key) => {
      const childValue = numericMetric(node?.[key]);
      const parentValue = numericMetric(parentNodeFor(node)?.[key]);
      if (childValue === null || parentValue === null) return null;
      return childValue - parentValue;
    };
    const nodeMetricDeltaPct = (node, key) => {
      const delta = nodeMetricDelta(node, key);
      const parentValue = numericMetric(parentNodeFor(node)?.[key]);
      if (delta === null || parentValue === null || parentValue === 0) return null;
      return Math.round((delta / Math.abs(parentValue)) * 1000) / 10;
    };
    const nodeMetricChipText = (node, key, label, kind = "ms") => {
      if (node?.status !== "done") return "";
      const delta = nodeMetricDelta(node, key);
      if (delta === null) return "";
      if (kind === "count") {
        if (delta === 0) return `${label} 0`;
        return `${label} ${delta < 0 ? "▼" : "▲"}${Math.abs(delta).toFixed(delta % 1 ? 1 : 0)}`;
      }
      const pct = nodeMetricDeltaPct(node, key);
      const pctText = compactPctDeltaText(pct);
      return pctText ? `${label} ${pctText}` : "";
    };
    const nodePrimaryDeltaText = node => {
      if (node?.status !== "done" || nodeMetricDelta(node, "compute_ms") === null) return "";
      return compactPctDeltaText(nodeMetricDeltaPct(node, "compute_ms"));
    };
    const nodeMetricChipClass = (node, key) => ["exp-delta", metricDeltaClass(nodeMetricDelta(node, key))];
    const formatNodeMetric = (node, key, kind = "ms") => {
      const value = kind === "count" ? formatCount(node?.[key]) : formatMs(node?.[key]);
      const delta = nodeMetricDelta(node, key);
      if (delta === null) return value;
      const deltaText = kind === "count" ? formatSignedCount(delta) : formatSignedMs(delta);
      return `${value} (${deltaText})`;
    };
    const formatHotKernelMetric = (node, kernel) => {
      const value = numericMetric(kernel?.avg_dur_ms);
      const absolute = formatMs(value);
      const serverDelta = numericMetric(kernel?.parent_delta_ms);
      if (serverDelta !== null) return `${absolute} (${formatSignedMs(serverDelta)})`;
      const parent = parentNodeFor(node);
      if (!parent || value === null) return absolute;
      const parentValue = numericMetric(parent.kernel_durations?.[kernel?.name]) ?? 0;
      return `${absolute} (${formatSignedMs(value - parentValue)})`;
    };
    const formatPct = value => Number.isFinite(Number(value)) ? `${Number(value) > 0 ? "+" : ""}${Number(value).toFixed(1)}%` : "n/a";
    const formatMetricPair = metric => `${formatMs(metric.parent)} → ${formatMs(metric.child)}`;
    const metricDeltaClass = value => {
      if (!Number.isFinite(Number(value))) return "neutral";
      return Number(value) < 0 ? "good" : Number(value) > 0 ? "bad" : "neutral";
    };
    const nodeMetricDetailRow = (node, key, label, kind = "ms", tone = "time") => {
      const delta = nodeMetricDelta(node, key);
      return {
        key,
        label,
        tone,
        value: kind === "count" ? formatCount(node?.[key]) : formatMs(node?.[key]),
        deltaText: delta === null ? "" : `(${kind === "count" ? formatSignedCount(delta) : formatSignedMs(delta)})`,
        deltaClass: metricDeltaClass(delta),
      };
    };
    const hotKernelDetailRow = (node, kernel, index = 0) => {
      const value = numericMetric(kernel?.avg_dur_ms);
      const serverDelta = numericMetric(kernel?.parent_delta_ms);
      const parent = parentNodeFor(node);
      const parentValue = parent ? numericMetric(parent.kernel_durations?.[kernel?.name]) : null;
      const delta = serverDelta !== null
        ? serverDelta
        : (parent && value !== null ? value - (parentValue ?? 0) : null);
      const name = String(kernel?.name || "");
      return {
        key: `hot-${index}-${name}`,
        label: `hot · ${shortKernelName(name)}`,
        title: name,
        tone: "hot",
        value: formatMs(value),
        deltaText: delta === null ? "" : `(${formatSignedMs(delta)})`,
        deltaClass: metricDeltaClass(delta),
      };
    };
    const edgePrimaryMetricEntry = edge => {
      const metrics = edge?.perf?.metrics || {};
      if (metrics.compute_ms && numericMetric(metrics.compute_ms.delta_pct) !== null) {
        return { key: "compute", metric: metrics.compute_ms };
      }
      if (metrics.e2e_ms && numericMetric(metrics.e2e_ms.delta_pct) !== null) {
        return { key: "e2e", metric: metrics.e2e_ms };
      }
      return { key: "compute", metric: {} };
    };
    const deltaMetric = edge => edgePrimaryMetricEntry(edge).metric || {};
    const deltaClass = edge => ["exp-delta", metricDeltaClass(deltaMetric(edge).delta_pct)];
    const edgeDeltaChipText = edge => {
      const entry = edgePrimaryMetricEntry(edge);
      const text = compactPctDeltaText(entry.metric?.delta_pct);
      return text ? `${entry.key} ${text}` : `${entry.key} n/a`;
    };
    const nodeOptimizationDelta = node => {
      const computeDelta = nodeMetricDelta(node, "compute_ms");
      if (computeDelta !== null) return computeDelta;
      return nodeMetricDelta(node, "e2e_ms");
    };
    const edgeOptimizationDelta = edge => {
      const computeDelta = numericMetric(edge?.perf?.metrics?.compute_ms?.delta_pct);
      if (computeDelta !== null) return computeDelta;
      return numericMetric(edge?.perf?.metrics?.e2e_ms?.delta_pct);
    };
    const nodeOptimizationClass = node => {
      if (isBestNode(node)) return "result-best";
      const deltaPct = nodeOptimizationDelta(node);
      if (deltaPct === null || deltaPct === 0) return "";
      return deltaPct < 0 ? "result-good" : "result-bad";
    };
    const edgeOptimizationClass = edge => {
      const deltaPct = edgeOptimizationDelta(edge);
      if (deltaPct === null || deltaPct === 0) return "";
      return deltaPct < 0 ? "result-good" : "result-bad";
    };
    const draftVariableInsertItems = computed(() => draftVariables().map((variable, index) => ({
      key: `${index}-${variable.name}-${variable.from}-${variable.to}`,
      label: variableDisplayLabel(variable),
      text: variableListLine(variable),
    })));
    const appendLinesToDraftTitle = lines => {
      const text = (lines || []).map(line => String(line || "").trim()).filter(Boolean).join("\n");
      if (!text) return;
      const current = String(edgeDraft.value.title || "").replace(/\s+$/, "");
      edgeDraft.value = {
        ...edgeDraft.value,
        title: current ? `${current}\n${text}` : text,
      };
    };
    const appendDraftVariableToTitle = text => appendLinesToDraftTitle([text]);
    const appendAllDraftVariablesToTitle = () => appendLinesToDraftTitle(draftVariableInsertItems.value.map(item => item.text));
    const edgeLabelVariables = edge =>
      edge?.id && edge.id === selectedEdgeId.value
        ? cleanVariables(draftVariables())
        : cleanVariables(edge?.variables || []);
    const edgeLabelDescription = edge =>
      String(edge?.id && edge.id === selectedEdgeId.value ? edgeDraft.value.title || "" : edge?.title || "").trim();
    const edgeLabelSizingText = edge => {
      const description = edgeLabelDescription(edge);
      const variables = edgeLabelVariables(edge).map(variableDisplayLabel);
      return [description, ...variables, (!description && !variables.length ? "优化关系" : "")].filter(Boolean).join("\n");
    };
    const edgeCanvasText = edge => {
      const description = edgeLabelDescription(edge);
      if (description) return description;
      const variableText = variableListText(edgeLabelVariables(edge));
      return variableText || "优化关系";
    };
    const edgeLabel = edge => edgeCanvasText(edge);
    const looksLikeFileName = value => /\.(json|gz|zip|tgz|tar|trace|pt)(\.|$)/i.test(String(value || ""));
    const shortId = id => String(id || "").slice(0, 8);
    const shortKernelName = name => {
      const text = String(name || "").replace(/^void\s+/, "");
      const beforeArgs = text.split("(")[0] || text;
      const parts = beforeArgs.split("::").filter(Boolean);
      const leaf = parts[parts.length - 1] || beforeArgs;
      return leaf.length > 34 ? `${leaf.slice(0, 31)}...` : leaf;
    };
    const nodeTitle = node => {
      const label = String(node?.label || "").trim();
      const fileName = String(node?.file_a_name || "").trim();
      if (label && label !== fileName && !looksLikeFileName(label)) return label;
      return `Job ${shortId(node?.id)}`;
    };
    const nodeEditableName = node => {
      const label = String(node?.label || "").trim();
      return label || nodeTitle(node);
    };
    const resetNodeNameDraft = () => {
      const value = selectedNode.value ? nodeEditableName(selectedNode.value) : "";
      nodeNameDraft.value = value;
      nodeNameOriginal.value = value;
    };
    const nodeNameDirty = computed(() => nodeNameDraft.value.trim() !== nodeNameOriginal.value.trim());
    const patchJobInList = (listRef, job) => {
      const index = listRef.value.findIndex(item => item.id === job.id);
      if (index >= 0) listRef.value.splice(index, 1, { ...listRef.value[index], label: job.label });
    };
    const syncRenamedJob = job => {
      if (!job?.id) return;
      patchJobInList(nodes, job);
      patchJobInList(unconnected, job);
      patchJobInList(candidateJobs, job);
    };
    const updateNodeAttachments = (nodeId, attachments) => {
      const normalized = Array.isArray(attachments) ? attachments : [];
      const patchList = listRef => {
        const index = listRef.value.findIndex(item => item.id === nodeId);
        if (index >= 0) listRef.value.splice(index, 1, { ...listRef.value[index], attachments: normalized });
      };
      patchList(nodes);
      patchList(unconnected);
    };
    const nodeAttachmentUrl = attachment => {
      const nodeId = selectedNode.value?.id || "";
      const attachmentId = attachment?.id || "";
      if (!nodeId || !attachmentId) return "";
      return attachment.url || `/api/jobs/${encodeURIComponent(nodeId)}/experiment-attachments/${encodeURIComponent(attachmentId)}`;
    };
    const uploadNodeAttachment = async event => {
      const input = event?.target;
      const file = input?.files?.[0];
      if (!file || !selectedNode.value?.id) {
        if (input) input.value = "";
        return;
      }
      if (file.size > 500 * 1024 * 1024) {
        showToast("附件不能超过 500MB", "error");
        input.value = "";
        return;
      }
      const nodeId = selectedNode.value.id;
      const form = new FormData();
      form.append("file", file);
      nodeAttachmentUploading.value = true;
      try {
        const payload = await fetchJson(
          `/api/jobs/${encodeURIComponent(nodeId)}/experiment-attachments`,
          { method: "POST", credentials: "include", body: form },
          "上传附件失败",
        );
        updateNodeAttachments(nodeId, payload.attachments || []);
        showToast("附件已上传", "success");
      } catch (e) {
        showToast(normalizeApiError(e, "上传附件失败"), "error");
      } finally {
        nodeAttachmentUploading.value = false;
        if (input) input.value = "";
      }
    };
    const downloadNodeAttachment = attachment => {
      const url = nodeAttachmentUrl(attachment);
      if (!url || typeof document === "undefined") return;
      const link = document.createElement("a");
      link.href = url;
      link.download = attachment?.filename || "attachment";
      link.rel = "noopener";
      document.body.appendChild(link);
      link.click();
      link.remove();
    };
    const deleteNodeAttachment = async attachment => {
      const nodeId = selectedNode.value?.id || "";
      const attachmentId = attachment?.id || "";
      if (!nodeId || !attachmentId) return;
      const name = attachment?.filename || "该附件";
      if (typeof window !== "undefined" && !window.confirm(`确定删除 ${name}？`)) return;
      nodeAttachmentDeletingId.value = attachmentId;
      try {
        const response = await fetch(
          `/api/jobs/${encodeURIComponent(nodeId)}/experiment-attachments/${encodeURIComponent(attachmentId)}`,
          { method: "DELETE", credentials: "include" },
        );
        if (!response.ok) {
          const payload = await readJsonResponse(response, {});
          throw new ApiRequestError(apiErrorMessage(response, payload, "删除附件失败"), {
            status: response.status,
            authExpired: response.status === 401,
          });
        }
        updateNodeAttachments(
          nodeId,
          selectedNodeAttachments.value.filter(item => item.id !== attachmentId),
        );
        showToast("附件已删除", "success");
      } catch (e) {
        showToast(normalizeApiError(e, "删除附件失败"), "error");
      } finally {
        nodeAttachmentDeletingId.value = "";
      }
    };
    const saveNodeName = async () => {
      if (!selectedNode.value?.id || nodeNameSaving.value || !nodeNameDirty.value) return;
      nodeNameSaving.value = true;
      try {
        const updated = await fetchJson(
          `/api/jobs/${encodeURIComponent(selectedNode.value.id)}`,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            credentials: "include",
            body: JSON.stringify({ label: nodeNameDraft.value.trim() }),
          },
          "保存节点名称失败",
        );
        syncRenamedJob(updated);
        const nextName = nodeEditableName({ ...selectedNode.value, label: updated.label });
        nodeNameDraft.value = nextName;
        nodeNameOriginal.value = nextName;
        await refreshSidebarData();
        showToast("节点名称已更新", "success");
      } catch (e) {
        showToast(normalizeApiError(e, "保存节点名称失败"), "error");
      } finally {
        nodeNameSaving.value = false;
      }
    };
    const nodeTitleById = id => {
      const node = nodeById.value[id] || candidateOptions.value.find(item => item.id === id);
      return node ? nodeTitle(node) : `Job ${shortId(id)}`;
    };
    const jobOptionLabel = job => `${nodeTitle(job)} · ${statusText(job.status)}`;
    const isNodeHighlighted = node => !hasSelection.value || lineage.value.nodeIds.has(node.id);
    const isEdgeHighlighted = edge => !hasSelection.value || lineage.value.edgeIds.has(edge.id) || hoverEdgeId.value === edge.id;
    watch(lineageMetricOptions, options => {
      if (!options.some(item => item.key === lineageMetricKey.value)) {
        lineageMetricKey.value = options[0]?.key || "compute_ms";
      }
    }, { immediate: true });
    watch(() => lineageChartCanvas.value, () => {
      if (viewMode.value === "chart") scheduleLineageChart();
    });
    watch(
      () => [lineageMetricKey.value, projectMetricTargets.value],
      () => {
        if (!projectMetricTargetSaving.value) refreshProjectMetricTargetDraft();
      },
      { deep: true },
    );
    watch(
      () => [viewMode.value, lineageMetricKey.value, currentMetricTargetValue.value, isDark.value, nodes.value, edges.value, lineageChartEmptyText.value],
      () => {
        if (viewMode.value !== "chart") {
          destroyLineageChart();
          return;
        }
        scheduleLineageChart();
      },
      { deep: true },
    );
    watch(viewMode, () => {
      selectedNodeId.value = "";
      selectedEdgeId.value = "";
      hoverEdgeId.value = "";
      panelCollapsed.value = viewMode.value !== "canvas";
    });
    watch(selectedNodeId, () => {
      resetNodeNameDraft();
      nodeCompareTargetId.value = "";
    });

    watch(() => route.params.pid, () => {
      selectedNodeId.value = "";
      selectedEdgeId.value = "";
      loadGraph();
    }, { immediate: true });

    return {
      viewportRef, lineageChartCanvas, loading, saving, nodes, unconnected, edges, selectedNodeId, selectedEdgeId,
      selectedNode, selectedEdge, selectedEdgePerfNote, hoverEdgeId, showAddEdge, panelCollapsed, addForm, edgeDraft,
      viewMode, lineageMetricKey, lineageMetricOptions, lineageChartEmptyText, lineageChartWarning,
      roiMetricKey, roiMetricOptions, roiGroupMode, roiBarMode, roiSort, currentRoiMetricLabel, roiSkippedEdgeCount, roiEmptyText,
      roiSummary, roiBarItems, sortedRoiRows, roiCombinedEdges,
      projectMetricTargetDraft, projectMetricTargetSaving, projectMetricTargetDirty, currentMetricTargetValue, currentMetricTargetTitle, currentMetricTargetValueLabel,
      nodeNameDraft, nodeNameSaving, nodeNameDirty, nodeCompareTargetId, nodeCompareSaving, nodeCompareTargetOptions,
      nodeAttachmentUploading, nodeAttachmentDeletingId,
      edgeVariableQuick, addVariableQuick, variableNameOptions,
      draftVariableInsertItems, selectedNodeTopKernels, selectedNodeDetailRows, selectedNodeAttachments,
      nodePrimaryDeltaText,
      view, projectName, displayNodes, edgePaths, canvasSize, bestNodeId,
      candidateOptions, hasSelection, sortedNodeMetricRows, nodeMetricSort,
      loadGraph, openAddEdge, closeAddEdge, submitAddEdge, selectNode, selectEdge, openJob,
      saveNodeName, resetNodeNameDraft, uploadNodeAttachment, downloadNodeAttachment, deleteNodeAttachment,
      appendQuickVariable, compareSelectedNode,
      saveProjectMetricTarget, clearProjectMetricTarget,
      saveEdge, deleteSelectedEdge, refreshPerf, createCompare, resetLayout,
      startPan, startNodeDrag, startEdgeLabelDrag, startEdgeLabelResize,
      nodeStyle, nodeWidth, edgeLabelStyle, zoomBy, onWheel,
      edgeLabel, edgeCanvasText, edgeLabelDescription, edgeLabelVariables, variableDisplayLabel, metricRows, formatMs, formatSignedMs, formatPct,
      formatCount, formatNodeMetric, formatNodeMetricNumber, formatNodeMetricValue,
      nodeMetricChipText, nodeMetricChipClass, deltaClass, edgeDeltaChipText,
      formatHotKernelMetric, shortKernelName, nodeTitle, nodeTitleById,
      formatMetricPair, metricDeltaClass, jobOptionLabel, isNodeHighlighted, isEdgeHighlighted,
      isBaselineNode, isBestNode, nodeOptimizationClass, edgeOptimizationClass,
      appendDraftVariableToTitle, appendAllDraftVariablesToTitle,
      setRoiSort, roiSortMark, roiGainClass, isFiniteRoiValue, formatRoiGainMs, formatRoiGainPct, formatHitRate,
      focusRoiGroup, focusEdge, openRoiBestEdge, setNodeMetricSort, nodeMetricSortMark, selectNodeFromRoi,
      fmtDate, fmtDateTime, fmtBytes, statusText,
    };
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Router definition
// ══════════════════════════════════════════════════════════════════════════════

const topVisibleModalMask = () => {
  const masks = [...document.querySelectorAll(".modal-mask")]
    .filter(mask => window.getComputedStyle(mask).display !== "none");
  return masks.reduce((top, mask) => {
    if (!top) return mask;
    const topZ = Number.parseInt(window.getComputedStyle(top).zIndex, 10) || 0;
    const maskZ = Number.parseInt(window.getComputedStyle(mask).zIndex, 10) || 0;
    return maskZ >= topZ ? mask : top;
  }, null);
};

const handleGlobalFocusTrap = event => {
  if (event.key !== "Tab") return;
  const topMask = topVisibleModalMask();
  if (!topMask) return;
  const focusable = [...topMask.querySelectorAll(
    "button:not([disabled]), input:not([disabled]):not([type='hidden']), textarea:not([disabled]), select:not([disabled]), a[href], [tabindex]:not([tabindex='-1'])"
  )].filter(element => element.getClientRects().length && element.getAttribute("aria-hidden") !== "true");
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  const active = document.activeElement;
  if (!topMask.contains(active)) {
    event.preventDefault();
    first.focus();
    return;
  }
  if (event.shiftKey && active === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && active === last) {
    event.preventDefault();
    first.focus();
  }
};

const handleGlobalEscape = event => {
  if (event.key !== "Escape") return;
  let handled = true;
  if (showConfirmModal.value) resolveConfirm(false);
  else if (showAiCodeViewer.value) closeAiCodeViewer();
  else if (showErrorModal.value) showErrorModal.value = false;
  else if (showAiPromptModal.value) closeAiPromptModal();
  else if (showStepReanalysisModal.value) closeStepReanalysisModal();
  else if (showBulkMoveProject.value) showBulkMoveProject.value = false;
  else if (showProjectBulkModal.value) closeProjectBulkModal();
  else if (showCompareModal.value) closeCompareModal();
  else if (showFeedbackComposer.value) closeFeedbackComposer();
  else if (showRenameProject.value) {
    if (!renameProjectLoading.value) showRenameProject.value = false;
  }
  else if (showNewProject.value) {
    if (!newProjectLoading.value) showNewProject.value = false;
  }
  else if (showRenameJob.value) showRenameJob.value = false;
  else if (showMoveProject.value) showMoveProject.value = false;
  else if (showDeletedProjects.value) showDeletedProjects.value = false;
  else if (showSettings.value) closeSettings();
  else if (showStorageManager.value) showStorageManager.value = false;
  else if (showAdminUsage.value) showAdminUsage.value = false;
  else if (showTritonCode.value) showTritonCode.value = false;
  else if (showGuide.value) showGuide.value = false;
  else if (showReleaseNotes.value) showReleaseNotes.value = false;
  else if (showTaskCenter.value) closeTaskCenter();
  else if (isReadingMode.value) exitReadingMode();
  else handled = false;
  if (handled) {
    event.preventDefault();
    event.stopPropagation();
  }
};
window.addEventListener("keydown", handleGlobalEscape);
window.addEventListener("keydown", handleGlobalFocusTrap);

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", component: Home },
    { path: "/feedback/:postId?", name: "feedback", component: Home },
    { path: "/project/:pid/tree", component: ExperimentTree },
    { path: "/job/:id", component: JobDetail },
    { path: "/job/:id/:tab", component: JobDetail },
  ],
});

const resetJobRuntimeState = () => {
  isReadingMode.value = false;
  if (ktChartInst.value)     { ktChartInst.value.destroy();     ktChartInst.value = null; }
  stopAiAnalysisPolling();
  cancelResultTableRequest();
  clearAiDiagnostics();
};

const clearSelectedJobRoute = () => {
  saveResultViewState();
  resetJobRuntimeState();
  loadJobRequestSeq += 1;
  clearInterval(pollTimer);
  pollTimer = null;
  selectedJobId.value = null;
  selectedJobHandle.value = null;
  selectedJob.value = null;
  jobLoading.value = false;
  resultTab.value = DEFAULT_RESULT_TAB;
  resultTableFile.value = "";
  activeResultStateJobId = null;
};

const loadJobRoute = async to => {
  const newJobHandle = to.params?.id || null;
  if (!newJobHandle) return true;

  saveResultViewState();
  resetJobRuntimeState();

  selectedJobId.value = null;
  selectedJobHandle.value = newJobHandle;
  selectedJob.value = null;
  jobLoading.value = true;
  resultTableFile.value = "";

  let loaded;
  try {
    loaded = await loadJob(newJobHandle);
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
    selectedJobHandle.value = null;
    clearAiDiagnostics();
    jobLoading.value = false;
    return { path: "/" };
  }

  const canonicalJobId = selectedJobId.value;
  const requestedTab = to.params?.tab || "";
  const validTabs = availableTabs.value.map(t => t.key);
  const targetTab = resolveResultTab(canonicalJobId, requestedTab, validTabs);
  activeResultStateJobId = canonicalJobId;
  if (targetTab.endsWith(".csv")) {
    await activateCsvTab(targetTab, { updateRoute: false, savePrevious: false });
  } else {
    skipNextResultTabWatch();
    resultTab.value = targetTab;
    restoreResultViewState(canonicalJobId, targetTab);
    rememberResultTabSelection(canonicalJobId, targetTab);
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
  focusCurrentJobInSidebar(selectedJob.value);
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
    return;
  }
  if (route.path === "/") {
    await router.replace({ path: "/" }).catch(() => {});
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

  if (showConfirmModal.value) resolveConfirm(false);
  if (showFeedbackComposer.value && !await closeFeedbackComposer()) return false;

  if (to.name === "feedback") {
    await openFeedbackRoute(to);
    return;
  }
  if (to.path.startsWith("/project/")) {
    clearSelectedJobRoute();
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

  const newJobHandle = to.params?.id || null;

  if (!newJobHandle) {
    // Navigated to home -- clean up
    clearSelectedJobRoute();
    if (from.params?.id || from.path.startsWith("/project/") || from.name === "feedback") {
      loadHomeDashboard({ silent: true });
    }
    return;
  }

  const requestedTabForSameJob = to.params?.tab || "";

  // Same job, just switch tab
  const sameLoadedJob = selectedJob.value && (
    newJobHandle === selectedJobHandle.value ||
    newJobHandle === selectedJobId.value ||
    newJobHandle === jobRouteHandle(selectedJob.value)
  );
  if (sameLoadedJob) {
    selectedJobHandle.value = newJobHandle;
    const canonicalJobId = selectedJobId.value;
    const validTabs = availableTabs.value.map(t => t.key);
    const targetTab = resolveResultTab(canonicalJobId, requestedTabForSameJob, validTabs);
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
    let projectBulkSearchTimer = null;
    let resultTableTimer = null;
    let feedbackImageMutationObserver = null;
    const refreshTaskCenterWhenVisible = () => {
      if (document.visibilityState !== "visible") return;
      if (!authChecked.value || (authRequired.value && !currentUser.value)) return;
      loadTaskCenter({ silent: true });
    };
    const isFeedbackRoute = computed(() => router.currentRoute.value.name === "feedback");
    const showHeaderMotto = computed(() => router.currentRoute.value.path === "/");
    const handleFeedbackImageResize = () => scheduleFeedbackMarkdownImageResize();
    const handleFeedbackImageToggle = event => toggleFeedbackMarkdownImage(event);
    const startFeedbackImageMutationObserver = () => {
      if (typeof MutationObserver === "undefined" || feedbackImageMutationObserver) return;
      const root = document.getElementById("app");
      if (!root) return;
      const hasMarkdownImage = node => (
        node?.nodeType === 1
        && (node.matches?.(".md-image") || node.querySelector?.(".md-image"))
      );
      feedbackImageMutationObserver = new MutationObserver(mutations => {
        if (mutations.some(mutation => Array.from(mutation.addedNodes || []).some(hasMarkdownImage))) {
          scheduleFeedbackMarkdownImageResize();
        }
      });
      feedbackImageMutationObserver.observe(root, { childList: true, subtree: true });
    };

    nextTick(() => {
      startFeedbackImageMutationObserver();
      scheduleFeedbackMarkdownImageResize();
    });
    window.addEventListener("resize", handleFeedbackImageResize);
    document.addEventListener("click", handleFeedbackImageToggle);
    document.addEventListener("click", closeTaskCenter);
    document.addEventListener("visibilitychange", refreshTaskCenterWhenVisible);
    clearInterval(taskCenterPollTimer);
    taskCenterPollTimer = setInterval(refreshTaskCenterWhenVisible, 5000);

    let modalReturnFocus = null;
    const modalVisibilitySources = [
      showCompareModal, showProjectBulkModal, showMoveProject, showBulkMoveProject,
      showRenameJob, showNewProject, showRenameProject, showDeletedProjects,
      showFeedbackComposer, showAdminUsage, showStorageManager, showTritonCode,
      showAiCodeViewer, showErrorModal, showAiPromptModal, showStepReanalysisModal,
      showConfirmModal, showSettings, showGuide, showReleaseNotes,
    ];
    watch(modalVisibilitySources, (values, previousValues) => {
      const hasOpenModal = values.some(Boolean);
      const hadOpenModal = previousValues.some(Boolean);
      if (hasOpenModal && !hadOpenModal) modalReturnFocus = document.activeElement;
      nextTick(() => {
        if (!hasOpenModal) {
          modalReturnFocus?.focus?.({ preventScroll: true });
          modalReturnFocus = null;
          return;
        }
        const topMask = topVisibleModalMask();
        const focusTarget = topMask?.querySelector(
          "[data-modal-autofocus], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), button:not([disabled])"
        );
        focusTarget?.focus?.({ preventScroll: true });
      });
    });

    // Watchers that need to live at the root level
    watch(resultTab, (v, previousTab) => {
      if (v !== "all_kernels_avg.csv" && kernelHostOpDetail.value.kernelName) {
        closeKernelHostOpDetail();
      }
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

    watch(selectedJobId, () => {
      if (kernelHostOpDetail.value.kernelName) closeKernelHostOpDetail();
    });

    watch(filterProject, () => {
      historyGroupsOffset.value = 0;
      historyJobsOffset.value = 0;
      compareJobsOffset.value = 0;
      historySelection.value = [];
      localStorage.setItem("tpa-filter-project", filterProject.value);
      if (suppressSidebarAutoRefresh) return;
      refreshSidebarData();
    });

    watch(historyProjectView, () => {
      historyGroupsOffset.value = 0;
      historyJobsOffset.value = 0;
      historySelection.value = [];
      localStorage.setItem("tpa-history-project-view", historyProjectView.value);
      if (suppressSidebarAutoRefresh) return;
      refreshSidebarData();
    });

    watch(historySearch, () => {
      clearTimeout(historySearchTimer);
      if (suppressSidebarAutoRefresh) return;
      historySearchTimer = setTimeout(() => {
        historyGroupsOffset.value = 0;
        historyJobsOffset.value = 0;
        historySelection.value = [];
        Promise.all([loadHistoryGroups(), loadHistoryJobs()]);
      }, 250);
    });

    watch(compareSearch, () => {
      clearTimeout(compareSearchTimer);
      compareSearchTimer = setTimeout(() => {
        if (!showCompareModal.value) return;
        compareJobsOffset.value = 0;
        loadCompareJobs();
      }, 250);
    });

    watch(projectBulkSearch, () => {
      clearTimeout(projectBulkSearchTimer);
      projectBulkSearchTimer = setTimeout(() => {
        if (!showProjectBulkModal.value) return;
        projectBulkJobsOffset.value = 0;
        loadProjectBulkJobs(true);
      }, 250);
    });

    watch(compareProjectId, () => {
      if (!showCompareModal.value) return;
      resetCompareSelections();
      compareJobsOffset.value = 0;
      loadCompareJobs();
    });

    watch(sidebarWidth, value => localStorage.setItem("tpa-sidebar-width", String(value)));
    watch(sidebarCollapsed, value => localStorage.setItem("tpa-sidebar-collapsed", String(value)));
    watch(sidebarTab, value => localStorage.setItem("tpa-sidebar-tab", value));
    watch(consoleHideWrote, value => localStorage.setItem("tpa-console-hide-wrote", String(value)));
    watch(isReadingMode, value => {
      document.body.classList.toggle("result-reading-active", value);
    });
    watch(collapsedGroups, value => {
      if (!historySearch.value.trim()) {
        localStorage.setItem(projectExpansionStorageKey(), JSON.stringify(value));
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

    watch(autoColWidthsSourceKey, refreshAutoColWidthsIfNeeded, { immediate: true });

    watch([showFeedbackComposer, selectedFeedbackPostId], () => {
      refreshFeedbackMarkdownEditors();
      scheduleFeedbackMarkdownImageResize();
    });
    watch(() => feedbackEditing.value.id, () => {
      refreshFeedbackMarkdownEditors();
      scheduleFeedbackMarkdownImageResize();
    });
    onBeforeUnmount(() => {
      clearInterval(appVersionCheckTimer);
      appVersionCheckTimer = null;
      clearTimeout(projectBulkSearchTimer);
      closeKernelHostOpDetail();
      destroyFeedbackMarkdownEditors();
      window.removeEventListener("resize", handleFeedbackImageResize);
      document.removeEventListener("click", handleFeedbackImageToggle);
      document.removeEventListener("click", closeTaskCenter);
      document.removeEventListener("visibilitychange", refreshTaskCenterWhenVisible);
      clearInterval(taskCenterPollTimer);
      taskCenterPollTimer = null;
      taskCenterController?.abort();
      homeDashboardController?.abort();
      feedbackImageMutationObserver?.disconnect();
      if (feedbackImageResizeRaf) window.cancelAnimationFrame(feedbackImageResizeRaf);
    });

    // Return everything the root template (index.html) needs
    return {
      // Layout/theme
      isDark, themePreference, setThemePreference, toggleTheme,
      userPrefs, saveUserPrefsFromSettings, applySidebarDefaultFromSettings,
      sidebarWidth, sidebarCollapsed, appVersion, isFeedbackRoute, showHeaderMotto,
      currentUserAvatarColor, profileAvatarColor, profileSourceLabel, avatarColorFor,
      toggleSidebar, startSidebarResize,
      authRequired, authChecked, authInitError, currentUser, isAdmin, loginForm, loginRememberUsername, loginLoading, loginError,
      loginCaptchaRequired, loginCaptchaImage,
      retryInitializeApp, submitLogin, refreshLoginCaptcha, logout,
      showTaskCenter, taskCenterLoading, taskCenterError,
      taskCenterActiveJobs, taskCenterFailedJobs, taskCenterActiveCount,
      toggleTaskCenter, closeTaskCenter, loadTaskCenter,
      openTaskCenterJob, copyTaskCenterError, taskCenterJobLabel, taskCenterJobMeta,

      // Sidebar data
      projects,
      selectedFilterProject, projectOptionLabel,
      historyGroupsTotal, historyGroupsLimit, historyGroupsOffset, historyGroupsLoading,
      historyJobs, historyJobsTotal, historyJobsLimit, historyJobsOffset, historyJobsLoading,
      historyProjectGroups, historyAllJobCount, activeHistoryProject, historyListTitle, historyListSubtitle,
      projectQuickViews, activeProjectView, historyProjectView,
      currentSidebarProjectId, isSidebarProjectActive, isRecentProjectActive,
      recentViewedProjectItems, recentProjectSubtitle, clearRecentViewedProjects,
      openRecentProject, openRecentProjectTree,
      historySearch, filterProject, sidebarTab, selectedJobId, selectedJob,
      collapsedGroups, groupedJobs, loadedHistoryJobIds,
      draggingProjectId, dragOverProjectId, projectOrderSaving,
      prevPage, nextPage, navigateToJob, loadHistoryGroupJobs,
      historyBulkMode, historySelection, toggleHistoryBulkMode,
      toggleSelectLoadedHistoryJobs, clearHistorySelection,
      handleHistoryJobClick, selectHistoryProject, selectHistoryProjectView,
      toggleProjectFavorite, projectMenuKey, toggleProjectMenu,
      startProjectBulkMode, openProjectBulkModal,
      openBulkMoveProject, bulkDeleteFiles, bulkDeleteJobs,

      // Compare
      compareSelection, selectedCompareJobs, compareLabel, compareProjectId, compareProjectLabel,
      showCompareModal, openCompareModal, openProjectCompareModal, closeCompareModal,
      batchCompareMode, batchBaselineId, selectedBatchBaseline, selectedBatchCandidates,
      batchCandidateIds, batchCompareLabelPrefix, batchCompareLoading, compareSubmitLoading,
      compareJobs, compareJobsTotal, compareJobsLimit, compareJobsOffset, compareJobsLoading, compareSearch,
      setBatchCompareMode, isCompareJobSelected, compareJobRoleLabel,
      toggleCompareSelect, removeCompareSelection, removeBatchBaseline, removeBatchCandidate,
      submitCompare, submitBatchCompare,
      prevComparePage, nextComparePage,

      // Modals
      showNewProject, newProjectName, newProjectDesc, newProjectShared, newProjectLoading,
      showRenameProject, renameProjectName, renameProjectLoading, projectMutationLoading, openRenameModal,
      confirmRenameProject, deleteProject, shareProject, unshareProject,
      showMoveProject, moveProjectTarget, confirmMoveProject,
      showBulkMoveProject, bulkMoveProjectTarget, confirmBulkMoveProject,
      showProjectBulkModal, projectBulkName, projectBulkJobs, projectBulkJobsTotal,
      projectBulkJobsOffset, projectBulkJobsLoading, projectBulkActionLoading, projectBulkSearch,
      projectBulkSelectedJobs, projectBulkLoadedAllSelected,
      closeProjectBulkModal, loadProjectBulkJobs, toggleProjectBulkSelection,
      toggleLoadedProjectBulkJobs, clearProjectBulkSelection,
      showRenameJob, renameJobName, confirmRenameJob,
      showDeletedProjects, deletedProjects, deletedProjectActionLoading, loadDeletedProjects,
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
      feedbackMessageHtml, feedbackVisibleAttachments, feedbackImageViewer, closeFeedbackImageViewer,
      openFeedbackBoard, refreshFeedbackBoard, loadFeedback, setFeedbackSort, setFeedbackFiles, clearFeedbackForm,
      handleFeedbackPaste, handleFeedbackDrop, handleFeedbackDragOver,
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
      adminUsage, adminUsageCards, adminUsageMetricOptions, adminUsageTrendMetric,
      adminUsageTrendOption, adminUsageTrendPoints, adminUsageTrendPolyline,
      adminUsageTrendArea, adminUsageTrendMaxValue, adminUsageTrendFirstDay,
      adminUsageTrendLastDay, adminUsageUserLastPath,
      openAdminUsage, loadAdminUsage,
      showTritonCode, tritonCodeContent, tritonCodeFilename,
      tritonCodeEditing, tritonCodeEditContent,
      runCustomTriton, editTritonCode, cancelEditTritonCode,
      customRunStatus, allowCodeExecution,
      showAiCodeViewer, aiCodeViewerLoading, aiCodeViewerError,
      aiCodeViewerPath, aiCodeViewerFilename, aiCodeViewerContent,
      aiCodeViewerSize, aiCodeViewerTruncated,
      closeAiCodeViewer, copyAiCodeViewer, downloadAiCodeViewer,
      showGuide, showSettings, profileForm, openSettings, closeSettings, saveProfile,
      showReleaseNotes, releaseNotes, openReleaseNotes,
      showErrorModal, errorModalMsg, errorModalTitle,
      copyTritonCode, copyErrorModal,
      showAiPromptModal, aiAnalysisPrompt, aiPromptForce,
      openAiPromptModal, closeAiPromptModal, confirmAiPromptModal,
      showStepReanalysisModal, stepReanalysisLoading, stepReanalysisLabel,
      stepReanalysisFilterA, stepReanalysisFilterB,
      openStepReanalysisModal, closeStepReanalysisModal, confirmStepReanalysis,
      toasts, showConfirmModal, confirmModal, resolveConfirm,
      openActionMenu, toggleActionMenu, closeActionMenu,

      // Misc
      fmtDate, fmtDateTime, fmtCount, statusIcon, statusText, toggleGroup,
      startProjectDrag, dragProjectOver, dropProject, endProjectDrag,
      createProject,
    };
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Bootstrap
// ══════════════════════════════════════════════════════════════════════════════

const app = createApp(App);
app.use(router);
app.mount("#app");
