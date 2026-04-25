const { createApp, ref, computed, watch, nextTick, onMounted } = Vue;

createApp({
  setup() {
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
      // Rebuild charts so colors update immediately on theme switch
      if (resultTab.value === 'chart' && selectedJob.value?.status === 'done') {
        nextTick(() => buildChart());
      }
    };

    // ── State ──────────────────────────────────────────────────────────────
    const projects      = ref([]);
    const jobs          = ref([]);
    const jobsTotal     = ref(0);
    const jobsLimit     = ref(50);
    const jobsOffset    = ref(0);
    const filterProject = ref("");
    const sidebarTab    = ref("jobs");
    const selectedJobId = ref(null);
    const selectedJob   = ref(null);
    const collapsedGroups = ref({});

    const fileA    = ref(null);
    const fileAName = ref("");
    const submitting = ref(false);
    const uploadProgress = ref(0);
    const form = ref({
      label: "",
      projectId: "",
      saveTritonCsv: true,
      saveTritonCode: true,
    });

    const resultTab   = ref("console");
    const tableSearch = ref("");
    const sortCol     = ref("");
    const sortAsc     = ref(true);
    const colWidths     = ref({});
    const colFilters    = ref({});
    const colFilterOps  = ref({});
    const ktChartInst     = ref(null);
    const ktChart         = ref(null);
    const ktPieChartInst  = ref(null);
    const ktPieChart      = ref(null);
    const ktPieChartInstB = ref(null);
    const ktPieChartB     = ref(null);

    const allowFileDownload = ref(true);

    // ── Triton execution status (keyed by code_path) ──────────────────────────
    // Status: 'idle' | 'running' | 'success' | 'failed'
    const tritonStatus = ref({});

    // ── Error display modal ──────────────────────────────────────────────────
    const showErrorModal = ref(false);
    const errorModalMsg = ref("");
    const errorModalTitle = ref("错误信息");

    // ── Auth ─────────────────────────────────────────────────────────────────
    const userToken = ref(localStorage.getItem("user_token") || null);

    const initAuth = async () => {
      // If we already have a token in localStorage, use it
      let token = localStorage.getItem("user_token");
      if (token) {
        userToken.value = token;
        // Verify this token is valid on server
        const r = await fetch("/api/auth/guest", {
          method: "POST",
          headers: token ? { "X-User-Token": token } : {},
          credentials: "include",
        });
        const data = await r.json();
        if (data.user_token !== token) {
          // Token was invalidated or server created new one, update localStorage
          localStorage.setItem("user_token", data.user_token);
          userToken.value = data.user_token;
        }
      } else {
        // No localStorage token, get one from server
        const r = await fetch("/api/auth/guest", { method: "POST", credentials: "include" });
        const data = await r.json();
        localStorage.setItem("user_token", data.user_token);
        userToken.value = data.user_token;
      }
    };

    // ── Layout ────────────────────────────────────────────────────────────────
    const sidebarWidth     = ref(240);
    const sidebarCollapsed = ref(false);

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

    // ── Project renaming ──────────────────────────────────────────────────────
    const showRenameProject = ref(false);
    const renameProjectId = ref("");
    const renameProjectName = ref("");

    // ── Move job to project ───────────────────────────────────────────────────
    const showMoveProject = ref(false);
    const moveProjectTarget = ref("");

    // ── Deleted projects recovery ──────────────────────────────────────────────
    const showDeletedProjects = ref(false);
    const deletedProjects = ref([]);

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
      if (!confirm("确定恢复该项目？")) return;
      try {
        const r = await fetch(`/api/deleted-projects/${projectId}/restore`, {
          method: "POST",
          credentials: "include",
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          alert("恢复失败: " + (err.detail || err.message || "未知错误"));
          return;
        }
        const restored = await r.json();
        await loadDeletedProjects();
        await loadProjects();
        await loadJobs();
      } catch (e) {
        alert("恢复出错: " + e.message);
      }
    };

    const permanentlyDeleteProject = async (projectId) => {
      if (!confirm("确定永久删除？此操作不可恢复。")) return;
      await fetch(`/api/deleted-projects/${projectId}`, {
        method: "DELETE",
        credentials: "include",
      });
      await loadDeletedProjects();
    };

    // ── Triton code viewer ────────────────────────────────────────────────────
    const showTritonCode = ref(false);
    const tritonCodeContent = ref("");
    const tritonCodeFilename = ref("");
    const tritonCodeEditing = ref(false);
    const tritonCodeEditContent = ref("");
    const customRunStatus = ref("");
    const currentTritonCodePath = ref("");

    // ── Guide ─────────────────────────────────────────────────────────────────
    const showGuide = ref(false);

    const compareSelection  = ref([]);
    const compareLabel      = ref("");
    const compareProjectId  = ref("");

    let pollTimer = null;

    // ── Memoization cache for groupedJobs ─────────────────────────────────────
    let groupedJobsCache = null;
    let groupedJobsCacheKey = null;
    const invalidateGroupedJobsCache = () => {
      groupedJobsCache = null;
      groupedJobsCacheKey = null;
    };

    const getGroupedJobsCacheKey = () =>
      `${filterProject.value}-${jobs.value.length}-${projects.value.length}`;

    // ── Computed ───────────────────────────────────────────────────────────
    const singleJobs = computed(() => {
      let filtered = jobs.value.filter(j => j.mode === "single" && j.status === "done");
      if (filterProject.value) {
        filtered = filterProject.value === "__none__"
          ? filtered.filter(j => !j.project_id)
          : filtered.filter(j => j.project_id === filterProject.value);
      }
      return filtered;
    });

    const groupedJobs = computed(() => {
      const cacheKey = getGroupedJobsCacheKey();
      if (groupedJobsCache && groupedJobsCacheKey === cacheKey) {
        return groupedJobsCache;
      }

      const filtered = filterProject.value
        ? filterProject.value === "__none__"
          ? jobs.value.filter(j => !j.project_id)
          : jobs.value.filter(j => j.project_id === filterProject.value)
        : jobs.value;

      const map = {};
      for (const job of filtered) {
        const p = projects.value.find(p => p.id === job.project_id);
        const projectId = p?.id || "__none__";
        const projectName = p ? p.name : "未分组";

        if (!map[projectId]) {
          map[projectId] = {
            type: "project",
            id: projectId,
            label: projectName,
            jobs: []
          };
        }
        map[projectId].jobs.push(job);
      }

      const result = Object.values(map).sort((a, b) => {
        if (a.id === "__none__") return 1;
        if (b.id === "__none__") return -1;
        return a.label.localeCompare(b.label);
      });

      groupedJobsCacheKey = cacheKey;
      groupedJobsCache = result;
      return result;
    });

    const availableTabs = computed(() => {
      if (!selectedJob.value?.results) return [];
      const res = selectedJob.value.results;
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
        "cncl_ops_avg.csv":         "CNCL Ops",
        "cncl_ops_cmp.csv":         "CNCL 对比",
      };
      for (const [file, label] of Object.entries(csvMap)) {
        if (res[file]) tabs.push({ key: file, label });
      }
      // Add per-step triton tabs (step_N_triton_kernels.csv), at most 3
      const tritonFiles = Object.keys(res).sort()
        .filter(f => f.match(/^step_\d+_triton_kernels\.csv$/))
        .slice(0, 3);
      for (const file of tritonFiles) {
        const stepNum = file.match(/^step_(\d+)_/)[1];
        tabs.push({ key: file, label: `Triton Step ${stepNum}` });
      }
      return tabs;
    });

    const isTritonStepTab = computed(() => {
      return resultTab.value && resultTab.value.match(/^step_\d+_triton_kernels\.csv$/);
    });

    const currentTable = computed(() => {
      const res = selectedJob.value?.results;
      if (!res || !resultTab.value.endsWith(".csv")) return { fields: [], rows: [] };
      return res[resultTab.value] || { fields: [], rows: [] };
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
      if (tableSearch.value) {
        const q = tableSearch.value.toLowerCase();
        rows = rows.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)));
      }
      for (const [field, val] of Object.entries(colFilters.value)) {
        if (!val) continue;
        const op = colFilterOps.value[field] || '~';
        rows = rows.filter(r => {
          const cell = r[field] ?? '';
          if (op === '~')  return  String(cell).toLowerCase().includes(val.toLowerCase());
          if (op === '!~') return !String(cell).toLowerCase().includes(val.toLowerCase());
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

    // ── Helpers ────────────────────────────────────────────────────────────
    const fmtDate = iso => iso ? iso.replace("T", " ").slice(0, 16) : "";

    const statusIcon = s => ({ pending: "⏳", running: "⟳", done: "✓", error: "✗" }[s] || s);

    const toggleGroup = label => {
      collapsedGroups.value[label] = !collapsedGroups.value[label];
    };

    const deltaCellClass = (field, value) => {
      if (!field.includes("delta") && field !== "pct_change") return "";
      const n = parseFloat(value);
      if (isNaN(n)) return "";
      return n > 0 ? "cell-neg" : n < 0 ? "cell-pos" : "";
    };

    // ── Data fetching ──────────────────────────────────────────────────────
    const loadConfig = async () => {
      const r = await fetch("/api/config");
      const cfg = await r.json();
      allowFileDownload.value = cfg.allow_file_download ?? true;
    };

    const loadProjects = async () => {
      try {
        const r = await fetch("/api/projects", { credentials: "include" });
        if (!r.ok) throw new Error("加载项目失败: HTTP " + r.status);
        projects.value = await r.json();
      } catch (e) {
        console.error("loadProjects error:", e);
      }
    };

    const loadJobs = async () => {
      const params = new URLSearchParams();
      if (filterProject.value) params.set("project_id", filterProject.value);
      params.set("limit", String(jobsLimit.value));
      params.set("offset", String(jobsOffset.value));
      const r = await fetch(`/api/jobs?${params}`, { credentials: "include" });
      const data = await r.json();
      jobs.value = data.data || [];
      jobsTotal.value = data.total || 0;
    };

    const loadJob = async id => {
      const r = await fetch(`/api/jobs/${id}`, { credentials: "include" });
      if (!r.ok) return;
      selectedJob.value = await r.json();
    };

    const selectJob = async id => {
      // Destroy existing charts before loading new job
      if (ktChartInst.value)     { ktChartInst.value.destroy();     ktChartInst.value = null; }
      if (ktPieChartInst.value)  { ktPieChartInst.value.destroy();  ktPieChartInst.value = null; }
      if (ktPieChartInstB.value) { ktPieChartInstB.value.destroy(); ktPieChartInstB.value = null; }

      selectedJobId.value = id;
      tableSearch.value = "";
      sortCol.value = "";
      clearColFilters();
      await loadJob(id);
      if (selectedJob.value?.status === "done") {
        resultTab.value = "console";
      }
      startPoll();
    };

    // Poll while running
    const startPoll = () => {
      clearInterval(pollTimer);
      pollTimer = setInterval(async () => {
        if (!selectedJobId.value) return clearInterval(pollTimer);
        const job = jobs.value.find(j => j.id === selectedJobId.value);
        if (job && (job.status === "done" || job.status === "error")) {
          clearInterval(pollTimer);
          return;
        }
        await loadJob(selectedJobId.value);
        await loadJobs();
        if (selectedJob.value?.status === "done" || selectedJob.value?.status === "error") {
          clearInterval(pollTimer);
          resultTab.value = "console";
        }
      }, 1500);
    };

    // ── Chart ──────────────────────────────────────────────────────────────
    const PIE_COLORS = [
      'rgba(99,102,241,.82)',  'rgba(234,88,12,.82)',   'rgba(16,163,74,.82)',
      'rgba(220,38,38,.82)',   'rgba(168,85,247,.82)',  'rgba(14,165,233,.82)',
      'rgba(245,158,11,.82)',  'rgba(20,184,166,.82)',  'rgba(244,63,94,.82)',
      'rgba(6,182,212,.82)',   'rgba(251,191,36,.82)',  'rgba(52,211,153,.82)',
      'rgba(239,68,68,.82)',   'rgba(139,92,246,.82)',  'rgba(34,197,94,.82)',
      'rgba(249,115,22,.82)',  'rgba(59,130,246,.82)',  'rgba(236,72,153,.82)',
      'rgba(132,204,22,.82)',  'rgba(20,184,166,.82)',
    ];
    // Cycle through PIE_COLORS for any number of labels
    const getColors = n => Array.from({ length: n }, (_, i) => PIE_COLORS[i % PIE_COLORS.length]);

    // Returns theme-aware color tokens for Chart.js
    const chartColors = () => {
      const dark = isDark.value;
      return {
        text:       dark ? '#cbd5e1' : '#475569',
        title:      dark ? '#e2e8f0' : '#1e293b',
        grid:       dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)',
        border:     dark ? '#1e293b' : '#ffffff',
      };
    };

    const buildPie = (canvas, labels, data, title) => {
      const pairs = labels.map((l, i) => ({ l, v: data[i] })).filter(p => p.v > 0);
      if (!pairs.length || !canvas) return null;
      const total = pairs.reduce((s, p) => s + p.v, 0);
      const cc = chartColors();
      return new Chart(canvas, {
        type: 'doughnut',
        data: {
          labels: pairs.map(p => p.l),
          datasets: [{ data: pairs.map(p => p.v),
            backgroundColor: getColors(pairs.length),
            borderWidth: 2, borderColor: cc.border }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            title: { display: true, text: title, font: { size: 13 }, color: cc.title },
            legend: {
              position: 'bottom',
              labels: { font: { size: 11 }, boxWidth: 12, padding: 10, color: cc.text,
                        generateLabels: chart => {
                          const ds = chart.data.datasets[0];
                          return chart.data.labels.map((l, i) => ({
                            text: `${l}  ${(ds.data[i] / total * 100).toFixed(1)}%`,
                            fillStyle: ds.backgroundColor[i],
                            strokeStyle: ds.backgroundColor[i],
                            fontColor: cc.text,
                            hidden: false, index: i,
                          }));
                        }},
            },
            tooltip: { callbacks: { label: ctx => {
              const pct = total ? (ctx.parsed / total * 100).toFixed(1) : 0;
              return ` ${ctx.label}: ${ctx.parsed.toFixed(2)} ms (${pct}%)`;
            }}},
          },
        },
      });
    };

    const buildChart = async () => {
      await nextTick();
      if (!ktChart.value || !selectedJob.value?.results) return;
      const res = selectedJob.value.results;
      const csvKey = res?.["kernel_types_cmp.csv"]
        ? "kernel_types_cmp.csv" : "kernel_types_avg.csv";
      const table = res?.[csvKey];
      if (!table) return;

      // Clean up existing charts first
      if (ktChartInst.value)     { ktChartInst.value.destroy();     ktChartInst.value = null; }
      if (ktPieChartInst.value)  { ktPieChartInst.value.destroy();  ktPieChartInst.value = null; }
      if (ktPieChartInstB.value) { ktPieChartInstB.value.destroy(); ktPieChartInstB.value = null; }

      const isCmp = csvKey === "kernel_types_cmp.csv";

      const labels = table.rows.map(r => r.type);

      // For horizontal bars, make the container tall enough so each bar is ~32px
      if (!isCmp) {
        ktChart.value.parentElement.style.height =
          Math.max(420, labels.length * 32 + 80) + 'px';
      } else {
        ktChart.value.parentElement.style.height = '420px';
      }

      // Bar chart — horizontal (indexAxis:'y') for single trace so type names are readable
      const barColors = getColors(labels.length);
      const datasets = [];
      if (!isCmp) {
        datasets.push({ label: "avg_dur_ms",
          data: table.rows.map(r => parseFloat(r.avg_dur_ms) || 0),
          backgroundColor: barColors, borderRadius: 3, barThickness: 22 });
      } else {
        datasets.push({ label: "A avg_dur_ms",
          data: table.rows.map(r => parseFloat(r.avg_dur_ms_A) || 0),
          backgroundColor: "rgba(99,102,241,0.75)", borderRadius: 3 });
        datasets.push({ label: "B avg_dur_ms",
          data: table.rows.map(r => parseFloat(r.avg_dur_ms_B) || 0),
          backgroundColor: "rgba(234,88,12,0.75)", borderRadius: 3 });
      }
      const durPcts = isCmp
        ? table.rows.map(r => `A:${r.dur_pct_A || ''} B:${r.dur_pct_B || ''}`)
        : table.rows.map(r => r.dur_pct || '');
      const cc = chartColors();
      ktChartInst.value = new Chart(ktChart.value, {
        type: "bar",
        data: { labels, datasets },
        options: {
          indexAxis: isCmp ? 'x' : 'y',   // horizontal for single, vertical for compare
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: isCmp, position: "top",
              labels: { color: cc.text, font: { size: 11 } } },
            title: { display: true, text: "Kernel 类型耗时 (ms)", font: { size: 13 }, color: cc.title },
            tooltip: { callbacks: { afterLabel: (ctx) => `  占比: ${durPcts[ctx.dataIndex]}` } },
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

      // Pie chart(s)
      if (!isCmp) {
        ktPieChartInst.value = buildPie(
          ktPieChart.value, labels,
          table.rows.map(r => parseFloat(r.avg_dur_ms) || 0),
          "耗时占比"
        );
      } else {
        ktPieChartInst.value = buildPie(
          ktPieChart.value, labels,
          table.rows.map(r => parseFloat(r.avg_dur_ms_A) || 0),
          "A 耗时占比"
        );
        ktPieChartInstB.value = buildPie(
          ktPieChartB.value, labels,
          table.rows.map(r => parseFloat(r.avg_dur_ms_B) || 0),
          "B 耗时占比"
        );
      }
    };

    watch(resultTab, v => {
      colWidths.value = {};
      colFilters.value = {};
      colFilterOps.value = {};
      if (v === "chart" && selectedJob.value?.status === "done") {
        nextTick(() => buildChart());
      }
    });
    watch(selectedJob, v => {
      if (v?.status === "done") nextTick(() => { if (resultTab.value === "chart") buildChart(); });
    }, { deep: true });

    // ── Uploads ────────────────────────────────────────────────────────────
    const onFileChange = (e) => {
      const f = e.target.files[0];
      if (!f) return;
      fileA.value = f;
      fileAName.value = f.name;
    };

    const onDrop = (e) => {
      const f = e.dataTransfer.files[0];
      if (!f) return;
      fileA.value = f;
      fileAName.value = f.name;
    };

    const clearFile = () => {
      fileA.value = null;
      fileAName.value = "";
    };

    // ── Submit ─────────────────────────────────────────────────────────────
    const submitJob = () => {
      if (!fileA.value || submitting.value) return;
      submitting.value = true;
      uploadProgress.value = 0;
      const fd = new FormData();
      fd.append("file_a", fileA.value);
      fd.append("label", form.value.label);
      fd.append("project_id", form.value.projectId);
      fd.append("save_triton_csv", form.value.saveTritonCsv);
      fd.append("save_triton_code", form.value.saveTritonCode);

      const xhr = new XMLHttpRequest();
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) uploadProgress.value = Math.round(e.loaded / e.total * 100);
      };
      xhr.onload = async () => {
        try {
          const job = JSON.parse(xhr.responseText);
          await loadJobs();
          await selectJob(job.id);
          fileA.value = null; fileAName.value = "";
          form.value.label = "";
          sidebarTab.value = "jobs";
        } finally {
          submitting.value = false;
          uploadProgress.value = 0;
        }
      };
      xhr.onerror = () => { submitting.value = false; uploadProgress.value = 0; };
      xhr.open("POST", "/api/jobs");
      xhr.withCredentials = true;
      xhr.send(fd);
    };

    // ── Job actions ────────────────────────────────────────────────────────
    const deleteJob = async () => {
      if (!selectedJobId.value) {
        alert("未选中任务，无法删除");
        return;
      }
      if (!confirm("确定删除该任务及所有关联文件？")) return;
      try {
        const response = await fetch(`/api/jobs/${selectedJobId.value}`, { method: "DELETE", credentials: "include" });
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          alert("删除失败: " + (errorData.detail || errorData.message || "未知错误"));
          return;
        }
        selectedJobId.value = null;
        selectedJob.value = null;
        resultTab.value = "console";
        await loadJobs();
      } catch (error) {
        alert("删除出错: " + error.message);
      }
    };

    const deleteFile = async slot => {
      if (!confirm(`确定删除原始 trace 文件？删除后该文件无法参与历史对比。`)) return;
      await fetch(`/api/jobs/${selectedJobId.value}/files/${slot}`, { method: "DELETE", credentials: "include" });
      await loadJob(selectedJobId.value);
    };

    const editLabel = async () => {
      const newLabel = prompt("新备注名称：", selectedJob.value?.label || "");
      if (newLabel === null) return;
      const token = localStorage.getItem("user_token");
      await fetch(`/api/jobs/${selectedJobId.value}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { "X-User-Token": token } : {}),
        },
        credentials: "include",
        body: JSON.stringify({ label: newLabel }),
      });
      await loadJob(selectedJobId.value);
      invalidateGroupedJobsCache();
      await loadJobs();
    };

    const moveProject = () => {
      moveProjectTarget.value = selectedJob.value?.project_id || "";
      showMoveProject.value = true;
    };

    const confirmMoveProject = async () => {
      if (!selectedJobId.value) {
        alert("未选中任务");
        return;
      }
      const token = localStorage.getItem("user_token");
      try {
        const r = await fetch(`/api/jobs/${selectedJobId.value}`, {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { "X-User-Token": token } : {}),
          },
          credentials: "include",
          body: JSON.stringify({ project_id: moveProjectTarget.value || null }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          alert("移动项目失败: " + (err.detail || r.status));
          return;
        }
      } catch (e) {
        alert("移动项目失败: " + e.message);
        return;
      }
      showMoveProject.value = false;
      await loadJob(selectedJobId.value);
      invalidateGroupedJobsCache();
      await loadJobs();
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
      if (!pid) { alert("项目ID无效"); return; }
      // Optimistically update local state immediately
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
        // Revert optimistic update on error
        if (proj) await loadProjects();
        alert("重命名失败: " + e.message);
        return;
      }
      showRenameProject.value = false;
      await loadProjects();
      invalidateGroupedJobsCache();
      if (filterProject.value === pid) {
        await loadJobs();
      }
    };

    const deleteProject = async (projectId) => {
      if (!confirm("确定删除该项目？项目内的任务将同时被删除。删除后10天内可以找回。")) return;
      const r = await fetch(`/api/projects/${projectId}`, {
        method: "DELETE",
        credentials: "include",
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        alert("删除失败: " + (err.detail || err.message || `HTTP ${r.status}`));
        return;
      }
      filterProject.value = "";
      selectedJobId.value = null;
      selectedJob.value = null;
      resultTab.value = "console";
      await loadProjects();
      invalidateGroupedJobsCache();
      await loadJobs();
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
      if (!selectedJobId.value) return;
      window.open(`/api/jobs/${selectedJobId.value}/results/${filename}`);
    };

    const runSingleTriton = async (codePath) => {
      if (!selectedJobId.value || !codePath) return;
      // Set status to running
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
          // Parse efficiency for status
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
          // Update tritonStatus with custom flag
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
      tritonCodeEditContent.value = tritonCodeContent.value;
      tritonCodeEditing.value = true;
    };

    const cancelEditTritonCode = () => {
      tritonCodeEditing.value = false;
      tritonCodeEditContent.value = "";
    };

    const clearInductorCache = async () => {
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

    const downloadTraceFile = slot => {
      if (!selectedJobId.value) return;
      window.open(`/api/jobs/${selectedJobId.value}/files/${slot}`);
    };

    const openInPerfetto = async (slot) => {
      const job = selectedJob.value;
      if (!job) return;
      const fname = (slot === 'a' ? job.file_a_name : job.file_b_name) || `trace_${slot}.json`;
      const PERFETTO = 'https://ui.perfetto.dev';

      // Open Perfetto immediately so the user sees progress
      const win = window.open(PERFETTO);
      if (!win) { alert('请允许浏览器弹出窗口后重试'); return; }

      // Fetch trace in background
      const resp = await fetch(`/api/jobs/${selectedJobId.value}/files/${slot}`, { credentials: "include" });
      if (!resp.ok) { win.close(); return; }
      const buffer = await resp.arrayBuffer();

      const send = () => {
        if (!win.closed)
          win.postMessage({ perfetto: { buffer, title: fname, fileName: fname } }, PERFETTO);
      };

      // Send when Perfetto signals ready (PING), with two timeout fallbacks
      const handler = (e) => {
        if (e.origin !== PERFETTO || !e.data?.perfetto) return;
        window.removeEventListener('message', handler);
        send();
      };
      window.addEventListener('message', handler);
      setTimeout(send, 2000);
      setTimeout(() => { window.removeEventListener('message', handler); send(); }, 8000);
    };

    const viewTritonCode = async (codePath) => {
      if (!selectedJobId.value || !codePath) return;
      currentTritonCodePath.value = codePath;
      const resp = await fetch(`/api/jobs/${selectedJobId.value}/triton-code/${codePath}`, { credentials: "include" });
      if (!resp.ok) { alert("无法加载代码文件"); return; }
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
        alert("已复制到剪贴板");
      } catch (e) {
        // Fallback for older browsers
        const textarea = document.createElement("textarea");
        textarea.value = tritonCodeContent.value;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
        alert("已复制到剪贴板");
      }
    };

    const copyErrorModal = async () => {
      if (!errorModalMsg.value) return;
      try {
        await navigator.clipboard.writeText(errorModalMsg.value);
        alert("已复制到剪贴板");
      } catch (e) {
        const textarea = document.createElement("textarea");
        textarea.value = errorModalMsg.value;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
        alert("已复制到剪贴板");
      }
    };

    // ── Compare ────────────────────────────────────────────────────────────
    const toggleCompareSelect = job => {
      if (!job.file_a_exists) return;
      const idx = compareSelection.value.indexOf(job.id);
      if (idx >= 0) {
        compareSelection.value.splice(idx, 1);
      } else if (compareSelection.value.length < 2) {
        compareSelection.value.push(job.id);
      } else {
        compareSelection.value = [compareSelection.value[1], job.id];
      }
    };

    watch(compareSelection, () => {
      if (compareSelection.value.length === 2) {
        const jobA = jobs.value.find(j => j.id === compareSelection.value[0]);
        const jobB = jobs.value.find(j => j.id === compareSelection.value[1]);
        if (jobA?.project_id && jobA.project_id === jobB?.project_id) {
          compareProjectId.value = jobA.project_id;
        } else {
          compareProjectId.value = "";
        }
      }
    });

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
      compareSelection.value = [];
      compareLabel.value = "";
      sidebarTab.value = "jobs";
      await loadJobs();
      await selectJob(job.id);
    };

    // ── Projects ───────────────────────────────────────────────────────────
    const createProject = async () => {
      if (!newProjectName.value.trim()) return;
      await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          name: newProjectName.value,
          description: newProjectDesc.value,
        }),
      });
      showNewProject.value = false;
      newProjectName.value = "";
      newProjectDesc.value = "";
      await loadProjects();
      invalidateGroupedJobsCache();
      await loadJobs();
    };

    watch(filterProject, () => {
      jobsOffset.value = 0;
      if (filterProject.value) {
        collapsedGroups.value[filterProject.value] = true;
      }
      loadJobs();
    });

    const prevPage = () => {
      if (jobsOffset.value > 0) {
        jobsOffset.value = Math.max(0, jobsOffset.value - jobsLimit.value);
        loadJobs();
      }
    };

    const nextPage = () => {
      if (jobsOffset.value + jobsLimit.value < jobsTotal.value) {
        jobsOffset.value += jobsLimit.value;
        loadJobs();
      }
    };

    onMounted(async () => {
      await initAuth();
      await loadConfig();
      await loadProjects();
      await loadJobs();
    });

    const formatConsole = (text) => {
      if (!text) return '';
      return text.split('\n').map(line => {
        const e = line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        if (/^={3,}/.test(line))  return `<span class="co-hdr">${e}</span>`;
        if (/^-{5,}/.test(line))  return `<span class="co-sep">${e}</span>`;
        if (/^Wrote /.test(line)) return `<span class="co-wrote">${e}</span>`;
        if (/^\s*$/.test(line))   return e;
        // highlight numbers within data lines
        const highlighted = e.replace(
          /(\b\d+\.?\d*%?|\+[\d.]+|[-][\d.]+)/g,
          '<span class="co-num">$1</span>'
        );
        return `<span class="co-line">${highlighted}</span>`;
      }).join('\n');
    };

    return {
      projects, jobs, jobsTotal, jobsLimit, jobsOffset,
      filterProject, sidebarTab, selectedJobId, selectedJob,
      collapsedGroups, groupedJobs, singleJobs,
      prevPage, nextPage,
      fileA, fileAName, submitting, uploadProgress, form,
      resultTab, tableSearch, sortCol, sortAsc, ktChart, ktPieChart, ktPieChartB,
      availableTabs, currentTable, filteredRows,
      showNewProject, newProjectName, newProjectDesc,
      showRenameProject, renameProjectName, openRenameModal, confirmRenameProject, deleteProject,
      showMoveProject, moveProjectTarget,
      compareSelection, compareLabel, compareProjectId,
      fmtDate, statusIcon, toggleGroup, deltaCellClass,
      onFileChange, onDrop, clearFile, submitJob,
      selectJob, deleteJob, deleteFile, editLabel, moveProject, confirmMoveProject,
      setSort, downloadCsv, colWidths, startResize,
      colFilters, colFilterOps, hasColFilters, clearColFilters,
      sidebarWidth, sidebarCollapsed,
      toggleSidebar, startSidebarResize,
      allowFileDownload, downloadTraceFile, openInPerfetto, viewTritonCode, copyTritonCode, copyErrorModal, runSingleTriton, tritonStatus, clearInductorCache,
      isTritonStepTab,
      showTritonCode, tritonCodeContent, tritonCodeFilename, tritonCodeEditing, tritonCodeEditContent,
      runCustomTriton, editTritonCode, cancelEditTritonCode, customRunStatus,
      showGuide, showErrorModal, errorModalMsg, errorModalTitle,
      toggleCompareSelect, submitCompare, createProject,
      showDeletedProjects, deletedProjects, loadDeletedProjects, isDeletedOver10Days, restoreProject, permanentlyDeleteProject,
      isDark, toggleTheme,
      formatConsole,
    };
  },
}).mount("#app");
