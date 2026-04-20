const { createApp, ref, computed, watch, nextTick, onMounted } = Vue;

createApp({
  setup() {
    // ── State ──────────────────────────────────────────────────────────────
    const projects      = ref([]);
    const jobs          = ref([]);
    const filterProject = ref("");
    const sidebarTab    = ref("jobs");
    const selectedJobId = ref(null);
    const selectedJob   = ref(null);
    const collapsedGroups = ref({});

    const fileA    = ref(null);
    const fileB    = ref(null);
    const fileAName = ref("");
    const fileBName = ref("");
    const submitting = ref(false);
    const uploadProgress = ref(0);
    const form = ref({
      kernelTypes: "gemm,embedding,pool",
      label: "",
      projectId: "",
      saveTritonCsv: false,
      saveTritonCode: false,
    });

    const resultTab   = ref("console");
    const tableSearch = ref("");
    const sortCol     = ref("");
    const sortAsc     = ref(true);
    const colWidths   = ref({});
    const colFilters  = ref({});
    const ktChartInst = ref(null);
    const ktChart     = ref(null);

    const allowFileDownload = ref(true);

    const showNewProject  = ref(false);
    const newProjectName  = ref("");
    const newProjectDesc  = ref("");

    const showMoveProject   = ref(false);
    const moveProjectTarget = ref("");

    const compareSelection  = ref([]);
    const compareKernelTypes = ref("gemm,embedding,pool");
    const compareLabel      = ref("");
    const compareProjectId  = ref("");

    let pollTimer = null;

    // ── Computed ───────────────────────────────────────────────────────────
    const singleJobs = computed(() =>
      jobs.value.filter(j => j.mode === "single" && j.status === "done")
    );

    const groupedJobs = computed(() => {
      const filtered = filterProject.value
        ? filterProject.value === "__none__"
          ? jobs.value.filter(j => !j.project_id)
          : jobs.value.filter(j => j.project_id === filterProject.value)
        : jobs.value;

      const map = {};
      for (const job of filtered) {
        const p = projects.value.find(p => p.id === job.project_id);
        const label = p ? p.name : "未分组";
        if (!map[label]) map[label] = { label, jobs: [] };
        map[label].jobs.push(job);
      }
      // Projects first, then ungrouped
      const order = [...projects.value.map(p => p.name), "未分组"];
      return order.filter(l => map[l]).map(l => map[l]);
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
      return tabs;
    });

    const currentTable = computed(() => {
      const res = selectedJob.value?.results;
      if (!res || !resultTab.value.endsWith(".csv")) return { fields: [], rows: [] };
      return res[resultTab.value] || { fields: [], rows: [] };
    });

    const hasColFilters = computed(() =>
      Object.values(colFilters.value).some(v => v)
    );

    const filteredRows = computed(() => {
      let rows = currentTable.value.rows || [];
      if (tableSearch.value) {
        const q = tableSearch.value.toLowerCase();
        rows = rows.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)));
      }
      for (const [field, val] of Object.entries(colFilters.value)) {
        if (!val) continue;
        const q = val.toLowerCase();
        rows = rows.filter(r => String(r[field] ?? '').toLowerCase().includes(q));
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
      const r = await fetch("/api/projects");
      projects.value = await r.json();
    };

    const loadJobs = async () => {
      const url = filterProject.value
        ? `/api/jobs?project_id=${filterProject.value}`
        : "/api/jobs";
      const r = await fetch(url);
      jobs.value = await r.json();
    };

    const loadJob = async id => {
      const r = await fetch(`/api/jobs/${id}`);
      if (!r.ok) return;
      selectedJob.value = await r.json();
    };

    const selectJob = async id => {
      selectedJobId.value = id;
      tableSearch.value = "";
      sortCol.value = "";
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
    const buildChart = async () => {
      await nextTick();
      if (!ktChart.value) return;
      const res = selectedJob.value?.results;
      const csvKey = res?.["kernel_types_cmp.csv"]
        ? "kernel_types_cmp.csv" : "kernel_types_avg.csv";
      const table = res?.[csvKey];
      if (!table) return;

      if (ktChartInst.value) ktChartInst.value.destroy();

      const labels = table.rows.map(r => r.type);
      const datasets = [];
      if (table.fields.includes("avg_dur_ms")) {
        datasets.push({ label: "avg_dur_ms", data: table.rows.map(r => parseFloat(r.avg_dur_ms) || 0),
          backgroundColor: "rgba(99,102,241,0.7)" });
      }
      if (table.fields.includes("avg_dur_ms_A")) {
        datasets.push({ label: "A avg_dur_ms", data: table.rows.map(r => parseFloat(r.avg_dur_ms_A) || 0),
          backgroundColor: "rgba(99,102,241,0.7)" });
        datasets.push({ label: "B avg_dur_ms", data: table.rows.map(r => parseFloat(r.avg_dur_ms_B) || 0),
          backgroundColor: "rgba(234,88,12,0.7)" });
      }

      ktChartInst.value = new Chart(ktChart.value, {
        type: "bar",
        data: { labels, datasets },
        options: {
          responsive: true,
          plugins: { legend: { position: "top" }, title: { display: true, text: "Kernel 类型耗时 (ms)" } },
          scales: { y: { beginAtZero: true } },
        },
      });
    };

    watch(resultTab, v => { colWidths.value = {}; colFilters.value = {}; if (v === "chart") buildChart(); });
    watch(selectedJob, v => {
      if (v?.status === "done") nextTick(() => { if (resultTab.value === "chart") buildChart(); });
    });

    // ── Uploads ────────────────────────────────────────────────────────────
    const onFileChange = (e, slot) => {
      const f = e.target.files[0];
      if (!f) return;
      if (slot === "a") { fileA.value = f; fileAName.value = f.name; }
      else              { fileB.value = f; fileBName.value = f.name; }
    };

    const onDrop = (e, slot) => {
      const f = e.dataTransfer.files[0];
      if (!f) return;
      if (slot === "a") { fileA.value = f; fileAName.value = f.name; }
      else              { fileB.value = f; fileBName.value = f.name; }
    };

    const clearFile = slot => {
      if (slot === "a") { fileA.value = null; fileAName.value = ""; }
      else              { fileB.value = null; fileBName.value = ""; }
    };

    // ── Submit ─────────────────────────────────────────────────────────────
    const submitJob = () => {
      if (!fileA.value || submitting.value) return;
      submitting.value = true;
      uploadProgress.value = 0;
      const fd = new FormData();
      fd.append("file_a", fileA.value);
      if (fileB.value) fd.append("file_b", fileB.value);
      fd.append("kernel_types", form.value.kernelTypes);
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
          fileB.value = null; fileBName.value = "";
          form.value.label = "";
          sidebarTab.value = "jobs";
        } finally {
          submitting.value = false;
          uploadProgress.value = 0;
        }
      };
      xhr.onerror = () => { submitting.value = false; uploadProgress.value = 0; };
      xhr.open("POST", "/api/jobs");
      xhr.send(fd);
    };

    // ── Job actions ────────────────────────────────────────────────────────
    const deleteJob = async () => {
      if (!selectedJobId.value) return;
      if (!confirm("确定删除该任务及所有关联文件？")) return;
      await fetch(`/api/jobs/${selectedJobId.value}`, { method: "DELETE" });
      selectedJobId.value = null;
      selectedJob.value = null;
      await loadJobs();
    };

    const deleteFile = async slot => {
      if (!confirm(`确定删除原始 trace 文件？删除后该文件无法参与历史对比。`)) return;
      await fetch(`/api/jobs/${selectedJobId.value}/files/${slot}`, { method: "DELETE" });
      await loadJob(selectedJobId.value);
    };

    const editLabel = async () => {
      const newLabel = prompt("新备注名称：", selectedJob.value?.label || "");
      if (newLabel === null) return;
      await fetch(`/api/jobs/${selectedJobId.value}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: newLabel }),
      });
      await loadJob(selectedJobId.value);
      await loadJobs();
    };

    const moveProject = () => {
      moveProjectTarget.value = selectedJob.value?.project_id || "";
      showMoveProject.value = true;
    };

    const confirmMoveProject = async () => {
      await fetch(`/api/jobs/${selectedJobId.value}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: moveProjectTarget.value || null }),
      });
      showMoveProject.value = false;
      await loadJob(selectedJobId.value);
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

    const downloadTraceFile = slot => {
      if (!selectedJobId.value) return;
      window.open(`/api/jobs/${selectedJobId.value}/files/${slot}`);
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

    const submitCompare = async () => {
      const [a, b] = compareSelection.value;
      const r = await fetch("/api/jobs/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id_a: a, job_id_b: b,
          kernel_types: compareKernelTypes.value,
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
        body: JSON.stringify({ name: newProjectName.value, description: newProjectDesc.value }),
      });
      showNewProject.value = false;
      newProjectName.value = "";
      newProjectDesc.value = "";
      await loadProjects();
    };

    watch(filterProject, loadJobs);

    onMounted(async () => {
      await loadConfig();
      await loadProjects();
      await loadJobs();
    });

    return {
      projects, jobs, filterProject, sidebarTab, selectedJobId, selectedJob,
      collapsedGroups, groupedJobs, singleJobs,
      fileA, fileB, fileAName, fileBName, submitting, uploadProgress, form,
      resultTab, tableSearch, sortCol, sortAsc, ktChart,
      availableTabs, currentTable, filteredRows,
      showNewProject, newProjectName, newProjectDesc,
      showMoveProject, moveProjectTarget, confirmMoveProject,
      compareSelection, compareKernelTypes, compareLabel, compareProjectId,
      fmtDate, statusIcon, toggleGroup, deltaCellClass,
      onFileChange, onDrop, clearFile, submitJob,
      selectJob, deleteJob, deleteFile, editLabel, moveProject,
      setSort, downloadCsv, colWidths, startResize,
      colFilters, hasColFilters,
      allowFileDownload, downloadTraceFile,
      toggleCompareSelect, submitCompare, createProject,
    };
  },
}).mount("#app");
