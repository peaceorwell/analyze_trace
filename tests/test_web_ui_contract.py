from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
STYLE_CSS = (ROOT / "web" / "static" / "style.css").read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_job_share_warns_before_publishing_private_project():
    share_job = _section(APP_JS, "const shareJob = async () => {", "const togglePinJob")

    assert "确认公开整个项目" in share_job
    assert "项目内所有任务都将对其他用户可见" in share_job
    assert share_job.index("await askConfirm") < share_job.index("fetch(`/api/jobs/${jobId}/share`")


def test_destructive_job_actions_keep_the_original_job_id():
    delete_job = _section(APP_JS, "const deleteJob = async () => {", "const deleteFile")
    delete_file = _section(APP_JS, "const deleteFile = async slot => {", "const editLabel")

    for action in (delete_job, delete_file):
        assert "const jobId = selectedJobId.value;" in action
        assert "${selectedJobId.value}" not in action
    assert "fetch(`/api/jobs/${jobId}`" in delete_job
    assert "fetch(`/api/jobs/${jobId}/files/${slot}?force=true`" in delete_file


def test_feedback_draft_and_nested_confirmation_are_protected():
    close_composer = _section(APP_JS, "const feedbackComposerHasDraft", "const closeFeedbackBoard")

    assert "if (feedbackSubmitting.value) return false" in close_composer
    assert "放弃发帖草稿" in close_composer
    assert 'class="modal-mask modal-mask-confirm"' in INDEX_HTML
    assert ".modal-mask-confirm { z-index: 180; }" in STYLE_CSS


def test_project_controls_and_modals_have_keyboard_contracts():
    project_row = _section(INDEX_HTML, 'class="project-tree-node"', 'class="project-tree-jobs"')
    new_project = _section(INDEX_HTML, "<!-- ── New project modal ── -->", "<!-- ── Rename project modal ── -->")

    assert 'class="project-tree-toggle"' in project_row
    assert 'role="button"' not in project_row
    assert '@submit.prevent="createProject"' in new_project
    assert 'type="submit"' in new_project
    assert 'role="dialog"' in new_project
    assert 'window.addEventListener("keydown", handleGlobalEscape);' in APP_JS


def test_refresh_bootstrap_does_not_flash_the_branded_auth_page():
    bootstrap = _section(
        INDEX_HTML,
        '<div v-else-if="!authChecked"',
        '<div v-else-if="authRequired && !currentUser"',
    )

    assert '<div id="app" v-cloak>' in INDEX_HTML
    assert 'class="app-bootstrap"' in bootstrap
    assert "Torch Profiler Analyzer" not in bootstrap
    assert "正在初始化" not in bootstrap
    assert "animation: app-bootstrap-reveal .16s ease .35s forwards" in STYLE_CSS


def test_csv_filter_defaults_to_text_for_the_implicit_contains_operator():
    assert ":type=\"(!colFilterOps[f] || ['~', '!~', '=='].includes(colFilterOps[f])) ? 'text' : 'number'\"" in APP_JS


def test_performance_overview_explains_compute_time_scope():
    assert "设备计算耗时" in APP_JS
    assert "A 计算耗时" in APP_JS
    assert "计算 Delta" in APP_JS
    assert "不等同于模型端到端耗时" in APP_JS
    assert "avg_count 合计" not in APP_JS


def test_header_task_center_tracks_active_and_failed_jobs_globally():
    assert 'class="task-center-popover"' in INDEX_HTML
    assert 'statuses: "pending,running,error"' in APP_JS
    assert "taskCenterPollTimer = setInterval(refreshTaskCenterWhenVisible, 5000)" in APP_JS
    assert "taskCenterActiveJobs" in APP_JS
    assert "copyTaskCenterError" in APP_JS


def test_large_csv_tables_have_safe_rendering_and_full_filtered_export():
    assert "const TABLE_MAX_RENDER_ROWS = 2000" in APP_JS
    assert "导出全部筛选结果" in APP_JS
    assert 'params.set("download", "true")' in APP_JS
    assert "tableFieldLabel(f)" in APP_JS
    assert "loadResultTable()\">重试" in APP_JS
    assert ".data-table thead th:first-child" in STYLE_CSS
    assert "position: sticky; left: 0" in STYLE_CSS


def test_home_page_prioritizes_active_recent_and_project_work():
    home = _section(APP_JS, "const Home = {", "const JobDetail = {")

    assert 'class="home-workbench"' in home
    assert "taskCenterActiveJobs" in home
    assert "homeRecentJobs" in home
    assert "homeRecentProjects" in home
    assert "继续你的性能分析" in home
    assert 'statuses: "done"' in APP_JS


def test_primary_web_interactions_are_keyboard_and_screen_reader_accessible():
    assert '<button class="header-brand"' in INDEX_HTML
    assert ':tabindex="historyBulkMode ? -1 : 0"' in INDEX_HTML
    assert '@keydown.space.prevent="handleHistoryJobClick(job)"' in INDEX_HTML
    assert ':aria-pressed="String(batchCompareMode)"' in INDEX_HTML
    assert 'role="checkbox"' in INDEX_HTML
    assert "const handleGlobalFocusTrap = event =>" in APP_JS
    assert '@keydown.space.prevent="setSort(f, $event)"' in APP_JS
    assert ':aria-sort="sortCol===f' in APP_JS
    assert 'aria-label="选择 A Trace 文件"' in APP_JS
    assert "fmtDate, fmtDateTime, fmtCount, statusIcon, statusText, toggleGroup" in APP_JS
    assert "@media (prefers-reduced-motion: reduce)" in STYLE_CSS
