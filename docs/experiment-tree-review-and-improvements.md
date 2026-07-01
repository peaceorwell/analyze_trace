# 实验树 · Review 修复 & 改进清单

> 基于对 Codex 首版实现的 review（静态审查 + curl 接口实测 + 浏览器渲染 + `pytest`）。
> 当前状态：功能完整，104 个 web 测试全过、无回归；§4.8 清理、环检测、路由参数、生命周期清理均正确。
> 本清单只列**待修复项**与**改进方向**，按优先级排，带勾选框，供 Codex 接续执行。所有行号以 review 当时为准，改动后以函数名为准。

---

## A. 待修复项（P0/P1）

### A1 [性能 · P1] GET 每次重读大量结果文件
- [ ] **问题**：`_read_node_metrics`（`web/server.py:3689`）对**每个入图节点 + 每个未连接 single job** 都读 ~3 个 CSV（`kernel_types_avg` / `aten_ops_avg` / 热点 kernel）+ 解析 `console_out`，且是**同步阻塞 IO 跑在 async 接口里**（`get_project_experiments`，调用点约 `:7485`、`:7551`）。`unconnected` 可能有几十个 job → 每次打开实验树都做几十×3 次磁盘读，既慢又阻塞事件循环。自愈逻辑（`:7437`）对 incomplete 边还会再各读两份，进一步放大。
- **修复方向（任选其一或组合）**：
  - [ ] **节点卡只读 e2e**：GET 里节点/未连接只读 `perfetto_context.json`（单文件，e2e_ms）；`compute_ms/top_kernels/aten_*` 等改为**悬停时按需懒加载**（新增轻接口 `GET /api/jobs/{jid}/experiment-metrics` 或复用现有结果接口）。
  - [ ] **落库缓存**：job 分析完成（done）时算一次完整指标，存到 jobs 表新列或一张 `job_metrics` 缓存表；GET 直接读缓存，避免每次解析 CSV。
  - [ ] 若暂不改结构：至少把 `_read_node_metrics` 的同步文件读放进 `run_in_threadpool`（FastAPI/Starlette）避免阻塞事件循环。
- **验收**：含 ~30 个 single job 的项目，GET `/api/projects/{pid}/experiments` P95 延迟显著下降；并发请求时不再卡其他接口。

### A2 [数据语义 · P1] 跨 trace 比较缺 step 口径校验
- [ ] **问题**：`compute_perf_delta`（`web/server.py:3709`）直接把两份 trace 的 e2e/compute 相减算百分比。但若两端 trace 的 **step 数 / `step_filter` 不同**，就是"苹果比橘子"，界面却给出精确百分比，可能误导。Codex 已加 `step_dur_ms`（每步）部分缓解，但没有不一致提示。
- **修复方向**：
  - [ ] 在 `perf_summary` 增加 `step_meta`：两端的 step 数 / step 名 / step 窗口时长。
  - [ ] 两端 step 数或单步时长口径差异超阈值时，置 `perf_summary.notes` 警告 + 前端边详情面板显示醒目提示（"两端 step 口径不一致，跨度对比仅供参考"）。
  - [ ] 可选：优先用 `step_dur_ms`（每步归一）而非整窗 e2e 作为主指标，并在 UI 标注口径。
- **验收**：用 step 数不同的两份 trace 建边 → 边详情出现口径警告；step 一致时无警告。

### A3 [UI · P2] 节点卡数值被截断
- [ ] **问题**：节点卡指标行如 `Comp… 54.09 ms`、delta `(+1.15 …`、`(−5.82 …` 被截断（见 review 截图）。
- **修复方向**：调 `.exp-node` 卡宽 / 指标行 `grid` 列宽 / 数值用 `tabular-nums` + 合理省略；delta 注释过长时换行或移到 tooltip。
- **验收**：默认缩放下三行指标（E2E/Kernel/Compute）数值与 delta 完整可读，深浅主题均不溢出。

### A4 [健壮性 · P2] GET 带写副作用
- [ ] **问题**：自愈在 `get_project_experiments`（`:7437-7445`）里对 incomplete 边做 `UPDATE`。作为自愈缓存可接受，但严格 REST 语义 GET 不应改状态；并发 GET 可能重复写。
- **修复方向**：保留自愈，但①只在确有变化时写；②或把自愈挪到"job 完成"的回调里（job done 时刷新引用它且未编辑的边），让 GET 纯只读。
- **验收**：建边时 job 未完成 → 之后完成 → 再次 GET 一次即非 incomplete；并发 GET 不产生重复写竞争。

---

## B. 改进方向（增强，非缺陷）

### B1 数据深度与可信
- [ ] perf 摘要扩指标：memcpy、host/device sync、exposed comm、计算占比，而不止 e2e/compute/comm。
- [ ] delta 的方向/单位在 UI 明确标注（负=变快、ms），`top_movers` 中 parent=0 导致 `delta_pct=null` 时用"新增/消失"文案而非空。

### B2 少干活、更聪明
- [ ] **从已有 compare 任务自动建议边**：项目里已存在 `source_job_a/b` 的对比任务 → 在实验树里显示为"待确认边"，一键转成 `experiment_edges` 并回填 `compare_job_id`。
- [ ] 节点指标服务端缓存（同 A1 落库方案，二者合并实现）。

### B3 用上结构化变量（方案 C 的红利）
- [ ] 按变量筛边（如"列出所有 dtype: fp16→bf16 的优化"）。
- [ ] 变量收益小榜单（"哪类变更平均 e2e 收益最大"）。

### B4 画布体验
- [ ] 交叉变多时切 `dagre`（CDN）做布局，渲染层不变。
- [ ] 子树折叠、节点搜索定位、整条优化路径一键高亮。
- [ ] 导出：把树导出为报告 / 图片。
- [ ] 浏览器实测补全：拖拽/缩放坐标回写、分支(多子)/合并(多父)布局压测（review 时仅静态确认）。

### B5 Phase 3（规格已留）
- [ ] 边一键生成详细对比并回填链接（spec §4.6，`POST .../edges/{eid}/compare`，前端处理"对比生成中"态）。
- [ ] 用已接入的 Claude AI 对**整条优化路径**出文字总结（"从 baseline 到最优，关键是这几步…"）。

### B6 可观测
- [ ] `audit_logs` 里 `experiment.*` 动作可在管理端复盘。
- [ ] 前端所有写操作错误统一 `showToast`（已基本到位，复核 layout/resize 路径）。

---

## C. 已验证正确（无需改，存档备查）
- §4.8 删除/移动清理 5 处全到位（delete_job / delete_project / bulk delete / 单条 PATCH 改项目 / bulk_move），移动仅在 `project_id` 真变化时触发。
- 错误码：环 409 / 重复 409 / 自连 400 / 坏 variables 400（curl 实测）。
- delta 数学正确（实测 64.22→61.04 = −5.0%）。
- 前端：路由 `:pid` 未撞 `:id`；`watch(route.params.pid, …, {immediate:true})` 切项目可重载；`onBeforeUnmount` 清理全部拖拽监听；mousemove/mouseup 成对增删；wheel 模板作用域无泄漏。
- `incomplete` 放宽忽略 comm（单卡合理）。
- `pytest tests/test_web_api.py` 104 passed，无回归。

---

## D. 执行建议
1. 先做 **A1 + A2**（性能 + 数据可信，影响最大），二者可与 **B2 缓存** 合并实现。
2. 再做 **A3 + A4**（UI/语义小修）。
3. B 系列按需排期；**B5 Phase 3** 价值高、依赖已就绪（compare 流程 + Claude AI 都现成）。
4. 每项改完跑 `pytest tests/test_web_api.py` 确认无回归，UI 项用浏览器预览复核深浅主题。
