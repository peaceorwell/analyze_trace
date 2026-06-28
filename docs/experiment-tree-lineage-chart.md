# 实验树 · 代际折线图视角 实现规格

> 给现有实验树新增一个**折线图视角**：横轴=子孙代数（generation），纵轴=主指标（默认 compute time），每个分支一种颜色，每个点可点击 + 悬停看详情。
> **纯前端**，无需后端改动：数据来自现有 `GET /api/projects/{pid}/experiments`（节点已带各项指标 + edges 的 parent/child 关系）；图表库用已全局加载的 **Chart.js v4**（`web/static/index.html:12`，全局 `Chart`，app 内 `new Chart(...)` 已有先例 `app.js:4216/4347`）。
> 改动范围：仅 `web/static/app.js` 的 `ExperimentTree` 组件 + `web/static/style.css` 的 `.exp-*`。

---

## 0. 概念
- **横轴 X = 子孙代数**：root（baseline，无入边）= 第 0 代；沿 parent→child 往下每层 +1。DAG 下取**从任一 root 到该节点的最长路径深度**（与画布分层布局同口径）。
- **纵轴 Y = 主指标**：默认 `compute_ms`（compute time），**可切换**（见 §4）。
- **每个分支一条折线、一种颜色**：一个分支 = 一条 root→leaf 路径；路径上的节点按代数连成线。
- **每个点可点击 + 悬停**：点击 → 打开该节点 job；悬停 → 显示节点详情（名称、代数、指标值、相对父节点 delta、状态）。

---

## 1. 视图切换（集成点）
- [ ] 在 `ExperimentTree` 工具条加视图切换：`画布` / `折线图`（`viewMode` ref，值 `'canvas' | 'chart'`，默认 `'canvas'`）。
- [ ] 两个视图**共用同一份已加载数据**（`nodes` / `edges`），切换不重新请求。
- [ ] `折线图` 视图右侧仍可保留项目信息面板；主区放一个 `<canvas>` + 指标选择器。
- [ ] 折线图视图不需要画布的平移/缩放/拖拽；Chart.js 自带交互即可。

---

## 2. 数据计算（前端，纯函数）
基于 `nodes`（含 `id,label,status` + 指标）与 `edges`（`parent_job_id,child_job_id`）：

### 2.1 代数（generation / X）
- [ ] 建邻接：`childrenOf[parent] += child`、`parentsOf[child] += parent`。
- [ ] roots = 没有入边的节点。
- [ ] `generation[node]` = 从任一 root 到该节点的**最长路径长度**（root=0）。按拓扑序递推：`gen[child] = max(gen[child], gen[parent]+1)`。DAG 已由后端环检测保证可拓扑排序。
- [ ] 多 root（森林）都从 0 起。

### 2.2 分支（每条折线）
- [ ] 枚举所有 **root→leaf 简单路径**（leaf = 没有出边的节点）；每条路径 = 一个 Chart.js dataset（一条线、一种颜色）。
- [ ] 节点被多条路径共享（主干/合并点）时，会在多条线上各出现一次（线在该点重合），符合预期。
- [ ] **路径数上限**：DAG 多分支多合并时路径可能组合爆炸。设上限（如 `MAX_BRANCHES = 24`）；超出时只画前 N 条 + 顶部提示"分支过多，仅显示部分；可在画布视图查看全图"。
- [ ] 分支命名：用该路径 leaf 节点的 label（或 `分支 N`）。

### 2.3 点
- [ ] 一条线上每个节点一个点：`{ x: generation[node], y: metric[node], nodeId: node.id, label, status }`。
- [ ] 同一路径内代数严格递增但**可能不连续**（某节点因更长祖先链而代数更大）——直接按真实 generation 作 x，线允许跨度，正确反映合并点被推后。
- [ ] **缺指标**（`metric[node] == null`，如未 done）：该点跳过（`spanGaps` 连接相邻有效点）或断开；并在 tooltip 标注"无数据"。

---

## 3. Chart.js 配置
- [ ] `type: 'line'`，`data.datasets` = 每分支一项：
  ```js
  { label: branchName, data: points, borderColor: color, backgroundColor: color,
    pointRadius: 4, pointHoverRadius: 6, tension: 0, spanGaps: true }
  ```
- [ ] X 轴：`type:'linear'`，`ticks.stepSize:1`、整数、`title:'子孙代数'`，从 0 开始。
- [ ] Y 轴：`title` = 当前指标名 + 单位（如 `compute time (ms)`）；`beginAtZero` 视指标而定（ms 类可 false 以放大差异，但需在轴标注）。
- [ ] 多分支共点重叠时，点击/悬停命中用 `interaction: { mode:'nearest', intersect:true }`。
- [ ] 颜色见 §5。
- [ ] 复用 app 既有模式：`ref` 持有 chart 实例，`watch` canvas 元素就绪后创建；切换视图/数据变化时 `chart.destroy()` 重建；组件 `onBeforeUnmount` 销毁（参考 `resetJobRuntimeState` 里 `ktChartInst.destroy()` 的做法）。

### 3.1 悬停 tooltip（节点详情）
- [ ] 自定义 `plugins.tooltip.callbacks`：
  - `title` → `节点 label · 第 {gen} 代`
  - `label` 行 → `{指标名} {value} ms`、`相对父节点 {±delta} ({±pct}%)`、`状态 {status}`、`文件 {file_a_name}`（可截断）。
  - delta 相对该节点在**当前分支里的前一个点**（父）计算；多父时取该路径上的父。
- [ ] tooltip 数字 `toFixed(2)`，pct 一位小数；缺值显示 `—`。

### 3.2 点击 → 打开 job
- [ ] `options.onClick(evt, elements, chart)`：用 `chart.getElementsAtEventForMode(evt,'nearest',{intersect:true},false)` 取命中点 → 取该点 `nodeId` → `router.push({ path: '/job/' + nodeId })`。
- [ ] hover 时 canvas `cursor:pointer`（`onHover` 里按是否命中点切换）。

---

## 4. 主指标选择器（"第一指标"可切换）
- [ ] 折线图上方放一个下拉，选 Y 轴指标；默认 `compute_ms`。
- [ ] 选项来自节点可用指标：`compute_ms`(compute time, 默认)、`e2e_ms`(端到端)、`comm_ms`(通信)、`kernel_count`、`aten_ops_ms` 等；label 用中文。
- [ ] 切换即重算 datasets + 重建 chart；Y 轴标题随之更新。
- [ ] 与画布视图相互独立（画布仍按 e2e 显示主指标即可，不要求联动）。

---

## 5. 配色（分支颜色，须与 app 协调）
- [ ] 分支调色板沿用 app/侧边栏体系，**有序且彼此可分**，例如：
  `['#6366f1'(indigo), '#10b981'(emerald), '#f59e0b'(amber), '#0ea5e9'(sky), '#ef4444'(red), '#8b5cf6'(violet), '#14b8a6'(teal), '#f97316'(orange)]`；分支数超过调色板长度则循环。
- [ ] 折线/点/图例同色；点边白描边提升可读性。
- [ ] 深浅主题都要清晰：网格线、轴文字用现有 `--text3/--border` 等 CSS 变量，不要写死灰色；Chart.js 的 `scales.*.grid.color`、`ticks.color` 用从 `getComputedStyle` 读出的主题色，并在切主题时重建（或直接用半透明色兼顾两态）。
- [ ] 图例（分支名）置于图上方或右侧，可点击切换显隐（Chart.js 默认支持）。

---

## 6. 边界与空态
- [ ] 0 条边：提示"先建立优化关系后查看代际趋势"，不渲染空图。
- [ ] 单链（无分支）：一条线。
- [ ] 某指标全为 null：提示该指标暂无数据，建议换指标。
- [ ] 分支过多（> 上限）：见 §2.2 提示。

---

## 7. 验收
- [ ] 工具条可在 `画布`/`折线图` 间切换，数据不重拉。
- [ ] X 轴为整数代数（baseline=0），Y 轴为所选指标(ms)；切指标即时更新。
- [ ] 多分支各一种颜色且可分；图例可切显隐。
- [ ] 悬停任意点出 tooltip（名称/代数/指标值/相对父 delta/状态/文件）；缺值显示 `—`。
- [ ] 点击任意点跳到对应 `/job/:id`；hover 时鼠标为 pointer。
- [ ] 深浅主题、无 console error；切换视图/项目/指标时无 Chart 实例泄漏（每次重建前 `destroy()`）。
- [ ] `pytest tests/test_web_api.py` 仍全过（纯前端，不应影响）。

## 8. 不动项
- 不改后端、不改 GET 返回结构（所需 `*_ms`/`*_count` 与 edges 已具备）。
- 不引新依赖（Chart.js v4 已加载）。
- 画布视图（DAG）行为保持不变，仅新增并列的折线图视图。
