# 实验树 / 优化谱系（Experiment Tree）实现规格

> 面向实现者（Codex）的落地文档。目标：在「同一项目下」把多个 timeline（trace 分析任务）组织成一张**有向无环图（DAG）**。每条边 = 一次优化，记录用户填的**变更项（自变量）**与系统自动算出的**性能变化（因变量）**，可手改描述、链到详细对比数据。画布支持悬停看详情、点击看血缘、拖拽排布。

## 0. 关键设计决策（已拍板）

1. **节点 = 现有的 `single` job**（不新建节点实体；一个 single job 就是一份 timeline 的分析结果）。
2. **图是 DAG**：允许一个节点有多个父 / 多个子（分支 + 合并），建边时必须做环检测。
3. **性能变化「先自动生成、后可手改」**：建边时自动从两端 job 的结果文件算出性能摘要；用户改了描述/摘要后，自动重算不再覆盖被改过的字段。
4. **变量结构化（方案 C）**：每条边带一组「变更项」`variables`（变量名 / 优化前值 / 优化后值），即这次优化改了哪些**自变量**；与自动算出的性能摘要（**因变量**）配对，构成一条完整的「受控实验记录」。变量是边的属性，只在建/改关系时填，不强制给每个节点做全量配置标注。
5. **画布交互**：自动分层布局为默认，叠加 **平移 / 缩放 / 「自动整理」复位 / 节点悬停浮层 / 点击选中并高亮血缘**；并允许**手动拖拽节点微调且持久化坐标**。
   - 坐标持久化用 `experiment_node_layout` 表，**per-project 共享**（同一项目所有协作者看到同一套排布）。
   - **分期**：自动布局 + 平移/缩放 + 悬停/点击属 **MVP**；**拖拽 + 坐标持久化属 MVP+**（紧接 MVP，是独立小特性，不阻塞主流程）。

> 决策 4/5 的两个默认取值（变量=方案 C、坐标=per-project 共享）若要改，只动 §2 的列/表与 §6 的对应 UI 即可。

## 1. 复用的现有代码（不要重复造）

### 后端 `web/server.py`
| 名称 | 签名 / 说明 |
|---|---|
| `get_db()` | `await get_db()` 取 aiosqlite 连接；用完 `await db.close()` |
| `ensure_user_row(db, request) -> Optional[str]` | 取当前 user_token（写操作前调用） |
| `load_accessible_job(db, request, job_id, columns="*") -> Optional[dict]` | 带权限过滤地读单个 job |
| `load_accessible_project(db, request, project_id) -> Optional[dict]` | 带权限读项目 |
| `validate_project_access(db, request, project_id)` | 无权限则抛 `HTTPException(404)` |
| `_job_access_clause(request, alias="") -> (sql, params)` | 生成 job 的权限 WHERE 片段 |
| `result_dir(job_id) -> str` | 返回该 job 结果目录（内含 `perfetto_context.json` 等） |
| `write_audit(db, request, action, *, resource_type="", resource_id="", details=None)` | 审计日志 |
| `row_to_dict(row)` | sqlite Row → dict |
| `enqueue_analysis_job(job_id)` | 把分析任务入队（仅 phase 3 生成 compare 时用） |

权限/写操作的标准骨架（照抄 `compare_jobs` @ `server.py:7021`）：
```python
db = await get_db()
user_token = await ensure_user_row(db, request)
await validate_project_access(db, request, project_id)
# ... 业务 ...
await write_audit(db, request, "experiment.xxx", resource_type="experiment_edge", resource_id=eid, details={...})
await db.commit()
await db.close()
```

### 前端 `web/static/app.js`
- HTTP 封装：`fetchJson(url, options = {}, fallback = "请求失败")`，**所有请求带** `credentials: "include"`；非 2xx 抛 `ApiRequestError`。
- 组件是「普通对象」：`const Xxx = { template: \`...\`, setup() { return {...} } }`，在 router 的 `routes` 数组注册。
- 全局状态是模块级 `ref` / `computed`，组件 `setup()` 直接引用并 `return`。
- 路由：`createWebHashHistory()`，定义在 `app.js` 末尾 `const router = createRouter({ routes: [...] })`（约 `:7199`）。
- 已有错误提示：`showToast(msg, "error")`、`normalizeApiError(e, fallback)`。
- 主区出口：`web/static/index.html:325` 的 `<main class="main"><router-view></router-view></main>`。

### 数据库 `web/db.py`
- 在 `init_db()` 里：第一段 `db.executescript("""...CREATE TABLE IF NOT EXISTS ...""")` 建表；中间一段 `add_column_if_missing(...)` 做列迁移；后面一段建索引。新增表/索引就追加进去即可，**幂等**。

---

## 2. 数据层（`web/db.py`）

在 `init_db()` 第一段 `executescript` 里，和其它 `CREATE TABLE` 并列加入：

```sql
CREATE TABLE IF NOT EXISTS experiment_edges (
    id                  TEXT PRIMARY KEY,
    project_id          TEXT REFERENCES projects(id) ON DELETE CASCADE,
    user_token          TEXT,
    parent_job_id       TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    child_job_id        TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    title               TEXT DEFAULT '',
    description         TEXT DEFAULT '',
    perf_summary        TEXT DEFAULT '',      -- 自动生成的 JSON（见 §3）
    perf_summary_edited INTEGER DEFAULT 0,    -- 1 = 用户改过，refresh 不覆盖
    variables           TEXT DEFAULT '[]',    -- 方案 C 变更项 JSON（见下）
    compare_job_id      TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

`variables`（自变量，用户填）JSON 结构：
```json
[
  {"name": "dtype",  "from": "fp16", "to": "bf16"},
  {"name": "tiling", "from": "64",   "to": "128"}
]
```
- `from` / `to` 一律按字符串存（值可能是数字也可能是枚举/开关），展示时原样回显。
- 允许空数组（用户没填变更项也能建边）。

节点坐标表（MVP+ 用，per-project 共享）：
```sql
CREATE TABLE IF NOT EXISTS experiment_node_layout (
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    job_id      TEXT NOT NULL REFERENCES jobs(id)     ON DELETE CASCADE,
    x           REAL,
    y           REAL,
    pinned      INTEGER DEFAULT 1,   -- 1 = 用户手动摆过，自动布局不再动它
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(project_id, job_id)
);
```

在 `add_column_if_missing` 那一段追加（防止表是早期 MVP 版本建的、缺 `variables` 列）：
```python
await add_column_if_missing(db, "experiment_edges", "variables", "TEXT DEFAULT '[]'")
```

在建索引那一段追加：
```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_expedge_pair    ON experiment_edges(project_id, parent_job_id, child_job_id);
CREATE INDEX        IF NOT EXISTS idx_expedge_project ON experiment_edges(project_id);
CREATE INDEX        IF NOT EXISTS idx_expedge_parent  ON experiment_edges(parent_job_id);
CREATE INDEX        IF NOT EXISTS idx_expedge_child   ON experiment_edges(child_job_id);
```
> `ON DELETE CASCADE`：项目或任一端 job 被删时，边自动消失。注意：SQLite 外键级联需连接开启 `PRAGMA foreign_keys=ON`；若现有连接未开启，则在 §4 的 DELETE job / DELETE project 流程里**显式补删** `experiment_edges`（先确认现库是否开了 FK，没开就手删）。

---

## 3. 自动性能摘要算法（`web/server.py` 新函数）

### 数据来源（每个 single job 的 `result_dir(job_id)/` 下）
| 文件 | 用到的列 | 含义 |
|---|---|---|
| `perfetto_context.json` | `["a"]["dur_ns"]` | profiled step 的端到端耗时（ns） |
| `kernel_types_avg.csv` | `type`, `avg_dur_ms` | 各 kernel 类型平均耗时（ms/step） |
| `cncl_ops_avg.csv` | `op_name`, `avg_dur_ms` | 通信算子耗时（ms/step） |

> 这些文件只有在 job `status='done'` 后才齐全。缺文件要降级（见末尾）。

解析要点（避免 500）：
- 用 `csv.DictReader`；`avg_dur_ms` 用 `float(...)`，遇空/非数字跳过该行。
- `e2e_ms = (json.load(...).get("a") or {}).get("dur_ns")`，缺 `a`/`dur_ns` → `None`。
- 文件不存在 / JSON 解析失败 → 该指标 `None`，不抛异常。

### 函数
```python
def _read_node_metrics(job_id: str) -> dict:
    """读单个 job 的关键指标，缺文件返回 None 值。
    return {
      "e2e_ms":     float|None,   # perfetto_context.json a.dur_ns / 1e6
      "compute_ms": float|None,   # sum(kernel_types_avg.avg_dur_ms)
      "comm_ms":    float|None,   # sum(cncl_ops_avg.avg_dur_ms)
      "by_type":    {type: avg_dur_ms, ...},  # kernel_types_avg
    }"""

def compute_perf_delta(parent_job_id: str, child_job_id: str) -> dict:
    """对比两端，产出存进 experiment_edges.perf_summary 的 JSON。"""
```

`delta_pct = round((child - parent) / parent * 100, 1)`（parent 为 0 或缺值时该指标 `delta_pct=None`）。
时间类指标 **负值 = 变快/变好**。

### `perf_summary` JSON 结构（schema=1）
```json
{
  "schema": 1,
  "generated_at": "2026-06-27T12:00:00Z",
  "metrics": {
    "e2e_ms":     {"parent": 120.4, "child": 100.1, "delta_pct": -16.9},
    "compute_ms": {"parent": 48.3,  "child": 40.2,  "delta_pct": -16.8},
    "comm_ms":    {"parent": 12.0,  "child": 11.5,  "delta_pct": -4.2}
  },
  "top_movers": [
    {"type": "gemm",     "parent_ms": 20.6, "child_ms": 14.1, "delta_ms": -6.5, "delta_pct": -31.6},
    {"type": "attention","parent_ms": 8.26, "child_ms": 8.10, "delta_ms": -0.16,"delta_pct": -1.9}
  ],
  "incomplete": false,
  "notes": ""
}
```
- `top_movers`：对 `by_type` 取两端并集，按 `|delta_ms|` 降序取前 5。
- 任一端指标缺失 → `incomplete=true`，能算的照常算，不能算的填 `null`。
- 所有展示用数字保留 1–2 位小数（用 `round`）。

---

## 4. 后端 REST 接口（`web/server.py`）

> 建议新代码集中放在 compare 相关 endpoint 之后（约 `server.py:7078` 后）。所有写接口都 `write_audit`。错误用 `HTTPException`。

### 4.1 `GET /api/projects/{pid}/experiments`
取整张图（含坐标 + 未入图的单文件任务，供冷启动展示）。
- 校验：`validate_project_access(db, request, pid)`（404）。
- 查 `experiment_edges WHERE project_id=pid`、`experiment_node_layout WHERE project_id=pid`。
- **入图节点** = 所有边里出现过的 `parent_job_id ∪ child_job_id`，每个读 `id,label,file_a_name,created_at,status,mode` + `e2e_ms` + 坐标 `x,y,pinned`（无记录则 `null`，由前端自动布局填）。
  - 节点这里**只需 e2e**，读单文件 `perfetto_context.json` 即可，别对每个节点跑完整 `_read_node_metrics()`（那会多读 2 个 CSV，浪费 IO）。
- **未连接节点** `unconnected` = 本项目里 `mode='single'` 且**未出现在任何边**的 job（精简字段 + `e2e_ms`），用于空画布/侧边「未连接池」。查询带 `_job_access_clause` 权限过滤，`id NOT IN (SELECT parent_job_id FROM experiment_edges WHERE project_id=? UNION SELECT child_job_id FROM experiment_edges WHERE project_id=?)`。
- **自愈未完成的 perf**：对 `perf.incomplete==true` 且 `perf_summary_edited==0` 的边，在 GET 时用 `compute_perf_delta` 重算并回写（处理「建边时 job 还没分析完、后来完成了」的情况，免得永远停在 incomplete）。重算仍不全则保持 incomplete。
- 返回：
```json
{
  "project_id": "...",
  "nodes": [
    {"id":"job1","label":"timeline_a","file_a_name":"a.json.gz","status":"done",
     "created_at":"...","e2e_ms":120.4,"x":260.0,"y":40.0,"pinned":1}
  ],
  "unconnected": [
    {"id":"job9","label":"timeline_x","status":"done","created_at":"...","e2e_ms":98.2}
  ],
  "edges": [
    {"id":"e1","parent_job_id":"job1","child_job_id":"job2","title":"算子融合",
     "description":"...","compare_job_id":null,"perf_summary_edited":0,
     "variables":[{"name":"dtype","from":"fp16","to":"bf16"}],
     "perf": { ...§3 的 JSON 解析后对象, 解析失败则 null }, "created_at":"..."}
  ]
}
```

### 4.2 `POST /api/projects/{pid}/experiments/edges`
建边。
- Body：`{ "parent_job_id": "...", "child_job_id": "...", "title": "", "description": "", "variables": [{"name","from","to"}] }`（title/description/variables 可空）。
- `variables` 入库前校验是数组、每项是含 `name` 的对象（`from`/`to` 缺省按空串）；存 `json.dumps`。
- 校验顺序（不通过即对应错误）：
  1. `validate_project_access(pid)` → 404。
  2. `parent_job_id == child_job_id` → 400「不能自连」。
  3. `load_accessible_job` 两端都存在 → 404；且两端 `project_id == pid` 且 `mode=='single'` → 400「节点必须是本项目的单文件任务」。
  4. 已存在同 `(pid,parent,child)` 边 → 409「关系已存在」。
  5. **环检测**（见 §5）：若加入 `parent→child` 会成环 → 409「会形成循环依赖」。
- 通过后：`perf = compute_perf_delta(parent, child)`；`INSERT`（`id=uuid4`，`user_token`，`perf_summary=json.dumps(perf)`，`perf_summary_edited=0`）。
- `write_audit("experiment.edge_create", resource_type="experiment_edge", resource_id=eid, details={parent,child,pid})`。
- 返回创建出的边（同 4.1 的 edge 结构，含解析后的 `perf`）。

### 4.3 `PATCH /api/experiments/edges/{eid}`
改 title / description / 手动改性能摘要 / 挂 compare。
- 读边 → `validate_project_access(edge.project_id)`（404）。
- Body 任意子集：`{ "title", "description", "variables"(数组), "perf_summary"(对象), "compare_job_id" }`。
- 若带 `perf_summary` → 存其 JSON 并置 `perf_summary_edited=1`（之后 refresh 不覆盖）。
- 若带 `compare_job_id` → 校验该 job 可访问且 `mode=='compare'`。
- `write_audit("experiment.edge_update", ...)`；返回更新后的边。

### 4.4 `DELETE /api/experiments/edges/{eid}` → 204
- 读边 → `validate_project_access` → 删除 → `write_audit("experiment.edge_delete", ...)`。

### 4.5 `POST /api/experiments/edges/{eid}/refresh-perf`
重算自动摘要。
- 读边 → 权限校验。
- 若 `perf_summary_edited==1`：直接返回现状并带 `{"skipped": true}`（不覆盖用户编辑）。
- 否则 `compute_perf_delta` 重算并存，返回更新后的边。

### 4.6 `POST /api/experiments/edges/{eid}/compare`（phase 3，可后做）
为边生成详细对比任务并回填链接。
- 复用 `compare_jobs` 的逻辑：以 `parent` 为 A、`child` 为 B、`project_id=edge.project_id` 创建 `mode='compare'` job，`enqueue_analysis_job(jid)`。
- `UPDATE experiment_edges SET compare_job_id=jid`。
- 返回 `{ "compare_job_id": jid }`。前端据此显示「对比生成中 / 查看对比详情」。

### 4.7 节点坐标（MVP+，拖拽持久化）
坐标随 §4.1 一起返回，写入用下面两个接口；权限都走 `validate_project_access(pid)`。
- `PUT /api/projects/{pid}/experiments/layout`：批量 upsert，Body `{ "positions": [{"job_id","x","y","pinned":1}] }`（拖拽结束时提交被移动的节点，`pinned=1`）。`INSERT ... ON CONFLICT(project_id,job_id) DO UPDATE`。
- `DELETE /api/projects/{pid}/experiments/layout`：清空该项目坐标 = 「自动整理」复位（也可前端只清 `pinned` 后重算，不落库）。
> 校验每个 `job_id` 属于本项目；非法项忽略或 400。写操作 `write_audit("experiment.layout_update", ...)`。

### 4.8 与现有删除/移动流程的集成（**必须做，否则产生脏数据**）
> 已确认：连接**没有** `PRAGMA foreign_keys=ON`（只设了 WAL/synchronous/busy_timeout），所以 `ON DELETE CASCADE` **不生效**，现有代码靠手动级联。新表必须在以下 4 处手动清理：

| 位置 | 现有行为 | 需补充 |
|---|---|---|
| `delete_job` (`server.py:7965`) | `DELETE FROM jobs WHERE id=?` | `DELETE FROM experiment_edges WHERE parent_job_id=? OR child_job_id=?`；`DELETE FROM experiment_node_layout WHERE job_id=?` |
| `delete_project` (`:6103`) | `DELETE FROM jobs WHERE project_id=?` 等 | `DELETE FROM experiment_edges WHERE project_id=?`；`DELETE FROM experiment_node_layout WHERE project_id=?` |
| bulk 删除 (`POST /api/jobs/bulk/delete` `:6578`) | `DELETE FROM jobs WHERE id IN (...)` | 对同一批 id 删 `experiment_edges`（parent/child 命中）+ `experiment_node_layout` |
| **job 改项目**：`PATCH /api/jobs/{jid}`(`:7928`，`project_id in body`) 和 `bulk_move_jobs`(`:6555`) | `UPDATE jobs SET project_id=?` | lineage 是项目级的：job 换项目后其旧边/坐标失配 → **删除该 job 参与的 `experiment_edges` 与 `experiment_node_layout`**（在 project_id 实际变化时） |

> 恢复流程（restore deleted project/job）只恢复 job，不恢复 lineage（边已删）——可接受，文档说明即可。

---

## 5. 环检测（DAG 保证）

在内存里用现有边建邻接表 `adj[parent] -> [child,...]`。判断加入 `(p, c)` 是否成环：
**从 `c` 出发沿 `adj` 能否到达 `p`**；能到达则会成环（因为加上 p→c 后 c…→p→c 闭环）。用 DFS/BFS + visited 集合，O(V+E)。注意 `p==c` 已在 §4.2 单独拦截。

```python
def _would_create_cycle(edges, parent, child) -> bool:
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for e in edges:
        adj[e["parent_job_id"]].append(e["child_job_id"])
    # 从 child 能否走到 parent
    seen, dq = set(), deque([child])
    while dq:
        n = dq.popleft()
        if n == parent:
            return True
        for m in adj[n]:
            if m not in seen:
                seen.add(m); dq.append(m)
    return False
```

---

## 6. 前端（`web/static/`）

### 6.1 路由 & 组件
`app.js` router `routes` 增加（**参数名必须用 `pid`，不能用 `id`**）：
```js
{ path: "/project/:pid/tree", component: ExperimentTree },
```
> ⚠️ 现有 `router.beforeEach`（`app.js:7349`）把 `to.params.id` 当 jobId 去 `loadJobRoute()`。若沿用 `:id`，项目 id 会被当任务加载 → 404 跳回首页。用 `:pid` 后 `to.params.id` 为 undefined，守卫会走「非 job 路由」分支。**另外在 `beforeEach` 顶部（`if (authRequired...)` 之后、`newJobId` 逻辑之前）加显式放行**：
> ```js
> if (to.path.startsWith('/project/')) { clearSelectedJobRoute(); return; }
> ```
- 新组件 `ExperimentTree = { template: \`...\`, setup() {...} }`，放在 `JobDetail` 定义之后、`router` 定义之前。
- `setup` 里取路由参数用 `const route = VueRouter.useRoute()`，`route.params.pid`；编程式跳转用模块级 `router` 或模板 `$router`。

### 6.2 状态与数据加载（`setup`）
- `route.params.id` = projectId；`watch` 之，变化时重载。
- `loadGraph()`：`fetchJson(\`/api/projects/${pid}/experiments\`)` → `nodes`(含 `x,y,pinned`)、`unconnected`、`edges`(含 `variables`,`perf`)。
- `loadCandidates()`：`fetchJson(\`/api/jobs?project_id=${pid}&limit=200\`)`，过滤 `mode==='single'`，供加边选择器（也可直接用 `unconnected` + 入图节点合并）。
- 局部 ref：`nodes, unconnected, edges, positions, selectedNodeId, selectedEdge, hoverNode, view{scale,tx,ty}, showAddEdge, addForm{parent,child,title,description,variables:[]}, saving`。

### 6.3 成品视图（页面长什么样）
一块**分层有向画布**（自上而下）：
- **节点卡**：timeline 名 + 关键指标（`e2e_ms`）+ 状态点；baseline（无入边）和「当前最优」（`e2e_ms` 最小的 done 节点）加标签，最优用 2px 强调边框；失败/缺结果节点降透明度。
- **有向边**：箭头表方向，边上挂「优化项名 + delta 芯片」；芯片颜色见 §6.5。
- **DAG**：分支（一父多子）、合并（多父一子）都渲染。
- 顶部工具条：`标记优化关系`、`自动整理`、缩放 `− / +`。
- 选中节点 → 高亮其上下游血缘路径；选中边 → 右侧详情面板。
- 空画布 / 侧边显示 `unconnected` 池，提示「拖入或新建关系」。

### 6.4 布局 + 平移缩放（MVP，零依赖）
分层有向布局（仅摆放**没有 pinned 坐标**的节点）：
1. 入度为 0 的节点是 roots，`layer=0`；`layer(node)=max(layer(parent))+1`（拓扑序）。
2. 同层按 `created_at` 升序，`x=colIndex*COL_W`，`y=layer*ROW_H`（如 `COL_W=200, ROW_H=140`）。
3. `pinned` 节点用其落库坐标，不参与自动摆放。
- 渲染：外层 `position:relative` 视口 + 内层 `transform: translate(tx,ty) scale(scale)` 的画布层；**节点 div、边 svg、边标签芯片都放在同一个 transform 层内**（保证 pan/zoom 时三者对齐）；边 svg 覆盖整个内容范围、`pointer-events:none`、画 `<path>`+箭头 marker；**边标签是绝对定位 HTML 芯片**（父子中点，`pointer-events:auto`，可点）。缩放时标签随之缩放（可接受；如不想缩放可对芯片反向 `scale(1/scale)`）。
- **平移**：拖画布空白处改 `tx,ty`；**缩放**：滚轮 / 工具条按钮改 `scale`（夹在 0.4–1.6）。
> 规模/交叉变复杂再引入 `dagre`（CDN），渲染层不变。先不引依赖。

### 6.5 交互：悬停 / 点击 / 编辑
- **节点悬停** → 浮层预览（不跳转）：`e2e/compute/comm`、文件名、日期、状态、Top kernel 类型。`e2e_ms` 等已在 GET 返回；`compute/comm/Top类型` 可在 hover 时懒加载该 job 的 `kernel_types_avg.csv`（带缓存，避免重复请求）。
- **节点点击** → 选中 `selectedNodeId`，**高亮其全部祖先/后代边与节点**（前端按 edges 做 BFS）；侧面板显示节点详情 + 「打开完整分析」→ `router.push(\`/job/${id}\`)`。**双击节点**直接打开分析。
- **边悬停** → 快速 delta tooltip；**边点击** → 右侧详情面板：
  - 标题、描述（`<textarea>`）、**变更项编辑**（变量名/前值/后值 多行，可加可删）→ 保存调 `PATCH`（带 `variables`）。
  - 性能摘要：`perf.metrics` 渲染成 delta 芯片（颜色：时间类 `delta_pct<0` 绿、`>0` 红/琥珀、`null`/`incomplete` 灰）；`top_movers` 列小表。
  - `incomplete` → 提示 + 「重新计算」（`POST .../refresh-perf`）。
  - 「查看对比详情 ↗」：有 `compare_job_id` → 跳 `/job/{compare_job_id}`；无 → 「生成详细对比」（`POST .../compare`，phase 3）。
- **加边（标记优化关系）** → 弹窗：parent/child 下拉 + 优化项名 + **变更项多行**（变量名/前/后）+ 描述 → `POST` → 成功 `loadGraph()`，失败 `showToast(normalizeApiError(e),"error")`。提交后画布即显示自动算出的 delta 芯片（自变量↔因变量配对）。
  - 下拉来源 = **本项目所有 `mode='single'` 的 job**（入图节点 + `unconnected` 都要列，因为常从已有节点继续往下连），显示 label + 日期。
- **删边**：详情面板「删除关系」→ `DELETE` → 重载。

### 6.6 拖拽 + 坐标持久化（MVP+）
- 节点卡可拖动（`mousedown`→`mousemove` 改其 `x,y`，按当前 `scale` 换算位移）；`mouseup` 时把被移动的节点以 `pinned=1` 提交 `PUT .../layout`。
- 区分「拖节点」与「平移画布」：在节点卡上起拖=移动节点；在空白起拖=平移。
- 新加入 / 未 pinned 的节点继续自动布局；`自动整理` 调 `DELETE .../layout` 后重算（或前端清 pinned 重排）。
- 拖拽是独立增量，**不做也不影响 MVP**（届时节点用自动坐标、只读）。

### 6.7 入口
- 项目右键菜单（`index.html:262` 那组 `<button>`）加一项：
  `<button type="button" @click="$router.push('/project/'+group.id+'/tree'); closeActionMenu()">实验树</button>`
- （可选）`JobDetail` 顶部操作区加「加入实验树」，预填 child=当前 job。

### 6.8 样式（`web/static/style.css`）
沿用现有设计语言：卡片 `var(--card)`/`var(--border)`/圆角 12px；轻字重（标题 700）；选中靛蓝；delta 芯片用语义色。新增类前缀统一 `exp-`（如 `.exp-canvas`,`.exp-node`,`.exp-edge-label`,`.exp-chip`,`.exp-panel`,`.exp-tooltip`）。悬停浮层用 `var(--shadow)` / 弹层样式；血缘高亮用靛蓝描边。深浅两主题都要可读（用 CSS 变量，别写死颜色）。

---

## 7. 验收标准

### 后端（curl，`AUTH_MODE=none` 默认本地）
1. 迁移后启动无错；`sqlite3 web/storage/jobs.db '.schema experiment_edges'` 有表。
2. 用项目下两个 done 的 single job 建边：
   `curl -XPOST localhost:8181/api/projects/<pid>/experiments/edges -H 'content-type: application/json' -d '{"parent_job_id":"<a>","child_job_id":"<b>","title":"算子融合"}'` → 201，返回里 `perf.metrics.e2e_ms.delta_pct` 是合理数值。
3. 反向再建 `b→a` → 409（环）。重复建 `a→b` → 409（已存在）。自连 → 400。
4. `GET /api/projects/<pid>/experiments` → nodes/edges 正确，节点带 `e2e_ms`。
5. `PATCH` 改 description → 200，`perf_summary_edited` 不受影响；带 `perf_summary` → `perf_summary_edited=1`；之后 `refresh-perf` 返回 `skipped:true`。
6. 建边带 `variables` → GET 原样回显；`PATCH` 改 `variables` → 持久化。
7. `DELETE` → 204，再 GET 不含该边。
8. 缺结果文件的 job 建边 → 仍 201，`perf.incomplete=true`，缺的指标为 `null`。
9. GET 返回 `unconnected` 含本项目里没入图的 single job。
10.（MVP+）`PUT .../layout` 存坐标后 GET 节点带回 `x,y,pinned`；`DELETE .../layout` 后坐标清空。

### 前端
- 项目菜单「实验树」进入 `/project/:id/tree`，渲染出节点 + 带 delta 芯片的边；空项目显示 `unconnected` 池。
- 节点悬停出浮层（不跳转）；单击选中并高亮血缘 + 「打开完整分析」；双击进 job 详情。
- 点边开面板可改 描述 / 变更项 并保存；改完刷新仍在。
- 加边/删边后图实时更新；分支（一父两子）、合并（两父一子）布局正确不重叠；平移缩放可用。
-（MVP+）拖动节点后位置被记住（刷新仍在）；「自动整理」复位。
- 深/浅主题切换均正常、无 console error。

---

## 8. 范围与分期
- **MVP（必做）**：§2 的 `experiment_edges`(含 `variables`) 迁移、§3 自动摘要、§4.1–4.5 接口、§4.8 的**边清理**（delete_job / delete_project / bulk 删除 / job 改项目）、§5 环检测、§6 前端视图（分层布局 + 平移缩放 + 节点悬停浮层 + 点击选中高亮血缘 + 加/删边 + 改描述/变更项 + delta 展示 + `unconnected` 池）。
- **MVP+（紧接 MVP）**：§2 `experiment_node_layout` 表、§4.7 坐标接口、§4.8 中 `experiment_node_layout` 的清理、§6.6 拖拽 + 持久化 + 「自动整理」。
- **Phase 3（后做）**：§4.6 一键生成对比并回填链接；用已接入的 Claude AI 对整条优化路径出文字总结。

## 9. 注意事项
- 自动摘要依赖 job `status='done'` 且结果文件存在，缺失要降级而非报错。
- 不要为渲染引入重型依赖；MVP 手写布局，预留切 `dagre` 的接口。
- **已确认连接未开 `PRAGMA foreign_keys`，`ON DELETE CASCADE` 不生效** → §4.8 的 4 处手动清理是**必做项**，不是可选。
- 所有数字展示前 `round`，避免浮点尾巴。
- 所有写接口 `write_audit`，权限一律走 project 的共享规则（`validate_project_access`）。
- `variables` / `perf_summary` / 坐标都按 JSON 存 TEXT；读出 `json.loads`，解析失败要降级（`variables→[]`、`perf→null`），不要抛 500。
- 节点坐标 per-project 共享：同项目多人看到同一排布；后续若要「每人各自摆」，给 `experiment_node_layout` 主键加 `user_token` 即可，前端不变。

---

## 10. 实施顺序清单（给 Codex）

> 按依赖顺序做，**每个阶段末尾的「关卡」过了再进下一阶段**。括号是对应章节。先准备一个项目 + 至少 2 个 `status='done'` 的 single job 作为测试数据。

### 阶段 0 · 准备
- [ ] 通读本规格，确认能跑通 `python web/server.py`（默认 `AUTH_MODE=none`，端口 8181）。
- [ ] 记下测试用 `项目id` 和两个 `job id`（`sqlite3 web/storage/jobs.db "SELECT id,project_id,mode,status FROM jobs"`）。

### 阶段 1 · 数据层（§2）
- [ ] `init_db()` 第一段加 `experiment_edges`（含 `variables`）建表。
- [ ] `add_column_if_missing(db,"experiment_edges","variables","TEXT DEFAULT '[]'")`。
- [ ] 加 `experiment_node_layout` 建表。
- [ ] 建索引段加 `idx_expedge_*`。
- [ ] **关卡**：重启无错，`.schema experiment_edges` / `experiment_node_layout` 都在。

### 阶段 2 · 后端核心（§3、§4.1–4.5、§5）
- [ ] `_read_node_metrics(job_id)` / e2e 单读（§3，注意缺文件/缺键降级）。
- [ ] `compute_perf_delta(parent,child)`（§3，输出 schema=1 JSON）。
- [ ] `_would_create_cycle(edges,parent,child)`（§5）。
- [ ] `GET /api/projects/{pid}/experiments`（§4.1，含 `unconnected` + 坐标 + 自愈 incomplete）。
- [ ] `POST .../edges`（§4.2，五步校验 + 自动算 perf + 存 `variables`）。
- [ ] `PATCH /api/experiments/edges/{eid}`（§4.3，描述/变更项/手改 perf/挂 compare）。
- [ ] `DELETE /api/experiments/edges/{eid}`（§4.4）。
- [ ] `POST .../edges/{eid}/refresh-perf`（§4.5）。
- [ ] **关卡（curl，§7 后端 1–9）**：建边 201 且 perf 合理；反向/重复/自连分别 409/409/400；GET 正确；PATCH 改 description/variables 持久化；缺结果文件建边 → `incomplete=true`；DELETE 后消失。

### 阶段 3 · 与现有流程集成（§4.8，**边清理，必做**）
- [ ] `delete_job`(:7965) 删 `experiment_edges`(parent/child) + `experiment_node_layout`(job)。
- [ ] `delete_project`(:6103) 删 `experiment_edges` + `experiment_node_layout`(by project)。
- [ ] `POST /api/jobs/bulk/delete`(:6578) 同上按 id 集合清理。
- [ ] job 改项目两处：`PATCH /api/jobs/{jid}`(:7928) 与 `bulk_move_jobs`(:6555)，在 `project_id` 实际变化时删该 job 的边/坐标。
- [ ] **关卡**：建边后删其中一个 job / 删项目 / 把 job 移到别的项目 → `SELECT COUNT(*) FROM experiment_edges` 不残留脏行。

### 阶段 4 · 前端 MVP（§6.1–6.5、6.7、6.8）
- [ ] 路由 `/project/:pid/tree` + `beforeEach` 顶部放行分支（§6.1，**别用 `:id`**）。
- [ ] `ExperimentTree` 组件：`loadGraph()` / 候选 job / 分层布局（§6.4）。
- [ ] 平移 + 缩放（§6.4）。
- [ ] 节点悬停浮层、单击选中高亮血缘、双击进详情（§6.5）。
- [ ] 边详情面板：描述 + 变更项编辑 + delta 芯片 + incomplete 重算 + 对比详情链接（§6.5）。
- [ ] 加边弹窗（下拉列全部 single job）+ 删边（§6.5）。
- [ ] `unconnected` 池 / 空状态（§6.3）。
- [ ] 项目右键菜单入口（§6.7）+ `exp-` 样式（§6.8）。
- [ ] **关卡（§7 前端）**：菜单进入能渲染图；悬停/点击/双击行为正确；加/删边实时更新；分支与合并布局不重叠；深浅主题无 console error。

### 阶段 5 · MVP+ 拖拽（§2 坐标表、§4.7、§4.8 坐标清理、§6.6）
- [ ] `PUT /api/projects/{pid}/experiments/layout` 批量 upsert + `DELETE .../layout`（§4.7）。
- [ ] §4.8 中 `experiment_node_layout` 的清理已随阶段 3 落地（复核）。
- [ ] 前端：拖节点改坐标、`mouseup` 提交 `pinned=1`；区分拖节点 vs 平移；「自动整理」复位（§6.6）。
- [ ] **关卡**：拖动后刷新位置保留；新节点仍自动布局；「自动整理」清回自动排布。

### 阶段 6 · 收尾
- [ ] `git status` 检查仅改动 `web/db.py`、`web/server.py`、`web/static/{index.html,app.js,style.css}`、本 doc。
- [ ] README/CHANGELOG 视情况加一句功能说明（可选）。
- [ ] （Phase 3，非本次）§4.6 一键生成对比 + AI 路径总结。
