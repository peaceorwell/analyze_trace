# 任务 URL 数字短 ID（8 位）实现规格

> 需求：新上传/新建的任务，其 URL 用 **8 位数字 ID**（如 `#/job/10000001`）；同时**旧的 UUID 链接继续可用**。
> 核心原则：**不替换 UUID**。UUID 仍是 jobs 主键 + 外键目标 + **文件系统目录名**（`job_dir = STORAGE_DIR/{uuid}`，`server.py:1095`），改它伤筋动骨。做法是给 jobs 加一个数字"门牌号" `seq` 作为对外句柄，内部一切照旧用 uuid。
> 关键洞察（决定改动面很小）：前端首次加载后把当前任务规范化成 uuid，之后所有子资源调用仍用 uuid；**后端真正必须认 seq 的只有入口 `GET /api/jobs/{jid}`**。其余靠"在 `load_*` 助手里顺带解析"覆盖，旧 uuid 天然兼容。
> 涉及文件：`web/db.py`、`web/server.py`、`web/static/app.js`。

---

## A. 数据层（`web/db.py`）

### A1 列与索引
- [ ] jobs 加列：`seq INTEGER`（在 `init_db` 第一段 `CREATE TABLE jobs` 内加，或用 `add_column_if_missing(db,"jobs","seq","INTEGER")` 兼容旧库）。
- [ ] 唯一索引：`CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_seq ON jobs(seq);`

### A2 单调、不复用的计数器（保证旧链接永不被新任务顶替）
> 用 `MAX(seq)+1` 会在删除任务后复用号码 → 旧 URL 指向新任务。必须用持久计数器。
- [ ] 计数器表 + 起始值（从 `10000000` 起，下一个即 `10000001`，天然 8 位，撑到 9000 万个任务）：
  ```sql
  CREATE TABLE IF NOT EXISTS job_seq_counter (id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL);
  INSERT OR IGNORE INTO job_seq_counter(id, value) VALUES (1, 10000000);
  ```
- [ ] **AFTER INSERT 触发器**自动给新任务发号（零遗漏，不管经哪条 INSERT 路径；jobs 有 10 处 INSERT，用触发器避免逐个改）：
  ```sql
  CREATE TRIGGER IF NOT EXISTS trg_jobs_assign_seq
  AFTER INSERT ON jobs WHEN NEW.seq IS NULL
  BEGIN
    UPDATE job_seq_counter SET value = value + 1 WHERE id = 1;
    UPDATE jobs SET seq = (SELECT value FROM job_seq_counter WHERE id = 1) WHERE id = NEW.id;
  END;
  ```
  > 触发器内是 UPDATE 不是 INSERT，不会自触发递归；WAL 串行写保证并发安全。

### A3 历史任务回填（一次性迁移）
- [ ] 按 `created_at, id` 顺序给所有 `seq IS NULL` 的旧任务依次赋 `10000001, 10000002, …`；赋完把 `job_seq_counter.value` 设为最后一个值，使后续新任务接着递增、与旧号不冲突。
- [ ] 迁移幂等：再次启动时 `seq IS NULL` 已无、`INSERT OR IGNORE` 计数器不重置。

---

## B. 后端解析（`web/server.py`）

### B1 句柄解析助手
- [ ] 新增：
  ```python
  async def resolve_job_pk(db, handle: str) -> Optional[str]:
      """把 URL 句柄解析成 jobs.id(uuid)。纯数字 → 按 seq 查；否则原样当 uuid。"""
      if handle is None:
          return None
      if handle.isdigit():
          row = await (await db.execute("SELECT id FROM jobs WHERE seq=?", (int(handle),))).fetchone()
          return row[0] if row else None
      return handle  # 已是 uuid
  ```
- [ ] 纯数字但查不到 → 返回 None（让调用方 404）。

### B2 接入点（覆盖面，按性价比）
- [ ] **必做**：`load_accessible_job` / `load_owned_job` 在用 `job_id` 前先 `job_id = await resolve_job_pk(db, job_id)`（覆盖 39 处调用，含入口 `GET /api/jobs/{jid}`）。
- [ ] **建议**：审计直接吃 `{jid}` 路径参数、又**不**经过 `load_*` 的少数端点（用 `result_dir(jid)`/`job_dir(jid)` 或裸 `WHERE id=?` 的，约 16 处裸查里属于此类的部分），在函数开头 `jid = await resolve_job_pk(db, jid) or 404`。漏改不会让旧链接断（uuid 仍通），只是该子资源暂不认 seq；但建议补全以支持"分享带 tab 的短链/子资源深链"。
- [ ] `seq` 随 `dict(row)` / `SELECT *` **自动出现在所有任务 JSON 里**（45 处序列化无需改），前端据此生成短链。

---

## C. 前端（`web/static/app.js`）

### C1 路由句柄 vs 规范 id（关键，别搞混）
现状：`loadJobRoute` 把 `selectedJobId.value = to.params.id`（即 URL 句柄）。若句柄是 seq，会污染各处 `ids.includes(selectedJobId.value)`（列表里都是 uuid）等比较。
- [ ] **`selectedJobId` 始终保存规范 uuid**：加载成功后用返回 job 的 `id`（uuid）赋值，**不要**用路由句柄。
- [ ] 新增 `selectedJobHandle` 记录当前 URL 句柄，用于"同任务仅切 tab"的判断：
  - `loadJobRoute(to)`：`handle = to.params.id`；若 `handle === selectedJobHandle` → 走切 tab 分支；否则 `job = await loadJob(handle)`（GET `/api/jobs/{handle}` 后端已解析），成功后 `selectedJobId = job.id`（uuid）、`selectedJobHandle = handle`。
- [ ] `loadJob` 的子资源/轮询继续用 `selectedJobId`(uuid)，行为不变。

### C2 链接生成一律用短 ID
- [ ] 所有由 job 对象生成 `/job/...` 的地方改用 `job.seq`（回退 `job.id`，理论上回填后恒有 seq）：`navigateToJob`、各 `router.push({path:'/job/'+...})`（约 18 处）。
- [ ] 分享 URL（`app.js:5672` 用 `selectedJobId.value`）改用**当前 job 的 `seq`** 生成短链；AI 深链（`app.js:3788` 的 `#/job/${jobId}/ai`）同理优先 seq。
- [ ] 路由定义 `/job/:id`、`/job/:id/:tab` 不变（`:id` 既收 seq 也收 uuid）。

### C3 兼容旧链接
- [ ] 旧 `#/job/{uuid}` 直接可用：后端 `resolve_job_pk` 见非数字按 uuid 处理；前端句柄流程对 uuid 同样成立。无需做任何重定向。

---

## D. 边界与正确性
- [ ] **不复用号码**：删除任务后 `seq` 不回收（计数器只增），旧短链不会指向新任务。
- [ ] compare/batch 等所有任务都是 jobs 行 → 触发器一律发号。
- [ ] 纯数字但不存在的句柄 → 404（前端走现有"任务不存在"回退到首页）。
- [ ] 极端：人为传入超大数字 → 查不到即 404，无副作用。
- [ ] 8 位上限：到 9999_9999（~9000 万任务）后进位成 9 位，URL 仍可用，不报错（仅不再是严格 8 位）。

---

## E. 验收
- [ ] 迁移后启动无错：`sqlite3 web/storage/jobs.db "SELECT id,seq FROM jobs ORDER BY seq LIMIT 3;"` 旧任务已得 8 位 seq；`job_seq_counter.value` = 最大 seq。
- [ ] 新建/上传一个任务 → 其 `seq` = 上一个 +1（触发器生效，无需改各 INSERT）。
- [ ] `curl /api/jobs/{seq}` 与 `curl /api/jobs/{uuid}` 返回同一任务；返回 JSON 含 `seq`。
- [ ] 前端列表点击进入的 URL 是 `#/job/{8位数字}`；刷新该短链正常加载；分享按钮给出短链。
- [ ] **旧 `#/job/{uuid}` 仍正常打开**；其子页 `/job/{uuid}/{tab}` 正常。
- [ ] 选中任务时，删除/批量等依赖 `selectedJobId` 的操作仍正确（证明 `selectedJobId` 是 uuid 而非句柄）。
- [ ] `pytest tests/test_web_api.py` 全过；建议补：seq 自动分配、seq/uuid 双解析、删除不复用号 的用例。

## F. 范围与不动项
- **不改**：jobs 主键、文件系统目录布局、外键、`experiment_edges`/`source_job_*`/`deleted_jobs`/审计的引用——全部继续用 uuid。
- 改动集中在：`db.py`（列+索引+计数器+触发器+回填）、`server.py`（`resolve_job_pk` + 接入 `load_*` 与少数裸查端点）、`app.js`（句柄/规范 id 分离 + 链接生成用 seq）。
- 兼容旧链接是天然的（uuid 全程合法），前端可**渐进迁移**链接生成点，漏改某处只是仍输出长链、不报错。

## G. 实施顺序
1. A 迁移（列/计数器/触发器/回填）→ `sqlite3` 验证。
2. B `resolve_job_pk` + 接入 `load_*`，`curl` 验证 seq/uuid 双通。
3. C 前端句柄分离 + 链接改 seq，浏览器验证新短链 + 旧 uuid 链接。
4. 补测试、跑 `pytest`。
