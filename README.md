# Torch Profiler Analyzer

Torch Profiler Analyzer 是一个面向 PyTorch Profiler Chrome Trace 的本地/内网性能分析工具。它可以解析 `.json`、`.json.gz`、`.gz`、`.json.zip`、`.zip`、`.tar.gz` 和 `.tgz` trace 文件，统计 GPU kernel、Triton kernel、ATen Ops、CNCL/NCCL 通信算子，并提供单 trace 分析、双 trace 对比、历史管理、AI 分析和 Web 可视化界面。

当前版本：`0.2.88`

## 主要功能

- **上传与对比**：单文件、批量上传、两个 trace 快速对比，以及基于历史任务的 A/B / 批量基线对比。
- **结果阅读**：默认进入性能总览，提供摘要卡片、Top 回退/改善、图表下钻、控制台全屏阅读和全量 CSV 表格能力。
- **定位细节**：Kernel 类型、所有 Kernel、Triton、ATen Ops、CNCL Ops、Triton Step 等页签支持搜索、筛选、排序、列显隐、分页和下载。
- **Step 重分析**：完成任务后可指定 step 派生新分析；对比任务支持 A/B 分别指定不同 step。
- **AI 分析**：Claude Code + 自定义 skill 生成 Markdown 报告，支持补充 Prompt、环境诊断、进度/耗时、历史版本、下载和完成通知。
- **团队协作**：LDAP 用户隔离、个人/共享项目、管理员统计，以及独立的 **灵感社区**（Issue 风格帖子、回复、图片、@ 候选、邮件通知）。
- **运维能力**：JSON 日志、审计日志、健康检查、Prometheus 指标、备份脚本和存储管理。

## 目录结构

```text
.
├── analyze_trace.py
├── trace_analyzer/
│   ├── __init__.py
│   └── core.py
├── web/
│   ├── auth.py
│   ├── backup.py
│   ├── db.py
│   ├── server.py
│   └── static/
│       ├── app.js
│       ├── favicon.svg
│       ├── index.html
│       └── style.css
├── tests/
├── scripts/init_deploy_dirs.sh
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## 快速开始

推荐使用 `uv`：

```bash
uv sync --extra web
uv run --extra web python web/server.py
```

浏览器访问：

```text
http://127.0.0.1:8181
```

指定监听地址和端口：

```bash
uv run --extra web python web/server.py --host 0.0.0.0 --port 8181
```

启用开发和测试依赖：

```bash
uv sync --extra web --extra dev
uv run --extra web --extra dev pytest -q
```

也可以使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[web]"
python web/server.py
```

## 推荐使用路径

1. 首页上传单 trace，或切到快速对比上传 A/B。
2. 先看 `性能总览`：总耗时、Top 回退/改善和占比图通常能定位第一批问题。
3. 需要证据时下钻到 `所有 Kernel`、`Triton`、`ATen Ops` 或 `CNCL Ops`，再下载当前页 CSV。
4. 需要自然语言总结时打开 `AI 分析`，可补充 Prompt；报告会保留多个版本。
5. 工具问题、优化建议或经验沉淀放到 `灵感社区`，被 @ 的同事会收到邮件。

## Web 使用

### 提交分析

Web 首页有两种上传模式：

- `单文件/批量`：拖拽或选择一个或多个 trace 文件，批量提交后每个文件生成一个任务。
- `快速对比`：同时上传 A/B 两个 trace，直接生成对比任务。
- 大 trace 分析期间，任务页会持续显示当前阶段和已用时间；如果正在解析 10GB+ trace，等待几十秒到数分钟是正常现象。

支持的输入格式：

- `.json`
- `.json.gz`
- `.gz`
- `.json.zip`
- `.zip`
- `.tar.gz`
- `.tgz`

上传时可以选择项目和填写别名。压缩包中会自动提取可用 JSON trace；服务端内部会统一保留压缩副本。普通体量 trace 会走快速 JSON 解析，超大 `.json.gz` 会保持压缩形态并单次流式读取 `traceEvents`，避免 10GB+ trace 解压落盘或一次性读入内存。下载原始 trace 时默认提供 `.json.gz`，便于保存大文件并保持工具兼容性。上传阶段不会裁剪 trace；如果只想分析某些 step，可以在任务完成后通过 `指定 Step 重分析` 创建新的派生任务。

### 结果页

任务完成后默认打开 `性能总览`：

- `性能总览`：摘要卡片、TopN 柱状图、占比图、对比回退/优化列表，默认排除通信类 kernel/op，支持点击下钻到相关表格。
- `控制台`：展示分析脚本输出，支持搜索、section 跳转、折叠生成文件日志和 Delta 着色。
- `Kernel 类型` / `类型对比`：按 family 聚合，点击类型行可跳到相关 Kernel 表格。
- `所有 Kernel`、`Triton`、`ATen Ops`、`CNCL Ops`：表格化查看明细；`Triton 对比` 会优先用 Triton code 指纹、code signature、多 step 指纹交集和规整化名称匹配 A/B kernel，减少末尾数字后缀不同导致的错位。
- `Triton Step N`：当保存了 per-step Triton CSV 时显示。
- `AI 分析`：服务端启用 Claude Code 后显示。

表格能力：

- 全局搜索、列筛选、排序。
- 列宽拖拽和列显隐。
- 每页数量可选，也支持快捷显示全部。
- 下载当前页 CSV。
- 下钻打开的 Kernel 表默认隐藏冗余列，方便聚焦相关 kernel。

其他结果页能力：

- 右上角 `全屏` 可让结果区域覆盖页面，适合看长表格、控制台和 AI 报告。
- 每个任务会记住上次停留的页签，以及表格搜索、排序、列宽、列筛选和列显隐。
- Perfetto 按钮可把 trace 打开到 Perfetto UI。
- AI 分析展示阶段进度和耗时；分析产物默认折叠在报告末尾，不抢占报告阅读空间。

### 历史、项目和对比

- 左侧侧栏包含 `历史` 和 `对比` 两个页签。
- 历史按项目分组展示，默认折叠时按项目条目分页。
- 侧栏支持项目过滤和任务/文件/项目搜索。
- 展开项目后按需加载任务，避免大量历史一次性渲染。
- 多选模式支持批量移动任务、删除任务、删除原始文件。
- 删除任务会清理该任务对应的 trace、压缩 trace、结果 CSV、Triton 代码、AI 分析产物等文件。
- 项目删除后进入回收站，可在保留期内恢复；也支持永久删除。
- `对比` 页签可选择两个已完成单文件任务创建对比，也可使用 `批量基线` 一次创建多个对比任务。
- 对比结果可交换 A/B 重新对比。
- 任务详情右上角 `...` 菜单支持 `指定 Step 重分析`。单 trace 填写 A step；对比任务可分别填写 A/B step，例如 A 选择 `0`、B 选择 `2-3`。留空的一侧表示使用全部 step，提交后会生成一个新任务。

### 灵感社区

右上角 `灵感社区` 会进入独立社区页，用于收集内部用户反馈和沉淀使用经验：

- 用户可以发布帖子，支持文字和最多 4 张图片。
- 进入帖子后可在帖子内回复交流；帖子和回复的发布者可以后续编辑正文。
- 帖子列表采用类似 GitHub Issue 的讨论列表风格，默认按最新更新时间排序，也可切换为发布时间或热度排序。
- 回复支持点赞、踩和常用表情反应，每个用户对同一条回复的同一表情只计一次。
- 新增帖子或回复会邮件通知管理员；正文支持 Emoji 和更多常用表情快捷插入，输入 `@英文名` 时会弹出候选，选择后自动补全，并额外通知 `英文名@cambricon.com`。
- 邮件正文包含 `打开留言` 链接，可直达对应帖子或回复；这需要配置 `TRACE_PUBLIC_BASE_URL`。
- 管理员可以删除帖子和回复，也可以在留言板中点击 `邮件诊断` 检查 SMTP/sendmail、DNS、端口连通性和收件人配置。

## Claude Code AI 分析

AI 分析默认关闭。开启后，已完成任务会出现 `AI 分析` 页签。点击开始/重新分析时会弹窗填写可选补充 Prompt；确认后流程如下：

1. 服务端先运行 AI 环境诊断。
2. 诊断检查 Claude 命令、skills 目录、单 trace skill、对比 skill、skills 挂载、基础 Claude 调用和工具权限探针。
3. 如果诊断失败，页面展示 Markdown 诊断报告和具体 stdout/stderr。
4. 如果诊断通过，调用 Claude Code 和对应 skill 生成 Markdown 分析报告。
5. 页面显示阶段进度、已耗时/总耗时，渲染 Markdown 报告，并支持复制、下载报告，以及点击报告中的产物文件名下载对应日志/DB/中间报告。
6. 如果页面不在前台，分析完成或失败时会触发浏览器通知；如果通知权限不可用，会退回到页面 toast 和标题提示。
7. 如果配置了邮件通道，分析完成或失败后会邮件通知触发人和任务所属人，邮件中包含 `AI 分析` 页直达链接和 Markdown 报告下载链接。

仓库默认使用：

- 单 trace skill：`.claude/skills/e2e-profiling-analyzer`
- 对比 skill：`.claude/skills/e2e-profiling-comparator`

常用环境变量：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TRACE_ENABLE_CLAUDE_ANALYSIS` | off | 设置为 `1` 后开启 AI 分析 |
| `TRACE_AI_ANALYSIS_CONCURRENCY` | `1` | 并发 Claude Code AI 分析任务数，建议从 1 开始按机器资源调大 |
| `TRACE_CLAUDE_COMMAND` | `claude` | Claude Code 命令 |
| `TRACE_CLAUDE_EXTRA_ARGS` | `--dangerously-skip-permissions` | 追加给 Claude Code 的参数 |
| `TRACE_CLAUDE_COMMAND_TEMPLATE` | 空 | 完整命令模板，可使用 `{prompt}`、`{trace_a}`、`{trace_b}`、`{skill}`、`{skills_dir}`、`{results_dir}`、`{analysis_dir}`、`{report_path}` |
| `TRACE_CLAUDE_CUSTOM_HEADERS` | `x-project: torch_mlu` | 注入到 Claude Code 子进程的 `ANTHROPIC_CUSTOM_HEADERS`；用于网关侧项目标识 |
| `TRACE_CLAUDE_SKILLS_DIR` | `.claude/skills` | Claude skills 目录 |
| `TRACE_CLAUDE_SINGLE_SKILL` | `e2e-profiling-analyzer` | 单 trace skill 名称 |
| `TRACE_CLAUDE_COMPARE_SKILL` | `e2e-profiling-comparator` | 对比 skill 名称 |
| `TRACE_CLAUDE_MODEL` | `Claude Code default` | AI 报告版本元信息中记录的后端模型名；若网关通过 `ANTHROPIC_MODEL` 选择模型，也可不单独设置 |
| `TRACE_CLAUDE_TIMEOUT_SECONDS` | `1800` | AI 分析超时时间 |
| `TRACE_CLAUDE_DIAGNOSTIC_TIMEOUT_SECONDS` | `60` | AI 环境诊断超时时间 |

示例：

```bash
TRACE_ENABLE_CLAUDE_ANALYSIS=1 \
TRACE_CLAUDE_COMMAND=/usr/local/node20/bin/claude \
TRACE_CLAUDE_EXTRA_ARGS=--dangerously-skip-permissions \
TRACE_CLAUDE_CUSTOM_HEADERS='x-project: torch_mlu' \
uv run --extra web python web/server.py --host 0.0.0.0 --port 8181
```

使用命令模板：

```bash
TRACE_ENABLE_CLAUDE_ANALYSIS=1 \
TRACE_CLAUDE_COMMAND_TEMPLATE='/usr/local/node20/bin/claude --dangerously-skip-permissions -p {prompt}' \
uv run --extra web python web/server.py
```

AI 分析目录位于任务结果目录下的 `ai_analysis/`。其中 `ai_analysis.md` 是当前最新报告；每次分析结束都会在 `ai_analysis/versions/` 下保留一个历史版本，记录生成时间、触发人、后端模型名、skill、补充 Prompt、耗时和状态。页面默认展示最新版本，也可以切换旧版本并下载对应 Markdown；其他小文本产物会在页面末尾折叠展示。

## 认证、用户隔离和管理员

默认 `AUTH_MODE=none`，适合单用户本地使用。内网部署可启用 LDAP：

```bash
AUTH_MODE=ldap
SESSION_SECRET="replace-with-a-long-random-secret"
LDAP_URL="ldaps://ldap.example.com:636"
LDAP_BASE_DN="DC=example,DC=com"
LDAP_BIND_DN="CN=svc_analyze_trace,OU=Service Accounts,DC=example,DC=com"
LDAP_BIND_PASSWORD="replace-with-service-account-password"
LDAP_USER_FILTER="(sAMAccountName={username})"
LDAP_REQUIRE_GROUP_DN="CN=analyze_trace_users,OU=Groups,DC=example,DC=com"
```

也可以使用 UPN 直连绑定：

```bash
AUTH_MODE=ldap
SESSION_SECRET="replace-with-a-long-random-secret"
LDAP_URL="ldaps://ldap.example.com:636"
LDAP_USER_DN_TEMPLATE="{username}@example.com"
```

可选项：

| 环境变量 | 说明 |
| --- | --- |
| `LDAP_DISPLAY_NAME_ATTR` | 显示名属性，默认 `displayName` |
| `LDAP_MAIL_ATTR` | 邮箱属性，默认 `mail` |
| `LDAP_TLS_CA_FILE` | LDAPS CA 证书路径 |
| `SESSION_COOKIE_SECURE=1` | HTTPS 部署建议开启 |
| `TRACE_ADMIN_USERS` | 管理员账号，支持用户名、邮箱或显示名，多个用逗号分隔 |
| `LOGIN_CAPTCHA_THRESHOLD` | 连续登录失败后要求验证码，默认 5 |
| `LOGIN_CAPTCHA_TTL_SECONDS` | 验证码有效期，默认 300 秒 |

启用认证后：

- 个人项目和任务按登录用户隔离。
- 用户可以创建共享项目，或把自己的项目转为共享项目。
- 共享项目对所有登录用户可读，可用于对比。
- 任务重命名、移动、删除和文件删除仍限制为任务创建者。
- 管理员可进行全局管理操作，例如删除留言板内容，并在右上角 `...` 菜单打开 `使用统计` 查看今日日活、近 7 日活跃、逐日请求量、任务量、AI 分析次数和今日活跃用户。

## 运维

### 目录和权限

建议生产环境使用独立数据目录：

```bash
sudo mkdir -p /data/analyze_trace/storage /data/analyze_trace/logs /data/analyze_trace/backups
sudo chown -R cambricon:cambricon /data/analyze_trace
```

关键环境变量：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TRACE_STORAGE_DIR` | `web/storage` | 数据库、上传文件和结果目录 |
| `TRACE_BACKUP_DIR` | `web/backups` | 备份目录 |
| `TRACE_LOG_FILE` | 空 | JSONL 日志文件 |
| `TRACE_LOG_LEVEL` | `INFO` | 日志级别 |
| `TRACE_LOG_TIMEZONE` | `Asia/Shanghai` | JSON 日志时间戳时区；可设为 `UTC`、`local` 或 IANA 时区名 |
| `TRACE_DB_TIMEOUT_SECONDS` | `30` | SQLite 写锁等待超时；服务会启用 WAL 与 busy timeout 以降低并发写入冲突 |
| `TRACE_UPLOAD_CONCURRENCY` | `3` | 同时处理上传/解压的请求数 |
| `TRACE_MAX_UPLOAD_BYTES` | `0` | 单个上传文件大小限制；`0` 表示不限制 |
| `TRACE_MAX_TRACE_JSON_BYTES` | `0` | 分析前允许的解压后 trace JSON 大小上限；`0` 表示不限制。该保护会直接检查 plain JSON/zip/tar 中可获得的 JSON 大小；普通 `.json.gz` 默认不额外预扫描，避免 10GB+ trace 被重复解压 |
| `TRACE_STRICT_GZIP_SIZE_CHECK` | `0` | 设置为 `1` 后，普通 `.json.gz` 也会在分析前流式扫描解压后大小。该模式更严格，但会让 gzip trace 至少多解压一遍，超大文件会明显变慢 |
| `TRACE_FAST_TRACE_JSON_BYTES` | `268435456` | 小/中等 trace 走 `orjson` 快速整文件解析的大小阈值；超出阈值会自动回退流式解析，`0` 表示全部使用流式解析。调高会显著提升中大 trace 速度，但需要按解压后 JSON 大小预留数倍内存 |
| `TRACE_MIN_STORAGE_FREE_BYTES` | `0` | 上传前要求保留的磁盘可用空间；`0` 表示不检查 |
| `TRACE_ANALYSIS_CONCURRENCY` | `1` | 并发分析任务数 |
| `TRACE_AI_ANALYSIS_CONCURRENCY` | `1` | 并发 Claude Code AI 分析任务数 |
| `TRACE_NO_DOWNLOAD` | 空 | 设置后禁止下载原始 trace |
| `TRACE_ENABLE_CODE_EXEC` | off | 设置为 `1` 后允许运行 Triton 代码和清除 cache |

邮件通知（留言板和 AI 分析）：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TRACE_FEEDBACK_ADMIN_EMAILS` | `zhouyusong@cambricon.com` | 新增帖子或回复时默认通知的管理员邮箱，多个用逗号分隔 |
| `TRACE_FEEDBACK_MENTION_DOMAIN` | `cambricon.com` | 留言中 `@英文名`、任务 owner 用户名映射到邮箱时使用的域名 |
| `TRACE_DISABLE_FEEDBACK_EMAIL` | off | 设置为 `1` 后关闭留言板和 AI 分析邮件通知 |
| `TRACE_PUBLIC_BASE_URL` | 空 | 对外访问地址；用于邮件里的应用链接、留言深链、AI 分析结果链接 |
| `TRACE_SMTP_HOST` / `SMTP_HOST` | 空 | SMTP 服务器；为空且未显式配置 sendmail 时不发送邮件，页面会提示缺少投递通道 |
| `TRACE_SMTP_PORT` / `SMTP_PORT` | `25` | SMTP 端口 |
| `TRACE_SMTP_USERNAME` / `SMTP_USERNAME` | 空 | SMTP 用户名 |
| `TRACE_SMTP_PASSWORD` / `SMTP_PASSWORD` | 空 | SMTP 密码 |
| `TRACE_SMTP_FROM` | `trace-analyzer@cambricon.com` | 发件人地址；建议申请同名公共邮箱或由 SMTP 中继允许该地址发信 |
| `TRACE_SMTP_SSL` / `SMTP_SSL` | off | 使用 SMTP SSL |
| `TRACE_SMTP_STARTTLS` / `SMTP_STARTTLS` | off | 使用 STARTTLS |
| `TRACE_SMTP_TIMEOUT_SECONDS` | `10` | SMTP 连接超时 |
| `TRACE_SENDMAIL_COMMAND` | 空 | SMTP 为空时的 sendmail 命令路径，例如 `/usr/sbin/sendmail`；需确认该命令能外发到公司邮箱 |
| `TRACE_ENABLE_SENDMAIL_AUTODETECT` | off | 设置为 `1` 后才自动探测系统 sendmail；默认关闭，避免本地 sendmail 只排队但无法投递时误报成功 |

### systemd 示例

`/etc/analyze_trace.env`：

```bash
TRACE_STORAGE_DIR=/data/analyze_trace/storage
TRACE_BACKUP_DIR=/data/analyze_trace/backups
TRACE_LOG_FILE=/data/analyze_trace/logs/app.jsonl
TRACE_ENABLE_CLAUDE_ANALYSIS=1
TRACE_CLAUDE_COMMAND=/usr/local/node20/bin/claude
TRACE_CLAUDE_EXTRA_ARGS=--dangerously-skip-permissions
TRACE_CLAUDE_CUSTOM_HEADERS="x-project: torch_mlu"
TRACE_PUBLIC_BASE_URL=http://tpa.cambricon.com:1818
# 向 IT 确认真正可解析、可连通的 SMTP 主机；不要直接使用占位示例
TRACE_SMTP_HOST=<it-provided-smtp-host>
TRACE_SMTP_PORT=25
TRACE_SMTP_FROM=trace-analyzer@cambricon.com
TRACE_FEEDBACK_ADMIN_EMAILS=zhouyusong@cambricon.com
```

`/etc/systemd/system/analyze-trace.service`：

```ini
[Unit]
Description=Torch Profiler Analyzer
After=network-online.target

[Service]
User=cambricon
Group=cambricon
WorkingDirectory=/opt/analyze_trace
EnvironmentFile=/etc/analyze_trace.env
ExecStart=/opt/analyze_trace/.venv/bin/python web/server.py --host 0.0.0.0 --port 1818
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now analyze-trace
sudo systemctl status analyze-trace --no-pager
```

### 日志和审计

- 服务输出 JSON 请求日志，字段包含 `request_id`、用户、IP、方法、路径、状态码和耗时；时间戳默认使用 `Asia/Shanghai`，可通过 `TRACE_LOG_TIMEZONE` 调整。
- 设置 `TRACE_LOG_FILE` 后会同时写入 JSONL 文件。
- 留言板发布帖子或回复会额外写入 `feedback_created` 业务日志，包含留言类型、作者、IP、图片数、@ 收件人和邮件通知状态；邮件发送成功/失败会写入 `feedback_email_sent` / `feedback_email_failed`。
- AI 分析结束后会写入 `ai_analysis_email_sent` / `ai_analysis_email_failed` / `ai_analysis_email_not_sent`，用于确认触发人和任务所属人的完成通知是否成功发送。
- 关键操作会写入 SQLite 的 `audit_logs` 表。
- 可通过 `GET /api/audit-logs?limit=100` 查看审计记录。
- 访问使用量会按天聚合到 SQLite 的 `usage_daily` 表；管理员接口 `GET /api/admin/usage?days=14` 用于查看日活、请求量和业务动作趋势。该统计从启用本版本后开始准确累计，历史数据不会从旧请求日志自动补齐。

### 备份

备份脚本会使用 SQLite backup API 生成一致性数据库快照，并打包 storage：

```bash
uv run python web/backup.py \
  --storage-dir /data/analyze_trace/storage \
  --backup-dir /data/analyze_trace/backups \
  --retention-days 14
```

建议使用 cron、systemd timer 或公司备份平台定时执行，并把备份目录同步到 NAS 或对象存储。

### 监控接口

| 接口 | 用途 |
| --- | --- |
| `/healthz` | 存活检查 |
| `/readyz` | DB、storage、backup、log file 可用性检查 |
| `/metrics` | Prometheus 文本指标 |

`/metrics` 包含请求量、请求耗时、任务状态数量、分析队列长度、磁盘容量和最近一次备份信息。

### 常见排障

服务状态和日志：

```bash
sudo systemctl status analyze-trace --no-pager
sudo journalctl -u analyze-trace -n 100 --no-pager
curl -fsS http://127.0.0.1:1818/healthz
curl -fsS http://127.0.0.1:1818/readyz
```

如果服务反复重启，优先检查 `TRACE_STORAGE_DIR`、`TRACE_LOG_FILE`、`TRACE_BACKUP_DIR` 对服务运行用户是否可写。

邮件通知排障：

```bash
getent hosts <smtp-host>
nc -vz <smtp-host> <smtp-port>
```

- `SMTP 主机无法解析`：`TRACE_SMTP_HOST` 不是可解析的真实 SMTP 地址，或服务器 DNS 不通。
- `SMTP 拒收发件人`：`TRACE_SMTP_FROM` 没有被 SMTP 中继允许；建议申请 `trace-analyzer@cambricon.com` 公共邮箱或让 IT 将该发件人加入白名单。
- 发布成功或 AI 分析完成后收不到邮件：在留言板点击管理员可见的 `邮件诊断`，并检查日志中的 `feedback_email_sent` / `feedback_email_failed`、`ai_analysis_email_sent` / `ai_analysis_email_failed`。

AI 分析排障：

- 在服务运行用户下执行 `command -v claude && claude --version`，确认 `TRACE_CLAUDE_COMMAND` 指向正确。
- AI 页签中的 `环境诊断` 会检查 Claude 命令、skills 目录、基础调用和工具权限；未通过时先按页面诊断明细修复。
- 如果通过终端能调用 Claude，但服务中失败，重点检查 systemd 的 `EnvironmentFile`、服务用户的 `HOME`、`.claude` 登录状态和目录写权限。

## Docker

```bash
docker-compose up -d
```

或者手动构建：

```bash
cd web
docker build -t trace-analyzer .
docker run -d -p 8181:8181 --name trace-analyzer -v trace_data:/app/storage trace-analyzer
```

域名直连部署时，应用需要监听 `0.0.0.0:1818`，并让 IT 或安全组放通 `1818/tcp`。对外访问地址为 `http://tpa.cambricon.com:1818/`，因此 `TRACE_PUBLIC_BASE_URL` 也要包含 `:1818`，否则邮件和分享链接会生成不带端口的地址。

## CLI 使用

安装：

```bash
pip install -e .
```

单文件分析：

```bash
analyze-trace trace.json.gz -o ./output
```

双文件对比：

```bash
analyze-trace baseline.json.gz optimized.json.gz -o ./output
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `trace_files` | 无 | 1 或 2 个 PyTorch Profiler trace 文件 |
| `-o, --output-dir` | `.` | 输出目录 |
| `-s, --save-triton-csv` | off | 输出逐 step 的 Triton kernel 详情 CSV |
| `-c, --save-triton-code` | off | 保存每个 Triton kernel 的生成源码 |
| `--steps` | 空 | 只分析指定 step，例如 `0`、`0,2,5`、`1-4`；双文件时作为 A/B 默认值 |
| `--steps-a` | 空 | 双文件对比时 A trace 的 step 选择 |
| `--steps-b` | 空 | 双文件对比时 B trace 的 step 选择 |

## 输出文件

单文件分析：

| 文件 | 内容 |
| --- | --- |
| `all_kernels_avg.csv` | 所有 GPU kernel 按名称聚合的平均耗时和调用次数 |
| `triton_kernels_avg.csv` | Triton kernel 的平均耗时、IO 量和平均 IO 效率 |
| `aten_ops_avg.csv` | ATen Ops 的平均耗时和调用次数 |
| `kernel_types_avg.csv` | Kernel family 聚合结果 |
| `cncl_ops_avg.csv` | CNCL/NCCL 通信算子聚合结果 |
| `step_N_triton_kernels.csv` | 开启 `-s` 后输出的逐 step Triton 明细 |
| `step_N_triton_codes/` | 开启 `-c` 后保存的 Triton 代码 |

双文件对比会额外输出：

| 文件 | 内容 |
| --- | --- |
| `all_kernels_cmp.csv` | 两个 trace 的 kernel 耗时和调用次数 delta |
| `triton_kernels_cmp.csv` | Triton kernel 对比；包含 `match_method`、`kernel_name_A`、`kernel_name_B`，优先按 exact name、code hash、多 step code hash 交集、code signature、规整化名称 + tiling、规整化名称匹配 |
| `aten_ops_cmp.csv` | ATen Ops 对比 |
| `kernel_types_cmp.csv` | Kernel family 对比 |
| `cncl_ops_cmp.csv` | CNCL/NCCL 通信算子对比 |

## 解析逻辑

分析流程会先识别 step 区间，再把 kernel、ATen 和通信事件归属到对应 step：

1. 优先识别 `ProfilerStep#N` 和 `step_N`。
2. 如果没有标准 step 标记，会 fallback 到 `run_step` 或整体可分析事件范围。
3. 每个 step 内先聚合，再对所有 step 求平均，降低单步抖动影响。

Kernel family 规则：

- `triton`：名称以 `triton_` 开头，并进一步细分为 `triton_mm`、`triton_reduce`、`triton_pointwise` 等。
- `collective`：TCDP 前缀或包含 `nccl`、`cncl`、`allreduce`、`allgather` 等通信关键词，单独统计，不计入 compute 分析。
- 语义聚类：匹配 `gemm`、`conv`、`embedding`、`pool`、`norm`、`attention` 等常见类型。
- fallback：无法匹配规则的 kernel 按名称前缀归类，兜底归入 `other`。

## 安全说明

- `TRACE_ENABLE_CODE_EXEC=1` 会允许执行 trace 中保存的 Triton 代码，只建议在可信环境使用。
- Claude Code AI 分析会让服务端进程调用本机 `claude` 命令，并需要对任务结果目录有读写权限。
- 对内开放时建议启用 LDAP、HTTPS、日志、备份和监控。

## 开源协议

本项目使用 BSD-3-Clause License，风格与 PyTorch 社区版采用的宽松开源协议一致。详见 [LICENSE](LICENSE)。
