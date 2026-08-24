# tpa_mcp — TPA 的 MCP 服务器

用 [FastMCP](https://gofastmcp.com) 把 [TPA](http://tpa.cambricon.com)（trace performance analyzer，PyTorch profiler 分析 Web 服务）的 REST API 封装成 Model Context Protocol 工具，让 Claude 可以直接查询/分析任务、读结果与 AI 报告；本地 stdio 模式还可上传 trace 触发新分析。

## 已并入 analyze server

本目录已合并进 analyze server（`web/server.py` 的 FastAPI app），在**同一进程、同一端口**暴露 MCP 端点：

```
MCP endpoint = http://<host>:8181/mcp/
```

- 现有 REST `/api/*` 端点与 MCP `/mcp/` 共存于一个 uvicorn 进程，无需单独部署。
- 鉴权按 per-user（每个客户端带自己的 analyze Bearer access token），**无需** `TPA_API_KEY`。
- 标准 Web 依赖已包含 `fastmcp>=3.4.7,<4`；若自定义环境缺少依赖，analyze server 会记录告警并跳过 `/mcp`。
- 也可用 `web/tpa_mcp/http_app.py` 独立跑一个 MCP 服务（见下文「独立 HTTP 运行」）。

## 架构与交互（用户 → /mcp → 回环 /api）

```
┌───────────────────────── analyze server (http://<host>:8181) ────────────────────────┐
│                                                                                      │
│   ┌──────────────┐      ┌──────────────────────────┐      ┌───────────────────────┐  │
│   │ /api/* REST  │      │ /mcp (FastMCP 挂载)       │      │ 鉴权层 auth_middleware │  │
│   │ (analyze 自身)│      │  · AnalyzeTokenVerifier  │      │  解析 Bearer →         │  │
│   └──────▲───────┘      │  · 10 个 tpa_* 工具        │      │  查 api_tokens 表      │  │
│          │              └──────────▲───────────────┘      └───────────▲───────────┘  │
│          │  工具回调自身 /api       │                                 │              │
│          └─────────────────────────┘                                 │              │
└──────────────────────────────────────────────────────────────────────┘              │
                                     ▲
                Claude Code 带 Bearer token 连 /mcp/
```

一次 MCP 工具调用（以 `tpa_list_jobs` 为例）经过：

1. **Claude Code 发请求**到 `http://<host>:8181/mcp/`，头带 `Authorization: Bearer <用户 token>`。
2. **父中间件 `auth_middleware`** 先过一遍；因路径是 `/mcp`，它**豁免 readonly-POST 检查**（MCP 工具调用全是 POST，readonly 拦截改由内层 `/api` 负责）。
3. **FastMCP `AnalyzeTokenVerifier`** 独立校验 token（查 analyze 的 `api_tokens` 表，sha256 比对），识别出是该用户的哪个 token。
4. **工具函数 `_client()`**：从 ASGI 的监听 socket（或可信配置 `TPA_INTERNAL_BASE_URL`）生成内部 API 地址，不读取客户端可控的 `Host`；用 `get_access_token()` 取得该用户 token。
5. **TpaClient 回环调用 analyze 自己的 `/api/jobs`**，带用户 token → 父中间件完整鉴权 + 按用户隔离数据 → 返回结果。
6. 结果经 MCP 返回给 Claude。

> **为什么回环调 `/api` 而不是直连 DB**：MCP 工具不带用户 token 直接碰数据。复用 analyze 的 `/api`，数据隔离、readonly 权限天然一致——REST 和 MCP 共用同一套 token 与权限逻辑，不维护两套。

## 鉴权（per-user，每个用户自己的 token）

这个 MCP 是给**用户操作 analyze server** 的接口：每个 MCP 客户端用**自己**在 analyze server 上创建的 **access token**（Bearer）做鉴权，工具调用再以该用户身份打 analyze 自己的 `/api/*`，数据按用户隔离。token 在 analyze 网页 `...` → `用户设置` → `访问令牌` 创建，或用 `POST /api/tokens` 创建（与 REST `/api` 用的是同一套 token）。

连接时把 token 通过请求头传进去：

```bash
claude mcp add tpa --transport http --url http://127.0.0.1:8181/mcp/ \
  --header "Authorization: Bearer <自己的 access token>"
```

- **方式一（推荐，经 analyze server，单进程）**：如上，各个用户带各自的 token 连 `/mcp/`。analyze server 内嵌的 FastMCP 用 `AnalyzeTokenVerifier` 校验 token（查 `api_tokens` 表），并按该用户身份代理后续 `/api` 调用。
- 环境变量 `TPA_API_KEY` 只在本地 stdio 模式中作为调用 REST API 的凭据。

`TPA_BASE_URL`：可选，默认 `http://tpa.cambricon.com`。

`TPA_INTERNAL_BASE_URL`：可选，仅供 HTTP MCP 回环 `/api` 使用。普通 TCP 部署会自动使用实际监听地址；Unix socket、进程内 TLS 或独立端口部署需显式配置可信的内部 API 地址。

## 连接方式

### 1. 经 analyze server（推荐，单进程）
起好 analyze server，每个用户带各自的 token 连 `http://<host>:8181/mcp/`。例如用 `claude mcp add` 指定远程端点并传入自己的 access token：

```bash
# 地址按实际部署改；token 换成你自己的 access token（见上方「鉴权」）
claude mcp add tpa --transport http --url http://127.0.0.1:8181/mcp/ \
  --header "Authorization: Bearer <自己的 access token>"
```

### 2. 本地 stdio 运行（调试用，不经 analyze server 鉴权）
```bash
claude mcp add tpa \
  --env TPA_API_KEY=$TPA_API_KEY \
  -- python3 </path/to>/analyze_trace/web/tpa_mcp/server.py
```

### 3. 独立 HTTP 运行（备选，单独端口）
若需要单独端口，可复用 analyze server 的 token 数据库并显式指定内部 API 地址：
```bash
cd </path/to>/analyze_trace/web && \
  TPA_INTERNAL_BASE_URL=http://127.0.0.1:8181 \
  </path/to>/.venv/bin/python -m uvicorn tpa_mcp.http_app:app \
  --host 0.0.0.0 --port 8080
# endpoint = http://<host>:8080/mcp
```
独立 HTTP 端点仍要求客户端携带存储在同一 `api_tokens` 表中的 Bearer Token；stdio 模式才直接使用 `TPA_API_KEY`。

## 工具

| 工具 | 说明 |
|---|---|
| `tpa_list_jobs` | 列出任务（可按 project/q/statuses/limit/offset 过滤） |
| `tpa_get_job` | 查某任务详情（seq 或 UUID），含控制台摘要与结果文件清单 |
| `tpa_get_job_status` | 紧凑状态查询（专供上传后轮询 status 与 AI 进度） |
| `tpa_list_projects` | 列出项目（可用 `q` 过滤） |
| `tpa_get_job_result` | 读结果表（all_kernels / triton_kernels / aten_ops / kernel_types，compare 带 `_cmp`），支持筛选/排序/分页 |
| `tpa_get_ai_report` | 读自动生成的 AI 分析报告（Markdown） |
| `tpa_get_ai_analysis_status` | 查 AI 分析状态/进度 |
| `tpa_start_ai_analysis` | 触发/重跑某已完成任务的 AI 分析（可传自定义 prompt） |
| `tpa_get_job_report_md` | 读任务的 `report.md` 纯文本摘要 |
| `tpa_upload_trace` | 仅本地 stdio：流式上传 1~2 个 trace 并异步返回 seq/job_id |

任务可用其**数字 seq**（如 `10000679`）或 **UUID id** 引用，两种都能直接定位。

## 异步分析流程

本地 stdio 的 `tpa_upload_trace` 上传后立即返回（不阻塞等待）。分析在 TPA 后台异步运行，配合以下工具完成「提交 → 轮询 → 取结果」：

```
1. seq = tpa_upload_trace(file_a=..., file_b=...).seq     # 立即返回
2. tpa_get_job_status(job=seq) ... 直到 status == "done"  # 轮询
3. tpa_get_job(job=seq) / tpa_get_job_result(job=seq, filename="all_kernels")  # 取结果
4. （可选）tpa_start_ai_analysis(job=seq)  →  tpa_get_ai_report(job=seq)      # AI 报告
```

所有工具都不做长轮询/阻塞，等待逻辑交给调用方（LLM 可自行多次调用 `tpa_get_job_status`）。

远程 HTTP MCP 不接受客户端传入的文件路径，因为该路径会被解释为服务端路径，既无法读取客户端文件也存在越权风险。请改用同一个 token 调 REST 上传：

```bash
curl -X POST http://<host>:8181/api/jobs \
  -H "Authorization: Bearer <自己的 access token>" \
  -F "file_a=@/path/to/trace.json" \
  -F "save_triton_csv=true"
```

取得响应中的 `id` 或 `seq` 后，再通过 MCP 查询状态和结果。

## 快速验证

**方式 A（推荐，走完整鉴权）**：起好合并后的 analyze server，用一个 per-user token 连 `/mcp/` 握手。协议级端到端验证见 `tests/`（若后续补充）。

**方式 B（本地 stdio 调试）**：直接以 `TPA_API_KEY` 身份调工具函数，验证凭据与连通性：

```bash
cd /home/luohaizhao/workspace/analyze_trace/web
export TPA_API_KEY=...
/home/luohaizhao/workspace/analyze_trace/.venv/bin/python -c \
  "from tpa_mcp.server import tpa_get_job; print(tpa_get_job('10000679'))"
```

完整协议级验证另见 `tests/`（若后续补充）。

## 说明

- HTTP 层只用 Python 标准库 `urllib`，除 `fastmcp` 外无第三方依赖。
- 本地 stdio 的 `tpa_upload_trace` 会按 1 MiB 分块流式上传，不会把整份 trace 拼进内存；它会真实创建 job 并占用 TPA 存储/算力。
- AI 分析报告按 job 按需生成，未生成时工具会返回可读提示而非报错。
