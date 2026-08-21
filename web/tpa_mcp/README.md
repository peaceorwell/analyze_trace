# tpa_mcp — TPA 的 MCP 服务器

用 [FastMCP](https://gofastmcp.com) 把 [TPA](http://tpa.cambricon.com)（trace performance analyzer，PyTorch profiler 分析 Web 服务）的 REST API 封装成 Model Context Protocol 工具，让 Claude 可以直接查询/分析任务、读结果与 AI 报告，并上传 trace 触发新分析。

## 已并入 analyze server

本目录已合并进 analyze server（`web/server.py` 的 FastAPI app）：设置环境变量 `TPA_API_KEY` 后，analyze server 会在**同一进程、同一端口**暴露 MCP 端点：

```
MCP endpoint = http://<host>:8181/mcp/
```

- 现有 REST `/api/*` 端点与 MCP `/mcp/` 共存于一个 uvicorn 进程，无需单独部署。
- 若运行环境未设置 `TPA_API_KEY`，analyze server **优雅跳过**，不挂 `/mcp`，其余功能不受影响。
- 也可用 `web/tpa_mcp/http_app.py` 独立跑一个 MCP 服务（见下文「独立 HTTP 运行」）。

## 鉴权

需一个 TPA **access token**（大段随机字符串，仅创建时显示一次）。在 TPA 网页右上角 `...` → `用户设置` → `访问令牌` 创建，或用 `POST /api/tokens` 创建。

- `TPA_API_KEY`：必填，访问令牌。
- `TPA_BASE_URL`：可选，默认 `http://tpa.cambricon.com`。

例：

```bash
export TPA_API_KEY=your_access_token_here
```

## 连接方式

### 1. 经 analyze server（推荐，单进程）
起好 analyze server（含 `TPA_API_KEY`），Claude Code / 其它 MCP 客户端连 `http://<host>:8181/mcp/`。例如用 `claude mcp add` 指定远程端点：

```bash
# 指向已合并进 analyze server 的 /mcp 端点（地址按实际部署改）
claude mcp add tpa --transport http --url http://127.0.0.1:8181/mcp/
```

### 2. 本地 stdio 运行（无需 analyze server）
```bash
claude mcp add tpa \
  --env TPA_API_KEY=$TPA_API_KEY \
  -- python3 /home/luohaizhao/workspace/analyze_trace/web/tpa_mcp/server.py
```

### 3. 独立 HTTP 运行（备选，单独端口）
若不想经 analyze server，可用本目录 `http_app.py` 单独跑一个 MCP 服务：
```bash
cd /home/luohaizhao/workspace/analyze_trace/web && \
  TPA_API_KEY=$TPA_API_KEY \
  /home/luohaizhao/workspace/analyze_trace/.venv/bin/python -m uvicorn tpa_mcp.http_app:app \
  --host 0.0.0.0 --port 8080
# endpoint = http://<host>:8080/mcp
```

`TPA_API_KEY` 已 export 到 shell 时可直接用 `$TPA_API_KEY`；否则把整段 token 写在命令行或环境里。启动的 shell 需已 export 该 key。

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
| `tpa_upload_trace` | 上传 1~2 个 trace 文件触发分析，**异步立即返回** seq/job_id，不阻塞 |

任务可用其**数字 seq**（如 `10000679`）或 **UUID id** 引用，两种都能直接定位。

## 异步分析流程

`tpa_upload_trace` 上传后立即返回（不阻塞等待）。分析在 TPA 后台异步运行，配合以下工具完成「提交 → 轮询 → 取结果」：

```
1. seq = tpa_upload_trace(file_a=..., file_b=...).seq     # 立即返回
2. tpa_get_job_status(job=seq) ... 直到 status == "done"  # 轮询
3. tpa_get_job(job=seq) / tpa_get_job_result(job=seq, filename="all_kernels")  # 取结果
4. （可选）tpa_start_ai_analysis(job=seq)  →  tpa_get_ai_report(job=seq)      # AI 报告
```

所有工具都不做长轮询/阻塞，等待逻辑交给调用方（LLM 可自行多次调用 `tpa_get_job_status`）。

## 快速验证

用 `TPA_API_KEY` 直接调工具（不经握手，验证凭据与连通性）：

```bash
cd /home/luohaizhao/workspace/analyze_trace/web
export TPA_API_KEY=...
/home/luohaizhao/workspace/analyze_trace/.venv/bin/python -c \
  "from tpa_mcp.server import tpa_get_job; print(tpa_get_job('10000679'))"
```

完整协议级验证另见 `tests/`（若后续补充）。

## 说明

- HTTP 层只用 Python 标准库 `urllib`，除 `fastmcp` 外无第三方依赖。
- 上传 `tpa_upload_trace` 会真实创建 job 并进入分析队列，会占用 TPA 存储/算力，请按需使用。
- AI 分析报告按 job 按需生成，未生成时工具会返回可读提示而非报错。
