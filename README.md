# Trace Analyzer

GPU 性能分析工具，解析 PyTorch Profiler 生成的 Chrome Trace JSON 文件，提取并统计 GPU kernel、Triton kernel、ATen 算子、CNCL/NCCL 通信算子的耗时数据，支持单文件分析与双文件对比。

提供命令行脚本和 Web 可视化界面两种使用方式。

---

## 目录结构

```
.
├── analyze_trace.py       # 核心分析脚本（命令行入口）
├── web/
│   ├── server.py          # Web 服务器（FastAPI）
│   ├── db.py              # SQLite 数据库操作
│   ├── requirements.txt   # Python 依赖
│   ├── Dockerfile         # Docker 镜像构建
│   └── static/
│       ├── index.html     # 前端页面
│       ├── app.js         # Vue 3 前端逻辑
│       ├── style.css      # 样式
│       └── favicon.svg    # 图标
├── docker-compose.yml     # Docker Compose 部署配置
└── tests/                 # 测试
```

---

## 快速开始

```bash
# 命令行分析
python analyze_trace.py trace.json -o ./output

# Web 界面
cd web && pip install -r requirements.txt && python server.py
```

浏览器访问 `http://127.0.0.1:8181`。

---

## Web 界面

提供完整的前端操作界面，支持文件上传、分析结果查看、历史管理和双文件对比。

### 启动

```bash
cd web
pip install -r requirements.txt

python server.py                        # 默认 127.0.0.1:8181
python server.py --host 0.0.0.0 --port 8080
python server.py --no-download          # 禁止用户下载原始 trace 文件
```

### Docker 部署

```bash
# 使用 docker-compose（推荐）
docker-compose up -d

# 或手动构建运行
cd web
docker build -t trace-analyzer .
docker run -d -p 8181:8181 --name trace-analyzer -v trace_data:/app/storage trace-analyzer
```

**数据持久化**：SQLite 数据库和上传的文件存储在 `/app/storage` 目录，挂载 volume 后数据不会丢失。

**禁用文件下载**：设置环境变量 `TRACE_NO_DOWNLOAD=1`：
```bash
TRACE_NO_DOWNLOAD=1 docker-compose up -d
```

**启用本地代码执行**：Triton 代码运行和清除 cache 默认关闭；在可信本机环境中设置环境变量 `TRACE_ENABLE_CODE_EXEC=1`：
```bash
TRACE_ENABLE_CODE_EXEC=1 docker-compose up -d
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8181` | 监听端口 |
| `--no-download` | off | 禁止下载上传的原始 trace 文件 |

### 功能特性

#### 提交分析
- 拖拽或点击上传 `.json` 或 `.json.gz` 文件（PyTorch Profiler 导出的 Chrome Trace 格式）
- 选择所属项目（可选）、填写备注（可选）
- 点击"提交分析"，后台异步处理，实时显示上传进度和运行状态

#### 结果查看
- **控制台**：原始文本输出，含 Per-Step 摘要、Top 10 热点 kernel、Kernel 类型分布
- **图表**：Kernel 类型耗时柱状图（横向）和占比饼图，collective 类型不计入 compute 分析
- **CSV 表格**：支持搜索、列排序、列宽拖拽调整、超长内容截断并 hover 显示全文
- **下载 CSV**：表格右上角一键下载当前视图的 CSV 文件
- **Triton 代码执行**：设置 `TRACE_ENABLE_CODE_EXEC=1` 后，在 Triton 或 Triton Step N 表格中点击"运行"执行 kernel 代码，显示效率（GB/s）
- **清除 Cache**：设置 `TRACE_ENABLE_CODE_EXEC=1` 后，Triton Step N 表格中可清除 `/tmp/torchinductor_*` 缓存目录

#### 历史管理
- 侧栏按项目分组显示历史任务，支持折叠/展开
- 可选择项目过滤器筛选特定项目的历史记录
- 支持分页浏览
- 点击任务可查看详情

#### 项目管理
- 点击"+ 新建项目"创建项目，用于归类管理分析任务
- 支持重命名项目、删除项目
- 可将任务移动到其他项目

#### 项目恢复
- **删除项目时**：项目及其下的所有任务保存到回收站，而非真正删除
- **恢复项目**：在"找回项目"中可恢复近 10 天内删除的项目，任务会一起恢复
- **永久删除**：超过 10 天的已删除项目会显示"已过期"标签，只能永久删除

#### 历史对比
- 在侧栏"对比"标签页选择两个已完成的单文件任务
- 可选填备注和所属项目
- 发起对比分析，无需重新上传文件，直接复用已有数据
- 源文件删除后无法参与对比，会显示"已删除"标签

#### 文件操作
- **下载**：可下载原始 trace 文件（受 `--no-download` 控制）
- **Perfetto 集成**：点击"Perfetto ↗"按钮在 Perfetto UI 中打开 trace 文件
- **删除文件**：删除原始 trace 文件，对应任务会标记"已删除"，无法再参与对比

#### 其他
- **深色/浅色模式切换**
- **侧栏可拖拽调整宽度**，可折叠
- **点击标题可返回上传页面**
- **界面内嵌使用指南**：右上角"使用指南"按钮

### 反向代理部署

使用 Docker 部署后，可通过 Caddy 配置域名 + 自动 HTTPS：

```yaml
# docker-compose.yml
services:
  trace-analyzer:
    build:
      context: ./web
      dockerfile: Dockerfile
    volumes:
      - trace_analyzer_data:/app/storage
    restart: unless-stopped

  caddy:
    image: caddy:2-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - ./data/caddy:/data
    depends_on:
      - trace-analyzer
    restart: unless-stopped

volumes:
  trace_analyzer_data:
```

```nginx
# Caddyfile
trace.example.com {
    reverse_proxy trace-analyzer:8181
}
```

Caddy 会自动从 Let's Encrypt 申请 SSL 证书并续期。

---

## 命令行使用

### 依赖

仅依赖 Python 标准库，无需额外安装。

### 单文件分析

```bash
python analyze_trace.py trace.json -o ./output
```

### 双文件对比

```bash
python analyze_trace.py baseline.json optimized.json -o ./output
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trace_files` | — | 1 或 2 个 PyTorch Profiler trace JSON 文件 |
| `-o, --output-dir` | `.` | 输出目录 |
| `-s, --save-triton-csv` | off | 输出逐 step 的 Triton kernel 详情 CSV |
| `-c, --save-triton-code` | off | 将每个 Triton kernel 的生成代码保存为 `.py` 文件 |

### 输出文件

**单文件模式：**

| 文件 | 内容 |
|------|------|
| `all_kernels_avg.csv` | 所有 GPU kernel 按名称聚合的平均耗时和调用次数 |
| `triton_kernels_avg.csv` | Triton kernel 的平均耗时、IO 量、IO 效率 |
| `aten_ops_avg.csv` | ATen 算子的平均耗时和调用次数 |
| `kernel_types_avg.csv` | 各 kernel 类型（triton / gemm / … / other）的平均耗时汇总 |
| `cncl_ops_avg.csv` | CNCL/NCCL 通信算子的平均耗时 |
| `step_N_triton_kernels.csv` | （`-s`）每个 ProfilerStep 的 Triton kernel 详情 |
| `step_N_triton_codes/` | （`-c`）每个 Triton kernel 的生成源码 `.py` 文件 |

**双文件对比模式**（额外输出 `*_cmp.csv`）：

| 文件 | 内容 |
|------|------|
| `all_kernels_cmp.csv` | 两个 trace 的 kernel 耗时对比，含 delta 和百分比变化 |
| `triton_kernels_cmp.csv` | Triton kernel 对比 |
| `aten_ops_cmp.csv` | ATen 算子对比 |
| `kernel_types_cmp.csv` | kernel 类型汇总对比 |
| `cncl_ops_cmp.csv` | CNCL 算子对比 |

---

## 工作原理

`analyze_trace.py` 对 Chrome Trace 格式的 JSON 执行两遍扫描：

1. **Pass 1** — 收集所有 `ProfilerStep#N` 事件，建立 `step_num → (start_ts, end_ts)` 映射
2. **Pass 2** — 遍历 `kernel` / `aten::*` / CNCL 事件，通过时间戳二分查找将每个事件归属到对应的 ProfilerStep

每个 ProfilerStep 内按 kernel 名称聚合耗时后，再对所有 step 求均值，消除单步抖动。

kernel 自动分类逻辑：

- **triton**：名称以 `triton_` 开头，进一步细分为 `triton_mm`、`triton_reduce`、`triton_pointwise` 等子类型
- **collective**：TCDP 前缀或包含 `nccl`、`cncl`、`allreduce`、`allgather` 等集合通信关键词，单独统计，**不计入** compute 分析
- **语义聚类**：通过内置规则匹配 `gemm`、`conv`、`embedding`、`pool`、`norm`、`attention` 等常见类型
- **fallback**：无法匹配规则的 kernel 按名称前缀归入对应 family（保留原始大小写），兜底归入 `other`

所有非 collective 的 kernel family 均在 Kernel Type Breakdown 和图表中展示。
