# Trace Analyzer

GPU 性能分析工具，解析 PyTorch Profiler 生成的 Chrome Trace JSON 文件，提取并统计 GPU kernel、Triton kernel、ATen 算子、CNCL/NCCL 通信算子的耗时数据，支持单文件分析与双文件对比。

提供命令行脚本和 Web 可视化界面两种使用方式。

---

## 目录结构

```
analyze_json/
├── analyze_trace.py       # 核心分析脚本（命令行入口）
└── web/
    ├── server.py          # Web 服务器（FastAPI）
    ├── db.py              # SQLite 数据库操作
    ├── requirements.txt   # Python 依赖
    └── static/
        ├── index.html     # 前端页面
        ├── app.js         # Vue 3 前端逻辑
        ├── style.css      # 样式
        └── favicon.svg    # 图标
```

---

## 命令行使用

### 依赖

仅依赖 Python 标准库，无需额外安装。

### 单文件分析

```bash
python analyze_json/analyze_trace.py trace.json -o ./output
```

### 双文件对比

```bash
python analyze_json/analyze_trace.py baseline.json optimized.json -o ./output
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trace_files` | — | 1 或 2 个 PyTorch Profiler trace JSON 文件 |
| `-o, --output-dir` | `.` | 输出目录 |
| `-k, --kernel-types` | `gemm,embedding,pool` | 自定义 kernel 分类关键词，逗号分隔，大小写不敏感的子串匹配，首次命中生效 |
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

### 控制台输出示例

```
=== Per-Step Summary (10 steps) ===
step     step_dur(ms)   kernels    compute_kernel_dur(ms)   triton     triton_dur(ms)   ...
-------------------------------------------------------------------------------------
0        125.431        1842       98.762                   312        45.123           ...
1        123.887        1840       97.501                   312        44.891           ...
...
avg      124.659        1841.0     98.131                   312.0      45.007           ...

=== Kernel Type Breakdown (avg across 10 steps) ===
type         avg_count    avg_dur_ms
------------------------------------
triton       312.0        45.007
gemm         128.0        31.244
collective   24.0         8.103
other        1377.0       14.777
```

---

## Web 界面

提供浏览器操作界面，支持上传文件、查看结果、历史管理和双文件对比。

### 启动

```bash
cd analyze_json/web
pip install -r requirements.txt

python server.py                        # 默认 127.0.0.1:8181
python server.py --host 0.0.0.0 --port 8080
python server.py --no-download          # 禁止用户下载原始 trace 文件
```

然后打开浏览器访问 `http://127.0.0.1:8181`。

### 功能

- **提交分析**：拖拽或点击上传 1~2 个 trace JSON 文件，填写备注和 kernel-types 后提交，后台异步分析，实时显示进度
- **结果查看**：
  - 控制台输出（原始文本）
  - CSV 数据表格（可搜索、列排序、列宽拖拽调整、超长内容截断并 hover 显示全文）
  - Kernel 类型耗时柱状图
  - 一键下载各 CSV 文件 / 原始 trace 文件
- **历史管理**：按项目分组，支持重命名、移动项目、删除任务、删除原始文件
- **项目分组**：自定义项目，将相关任务归类管理
- **历史对比**：在"对比"标签页选择两个已完成的单文件任务直接发起对比分析，无需重新上传文件
- **Perfetto 集成**：点击"Perfetto ↗"按钮可在 Perfetto UI 中打开 trace 文件
- **gz 文件支持**：上传 `.json.gz` 文件时自动解压，下载时保留原始压缩格式

### 用户隔离与分享

- **无登录访问**：首次访问自动生成用户标识（UUID），数据按用户隔离
- **项目分享设置**：
  - **无密码公开**：任何人都可以查看项目内容
  - **密码保护**：访问者需输入密码才能查看
  - **私有**：仅项目创建者可查看
- **跨用户对比**：可选择任意两个可访问的任务进行对比分析

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8181` | 监听端口 |
| `--no-download` | off | 禁止下载上传的原始 trace 文件 |

---

## 工作原理

`analyze_trace.py` 对 Chrome Trace 格式的 JSON 执行两遍扫描：

1. **Pass 1** — 收集所有 `ProfilerStep#N` 事件，建立 `step_num → (start_ts, end_ts)` 映射
2. **Pass 2** — 遍历 `kernel` / `aten::*` / CNCL 事件，通过时间戳二分查找将每个事件归属到对应的 ProfilerStep

每个 ProfilerStep 内按 kernel 名称聚合耗时后，再对所有 step 求均值，消除单步抖动。

kernel 分类优先级：`triton` (名称前缀 `triton_`) > 用户自定义 `-k` 关键词 > `collective` (含 Collective name) > `other`。