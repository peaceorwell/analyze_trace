# 实验树 · 前端设计 / 配色 / 交互改进规格

> 面向 Codex 的纯前端改进（只动 `web/static/style.css` 的 `--exp-*` 变量与 `.exp-*` 规则、`web/static/app.js` 的 `ExperimentTree` 模板/节点结构与少量布局逻辑；**不动后端、不改接口**）。
> 现状：Codex 已实现一套语义配色（`--exp-best`翠绿 / `--exp-good`青 / `--exp-bad`玫红 / 默认 slate / 边 indigo），功能可用。本规格在此之上做"协调化"重构。
> 类名/变量以当前实现为准：节点 `.exp-node`（带 `.result-best/.result-good/.result-bad`）、`.exp-node-head`、`.exp-status-dot`、`.exp-node-stats`；边 `.exp-edge-label`、`.exp-edge-path`、`.exp-edge-arrow`；画布 `.exp-canvas`、`.exp-layer`；变量在 `style.css` 暗色 `:root`（约 `:33-59`）和浅色 `[data-theme="light"]`（约 `:118-140`）。

---

## 0. 设计总原则（决定后面所有改动）
**把语义色从「节点」挪到「边」。**
- 节点 = 中性（一个 timeline 是个"物"，本身无好坏）：所有节点统一中性卡片色，只用**小标记**区分 baseline / 当前最优。
- 边 = 带语义色（"优化"是变化，好坏发生在边上）：delta 芯片绿=变快、红=退化。
- 收益：画布安静、重点集中、暗色一致、与 app 的 indigo+orange 调性对齐。

---

## A. 配色重构（P0，影响最大）

### A1 节点中性化
- [ ] 移除按结果给整张节点卡染色：`.exp-node.result-best/.result-good/.result-bad` 不再改 `background/border` 为绿/青/红。
- [ ] 所有节点统一用 `--exp-node-bg` / `--exp-node-border`（已存在，中性）。
- [ ] **baseline**（无入边节点）：加一个 `基线` 标签（`--bg-accent`/indigo 浅底），不靠颜色。
- [ ] **当前最优**（done 节点里 e2e 最小）：用 **indigo 描边 + 右上角「最优」角标**（呼应品牌主色 `--purple`/侧边栏 qv），不要用第三种绿。
- [ ] **退化节点**（相对父 e2e 变慢）：不整块灌红，仅保留中性卡 + 该节点入边的红色 delta 芯片表达。

### A2 语义色对齐 app 体系（只保留两种结果色 + 中性）
- [ ] 改善 = **一种绿**：用 emerald（与侧边栏 `qv-mine` 的 `#10b981` 同源），替换 `--exp-good` 的青色一族（`#06b6d4/#22d3ee`）。删除"best 翠绿 / good 青"两套绿，避免一眼分不清。
- [ ] 退化 = app 的红：`--exp-bad-*` 收敛到 `--red`(`#ef4444`) 系；**软化**：用浅红 tint + 细红边（参考 `--red-bg`），删掉 `rgba(136,19,55,.82)` 这种深玫红整块填充。
- [ ] 中性/基线强调 = `--purple`/`--purple-l`（indigo）。
- [ ] 边的连线/箭头保持 indigo（`--exp-edge-active`），与选中态区分（见 C3）。
- [ ] 浅色 + 暗色两套变量都要同步改（`:33-59` 与 `:118-140`），保证暗色默认节点不再发白割裂。

### A3 delta 芯片统一规则
- [ ] 时间类指标：`delta_pct<0`（变快）→ 绿芯片（`--bg-success`/`--green` 系）；`>0`（变慢）→ 红芯片（`--red` 系）；`null`/缺失 → 灰。
- [ ] 芯片用 `tabular-nums`，箭头 ▼/▲ + 百分比，短文案（`e2e ▼16.9%`）。

---

## B. 节点卡信息层级（P0）

### B1 主指标当主角
- [ ] 卡片顶部：节点名（+ 基线/最优标记）一行；下面一个**大号 e2e 数值**（~22px，`tabular-nums`）+ 单位 `ms · e2e` + 一个 e2e delta 芯片。
- [ ] 次要指标（compute / kernel 等）缩小成一行小字（`--text-secondary`），或仅 hover 展开；不要三行平铺占满。

### B2 消除截断
- [ ] **值与 delta 拆开**：值单独显示，delta 做成右侧/下方独立彩色芯片，杜绝 `Comp… 54.09 ms (+1.15 …` 这种一行挤俩数被截断。
- [ ] 数值统一 `font-variant-numeric: tabular-nums`；过长名称 `text-overflow:ellipsis`，但**数值与 delta 永不截断**（必要时加宽卡或换行）。
- [ ] 复核默认缩放下 E2E/compute/kernel 全部完整可读，深浅主题均不溢出。

---

## C. 交互（P1）

### C1 边的状态语义
- [ ] **已建立的关系用实线 + 实心 pill**（现为虚线，读起来像草稿）。
- [ ] 虚线/半透明只保留给"对比生成中"（`compare_job_id` 异步生成未完成）这类未决态。

### C2 未连接任务 → 可连入，而非只"进入"
- [ ] 右栏未连接列表的主操作从「进入」改为「**连入树 / 设为优化结果**」（打开建边弹窗并预填该 job）；「进入 job 详情」降为次要。
- [ ] 更进一步（可选）：未连接节点以**幽灵卡片**摆在画布底部托盘，从托盘拖/点即可连入，连接动作一目了然。

### C3 三态分离：选中 / 最优 / 悬停
- [ ] **选中态独立**：用 indigo 实心 ring（`box-shadow: 0 0 0 2px var(--purple)`）表示当前选中，**不要**和"最优"的语义标记复用同一视觉。
- [ ] 悬停：轻微抬升/边色加深即可，别与选中混淆。

### C4 画布可操作性暗示
- [ ] cursor：画布空白 `grab`/`grabbing`，节点 `move`，关系标签 resize 手柄 `nwse-resize`。
- [ ] 工具条或画布角落加一行浅提示：`拖拽排布 · 滚轮缩放 · 拖空白平移`。
- [ ] 关系标签 resize 手柄 hover 时再显现并加深；节点不提供单独缩放，只跟随画布整体缩放。

### C5 加载自动适配
- [ ] 进入/刷新后 **auto-fit 居中**：根据节点包围盒设置初始 `view.tx/ty/scale`，让内容居中且不被工具条裁切（现状内容挤左上、baseline 顶部被裁）。
- [ ] 「自动整理」后同样 auto-fit。

---

## D. 细节
- [ ] 网格点（`--exp-grid-dot`）暗色下可再降一档，减少底噪。
- [ ] 主 CTA「标记优化关系」在空/小树时更突出（primary 实心），有内容后可降为常规。
- [ ] 节点 `status` 非 done（分析中/失败）时，指标显示「—」并加状态标识，不显示误导数字。

---

## E. 验收
- [ ] 画布上**节点为中性、颜色集中在边的 delta 芯片**；满屏不再是大色块。
- [ ] best/baseline 用标记区分，**改善只有一种绿、退化为软化红**，与侧边栏 qv（emerald/amber/sky）和品牌 indigo+orange 视觉一致。
- [ ] 节点卡主指标醒目、值与 delta 不截断；深浅主题都完整可读，**暗色默认节点不再发白割裂**。
- [ ] 边为实线 pill；未连接项可一键连入；选中态是 indigo ring，与最优标记不撞。
- [ ] 进入/刷新/自动整理后画布内容居中、不被裁切。
- [ ] `pytest tests/test_web_api.py` 仍 104 passed（本批为纯前端，不应影响）；浏览器预览复核深浅主题、无 console error。

## F. 范围与不动项
- 仅前端：`style.css`（`--exp-*` 与 `.exp-*`）、`app.js`（`ExperimentTree` 模板与节点/布局逻辑）。
- **不改**后端接口、数据结构、`perf_summary` 字段；node 颜色判定所需的 e2e/parent 关系数据 GET 已返回，前端据此判断 baseline/最优/改善退化即可。
