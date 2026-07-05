# 用户设置功能 —— 执行方案

> 目标读者：一个冷启动的执行者（codex），没有本轮对话上下文。本文自包含，按顺序落地即可。
> 技术栈：后端 FastAPI（`web/server.py` 单文件） + `aiosqlite`（`web/db.py`），前端 Vue 3（`web/static/app.js` 单文件 setup 风格 + `web/static/index.html` 内联模板）。

## 1. 需求

给 Web 应用增加一个「用户设置」入口（弹窗），包含两类内容：

1. **显示偏好（纯前端，`localStorage`）**——不落数据库：
   - 主题：浅色 / 深色 / 跟随系统（复用现有 `toggleTheme` 逻辑）
   - 表格密度：紧凑 / 默认
   - 默认落地页：历史任务 / 上传页
   - 侧栏默认展开 / 收起
2. **账号资料（后端持久化）**——需要被其他用户在「灵感社区」看到，因此必须落库：
   - 自定义昵称（覆盖 LDAP 返回的 `display_name`，留空则回落 LDAP 名）
   - 头像颜色（从固定色板里选，**不做图片上传**，避免引入文件存储/裁剪）
   - 只读展示：登录用户名、邮箱、来源（LDAP）

设计约束：轻量。账号资料只加 1 张表的 2 列 + 2 个接口 + 登录处 1 段合并逻辑，头像用「色板 + 首字母」而非图片。

## 2. 关键现状（已核对）

- 用户会话：登录成功后写 `request.session["user"] = {username, display_name, email}`，见 `web/server.py` `login()`（约 4969 行）。`current_user(request)` 读 `request.state.user`（506 行）。
- 未开启鉴权时（`AUTH_ENABLED` 为假）`current_user_token()` 返回 `None`，此时身份统一按 `"local"` 处理。设置接口需兼容这种模式。
- `users` 表当前只有 `user_token`、`created_at` 两列（`web/db.py` 约 52 行）。已有 `add_column_if_missing(db, table, col, def)` 迁移助手（38 行），加列走它。
- `ensure_user_row(db, request)`（server.py 约 719 行）会 `INSERT OR IGNORE` 保证 `users` 行存在。
- 前端：`currentUser`（app.js 139 行）由 `loadMe()`（2203 行）从 `GET /api/me` 填充；`authRequired`（136 行）。Header 用户名展示在 `index.html:156`，「更多」下拉菜单在 `index.html:164`。已有 `showGuide` 弹窗模式：`const showGuide = ref(false)`（app.js 627）+ `index.html:1518` 的 `modal-mask`。
- 头像首字母函数 `feedbackUserInitial(value)`（app.js 2409）已存在；社区头像样式 `.feedback-avatar`（style.css 5519）。
- 发帖/回复作者信息取自 `_feedback_author(request)`（server.py 5259），返回 `(token, display)`，写入 `feedback_messages.user_display`。
- 主题预加载脚本在 `index.html:17` 一带，key 为 `tpa-theme`。

## 3. 固定色板（前后端共用枚举）

头像颜色只允许以下 key（后端校验、前端渲染各持一份，保持一致）：

```
slate, blue, indigo, violet, rose, amber, emerald, teal
```

- 每个 key 对应一个背景色 + 文字色，在 `style.css` 里用 `.avatar[data-color="blue"]` 之类落地。
- 存量数据 / 未设置时：按 `username` 做稳定哈希（如逐字符累加取模色板长度）得默认色，保证同一人颜色稳定。

## 4. 后端改动（`web/server.py` + `web/db.py`）

### 4.1 迁移（db.py，`init_db()` 里现有 `add_column_if_missing` 段落，约 239 行后追加）

```python
await add_column_if_missing(db, "users", "display_name_override", "TEXT DEFAULT NULL")
await add_column_if_missing(db, "users", "avatar_color", "TEXT DEFAULT NULL")
await add_column_if_missing(db, "feedback_messages", "avatar_color", "TEXT DEFAULT ''")
```

- `feedback_messages.avatar_color` 用于**快照**发帖时的头像色，避免用户改色后「改写历史帖子」，也省去展示时 JOIN `users`。

### 4.2 常量与校验助手

```python
AVATAR_COLORS = ("slate", "blue", "indigo", "violet", "rose", "amber", "emerald", "teal")
DISPLAY_NAME_MAX = 20

def _default_avatar_color(identity: str) -> str:
    key = (identity or "local")
    total = sum(ord(c) for c in key)
    return AVATAR_COLORS[total % len(AVATAR_COLORS)]

def _profile_identity(request) -> str:
    # 兼容未鉴权模式：优先 username，否则 "local"
    return current_user(request).get("username") or ("anonymous" if AUTH_ENABLED else "local")
```

### 4.3 读取用户覆盖字段的助手

```python
async def _load_user_overrides(db, identity: str) -> dict:
    row = await (await db.execute(
        "SELECT display_name_override, avatar_color FROM users WHERE user_token=?",
        (identity,),
    )).fetchone()
    if not row:
        return {"display_name_override": None, "avatar_color": None}
    return {"display_name_override": row[0], "avatar_color": row[1]}
```

### 4.4 接口：`GET /api/user/profile`

- 若 `AUTH_ENABLED` 且未登录 → 401（复用 `current_user_token(request)` 触发）。
- 逻辑：`ensure_user_row` → `_load_user_overrides`，返回：

```json
{
  "username": "...",
  "email": "...",
  "source": "ldap",
  "display_name_ldap": "LDAP 原始名",
  "display_name_override": null,
  "display_name_effective": "覆盖优先，否则 LDAP 名，否则 username",
  "avatar_color": "blue 或 null",
  "avatar_color_effective": "有则用，无则 _default_avatar_color(username)",
  "avatar_colors": ["slate", ...]
}
```

- `display_name_ldap` 取 `current_user(request).get("display_name")`（注意：session 里可能已被覆盖，见 4.6；建议 session 里额外存一份 `display_name_ldap` 原值，或直接以 override 是否存在来判断——见下）。

### 4.5 接口：`PUT /api/user/profile`

- Body：`{"display_name": str|null, "avatar_color": str|null}`。
- 校验：
  - `display_name`：`strip` 后长度 `<= DISPLAY_NAME_MAX`；空串按 `NULL` 存（回落 LDAP）。
  - `avatar_color`：必须属于 `AVATAR_COLORS` 或为 `null`；否则 422。
- 落库：`UPDATE users SET display_name_override=?, avatar_color=? WHERE user_token=?`（先 `ensure_user_row`）。
- **同步当前会话**，让改动即时生效、无需重登：
  ```python
  effective = display_name_override or session_ldap_name or username
  request.session["user"]["display_name"] = effective
  request.session["user"]["avatar_color"] = avatar_color or _default_avatar_color(username)
  request.state.user = request.session["user"]
  ```
- 返回体同 `GET /api/user/profile`。
- 记一条 `write_audit(db, request, "user.profile.update", ...)`（与现有审计写法一致）。

### 4.6 登录处合并（`login()`，约 4969 行）

写 session 前查一次覆盖字段，让重登后自定义仍在。建议 session 结构：

```python
overrides = await _load_user_overrides(db, user["username"])  # 需在 get_db 之后
ldap_name = user.get("display_name") or user["username"]
request.session["user"] = {
    "username": user["username"],
    "display_name": overrides["display_name_override"] or ldap_name,
    "display_name_ldap": ldap_name,          # 原始 LDAP 名，供 GET profile 展示 & 回落
    "email": user.get("email") or "",
    "avatar_color": overrides["avatar_color"] or _default_avatar_color(user["username"]),
}
```

注意：现有代码里 `ensure_user_row` 在 `get_db()` 之后调用，把 `_load_user_overrides` 也放到那个 `db` 块内，调整语句顺序即可。

### 4.7 `/api/me` 附带头像色

`get_me()`（4910 行）当前直接回 `current_user(request)`。session 里已含 `avatar_color`，会自动带出，前端可直接用。无需额外改动，但确认 `user` 字典里含 `avatar_color` 与 `display_name`。

### 4.8 社区作者信息带上头像色

- `_feedback_author(request)`（5259 行）改为返回三元组 `(token, display, avatar_color)`：
  ```python
  def _feedback_author(request):
      user = current_user(request)
      token = user.get("username") or request_user(request)
      display = user.get("display_name") or token
      color = user.get("avatar_color") or _default_avatar_color(token)
      return token, display, color
  ```
- 更新全部调用点（grep `_feedback_author`：5259/6213/6308/6415/6511/6611）。只关心 color 的发帖/回复处（6415 一带）把它写进 `feedback_messages.avatar_color`；只取 token 的调用点用 `token, _, _ = _feedback_author(...)`。
- 帖子/回复的读取查询里 SELECT 出 `avatar_color`，随 `user_display` 一起返回给前端（存量为空串，前端回落哈希默认色）。

## 5. 前端改动

### 5.1 显示偏好（纯前端）

- 新增 `localStorage` key `tpa-settings`，存 `{ tableDensity, defaultLanding, sidebarDefaultCollapsed }`。主题继续用现有 `tpa-theme`，不迁移。
- app.js：`const userPrefs = reactive({...默认值})`，初始化时 `Object.assign` 读取 `localStorage`；`saveUserPrefs()` 写回并应用到 DOM。
- 应用方式与 `data-theme` 一致：在根节点设 `document.documentElement.setAttribute('data-density', userPrefs.tableDensity)`；`defaultLanding` 影响登录后首次 `$router.push` 目标；`sidebarDefaultCollapsed` 作为 `sidebarCollapsed` 初值（仅当无当次会话覆盖时）。
- `style.css`：加 `[data-density="compact"]` 下表格 `padding`/`line-height` 收紧的一版覆盖（复用现有表格类，改变量或直接覆盖 padding）。

### 5.2 设置弹窗

- app.js：`const showSettings = ref(false)`；`const profileForm = reactive({ display_name:"", avatar_color:null, loading:false, saving:false, error:"" })`。
- 打开时 `openSettings()`：`showSettings.value=true` 并 `GET /api/user/profile` 填 `profileForm`（未鉴权模式下账号资料区可整块隐藏或标注「本地模式」）。
- `saveProfile()`：`PUT /api/user/profile`，成功后把返回的 `display_name_effective`/`avatar_color_effective` 合并进 `currentUser.value`，并调 `loadMe()` 或直接赋值刷新 header。
- index.html：
  - 「更多」下拉（`index.html:164` 内）加一项：`<button type="button" @click="openSettings(); closeActionMenu()">用户设置</button>`。
  - 新增弹窗 markup（仿 `showGuide` 的 `modal-mask`）：
    - 「显示偏好」：主题三选（可复用现有 toggle 或做成 radio）、表格密度、默认落地页、侧栏默认状态。
    - 「账号资料」：昵称 input（placeholder 显示 LDAP 名，留空即回落）+ 一排色板圆点单选头像色 + 只读的用户名/邮箱/来源。

### 5.3 头像组件化

- header 用户名旁（`index.html:156`）加一个头像圆点：背景色取 `currentUser.avatar_color`，文字取 `feedbackUserInitial(currentUser.display_name || currentUser.username)`。
- 社区渲染头像处优先用消息快照的 `item.avatar_color`（存量为空 → 用基于 `user_display`/`user_token` 的哈希默认色，前端补一个和后端 `_default_avatar_color` 同规则的 `avatarColorFor(name)` 函数）。
- style.css：抽一个通用 `.avatar[data-color]` 色板规则（8 个 key 各一条），`.feedback-avatar` 可复用同一套背景色变量。

## 6. 测试

- 后端：`tests/test_web_api.py` 加用例——`PUT /api/user/profile` 校验（超长昵称 422、非法色 422、正常 200 且 `GET` 回读一致）；未鉴权模式下接口行为；社区发帖后 `avatar_color` 落到消息并回读。
- 手工核对：改昵称/头像后 header 立即变化、退出重登仍在、社区历史帖头像色不被后续改色影响。

## 7. 落地顺序（建议）

1. db.py 加三列迁移。
2. server.py：常量 + 助手 + `_load_user_overrides` + 两个接口 + `login()` 合并 + `_feedback_author` 三元组及调用点。
3. 前端：显示偏好（localStorage）→ 设置弹窗 → 头像展示 → 社区头像快照。
4. 补测试，跑 `pytest`。

## 8. 明确不做

- 不做头像图片上传/存储/裁剪（只用色板 + 首字母）。
- 不做改密码（密码在 LDAP 侧管理）。
- 显示偏好不落库、不跨设备同步。
