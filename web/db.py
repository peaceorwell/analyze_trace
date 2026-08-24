import aiosqlite
import os

DEFAULT_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
DB_PATH = os.environ.get(
    "TRACE_DB_PATH",
    os.path.join(os.environ.get("TRACE_STORAGE_DIR", DEFAULT_STORAGE_DIR), "jobs.db"),
)
DB_TIMEOUT_SECONDS = float(os.environ.get("TRACE_DB_TIMEOUT_SECONDS", "30"))
DB_BUSY_TIMEOUT_MS = max(1000, int(DB_TIMEOUT_SECONDS * 1000))


async def configure_connection(db):
    await db.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")


async def get_db():
    db = await aiosqlite.connect(DB_PATH, timeout=DB_TIMEOUT_SECONDS)
    db.row_factory = aiosqlite.Row
    await configure_connection(db)
    return db


async def table_exists(db, table_name):
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return await cursor.fetchone() is not None


async def column_exists(db, table_name, column_name):
    cursor = await db.execute(f"PRAGMA table_info({table_name})")
    rows = await cursor.fetchall()
    return any(row[1] == column_name for row in rows)


async def add_column_if_missing(db, table_name, column_name, column_def):
    if not await table_exists(db, table_name):
        return
    if not await column_exists(db, table_name, column_name):
        await db.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH, timeout=DB_TIMEOUT_SECONDS) as db:
        await configure_connection(db)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_token  TEXT PRIMARY KEY,
                email       TEXT DEFAULT NULL,
                display_name_override TEXT DEFAULT NULL,
                avatar_color TEXT DEFAULT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS folders (
                id           TEXT PRIMARY KEY,
                user_token   TEXT REFERENCES users(user_token) ON DELETE CASCADE,
                name         TEXT NOT NULL,
                password_hash TEXT DEFAULT NULL,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS projects (
                id           TEXT PRIMARY KEY,
                user_token   TEXT REFERENCES users(user_token) ON DELETE CASCADE,
                folder_id    TEXT REFERENCES folders(id) ON DELETE SET NULL,
                name         TEXT NOT NULL,
                description  TEXT DEFAULT '',
                password_hash TEXT DEFAULT NULL,
                is_public    INTEGER DEFAULT 0,
                compute_target_ms REAL,
                metric_targets_json TEXT DEFAULT '{}',
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS project_favorites (
                project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                user_token   TEXT NOT NULL,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(project_id, user_token)
            );

            CREATE TABLE IF NOT EXISTS project_orders (
                project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                user_token   TEXT NOT NULL,
                sort_order   INTEGER NOT NULL,
                updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(project_id, user_token)
            );

            CREATE TABLE IF NOT EXISTS resource_access_requests (
                id                   TEXT PRIMARY KEY,
                resource_kind        TEXT NOT NULL CHECK(resource_kind IN ('project','job')),
                resource_id          TEXT NOT NULL,
                project_id           TEXT,
                owner_user_token     TEXT NOT NULL,
                requester_user_token TEXT NOT NULL,
                requester_display    TEXT DEFAULT '',
                requester_email      TEXT DEFAULT '',
                status               TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved')),
                notification_status  TEXT DEFAULT '',
                notification_detail  TEXT DEFAULT '',
                created_at           DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at           DATETIME DEFAULT CURRENT_TIMESTAMP,
                decided_at           DATETIME DEFAULT NULL,
                decided_by           TEXT DEFAULT '',
                UNIQUE(resource_kind, resource_id, requester_user_token)
            );

            CREATE TABLE IF NOT EXISTS resource_access_grants (
                resource_kind TEXT NOT NULL CHECK(resource_kind IN ('project','job')),
                resource_id   TEXT NOT NULL,
                user_token    TEXT NOT NULL,
                granted_by    TEXT NOT NULL,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(resource_kind, resource_id, user_token)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id               TEXT PRIMARY KEY,
                seq              INTEGER,
                project_id       TEXT REFERENCES projects(id) ON DELETE CASCADE,
                user_token       TEXT REFERENCES users(user_token) ON DELETE CASCADE,
                created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
                label            TEXT DEFAULT '',
                is_pinned        INTEGER DEFAULT 0,
                mode             TEXT CHECK(mode IN ('single','compare')) NOT NULL,
                input_kind       TEXT DEFAULT 'timeline',

                file_a_name      TEXT,
                file_a_path      TEXT,
                file_a_gzip_path TEXT,
                file_a_exists    INTEGER DEFAULT 1,
                file_b_name      TEXT,
                file_b_path      TEXT,
                file_b_gzip_path TEXT,
                file_b_exists    INTEGER DEFAULT 1,

                source_job_a     TEXT REFERENCES jobs(id) ON DELETE SET NULL,
                source_job_b     TEXT REFERENCES jobs(id) ON DELETE SET NULL,

                step_filter_a    TEXT DEFAULT '',
                step_filter_b    TEXT DEFAULT '',
                label_filter_a   TEXT DEFAULT '',
                label_filter_b   TEXT DEFAULT '',

                save_triton_csv  INTEGER DEFAULT 0,
                save_triton_code INTEGER DEFAULT 0,

                status           TEXT CHECK(status IN ('pending','running','done','error'))
                                     DEFAULT 'pending',
                console_out      TEXT DEFAULT '',
                error_msg        TEXT DEFAULT '',
                result_dir       TEXT DEFAULT '',

                owned_bytes          INTEGER,
                result_bytes         INTEGER,
                original_trace_bytes INTEGER
            );

            CREATE TABLE IF NOT EXISTS experiment_edges (
                id                  TEXT PRIMARY KEY,
                project_id          TEXT REFERENCES projects(id) ON DELETE CASCADE,
                user_token          TEXT,
                parent_job_id       TEXT REFERENCES jobs(id) ON DELETE CASCADE,
                child_job_id        TEXT REFERENCES jobs(id) ON DELETE CASCADE,
                title               TEXT DEFAULT '',
                description         TEXT DEFAULT '',
                perf_summary        TEXT DEFAULT '',
                perf_summary_edited INTEGER DEFAULT 0,
                variables           TEXT DEFAULT '[]',
                label_x             REAL,
                label_y             REAL,
                label_width         REAL,
                label_height        REAL,
                compare_job_id      TEXT REFERENCES jobs(id) ON DELETE SET NULL,
                created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS experiment_node_layout (
                project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                x           REAL,
                y           REAL,
                scale       REAL DEFAULT 1.0,
                width       REAL,
                height      REAL,
                pinned      INTEGER DEFAULT 1,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(project_id, job_id)
            );

            CREATE TABLE IF NOT EXISTS deleted_projects (
                id           TEXT PRIMARY KEY,
                user_token   TEXT,
                folder_id    TEXT,
                name         TEXT NOT NULL,
                description  TEXT DEFAULT '',
                password_hash TEXT DEFAULT NULL,
                is_public    INTEGER DEFAULT 0,
                compute_target_ms REAL,
                metric_targets_json TEXT DEFAULT '{}',
                created_at   DATETIME,
                deleted_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                experiment_edges_json TEXT DEFAULT '[]',
                experiment_node_layout_json TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id            TEXT PRIMARY KEY,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
                user          TEXT DEFAULT 'local',
                action        TEXT NOT NULL,
                resource_type TEXT DEFAULT '',
                resource_id   TEXT DEFAULT '',
                ip            TEXT DEFAULT '',
                detail_json   TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS usage_daily (
                day           TEXT NOT NULL,
                user_token    TEXT NOT NULL,
                display_name  TEXT DEFAULT '',
                request_count INTEGER DEFAULT 0,
                last_path     TEXT DEFAULT '',
                last_method   TEXT DEFAULT '',
                last_status   INTEGER DEFAULT 0,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen_at  TEXT DEFAULT '',
                PRIMARY KEY(day, user_token)
            );

            CREATE TABLE IF NOT EXISTS feedback_messages (
                id           TEXT PRIMARY KEY,
                parent_id    TEXT REFERENCES feedback_messages(id) ON DELETE CASCADE,
                user_token   TEXT DEFAULT '',
                user_display TEXT DEFAULT '',
                avatar_color TEXT DEFAULT '',
                body         TEXT DEFAULT '',
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                edited_at    DATETIME DEFAULT NULL,
                edit_count   INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS feedback_attachments (
                id           TEXT PRIMARY KEY,
                message_id   TEXT REFERENCES feedback_messages(id) ON DELETE CASCADE,
                filename     TEXT DEFAULT '',
                stored_path  TEXT NOT NULL,
                content_type TEXT DEFAULT '',
                size_bytes   INTEGER DEFAULT 0,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS feedback_reactions (
                message_id   TEXT REFERENCES feedback_messages(id) ON DELETE CASCADE,
                user_token   TEXT DEFAULT '',
                emoji        TEXT NOT NULL,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(message_id, user_token, emoji)
            );

            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                applied_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS api_tokens (
                token_hash   TEXT PRIMARY KEY,
                user_token   TEXT NOT NULL REFERENCES users(user_token),
                name         TEXT NOT NULL DEFAULT '',
                scope        TEXT NOT NULL DEFAULT 'readonly',
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                revoked      INTEGER DEFAULT 0,
                revoked_at   DATETIME DEFAULT NULL,
                last_used_at TEXT DEFAULT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_api_tokens_user ON api_tokens(user_token);
        """)

        # Column migrations for databases created by earlier versions.
        await add_column_if_missing(db, "users", "created_at", "DATETIME")
        await add_column_if_missing(db, "users", "email", "TEXT DEFAULT NULL")
        await add_column_if_missing(db, "users", "display_name_override", "TEXT DEFAULT NULL")
        await add_column_if_missing(db, "users", "avatar_color", "TEXT DEFAULT NULL")
        await add_column_if_missing(db, "projects", "user_token", "TEXT")
        await add_column_if_missing(db, "projects", "password_hash", "TEXT DEFAULT NULL")
        await add_column_if_missing(db, "projects", "is_public", "INTEGER DEFAULT 0")
        await add_column_if_missing(db, "projects", "compute_target_ms", "REAL")
        await add_column_if_missing(db, "projects", "metric_targets_json", "TEXT DEFAULT '{}'")
        await add_column_if_missing(db, "projects", "created_at", "DATETIME")
        await add_column_if_missing(db, "projects", "folder_id", "TEXT")
        await add_column_if_missing(db, "deleted_projects", "compute_target_ms", "REAL")
        await add_column_if_missing(db, "deleted_projects", "metric_targets_json", "TEXT DEFAULT '{}'")
        await add_column_if_missing(db, "jobs", "user_token", "TEXT")
        await add_column_if_missing(db, "jobs", "owned_bytes", "INTEGER")
        await add_column_if_missing(db, "jobs", "result_bytes", "INTEGER")
        await add_column_if_missing(db, "jobs", "original_trace_bytes", "INTEGER")
        await add_column_if_missing(db, "jobs", "is_pinned", "INTEGER DEFAULT 0")
        await add_column_if_missing(db, "jobs", "step_filter_a", "TEXT DEFAULT ''")
        await add_column_if_missing(db, "jobs", "step_filter_b", "TEXT DEFAULT ''")
        await add_column_if_missing(db, "jobs", "label_filter_a", "TEXT DEFAULT ''")
        await add_column_if_missing(db, "jobs", "label_filter_b", "TEXT DEFAULT ''")
        await add_column_if_missing(db, "jobs", "seq", "INTEGER")
        await add_column_if_missing(db, "jobs", "input_kind", "TEXT DEFAULT 'timeline'")
        await add_column_if_missing(db, "folders", "password_hash", "TEXT DEFAULT NULL")
        await add_column_if_missing(db, "feedback_messages", "edited_at", "DATETIME DEFAULT NULL")
        await add_column_if_missing(db, "feedback_messages", "edit_count", "INTEGER DEFAULT 0")
        await add_column_if_missing(db, "feedback_messages", "avatar_color", "TEXT DEFAULT ''")
        await add_column_if_missing(db, "experiment_edges", "variables", "TEXT DEFAULT '[]'")
        await add_column_if_missing(db, "experiment_edges", "label_x", "REAL")
        await add_column_if_missing(db, "experiment_edges", "label_y", "REAL")
        await add_column_if_missing(db, "experiment_edges", "label_width", "REAL")
        await add_column_if_missing(db, "experiment_edges", "label_height", "REAL")
        await add_column_if_missing(db, "experiment_node_layout", "scale", "REAL DEFAULT 1.0")
        await add_column_if_missing(db, "experiment_node_layout", "width", "REAL")
        await add_column_if_missing(db, "experiment_node_layout", "height", "REAL")

        await db.executescript("""
            CREATE TABLE IF NOT EXISTS job_seq_counter (
                id    INTEGER PRIMARY KEY CHECK(id=1),
                value INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO job_seq_counter(id, value) VALUES (1, 10000000);
        """)
        current_counter_row = await (
            await db.execute("SELECT value FROM job_seq_counter WHERE id=1")
        ).fetchone()
        max_seq_row = await (
            await db.execute("SELECT COALESCE(MAX(seq), 10000000) FROM jobs")
        ).fetchone()
        next_seq = max(
            int(current_counter_row[0] if current_counter_row else 10000000),
            int(max_seq_row[0] if max_seq_row else 10000000),
            10000000,
        )
        missing_seq_rows = await (
            await db.execute("SELECT id FROM jobs WHERE seq IS NULL ORDER BY created_at, id")
        ).fetchall()
        for row in missing_seq_rows:
            next_seq += 1
            await db.execute("UPDATE jobs SET seq=? WHERE id=?", (next_seq, row[0]))
        await db.execute("UPDATE job_seq_counter SET value=? WHERE id=1", (next_seq,))

        await db.executescript("""
            CREATE TABLE IF NOT EXISTS deleted_jobs (
                id               TEXT PRIMARY KEY,
                seq              INTEGER,
                project_id       TEXT,
                user_token       TEXT,
                created_at       DATETIME,
                label            TEXT DEFAULT '',
                is_pinned        INTEGER DEFAULT 0,
                mode             TEXT,
                input_kind       TEXT DEFAULT 'timeline',

                file_a_name      TEXT,
                file_a_path      TEXT,
                file_a_gzip_path TEXT,
                file_a_exists    INTEGER DEFAULT 1,
                file_b_name      TEXT,
                file_b_path      TEXT,
                file_b_gzip_path TEXT,
                file_b_exists    INTEGER DEFAULT 1,

                source_job_a     TEXT,
                source_job_b     TEXT,

                step_filter_a    TEXT DEFAULT '',
                step_filter_b    TEXT DEFAULT '',
                label_filter_a   TEXT DEFAULT '',
                label_filter_b   TEXT DEFAULT '',

                save_triton_csv  INTEGER DEFAULT 0,
                save_triton_code INTEGER DEFAULT 0,

                status           TEXT DEFAULT 'pending',
                console_out      TEXT DEFAULT '',
                error_msg        TEXT DEFAULT '',
                result_dir       TEXT DEFAULT '',
                deleted_at       DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_deleted_projects_user ON deleted_projects(user_token);
            CREATE INDEX IF NOT EXISTS idx_deleted_projects_deleted_at ON deleted_projects(deleted_at);
            CREATE INDEX IF NOT EXISTS idx_deleted_jobs_project ON deleted_jobs(project_id);
            CREATE INDEX IF NOT EXISTS idx_deleted_jobs_deleted_at ON deleted_jobs(deleted_at);
            CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_token);
            CREATE INDEX IF NOT EXISTS idx_project_favorites_user ON project_favorites(user_token, created_at);
            CREATE INDEX IF NOT EXISTS idx_project_orders_user_order ON project_orders(user_token, sort_order);
            CREATE INDEX IF NOT EXISTS idx_resource_access_requests_owner_status
                ON resource_access_requests(owner_user_token, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_resource_access_requests_requester
                ON resource_access_requests(requester_user_token, created_at);
            CREATE INDEX IF NOT EXISTS idx_resource_access_requests_project
                ON resource_access_requests(project_id, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_resource_access_grants_user
                ON resource_access_grants(user_token, resource_kind, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_token);
            CREATE INDEX IF NOT EXISTS idx_jobs_project_created ON jobs(project_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_mode_status_created ON jobs(mode, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_pinned_created ON jobs(is_pinned, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_seq ON jobs(seq);
            CREATE INDEX IF NOT EXISTS idx_jobs_source_a ON jobs(source_job_a);
            CREATE INDEX IF NOT EXISTS idx_jobs_source_b ON jobs(source_job_b);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_expedge_pair ON experiment_edges(project_id, parent_job_id, child_job_id);
            CREATE INDEX IF NOT EXISTS idx_expedge_project ON experiment_edges(project_id);
            CREATE INDEX IF NOT EXISTS idx_expedge_parent ON experiment_edges(parent_job_id);
            CREATE INDEX IF NOT EXISTS idx_expedge_child ON experiment_edges(child_job_id);
            CREATE INDEX IF NOT EXISTS idx_folders_user ON folders(user_token);
            CREATE INDEX IF NOT EXISTS idx_projects_folder ON projects(folder_id);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
            CREATE INDEX IF NOT EXISTS idx_usage_daily_day ON usage_daily(day);
            CREATE INDEX IF NOT EXISTS idx_usage_daily_user ON usage_daily(user_token);
            CREATE INDEX IF NOT EXISTS idx_feedback_messages_parent_created ON feedback_messages(parent_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_feedback_attachments_message ON feedback_attachments(message_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_reactions_message ON feedback_reactions(message_id);
            CREATE TRIGGER IF NOT EXISTS trg_jobs_assign_seq
            AFTER INSERT ON jobs WHEN NEW.seq IS NULL
            BEGIN
                UPDATE job_seq_counter SET value = value + 1 WHERE id = 1;
                UPDATE jobs SET seq = (SELECT value FROM job_seq_counter WHERE id = 1) WHERE id = NEW.id;
            END;
        """)
        await add_column_if_missing(db, "deleted_jobs", "is_pinned", "INTEGER DEFAULT 0")
        await add_column_if_missing(db, "deleted_jobs", "seq", "INTEGER")
        await add_column_if_missing(db, "deleted_jobs", "input_kind", "TEXT DEFAULT 'timeline'")
        await add_column_if_missing(db, "deleted_jobs", "step_filter_a", "TEXT DEFAULT ''")
        await add_column_if_missing(db, "deleted_jobs", "step_filter_b", "TEXT DEFAULT ''")
        await add_column_if_missing(db, "deleted_jobs", "label_filter_a", "TEXT DEFAULT ''")
        await add_column_if_missing(db, "deleted_jobs", "label_filter_b", "TEXT DEFAULT ''")
        await add_column_if_missing(db, "deleted_projects", "experiment_edges_json", "TEXT DEFAULT '[]'")
        await add_column_if_missing(db, "deleted_projects", "experiment_node_layout_json", "TEXT DEFAULT '[]'")
        await db.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES(1)")

        await db.commit()


async def row_to_dict(row):
    if row is None:
        return None
    return dict(row)
