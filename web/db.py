import aiosqlite
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "storage", "jobs.db")


async def get_db():
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
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
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_token  TEXT PRIMARY KEY,
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
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id               TEXT PRIMARY KEY,
                project_id       TEXT REFERENCES projects(id) ON DELETE CASCADE,
                user_token       TEXT REFERENCES users(user_token) ON DELETE CASCADE,
                created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
                label            TEXT DEFAULT '',
                is_pinned        INTEGER DEFAULT 0,
                mode             TEXT CHECK(mode IN ('single','compare')) NOT NULL,

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

            CREATE TABLE IF NOT EXISTS deleted_projects (
                id           TEXT PRIMARY KEY,
                user_token   TEXT,
                folder_id    TEXT,
                name         TEXT NOT NULL,
                description  TEXT DEFAULT '',
                password_hash TEXT DEFAULT NULL,
                is_public    INTEGER DEFAULT 0,
                created_at   DATETIME,
                deleted_at   DATETIME DEFAULT CURRENT_TIMESTAMP
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

            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                applied_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Column migrations for databases created by earlier versions.
        await add_column_if_missing(db, "users", "created_at", "DATETIME")
        await add_column_if_missing(db, "projects", "user_token", "TEXT")
        await add_column_if_missing(db, "projects", "password_hash", "TEXT DEFAULT NULL")
        await add_column_if_missing(db, "projects", "is_public", "INTEGER DEFAULT 0")
        await add_column_if_missing(db, "projects", "created_at", "DATETIME")
        await add_column_if_missing(db, "projects", "folder_id", "TEXT")
        await add_column_if_missing(db, "jobs", "user_token", "TEXT")
        await add_column_if_missing(db, "jobs", "owned_bytes", "INTEGER")
        await add_column_if_missing(db, "jobs", "result_bytes", "INTEGER")
        await add_column_if_missing(db, "jobs", "original_trace_bytes", "INTEGER")
        await add_column_if_missing(db, "jobs", "is_pinned", "INTEGER DEFAULT 0")
        await add_column_if_missing(db, "folders", "password_hash", "TEXT DEFAULT NULL")

        await db.executescript("""
            CREATE TABLE IF NOT EXISTS deleted_jobs (
                id               TEXT PRIMARY KEY,
                project_id       TEXT,
                user_token       TEXT,
                created_at       DATETIME,
                label            TEXT DEFAULT '',
                is_pinned        INTEGER DEFAULT 0,
                mode             TEXT,

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
            CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_token);
            CREATE INDEX IF NOT EXISTS idx_jobs_project_created ON jobs(project_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_mode_status_created ON jobs(mode, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_pinned_created ON jobs(is_pinned, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_source_a ON jobs(source_job_a);
            CREATE INDEX IF NOT EXISTS idx_jobs_source_b ON jobs(source_job_b);
            CREATE INDEX IF NOT EXISTS idx_folders_user ON folders(user_token);
            CREATE INDEX IF NOT EXISTS idx_projects_folder ON projects(folder_id);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
        """)
        await add_column_if_missing(db, "deleted_jobs", "is_pinned", "INTEGER DEFAULT 0")
        await db.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES(1)")

        await db.commit()


async def row_to_dict(row):
    if row is None:
        return None
    return dict(row)
