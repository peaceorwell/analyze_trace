import aiosqlite
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "storage", "jobs.db")


async def get_db():
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


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

                kernel_types     TEXT DEFAULT 'gemm,embedding,pool',
                save_triton_csv  INTEGER DEFAULT 0,
                save_triton_code INTEGER DEFAULT 0,

                status           TEXT CHECK(status IN ('pending','running','done','error'))
                                     DEFAULT 'pending',
                console_out      TEXT DEFAULT '',
                error_msg        TEXT DEFAULT '',
                result_dir       TEXT DEFAULT ''
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
        """)

        # Migration for existing databases
        try:
            await db.execute("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE projects ADD COLUMN user_token TEXT")
            await db.execute("ALTER TABLE projects ADD COLUMN password_hash TEXT DEFAULT NULL")
            await db.execute("ALTER TABLE projects ADD COLUMN is_public INTEGER DEFAULT 0")
            await db.execute("ALTER TABLE projects ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE jobs ADD COLUMN user_token TEXT")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE projects ADD COLUMN folder_id TEXT")
        except Exception:
            pass

        # Migration: add deleted_projects table for soft delete recovery
        try:
            await db.execute("SELECT id FROM deleted_projects LIMIT 1")
        except Exception:
            await db.execute("""
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
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_deleted_projects_user ON deleted_projects(user_token)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_deleted_projects_deleted_at ON deleted_projects(deleted_at)")

        # Create folders table if not exists (for existing databases)
        try:
            await db.execute("SELECT id FROM folders LIMIT 1")
        except Exception:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS folders (
                    id           TEXT PRIMARY KEY,
                    user_token   TEXT REFERENCES users(user_token) ON DELETE CASCADE,
                    name         TEXT NOT NULL,
                    password_hash TEXT DEFAULT NULL,
                    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_folders_user ON folders(user_token)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_projects_folder ON projects(folder_id)")

        # Migration: add password_hash to folders for existing tables
        try:
            await db.execute("ALTER TABLE folders ADD COLUMN password_hash TEXT DEFAULT NULL")
        except Exception:
            pass

        # Create indexes for user_token lookups
        try:
            await db.execute("CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_token)")
        except Exception:
            pass
        try:
            await db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_token)")
        except Exception:
            pass
        try:
            await db.execute("CREATE INDEX IF NOT EXISTS idx_folders_user ON folders(user_token)")
        except Exception:
            pass
        try:
            await db.execute("CREATE INDEX IF NOT EXISTS idx_projects_folder ON projects(folder_id)")
        except Exception:
            pass

        await db.commit()


async def row_to_dict(row):
    if row is None:
        return None
    return dict(row)