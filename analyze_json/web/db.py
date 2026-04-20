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
            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id               TEXT PRIMARY KEY,
                project_id       TEXT REFERENCES projects(id) ON DELETE SET NULL,
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
        """)
        # Add gzip_path columns if they don't exist (for existing databases)
        for col in ["file_a_gzip_path", "file_b_gzip_path"]:
            try:
                await db.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
            except Exception:
                pass  # Column already exists
        await db.commit()


async def row_to_dict(row):
    if row is None:
        return None
    return dict(row)
