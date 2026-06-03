#!/usr/bin/env python3
"""Create an operational backup for the analyze_trace web app.

The archive contains a consistent SQLite snapshot plus the storage tree.  Use
this from cron/systemd timer or the company's backup runner.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone


ROOT = os.path.dirname(__file__)
DEFAULT_STORAGE_DIR = os.path.join(ROOT, "storage")
DEFAULT_BACKUP_DIR = os.environ.get("TRACE_BACKUP_DIR", os.path.join(ROOT, "backups"))
DB_FILENAME = "jobs.db"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_sqlite(db_path, snapshot_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    target = sqlite3.connect(snapshot_path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def should_skip_file(path, db_path, backup_dir):
    real = os.path.realpath(path)
    db_real = os.path.realpath(db_path)
    backup_real = os.path.realpath(backup_dir)
    if real in {db_real, db_real + "-wal", db_real + "-shm"}:
        return True
    return real == backup_real or real.startswith(backup_real + os.sep)


def create_backup(storage_dir, backup_dir, retention_days):
    storage_dir = os.path.abspath(storage_dir)
    backup_dir = os.path.abspath(backup_dir)
    os.makedirs(backup_dir, exist_ok=True)

    created = datetime.now(timezone.utc)
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    archive_path = os.path.join(backup_dir, f"analyze_trace_backup_{stamp}.tar.gz")
    db_path = os.path.join(storage_dir, DB_FILENAME)

    with tempfile.TemporaryDirectory(prefix="analyze-trace-backup-") as tmpdir:
        db_snapshot = os.path.join(tmpdir, DB_FILENAME)
        backup_sqlite(db_path, db_snapshot)

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(db_snapshot, arcname=os.path.join("storage", DB_FILENAME))
            for root, _, files in os.walk(storage_dir):
                for filename in files:
                    path = os.path.join(root, filename)
                    if should_skip_file(path, db_path, backup_dir):
                        continue
                    rel = os.path.relpath(path, storage_dir)
                    tar.add(path, arcname=os.path.join("storage", rel))

    manifest = {
        "created_at": created.isoformat(),
        "created_at_epoch": int(created.timestamp()),
        "archive": archive_path,
        "filename": os.path.basename(archive_path),
        "size_bytes": os.path.getsize(archive_path),
        "sha256": sha256_file(archive_path),
        "storage_dir": storage_dir,
        "retention_days": retention_days,
    }
    latest_path = os.path.join(backup_dir, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    prune_old_backups(backup_dir, retention_days, keep={os.path.realpath(archive_path)})
    return manifest


def prune_old_backups(backup_dir, retention_days, keep=None):
    if retention_days <= 0:
        return
    keep = keep or set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for filename in os.listdir(backup_dir):
        if not filename.startswith("analyze_trace_backup_") or not filename.endswith(".tar.gz"):
            continue
        path = os.path.join(backup_dir, filename)
        if os.path.realpath(path) in keep:
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
        if mtime < cutoff:
            os.remove(path)


def main():
    parser = argparse.ArgumentParser(description="Back up analyze_trace web storage.")
    parser.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--backup-dir", default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--retention-days", type=int, default=14)
    args = parser.parse_args()

    manifest = create_backup(args.storage_dir, args.backup_dir, args.retention_days)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
