#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-cambricon}"
APP_GROUP="${APP_GROUP:-$APP_USER}"
BASE_DIR="${ANALYZE_TRACE_DATA_DIR:-/data/analyze_trace}"

install -d -o "$APP_USER" -g "$APP_GROUP" -m 750 "$BASE_DIR"
install -d -o "$APP_USER" -g "$APP_GROUP" -m 750 "$BASE_DIR/storage"
install -d -o "$APP_USER" -g "$APP_GROUP" -m 750 "$BASE_DIR/logs"
install -d -o "$APP_USER" -g "$APP_GROUP" -m 750 "$BASE_DIR/backups"
install -o "$APP_USER" -g "$APP_GROUP" -m 640 /dev/null "$BASE_DIR/logs/app.jsonl"

echo "Prepared $BASE_DIR for $APP_USER:$APP_GROUP"
echo "Suggested env:"
echo "TRACE_STORAGE_DIR=$BASE_DIR/storage"
echo "TRACE_LOG_FILE=$BASE_DIR/logs/app.jsonl"
echo "TRACE_BACKUP_DIR=$BASE_DIR/backups"
