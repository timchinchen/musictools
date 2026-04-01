"""SQLite-backed metadata for uploaded MP3s (history list)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "upload_history.sqlite"


def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                file_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_uploads_created_at ON uploads(created_at)")


def record_upload(file_id: str, filename: str, size_bytes: int) -> None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            """
            INSERT INTO uploads (file_id, filename, size_bytes, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (file_id, filename, int(size_bytes), now),
        )


def list_uploads(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            """
            SELECT file_id, filename, size_bytes, created_at
            FROM uploads
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        ).fetchall()
    return [dict(r) for r in rows]
