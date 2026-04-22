from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class ConversationMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).expanduser())
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                ON messages(session_id, created_at DESC, id DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, key)
                )
                """
            )

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        payload = json.dumps(metadata or {}, ensure_ascii=True)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO messages(session_id, role, content, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, role, content, payload, now),
                )

    def get_recent_messages(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (session_id, int(limit)),
            ).fetchall()

        result: list[dict[str, Any]] = []
        for row in reversed(rows):
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            result.append(
                {
                    "role": str(row["role"]),
                    "content": str(row["content"]),
                    "metadata": metadata,
                    "created_at": float(row["created_at"]),
                }
            )
        return result

    def set_fact(self, session_id: str, key: str, value: str) -> None:
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO facts(session_id, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id, key)
                    DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (session_id, key, value, now),
                )

    def get_facts(self, session_id: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, value
                FROM facts
                WHERE session_id = ?
                ORDER BY key ASC
                """,
                (session_id,),
            ).fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def summarize_context(self, session_id: str, recent_limit: int = 12) -> dict[str, Any]:
        return {
            "facts": self.get_facts(session_id),
            "recent_messages": self.get_recent_messages(session_id, limit=recent_limit),
        }
