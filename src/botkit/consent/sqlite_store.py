# Consent — ConsentStore поверх SQLite. Журнал только дописывается: статус
# определяется последней записью, а не перезаписываемым флагом — по журналу
# видно, какой текст и когда видел студент.

from __future__ import annotations

import sqlite3
from asyncio import to_thread
from datetime import UTC, datetime

from botkit.consent.base import Decision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consent_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    decision TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consent_events_actor
    ON consent_events(actor_id, policy_version, event_id);
"""

_DECISIONS = ("granted", "declined", "revoked")


class SQLiteConsentStore:
    """Один файл = один процесс-writer, как остальные SQLite-сторы фреймворка.
    Можно указывать тот же файл БД, что у других сторов бота: таблица своя."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def record(
        self, *, actor_id: str, policy_version: str, text_sha256: str, decision: Decision
    ) -> None:
        if decision not in _DECISIONS:
            raise ValueError(f"unknown consent decision: {decision!r}")
        await to_thread(self._record_sync, actor_id, policy_version, text_sha256, decision)

    def _record_sync(self, actor_id: str, policy_version: str, text_sha256: str, decision: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO consent_events "
                "(actor_id, policy_version, text_sha256, decision, created_at) VALUES (?, ?, ?, ?, ?)",
                (actor_id, policy_version, text_sha256, decision, datetime.now(UTC).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    async def current(self, actor_id: str, policy_version: str) -> Decision | None:
        return await to_thread(self._current_sync, actor_id, policy_version)

    def _current_sync(self, actor_id: str, policy_version: str) -> Decision | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT decision FROM consent_events WHERE actor_id = ? AND policy_version = ? "
                "ORDER BY event_id DESC LIMIT 1",
                (actor_id, policy_version),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else row["decision"]

    async def list_events(self, actor_id: str) -> list[dict]:
        return await to_thread(self._list_events_sync, actor_id)

    def _list_events_sync(self, actor_id: str) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM consent_events WHERE actor_id = ? ORDER BY event_id", (actor_id,)
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]
