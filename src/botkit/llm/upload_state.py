"""Resumable upload metadata stores. Audio and transcripts must never be stored here."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from typing import Protocol

TTL = 24 * 60 * 60


class UploadStore(Protocol):
    """Application stores may implement this interface without any botkit database dependency."""

    async def get(self, key: str) -> dict: ...
    async def put(self, key: str, value: dict) -> None: ...
    async def prune(self) -> None: ...


def _validate(key: str, value: dict) -> None:
    if not re.fullmatch(r"[a-f0-9]{64}", key):
        raise ValueError("Upload state key must be a SHA256 fingerprint")
    if set(value) - {"upload_id", "job_id", "received_bytes"}:
        raise ValueError("Only upload metadata may be stored")
    for name, item in value.items():
        if name == "received_bytes":
            if type(item) is not int or item < 0:
                raise ValueError("Invalid upload offset")
        elif not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", item):
            raise ValueError("Invalid upload identifier")


class MemoryUploadStore:
    """Bounded per-process metadata; use SQLiteUploadStore to survive a restart."""

    def __init__(self):
        self.rows = OrderedDict()

    async def get(self, key: str) -> dict:
        await self.prune()
        return dict(self.rows.get(key, (0, {}))[1])

    async def put(self, key: str, value: dict) -> None:
        _validate(key, value)
        self.rows[key] = (time.time(), dict(value))
        self.rows.move_to_end(key)
        while len(self.rows) > 1000:
            self.rows.popitem(last=False)

    async def prune(self) -> None:
        cutoff = time.time() - TTL
        self.rows = OrderedDict((k, row) for k, row in self.rows.items() if row[0] > cutoff)


class SQLiteUploadStore:
    """Durable metadata with a 24-hour TTL, using only the Python standard library.

    Connections are short-lived and owned here. The client serializes identical
    uploads inside one transcriber instance; cross-process scheduling is the
    application's responsibility (this store is not a distributed upload lock).
    """

    def __init__(self, path: str | Path):
        if str(path) == ":memory:":
            raise ValueError("Use MemoryUploadStore for in-memory metadata")
        self.path = Path(path)

    def _run(self, operation: str, key: str = "", value: dict | None = None) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS pneuma_upload_state (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL
            )""")
            db.execute("DELETE FROM pneuma_upload_state WHERE updated_at <= ?", (time.time() - TTL,))
            if operation == "put":
                db.execute(
                    "INSERT INTO pneuma_upload_state VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (key, json.dumps(value), time.time()),
                )
            elif operation == "get":
                row = db.execute("SELECT value FROM pneuma_upload_state WHERE key=?", (key,)).fetchone()
                if row:
                    result = json.loads(row[0])
                    _validate(key, result)
                    return result
        return {}

    async def get(self, key: str) -> dict:
        return await asyncio.to_thread(self._run, "get", key)

    async def put(self, key: str, value: dict) -> None:
        _validate(key, value)
        await asyncio.to_thread(self._run, "put", key, dict(value))

    async def prune(self) -> None:
        await asyncio.to_thread(self._run, "prune")
