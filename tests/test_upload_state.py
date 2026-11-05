import asyncio
import json
import sqlite3
import time

import httpx
import pytest
from test_pneuma_upload import API, config, transcribe

from botkit.llm import ChunkedPneumaTranscriber, MemoryUploadStore, SQLiteUploadStore
from botkit.llm.upload_state import TTL
from botkit.usage import usage_context


async def test_sqlite_resume_after_restart_without_audio_on_disk(tmp_path):
    api = API()
    db = tmp_path / "nested" / "uploads.db"
    store = SQLiteUploadStore(db)

    def interrupted(request):
        if request.method == "PATCH" and request.headers["Upload-Offset"] == "4":
            raise asyncio.CancelledError()
        return api(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(interrupted)) as client:
        with pytest.raises(asyncio.CancelledError):
            await transcribe(ChunkedPneumaTranscriber(config(), client=client, store=store))
    assert api.uploaded == b"0123"
    with sqlite3.connect(db) as connection:
        rows = connection.execute("SELECT key, value FROM pneuma_upload_state").fetchall()
    assert len(rows) == 1 and len(rows[0][0]) == 64
    assert json.loads(rows[0][1]) == {"upload_id": "upload-1", "received_bytes": 4}
    assert b"0123456789" not in db.read_bytes() and b"Student Name" not in db.read_bytes()
    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as client:
        resumed = ChunkedPneumaTranscriber(config(), client=client, store=SQLiteUploadStore(db))
        assert (await transcribe(resumed)).response == api.result["transcript"]
        await transcribe(resumed)
    assert api.uploaded == b"0123456789"
    assert len([call for call in api.calls if call[:2] == ("POST", "/uploads")]) == 1


@pytest.mark.parametrize("persistent", [False, True])
async def test_store_rejects_text_credentials_and_bad_identifiers(tmp_path, persistent):
    store = SQLiteUploadStore(tmp_path / "state.db") if persistent else MemoryUploadStore()
    for value in (
        {"audio": "binary"},
        {"api_key": "secret"},
        {"job_id": "../secret"},
        {"received_bytes": True},
    ):
        with pytest.raises(ValueError):
            await store.put("a" * 64, value)
    with pytest.raises(ValueError):
        await store.put("user-name-not-hash", {"job_id": "valid"})
    await store.put("a" * 64, {"job_id": "valid"})
    result = await store.get("a" * 64)
    result["job_id"] = "modified"
    assert (await store.get("a" * 64))["job_id"] == "valid"


async def test_metadata_ttl_and_memory_bound(tmp_path, monkeypatch):
    memory, disk = MemoryUploadStore(), SQLiteUploadStore(tmp_path / "state.db")
    await disk.put("b" * 64, {"job_id": "expired"})
    for index in range(1001):
        await memory.put(f"{index:064x}", {"job_id": str(index)})
    assert len(memory.rows) == 1000
    assert await memory.get("0" * 64) == {}
    future = time.time() + TTL + 1
    monkeypatch.setattr("botkit.llm.upload_state.time.time", lambda: future)
    await memory.prune()
    await disk.prune()
    assert not memory.rows and await disk.get("b" * 64) == {}
    with sqlite3.connect(disk.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pneuma_upload_state").fetchone()[0] == 0


async def test_context_identity_required_and_different_requests_get_different_keys():
    api = API()
    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as client:
        provider = ChunkedPneumaTranscriber(config(user_id=""), client=client)
        with pytest.raises(ValueError, match="pseudonymous"):
            await transcribe(provider)
        with usage_context(user_id="stu_private", request_id="first-message"):
            await transcribe(provider)
        assert len(provider.store.rows) == 1
        api.uploaded, api.job = b"", None
        with usage_context(user_id="stu_private", request_id="another-message"):
            await transcribe(provider)
        assert len(provider.store.rows) == 2


async def test_concurrent_identical_uploads_share_one_job():
    api = API()
    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as client:
        provider = ChunkedPneumaTranscriber(config(), client=client)
        await asyncio.gather(transcribe(provider), transcribe(provider))
    assert len([call for call in api.calls if call[:2] == ("POST", "/uploads")]) == 1
