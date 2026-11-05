"""Tests of the operator harness; every external HTTP boundary is mocked."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from botkit.llm import (
    FallbackLLM,
    LLMConfig,
    LLMGateway,
    LLMResponse,
    PneumaConfig,
    PneumaTranscriber,
    TranscriptionPipeline,
    TranscriptResult,
)
from botkit.transport import Attachment, CallbackQuery, IncomingMessage, MemoryAdapter
from botkit.usage import SQLiteUsageTracker, usage_context

spec = importlib.util.spec_from_file_location(
    "acceptance_harness", Path(__file__).parents[1] / "scripts" / "acceptance.py"
)
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)


@pytest.fixture
def report(tmp_path):
    return harness.Report(tmp_path / "report", "test")


@pytest.fixture
def fixtures(report):
    return harness.make_fixtures(report.directory / "fixtures")


@pytest.fixture
def tracker(report):
    return SQLiteUsageTracker(report.directory / "usage.sqlite3")


def test_report_missing_and_failures_stay_visible(report):
    report.expect("one", "two")
    assert report.data["result"] == "incomplete"
    report.record("one")
    assert report.data["missing"] == ["two"]
    report.record("two", "failed")
    report.record("two")
    assert report.exit_code == 1
    saved = json.loads((report.directory / "report.json").read_text(encoding="utf-8"))
    assert saved["publication_allowed"] is False
    assert saved["result"] == "failed"


async def test_exception_and_http_url_never_exposed(report, capsys):
    async def failure():
        raise RuntimeError("secret-api-key https://signed-url.test/?token=secret document-content")

    assert not await report.check("failure", failure)
    output = capsys.readouterr().out + (report.directory / "report.json").read_text(encoding="utf-8")
    assert "secret" not in output and "document-content" not in output
    assert report.data["cases"][-1]["error_type"] == "RuntimeError"


async def test_checks_have_deadlines_and_do_not_swallow_cancel(report):
    async def wait():
        await asyncio.Event().wait()

    assert not await report.check("deadline", wait, timeout=0.01)
    assert report.data["cases"][-1]["error_type"] == "TimeoutError"
    task = asyncio.create_task(report.check("cancel", wait))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_local_fixtures_and_usage_are_real(report, fixtures, tracker):
    await harness.local_checks(report, fixtures, tracker)
    result = await harness.usage_check(tracker)
    assert result["events"] == 3
    assert result["llm_events"] == 0
    assert report.exit_code == 0
    assert (report.directory / "fixtures" / "rendered-scan.png").is_file()


async def test_known_regressions_are_reported_not_hidden(report):
    await harness.regression_checks(report)
    assert {c["name"] for c in report.data["cases"]} == {
        "regression:telegram_native_handlers",
        "regression:telegram_media",
        "regression:vk_native_handlers",
        "regression:vk_callback_ids",
    }
    reasons = {c.get("reason") for c in report.data["cases"] if c["status"] == "failed"}
    # Fixing a product defect must turn the harness green, not break this harness test.
    assert reasons <= {
        "native_handlers_intercepted",
        "animation_or_video_note_missing",
        "native_handlers_blocked",
        "conversation_id_sent_as_global_message_id",
    }
    assert report.exit_code == int(bool(reasons))


def completion(content):
    return httpx.Response(
        200,
        json={
            "id": "test",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


async def test_model_matrix_forced_fallback_and_tracking(report, fixtures, tracker):
    calls = []

    def endpoint(request):
        body = json.loads(request.content)
        calls.append(body)
        if isinstance(body["messages"][-1]["content"], list):
            return completion(json.dumps({"text": "BOTKIT-7319 TOTAL 42", "tables": [], "warnings": []}))
        return completion('{"response":"BOTKIT_OK"}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        providers = [
            LLMGateway(
                LLMConfig("vllm", name, "https://mock.test/v1", "fake", max_retries=0),
                client=client,
                tracker=tracker,
            )
            for name in ("primary", "backup")
        ]
        chain = FallbackLLM(providers[0], providers[1:], tracker=tracker)
        try:
            await harness.model_checks(report, chain, fixtures, tracker)
        finally:
            await chain.aclose()
    assert report.exit_code == 0
    assert len(calls) == 9  # 3 per provider + chain + two actual backup calls
    assert [c["model"] for c in calls[-2:]] == ["backup", "backup"]
    stats = await harness.usage_check(tracker, require_llm=True)
    assert stats["llm_events"] == 11  # plus two locally injected 503 primary attempts
    rows = await tracker.events(kind="llm")
    assert sum(r["status"] == "error" for r in rows) == 2
    assert all(not r["input_text"] and not r["output_text"] for r in rows)


async def test_broken_primary_not_hidden_by_successful_backup(report, fixtures, tracker):
    def endpoint(request):
        body = json.loads(request.content)
        if body["model"] == "primary":
            return httpx.Response(503)
        if isinstance(body["messages"][-1]["content"], list):
            return completion('{"text":"7319 42","tables":[],"warnings":[]}')
        return completion('{"response":"BOTKIT_OK"}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        providers = [
            LLMGateway(
                LLMConfig("vllm", name, "https://mock.test/v1", "fake", max_retries=0),
                client=client,
                tracker=tracker,
            )
            for name in ("primary", "backup")
        ]
        chain = FallbackLLM(providers[0], providers[1:], tracker=tracker)
        try:
            await harness.model_checks(report, chain, fixtures, tracker)
        finally:
            await chain.aclose()
    assert report.exit_code == 1
    assert len([c for c in report.data["cases"] if c["status"] == "failed"]) == 3
    assert next(c for c in report.data["cases"] if c["name"] == "configured_chain:text")["status"] == "passed"


@pytest.mark.parametrize("mode", ["models", "telegram", "vk"])
async def test_live_requires_explicit_flag_before_loading_env(mode):
    with pytest.raises(SystemExit) as exc:
        await harness.main(["--mode", mode, "--env", "does-not-exist"])
    assert exc.value.code == 2


async def test_messenger_accepts_any_user_and_chat_with_shared_budget(report, fixtures, tracker):
    llm = SimpleNamespace(ainvoke=AsyncMock(), ainvoke_with_image_b64=AsyncMock())
    adapter = MemoryAdapter(tracker=tracker, bot_id="acceptance")
    session = harness.MessengerSession(adapter, llm, tracker, report, fixtures, max_calls=1)
    try:
        await asyncio.gather(
            adapter.dispatch(IncomingMessage("memory", "20", "10", "/start", message_id="1")),
            adapter.dispatch(IncomingMessage("memory", "40", "30", "hello", message_id="1")),
        )
        assert [m["chat_id"] for m in adapter.sent] == ["10", "30"]
        llm.ainvoke.assert_not_awaited()
        message = IncomingMessage("memory", "20", "10", "hello", message_id="2")
        await adapter.dispatch(message)
        await adapter.dispatch(message)
        assert len(adapter.sent) == 3
        assert any(c["name"] == "messenger:duplicate_message" for c in report.data["cases"])
        await session.llm.ainvoke("x")
        with pytest.raises(harness.CheckFailure, match="session_llm_budget_exhausted"):
            await session.llm.ainvoke_with_image_b64("x", "image")
        assert llm.ainvoke.await_count == 1
        llm.ainvoke_with_image_b64.assert_not_awaited()
    finally:
        await adapter.aclose()


async def test_delete_targets_only_freshly_sent_message(report, fixtures, tracker):
    adapter = MemoryAdapter()
    adapter.edit = AsyncMock()
    adapter.delete = AsyncMock(return_value=True)
    adapter.send_attachment = AsyncMock(return_value="uploaded")
    session = harness.MessengerSession(adapter, object(), tracker, report, fixtures)
    try:
        await session.smoke("10")
        target = adapter.edit.await_args.args[1]
        assert adapter.sent[int(target) - 1]["text"].startswith("botkit: это сообщение")
        assert adapter.delete.await_args.args == ("10", target)
        assert not session.done.is_set()
        assert "messenger:callback" in report.data["missing"]
    finally:
        await adapter.aclose()


async def test_callbacks_from_any_user_are_scoped_to_their_own_chat(report, fixtures, tracker):
    adapter = MemoryAdapter()
    adapter.answer_callback = AsyncMock()
    session = harness.MessengerSession(adapter, object(), tracker, report, fixtures)
    session.callback_messages = {"10": ("nonce-a", "1"), "30": ("nonce-b", "2")}
    try:
        await session.callback(CallbackQuery("memory", "bad", "99", "30", "nonce-a"))
        assert adapter.sent == []
        adapter.answer_callback.assert_not_awaited()
        await asyncio.gather(
            session.callback(CallbackQuery("memory", "a", "999", "10", "nonce-a")),
            session.callback(CallbackQuery("memory", "b", "888", "30", "nonce-b")),
        )
        assert [m["chat_id"] for m in adapter.sent] == ["10", "30"]
        assert adapter.answer_callback.await_count == 2
    finally:
        await adapter.aclose()


async def test_no_false_usage_failure_before_successful_model_call(report, fixtures, tracker):
    await harness.local_checks(report, fixtures, tracker)
    report.record("startup", "failed", reason="messenger_token_missing")
    assert not harness.successful_llm_check(report)
    await harness.usage_check(tracker, require_llm=harness.successful_llm_check(report))
    report.record("messenger:llm")
    assert harness.successful_llm_check(report)
    with pytest.raises(harness.CheckFailure, match="no_successful_llm_usage_event"):
        await harness.usage_check(tracker, require_llm=harness.successful_llm_check(report))


@pytest.mark.parametrize("platform", ["telegram", "vk"])
async def test_polling_needs_no_chat_ids_and_sends_no_unsolicited_invitation(
    monkeypatch, report, fixtures, tracker, platform
):
    import botkit.runtime

    adapter = MemoryAdapter()
    adapter.call_api = AsyncMock(return_value=SimpleNamespace(url=""))

    async def polling():
        assert adapter.sent == []
        await adapter.dispatch(IncomingMessage("memory", "777", "555", "/start", message_id="1"))
        await adapter.dispatch(IncomingMessage("memory", "888", "666", "/finish", message_id="1"))

    adapter.run = polling
    monkeypatch.setattr(botkit.runtime, "make_adapter", lambda *args, **kwargs: adapter)
    monkeypatch.setenv(platform.upper() + "_BOT_TOKEN", "dummy")
    # Old access settings no longer constrain callers or prevent startup.
    monkeypatch.setenv("ACCEPTANCE_" + platform.upper() + "_CHAT_ID", "")
    monkeypatch.setenv("ACCEPTANCE_" + platform.upper() + "_USER_IDS", "")
    args = SimpleNamespace(mode=platform, seconds=1, max_calls=2)
    await harness.messenger_checks(args, report, object(), tracker, fixtures)
    assert [m["chat_id"] for m in adapter.sent] == ["555", "666"]
    assert report.data["messenger_access"] == "all_users_and_chats"
    assert report.data["max_logical_model_calls"] == 2
    assert not any(c["status"] == "failed" for c in report.data["cases"])


async def test_existing_webhook_does_not_start_polling_or_delete_webhook(
    monkeypatch, report, fixtures, tracker
):
    import botkit.runtime

    adapter = SimpleNamespace(
        call_api=AsyncMock(return_value=SimpleNamespace(url="https://existing.test/webhook")),
        aclose=AsyncMock(),
        run=AsyncMock(),
        send=AsyncMock(),
    )
    monkeypatch.setattr(botkit.runtime, "make_adapter", lambda *args, **kwargs: adapter)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("ACCEPTANCE_TELEGRAM_CHAT_ID", "10")
    monkeypatch.setenv("ACCEPTANCE_TELEGRAM_USER_IDS", "20")
    args = SimpleNamespace(mode="telegram", seconds=1, max_calls=1)
    with pytest.raises(harness.CheckFailure, match="existing_webhook"):
        await harness.messenger_checks(args, report, object(), tracker, fixtures)
    adapter.call_api.assert_awaited_once_with("get_webhook_info")
    adapter.send.assert_not_awaited()
    adapter.run.assert_not_awaited()
    adapter.aclose.assert_awaited_once()


async def test_messenger_pneuma_asr_cases_and_shared_budget(report, fixtures, tracker):
    def server(request):
        if request.method == "POST":
            return httpx.Response(200, json={"job_id": "d" * 32, "status": "completed"})
        return httpx.Response(
            200, json={"status": "completed", "ready": True, "transcript": "Private speech"}
        )

    adapter = MemoryAdapter(tracker=tracker, bot_id="acceptance")
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        asr = PneumaTranscriber(PneumaConfig("https://asr.test"), client=client, tracker=tracker)
        session = harness.MessengerSession(
            adapter, object(), tracker, report, fixtures, max_calls=2, transcriber=asr
        )
        for i, (mime, name) in enumerate([("audio/ogg", None), ("application/octet-stream", "voice.mp3")]):
            with usage_context(bot_id="acceptance", skill_context="messenger_session"):
                await adapter.dispatch(
                    IncomingMessage(
                        "memory",
                        "u",
                        "c",
                        "",
                        message_id=str(i),
                        attachments=[
                            Attachment("audio", mime, AsyncMock(return_value=b"audio"), filename=name)
                        ],
                    )
                )
        assert not {"messenger:voice_in", "messenger:mp3_in"}.intersection(report.data["missing"])
        assert not any(c["status"] == "failed" for c in report.data["cases"])
        assert harness.successful_llm_check(report)
        assert (await harness.usage_check(tracker, require_llm=True))["llm_events"] == 2
        with pytest.raises(harness.CheckFailure, match="budget"):
            await session.llm.ainvoke("prompt")
        assert "Private speech" not in json.dumps(report.data)
    await adapter.aclose()


async def test_pipeline_reserves_asr_and_llm_budget_before_upload():
    asr, processor = AsyncMock(), AsyncMock()
    asr.atranscribe.return_value = LLMResponse("речь", "речь", "raw_text_fallback")
    processor.aprocess.return_value = TranscriptResult("Речь.", True)
    pipeline = TranscriptionPipeline(asr, processor)
    budget = harness.BudgetLLM(object(), 1, pipeline)
    with pytest.raises(harness.CheckFailure, match="budget"):
        await budget.atranscribe(b"voice", mime_type="audio/ogg", filename="a.ogg")
    asr.atranscribe.assert_not_awaited()
    assert budget.used == 0
    budget.limit = 2
    assert (await budget.atranscribe(b"voice", mime_type="audio/ogg", filename="a.ogg")).response == "Речь."
    assert budget.used == 2
    processor.aprocess.assert_awaited_once()
