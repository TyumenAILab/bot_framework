import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from botkit.platform import LLMConfig, LLMGateway, SQLiteUsageTracker
from botkit.runtime import ChatBot
from botkit.transport import IncomingMessage, MemoryAdapter


async def test_same_handler_two_platforms_end_to_end(tmp_path):
    tracker = SQLiteUsageTracker(tmp_path / "usage.db")
    one, two = MemoryAdapter(tracker=tracker, bot_id="one"), MemoryAdapter(tracker=tracker, bot_id="two")
    one.platform, two.platform = "telegram", "vk"
    body = {
        "choices": [{"message": {"content": '{"response":"Готово"}'}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        llm = LLMGateway(LLMConfig("vllm", "model", "http://test/v1"), client=client, tracker=tracker)
        ChatBot(llm, [one, two])
        await asyncio.gather(
            one.dispatch(IncomingMessage("telegram", "same-id", "c", "text")),
            two.dispatch(IncomingMessage("vk", "same-id", "c", "text")),
        )
        assert one.sent[0]["text"] == two.sent[0]["text"] == "Готово"
        assert (await tracker.stats(platform="vk")).calls == 1
        assert (await tracker.stats(platform="telegram")).calls == 1
        assert (await tracker.stats(bot_id="one", skill_context="CHAT")).output_tokens == 2
        await one.aclose()
        await two.aclose()


async def test_cli_wires_asr_from_env_factory_and_closes(monkeypatch, tmp_path):
    import botkit.__main__ as cli
    from botkit.llm import LLMResponse

    llm = SimpleNamespace(aclose=AsyncMock())
    asr = SimpleNamespace(
        aclose=AsyncMock(),
        atranscribe=AsyncMock(return_value=LLMResponse("speech", "speech", "raw_text_fallback")),
    )
    adapter = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(cli, "load_llm", lambda **kwargs: llm)
    monkeypatch.setattr(cli, "load_transcriber", lambda **kwargs: asr)
    monkeypatch.setattr(cli, "make_adapter", lambda *args, **kwargs: adapter)

    class RunBot:
        def __init__(self, llm_arg, adapters, *, extraction):
            assert llm_arg is llm and adapters == [adapter]
            self.extraction = extraction

        async def run(self):
            result = await self.extraction.extract_bytes(b"audio", "audio/mpeg")
            assert result.text == "speech"

    monkeypatch.setattr(cli, "ChatBot", RunBot)
    await cli.execute(SimpleNamespace(command="run", platform=["telegram"], db=str(tmp_path / "cli.db")))
    asr.atranscribe.assert_awaited_once()
    asr.aclose.assert_awaited_once()
    llm.aclose.assert_awaited_once()
    adapter.aclose.assert_awaited_once()


async def test_runtime_passes_transcript_not_audio_to_chat_model():
    from botkit.extraction import ContentExtractionService
    from botkit.llm import LLMResponse, PneumaConfig, PneumaTranscriber
    from botkit.transport import Attachment

    def server(request):
        return httpx.Response(
            200,
            json=(
                {"job_id": "e" * 32, "status": "completed"}
                if request.method == "POST"
                else {"status": "completed", "ready": True, "transcript": "Речь с голосового"}
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        asr = PneumaTranscriber(PneumaConfig("https://asr.test"), client=client)
        llm = SimpleNamespace(
            achat=AsyncMock(return_value=LLMResponse("Ответ", "Ответ", "raw_text_fallback"))
        )
        adapter = MemoryAdapter()
        ChatBot(llm, [adapter], extraction=ContentExtractionService(transcriber=asr))
        await adapter.dispatch(
            IncomingMessage(
                "memory",
                "u",
                "c",
                "",
                attachments=[
                    Attachment("audio", "audio/ogg", AsyncMock(return_value=b"private-audio-bytes"))
                ],
            )
        )
        payload = json.dumps(llm.achat.await_args.args[0], ensure_ascii=False)
        assert "Речь с голосового" in payload and "private-audio-bytes" not in payload
        assert "audio_transcript_unverified" in payload
        assert adapter.sent[-1]["text"] == "Ответ"
        await adapter.aclose()
