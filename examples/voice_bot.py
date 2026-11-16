"""Голосовой бот: принимает голосовое → текст (ASR) → ответ LLM → текст + голосовое (TTS).

Провайдеры выбираются в .env (ASR_PROVIDER, TTS_PROVIDER), код не меняется.
Запуск из корня проекта: BOT_PLATFORM=vk VK_BOT_TOKEN=... python examples/voice_bot.py
"""

import asyncio
import os

from dotenv import load_dotenv

from botkit.asr import load_asr, transcribe_attachment
from botkit.platform import PriceBook, SQLiteUsageTracker, load_llm, usage_context
from botkit.runtime import make_adapter
from botkit.transport import IncomingMessage
from botkit.tts import load_tts, speak


async def main() -> None:
    load_dotenv()
    tracker = SQLiteUsageTracker("data/voice.sqlite3", prices=PriceBook("config/prices.json"))
    llm = load_llm(tracker=tracker)
    asr = load_asr(tracker=tracker)  # None, если ASR_PROVIDER=disabled
    tts = load_tts(tracker=tracker)  # None, если TTS_PROVIDER=disabled
    platform = os.getenv("BOT_PLATFORM", "vk")
    adapter = make_adapter(platform, os.environ[f"{platform.upper()}_BOT_TOKEN"], tracker=tracker, bot_id="voice")

    async def reply(message: IncomingMessage) -> None:
        text = message.text
        voices = [a for a in message.attachments if a.kind in {"audio", "voice"}]
        if voices:
            if asr is None:
                await adapter.send(message.chat_id, "Голосовые сообщения пока не поддерживаются.")
                return
            async with await adapter.start_typing(message.chat_id):
                try:
                    text = (await transcribe_attachment(asr, voices[0], timeout=120)).response
                except Exception:
                    await adapter.send(message.chat_id, "Не удалось распознать голосовое сообщение.")
                    return
            await adapter.send(message.chat_id, f"Распознано: {text}")
        if not text.strip():
            return
        with usage_context(skill_context="VOICE_CHAT"):
            answer = (await llm.ainvoke(text)).response
        await adapter.send(message.chat_id, answer)  # текст остаётся всегда
        if tts is not None:
            # None = озвучить не удалось; текст уже отправлен, поэтому ничего больше не делаем.
            await speak(adapter, message.chat_id, answer[:1000])

    adapter.on_message(reply)
    try:
        await adapter.run()
    finally:
        await adapter.aclose()
        await llm.aclose()
        for closable in (asr, tts):
            if closable is not None:
                await closable.aclose()


if __name__ == "__main__":
    asyncio.run(main())
