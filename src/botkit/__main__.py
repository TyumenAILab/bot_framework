from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import asdict

import httpx
from dotenv import load_dotenv

from .extraction import ContentExtractionService, ExtractionConfig
from .llm import LLMGateway, load_llm, load_transcriber
from .llm.config import LLMConfig
from .runtime import ChatBot, make_adapter
from .transport import IncomingMessage, MemoryAdapter
from .usage import PriceBook, SQLiteUsageTracker


async def demo(db: str) -> None:
    """Exercise routing -> HTTP protocol -> parsing -> accounting -> reply offline."""

    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "offline-demo",
                "model": "demo-model",
                "choices": [
                    {
                        "message": {"content": '{"response": "Готово: A1 + A2 + A4 работают вместе."}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 18, "completion_tokens": 12},
            },
        )

    tracker = SQLiteUsageTracker(db)
    adapter = MemoryAdapter(tracker=tracker, bot_id="offline-demo")
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        llm = LLMGateway(
            LLMConfig("openai_compatible", "demo-model", "http://offline.test/v1"),
            tracker=tracker,
            client=client,
        )
        ChatBot(llm, [adapter])
        try:
            await adapter.dispatch(IncomingMessage("memory", "demo-user", "demo-chat", "Привет"))
            print(adapter.sent[-1]["text"])
            print(
                json.dumps(
                    asdict(await tracker.stats(bot_id="offline-demo")),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
        finally:
            await adapter.aclose()


async def execute(args: argparse.Namespace) -> None:
    if args.command == "demo":
        await demo(args.db or "data/demo.sqlite3")
        return
    tracker = SQLiteUsageTracker(
        args.db or os.getenv("USAGE_DB", "data/usage.sqlite3"),
        prices=PriceBook(os.getenv("PRICE_FILE", "config/prices.json")),
        log_text=os.getenv("LOG_TEXT", "false").lower() == "true",
        text_limit=int(os.getenv("LOG_TEXT_LIMIT", "20000")),
    )
    if args.command in {"stats", "export"}:
        filters = {
            "user_id": args.user,
            "skill_context": args.skill,
            "bot_id": args.bot,
            "platform": args.platform,
            "kind": args.kind,
        }
        if args.command == "stats":
            print(
                json.dumps(asdict(await tracker.stats(**filters)), ensure_ascii=False, indent=2, default=str)
            )
        else:
            print(f"Exported {await tracker.export_jsonl(args.output, **filters)} events")
        return
    llm = load_llm(tracker=tracker)
    adapters = []
    async with AsyncExitStack() as stack:
        stack.push_async_callback(llm.aclose)
        transcriber = load_transcriber(tracker=tracker)
        if transcriber is not None:
            stack.push_async_callback(transcriber.aclose)
        for platform in args.platform:
            adapters.append(
                make_adapter(
                    platform,
                    os.getenv(f"{platform.upper()}_BOT_TOKEN", ""),
                    tracker=tracker,
                    bot_id=os.getenv("BOT_ID", "example-bot"),
                    max_attachment_bytes=int(os.getenv("MAX_ATTACHMENT_BYTES", str(20 * 1024 * 1024))),
                )
            )
            stack.push_async_callback(adapters[-1].aclose)
        extraction = ContentExtractionService(
            llm, tracker=tracker, config=ExtractionConfig.from_env(), transcriber=transcriber
        )
        await ChatBot(llm, adapters, extraction=extraction).run()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    # httpx INFO includes URLs; avoid exposing signed attachment URLs in default logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="botkit A1/A2/A4")
    sub = parser.add_subparsers(dest="command", required=True)
    demo_parser = sub.add_parser("demo", help="Offline integration smoke run; no keys or network")
    demo_parser.add_argument("--db")
    run = sub.add_parser("run", help="Run the same bot in selected messengers")
    run.add_argument("--platform", choices=["telegram", "vk"], nargs="+", required=True)
    run.add_argument("--db")
    for command in ("stats", "export"):
        item = sub.add_parser(command)
        item.add_argument("--db")
        item.add_argument("--user")
        item.add_argument("--skill")
        item.add_argument("--bot")
        item.add_argument("--platform")
        item.add_argument("--kind", default="llm")
        if command == "export":
            item.add_argument("--output", required=True)
    try:
        asyncio.run(execute(parser.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
