"""Explicit Pneuma smoke test, with optional local LLM postprocessing from env. No messenger needed."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from botkit.extraction import ContentExtractionService, ExtractionConfig
from botkit.llm import ProviderError, load_transcriber
from botkit.transport import Attachment
from botkit.usage import SQLiteUsageTracker, usage_context


async def run(args) -> None:
    if not args.env.is_file():
        raise ValueError("Env file not found; copy .env.example and fill ASR settings")
    load_dotenv(args.env, override=True)
    if args.output and args.output.exists():
        raise ValueError("Output already exists; choose a NEW filename")
    config = ExtractionConfig.from_env()
    size = args.file.stat().st_size
    if not 0 < size <= config.max_bytes:
        raise ValueError("Audio is empty or exceeds MAX_ATTACHMENT_BYTES")
    tracker = SQLiteUsageTracker(args.db, log_text=False)
    asr = load_transcriber(tracker=tracker)
    if asr is None:
        raise ValueError("Set ASR_PROVIDER=pneuma and PNEUMA_BASE_URL in the env file")

    def read_bounded():
        with args.file.open("rb") as source:
            return source.read(config.max_bytes + 1)

    async def read():
        return await asyncio.to_thread(read_bounded)

    async with asr:
        service = ContentExtractionService(transcriber=asr, config=config, tracker=tracker)
        with usage_context(
            bot_id="pneuma-smoke",
            user_id="pneuma-smoke",
            request_id="single-file-smoke",
            skill_context="AUDIO_TRANSCRIPTION",
        ):
            result = await service.extract(
                Attachment("audio", "application/octet-stream", read, filename=args.file.name, size=size)
            )
    if result.kind != "transcript":
        raise ValueError("The supplied file is not a supported audio attachment")
    if args.output:
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(asdict(result), target, ensure_ascii=False, indent=2)
    print(result.text or "[Empty transcript: no recognized speech]")
    print(json.dumps({"warnings": result.warnings, "metadata": result.metadata}, ensure_ascii=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="MP3, OGG/Opus voice, WAV, M4A, FLAC, AAC or WMA")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--db", type=Path, default=Path("data/transcription.sqlite3"))
    parser.add_argument("--output", type=Path, help="Write transcript and metadata to a NEW JSON file")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow Pneuma upload and optional LLM processing configured in env",
    )
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("Sending audio requires --live; the server may store the audio and transcript")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run(args))
    except ProviderError as exc:
        print(f"Transcription failed: {exc} (HTTP {exc.status_code or 'n/a'})")
        return 1
    except (ValueError, OSError, TimeoutError) as exc:
        # Paths/env/parser errors can contain sensitive values; print only the type.
        print(f"Transcription failed: {type(exc).__name__}. Check file, env and server availability.")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
