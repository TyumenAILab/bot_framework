"""Offline checks of the operator's standalone ASR example."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from botkit.llm import PneumaConfig, PneumaTranscriber
from botkit.usage import SQLiteUsageTracker

spec = importlib.util.spec_from_file_location(
    "transcribe_example", Path(__file__).parents[1] / "examples" / "transcribe_audio.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_no_live_flag_stops_before_loading_env_or_network():
    with pytest.raises(SystemExit) as error:
        example.main(["not-a-real-file.mp3", "--env", "not-a-real-env"])
    assert error.value.code == 2


async def test_single_file_with_output_and_usage_without_llm(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"ID3synthetic-not-real-audio")
    env = tmp_path / "test.env"
    env.write_text("# Test only\n", encoding="utf-8")
    args = SimpleNamespace(file=audio, env=env, output=tmp_path / "transcript.json", db=tmp_path / "usage.db")
    calls = []

    def server(request):
        calls.append(request)
        if request.method == "POST":
            assert audio.read_bytes() in request.content
            return httpx.Response(200, json={"job_id": "f" * 32, "status": "completed"})
        return httpx.Response(200, json={"status": "completed", "ready": True, "transcript": "Речь"})

    monkeypatch.setenv("MAX_ATTACHMENT_BYTES", "1024")
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        monkeypatch.setattr(
            example,
            "load_transcriber",
            lambda **kwargs: PneumaTranscriber(PneumaConfig("https://asr.test"), client=client, **kwargs),
        )
        await example.run(args)
        output = json.loads(args.output.read_text(encoding="utf-8"))
        assert output["kind"] == "transcript" and output["text"] == "Речь"
        assert "Речь" in capsys.readouterr().out
        assert len(calls) == 2
        with pytest.raises(ValueError, match="already exists"):
            await example.run(args)
        assert len(calls) == 2  # refuse overwrite before another upload
        assert json.loads(args.output.read_text(encoding="utf-8")) == output
    events = await SQLiteUsageTracker(args.db).events()
    assert {e["operation"] for e in events} == {"transcription", "extract"}
    assert all(e["output_text"] == "" for e in events)


async def test_standalone_fails_before_upload_for_oversized_audio(monkeypatch, tmp_path):
    env, audio = tmp_path / "test.env", tmp_path / "test.mp3"
    env.write_text("# Test only\n", encoding="utf-8")
    audio.write_bytes(b"1234")
    monkeypatch.setenv("MAX_ATTACHMENT_BYTES", "3")
    monkeypatch.setattr(
        example, "load_transcriber", lambda **_: pytest.fail("Must reject before ASR creation")
    )
    with pytest.raises(ValueError, match="MAX_ATTACHMENT_BYTES"):
        await example.run(SimpleNamespace(env=env, file=audio, output=None))
