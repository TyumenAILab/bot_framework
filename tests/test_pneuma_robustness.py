"""Offline checks for malformed ASR status/identifiers, without real audio/API."""

import httpx
import pytest

from botkit.llm import PneumaConfig, PneumaTranscriber, ProviderError
from botkit.llm.pneuma_upload import ChunkedPneumaTranscriber, UploadConfig, UploadError
from botkit.usage import UsageEvent

CONFIG = UploadConfig("https://asr.test/api", user_id="synthetic-user", poll_interval=0.001, retry_delays=())


@pytest.mark.parametrize(
    "result",
    [
        {"ready": 1},
        {"ready": "false"},
        {"ready": []},
        {"status": "alien"},
        {"status": {"private": "diagnostic"}},
    ],
)
async def test_malformed_result_never_becomes_a_transcript(result):
    def server(request):
        body = (
            {"transcript": "Unverified stale text.", **result}
            if request.url.path.endswith("/result")
            else {"status": "completed"}
        )
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        provider = ChunkedPneumaTranscriber(CONFIG, client=client)
        with pytest.raises(UploadError):
            await provider._result("job-1", UsageEvent())


async def test_pending_result_is_polled_instead_of_returning_stale_text():
    results = iter(
        [
            {"status": "running", "transcript": "Stale text."},
            {"status": "completed", "ready": True, "transcript": "Final text."},
        ]
    )

    def server(request):
        return httpx.Response(
            200, json=next(results) if request.url.path.endswith("/result") else {"status": "completed"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        provider = ChunkedPneumaTranscriber(CONFIG, client=client)
        assert await provider._result("job-1", UsageEvent()) == "Final text."


@pytest.mark.parametrize(
    "body",
    [
        {"job_id": "a", "transcription_job_id": "b"},
        {"job_id": "a", "job": {"id": "b"}},
        {"job_id": False},
        {"job": []},
    ],
)
def test_conflicting_or_malformed_job_ids_are_rejected(body):
    with pytest.raises(UploadError):
        ChunkedPneumaTranscriber._job(body)


def test_zero_is_not_lost_when_numeric_job_identifiers_are_supported():
    assert ChunkedPneumaTranscriber._job({"job_id": 0}) == "0"


def test_oversized_numeric_offset_has_safe_error_contract():
    with pytest.raises(UploadError):
        ChunkedPneumaTranscriber._offset({"received_bytes": "9" * 5000}, 10)


def test_excessive_transcript_nesting_has_safe_error_contract():
    value = "text"
    for _ in range(500):
        value = [value]
    with pytest.raises(UploadError):
        ChunkedPneumaTranscriber._transcript(value)


@pytest.mark.parametrize("legacy", [True, False])
async def test_excessive_json_nesting_is_a_provider_failure(legacy):
    body = b"[" * 2000 + b"0" + b"]" * 2000
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    ) as client:
        provider = (
            PneumaTranscriber(PneumaConfig("https://asr.test"), client=client)
            if legacy
            else ChunkedPneumaTranscriber(CONFIG, client=client)
        )
        with pytest.raises(ProviderError):
            await provider._request("GET", "synthetic-job")
