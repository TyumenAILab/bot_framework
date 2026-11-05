import json
from dataclasses import replace
from urllib.parse import parse_qs

import httpx
import pytest

from botkit.llm.pneuma_upload import ChunkedPneumaTranscriber, UploadConfig, UploadError
from botkit.llm.upload_state import MemoryUploadStore


class API:
    def __init__(self):
        self.uploaded = b""
        self.calls = []
        self.job = None
        self.form = False
        self.lost_patch = False
        self.partial_patch = False
        self.lost_complete = False
        self.wait_once = False
        self.result = {"transcript": "мой ответ не изменился"}

    def __call__(self, request):
        path = request.url.path.removeprefix("/api")
        self.calls.append((request.method, path, request))
        assert request.headers["Authorization"] == "Bearer secret"
        if request.method == "POST" and path in {"/uploads", "/uploads/upload-1/complete"}:
            if request.headers["content-type"] == "application/json":
                if self.form:
                    return httpx.Response(422, json={"detail": "use form"})
                data = json.loads(request.content)
            else:
                data = {key: values[0] for key, values in parse_qs(request.content.decode()).items()}
            assert data["filename"] == "voice.ogg"
            assert int(data["size"]) == 10
            assert data["user_id"] == "stu_private" and data["device"] == "cuda"
            if path == "/uploads":
                return httpx.Response(200, json={"upload_id": "upload-1", "received_bytes": 0})
            self.job = "job-1"
            if self.lost_complete:
                self.lost_complete = False
                raise httpx.ReadError("secret server error", request=request)
            return httpx.Response(200, json={"job": {"id": self.job}})
        if path == "/uploads/upload-1" and request.method == "PATCH":
            offset = int(request.headers["Upload-Offset"])
            assert offset == len(self.uploaded)
            assert request.headers["Content-Type"] == "application/octet-stream"
            part = request.content[:2] if self.partial_patch else request.content
            self.partial_patch = False
            self.uploaded += part
            if self.lost_patch:
                self.lost_patch = False
                raise httpx.ReadTimeout("sensitive diagnostic", request=request)
            return httpx.Response(200, json={"received_bytes": len(self.uploaded)})
        if path == "/uploads/upload-1":
            value = {"received_bytes": len(self.uploaded)}
            if self.job:
                value["transcription_job_id"] = self.job
            return httpx.Response(200, json=value)
        if path == "/transcriptions/job-1":
            status = "running" if self.wait_once else "succeeded"
            self.wait_once = False
            return httpx.Response(200, json={"job_id": "job-1", "status": status})
        if path == "/transcriptions/job-1/result":
            return httpx.Response(200, json=self.result)
        raise AssertionError((request.method, path))


def config(**options):
    return replace(
        UploadConfig(
            "https://pneuma.test/api",
            api_key="secret",
            user_id="stu_private",
            chunk_bytes=4,
            poll_interval=0.001,
            retry_delays=(0, 0),
        ),
        **options,
    )


async def transcribe(provider):
    return await provider.atranscribe(
        b"0123456789", mime_type="audio/ogg", filename="C:/Private/Student Name.ogg"
    )


@pytest.mark.parametrize("form", [False, True])
@pytest.mark.parametrize(
    "lost_patch,partial_patch,lost_complete",
    [
        (False, False, False),
        (True, False, False),
        (True, True, True),
    ],
)
async def test_reference_contract_and_lost_reply_reconciliation(
    form, lost_patch, partial_patch, lost_complete
):
    api = API()
    api.form, api.lost_patch, api.partial_patch, api.lost_complete = (
        form,
        lost_patch,
        partial_patch,
        lost_complete,
    )
    api.wait_once = True
    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as client:
        provider = ChunkedPneumaTranscriber(config(), client=client)
        result = await transcribe(provider)
        assert result.response == api.result["transcript"]
        assert api.uploaded == b"0123456789"
        assert (
            sum(
                method == "POST"
                and path.endswith("/complete")
                and (not form or request.headers["content-type"] != "application/json")
                for method, path, request in api.calls
            )
            == 1
        )
        await provider.aclose()
        assert not client.is_closed


@pytest.mark.parametrize("offset", [-1, 11, True, 1.5, "NaN"])
async def test_bad_offsets_fail_without_sending_audio(offset):
    calls = []

    def server(request):
        calls.append(request)
        return httpx.Response(200, json={"upload_id": "u", "received_bytes": offset})

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        with pytest.raises(UploadError, match="offset"):
            await transcribe(ChunkedPneumaTranscriber(config(), client=client))
        assert len(calls) == 1


@pytest.mark.parametrize("identifier", ["../../secret", "https://evil.test", "bad?x", "", "a/b"])
async def test_server_identifiers_cannot_change_destination(identifier):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"upload_id": identifier}))
    ) as client:
        with pytest.raises(UploadError, match="identifier"):
            await transcribe(ChunkedPneumaTranscriber(config(), client=client))


@pytest.mark.parametrize("status", [301, 401, 403, 500])
async def test_create_not_retried_and_errors_redacted(status):
    calls = []

    def server(request):
        calls.append(request)
        return httpx.Response(
            status, content=b"private-key raw transcript", headers={"location": "https://evil.test"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=True) as client:
        with pytest.raises(UploadError) as error:
            await transcribe(ChunkedPneumaTranscriber(config(), client=client))
        assert "private-key" not in str(error.value) and len(calls) == 1


async def test_timeout_retains_job_and_never_invents_cancel_endpoint():
    api = API()

    def pending(request):
        if request.url.path.endswith("/transcriptions/job-1"):
            return httpx.Response(200, json={"status": "running"})
        return api(request)

    store = MemoryUploadStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(pending)) as client:
        provider = ChunkedPneumaTranscriber(config(timeout=0.03), client=client, store=store)
        with pytest.raises(TimeoutError):
            await transcribe(provider)
        assert any(row[1].get("job_id") == "job-1" for row in store.rows.values())
        assert not any("cancel" in path for _, path, _ in api.calls)


@pytest.mark.parametrize(
    "transcript", ["Речь.", {"text": "Речь."}, [{"text": "Речь."}], {"segments": [{"text": "Речь."}]}]
)
async def test_documented_transcript_variants(transcript):
    api = API()
    api.result = {"transcript": transcript}
    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as client:
        assert (await transcribe(ChunkedPneumaTranscriber(config(), client=client))).response == "Речь."


async def test_missing_transcript_and_failed_job_not_returned_as_text():
    api = API()
    api.result = {"transcript": {"error": "secret-body"}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as client:
        provider = ChunkedPneumaTranscriber(config(), client=client)
        with pytest.raises(UploadError, match="textual transcript"):
            await transcribe(provider)
        api.result = {"status": "failed", "transcript": "must not return"}
        with pytest.raises(UploadError, match="processing failed"):
            await transcribe(provider)


async def test_lost_patch_and_temporarily_unavailable_reconciliation():
    api = API()
    api.lost_patch = True
    failed_gets = 0

    def server(request):
        nonlocal failed_gets
        if request.method == "GET" and request.url.path.endswith("/uploads/upload-1") and failed_gets < 2:
            failed_gets += 1
            return httpx.Response(503, json={"error": "temporary"})
        return api(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        await transcribe(ChunkedPneumaTranscriber(config(), client=client))
    assert api.uploaded == b"0123456789" and failed_gets == 2
    offsets = [int(request.headers["Upload-Offset"]) for method, _, request in api.calls if method == "PATCH"]
    assert offsets == [0, 4, 8]


async def test_response_size_is_bounded():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 200))
    ) as client:
        with pytest.raises(UploadError, match="too large"):
            await transcribe(ChunkedPneumaTranscriber(config(max_response_bytes=100), client=client))
