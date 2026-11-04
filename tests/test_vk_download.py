import asyncio
from types import SimpleNamespace

import httpx
import pytest

from botkit.transport.vk import VKAdapter, VKAttachmentDownloadError


@pytest.mark.parametrize("location,violation", [
    ("http://userapi.com/file", "scheme"),
    ("https://unknown.example/file?TOKEN_SENTINEL", "host"),
    ("https://userapi.com:8443/file", "port"),
    ("https://user:secret@userapi.com/file", "userinfo"),
    ("https://attacker.example/file#userapi.com", "host"),
])
async def test_redirect_diagnostic_is_categorical_and_preserves_boundary(location, violation):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(302, headers={"Location": location})
    )) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/file")
    assert caught.value.violation == violation
    assert caught.value.reason == "unsafe_url"
    assert "secret" not in str(caught.value).lower()
    assert "unknown.example" not in str(caught.value)
    assert "TOKEN_SENTINEL" not in repr(caught.value)


async def test_fragment_is_removed_before_validated_redirect_and_http_request():
    requests = []

    def endpoint(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, headers={
                "Location": "https://cdn.vkuser.net/final?x=1#TOKEN_SENTINEL@localhost"
            })
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        assert await adapter(client)._download("https://doc.userapi.com/a#INITIAL_SECRET") == b"ok"
    assert str(requests[0].url) == "https://doc.userapi.com/a"
    assert str(requests[1].url) == "https://cdn.vkuser.net/final?x=1"
    assert all("authorization" not in r.headers and "cookie" not in r.headers for r in requests)


@pytest.mark.parametrize("location", [
    "https://attacker.example/file#https://userapi.com/file",
    "https://user:secret@userapi.com/file#ignored",
    "https://localhost/file#userapi.com",
    "https://userapi.com.attacker.example/file#ok",
    "https://userapi.com/file#bad\r\nX-Header: injected",
])
async def test_fragment_cannot_hide_an_untrusted_redirect(location):
    calls = []
    def endpoint(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": location})
    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        with pytest.raises(VKAttachmentDownloadError):
            await adapter(client)._download("https://doc.userapi.com/file")
    assert len(calls) == 1


async def test_fragment_only_redirect_cannot_evade_loop_detection():
    calls = []
    def endpoint(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "#different"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/file")
    assert caught.value.reason == "redirect_loop" and len(calls) == 1


async def test_fragment_only_initial_url_is_rejected_before_request():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: pytest.fail("a fragment alone must not trigger a request")
    )) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("#ignored")
    assert caught.value.violation == "malformed"


def adapter(client, *, max_bytes=1024):
    labeler = SimpleNamespace(
        message=lambda **_: lambda handler: handler, raw_event=lambda _, **kwargs: lambda handler: handler
    )
    bot = SimpleNamespace(api=object(), labeler=labeler)
    return VKAdapter(bot=bot, download_client=client, max_attachment_bytes=max_bytes)


@pytest.mark.parametrize(
    "host",
    ["vk.com", "vk.ru", "userapi.com", "vkuser.net", "vkuserphoto.ru", "vkuseraudio.net", "vk-cdn.net", "vk-cdn.me"],
)
async def test_vk_download_accepts_explicit_443_on_vetted_domains_and_subdomains(host):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"data"))
    ) as client:
        for hostname in [host, "cdn." + host]:
            assert await adapter(client)._download(f"https://{hostname}:443/file") == b"data"


async def test_vk_document_redirects_to_vkuserphoto_cdn_without_credentials():
    requests = []

    def endpoint(request):
        requests.append(request)
        assert not any(k in request.headers for k in ("cookie", "authorization", "proxy-authorization", "referer"))
        if len(requests) == 1:
            return httpx.Response(302, headers={"Location": "https://psv4.vkuserphoto.ru/file?signature=PRIVATE"})
        assert request.url.host == "psv4.vkuserphoto.ru"
        return httpx.Response(200, content=b"document")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(endpoint), follow_redirects=True,
        headers={"Authorization": "Bearer private", "Referer": "https://private.test"},
        cookies={"private": "cookie"},
    ) as client:
        assert await adapter(client)._download("https://vk.ru/document?signature=PRIVATE") == b"document"
    assert len(requests) == 2


@pytest.mark.parametrize("host", [
    "vkuserphoto.ru.attacker.test", "fakevkuserphoto.ru", "psv4.vkuserphoto.ru.attacker.test",
])
async def test_vkuserphoto_lookalike_redirect_is_rejected(host):
    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": f"https://{host}/file?signature=PRIVATE"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://vk.ru/document")
    assert len(requests) == 1
    assert caught.value.stage == "redirect"
    assert caught.value.violation == "host"
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_vk_download_follows_validated_redirects_without_forwarding_credentials(status):
    requests = []

    def endpoint(request):
        requests.append(request)
        assert request.method == "GET"
        assert not any(
            name in request.headers for name in ("authorization", "proxy-authorization", "cookie", "referer")
        )
        if len(requests) == 1:
            return httpx.Response(
                status,
                headers={
                    "Location": "https://cdn.vkuser.net/file?signature=redirect-secret",
                    "Set-Cookie": "secret=response-cookie; Domain=.vkuser.net; Secure",
                },
            )
        if len(requests) == 2:
            return httpx.Response(302, headers={"Location": "/final?signature=relative-secret"})
        return httpx.Response(200, content=b"document")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(endpoint),
        follow_redirects=True,
        headers={
            "Authorization": "Bearer private-default-token",
            "Proxy-Authorization": "Basic private-proxy-token",
            "Referer": "https://private.test/sensitive",
        },
        cookies={"private": "cookie"},
        auth=("private-user", "private-password"),
    ) as client:
        assert await adapter(client)._download(
            "https://psv4.userapi.com/document?signature=initial-secret"
        ) == (b"document")
    assert len(requests) == 3
    assert requests[-1].url.path == "/final"


@pytest.mark.parametrize("status", [201, 204, 206])
async def test_vk_download_accepts_non_200_success_status(status):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=b"data"))
    ) as client:
        assert await adapter(client)._download("https://doc.userapi.com/file") == b"data"


@pytest.mark.parametrize("status", [300, 304, 305, 400, 401, 403, 404, 429, 500, 503])
async def test_vk_download_http_status_errors_are_safe_and_do_not_read_body(status):
    class UnreadableBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("Do not read an unsuccessful response body")
            yield b"private document"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                status, stream=UnreadableBody(), headers={"Location": "/private-location"}
            )
        )
    ) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/file?signature=private-token")
    assert caught.value.stage == "request"
    assert caught.value.status_code == status
    assert caught.value.reason == "http_status"
    assert str(status) in str(caught.value)
    assert not hasattr(caught.value, "request")
    assert not hasattr(caught.value, "response")
    assert "private" not in repr(caught.value)
    assert "userapi" not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://doc.userapi.com/file",
        "ftp://doc.userapi.com/file",
        "file:///etc/passwd",
        "https://127.0.0.1/file",
        "https://10.0.0.1/file",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/file",
        "https://[::ffff:127.0.0.1]/file",
        "https://[2001:db8::1]/file",
        "https://2130706433/file",
        "https://localhost/file",
        "https://internal.service/file",
        "https://example.com/file",
        "https://userapi.com.attacker.test/file",
        "https://fakeuserapi.com/file",
        "https://userapi.com./file",
        "https://user:password@userapi.com/file",
        "https://@userapi.com/file",
        "https://userapi.com:8443/file",
        "https://userapi.com\\@internal.test/file",
        "https://userapi.com/file\r\nX-Private: token",
        "https://userapi.com/file bad",
        "https://[invalid/file",
    ],
)
async def test_vk_download_rejects_untrusted_initial_urls_before_network(url):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("Unsafe URL must not be requested"))
    ) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download(url)
    assert caught.value.stage == "validate"
    assert caught.value.reason == "unsafe_url"
    assert caught.value.status_code is None
    assert url not in str(caught.value)


@pytest.mark.parametrize(
    "location",
    [
        "http://userapi.com/file",
        "//127.0.0.1/file",
        "https://[::ffff:10.0.0.1]/file",
        "https://internal.service/file",
        "https://userapi.com.attacker.test/file?private-secret",
        "https://userapi.com:8443/file",
        "https://user:private-password@userapi.com/file",
        "https://userapi.com\\@internal.test/file",
        "https://[invalid/file",
    ],
)
async def test_vk_download_revalidates_each_redirect(location):
    requested = []

    def endpoint(request):
        requested.append(request)
        return httpx.Response(302, headers={"Location": location})

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint), follow_redirects=True) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/file")
    assert len(requested) == 1
    assert caught.value.stage == "redirect"
    assert caught.value.status_code == 302
    assert caught.value.reason == "unsafe_url"
    assert "private" not in str(caught.value)
    assert "internal" not in repr(caught.value)


@pytest.mark.parametrize("headers", [{}, {"Location": ""}, [("Location", "/a"), ("Location", "/b")]])
async def test_vk_download_rejects_missing_or_ambiguous_redirect(headers):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(302, headers=headers))
    ) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/file")
    assert caught.value.stage == "redirect"
    assert caught.value.status_code == 302
    assert caught.value.reason == "invalid_location"


async def test_vk_download_detects_redirect_loop_without_another_request():
    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "/same?signature=private-secret"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/same?signature=private-secret")
    assert len(requests) == 1
    assert caught.value.reason == "redirect_loop"
    assert caught.value.status_code == 302


async def test_vk_download_caps_redirect_chain():
    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(307, headers={"Location": f"/next-{len(requests)}?signature=private-secret"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/start")
    assert len(requests) == 6  # initial request plus five permitted redirects
    assert caught.value.reason == "redirect_limit"
    assert caught.value.status_code == 307


async def test_vk_download_closes_and_does_not_read_redirect_bodies():
    closed = []

    class RedirectBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("A redirect body must not be read")
            yield b"private body"

        async def aclose(self):
            closed.append(True)

    def endpoint(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/file"}, stream=RedirectBody())
        assert closed  # release the first response before sending the next GET
        return httpx.Response(200, content=b"data")

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        assert await adapter(client)._download("https://doc.userapi.com/start") == b"data"


@pytest.mark.parametrize("content_length", ["not-a-number", "-1", "2, 2", "9" * 100])
async def test_vk_download_validates_untrusted_content_length(content_length):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"Content-Length": content_length}, content=b"abc")
        )
    ) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/file?signature=private-secret")
    assert caught.value.stage == "size"
    assert caught.value.reason == "invalid_length"
    assert "private" not in str(caught.value)


async def test_vk_download_rejects_large_declared_length_after_redirect_without_reading():
    class UnreadableBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("Oversized declared body must not be read")
            yield b"private document"

    def endpoint(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/file"})
        return httpx.Response(200, headers={"Content-Length": "5"}, stream=UnreadableBody())

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client, max_bytes=4)._download("https://doc.userapi.com/start")
    assert caught.value.stage == "size"
    assert caught.value.reason == "size_limit"


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "2"}])
async def test_vk_download_enforces_actual_streamed_size_and_closes_response(headers):
    closed = []
    yielded = []

    class OversizedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in [b"12", b"345", b"must-not-read"]:
                yielded.append(chunk)
                yield chunk

        async def aclose(self):
            closed.append(True)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, headers=headers, stream=OversizedBody()))
    ) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client, max_bytes=4)._download("https://doc.userapi.com/file")
    assert caught.value.reason == "size_limit"
    assert yielded == [b"12", b"345"]
    assert closed


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError])
async def test_vk_download_sanitizes_transport_failures(error_type):
    def endpoint(request):
        raise error_type("private-token https://private.internal/file document-text", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/file?signature=private-secret")
    import traceback

    formatted = "".join(traceback.format_exception(caught.value))
    # The source line itself includes the synthetic URL, so inspect the exception
    # section and causal chain rather than test-runner source context.
    assert "private.internal" not in formatted
    assert "document-text" not in formatted
    assert "private-token" not in formatted
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.stage == "request"
    assert caught.value.reason == "transport_error"


async def test_vk_download_sanitizes_stream_failure_without_exposing_partial_document():
    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"private document contents"
            raise httpx.ReadError("private-stream-error")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=BrokenBody()))
    ) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/file")
    assert caught.value.stage == "read"
    assert caught.value.status_code == 200
    assert "private" not in str(caught.value)
    assert caught.value.__context__ is None


async def test_vk_download_preserves_cancellation_and_closes_response():
    closed = []

    class CancelledBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise asyncio.CancelledError
            yield b"not read"

        async def aclose(self):
            closed.append(True)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=CancelledBody()))
    ) as client:
        with pytest.raises(asyncio.CancelledError):
            await adapter(client)._download("https://doc.userapi.com/file")
    assert closed


@pytest.mark.parametrize("encoding", ["gzip", "br", "deflate", "gzip, identity", ""])
async def test_vk_download_rejects_unexpected_compression_before_read_and_closes_response(encoding):
    closed = []

    class UnreadableCompressedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("Compressed body must not be read or decompressed")
            yield b"compressed private document"

        async def aclose(self):
            closed.append(True)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"Content-Encoding": encoding}, stream=UnreadableCompressedBody()
            )
        )
    ) as client:
        with pytest.raises(VKAttachmentDownloadError) as caught:
            await adapter(client)._download("https://doc.userapi.com/file")
    assert caught.value.reason == "unsupported_encoding"
    assert caught.value.stage == "read" and caught.value.status_code == 200
    assert closed


@pytest.mark.parametrize("encoding", [None, "identity", "IDENTITY", "  identity  "])
async def test_vk_download_accepts_identity_encoding_without_case_or_padding_regression(encoding):
    headers = {} if encoding is None else {"Content-Encoding": encoding}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, headers=headers, content=b"data"))
    ) as client:
        assert await adapter(client)._download("https://doc.userapi.com/file") == b"data"
