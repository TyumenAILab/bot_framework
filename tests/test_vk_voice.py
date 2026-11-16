from types import SimpleNamespace

import pytest

from botkit.asr import transcribe_attachment
from botkit.llm import LLMResponse
from botkit.transport.vk import VKAdapter, VKUploadError
from botkit.tts import SynthesizedAudio, speak


class FakeHttp:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def request_text(self, url, method, data):
        self.calls.append((url, method, data["file"].name))
        return self.reply.pop(0) if isinstance(self.reply, list) else self.reply


class FakeApi:
    def __init__(self, reply):
        self.http_client, self.requests = FakeHttp(reply), []

    async def request(self, method, params):
        self.requests.append((method, params))
        if method == "docs.getMessagesUploadServer":
            return {"response": {"upload_url": "https://upload.example/x"}}
        if method == "messages.send":
            return {"response": 77}
        return {
            "response": {"type": "audio_message", "audio_message": {"owner_id": 1, "id": 2, "access_key": "k"}}
        }


def adapter(reply):
    instance = VKAdapter.__new__(VKAdapter)
    instance.api = FakeApi(reply)
    instance.upload_retry_delay = 0
    instance.max_attachment_bytes = 1024
    instance.tracker = None
    instance.bot_id = "bot"
    instance.platform = "vk"
    return instance


def methods(vk):
    return [m for m, _ in vk.api.requests]


async def test_voice_upload_returns_an_attachment_string_and_names_the_file():
    vk = adapter('{"file": "abc"}')
    assert await vk._upload_voice(b"OggS", peer_id=5, filename="question.ogg") == "audio_message1_2_k"
    assert vk.api.http_client.calls == [("https://upload.example/x", "POST", "question.ogg")]
    assert methods(vk) == ["docs.getMessagesUploadServer", "docs.save"]
    assert vk.api.requests[0][1] == {"type": "audio_message", "peer_id": 5}


@pytest.mark.parametrize("reply", ["<html>Bad request</html>", "", "[1]", "null"])
async def test_non_json_reply_explains_itself_instead_of_a_bare_jsondecodeerror(reply):
    vk = adapter(reply)
    with pytest.raises(VKUploadError) as info:
        await vk._upload_voice(b"OggS", peer_id=5, filename="question.ogg")
    assert "did not return JSON after 3 attempts" in info.value.detail
    assert "Could not upload the file to VK" in str(info.value) and len(vk.api.http_client.calls) == 3
    assert "Bad request" not in str(info.value)  # the server reply is never echoed


async def test_a_temporary_5xx_page_is_retried_with_a_fresh_upload_server():
    page = "<!DOCTYPE HTML><html><title>The page is temporarily unavailable</title></html>"
    vk = adapter([page, page, '{"file": "abc"}'])
    assert await vk._upload_voice(b"OggS", peer_id=5, filename="question.ogg") == "audio_message1_2_k"
    assert methods(vk).count("docs.getMessagesUploadServer") == 3 and methods(vk).count("docs.save") == 1


async def test_json_without_a_file_field_is_reported():
    with pytest.raises(VKUploadError, match="rejected"):
        await adapter('{"error": "wrong format"}')._upload_voice(b"x", peer_id=5, filename="q.ogg")


@pytest.mark.parametrize("data", [b"", b"x" * 2000])
async def test_empty_or_oversized_audio_is_rejected_before_upload(data):
    vk = adapter('{"file": "abc"}')
    with pytest.raises(ValueError, match="size limit"):
        await vk._upload_voice(data, peer_id=5, filename="q.ogg")
    assert not vk.api.requests


async def test_send_attachment_voice_uploads_natively_and_sends_the_message():
    vk = adapter('{"file": "abc"}')
    message_id = await vk.send_attachment("5", b"OggS", filename="q.ogg", kind="voice", caption="Question")
    assert message_id == "77"
    assert methods(vk) == ["docs.getMessagesUploadServer", "docs.save", "messages.send"]
    send = vk.api.requests[-1][1]
    assert send["peer_id"] == 5 and send["message"] == "Question" and send["attachment"] == "audio_message1_2_k"


async def test_speak_delivers_a_synthesized_voice_message_through_vk():
    class Tts:
        async def synthesize(self, text, *, voice=None):
            return SynthesizedAudio(b"OggS" + text.encode(), "audio/ogg", 2.0)

        async def aclose(self):
            pass

    vk = adapter('{"file": "abc"}')

    class Typing:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    async def start_typing(chat_id):
        return Typing()

    vk.start_typing = start_typing
    sent = await speak(vk, "5", "Hello", Tts(), caption="cap")
    assert sent is not None and sent.seconds == 2.0
    assert vk.api.http_client.calls[0][2] == "voice.ogg" and methods(vk)[-1] == "messages.send"


async def test_incoming_vk_voice_message_is_transcribed_end_to_end():
    """audio_message → normalized Attachment → any recognizer → text."""
    vk = adapter("")
    vk._users = {}
    downloaded = []

    async def download(url):
        downloaded.append(url)
        return b"OggS-voice"

    vk._download = download

    async def identity(user_id):
        return None, None

    vk._identity = identity
    message = SimpleNamespace(
        attachments=[
            SimpleNamespace(
                type=SimpleNamespace(value="audio_message"),
                audio_message=SimpleNamespace(id=9, link_ogg="https://cs.vkuseraudio.net/x.ogg", link_mp3=None),
            )
        ],
        from_id=1, peer_id=1, text="", id=3,
    )
    incoming = await vk.normalize(message)
    (attachment,) = incoming.attachments

    class Recognizer:
        async def atranscribe(self, data, *, mime_type, filename, timeout=None, language=None):
            assert (data, mime_type, filename) == (b"OggS-voice", "audio/ogg", "voice.ogg")
            return LLMResponse("привет", "привет", "raw_text_fallback")

    assert (await transcribe_attachment(Recognizer(), attachment)).response == "привет"
    assert downloaded == ["https://cs.vkuseraudio.net/x.ogg"]
