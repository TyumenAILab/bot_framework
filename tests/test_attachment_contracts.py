from array import array
from unittest.mock import AsyncMock

import pytest

from botkit.transport.base import Attachment
from botkit.transport.telegram import _LimitedBuffer


@pytest.mark.parametrize("size", [-1, True, False, 1.5, float("nan"), float("inf"), "2"])
async def test_attachment_rejects_invalid_declared_size_before_calling_reader(size):
    reader = AsyncMock(return_value=b"data")
    with pytest.raises(ValueError, match="size"):
        await Attachment("document", "text/plain", reader, size=size).read_bytes()
    reader.assert_not_awaited()


@pytest.mark.parametrize("limit", [-1, 0, True, False, 1.5, float("nan"), float("inf"), "2"])
async def test_attachment_rejects_invalid_byte_limit_before_calling_reader(limit):
    reader = AsyncMock(return_value=b"data")
    with pytest.raises(ValueError, match="limit"):
        await Attachment("document", "text/plain", reader, max_bytes=limit).read_bytes()
    reader.assert_not_awaited()


@pytest.mark.parametrize("value", [None, "private document", [1, 2], object()])
async def test_attachment_rejects_non_binary_reader_results_without_echoing_data(value):
    with pytest.raises(TypeError, match="bytes") as caught:
        await Attachment("document", "text/plain", AsyncMock(return_value=value)).read_bytes()
    assert "private document" not in str(caught.value)


@pytest.mark.parametrize("value", [b"data", bytearray(b"data"), memoryview(b"data")])
async def test_attachment_freezes_supported_binary_results(value):
    result = await Attachment("document", "text/plain", AsyncMock(return_value=value), size=0).read_bytes()
    assert result == b"data" and type(result) is bytes


async def test_attachment_bounds_memoryview_by_bytes_not_element_count():
    value = memoryview(array("I", [1, 2]))
    assert len(value) == 2 and value.nbytes == 8
    with pytest.raises(ValueError, match="size limit"):
        await Attachment(
            "document", "application/octet-stream", AsyncMock(return_value=value), max_bytes=4
        ).read_bytes()


def test_limited_buffer_bounds_memoryview_by_bytes_not_element_count():
    value = memoryview(array("I", [1, 2]))
    with _LimitedBuffer(4) as target:
        with pytest.raises(ValueError, match="size limit"):
            target.write(value)
        assert target.getvalue() == b""


def test_limited_buffer_writelines_cannot_bypass_write_limit():
    with _LimitedBuffer(4) as target:
        with pytest.raises(ValueError, match="size limit"):
            target.writelines([b"12", b"345"])
        assert target.getvalue() == b"12"
