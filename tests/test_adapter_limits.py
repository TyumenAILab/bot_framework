import pytest

from botkit.transport import MemoryAdapter


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, float("nan"), float("inf"), "1024", None])
def test_invalid_download_limit_is_rejected_at_adapter_construction(limit):
    with pytest.raises(ValueError, match="max_attachment_bytes must be a positive integer"):
        MemoryAdapter(max_attachment_bytes=limit)


def test_finite_integer_download_limit_is_preserved():
    assert MemoryAdapter(max_attachment_bytes=1024).max_attachment_bytes == 1024
