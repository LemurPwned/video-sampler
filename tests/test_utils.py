import warnings
import pytest

from video_sampler.config import SaveFormatConfig
from video_sampler.utils import slugify, batched


def test_slugify():
    assert slugify("Hello World") == "hello-world"
    assert slugify("Hello World", allow_unicode=True) == "hello-world"
    assert slugify("Hello World", allow_unicode=False) == "hello-world"
    assert slugify("Hello World 123") == "hello-world-123"
    assert slugify("Hello World 123", allow_unicode=True) == "hello-world-123"
    assert slugify("Hello World 123", allow_unicode=False) == "hello-world-123"


def test_slugify_weird_chars():
    assert slugify("Hello World 123!@#$%^&*()") == "hello-world-123"


def test_slugify_japanese():
    assert slugify("こんにちは世界", allow_unicode=True) == "こんにちは世界"


def test_batched_regular():
    items = list(batched(range(5), 2))
    assert items == [(0, 1), (2, 3), (4,)]


def test_batched_invalid():
    with pytest.raises(ValueError):
        list(batched([1, 2], 0))


def test_save_format_warning():
    with warnings.catch_warnings(record=True) as w:
        SaveFormatConfig(encode_time_b64=True, avoid_dot=True)
        assert any("avoid_dot will be ignored" in str(wi.message) for wi in w)
