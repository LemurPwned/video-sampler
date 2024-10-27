from video_sampler.utils import slugify


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
