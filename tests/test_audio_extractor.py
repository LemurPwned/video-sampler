import sys

import pytest

from video_sampler.language.keyword_capture import AudioKeywordExtractor


def test_audio_extractor_requires_whisper(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(ImportError):
        AudioKeywordExtractor(["test"])
