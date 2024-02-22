import pytest
from video_sampler.integrations.yt_dlp_plugin import YTDLPPlugin
from video_sampler.language.keyword_capture import download_sub, KeywordExtractor
from video_sampler.sampler import SegmentSampler, SamplerConfig, Worker
import tempfile
import json
import os


@pytest.fixture
def subtitles() -> list[list]:
    return json.load(open("./tests/assets/subs.json"))


def test_keyword_extractor(subtitles):
    ke = KeywordExtractor(["cat", "kitten", "feline"])
    c = sum(1 for _ in ke.capture_keyword_segments(subtitles))
    assert c == 4


def test_segment_sampler(random_video):
    ytdlp = YTDLPPlugin()
    ke = KeywordExtractor(["cat", "kitten", "feline"])
    title, url, subs = next(ytdlp.generate_urls(random_video, get_subs=True))
    worker = Worker(
        cfg=SamplerConfig(),
        processor_cls=SegmentSampler,
        extra_processor_args={
            "segment_generator": ke.capture_keyword_segments(subs),
        },
    )
    with tempfile.TemporaryDirectory() as tempdir:
        worker.launch(video_path=url, output_path=tempdir, pretty_video_name=title)

        assert len(os.listdir(tempdir)) > 0


def test_single_url_gen(random_video):
    ytdlp = YTDLPPlugin()
    title, url, subs = next(ytdlp.generate_urls(random_video, get_subs=True))
    assert title, f"Expected a title, got {title}"
    assert url.endswith(".m3u8"), f"Expected a video URL, got {url}"
    assert subs and len(subs) > 0, f"Expected subtitles, got {subs}"


def test_search_url_gen():
    ytdlp = YTDLPPlugin()
    expected_results, results = 5, 0
    search_url = f"ytsearch{expected_results}:cute cats"
    for title, url, subs in ytdlp.generate_urls(search_url, get_subs=True):
        assert title, f"Expected a title, got {title}"
        assert url.endswith(".m3u8"), f"Expected a video URL, got {url}"
        assert subs and len(subs) > 0, f"Expected subtitles, got {subs}"
        results += 1
    assert results == 5, f"Expected 5 results, got {results}"
