import pytest

from video_sampler.config import SamplerConfig


def test_valid_start_stop_times():
    config = SamplerConfig(start_time_s=10, end_time_s=20)
    assert config.start_time_s == 10
    assert config.end_time_s == 20


def test_invalid_start_stop_times():
    with pytest.raises(
        ValueError, match="start_time_s must be strictly less than the end_time_s"
    ):
        SamplerConfig(start_time_s=40, end_time_s=10)


def test_start_time_equals_end_time():
    with pytest.raises(
        ValueError, match="start_time_s must be strictly less than the end_time_s"
    ):
        SamplerConfig(start_time_s=10, end_time_s=10)


def test_no_end_time():
    config = SamplerConfig(start_time_s=10, end_time_s=None)
    assert config.start_time_s == 10
    assert config.end_time_s is None
