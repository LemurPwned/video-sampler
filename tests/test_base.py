from PIL import Image

from video_sampler.sampler import SamplerConfig, VideoSampler
from video_sampler.schemas import FrameObject


def verify_frame_res(res: list[FrameObject]):
    for frame_obj in res:
        if frame_obj.frame:
            assert isinstance(frame_obj.frame, Image.Image)
            assert isinstance(frame_obj.metadata, dict)
            assert "frame_time" in frame_obj.metadata


def test_pass_gating(base_video: str):
    config = SamplerConfig()
    sampler = VideoSampler(config)
    samples = 0
    for res in sampler.sample(base_video):
        verify_frame_res(res)
        samples += len(res)
    assert samples > 1, f"Expected more than 1 sample, got {samples}"
    stats = sampler.stats
    assert stats["produced"] == stats["gated"]


def test_clip_gating(base_video: str):
    config = SamplerConfig()
    config.gate_config = dict(
        type="clip",
        pos_samples=["a cat", "an animal"],
        neg_samples=["an empty background", "a forst with no animals"],
        model_name="ViT-B-32",
        batch_size=32,
        pos_margin=0.2,
        neg_margin=0.3,
    )
    sampler = VideoSampler(config)
    samples = 0
    for res in sampler.sample(base_video):
        verify_frame_res(res)
        samples += len(res)
    assert samples > 1, f"Expected more than 1 sample, got {samples}"


def test_grid(base_video: str):
    config = SamplerConfig()
    config.buffer_config = dict(
        type="grid", hash_size=4, size=30, max_hits=2, grid_x=4, grid_y=4, debug=True
    )
    sampler = VideoSampler(config)
    samples = 0
    for res in sampler.sample(base_video):
        verify_frame_res(res)
        samples += len(res)
    assert samples > 1, f"Expected more than 1 sample, got {samples}"
