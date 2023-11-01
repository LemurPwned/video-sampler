from video_sampler.sampler import SamplerConfig, VideoSampler
from PIL import Image
from itertools import chain
    


def test_pass_gating(base_video: str):
    config = SamplerConfig()
    sampler = VideoSampler(config)
    for res in chain(sampler.sample(base_video)):
        for frame, meta in res:
            if frame: 
                assert isinstance(frame, Image.Image)
                assert isinstance(meta, dict)
                assert "frame_time" in meta

    stats = sampler.stats
    assert stats["produced"] == stats['gated']

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
    for res in chain(sampler.sample(base_video)):
        for frame, meta in res:
            if frame: 
                assert isinstance(frame, Image.Image)
                assert isinstance(meta, dict)
                assert "frame_time" in meta
