import os
import random
import tempfile
from pathlib import Path

import av
import pytest
from PIL import Image

from video_sampler.config import ImageSamplerConfig, SamplerConfig
from video_sampler.language.keyword_capture import parse_srt_subtitle
from video_sampler.samplers import ImageSampler, SegmentSampler, VideoSampler
from video_sampler.schemas import FrameObject


# Helper function to create a temporary video file
def create_temp_video(duration=5):
    import av

    # set seed for
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        container = av.open(temp_video.name, mode="w")
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 480
        stream.height = 320
        stream.pix_fmt = "yuv420p"
        random.seed(42)
        for _ in range(duration * 30):
            img = Image.new(
                "RGB",
                (480, 320),
                color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            )

            frame = av.VideoFrame.from_image(img)
            packet = stream.encode(frame)
            container.mux(packet)

        # Flush stream
        packet = stream.encode()
        container.mux(packet)

        # Close the container
        container.close()

    return temp_video.name


# Helper function to create temporary image files
def create_temp_images(num_images=5):
    temp_dir = tempfile.mkdtemp()
    for i in range(num_images):
        img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
        img.save(os.path.join(temp_dir, f"image_{i}.jpg"))
    return temp_dir


@pytest.fixture
def temp_video():
    video_path = create_temp_video()
    yield video_path
    os.unlink(video_path)


@pytest.fixture
def temp_images():
    image_dir = create_temp_images()
    yield image_dir
    for file in Path(image_dir).glob("*"):
        os.unlink(file)
    os.rmdir(image_dir)


def test_video_sampler_fine_grained(temp_video):
    config = SamplerConfig(
        min_frame_interval_sec=0.5,
        keyframes_only=False,
        buffer_config={"type": "hash", "hash_size": 8, "size": 5},
    )
    sampler = VideoSampler(config)

    frames_list = list(sampler.sample(temp_video))
    frame_times = [
        frame.metadata["frame_time"]
        for batch in frames_list
        for frame in batch
        if frame.frame
    ]

    assert all(
        isinstance(frame.frame, Image.Image)
        for batch in frames_list
        for frame in batch
        if frame.frame
    )
    assert all(t2 - t1 >= 0.5 for t1, t2 in zip(frame_times, frame_times[1:]))
    assert len(frame_times) >= len(frames_list) // 2, "Too few frames sampled"


def test_video_sampler(temp_video):
    config = SamplerConfig(
        min_frame_interval_sec=0.5,
        keyframes_only=False,
        buffer_config={"type": "hash", "hash_size": 8, "size": 5},
    )
    sampler = VideoSampler(config)

    frame_count = 0
    frame_times = []
    for frames in sampler.sample(temp_video):
        frame_obj: FrameObject
        for frame_obj in frames:
            if frame_obj.frame:
                assert isinstance(
                    frame_obj.frame, Image.Image
                ), f"Expected frame to be an Image, got {type(frame_obj.frame)}"
                assert (
                    "frame_time" in frame_obj.metadata
                ), "Expected frame_time in metadata"
                frame_count += 1
                frame_times.append(frame_obj.metadata["frame_time"])
    assert frame_count > 0, "No frames were sampled"
    assert frame_times == sorted(frame_times), "Frames not in chronological order"
    assert frame_count < 150, "Too many frames were sampled"


def test_image_sampler(temp_images):
    config = ImageSamplerConfig(
        min_frame_interval_sec=0,
        buffer_config={"type": "hash", "hash_size": 8, "size": 5},
        frame_time_regex=r"image_(\d+)",
    )
    sampler = ImageSampler(config)

    frame_count = 0
    for frames in sampler.sample(temp_images):
        for frame_obj in frames:
            if frame_obj.frame:
                assert isinstance(frame_obj.frame, Image.Image)
                assert "frame_time" in frame_obj.metadata
                frame_count += 1

    assert frame_count > 0, "No frames were sampled"


def test_segment_sampler(temp_video):
    config = SamplerConfig(
        min_frame_interval_sec=0.5,
        keyframes_only=False,
        buffer_config={"type": "hash", "hash_size": 8, "size": 5},
        extractor_config={"type": "keyword", "args": {"keywords": ["test"]}},
    )
    sampler = SegmentSampler(config)

    # Create a simple subtitle file
    subs = """1
00:00:00,000 --> 00:00:02,000
This is a test subtitle

2
00:00:02,000 --> 00:00:04,000
Another test subtitle
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False
    ) as temp_subs:
        temp_subs.write(subs)
        temp_subs.flush()

    frame_count = 0
    subtitle_list = parse_srt_subtitle(subs)
    for frames in sampler.sample(temp_video, subs=subtitle_list):
        for frame_obj in frames:
            if frame_obj.frame:
                assert isinstance(frame_obj.frame, Image.Image)
                assert "frame_time" in frame_obj.metadata
                frame_count += 1

    assert frame_count > 0, "No frames were sampled"
    assert frame_count < 150, "Too many frames were sampled"

    os.unlink(temp_subs.name)


def test_sampler_config():
    config = SamplerConfig(
        min_frame_interval_sec=1.0,
        keyframes_only=True,
        buffer_config={"type": "hash", "hash_size": 8, "size": 10},
        gate_config={"type": "pass"},
    )
    assert config.min_frame_interval_sec == 1.0
    assert config.keyframes_only is True
    assert config.buffer_config["type"] == "hash"
    assert config.gate_config["type"] == "pass"


def test_image_sampler_config():
    config = ImageSamplerConfig(
        min_frame_interval_sec=0,
        buffer_config={"type": "hash", "hash_size": 8, "size": 5},
        frame_time_regex=r"image_(\d+)",
    )
    assert config.min_frame_interval_sec == 0
    assert config.buffer_config["type"] == "hash"
    assert config.frame_time_regex == r"image_(\d+)"


def test_video_sampler_invalid_file():
    config = SamplerConfig(
        min_frame_interval_sec=0.5,
        keyframes_only=False,
        buffer_config={"type": "hash", "hash_size": 8, "size": 5},
    )
    sampler = VideoSampler(config)

    with pytest.raises(FileNotFoundError):
        list(sampler.sample("non_existent_video.mp4"))


def test_video_sampler_invalid_format():
    config = SamplerConfig(
        min_frame_interval_sec=0.5,
        keyframes_only=False,
        buffer_config={"type": "hash", "hash_size": 8, "size": 5},
    )
    sampler = VideoSampler(config)

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w") as temp_file:
        temp_file.write("This is not a video file")
        temp_file.flush()

        with pytest.raises(av.error.InvalidDataError):
            list(sampler.sample(temp_file.name))


def test_image_sampler_empty_directory():
    config = ImageSamplerConfig(
        min_frame_interval_sec=0,
        buffer_config={"type": "hash", "hash_size": 8, "size": 5},
        frame_time_regex=r"image_(\d+)",
    )
    sampler = ImageSampler(config)

    with tempfile.TemporaryDirectory() as temp_dir:
        frames = []
        for frame_objs in sampler.sample(temp_dir):
            frames.extend(frame_obj for frame_obj in frame_objs if frame_obj.frame)
        assert not frames, "Expected no frames from an empty directory"


def test_image_sampler_invalid_directory():
    config = ImageSamplerConfig(
        min_frame_interval_sec=0,
        buffer_config={"type": "hash", "hash_size": 8, "size": 5},
        frame_time_regex=r"image_(\d+)",
    )
    sampler = ImageSampler(config)

    with pytest.raises(FileNotFoundError):
        list(sampler.sample("/non/existent/directory"))
