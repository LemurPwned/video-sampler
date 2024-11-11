import os
import tempfile
from queue import Queue
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from video_sampler.config import SamplerConfig, SaveFormatConfig
from video_sampler.schemas import FrameObject
from video_sampler.worker import Worker


@pytest.fixture
def sample_config():
    return SamplerConfig(
        fps=1,
        queue_wait=0.1,
        save_format=SaveFormatConfig(),
        summary_config={},
        debug=False,
        print_stats=False,
    )


@pytest.fixture
def worker(sample_config):
    return Worker(cfg=sample_config)


def test_worker_initialization(worker):
    assert isinstance(worker.q, Queue)
    assert worker.devnull is False
    assert worker.pool is None


def test_format_output_path(worker, base_video):
    output_path = str(base_video)
    frame_time = 10.5
    expected_path = os.path.join(output_path, "10.5.jpg")
    assert worker.format_output_path(output_path, frame_time) == expected_path


def test_format_output_path_with_avoid_dot(worker, base_video):
    worker.cfg.save_format.avoid_dot = True
    output_path = str(base_video)
    frame_time = 10.5
    expected_path = os.path.join(output_path, "TIMESEC_10_5.jpg")
    assert worker.format_output_path(output_path, frame_time) == expected_path


def test_format_output_path_with_encode_time_b64(worker, base_video):
    worker.cfg.save_format.encode_time_b64 = True
    output_path = str(base_video)
    frame_time = 10.5
    expected_path = os.path.join(output_path, "TIMEB64_MTAuNQ==.jpg")
    assert worker.format_output_path(output_path, frame_time) == expected_path


@patch("video_sampler.worker.Thread")
@patch("video_sampler.worker.VideoSampler")
def test_launch(mock_video_sampler, mock_thread, worker, base_video):
    # mock video sampler is needed because we check the calls
    with tempfile.TemporaryDirectory() as output_path:
        worker.queue_reader = Mock()
        worker.collect_summaries = Mock()

        worker.launch(base_video, output_path)

        mock_thread.assert_called_once()
        worker.queue_reader.assert_called_once_with(
            output_path, read_interval=worker.cfg.queue_wait
        )
        worker.collect_summaries.assert_called_once_with(output_path)


@pytest.mark.parametrize(
    "devnull,output_path,expected",
    [
        (True, "some_path", ValueError),
        (True, "", None),
        (False, "", None),
        (False, "some_path", None),
    ],
)
def test_launch_devnull_output_path_combinations(
    base_video, worker, devnull, output_path, expected
):
    worker.devnull = devnull
    worker.queue_reader = Mock()
    worker.collect_summaries = Mock()

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            worker.launch(base_video, output_path)
    else:
        worker.launch(base_video, output_path)


def test_queue_reader(worker):
    mock_image = Mock(spec=Image.Image)
    frame_object = FrameObject(frame=mock_image, metadata={"frame_time": 1.0})
    end_object = FrameObject(frame=None, metadata={"end": True})

    worker.q.put([frame_object, end_object])
    with tempfile.TemporaryDirectory() as temp_dir:
        worker.queue_reader(temp_dir, read_interval=0.01)

        expected_save_path = worker.format_output_path(temp_dir, 1.0)
        mock_image.save.assert_called_once_with(expected_save_path)


def test_queue_reader_parallel(worker):
    import threading

    mock_images = [Mock(spec=Image.Image) for _ in range(3)]
    frame_objects = [
        FrameObject(frame=img, metadata={"frame_time": i})
        for i, img in enumerate(mock_images)
    ]
    end_object = FrameObject(frame=None, metadata={"end": True})

    with tempfile.TemporaryDirectory() as temp_dir:
        threading.Thread(
            target=lambda: [worker.q.put([fo]) for fo in frame_objects + [end_object]]
        ).start()
        worker.queue_reader(temp_dir, read_interval=0.01)

        for i, mock_image in enumerate(mock_images):
            mock_image.save.assert_called_once_with(
                worker.format_output_path(temp_dir, int(i))
            )
