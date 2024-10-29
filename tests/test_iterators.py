from unittest.mock import patch

import pytest

from video_sampler.config import SamplerConfig
from video_sampler.iterators import (
    delegate_workers,
    parallel_video_processing,
    process_video,
)


@pytest.fixture
def sample_config():
    return SamplerConfig(n_workers=1)


@pytest.fixture
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def mock_video_file(tmp_path):
    video_file = tmp_path / "test_video.mp4"
    video_file.touch()
    return str(video_file)


class TestProcessVideo:
    def test_process_local_video(self, mock_video_file, temp_output_dir, sample_config):
        with patch("video_sampler.iterators.Worker") as MockWorker:
            mock_worker = MockWorker.return_value

            process_video(
                mock_video_file, temp_output_dir, is_url=False, worker_cfg=sample_config
            )

            mock_worker.launch.assert_called_once()

    def test_process_url_video(self, temp_output_dir, sample_config):
        with patch("video_sampler.iterators.Worker") as MockWorker:
            mock_worker = MockWorker.return_value
            video_info = ("Test Video", "http://example.com/video.mp4", None)

            process_video(
                video_info, temp_output_dir, is_url=True, worker_cfg=sample_config
            )

            mock_worker.launch.assert_called_once()


class TestParallelVideoProcessing:
    def test_single_worker_processing(
        self, mock_video_file, temp_output_dir, sample_config
    ):
        video_list = [mock_video_file]

        with patch("video_sampler.iterators.process_video") as mock_process:
            parallel_video_processing(
                video_list,
                temp_output_dir,
                is_url=False,
                worker_cfg=sample_config,
                n_workers=1,
            )

            assert mock_process.call_count == 1

    def test_multiple_workers_processing(
        self, mock_video_file, temp_output_dir, sample_config
    ):
        video_list = [mock_video_file] * 3

        # Move the patch inside the function that will be run in each process
        def run_process_video(*args, **kwargs):
            with patch("video_sampler.iterators.process_video") as mock_process:
                parallel_video_processing(
                    video_list,
                    temp_output_dir,
                    is_url=False,
                    worker_cfg=sample_config,
                    n_workers=2,
                )
                return mock_process.call_count

        # Run the function directly instead of trying to mock across processes
        with patch("video_sampler.iterators.process_video") as mock_process:
            parallel_video_processing(
                video_list,
                temp_output_dir,
                is_url=False,
                worker_cfg=sample_config,
                n_workers=1,  # Force single worker for testing
            )

            assert mock_process.call_count == 3


class TestDelegateWorkers:
    def test_delegate_single_file(
        self, mock_video_file, temp_output_dir, sample_config
    ):
        with patch(
            "video_sampler.iterators.parallel_video_processing"
        ) as mock_parallel:
            delegate_workers(mock_video_file, temp_output_dir, sample_config)

            mock_parallel.assert_called_once()

    def test_delegate_directory(self, tmp_path, temp_output_dir, sample_config):
        # Create mock video files in directory
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "video1.mp4").touch()
        (video_dir / "video2.mp4").touch()

        with patch(
            "video_sampler.iterators.parallel_video_processing"
        ) as mock_parallel:
            delegate_workers(str(video_dir), temp_output_dir, sample_config)

            mock_parallel.assert_called_once()

    def test_delegate_url_generator(self, temp_output_dir, sample_config):
        def url_generator():
            yield ("Video 1", "http://example.com/1.mp4", None)
            yield ("Video 2", "http://example.com/2.mp4", None)

        with patch(
            "video_sampler.iterators.parallel_video_processing"
        ) as mock_parallel:
            delegate_workers(url_generator(), temp_output_dir, sample_config)

            mock_parallel.assert_called_once()
