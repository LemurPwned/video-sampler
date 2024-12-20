from collections.abc import Iterable
from queue import Queue

import av

from ..config import SamplerConfig
from ..language.keyword_capture import create_extractor, subtitle_line
from ..logging import Color, console
from ..schemas import PROCESSING_DONE_ITERABLE, FrameObject
from .base_sampler import BaseSampler


class VideoSampler(BaseSampler):
    """
    The fundamental class for sampling video frames.

    Args:
        cfg (SamplerConfig): The configuration for the video sampler.

    Attributes:
        cfg (SamplerConfig): The configuration for the video sampler.
        frame_buffer (FrameBuffer): The frame buffer used for sampling frames.
        gate (Gate): The gate used for filtering frames.
        stats (Counter): A counter for tracking statistics.

    Methods:
        sample(video_path) -> Iterable[list[FrameObject]]:
            Generates sample frames from a video.
        write_queue(video_path, q):
            Writes sampled frames to a queue.

    """

    def __init__(self, cfg: SamplerConfig) -> None:
        super().__init__(cfg)

    def sample(
        self, video_path: str, subs: str | None = None
    ) -> Iterable[list[FrameObject]]:
        """Generate sample frames from a video.

        Args:
            video_path (str): The path to the video file.
            subs (str): Unused in video sampler

        Yields:
            Iterable[list[FrameObject]]: A generator that yields a list
                    of FrameObjects representing sampled frames.
        """
        self.init_sampler()
        with av.open(
            video_path,
            metadata_errors="ignore",
        ) as container:
            stream = container.streams.video[0]
            if self.cfg.keyframes_only:
                stream.codec_context.skip_frame = "NONKEY"
            prev_time = -10
            if self.cfg.start_time_s > 0 and self.cfg.precise_seek is False:
                """
                This is an inaccurate seek, so you may get frames before
                the start time or much later.
                """
                frame_container_pts = (
                    int(self.cfg.start_time_s / float(stream.time_base))
                    + stream.start_time
                )
                console.print(
                    f"Seeking to {frame_container_pts} pts based of {stream.time_base} "
                    f"with inbuilt video offset {stream.start_time}",
                    style=f"bold {Color.yellow.value}",
                )
                try:
                    container.seek(
                        frame_container_pts,
                        backward=True,
                        any_frame=False,
                        stream=stream,
                    )
                except av.AVError as e:
                    console.print(
                        f"Error seeking to {frame_container_pts} pts. Will default to precise seek.",
                        f"\n\t{e}",
                        style=f"bold {Color.red.value}",
                    )
            try:
                avg_fps = float(stream.average_rate)
            except (AttributeError, TypeError):
                console.print(
                    "Failed to get average FPS, defaulting to 1. If you are using a URL handle, this is expected.",
                    style=f"bold {Color.yellow.value}",
                )
                avg_fps = 1
            for frame in container.decode(stream):
                if frame is None or frame.is_corrupt:
                    self.debug_print("Frame is None or corrupt, skipping.")
                    continue
                try:
                    ftime = frame.time
                except AttributeError:
                    self.debug_print("Failed to get frame time, skipping frame.")
                    continue
                if self.cfg.start_time_s > 0 and ftime < self.cfg.start_time_s:
                    self.debug_print(
                        f"Frame time {ftime} is before start time {self.cfg.start_time_s}, skipping."
                    )
                    continue

                if self.cfg.end_time_s is not None and ftime > self.cfg.end_time_s:
                    self.debug_print(
                        f"Frame time {ftime} is after end time {self.cfg.end_time_s}, stopping."
                    )
                    break
                frame_index = int(ftime * avg_fps)
                # skip frames if keyframes_only is True
                time_diff = ftime - prev_time
                self.stats["total"] += 1
                if time_diff < self.cfg.min_frame_interval_sec:
                    self.debug_print(
                        f"Frame time {ftime} is too close to previous frame {prev_time}, skipping."
                    )
                    continue
                prev_time = ftime
                frame_pil = frame.to_image()
                yield from self.process_frame(frame_index, frame_pil, ftime)
        # flush buffer
        yield from self.flush_buffer()

    def write_queue(self, video_path: str, q: Queue, subs: str | None = None) -> None:
        try:
            item: tuple[FrameObject, int]
            for item in self.sample(video_path=video_path, subs=subs):
                q.put(item)
        except (av.IsADirectoryError, av.InvalidDataError) as e:
            console.print(
                f"Error while processing {video_path}",
                f"\n\t{e}",
                style=f"bold {Color.red.value}",
            )
            q.put(PROCESSING_DONE_ITERABLE)


class SegmentSampler(VideoSampler):
    """
    A class for sampling video frames based on subtitle segments.

    Args:
        cfg (SamplerConfig): The configuration for the video sampler.
        segment_generator (Iterable[subtitle_line]): An iterable of subtitle segments.

    Methods:
        sample(video_path) -> Iterable[list[FrameObject]]:
            Generates sample frames from a video.
        write_queue(video_path, q):
            Writes sampled frames to a queue.
    """

    def __init__(self, cfg: SamplerConfig) -> None:
        super().__init__(cfg)
        self.extractor = create_extractor(cfg.extractor_config)

    def sample(
        self, video_path: str, subs: str | None = None
    ) -> Iterable[list[FrameObject]]:
        """Generate sample frames from a video.

        Args:
            video_path (str): The path to the video file.
            subs (str): Subtitles for the video file.

        Yields:
            Iterable[list[FrameObject]]: A generator that yields a list of FrameObjects representing sampled frames.
        """
        segment_generator: Iterable[subtitle_line] = self.extractor.generate_segments(
            subs
        )
        self.init_sampler()
        next_segment = next(segment_generator)
        segment_boundary_end_sec = next_segment.end_time / 1000
        segment_boundary_start_sec = next_segment.start_time / 1000
        absolute_stop = False
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if self.cfg.keyframes_only:
                stream.codec_context.skip_frame = "NONKEY"
            prev_time = -10
            for frame_indx, frame in enumerate(container.decode(stream)):
                if frame is None:
                    continue
                try:
                    ftime = frame.time
                except AttributeError:
                    continue
                reiters = 0
                # find the next segment that starts after the current frame
                while ftime > segment_boundary_end_sec:
                    console.print(
                        f"Seeking to next segment: {segment_boundary_end_sec}/{ftime}",
                        style=f"bold {Color.yellow.value}",
                    )
                    try:
                        next_segment = next(segment_generator)
                        reiters += 1
                        segment_boundary_end_sec = next_segment.end_time / 1000
                        segment_boundary_start_sec = next_segment.start_time / 1000
                    except StopIteration:
                        absolute_stop = True
                        break
                if reiters > 0:
                    console.print(
                        f"Skipped {reiters} segments!",
                        style=f"bold {Color.red.value}",
                    )
                if absolute_stop:
                    break
                # we haven't found the next segment yet
                # the other condition, is where we are after the segment
                # but this is handled by the while loop above
                if ftime <= segment_boundary_start_sec:
                    continue

                self.stats["total"] += 1
                time_diff = ftime - prev_time
                if time_diff < self.cfg.min_frame_interval_sec:
                    continue
                prev_time = ftime
                frame_pil = frame.to_image()
                yield from self.process_frame(frame_indx, frame_pil, ftime)
        # flush buffer
        yield from self.flush_buffer()

    def write_queue(self, video_path: str, q: Queue, subs: str | None = None):
        super().write_queue(video_path, q, subs=subs)
