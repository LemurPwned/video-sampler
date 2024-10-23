import base64
import os
import time
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from queue import Queue
from threading import Thread

import av
import av.enum
from PIL import Image

from .buffer import create_buffer
from .config import SamplerConfig
from .gating import create_gate
from .language.keyword_capture import create_extractor, subtitle_line
from .logging import Color, console
from .schemas import PROCESSING_DONE_ITERABLE, FrameObject


class VideoSampler:
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
        self.cfg: SamplerConfig = deepcopy(cfg)
        self.frame_buffer = create_buffer(self.cfg.buffer_config)
        self.gate = create_gate(self.cfg.gate_config)
        self.stats = Counter()

    def flush_buffer(self):
        """Flushes the frame buffer and yields gated frames"""
        for res in self.frame_buffer.final_flush():
            if res:
                self.stats["produced"] += 1
                gated_obj = self.gate(*res)
                self.stats["gated"] += gated_obj.N
                if gated_obj.frames:
                    yield gated_obj.frames
        gated_obj = self.gate.flush()
        self.stats["gated"] += gated_obj.N
        if gated_obj.frames:
            yield gated_obj.frames
        yield PROCESSING_DONE_ITERABLE

    def _init_sampler(self):
        self.stats.clear()
        self.frame_buffer.clear()

    def _process_frame(self, frame_indx, frame, ftime):
        frame_pil: Image = frame.to_image()
        if self.cfg.debug:
            buf = self.frame_buffer.get_buffer_state()
            console.print(
                f"Frame {frame_indx}\ttime: {ftime}",
                f"\t Buffer ({len(buf)}): {buf}",
                style=f"bold {Color.green.value}",
            )
        frame_meta = {"frame_time": ftime, "frame_indx": frame_indx}
        self.stats["decoded"] += 1
        if res := self.frame_buffer.add(
            frame_pil,
            metadata=frame_meta,
        ):
            gated_obj = self.gate(*res)
            self.stats["produced"] += 1
            self.stats["gated"] += gated_obj.N
            if gated_obj.frames:
                yield gated_obj.frames

    def sample(self, video_path: str, subs: str = None) -> Iterable[list[FrameObject]]:
        """Generate sample frames from a video.

        Args:
            video_path (str): The path to the video file.
            subs (str): Unused in video sampler

        Yields:
            Iterable[list[FrameObject]]: A generator that yields a list of FrameObjects representing sampled frames.
        """
        self._init_sampler()
        with av.open(video_path) as container:
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
            avg_fps = float(stream.average_rate)
            for frame in container.decode(stream):
                if frame is None or frame.is_corrupt:
                    continue
                try:
                    ftime = frame.time
                except AttributeError:
                    continue
                if self.cfg.start_time_s > 0 and ftime < self.cfg.start_time_s:
                    continue

                if self.cfg.end_time_s is not None and ftime > self.cfg.end_time_s:
                    break
                frame_index = int(ftime * avg_fps)
                # skip frames if keyframes_only is True
                time_diff = ftime - prev_time
                self.stats["total"] += 1
                if time_diff < self.cfg.min_frame_interval_sec:
                    continue
                prev_time = ftime

                yield from self._process_frame(frame_index, frame, ftime)
        # flush buffer
        yield from self.flush_buffer()

    def write_queue(self, video_path: str, q: Queue, subs: str = None):
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

    def sample(self, video_path: str, subs: str = None) -> Iterable[list[FrameObject]]:
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
        self._init_sampler()
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

                yield from self._process_frame(frame_indx, frame, ftime)
        # flush buffer
        yield from self.flush_buffer()

    def write_queue(self, video_path: str, q: Queue, subs: str = None):
        super().write_queue(video_path, q, subs=subs)


class Worker:
    def __init__(
        self,
        cfg: SamplerConfig,
        devnull: bool = False,
        sampler_cls: VideoSampler = VideoSampler,
        extra_sampler_args: dict = None,
    ) -> None:
        if extra_sampler_args is None:
            extra_sampler_args = {}
        self.cfg: SamplerConfig = cfg
        self.sampler: VideoSampler = sampler_cls(cfg=cfg, **extra_sampler_args)
        self.q = Queue()
        self.devnull = devnull
        self.__initialise_summary_objs()

    def __initialise_summary_objs(self):
        self.pool = None
        self.futures = {}
        if self.cfg.summary_config:
            from concurrent.futures import ThreadPoolExecutor

            from .integrations.llava_chat import ImageDescriptionDefault

            console.print("Initialising summary pool...", style="bold yellow")
            self.pool = ThreadPoolExecutor(
                max_workers=self.cfg.summary_config.get("max_workers", 2)
            )
            self.desc_client = ImageDescriptionDefault(
                url=self.cfg.summary_config.get("url")
            )

    def collect_summaries(self, savepath: str):
        if not self.pool:
            return
        console.print(
            f"Waiting for summary pool to finish: [{len(self.futures)}] items...",
            style="bold yellow",
        )
        summary_info = []
        for k, v in self.futures.items():
            if summary := v.result():
                summary_info.append({"time": k, "summary": summary})
                if self.cfg.debug:
                    console.print(
                        f"Summary for frame {k}",
                        f"\t{summary}",
                        style="bold green",
                    )
        import json

        # save as a jsonl
        try:
            with open(os.path.join(savepath, "summaries.jsonl"), "w") as f:
                for item in summary_info:
                    f.write(json.dumps(item) + "\n")
        except OSError as e:
            console.print(f"Failed to write to file: {e}", style="bold red")

    def launch(
        self,
        video_path: str,
        output_path: str = "",
        pretty_video_name: str = "",
        subs: str = None,
    ) -> None:
        """
        Launch the worker.

        Args:
            video_path (str): Path to the video file.
            output_path (str, optional): Path to the output folder. Defaults to "".
            pretty_video_name (str, optional): Name of the video file for pretty printing (useful for urls).
                                                Defaults to "".
        """
        if not pretty_video_name:
            pretty_video_name = os.path.basename(video_path)
        if output_path and self.devnull:
            raise ValueError("Cannot write to disk when devnull is True")
        if output_path:
            os.makedirs(output_path, exist_ok=True)

        proc_thread = Thread(
            target=self.sampler.write_queue, args=(video_path, self.q, subs)
        )
        proc_thread.start()
        self.queue_reader(output_path, read_interval=self.cfg.queue_wait)
        proc_thread.join()
        self.collect_summaries(output_path)
        if self.cfg.print_stats:
            console.print(
                f"Stats for: {pretty_video_name}",
                f"\n\tTotal frames: {self.sampler.stats['total']}",
                f"\n\tDecoded frames: {self.sampler.stats['decoded']}",
                f"\n\tProduced frames: {self.sampler.stats['produced']}",
                f"\n\tGated frames: {self.sampler.stats['gated']}",
                style=f"bold {Color.magenta.value}",
            )

    def format_output_path(self, output_path: str, frame_time: float) -> str:
        """Format the output path for a frame."""
        ft = str(frame_time)
        if self.cfg.save_format.encode_time_b64:
            ft = base64.encodebytes(ft.encode()).decode().rstrip()
            ft = f"TIMEB64_{ft}"
        elif self.cfg.save_format.avoid_dot:
            ft = ft.replace(".", "_")
            ft = f"TIMESEC_{ft}"
        if self.cfg.save_format.include_filename:
            vbsn = os.path.basename(output_path)
            # remove extension
            vbsn = os.path.splitext(vbsn)[0]
            ft = f"{vbsn}_{ft}"
        return os.path.join(output_path, f"{ft}.jpg")

    def queue_reader(self, output_path, read_interval=0.1) -> None:
        """
        Reads frames from the queue and saves them as JPEG images.

        Args:
            output_path (str): The directory path where the frames will be saved.
            read_interval (float, optional): The time interval between reading frames from the queue.
                    Defaults to 0.1 seconds.
        """
        last_summary_time = -10
        self.futures = {}  # clear futures
        while True:
            if not self.q.empty():
                frame_object: FrameObject
                for frame_object in self.q.get():
                    if frame_object.metadata.get("end", False):
                        return
                    if frame_object.frame is not None and (
                        not self.devnull and isinstance(frame_object.frame, Image.Image)
                    ):
                        frame_object.frame.save(
                            self.format_output_path(
                                output_path, frame_object.metadata["frame_time"]
                            )
                        )
                        if self.pool:
                            ftime = frame_object.metadata["frame_time"]
                            if ftime - last_summary_time < self.cfg.summary_config.get(
                                "min_sum_interval", 30
                            ):  # seconds
                                continue

                            future = self.pool.submit(
                                self.desc_client.summarise_image, frame_object.frame
                            )
                            if self.cfg.debug:
                                console.print(
                                    f"Submitting summary for frame {ftime}",
                                    style="bold yellow",
                                )
                            self.futures[ftime] = future
                            last_summary_time = ftime
            time.sleep(read_interval)
