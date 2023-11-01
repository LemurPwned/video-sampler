import os
import time
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from queue import Queue
from threading import Thread

import av
from PIL import Image

from .buffer import SamplerConfig, create_buffer
from .gating import create_gate
from .logging import Color, console


class VideoSampler:
    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = deepcopy(cfg)
        self.frame_buffer = create_buffer(self.cfg.buffer_config)
        self.gate = create_gate(self.cfg.gate_config)
        self.stats = Counter()

    def sample(self, video_path: str) -> Iterable[tuple[Image.Image | None, dict]]:
        """Generate sample frames from a video"""
        self.stats.clear()
        self.frame_buffer.clear()
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if self.cfg.keyframes_only:
                stream.codec_context.skip_frame = "NONKEY"
            prev_time = -10
            for frame_indx, frame in enumerate(container.decode(stream)):
                # skip frames if keyframes_only is True
                time_diff = frame.time - prev_time
                self.stats["total"] += 1
                if time_diff < self.cfg.min_frame_interval_sec:
                    continue
                prev_time = frame.time

                frame_pil: Image = frame.to_image()
                if self.cfg.debug:
                    buf = self.frame_buffer.get_buffer_state()
                    console.print(
                        f"Frame {frame_indx}\ttime: {frame.time}",
                        f"\t Buffer ({len(buf)}): {buf}",
                        style=f"bold {Color.green.value}",
                    )
                frame_meta = {"frame_time": frame.time, "frame_indx": frame_indx}
                res = self.frame_buffer.add(
                    frame_pil,
                    metadata=frame_meta,
                )
                self.stats["decoded"] += 1
                if res:
                    self.stats["produced"] += 1
                    gated, N = self.gate(*res)
                    self.stats["gated"] += N
                    if gated:
                        yield gated

        # flush buffer
        for res in self.frame_buffer.final_flush():
            if res:
                self.stats["produced"] += 1
                gated, N = self.gate(*res)
                self.stats["gated"] += N
                if gated:
                    yield gated
        gated, N = self.gate.flush()
        self.stats["gated"] += N
        if gated:
            yield gated
        yield ((None, {"end": True}),)

    def write_queue(self, video_path: str, q: Queue):
        try:
            for item in self.sample(video_path=video_path):
                q.put(item)
        except (av.IsADirectoryError, av.InvalidDataError) as e:
            console.print(
                f"Error while processing {video_path}",
                f"\n\t{e}",
                style=f"bold {Color.red.value}",
            )
            q.put(((None, {"end": True}),))


class Worker:
    def __init__(self, cfg: SamplerConfig, devnull: bool = False) -> None:
        self.cfg = cfg
        self.processor = VideoSampler(cfg=cfg)
        self.q = Queue()
        self.devnull = devnull

    def launch(self, video_path: str, output_path: str = "") -> None:
        if output_path and self.devnull:
            raise ValueError("Cannot write to disk when devnull is True")
        if output_path:
            os.makedirs(output_path, exist_ok=True)
        proc_thread = Thread(
            target=self.processor.write_queue, args=(video_path, self.q)
        )
        proc_thread.start()
        self.queue_reader(output_path, read_interval=self.cfg.queue_wait)
        proc_thread.join()
        if self.cfg.print_stats:
            console.print(
                f"Stats for: {os.path.basename(video_path)}",
                f"\n\tTotal frames: {self.processor.stats['total']}",
                f"\n\tDecoded frames: {self.processor.stats['decoded']}",
                f"\n\tProduced frames: {self.processor.stats['produced']}",
                f"\n\tGated frames: {self.processor.stats['gated']}",
                style=f"bold {Color.magenta.value}",
            )

    def queue_reader(self, output_path, read_interval=0.1) -> None:
        while True:
            if not self.q.empty():
                for frame, metadata in self.q.get():
                    if metadata.get("end", False):
                        return
                    if frame is not None and (
                        not self.devnull and isinstance(frame, Image.Image)
                    ):
                        frame.save(
                            os.path.join(output_path, f"{metadata['frame_time']}.jpg")
                        )
            time.sleep(read_interval)
