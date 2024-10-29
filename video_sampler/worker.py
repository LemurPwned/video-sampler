import base64
import json
import os
import time
from queue import Queue
from threading import Thread

from PIL import Image

from .config import ImageSamplerConfig, SamplerConfig
from .logging import Color, console
from .samplers.video_sampler import VideoSampler
from .schemas import FrameObject


class Worker:
    def __init__(
        self,
        cfg: SamplerConfig,
        devnull: bool = False,
        sampler_cls: VideoSampler = VideoSampler,
        extra_sampler_args: dict | None = None,
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
        if isinstance(self.cfg, ImageSamplerConfig):
            output_path = os.path.join(output_path, os.path.basename(video_path))
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
