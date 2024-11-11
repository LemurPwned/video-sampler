import glob
import os
import re
from collections.abc import Iterable
from queue import Queue

from PIL import Image

from ..config import ImageSamplerConfig
from ..logging import Color, console
from ..schemas import PROCESSING_DONE_ITERABLE, FrameObject
from .base_sampler import BaseSampler


class ImageSampler(BaseSampler):
    """
    Image sampler -- sample frames from a folder of images

    Args:
        cfg (ImageSamplerConfig): Image sampler config

    Methods:
        sample(image_folder: str) -> Iterable[list[FrameObject]]: Sample frames from image folder
        write_queue(image_path: str, q: Queue, _: str = None): Write frames to queue
    """

    def __init__(self, cfg: ImageSamplerConfig):
        super().__init__(cfg)
        self.rgx = None
        if cfg.frame_time_regex:
            console.print(
                f"Using frame time regex: {cfg.frame_time_regex}", style="bold yellow"
            )
            self.rgx = re.compile(cfg.frame_time_regex)

    def extract_frame_time(self, image_path: str, default: str | None = None) -> str:
        """
        Extract frame time from image path
        Args:
            image_path (str): Path to image
            default (str | None): Default frame time to return if no regex is set

        Returns:
            str: Frame time
        """
        if self.rgx:
            if match := self.rgx.search(image_path):
                return float(match.group(1))
            else:
                console.print(
                    f"No frame time found in {image_path} with regex {self.rgx}",
                    style="bold red",
                )
        if default is None:
            raise ValueError(
                f"Frame time regex is not set, can't extract frame name from {image_path}"
            )
        return default

    def sample(self, image_folder: str) -> Iterable[list[FrameObject]]:
        """
        Sample frames from image folder
        Args:
            image_folder (str): Path to image folder or glob pattern

        Returns:
            Iterable[list[FrameObject]]: Iterable of frames
        """
        self.init_sampler()
        if "*" in image_folder:
            image_paths = glob.iglob(image_folder)
        else:
            # iterable over all files in image_folder
            image_paths = (
                os.path.join(image_folder, f) for f in os.listdir(image_folder)
            )
        generator = sorted(
            image_paths,
            key=lambda x: self.extract_frame_time(os.path.basename(x), default=x),
        )
        for frame_indx, image_path in enumerate(generator):
            frame = Image.open(image_path)
            yield from self.process_frame(
                frame_indx=frame_indx,
                frame=frame,
                ftime=self.extract_frame_time(os.path.basename(image_path), frame_indx),
            )

        yield from self.flush_buffer()

    def write_queue(self, image_path: str, q: Queue, _: str = None):
        """
        Write frames to queue.
        Args:
            image_path (str): Path to image
            q (Queue): Queue to write frames to
        """
        try:
            for item in self.sample(image_path):
                q.put(item)
        except Exception as e:
            console.print(
                f"Error while processing {image_path}",
                f"\n\t{e}",
                style=f"bold {Color.red.value}",
            )
            q.put(PROCESSING_DONE_ITERABLE)
