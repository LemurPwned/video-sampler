import glob
import os
from collections.abc import Generator, Iterable

from tqdm import tqdm

from .logging import Color, console
from .sampler import SamplerConfig, Worker
from .utils import slugify


def local_path_iterable(
    video_iterable: Iterable[str], output_path: str, worker: Worker
):
    for video in tqdm(video_iterable, desc="Processing videos..."):
        video_subpath = os.path.join(output_path, os.path.basename(video))
        worker.launch(
            video_path=video,
            output_path=video_subpath,
        )


def url_iterable(
    video_iterable: Iterable[str], output_path: str, worker: Worker
) -> None:
    for video_title, video_url, _ in tqdm(video_iterable, desc="Processing urls..."):
        video_filename = slugify(video_title)
        video_subpath = os.path.join(output_path, video_filename)
        worker.launch(
            video_path=video_url,
            output_path=video_subpath,
            pretty_video_name=video_filename,
        )


def delegate_workers(video_path: str | Generator, output_path: str, cfg: SamplerConfig):
    msg = "Detected input as a file"
    processor = local_path_iterable
    if isinstance(video_path, Generator):
        videos = video_path
        msg = "Detected input as an URL generator"
        processor = url_iterable
    elif not os.path.isfile(video_path):
        if "*" not in video_path:
            videos = glob.glob(os.path.join(video_path, "*"))
        else:
            videos = glob.glob(video_path)
        msg = f"Detected input as a folder with {len(videos)} files"
    else:
        videos = iter([video_path])
    console.print(msg, style=f"bold {Color.cyan.value}")

    worker = Worker(
        cfg=cfg,
    )
    processor(videos, output_path, worker)
    console.print("All videos processed", style=f"bold {Color.green.value}")
