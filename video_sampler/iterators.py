import glob
import os
import warnings
from collections.abc import Generator, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from .config import SamplerConfig
from .logging import Color, console
from .sampler import VideoSampler, Worker
from .utils import slugify


def process_video(
    video_info: str | tuple[str, str, str | None],
    output_path: str,
    is_url: bool,
    worker_cfg: SamplerConfig,
    sampler_cls: VideoSampler = VideoSampler,
):
    """Process a video file or URL.

    Args:
        video_info (Union[str, tuple]): A video file path or a tuple containing the video title,
            URL and subtitles.
        output_path (str): Path to the output folder.
        worker (Worker): Worker instance to process the videos.
        is_url (bool): Flag to indicate if the video is a URL.
    """
    worker = Worker(cfg=worker_cfg, sampler_cls=sampler_cls)
    try:
        if is_url:
            video_title, video_url, subs = video_info
            video_filename = slugify(video_title)
            video_subpath = os.path.join(output_path, video_filename)
            worker.launch(
                video_path=video_url,
                output_path=video_subpath,
                pretty_video_name=video_filename,
                subs=subs,
            )
        else:
            video_subpath = os.path.join(output_path, os.path.basename(video_info))
            worker.launch(
                video_path=video_info,
                output_path=video_subpath,
            )
    except Exception as e:
        console.print(
            f"Error processing video {video_info}: {e}", style=f"bold {Color.red.value}"
        )


def parallel_video_processing(
    video_iterable: Iterable[str | tuple],
    output_path: str,
    is_url: bool,
    worker_cfg: SamplerConfig,
    sampler_cls=VideoSampler,
    n_workers: int = None,
):  # sourcery skip: for-append-to-extend
    """Process a list of local video files or video URLs in parallel.

    Args:
        video_iterable (Iterable[Union[str, tuple]]): An iterable of video file paths or video URLs.
        output_path (str): Path to the output folder.
        is_url (bool): Flag to indicate if the video is a URL.
        worker_cfg (SamplerConfig): Configuration for the worker.
        n_workers (int): Number of workers to use.
    """
    if n_workers == -1:
        n_workers = None
    if n_workers is not None and n_workers == 1:
        for video in tqdm(video_iterable, desc="Processing videos..."):
            process_video(
                video,
                output_path,
                worker_cfg=worker_cfg,
                is_url=is_url,
                sampler_cls=sampler_cls,
            )
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            console.print(
                f"Using {executor._max_workers} workers",
                style=f"bold {Color.green.value}",
            )
            executor._max_workers
            for video in video_iterable:
                futures.append(
                    executor.submit(
                        process_video,
                        video,
                        output_path,
                        is_url=is_url,
                        worker_cfg=worker_cfg,
                        sampler_cls=sampler_cls,
                    )
                )
            for future in tqdm(as_completed(futures), desc="Processing videos..."):
                future.result()


def delegate_workers(
    video_path: str | Generator,
    output_path: str,
    cfg: SamplerConfig,
    sampler_cls: VideoSampler = VideoSampler,
):
    """Delegate the processing of a list of videos to a worker instance.

    Args:
        video_path (str | Generator): Path to a video file, a generator of URLs or a list of video files.
        output_path (str): Path to the output folder.
        cfg (SamplerConfig): Configuration for the worker.
    """
    msg = "Detected input as a file"
    is_url = False
    if isinstance(video_path, Generator):
        videos = video_path
        msg = "Detected input as an URL generator"
        is_url = True
    elif not os.path.isfile(video_path):
        if "*" not in video_path:
            videos = glob.glob(os.path.join(video_path, "*"))
        else:
            videos = glob.glob(video_path)
        msg = f"Detected input as a folder with {len(videos)} files"
    else:
        videos = iter([video_path])
    console.print(msg, style=f"bold {Color.cyan.value}")
    if sampler_cls is None:
        warnings.warn(
            "Sampler class was not specified, defaulting to Video Sampler", stacklevel=2
        )
        sampler_cls = VideoSampler
    parallel_video_processing(
        videos,
        output_path,
        is_url=is_url,
        n_workers=cfg.n_workers,
        sampler_cls=sampler_cls,
        worker_cfg=cfg,
    )
    console.print("All videos processed", style=f"bold {Color.green.value}")
