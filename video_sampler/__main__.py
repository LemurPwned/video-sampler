# type: ignore[attr-defined]

import shlex
from collections.abc import Generator
from typing import Annotated

import typer

from . import version
from .buffer import check_args_validity
from .config import SamplerConfig
from .iterators import delegate_workers
from .logging import Color, console
from .sampler import SegmentSampler, VideoSampler
from .schemas import BufferType

app = typer.Typer(
    name="video-sampler",
    help="Video sampler allows you to efficiently sample video frames"
    " from a video file or a list of video files or urls.",
    add_completion=True,
)


def _ytdlp_plugin(
    yt_extra_args: str, video_path: str | Generator, get_subs: bool = False
):
    """Use yt-dlp to download videos from urls. Default is False.
    Enabling this will treat video_path as an input to ytdlp command.
    Parse the extra arguments for YouTube-DLP extraction in classic format.

    Examples:
    --------
    >>> _ytdlp_plugin("--format bestvideo+bestaudio")
    >>> _ytdlp_plugin("--datebefore 20190101")
    >>> _ytdlp_plugin('--match-filter "original_url!*=/shorts/ & url!*=/shorts/"')

    """
    # the above import will fail if yt-dlp is not installed and prints an error message
    import yt_dlp

    from video_sampler.integrations import YTDLPPlugin

    if yt_extra_args is not None:
        yt_extra_args = yt_dlp.parse_options(shlex.split(yt_extra_args)).ydl_opts
        default_opts = yt_dlp.parse_options([]).ydl_opts
        yt_extra_args = {k: v for k, v in yt_extra_args.items() if default_opts[k] != v}
    plugin = YTDLPPlugin()

    video_path = plugin.generate_urls(
        video_path, extra_info_extract_opts=yt_extra_args, get_subs=get_subs
    )
    return video_path


def _create_from_config(
    cfg: SamplerConfig,
    video_path: str | Generator,
    output_path: str,
    sampler_cls: VideoSampler = VideoSampler,
):
    # create a test buffer
    try:
        check_args_validity(cfg)
    except AssertionError as e:
        console.print(
            "Error while creating buffer",
            f"\n\t{e}",
            style=f"bold {Color.red.value}",
        )
        raise typer.Exit(code=-1) from e

    console.print(cfg, style=f"bold {Color.yellow.value}")
    delegate_workers(
        video_path=video_path,
        output_path=output_path,
        cfg=cfg,
        sampler_cls=sampler_cls,
    )


def version_callback(print_version: bool = True) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]video-sampler[/] version: [bold blue]{version}[/]")
        raise typer.Exit()


@app.command(name="hash")
def main(
    video_path: str = typer.Argument(
        ..., help="Path to the video file or a glob pattern."
    ),
    output_path: str = typer.Argument(..., help="Path to the output folder."),
    min_frame_interval_sec: float = typer.Option(
        1.0, help="Minimum frame interval in seconds."
    ),
    stats: bool = typer.Option(True, help="Print stats."),
    keyframes_only: bool = typer.Option(True, help="Only sample keyframes."),
    buffer_size: int = typer.Option(10, help="Size of the buffer."),
    hash_size: int = typer.Option(4, help="Size of the hash."),
    queue_wait: float = typer.Option(0.1, help="Time to wait for the queue."),
    start_time_s: int = typer.Option(
        0, help="The starting time for sampling in seconds."
    ),
    end_time_s: int = typer.Option(
        None, help="The ending time for sampling in seconds. None for no end."
    ),
    debug: bool = typer.Option(False, help="Enable debug mode."),
    threshold: float = typer.Option(
        20.0, help="Threshold for the blur gate. If 0 then no blur gate is used."
    ),
    blur_method: str = typer.Option(
        "fft", help="Method to use for blur gate. Can be fft or variance."
    ),
    ytdlp: bool = typer.Option(
        False,
        help="Use yt-dlp to download videos from urls. Default is False."
        " Enabling this will treat video_path as an input to ytdlp command.",
    ),
    yt_extra_args: str = typer.Option(
        None, help="Extra arguments for YouTube-DLP extraction in classic format."
    ),
    keywords: str = typer.Option(
        None, help="Comma separated positive keywords for text extraction."
    ),
    summary_url: str = typer.Option(
        None, help="URL to summarise the video using LLaMA."
    ),
    summary_interval: int = typer.Option(
        -1, help="Interval in seconds to summarise the video."
    ),
    n_workers: int = typer.Option(
        1, help="Number of workers to use. Default is 1. Use -1 to use all CPUs."
    ),
) -> None:
    """Default buffer is the perceptual hash buffer"""
    extractor_cfg = {}
    sampler_cls = VideoSampler
    subs_enable = False
    if keywords is not None:
        if not ytdlp:
            raise ValueError("Subtitle extraction supported only with yt-dlp!")
        keywords_ = [s.strip() for s in keywords.split(",")]
        extractor_cfg = {"type": "keyword", "args": {"keywords": keywords_}}
        sampler_cls = SegmentSampler
        subs_enable = True
    summary_config = {}
    if summary_interval > 0:
        summary_config = {"url": summary_url, "min_sum_interval": summary_interval}
    elif summary_url is not None:
        console.print(
            "Set summary interval to be greater than 0 to enable summary feature.",
            style=f"bold {Color.red.value}",
        )
        return typer.Exit(code=-1)
    cfg = SamplerConfig(
        min_frame_interval_sec=min_frame_interval_sec,
        keyframes_only=keyframes_only,
        queue_wait=queue_wait,
        print_stats=stats,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        debug=debug,
        buffer_config={
            "type": "hash",
            "size": buffer_size,
            "debug": debug,
            "hash_size": hash_size,
        },
        summary_config=summary_config,
        gate_config=(
            {
                "type": "blur",
                "method": blur_method,
                "threshold": threshold,
            }
            if threshold > 0
            else {
                "type": "pass",
            }
        ),
        extractor_config=extractor_cfg,
        n_workers=n_workers,
    )
    if ytdlp:
        video_path = _ytdlp_plugin(yt_extra_args, video_path, get_subs=subs_enable)
    _create_from_config(
        cfg=cfg, video_path=video_path, output_path=output_path, sampler_cls=sampler_cls
    )


@app.command(name="buffer")
def buffer(
    buffer_type: Annotated[BufferType, typer.Argument(case_sensitive=False)],
    video_path: str = typer.Argument(
        ..., help="Path to the video file or a glob pattern."
    ),
    output_path: str = typer.Argument(..., help="Path to the output folder."),
    min_frame_interval_sec: float = typer.Option(
        1.0, help="Minimum frame interval in seconds."
    ),
    stats: bool = typer.Option(True, help="Print stats."),
    keyframes_only: bool = typer.Option(True, help="Only sample keyframes."),
    buffer_size: int = typer.Option(10, help="Size of the buffer."),
    hash_size: int = typer.Option(4, help="Size of the hash."),
    expiry: int = typer.Option(4, help="Expiry time for the buffer."),
    queue_wait: float = typer.Option(0.1, help="Time to wait for the queue."),
    start_time_s: int = typer.Option(
        0, help="The starting time for sampling in seconds."
    ),
    end_time_s: int = typer.Option(
        None, help="The ending time for sampling in seconds. None for no end."
    ),
    debug: bool = typer.Option(False, help="Enable debug mode."),
    grid_size: int = typer.Option(4, help="Grid size for the grid buffer."),
    max_hits: int = typer.Option(2, help="Max hits for the grid buffer."),
    threshold: float = typer.Option(
        20.0, help="Threshold for the blur gate. If 0 then no blur gate is used."
    ),
    blur_method: str = typer.Option(
        "fft", help="Method to use for blur gate. Can be fft or variance."
    ),
    ytdlp: bool = typer.Option(
        False,
        help="Use yt-dlp to download videos from urls. Default is False."
        " Enabling this will treat video_path as an input to ytdlp command.",
    ),
    yt_extra_args: str = typer.Option(
        None, help="Extra arguments for YouTube-DLP extraction in classic format."
    ),
    n_workers: int = typer.Option(
        1, help="Number of workers to use. Default is 1. Use -1 to use all CPUs."
    ),
):
    """Buffer type can be one of entropy, gzip, hash, passthrough"""
    cfg = SamplerConfig(
        min_frame_interval_sec=min_frame_interval_sec,
        keyframes_only=keyframes_only,
        queue_wait=queue_wait,
        print_stats=stats,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        debug=debug,
        buffer_config={
            "type": buffer_type,
            "size": buffer_size,
            "debug": debug,
            "hash_size": hash_size,
            "expiry": expiry,
            "grid_x": grid_size,
            "grid_y": grid_size,
            "max_hits": max_hits,
        },
        gate_config=(
            {
                "type": "blur",
                "method": blur_method,
                "threshold": threshold,
            }
            if threshold > 0
            else {
                "type": "pass",
            }
        ),
        n_workers=n_workers,
    )
    if ytdlp:
        video_path = _ytdlp_plugin(yt_extra_args, video_path)
    _create_from_config(cfg=cfg, video_path=video_path, output_path=output_path)


@app.command(name="clip")
def clip(
    video_path: str = typer.Argument(
        ..., help="Path to the video file or a glob pattern."
    ),
    output_path: str = typer.Argument(..., help="Path to the output folder."),
    pos_samples: str = typer.Option(
        None, help="Comma separated positive samples to use for gating."
    ),
    neg_samples: str = typer.Option(
        None, help="Comma separated negative samples to use for gating."
    ),
    pos_margin: float = typer.Option(0.2, help="Positive margin for gating."),
    neg_margin: float = typer.Option(0.3, help="Negative margin for gating."),
    batch_size: int = typer.Option(32, help="Batch size for CLIP."),
    model_name: str = typer.Option("ViT-B-32", help="Model name for CLIP."),
    min_frame_interval_sec: float = typer.Option(
        1.0, help="Minimum frame interval in seconds."
    ),
    stats: bool = typer.Option(True, help="Print stats."),
    keyframes_only: bool = typer.Option(True, help="Only sample keyframes."),
    buffer_size: int = typer.Option(10, help="Size of the buffer."),
    hash_size: int = typer.Option(4, help="Size of the hash."),
    queue_wait: float = typer.Option(0.1, help="Time to wait for the queue."),
    start_time_s: int = typer.Option(
        0, help="The starting time for sampling in seconds."
    ),
    end_time_s: int = typer.Option(
        None, help="The ending time for sampling in seconds. None for no end."
    ),
    debug: bool = typer.Option(False, help="Enable debug mode."),
    ytdlp: bool = typer.Option(
        False,
        help="Use yt-dlp to download videos from urls. Default is False."
        " Enabling this will treat video_path as an input to ytdlp command.",
    ),
    yt_extra_args: str = typer.Option(
        None, help="Extra arguments for YouTube-DLP extraction in classic format."
    ),
    n_workers: int = typer.Option(
        1, help="Number of workers to use. Default is 1. Use -1 to use all CPUs."
    ),
):
    """Buffer type can be only of type hash when using CLIP gating."""
    if pos_samples is not None:
        pos_samples = [s.strip() for s in pos_samples.split(",")]
    if neg_samples is not None:
        neg_samples = [s.strip() for s in neg_samples.split(",")]
    console.print(
        f"Using {len(pos_samples)} positive samples and {len(neg_samples)} negative samples",
        style=f"bold {Color.yellow.value}",
    )
    cfg = SamplerConfig(
        min_frame_interval_sec=min_frame_interval_sec,
        keyframes_only=keyframes_only,
        queue_wait=queue_wait,
        print_stats=stats,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        debug=debug,
        buffer_config={
            "type": "hash",
            "size": buffer_size,
            "debug": debug,
            "hash_size": hash_size,
        },
        gate_config={
            "type": "clip",
            "pos_samples": pos_samples,
            "neg_samples": neg_samples,
            "pos_margin": pos_margin,
            "neg_margin": neg_margin,
            "model_name": model_name,
            "batch_size": batch_size,
        },
        n_workers=n_workers,
    )
    if ytdlp:
        video_path = _ytdlp_plugin(yt_extra_args, video_path)
    _create_from_config(cfg=cfg, video_path=video_path, output_path=output_path)


@app.command(name="config")
def from_config(
    config_path: str = typer.Argument(..., help="Path to the configuration file."),
    video_path: str = typer.Argument(
        ..., help="Path to the video file or a glob pattern."
    ),
    output_path: str = typer.Argument(..., help="Path to the output folder."),
    ytdlp: bool = typer.Option(
        False,
        help="Use yt-dlp to download videos from urls. Default is False."
        " Enabling this will treat video_path as an input to ytdlp command.",
    ),
    yt_extra_args: str = typer.Option(
        None, help="Extra arguments for YouTube-DLP extraction in classic format."
    ),
):
    """Create a sampler from a configuration file."""

    cfg = SamplerConfig.from_yaml(config_path)
    if ytdlp:
        video_path = _ytdlp_plugin(yt_extra_args, video_path)
    _create_from_config(cfg=cfg, video_path=video_path, output_path=output_path)


def main_loop():
    app()


if __name__ == "__main__":
    main_loop()
