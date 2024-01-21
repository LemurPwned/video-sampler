# type: ignore[attr-defined]

import glob
import os
from typing import Annotated

import typer
from tqdm import tqdm

from video_sampler import version
from video_sampler.buffer import SamplerConfig, check_args_validity
from video_sampler.logging import Color, console
from video_sampler.sampler import Worker
from video_sampler.schemas import BufferType

app = typer.Typer(
    name="video-sampler",
    help="Video sampler allows you to efficiently sample video frames",
    add_completion=True,
)


def _create_from_config(cfg: SamplerConfig, video_path: str, output_path: str):
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
    delegate_workers(video_path=video_path, output_path=output_path, cfg=cfg)


def version_callback(print_version: bool = True) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]video-sampler[/] version: [bold blue]{version}[/]")
        raise typer.Exit()


def delegate_workers(video_path: str, output_path: str, cfg: SamplerConfig):
    msg = "Detected input as a file"
    if not os.path.isfile(video_path):
        if "*" not in video_path:
            videos = glob.glob(os.path.join(video_path, "*"))
        else:
            videos = glob.glob(video_path)
        msg = f"Detected input as a folder with {len(videos)} files"
    else:
        videos = [video_path]
    console.print(msg, style=f"bold {Color.cyan.value}")

    worker = Worker(
        cfg=cfg,
    )
    for video in tqdm(videos, desc="Processing videos..."):
        video_subpath = os.path.join(output_path, os.path.basename(video))
        worker.launch(
            video_path=video,
            output_path=video_subpath,
        )


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
    debug: bool = typer.Option(False, help="Enable debug mode."),
    threshold: float = typer.Option(
        20.0, help="Threshold for the blur gate. If 0 then no blur gate is used."
    ),
    blur_method: str = typer.Option(
        "fft", help="Method to use for blur gate. Can be fft or variance."
    ),
) -> None:
    """Default buffer is the perceptual hash buffer"""

    cfg = SamplerConfig(
        min_frame_interval_sec=min_frame_interval_sec,
        keyframes_only=keyframes_only,
        queue_wait=queue_wait,
        print_stats=stats,
        debug=debug,
        buffer_config={
            "type": "hash",
            "size": buffer_size,
            "debug": debug,
            "hash_size": hash_size,
        },
        gate_config={
            "type": "blur",
            "method": blur_method,
            "threshold": threshold,
        }
        if threshold > 0
        else {
            "type": "pass",
        },
    )
    _create_from_config(cfg=cfg, video_path=video_path, output_path=output_path)


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
    debug: bool = typer.Option(False, help="Enable debug mode."),
    grid_size: int = typer.Option(4, help="Grid size for the grid buffer."),
    max_hits: int = typer.Option(2, help="Max hits for the grid buffer."),
    threshold: float = typer.Option(
        20.0, help="Threshold for the blur gate. If 0 then no blur gate is used."
    ),
    blur_method: str = typer.Option(
        "fft", help="Method to use for blur gate. Can be fft or variance."
    ),
):
    """Buffer type can be one of entropy, gzip, hash, passthrough"""
    cfg = SamplerConfig(
        min_frame_interval_sec=min_frame_interval_sec,
        keyframes_only=keyframes_only,
        queue_wait=queue_wait,
        print_stats=stats,
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
        gate_config={
            "type": "blur",
            "method": blur_method,
            "threshold": threshold,
        }
        if threshold > 0
        else {
            "type": "pass",
        },
    )
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
    debug: bool = typer.Option(False, help="Enable debug mode."),
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
    )
    _create_from_config(cfg=cfg, video_path=video_path, output_path=output_path)


def main_loop():
    app()


if __name__ == "__main__":
    main_loop()
