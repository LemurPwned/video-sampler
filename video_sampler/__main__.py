# type: ignore[attr-defined]

import glob
import os

import typer
from tqdm import tqdm

from video_sampler import version
from video_sampler.logging import Color, console
from video_sampler.sampler import SamplerConfig, Worker

app = typer.Typer(
    name="video-sampler",
    help="Video sampler allows you to efficiently sample video frames",
    add_completion=False,
)


def version_callback(print_version: bool = True) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]video-sampler[/] version: [bold blue]{version}[/]")
        raise typer.Exit()


@app.command(name="")
def main(
    video_path: str = typer.Argument(
        ..., help="Path to the video file or a glob pattern."
    ),
    output_path: str = typer.Argument(..., help="Path to the output folder."),
    min_frame_interval_sec: float = typer.Option(
        1.0, help="Minimum frame interval in seconds."
    ),
    keyframes_only: bool = typer.Option(True, help="Only sample keyframes."),
    buffer_size: int = typer.Option(10, help="Size of the buffer."),
    hash_size: int = typer.Option(4, help="Size of the hash."),
    queue_wait: float = typer.Option(0.1, help="Time to wait for the queue."),
    debug: bool = typer.Option(False, help="Enable debug mode."),
) -> None:
    """Print a greeting with a giving name."""

    cfg = SamplerConfig(
        min_frame_interval_sec=min_frame_interval_sec,
        keyframes_only=keyframes_only,
        buffer_size=buffer_size,
        hash_size=hash_size,
        queue_wait=queue_wait,
        debug=debug,
    )
    console.print(cfg, style=f"bold {Color.yellow.value}")

    videos = [video_path]
    msg = "Detected input as a file"
    if not os.path.isfile(video_path):
        videos = glob.glob(video_path)
        msg = f"Detected input as a folder with {len(videos)} files"
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


if __name__ == "__main__":
    app()
