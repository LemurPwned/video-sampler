# type: ignore[attr-defined]

import glob
import os
from enum import Enum

import typer
from rich.console import Console
from tqdm import tqdm

from video_sampler import version
from video_sampler.sampler import SamplerConfig, Worker


class Color(str, Enum):
    white = "white"
    red = "red"
    cyan = "cyan"
    magenta = "magenta"
    yellow = "yellow"
    green = "green"


app = typer.Typer(
    name="video-sampler",
    help="Video sampler allows you to efficiently sample video frames",
    add_completion=False,
)
console = Console()


def version_callback(print_version: bool) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]video-sampler[/] version: [bold blue]{version}[/]")
        raise typer.Exit()



@app.command(name="")
def main(
    video_path: str = typer.Argument(..., help="Path to the video file or a folder."),
    output_path: str = typer.Argument(..., help="Path to the output folder."),
    min_frame_interval_sec: int = typer.Option(1, help="Minimum frame interval in seconds."),
    keyframes_only: bool = typer.Option(True, help="Only sample keyframes."),
    buffer_size: int = typer.Option(10, help="Size of the buffer."),
    hash_size: int = typer.Option(8, help="Size of the hash."),
    queue_wait: float = typer.Option(0.1, help="Time to wait for the queue."),
) -> None:
    """Print a greeting with a giving name."""

    cfg = SamplerConfig(
        min_frame_interval_sec=min_frame_interval_sec,
        keyframes_only=keyframes_only,
        buffer_size=buffer_size,
        hash_size=hash_size,
        queue_wait=queue_wait,
    )

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
