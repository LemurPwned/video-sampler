# type: ignore[attr-defined]
"""Video sampler allows you to efficiently sample video frames"""

from importlib import metadata as importlib_metadata

from .buffer import SamplerConfig
from .sampler import VideoSampler, Worker


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

__all__ = ["SamplerConfig", "VideoSampler", "Worker", "version"]
