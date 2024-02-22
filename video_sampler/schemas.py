from dataclasses import dataclass, field
from enum import Enum

from PIL import Image


class BufferType(str, Enum):
    entropy = "entropy"
    gzip = "gzip"
    hash = "hash"
    passthrough = "passthrough"
    grid = "grid"


@dataclass
class FrameObject:
    """
    A frame object.

    Args:
        frame (Image.Image): The frame image.
        metadata (dict): The metadata for the frame.
    """

    frame: Image.Image
    metadata: dict


PROCESSING_DONE = FrameObject(None, metadata={"end": True})
PROCESSING_DONE_ITERABLE = (PROCESSING_DONE,)
EMPTY_FRAME_OBJECT = FrameObject(None, metadata={})
EMPTY_FRAME_OBJECT_ITERABLE = (EMPTY_FRAME_OBJECT,)


@dataclass
class GatedObject:
    frames: list[FrameObject] = field(default_factory=list)
    N: int = 0


EMPTY_GATED_OBJECT = GatedObject(None, 0)
