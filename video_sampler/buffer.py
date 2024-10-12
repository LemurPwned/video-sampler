import gzip
import heapq
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from imagehash import average_hash, phash
from PIL import Image

from .config import SamplerConfig
from .logging import Color, console


class FrameBuffer(ABC):
    @abstractmethod
    def add(self, item: Image.Image, metadata: dict[str, Any]) -> None | tuple:
        pass

    @abstractmethod
    def final_flush(self) -> Iterable[tuple[Image.Image | None, dict]]:
        """Flush the buffer and return the remaining items"""
        pass

    @abstractmethod
    def get_buffer_state(self) -> list[str]:
        """Return the current state of the buffer"""
        pass

    @abstractmethod
    def clear(self):
        """Clear the buffer"""
        pass


class PassThroughBuffer(FrameBuffer):
    def __init__(self) -> None: ...

    def get_buffer_state(self) -> list[str]:
        return []

    def add(self, item: Image.Image, metadata: dict[str, Any]):
        return (item, metadata)

    def final_flush(self) -> Iterable[tuple[Image.Image | None, dict]]:
        yield None

    def clear(self):
        pass


class HashBuffer(FrameBuffer):
    """
    A buffer that stores frames with their corresponding metadata and
    checks for duplicates based on image hashes.
    Args:
        size (int): The maximum size of the buffer.
        debug_flag (bool, optional): Flag indicating whether to enable debug mode. Defaults to False.
        hash_size (int, optional): The size of the image hash. Defaults to 4.

    Methods:
        get_buffer_state() -> list[str]:
            Returns the current state of the buffer as a list of image hashes.

        add(item: Image.Image, metadata: dict[str, Any])
            Adds an item to the buffer along with its metadata.

        final_flush() -> Iterable[tuple[Image.Image | None, dict]]:
            Yields the stored items and their metadata in the buffer.

        clear()
            Clears the buffer.

    Private Methods:
        __add(item: Image.Image, hash_: str, metadata: dict)
            Adds an item to the buffer with the given hash and metadata.

        __check_duplicate(hash_: str) -> bool:
            Checks if the given hash already exists in the buffer and renews its validity if found.

    """

    def __init__(self, size: int, debug_flag: bool = False, hash_size: int = 4) -> None:
        self.ordered_buffer = OrderedDict()
        self.max_size = size
        self.debug_flag = debug_flag
        self.hash_size = hash_size

    def get_buffer_state(self) -> list[str]:
        return list(self.ordered_buffer.keys())

    def add(self, item: Image.Image, metadata: dict[str, Any]):
        hash_ = str(phash(item, hash_size=self.hash_size))
        if not self._check_duplicate(hash_):
            return self.__add(hash_, item, metadata)
        return None

    def __add(self, hash_: str, item: Image.Image, metadata: dict):
        self.ordered_buffer[hash_] = (item, metadata)
        if len(self.ordered_buffer) >= self.max_size:
            return self.ordered_buffer.popitem(last=False)[1]
        return None

    def _check_duplicate(self, hash_: str) -> bool:
        if hash_ in self.ordered_buffer:
            # renew the hash validity
            if self.debug_flag:
                console.print(
                    f"Renewing {hash_}",
                    style=f"bold {Color.red.value}",
                )
            self.ordered_buffer.move_to_end(hash_)
            return True
        return False

    def final_flush(self) -> Iterable[tuple[Image.Image | None, dict]]:
        yield from self.ordered_buffer.values()

    def clear(self):
        self.ordered_buffer.clear()


class GridBuffer(HashBuffer):
    """
    A class representing a grid-based buffer for images.
    Splits the image into a grid and stores the hashes of the grid cells in a mosaic buffer.

    Args:
        size (int): The maximum size of the buffer.
        debug_flag (bool, optional): A flag indicating whether debug information should be printed.
        hash_size (int, optional): The size of the hash.
        grid_x (int, optional): The number of grid cells in the x-axis.
        grid_y (int, optional): The number of grid cells in the y-axis.
        max_hits (int, optional): The maximum number of hits allowed for a hash.

    Attributes:
        grid_x (int): The number of grid cells in the x-axis.
        grid_y (int): The number of grid cells in the y-axis.
        max_hits (int): The maximum number of hits allowed for a hash.
        mosaic_buffer (dict): A dictionary storing the mosaic buffer.

    Methods:
        add(item, metadata):
            Adds an image to the buffer along with its metadata.
        clear():
            Clears the buffer and the mosaic buffer.
        update_ttl_buffer():
            Updates the buffer by expiring images that are not in the grid.

    """

    def __init__(
        self,
        size: int,
        debug_flag: bool = False,
        hash_size: int = 4,
        grid_x: int = 4,
        grid_y: int = 4,
        max_hits: int = 1,
    ) -> None:
        super().__init__(size, debug_flag, hash_size)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.max_hits = max_hits
        self.mosaic_buffer = {}

    def __get_grid_hash(self, item: Image.Image) -> Iterable[str]:
        """Compute grid hashes for a given image"""
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                yield str(
                    phash(
                        item.crop(
                            (
                                x * item.width / self.grid_x,
                                y * item.height / self.grid_y,
                                (x + 1) * item.width / self.grid_x,
                                (y + 1) * item.height / self.grid_y,
                            )
                        ),
                        hash_size=self.hash_size,
                    )
                )

    def _check_mosaic(self, mosaic_hash: str):
        return mosaic_hash in self.mosaic_buffer

    def update_ttl_buffer(self):
        # expire the images that are not in the grid
        if len(self.ordered_buffer) >= self.max_size:
            to_return_hash, return_data = self.ordered_buffer.popitem(last=False)
            if to_return_hash is not None:
                removal_keys = [
                    img_hash
                    for img_hash, mosaic_hash in self.mosaic_buffer.items()
                    if mosaic_hash == to_return_hash
                ]
                for key in removal_keys:
                    del self.mosaic_buffer[key]
            return return_data
        return None

    def add(self, item: Image.Image, metadata: dict[str, Any]):
        hash_ = str(phash(item, hash_size=self.hash_size))
        if not self._check_duplicate(hash_):
            # not automatically rejected, check the mosaic buffer
            hash_hits = 0
            hash_sets = []
            for el_hash_ in self.__get_grid_hash(item):
                if el_hash_ in self.mosaic_buffer:
                    hash_hits += 1
                hash_sets.append(el_hash_)

            if hash_hits < self.max_hits:
                # add image hash to the ttl counter
                self.ordered_buffer[hash_] = (item, metadata)
                # add the image to the mosaic buffer
                # this also automatically overwrites the deleted hashes
                for el_hash in hash_sets:
                    self.mosaic_buffer[el_hash] = hash_

            if self.debug_flag:
                console.print(
                    f"\tHash hits: {hash_hits}"
                    f"\tHash sets: {len(hash_sets)}"
                    f"\tHash buffer: {len(self.get_buffer_state())}"
                    f"\tMosaic buffer: {len(self.mosaic_buffer)}"
                )
        return self.update_ttl_buffer()

    def clear(self):
        super().clear()
        self.mosaic_buffer = {}


class SlidingTopKBuffer(FrameBuffer):
    """
    A class representing a sliding top-k buffer for frames.

    Args:
        size (int): The maximum size of the buffer.
        debug_flag (bool, optional): A flag indicating whether debug information should be printed.
        expiry (int, optional): The expiry count for frames.
        hash_size (int, optional): The size of the hash.

    Attributes:
        sliding_buffer (list): The sliding buffer implemented as a min heap.
        max_size (int): The maximum size of the buffer.
        debug_flag (bool): A flag indicating whether debug information should be printed.
        expiry_count (int): The expiry count for frames.
        hash_size (int): The size of the hash.

    Methods:
        get_buffer_state() -> list[str]:
            Returns the current state of the buffer.
        add(item, metadata):
            Adds a frame to the buffer along with its metadata.
        final_flush() -> Iterable[tuple[Image.Image | None, dict]]:
            Performs a final flush of the buffer and yields the remaining frames.
        clear():
            Clears the buffer.

    """

    def __init__(
        self, size: int, debug_flag: bool = False, expiry: int = 30, hash_size: int = 8
    ) -> None:
        # it's a min heap with a fixed size
        self.sliding_buffer = []
        self.max_size = size
        self.debug_flag = debug_flag
        self.expiry_count = expiry
        self.hash_size = hash_size
        assert (
            self.expiry_count > self.max_size
        ), "expiry count must be greater than max size"
        console.print(
            f"Creating sliding buffer of size {self.max_size} and expiry {expiry}",
            style=f"bold {Color.red.value}",
        )

    def get_buffer_state(self) -> list[str]:
        return [item[:3] for item in self.sliding_buffer]

    def add(self, item: Image.Image, metadata: dict[str, Any]):
        assert "index" in metadata, "metadata must have index key for sliding buffer"
        average_hash_ = str(average_hash(item, hash_size=self.hash_size))
        to_return = None
        if not self.__check_duplicate(average_hash_):
            heapq.heappush(
                self.sliding_buffer,
                [metadata["index"], 0, average_hash_, item, metadata],
            )
            if len(self.sliding_buffer) >= self.max_size:
                to_return = heapq.heappop(self.sliding_buffer)[-2:]
        # update the expiry count
        expired_indx = -1
        for i in range(len(self.sliding_buffer)):
            self.sliding_buffer[i][1] += 1
            if self.sliding_buffer[i][1] >= self.expiry_count:
                expired_indx = i
        # at any point only one item can be expired
        if expired_indx != -1:
            self.sliding_buffer.pop(expired_indx)  # just drop
        return to_return

    def __check_duplicate(self, hash_: str) -> bool:
        for item in self.sliding_buffer:
            if item[2] == hash_:
                # renew the hash validity
                if self.debug_flag:
                    console.print(
                        f"Renewing {hash_}",
                        style=f"bold {Color.red.value}",
                    )
                item[1] = 0
                return True
        return False

    def final_flush(self) -> Iterable[tuple[Image.Image | None, dict]]:
        if len(self.sliding_buffer):
            yield heapq.heappop(self.sliding_buffer)[-2:]
        yield None, {}

    def clear(self):
        self.sliding_buffer.clear()


class GzipBuffer(FrameBuffer):
    """Measure compression size as a function of the image usability"""

    def __init__(
        self, size: int, expiry: int, debug_flag: bool = False, hash_size: int = 8
    ) -> None:
        self.sliding_top_k_buffer = SlidingTopKBuffer(
            size=size, expiry=expiry, debug_flag=debug_flag, hash_size=hash_size
        )

    def get_buffer_state(self) -> list[str]:
        return self.sliding_top_k_buffer.get_buffer_state()

    def add(self, item: Image.Image, metadata: dict[str, Any]):
        compressed_l = len(gzip.compress(item.tobytes()))
        return self.sliding_top_k_buffer.add(item, {**metadata, "index": -compressed_l})

    def final_flush(self) -> Iterable[tuple[Image.Image | None, dict]]:
        return self.sliding_top_k_buffer.final_flush()

    def clear(self):
        self.sliding_top_k_buffer.clear()


class EntropyByffer(FrameBuffer):
    """Measure image entropy as a function of the image usability"""

    def __init__(
        self, size: int, expiry: int, debug_flag: bool = False, hash_size: int = 8
    ) -> None:
        self.sliding_top_k_buffer = SlidingTopKBuffer(
            size=size, expiry=expiry, debug_flag=debug_flag, hash_size=hash_size
        )

    def get_buffer_state(self) -> list[str]:
        return self.sliding_top_k_buffer.get_buffer_state()

    def add(self, item: Image.Image, metadata: dict[str, Any]):
        return self.sliding_top_k_buffer.add(
            item, {**metadata, "index": -item.entropy()}
        )

    def final_flush(self) -> Iterable[tuple[Image.Image | None, dict]]:
        return self.sliding_top_k_buffer.final_flush()

    def clear(self):
        self.sliding_top_k_buffer.clear()


def check_args_validity(cfg: SamplerConfig):
    assert (
        cfg.min_frame_interval_sec > 0
    ), "min_frame_interval_sec must be greater than 0"
    assert cfg.buffer_config["size"] > 0, "buffer size must be greater than 0"
    arg_check = {
        "hash": ("hash_size", "size"),
        "gzip": ("hash_size", "size", "expiry"),
        "entropy": ("hash_size", "size", "expiry"),
        "gating": ("hash_size", "size"),
        "grid": ("hash_size", "size", "grid_x", "grid_y", "max_hits"),
        "passthrough": (),
    }
    for arg in arg_check[cfg.buffer_config["type"]]:
        assert arg in cfg.buffer_config, f"{arg} must be present in buffer config"
        assert (
            cfg.buffer_config[arg] is not None and cfg.buffer_config[arg] > 0
        ), f"{arg} must be greater than 0 and must not be None"


def create_buffer(buffer_config: dict[str, Any]):
    """Create a buffer based on the config"""
    console.print(
        f"Creating buffer of type {buffer_config['type']}",
        style=f"bold {Color.red.value}",
    )
    if buffer_config["type"] == "hash":
        return HashBuffer(
            size=buffer_config["size"],
            debug_flag=buffer_config["debug"],
            hash_size=buffer_config["hash_size"],
        )
    elif buffer_config["type"] == "grid":
        return GridBuffer(
            size=buffer_config["size"],
            debug_flag=buffer_config["debug"],
            hash_size=buffer_config["hash_size"],
            grid_x=buffer_config["grid_x"],
            grid_y=buffer_config["grid_y"],
            max_hits=buffer_config["max_hits"],
        )
    elif buffer_config["type"] == "sliding_top_k":
        return SlidingTopKBuffer(
            size=buffer_config["size"],
            debug_flag=buffer_config["debug"],
            expiry=buffer_config["expiry"],
        )
    elif buffer_config["type"] == "passthrough":
        return PassThroughBuffer()
    elif buffer_config["type"] == "gzip":
        return GzipBuffer(
            size=buffer_config["size"],
            debug_flag=buffer_config["debug"],
            hash_size=buffer_config["hash_size"],
            expiry=buffer_config["expiry"],
        )
    elif buffer_config["type"] == "entropy":
        return EntropyByffer(
            size=buffer_config["size"],
            debug_flag=buffer_config["debug"],
            hash_size=buffer_config["hash_size"],
            expiry=buffer_config["expiry"],
        )
    else:
        raise ValueError(f"Unknown buffer type {buffer_config['type']}")
