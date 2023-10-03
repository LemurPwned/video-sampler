import gzip
import heapq
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from imagehash import average_hash, phash
from PIL import Image

from .logging import Color, console


class FrameBuffer(ABC):
    @abstractmethod
    def add(self, item: Image.Image, metadata: dict[str, Any]) -> None | tuple:
        pass

    @abstractmethod
    def final_flush(self) -> Iterable[tuple[Image.Image | None, dict]]:
        """Flush the buffer and return the remaining items"""
        pass


class PassThroughBuffer(FrameBuffer):
    def __init__(self) -> None:
        ...

    def get_buffer_state(self) -> list[str]:
        return []

    def add(self, item: Image.Image, metadata: dict[str, Any]):
        return (item, metadata)

    def final_flush(self) -> Iterable[tuple[Image.Image | None, dict]]:
        yield None


class HashBuffer(FrameBuffer):
    def __init__(self, size: int, debug_flag: bool = False, hash_size: int = 4) -> None:
        self.ordered_buffer = OrderedDict()
        self.max_size = size
        self.debug_flag = debug_flag
        self.hash_size = hash_size

    def get_buffer_state(self) -> list[str]:
        return list(self.ordered_buffer.keys())

    def add(self, item: Image.Image, metadata: dict[str, Any]):
        hash_ = str(phash(item, hash_size=self.hash_size))
        if not self.__check_duplicate(hash_):
            return self.__add(item, hash_, metadata)
        return None

    def __add(self, item: Image.Image, hash_: str, metadata: dict):
        self.ordered_buffer[hash_] = (item, metadata)
        if len(self.ordered_buffer) >= self.max_size:
            return self.ordered_buffer.popitem(last=False)[1]
        return None

    def __check_duplicate(self, hash_: str) -> bool:
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
        for _, item in self.ordered_buffer.values():
            yield item


class SlidingTopKBuffer(FrameBuffer):
    def __init__(self, size: int, debug_flag: bool = False, expiry: int = 30) -> None:
        # it's a min heap with fixed size
        self.sliding_buffer = []
        self.max_size = size
        self.debug_flag = debug_flag
        self.expiry_count = expiry
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
        average_hash_ = str(average_hash(item))
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


class GzipBuffer(FrameBuffer):
    def __init__(self, size: int, debug_flag: bool = False) -> None:
        self.sliding_top_k_buffer = SlidingTopKBuffer(size=size, debug_flag=debug_flag)

    def get_buffer_state(self) -> list[str]:
        return self.sliding_top_k_buffer.get_buffer_state()

    def add(self, item: Image.Image, metadata: dict[str, Any]):
        compressed_l = len(gzip.compress(item.tobytes()))
        return self.sliding_top_k_buffer.add(item, {**metadata, "index": -compressed_l})

    def final_flush(self) -> Iterable[tuple[Image.Image | None, dict]]:
        return self.sliding_top_k_buffer.final_flush()


def create_buffer(buffer_config):
    console.print(
        f"Creating buffer of type {buffer_config['type']}",
        style=f"bold {Color.red.value}",
    )
    if buffer_config["type"] == "hash":
        return HashBuffer(
            buffer_config["size"], buffer_config["debug"], buffer_config["hash_size"]
        )
    elif buffer_config["type"] == "sliding_top_k":
        return SlidingTopKBuffer(
            buffer_config["size"], buffer_config["debug"], buffer_config["expiry"]
        )
    elif buffer_config["type"] == "passthrough":
        return PassThroughBuffer()
    elif buffer_config["type"] == "gzip":
        return GzipBuffer(buffer_config["size"], buffer_config["debug"])
    else:
        raise ValueError(f"Unknown buffer type {buffer_config['type']}")
