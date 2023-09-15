import os
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Dict, Iterable, Tuple, Union

import av
from imagehash import phash
from PIL import Image


@dataclass
class SamplerConfig:
    min_frame_interval_sec: int = 1
    keyframes_only: bool = True
    buffer_size: int = 10
    hash_size: int = 8
    queue_wait: float = 0.1

class HashBuffer:
    def __init__(self, size: int ) -> None:
        self.buffer  = [None] * size

    def add(self, item, hash_, metadata={}):
        if not self.__check_duplicate(hash_):
            return self.__add(item, metadata)
        return None

    def __add(self, item, metadata={}):
        self.buffer.append((item, metadata))
        return self.buffer.pop(0)

    def __check_duplicate(self, hash_)-> bool:
        if hash_ in self.buffer:
            return True
        return False


class VideoSampler:
    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = SamplerConfig
        self.hash_buf = HashBuffer(cfg.buffer_size)

    def compute_hash(self, frame_img: Image) -> str:
        return str(phash(frame_img, hash_size=self.cfg.hash_size))


    def sample(self, video_path: str) -> Iterable[Tuple[Union[Image.Image, None], Dict]]:
        """Generate sample frames from a video"""
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if self.cfg.keyframes_only:
                stream.codec_context.skip_frame = "NONKEY"
            prev_time = - 10
            for frame_indx, frame in enumerate(container.decode(stream)):
                # skip frames if keyframes_only is True
                time_diff = frame.time - prev_time
                if time_diff < self.cfg.min_frame_interval_sec:
                    continue
                prev_time = frame.time

                frame_npy: Image = frame.to_image()
                frame_hash = self.compute_hash(frame_npy)
                res = self.hash_buf.add(frame_npy, frame_hash, metadata={"frame_time": frame.time, "frame_indx": frame_indx})
                if res:
                    yield res

        # flush buffer
        for item in self.hash_buf.buffer:
            if item:
                yield item
        yield None, {
            "end": True
        }

    def write_queue(self, video_path: str, q: Queue):
        for item in self.sample(video_path=video_path):
            q.put(item)

class Worker:
    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = cfg
        self.processor = VideoSampler(cfg=cfg)
        self.q = Queue()

    def launch(self, video_path: str, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        proc_thread = Thread(target=self.processor.write_queue, args=(video_path, self.q))
        proc_thread.start()
        self.queue_reader(output_path, read_interval=self.cfg.queue_wait)
        proc_thread.join()

    def queue_reader(self, output_path, read_interval=.1) -> None:
        while True:
            if not self.q.empty():
                item = self.q.get()
                frame, metadata = item
                if frame is not None:
                    if isinstance(frame, Image.Image):
                        frame.save(os.path.join(output_path, f"{metadata['frame_time']}.jpg"))
                    # with open(os.path.join(output_path, f"{metadata['frame_time']}.jpg"), "wb") as f:
                        # f.write(frame.tobytes())
                if metadata.get("end", False):
                    break
            time.sleep(read_interval)
