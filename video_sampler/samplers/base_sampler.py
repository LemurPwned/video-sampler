from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from queue import Queue

from PIL import Image

from ..buffer import FrameBuffer, create_buffer
from ..config import SamplerConfig
from ..gating import BlurGate, ClipGate, PassGate, create_gate
from ..logging import Color, console
from ..schemas import PROCESSING_DONE_ITERABLE, FrameObject, GatedObject


class BaseSampler:
    def __init__(self, cfg: SamplerConfig):
        self.cfg: SamplerConfig = deepcopy(cfg)
        self.frame_buffer: FrameBuffer = create_buffer(self.cfg.buffer_config)
        self.gate: BlurGate | ClipGate | PassGate = create_gate(self.cfg.gate_config)
        self.stats = Counter()

    def sample(self, _: str) -> Iterable[list[FrameObject]]:
        raise NotImplementedError("sample method must be implemented")

    def write_queue(self, _: str, q: Queue, subs: str | None = None):
        raise NotImplementedError("write_queue method must be implemented")

    def init_sampler(self):
        self.stats.clear()
        self.frame_buffer.clear()

    def flush_buffer(self) -> Iterable[list[FrameObject]]:
        """Flushes the frame buffer and yields gated frames"""
        for res in self.frame_buffer.final_flush():
            if res:
                self.stats["produced"] += 1
                gated_obj: GatedObject = self.gate(*res)
                self.stats["gated"] += gated_obj.N
                if gated_obj.frames:
                    yield gated_obj.frames
        gated_obj: GatedObject = self.gate.flush()
        self.stats["gated"] += gated_obj.N
        if gated_obj.frames:
            yield gated_obj.frames
        yield PROCESSING_DONE_ITERABLE

    def process_frame(
        self, frame_indx: int, frame: Image, ftime: float
    ) -> Iterable[list[FrameObject]]:
        if self.cfg.debug:
            buf = self.frame_buffer.get_buffer_state()
            console.print(
                f"Frame {frame_indx}\ttime: {ftime}",
                f"\t Buffer ({len(buf)}): {buf}",
                style=f"bold {Color.green.value}",
            )
        frame_meta = {"frame_time": ftime, "frame_indx": frame_indx}
        self.stats["decoded"] += 1
        if res := self.frame_buffer.add(
            frame,
            metadata=frame_meta,
        ):
            gated_obj: GatedObject = self.gate(*res)
            self.stats["produced"] += 1
            self.stats["gated"] += gated_obj.N
            if gated_obj.frames:
                yield gated_obj.frames

    def debug_print(self, message: str):
        if self.cfg.debug:
            console.print(message, style=f"bold {Color.red.value}")
