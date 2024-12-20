import os
from collections.abc import Iterable

import numpy as np
from PIL import Image
from typer import Exit

from ..config import SamplerConfig
from ..logging import Color, console
from ..schemas import FrameObject
from .video_sampler import VideoSampler

try:
    import cv2
    import pycuda.driver as cuda
    import PyNvVideoCodec as nvc
except ImportError:
    console.print(
        "GPUVideoSampler requires pycuda and PyNvVideoCodec to be installed.",
        style=f"bold {Color.red.value}",
    )


class GPUVideoSampler(VideoSampler):
    def __init__(self, cfg: SamplerConfig) -> None:
        super().__init__(cfg)

    def sample(
        self, video_path: str, subs: str | None = None
    ) -> Iterable[list[FrameObject]]:
        """Generate sample frames from a video using a GPU decoder.

        Args:
            video_path (str): The path to the video file.
            subs (str): Unused in video sampler

        Yields:
            Iterable[list[FrameObject]]: A generator that yields a list
                    of FrameObjects representing sampled frames.
        """
        self.init_sampler()
        try:
            cuda.init()
            dev_id = int(os.environ.get("GPU_ID", 0))
            device_count = cuda.Device.count()
            if dev_id >= device_count:
                console.print(
                    f"Requested GPU ID {dev_id} is out of range. Available IDs: 0-{device_count - 1}",
                    style=f"bold {Color.red.value}",
                )
                return Exit(code=-1)
            cudaDevice = cuda.Device(dev_id)
            cudaCtx = cudaDevice.retain_primary_context()
            console.print(
                f"Context created on device: {cudaDevice.name()}",
                style=f"bold {Color.green.value}",
            )
            cudaCtx.push()
            cudaStreamNvDec = cuda.Stream()
            nvDmx = nvc.CreateDemuxer(filename=video_path)
            nvDec = nvc.CreateDecoder(
                gpuid=0,
                codec=nvDmx.GetNvCodecId(),
                cudacontext=cudaCtx.handle,
                cudastream=cudaStreamNvDec.handle,
                enableasyncallocations=False,
            )
            fps_est = nvDmx.FrameRate()
            assert (
                fps_est > 0
            ), f"Failed to get FPS from the video using GPU decoder. Got: {fps_est}"

            console.print(
                f"GPU decoder currently produces est. timestamps based on FPS: {fps_est}",
                style=f"bold {Color.yellow.value}",
            )
            if self.cfg.keyframes_only:
                console.print(
                    "Keyframes only mode is not supported with GPU decoder. Argument is ignored.",
                    style=f"bold {Color.red.value}",
                )

            frame_indx = 0
            # currently stuff like
            # packet.pts, decodedFrame.timestamp etc. they don't work
            # so we use the frame_indx to keep track of the frame number
            prev_time = -10
            for packet in nvDmx:
                for decodedFrame in nvDec.Decode(packet):
                    self.stats["total"] += 1
                    ftime = frame_indx / fps_est
                    if self.cfg.start_time_s > 0 and ftime < self.cfg.start_time_s:
                        self.debug_print(
                            f"Frame time {ftime} is before start time {self.cfg.start_time_s}, skipping."
                        )
                        continue

                    if self.cfg.end_time_s is not None and ftime > self.cfg.end_time_s:
                        self.debug_print(
                            f"Frame time {ftime} is after end time {self.cfg.end_time_s}, stopping."
                        )
                        break
                    frame_indx += 1
                    time_diff = ftime - prev_time
                    if time_diff < self.cfg.min_frame_interval_sec:
                        continue
                    prev_time = ftime

                    cuda_ptr = decodedFrame.GetPtrToPlane(0)
                    numpy_array = np.ndarray(shape=(decodedFrame.shape), dtype=np.uint8)
                    cuda.memcpy_dtoh(numpy_array, cuda_ptr)
                    numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_YUV2RGB_NV12)
                    pil_image = Image.fromarray(numpy_array)  # Convert to PIL
                    yield from self.process_frame(frame_indx, pil_image, ftime)
            # flush buffer
            yield from self.flush_buffer()
            cudaCtx.pop()
            console.print(
                "Context removed.\nEnd of decode session",
                style=f"bold {Color.green.value}",
            )
        except Exception as e:
            console.print(
                f"Error during GPU decoding: {e}", style=f"bold {Color.red.value}"
            )
            if "cudaCtx" in locals():
                cudaCtx.pop()
                cudaCtx.detach()
            raise e
