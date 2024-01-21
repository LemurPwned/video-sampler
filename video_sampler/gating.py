import contextlib
from typing import Any, Literal

import numpy as np
from PIL import Image

from .schemas import EMPTY_GATED_OBJECT, FrameObject, GatedObject
from .utils import batched

with contextlib.suppress(ImportError):
    import cv2
with contextlib.suppress(ImportError):
    import open_clip
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_model(model_name: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model.to(DEVICE)
    return model, preprocess, tokenizer


class PassGate:
    def __call__(self, frame: Image.Image, meta: dict, last=False) -> GatedObject:
        return self.flush() if last else GatedObject([FrameObject(frame, meta)], 1)

    def flush(self):
        return EMPTY_GATED_OBJECT


class BlurGate:
    def __init__(
        self, method: Literal["fft", "laplacian"] = "laplacian", threshold: float = 100
    ) -> None:
        """Gate frames based on bluriness.
        :param method: The method to use for blur detection. Can be "fft" or "laplacian".
        :param threshold: The threshold for bluriness. The higher the threshold, the less
            blurry the image needs to be to be discarded.
            Those are different depending on the method:
            - 20 is a good start for fft
            - 100 is a good start for laplacian.
        """
        self.is_blurry = None
        if method == "fft":
            self.is_blurry = self._is_blurry_fft
        elif method == "laplacian":
            self.is_blurry = self._is_blurry_laplacian
        else:
            raise ValueError(f"Unknown blur method {method}")
        self.threshold = threshold

    def __call__(self, frame: Image.Image, meta: dict, last=False) -> GatedObject:
        if self.is_blurry(frame) or last:
            return EMPTY_GATED_OBJECT
        return GatedObject([FrameObject(frame, meta)], 1)

    def _is_blurry_laplacian(self, frame: Image.Image) -> bool:
        """Check if the image is blurry with laplacian method."""
        return (
            cv2.Laplacian(
                cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()
            < self.threshold
        )

    def _is_blurry_fft(self, frame: Image.Image) -> bool:
        """Check if the image is blurry with fft method."""
        f = np.fft.fft2(frame)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-12)
        return magnitude_spectrum.mean() < self.threshold

    def flush(self):
        return EMPTY_GATED_OBJECT


class ClipGate:
    def __init__(
        self,
        pos_samples: list[str] = None,
        neg_samples: list[str] = None,
        model_name: str = "ViT-B-32",
        batch_size: int = 32,
        pos_margin: float = 0.2,
        neg_margin: float = 0.3,
    ) -> None:
        self.model, self.preprocess, self.tokenizer = create_model(
            model_name=model_name
        )
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.batch_size = batch_size
        self.frame_accumulator = []
        self.metadata_accumulator = []
        if pos_samples is None:
            self.pos_samples = torch.zeros((1, 512))
        else:
            self.pos_samples = self._preproc_samples(pos_samples)
        if neg_samples is None:
            self.neg_samples = torch.zeros((1, 512))
        else:
            self.neg_samples = self._preproc_samples(neg_samples)

    def __call__(self, frame: Image.Image, meta: dict, last=False) -> Any:
        return self.flush() if last else self.add_frame(frame, meta)

    def _preproc_samples(self, sample_texts: list[str]):
        inputs = self.tokenizer(sample_texts)
        embeds = torch.zeros((len(sample_texts), 512))
        with torch.no_grad():
            for i, batch in enumerate(batched(inputs, n=self.batch_size)):
                batch = torch.stack(batch)
                text_embeds = self.model.encode_text(batch.to(DEVICE))
                embeds[
                    i * self.batch_size : (i + 1) * self.batch_size
                ] = text_embeds.cpu()
        embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds

    def _embed_frames(self, frames: list[Image.Image]):
        """Compute the embeddings for each frame."""
        inputs = torch.stack([self.preprocess(frame) for frame in frames]).to(DEVICE)
        with torch.no_grad():
            image_embeds = self.model.encode_image(inputs).cpu()
            image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

    def _get_margins(self, frame_embeds: torch.Tensor):
        """Compute the margins for each frame."""
        org_indx = np.arange(frame_embeds.shape[0])
        neg_distance = frame_embeds @ self.neg_samples.T
        pos_distance = frame_embeds @ self.pos_samples.T
        neg_margin, _ = neg_distance.max(axis=-1)
        pos_margin, _ = pos_distance.max(axis=-1)
        incl_samples = torch.argwhere(
            (neg_margin < self.neg_margin) & (pos_margin >= self.pos_margin)
        )
        return org_indx[incl_samples].ravel()

    def add_frame(self, frame: Image.Image, metadata: dict) -> GatedObject:
        self.frame_accumulator.append(frame)
        self.metadata_accumulator.append(metadata)
        if len(self.frame_accumulator) == self.batch_size:
            return self.__process_metadata()
        return EMPTY_GATED_OBJECT

    def flush(self):
        return self.__process_metadata()

    def __process_metadata(self) -> GatedObject:
        frame_embeds = self._embed_frames(self.frame_accumulator)
        selected_frames = self._get_margins(frame_embeds)
        to_return = [
            FrameObject(self.frame_accumulator[i], self.metadata_accumulator[i])
            for i in range(len(self.frame_accumulator))
            if i in selected_frames
        ]
        self.frame_accumulator.clear()
        self.metadata_accumulator.clear()
        return GatedObject(to_return, len(selected_frames))


def create_gate(gate_config: dict):
    gate_type = gate_config["type"]
    del gate_config["type"]
    if gate_type == "pass":
        return PassGate()
    elif gate_type == "clip":
        return ClipGate(**gate_config)
    elif gate_type == "blur":
        return BlurGate(**gate_config)
    else:
        raise ValueError(f"Unknown gate type {gate_type}")
