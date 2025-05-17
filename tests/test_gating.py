import numpy as np
from PIL import Image
import pytest

from video_sampler.gating import BlurGate, PassGate, create_gate
from video_sampler.schemas import EMPTY_GATED_OBJECT, GatedObject


def test_pass_gate_returns_frame():
    gate = PassGate()
    img = Image.new("RGB", (10, 10), color="blue")
    res = gate(img, {"t": 0})
    assert isinstance(res, GatedObject)
    assert len(res.frames) == 1
    assert res.frames[0].frame == img


def test_blur_gate_fft_blurry():
    gate = BlurGate(method="fft", threshold=100)
    img = Image.new("RGB", (10, 10), color="red")
    res = gate(img, {"t": 0})
    assert res == EMPTY_GATED_OBJECT


def test_create_gate_pass_and_blur():
    assert isinstance(create_gate({"type": "pass"}), PassGate)
    assert isinstance(
        create_gate({"type": "blur", "method": "fft", "threshold": 100}), BlurGate
    )


def test_create_gate_unknown():
    with pytest.raises(ValueError):
        create_gate({"type": "unknown"})


def test_create_gate_clip_missing_dependency():
    with pytest.raises(Exception):
        create_gate({"type": "clip"})
