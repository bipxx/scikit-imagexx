"""Deterministic input recipes and snapshots for the frozen thin contract."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


CONTRACT = Path(__file__).resolve().parents[2] / "coverage/contracts/thin-v1.json"


def contract():
    return json.loads(CONTRACT.read_bytes())


def fixture(spec):
    h, w = spec["shape"]
    r, c = np.ogrid[:h, :w]
    if spec["recipe"] == "block":
        image = ((r >= h // 8) & (r < h - h // 8) & (c >= w // 8) & (c < w - w // 8))
        image |= (c == w // 2) & (r < h // 3)
    elif spec["recipe"] == "checkerboard":
        image = (r + c) % 2 == 0
    else:
        image = np.where((r % 17 < 5) | (c % 23 < 7), 1 + (r + 9 * c) % 251, 0)
    return image.astype(spec["dtype"])


def fixed_cases():
    for spec in contract()["fixed_cases"]:
        yield spec["id"], fixture(spec), spec["iterations"]


def controls():
    image = fixture(contract()["fixed_cases"][0])
    readonly = image.copy()
    readonly.flags.writeable = False
    noncanonical = np.zeros((4, 7), np.uint8)
    noncanonical[1:3, 1:6] = np.arange(1, 6, dtype=np.uint8)
    yield "cap-0", image, 0
    yield "cap-1", image, 1
    yield "cap-2", image, 2
    yield "negative-cap", image, -1
    yield "fractional-cap", image, 1.5
    yield "zero-field", np.zeros_like(image), None
    yield "edge-full", np.ones((5, 8), bool), None
    yield "one-row", np.ones((1, 8), bool), None
    yield "one-column", np.ones((8, 1), bool), None
    yield "readonly", readonly, None
    yield "tagged-bool", image.view(np.dtype(bool, metadata={"kind": "thin"})), None
    yield "equivalent-u8", image.astype(np.uint8).view(np.dtype("uint8").newbyteorder("=")), None
    yield "strided", image[:, ::2], None
    yield "noncanonical-bool", noncanonical.view(bool), 1
    yield "empty", np.empty((0, 4), bool), None
    yield "rank-one", image.ravel(), None


def snapshot(array):
    return dict(
        type=type(array).__module__ + "." + type(array).__name__, shape=list(array.shape),
        strides=list(array.strides), dtype=array.dtype.str,
        metadata=None if array.dtype.metadata is None else dict(array.dtype.metadata),
        c=array.flags.c_contiguous, f=array.flags.f_contiguous, owns=array.flags.owndata,
        aligned=array.flags.aligned, writeable=array.flags.writeable, base_type=type(array.base).__name__,
        sha256=sha256(array.tobytes()).hexdigest(),
    )
