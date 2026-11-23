"""Exact frozen planar hull inputs and observable array snapshots."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def contract():
    path = Path(__file__).resolve().parents[2] / "coverage/contracts/convex-hull-image-v1.json"
    return json.loads(path.read_bytes())


def fixed_cases():
    for case in contract()["fixed_cases"]:
        h, w = case["shape"]
        image = np.zeros((h, w), bool)
        if case["recipe"] == "concave":
            image[h // 6:5 * h // 6, w // 7:w // 3] = True
            image[2 * h // 3:5 * h // 6, w // 7:6 * w // 7] = True
            image[h // 3, w - w // 9 - 1] = True
        elif case["recipe"] == "sparse":
            k = np.arange(157)
            image[(37 * k + 5) % h, (63 * k + 7) % w] = True
            image[0, w // 2] = image[h - 1, w // 3] = True
        else:
            image[14:19, 11:w - 19] = True
            image[h - 29:h - 23, 21:w - 31] = True
            image[14:h - 17, 11:16] = True
            image[22:h - 23, w - 28:w - 19] = True
        yield case["id"], image, dict(
            offset_coordinates=case["offset"], include_borders=case["borders"],
        )


def controls():
    point = np.zeros((8, 9), bool)
    point[0, 1] = True
    line = np.zeros_like(point)
    line[3, 2:7] = True
    triangle = np.zeros((7, 7), bool)
    triangle[1, 1] = triangle[1, 5] = triangle[5, 1] = True
    yield "single-diamond", point, {}
    yield "single-no-offset", point, {"offset_coordinates": False}
    yield "line-diamond", line, {}
    yield "line-no-offset", line, {"offset_coordinates": False}
    yield "triangle-include", triangle, {"offset_coordinates": False}
    yield "triangle-exclude", triangle, {"offset_coordinates": False, "include_borders": False}
    yield "triangle-tolerance-zero", triangle, {"offset_coordinates": False, "tolerance": 0.0}
    yield "triangle-offset", triangle, {}
    yield "array-zero", np.zeros((5, 8), bool), {}
    yield "shape-empty", np.empty((0, 8), bool), {}
    yield "strided", triangle[::-1, ::2], {}
    yield "tagged-bool", triangle.view(np.dtype(bool, metadata={"name": "hull"})), {}
    yield "binary-uint8", triangle.astype(np.uint8), {}


def snapshot(array):
    return dict(
        type=type(array).__module__ + "." + type(array).__name__, dtype=array.dtype.str,
        metadata=None if array.dtype.metadata is None else dict(array.dtype.metadata),
        shape=list(array.shape), strides=list(array.strides), owns=array.flags.owndata,
        c=array.flags.c_contiguous, f=array.flags.f_contiguous, aligned=array.flags.aligned,
        writeable=array.flags.writeable, base_type=type(array.base).__name__,
        sha256=sha256(array.tobytes()).hexdigest(),
    )
