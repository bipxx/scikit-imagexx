"""Recipes and observable results from the locked two-profile reference."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


CONTRACT = Path(__file__).resolve().parents[2] / "coverage/contracts/match-descriptors-v1.json"


def contract():
    return json.loads(CONTRACT.read_bytes())


def fixed_cases():
    for case in contract()["fixed_cases"]:
        rng = np.random.default_rng(case["seed"])
        left, right = (
            rng.integers(0, 2, size=case[key], dtype=np.uint8).view(np.bool_)
            for key in ("left", "right")
        )
        yield case["id"], left, right, case["kwargs"]


def controls():
    a = np.zeros((2, 4), bool)
    b = np.zeros((3, 4), bool)
    one = np.zeros((1, 4), bool)
    graded = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], bool)
    smaller = np.zeros((1, 7), bool)
    smaller_right = np.array([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]], bool)
    raw_a = np.array([[2, 0, 255], [1, 0, 1], [3, 1, 1]], np.uint8).view(bool)
    raw_b = np.array([[1, 0, 1], [2, 0, 255], [0, 0, 0]], np.uint8).view(bool)
    yield "mutual-first-ties", a, b, {}
    yield "first-ties-no-cross", a, b, {"cross_check": False}
    yield "distance-equality", one, graded, {"max_distance": 0.25}
    yield "distance-next-up", one, graded, {"max_distance": math.nextafter(0.25, math.inf)}
    yield "distance-next-down", one, graded, {"max_distance": math.nextafter(0.25, -math.inf)}
    yield "ratio-equality", one, graded, {"max_ratio": 0.5}
    yield "ratio-next-up", one, graded, {"max_ratio": math.nextafter(0.5, math.inf)}
    yield "ratio-next-down", one, graded, {"max_ratio": math.nextafter(0.5, -math.inf)}
    yield "zero-runner", one, b, {"max_ratio": 0.01}
    yield "only-right", one, graded[:1], {"max_ratio": 0.01}
    yield "ratio-zero", one, b, {"max_ratio": 0.0}
    yield "explicit-hamming", one, graded, {"metric": "hamming"}
    yield "width-seven-distance", smaller, smaller_right, {"max_distance": 3 / 7}
    yield "width-seven-next-up", smaller, smaller_right, {"max_distance": math.nextafter(3 / 7, math.inf)}
    yield "width-seven-ratio", smaller, smaller_right, {"max_ratio": (3 / 7) / (5 / 7)}
    yield "foreign-bool-byte", raw_a, raw_b, {"cross_check": False}
    yield "foreign-bool-byte-mutual", raw_a, raw_b, {}
    yield "mismatched-length", one, smaller, {}
    yield "empty-left-cross", a[:0], b, {}
    yield "empty-left-no-cross", a[:0], b, {"cross_check": False}
    yield "empty-right", a, b[:0], {}
    yield "strided", a[:, ::-1], graded, {}
    yield "fortran", np.asfortranarray(a), graded, {}
    yield "mixed-dtype", one, graded.astype(np.uint8), {}
    yield "euclidean", one, graded, {"metric": "euclidean"}
    yield "arbitrary-ignored-p", one, graded, {"p": None}


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def assert_fixed_inputs(case, left, right):
    assert [digest(left), digest(right)] == case["input_raw_sha256"]


def assert_result(value, case, left, right):
    expected = case["output"]
    assert type(value) is np.ndarray and value.base is None
    assert value.dtype == np.dtype(np.intp) and value.dtype.str == expected["dtype"]
    assert value.shape == tuple(expected["shape"]) and value.strides == tuple(expected["strides"])
    assert value.flags.c_contiguous == expected["c"] and value.flags.f_contiguous == expected["f"]
    assert value.flags.owndata and value.flags.writeable and value.flags.aligned
    assert not np.shares_memory(value, left) and not np.shares_memory(value, right)
    assert digest(value) == expected["raw_sha256"]
