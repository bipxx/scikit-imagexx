"""Frozen stock oracles and compatibility boundaries for convex hull objects."""

from hashlib import sha256
import inspect
import json
from pathlib import Path
from runpy import run_path
from unittest import TestCase
from unittest.mock import patch
import warnings

import numpy as np
import skimagexx
from skimagexx import _convex_hull_object as route
from skimagexx import morphology


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/convex_hull_object_026.json").read_bytes())
ADAPTER = run_path(str(ROOT / "benchmarks/convex_hull_object.py"))


def signature(value):
    return {
        "type": type(value).__module__ + "." + type(value).__name__,
        "dtype": value.dtype.str,
        "metadata": None if value.dtype.metadata is None else dict(value.dtype.metadata),
        "shape": list(value.shape),
        "strides": list(value.strides),
        "owns": value.flags.owndata,
        "c": value.flags.c_contiguous,
        "f": value.flags.f_contiguous,
        "aligned": value.flags.aligned,
        "writeable": value.flags.writeable,
        "base_type": type(value.base).__name__,
        "sha256": sha256(value.tobytes()).hexdigest(),
    }


def cases():
    fixture = ADAPTER["fixture"]
    for name, recipe, shape, kwargs in ADAPTER["FIXED"]:
        yield name, fixture(recipe, shape), kwargs
    diagonal = fixture("diagonal", (13, 17))
    overlap = fixture("opposite-corners-overlap", (21, 26))
    prior = np.zeros((11, 14), bool)
    prior[1:5, 1] = prior[4, 1:5] = prior[5:9, 5] = prior[8, 5:10] = True
    colored = prior.astype(np.uint8)
    colored[:, 3:] *= 2
    yield from (
        ("objects-conn-1", prior, {"connectivity": 1}),
        ("objects-conn-2", prior, {}),
        ("objects-zero", np.zeros((7, 8), bool), {}),
        ("multilevel-object", colored, {}),
        ("objects-invalid-conn", prior, {"connectivity": 0}),
        ("zero-height", np.empty((0, 6), bool), {}),
        ("one-dim-default", np.ones(3, bool), {}),
        ("three-dim-invalid-conn", np.ones((2, 2, 2), bool), {"connectivity": 0}),
        ("one-pixel", np.ones((1, 1), bool), {}),
        ("binary-uint8-overlap", overlap.astype(np.uint8), {}),
        ("float-labels", overlap.astype(np.float64) * 1.5, {}),
        ("tagged", diagonal.view(np.dtype(bool, metadata={"key": "hull"})), {}),
        ("strided", diagonal[:, ::-1], {}),
        ("noncanonical-storage", (diagonal.astype(np.uint8) * 2).view(np.bool_), {}),
    )


class TestConvexHullObject(TestCase):
    def test_frozen_contract(self):
        previous = skimagexx.set_fallback_enabled(False)
        self.addCleanup(skimagexx.set_fallback_enabled, previous)
        function = morphology.convex_hull_object
        assert str(inspect.signature(function)) == "(image, *, connectivity=2)"
        for name, image, kwargs in cases():
            with self.subTest(case=name), warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                expected = FIXTURE["rows"][name]
                initial = signature(image)
                assert initial == expected["input"]
                if name in FIXTURE["compatibility"]:
                    with patch.object(route, "fallback", wraps=route.fallback) as compatibility:
                        with self.assertRaisesRegex(NotImplementedError, "fallback is disabled"):
                            function(image, **kwargs)
                        compatibility.assert_called_once()
                        assert compatibility.call_args.args[0:2] == ("morphology", "convex_hull_object")
                        assert compatibility.call_args.args[2] is image
                elif "error" in expected:
                    with self.assertRaises(ValueError) as failure:
                        function(image, **kwargs)
                    assert str(failure.exception) == expected["message"]
                else:
                    result = function(image, **kwargs)
                    assert signature(result) == expected["output"]
                    assert not np.shares_memory(result, image)
                assert signature(image) == initial and not notices
