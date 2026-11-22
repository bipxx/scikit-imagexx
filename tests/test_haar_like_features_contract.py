"""Frozen known-safe Haar public results and malformed-input boundaries."""

import hashlib
import importlib
import importlib.util
import inspect
import os
import sys
import types
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import haar_feature_cases as recipe

_FIXTURE = Path(__file__).parent / "fixtures/haar_like_features_original_minimum_v2.json"
_CAPTURE_SHA = "97ee6044b8fe57ce516b724ed5cf787a7e324f3c1c2d12563e9860166c455352"
_SPECS = {spec["id"]: spec for spec in recipe.cases()}
_REFERENCE = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_reference_authority():
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _CAPTURE_SHA
    recipe_path = Path(recipe.__file__)
    assert hashlib.sha256(recipe_path.read_bytes()).hexdigest() == _REFERENCE["recipe_sha256"]
    excluded = _REFERENCE["unsafe_exclusions"]
    assert excluded == ["value-subset-few-rectangles"]
    assert set(_SPECS) == {row["id"] for row in _REFERENCE["cases"]} | set(excluded)


@pytest.fixture(scope="module")
def feature():
    extension = os.environ.get("HAAR_FEATURES_CORE")
    if extension:
        name = "_haar_features_private"
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/skimagexx")]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(name + "._core", extension)
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        package._core = core
        sys.modules[spec.name] = core
        owner = importlib.import_module(name + "._haar_like_features")
        return owner, core, importlib.import_module(name + "._compat")

    public = importlib.import_module("skimagexx.feature")
    owner = importlib.import_module("skimagexx._haar_like_features")
    for name in ("haar_like_feature_coord", "haar_like_feature"):
        assert getattr(public, name) is getattr(owner, name)
        assert public.__all__.count(name) == 1
    core = importlib.import_module("skimagexx._core")
    return public, core, importlib.import_module("skimagexx._compat")


@pytest.mark.parametrize("row", _REFERENCE["cases"], ids=lambda row: row["id"])
def test_public_haar_reference(feature, row):
    spec = _SPECS[row["id"]]
    args, kwargs = recipe.materialize(spec, np)
    is_value = row["which"] == "value"
    product, _, _ = feature
    target = product.haar_like_feature if is_value else product.haar_like_feature_coord
    before = recipe.fingerprint(args[0]) if is_value else None
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        if "error" in row:
            import builtins
            with pytest.raises(getattr(builtins, row["error"]["kind"])) as actual:
                target(*args, **kwargs)
            assert str(actual.value) == row["error"]["text"]
        else:
            result = target(*args, **kwargs)
            assert type(result) is (np.ndarray if is_value else tuple)
            assert [recipe.summary(array, np) for array in ((result,) if is_value else result)] == row["output"]
        assert [dict(kind=entry.category.__name__, text=str(entry.message)) for entry in observed] == row["warnings"]
    if is_value:
        assert (recipe.fingerprint(args[0]) == before) == row["input_content_preserved"]


def test_short_or_out_of_bounds_rectangles_stop_before_read(feature):
    opts = _SPECS["value-subset-few-rectangles"]
    args, kwargs = recipe.materialize(opts, np)
    with pytest.raises(ValueError, match="too few rectangles"):
        feature[0].haar_like_feature(*args, **kwargs)
    coords = np.empty(1, dtype=object)
    coords[0] = [[(0, 0), (0, 0)], [(0, 1), (0, 4)]]
    with pytest.raises(ValueError, match="outside"):
        feature[0].haar_like_feature(np.zeros((4, 4)), 0, 0, 4, 4, np.array(["type-2-x"]), coords)


def test_truncated_and_unverified_roi_stop_before_copy(feature):
    with pytest.raises(ValueError, match="outside"):
        feature[0].haar_like_feature(np.zeros((3, 3)), 1, 1, 4, 4)
    invalid = np.lib.stride_tricks.as_strided(np.uint8([1]), shape=(4, 4), strides=(4, 1))
    with pytest.raises(ValueError, match="unverified backing buffer"):
        feature[0].haar_like_feature(invalid, 0, 0, 4, 4)


@pytest.mark.parametrize("fallback_enabled", [False, True])
@pytest.mark.parametrize("case,expected", [
    ("coord-tiny", "haar_coordinates"),
    ("value-small", "haar_generated_values"),
    ("value-subset", "haar_selected_values"),
])
def test_compiled_route_ignores_fallback_switch(feature, fallback_enabled, case, expected):
    product, core, compat = feature
    args, kwargs = recipe.materialize(_SPECS[case], np)
    target = product.haar_like_feature_coord if case.startswith("coord") else product.haar_like_feature
    entry = getattr(core, expected)
    assert inspect.isbuiltin(entry) and entry.__module__ == core.__name__
    observed = []
    prior_profile = sys.getprofile()
    previous = compat.set_fallback_enabled(fallback_enabled)

    def record(frame, event, value):
        if event == "c_return" and value is entry:
            observed.append(value.__name__)

    try:
        sys.setprofile(record)
        target(*args, **kwargs)
    finally:
        sys.setprofile(prior_profile)
        compat.set_fallback_enabled(previous)
    assert observed == [expected]


@pytest.mark.parametrize("case", ["coord-tiny", "coord-zero-width"])
def test_coordinate_arrays_own_native_object_storage(feature, monkeypatch, case):
    function = feature[0].haar_like_feature_coord
    row = next(row for row in _REFERENCE["cases"] if row["id"] == case)
    args, kwargs = recipe.materialize(_SPECS[case], np)

    def unexpected_empty(*args, **kwargs):
        pytest.fail("coordinate binding used the Python NumPy allocator")

    with monkeypatch.context() as context:
        context.setattr(np, "empty", unexpected_empty)
        arrays = function(*args, **kwargs)
    assert type(arrays) is tuple and len(arrays) == 2
    assert all(type(array) is np.ndarray and array.base is None for array in arrays)
    assert arrays[0] is not arrays[1] and not np.shares_memory(*arrays)
    assert [recipe.summary(array, np) for array in arrays] == row["output"]


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
                                   "int64", "uint64", "float32", "float64"])
@pytest.mark.parametrize("selected", [False, True])
def test_equivalent_and_tagged_integral_descriptors(feature, dtype, selected):
    case = "value-subset" if selected else ("value-unsupported-bool" if dtype == "bool" else f"value-dtype-{dtype}")
    args, options = recipe.materialize(dict(_SPECS[case], dtype=dtype), np)
    image = args[0]
    before = recipe.fingerprint(image)
    canonical = image.dtype
    equivalent = canonical.newbyteorder(">").newbyteorder("=")
    tagged = np.dtype(canonical, metadata={"units": "linear", "source": "parity"})
    assert equivalent is not canonical and equivalent == canonical
    target = feature[0].haar_like_feature
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        expected = target(*args, **options)
        assert expected.dtype == np.dtype(dtype if canonical.kind == "f" else "int64")
        assert expected.dtype.metadata is None
        for descriptor in (equivalent, tagged):
            view = image.view(descriptor)
            actual = target(view, *args[1:], **options)
            assert recipe.summary(actual, np) == recipe.summary(expected, np)
            assert actual.dtype.metadata is None and actual.base is None
            assert actual.flags.aligned and actual.flags.c_contiguous
            assert not np.shares_memory(actual, view)
    assert not observed
    assert recipe.fingerprint(image) == before
