"""Offline contract checks for the worker's data entry points."""

from inspect import signature
from pathlib import Path
import warnings

import numpy as np
import pytest
from skimage import data as original
from skimage.data import _fetchers
from skimage.io._plugins import imageio_plugin
from skimage.io.manage_plugins import plugin_store

from skimagexx import FallbackWarning, _core, set_fallback_enabled
from skimagexx import _data_loaders_b as candidate


BUNDLED = ("logo", "microaneurysms", "moon", "page", "retina", "rocket", "stereo_motorcycle")
REMOTE = [
    ("lily", {"data/lily.tif": "multipage_rgb.tif"}),
    ("nickel_solidification", {"data/solidification.tif": "multipage.tif"}),
    ("palisades_of_vogt", {"data/palisades_of_vogt.tif": "multipage.tif"}),
    ("protein_transport", {"data/protein_transport.tif": "multipage_rgb.tif"}),
    ("skin", {"data/skin.jpg": "rocket.jpg"}),
    ("vortex", {"data/pivchallenge-B-B001_1.tif": "multipage.tif",
                "data/pivchallenge-B-B001_2.tif": "multipage_rgb.tif"}),
]


@pytest.fixture
def strict():
    previous = set_fallback_enabled(False)
    try:
        yield
    finally:
        set_fallback_enabled(previous)


def _compare(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            _compare(got, want)
        return
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.dtype.str == expected.dtype.str
    assert actual.dtype.metadata == expected.dtype.metadata
    assert actual.strides == expected.strides
    for flag in ("OWNDATA", "WRITEABLE", "C_CONTIGUOUS", "F_CONTIGUOUS", "ALIGNED"):
        assert actual.flags[flag] == expected.flags[flag]
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("name", BUNDLED + tuple(name for name, _ in REMOTE))
def test_signature(name):
    assert signature(getattr(candidate, name)) == signature(getattr(original, name))


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_hot_swap_is_exact_and_returns_fresh_arrays(name, strict, monkeypatch):
    reference = getattr(original, name)()
    monkeypatch.setattr(original, name, getattr(candidate, name))
    actual = getattr(original, name)()
    _compare(actual, reference)
    images = actual if isinstance(actual, tuple) else (actual,)
    for image in images:
        image.flat[0] = 0 if image.flat[0] else 1
    _compare(getattr(original, name)(), reference)


@pytest.mark.parametrize("name,files", REMOTE)
def test_remote_delegates_stock_fetch_and_decode_without_network(name, files, monkeypatch):
    calls = []
    installed_data = Path(_fetchers.__file__).parent

    def use_local_file(key):
        calls.append(key)
        return str(installed_data / files[key])

    monkeypatch.setattr(_fetchers, "_fetch", use_local_file)
    expected = getattr(original, name)()
    assert calls == list(files)
    calls.clear()
    previous = set_fallback_enabled(True)
    try:
        with pytest.warns(FallbackWarning, match=rf"skimage\.data\.{name}"):
            actual = getattr(candidate, name)()
    finally:
        set_fallback_enabled(previous)
    _compare(actual, expected)
    assert calls == list(files)


@pytest.mark.parametrize("name,files", REMOTE)
def test_remote_strict_mode_stops_before_fetch(name, files, strict, monkeypatch):
    def unexpected_fetch(key):
        pytest.fail(f"unexpected fetch: {key}")

    monkeypatch.setattr(_fetchers, "_fetch", unexpected_fetch)
    with pytest.raises(NotImplementedError, match=rf"skimage\.data\.{name}"):
        getattr(candidate, name)()


def test_plugin_override_obeys_stock_and_fallback_policy(monkeypatch):
    def alternate(filename):
        assert isinstance(filename, str)
        return np.full((2, 3), 41, dtype=np.uint8)

    assert plugin_store["imread"][0][1] is imageio_plugin.imread
    monkeypatch.setitem(plugin_store, "imread", [("alternate", alternate)])
    previous = set_fallback_enabled(True)
    try:
        with pytest.warns(FallbackWarning, match="data.moon"):
            _compare(candidate.moon(), original.moon())
        set_fallback_enabled(False)
        with pytest.raises(NotImplementedError, match="data.moon"):
            candidate.moon()
    finally:
        set_fallback_enabled(previous)


def test_corrupt_cache_uses_verified_distribution_asset(monkeypatch, strict):
    shipped = Path(_fetchers.__file__).with_name("moon.png")
    cache_dir = shipped.parent / "unavailable-cache"
    original_read = Path.read_bytes
    reads = []

    def read_bytes(path):
        reads.append(path)
        if path == cache_dir / shipped.name:
            return b"corrupt cached bytes"
        return original_read(path)

    monkeypatch.setattr(_fetchers, "data_dir", str(cache_dir))
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        _compare(candidate.moon(), original.moon())
    assert reads == [cache_dir / shipped.name, shipped]


def test_stale_decoded_entry_is_not_used_for_unverified_source(monkeypatch, strict):
    candidate.moon()
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"invalid on-disk content")
    with pytest.raises(NotImplementedError, match="data.moon"):
        candidate.moon()


@pytest.mark.parametrize("descriptor", ["B", "uint8", "<f4", ">f4"])
def test_native_buffer_preserves_dtype_aliases_and_metadata(descriptor):
    dtype = np.dtype(descriptor, metadata={"source": "offline"})
    source = np.arange(6).astype(dtype).reshape(2, 3).copy()
    result = _core.data_loaders_b_materialize(source.tobytes(), source.dtype, source.shape)
    _compare(result, source)


@pytest.mark.parametrize("shape", [(0, 3), (2, -1), (2, 4), (2, 2), (2**50, 3), ()])
def test_native_rejects_invalid_buffer_shape(shape):
    with pytest.raises(ValueError):
        _core.data_loaders_b_materialize(b"abcdxy", np.dtype("uint8"), shape)


def test_native_rejects_object_dtype():
    with pytest.raises(TypeError):
        _core.data_loaders_b_materialize(b"abcd", np.dtype("O"), (2, 2))
