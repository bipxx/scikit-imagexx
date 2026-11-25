"""Offline checks of the data loader leaf's public call contracts."""

from inspect import signature
from os import path
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from skimage import data as original, io
from skimage.data import _fetchers
from skimagexx import _data_loaders_a as candidate
from skimagexx._compat import FallbackWarning, set_fallback_enabled


BUNDLED_ARRAYS = (
    "astronaut", "cell", "coins", "colorwheel", "hubble_deep_field",
    "immunohistochemistry", "lfw_subset",
)
REMOTE_ARRAYS = ("brain", "cells3d", "eagle", "human_mitosis", "kidney")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected cache creation or download")

    monkeypatch.setattr(_fetchers, "_image_fetcher",
                        SimpleNamespace(abspath=_fetchers._DISTRIBUTION_DIR, fetch=forbidden))
    monkeypatch.setattr(_fetchers, "_ensure_cache_dir", forbidden)
    monkeypatch.setattr(candidate, "_verified", {})
    monkeypatch.setattr(candidate, "_decoded", {})


def assert_array_contract(actual, expected):
    assert type(actual) is type(expected) is np.ndarray
    assert (actual.shape, actual.strides, actual.dtype.str) == (
        expected.shape, expected.strides, expected.dtype.str)
    assert actual.dtype.metadata == expected.dtype.metadata
    assert actual.flags.num == expected.flags.num
    assert actual.tobytes() == expected.tobytes()
    assert type(actual.base) is type(expected.base)
    if actual.base is not None:
        assert actual.base.shape == expected.base.shape
        assert actual.base.flags.num == expected.base.flags.num


@pytest.mark.parametrize("name", (
    *BUNDLED_ARRAYS, *REMOTE_ARRAYS, "download_all", "lbp_frontal_face_cascade_filename",
))
def test_signature(name):
    assert signature(getattr(candidate, name)) == signature(getattr(original, name))


@pytest.mark.parametrize("name", BUNDLED_ARRAYS)
def test_bundled_array(name, monkeypatch):
    expected = getattr(original, name)()
    previous = set_fallback_enabled(False)
    try:
        first = getattr(candidate, name)()
        assert name in candidate._decoded
        copy = candidate.data_loaders_a_copy
        copied = []

        def record_copy(*args):
            copied.append(len(args[0]))
            return copy(*args)

        monkeypatch.setattr(candidate, "data_loaders_a_copy", record_copy)
        second = getattr(candidate, name)()
    finally:
        set_fallback_enabled(previous)
    assert copied == [expected.nbytes]
    for actual in (first, second):
        assert_array_contract(actual, expected)
    assert not np.shares_memory(first, second)
    first.flat[0] = 0 if first.flat[0] else 1
    assert second.tobytes() == expected.tobytes()


def test_cascade_path():
    actual = candidate.lbp_frontal_face_cascade_filename()
    expected = original.lbp_frontal_face_cascade_filename()
    assert type(actual) is type(expected) is str
    assert actual == expected
    assert Path(actual).read_bytes().startswith(b"<?xml")


@pytest.mark.parametrize("name", REMOTE_ARRAYS)
def test_offline_cached_remote_route(name, monkeypatch):
    # Verify the real decoder and cache flow with an explicitly supplied local
    # fixture standing in for the registry entry. No remote bytes are required.
    filename = candidate._FILES[name]
    fixture = {"eagle": "astronaut.png", "kidney": "multipage_rgb.tif"}.get(name, "multipage.tif")
    cache = path.join(_fetchers._DISTRIBUTION_DIR, "offline-test-cache")
    cached_path = path.join(cache, "data/" + filename)
    fixture_path = path.join(_fetchers._DISTRIBUTION_DIR, "data", fixture)
    monkeypatch.setattr(_fetchers._image_fetcher, "abspath", cache)
    real_hash, real_stamp, read = _fetchers._has_hash, candidate._stamp, io.imread
    monkeypatch.setattr(_fetchers, "_has_hash",
                        lambda file, digest: file == cached_path or real_hash(file, digest))
    monkeypatch.setattr(candidate, "_stamp",
                        lambda file: real_stamp(fixture_path if file == cached_path else file))
    opened = []

    def read_fixture(file, **kwargs):
        assert file == cached_path
        opened.append(file)
        return read(fixture_path, **kwargs)

    monkeypatch.setattr(io, "imread", read_fixture)
    previous = set_fallback_enabled(False)
    try:
        first = getattr(candidate, name)()
        expected = getattr(original, name)()
        second = getattr(candidate, name)()
    finally:
        set_fallback_enabled(previous)
    assert opened == [cached_path, cached_path]
    assert_array_contract(first, expected)
    assert_array_contract(second, expected)


@pytest.mark.parametrize("name", REMOTE_ARRAYS)
def test_missing_remote_uses_fallback_policy(name, monkeypatch):
    monkeypatch.setattr(_fetchers, "_has_hash", lambda *args: False)
    monkeypatch.setattr(_fetchers, "_image_fetcher", None)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    previous = set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            getattr(candidate, name)()
        set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match=f"skimage.data.{name}"):
            with pytest.raises(ModuleNotFoundError, match="pooch"):
                getattr(candidate, name)()
    finally:
        set_fallback_enabled(previous)


def test_cache_file_replacement_and_precedence(monkeypatch):
    name = "astronaut"
    fixture = original.astronaut()
    cache = path.join(_fetchers._DISTRIBUTION_DIR, "offline-test-cache")
    cached_path = path.join(cache, "data/astronaut.png")
    current = {"stamp": (1,), "valid": True}
    monkeypatch.setattr(_fetchers._image_fetcher, "abspath", cache)
    real_stamp, real_hash, read = candidate._stamp, _fetchers._has_hash, io.imread
    monkeypatch.setattr(candidate, "_stamp",
                        lambda file: current["stamp"] if file == cached_path else real_stamp(file))
    monkeypatch.setattr(_fetchers, "_has_hash",
                        lambda file, digest: current["valid"] if file == cached_path
                        else real_hash(file, digest))
    opened = []

    def read_fixture(file, **kwargs):
        opened.append(file)
        return fixture.copy() if file == cached_path else read(file, **kwargs)

    monkeypatch.setattr(io, "imread", read_fixture)
    assert_array_contract(candidate.astronaut(), fixture)
    assert_array_contract(candidate.astronaut(), fixture)
    assert opened == [cached_path]
    current.update(stamp=(2,), valid=False)
    assert_array_contract(candidate.astronaut(), fixture)
    assert opened == [cached_path, _fetchers._fetch("data/astronaut.png")]


def test_download_all_uses_provider_and_restores_directory(monkeypatch):
    assert candidate.download_all is original.download_all
    keys = {"data/astronaut.png": None, "data/coins.png": None}
    provider = SimpleNamespace(path="initial", registry=keys)
    created, requested = [], []
    monkeypatch.setattr(_fetchers, "_image_fetcher", provider)
    monkeypatch.setattr(_fetchers, "_ensure_cache_dir", lambda target_dir: created.append(target_dir))

    def fetch(key):
        requested.append((provider.path, key))
        return path.join(provider.path, key)

    monkeypatch.setattr(_fetchers, "_fetch", fetch)
    assert candidate.download_all(Path("custom")) is None
    assert created == ["custom"] and requested == [("custom", key) for key in keys]
    assert provider.path == "initial"

    def interrupted(key):
        raise ConnectionError(key)

    monkeypatch.setattr(_fetchers, "_fetch", interrupted)
    with pytest.raises(ConnectionError, match="data/astronaut.png"):
        candidate.download_all("interrupted")
    assert provider.path == "initial"
    monkeypatch.setattr(_fetchers, "_image_fetcher", None)
    with pytest.raises(ModuleNotFoundError, match="pooch"):
        candidate.download_all()


def test_alternate_image_plugin_obeys_fallback_policy(monkeypatch):
    from skimage.io.manage_plugins import plugin_store

    pixels = np.zeros((2, 2), dtype=np.uint8)
    monkeypatch.setitem(plugin_store, "imread", [("alternate", lambda *args: pixels)])
    previous = set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            candidate.astronaut()
        set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match="skimage.data.astronaut"):
            assert candidate.astronaut() is pixels
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize("alias", ["u1", np.ubyte, "<u2", ">u2", np.double])
def test_native_dtype_and_metadata(alias):
    dtype = np.dtype(alias, metadata={"sample": "native"})
    value = np.arange(6, dtype=dtype).reshape(2, 3)
    actual = candidate.data_loaders_a_copy(value.tobytes(), dtype, value.base.shape).reshape(value.shape)
    assert_array_contract(actual, value)


@pytest.mark.parametrize("shape", [(4,), (-1,), (0,), (1 << 62, 8)])
def test_native_rejects_mismatched_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        candidate.data_loaders_a_copy(b"hi", np.dtype("u1"), shape)
