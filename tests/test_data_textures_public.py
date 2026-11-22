"""Installed exact package bytes and fallback policy for the texture family."""

from hashlib import sha256
from importlib.resources import files
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from skimage import data as original

import skimagexx as xx
from skimagexx import _data_textures
from skimagexx._compat import FallbackWarning


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "coverage/contracts/data-texture-assets-v1.json").read_bytes())


@pytest.mark.parametrize("name", tuple(MANIFEST["assets"]))
def test_texture_installed_strict(name):
    record = MANIFEST["assets"][name]["representation"]
    public = getattr(xx.data, name)
    target = getattr(_data_textures, name)
    supplier = getattr(original, name)
    assert public is target
    assert str(inspect.signature(public)) == str(inspect.signature(supplier)) == "()"
    before = xx.set_fallback_enabled(False)
    try:
        first, second = public(), public()
    finally:
        xx.set_fallback_enabled(before)
    for array in (first, second):
        assert type(array) is np.ndarray and array.dtype == np.uint8
        assert list(array.shape) == record["shape"] and list(array.strides) == record["strides"]
        assert array.flags.owndata and array.flags.writeable and array.flags.c_contiguous
        assert array.base is None and sha256(array.tobytes()).hexdigest() == record["decoded_sha256"]
    assert not np.shares_memory(first, second)
    first[0, 0] ^= 255
    assert sha256(second.tobytes()).hexdigest() == record["decoded_sha256"]


@pytest.mark.parametrize("name", tuple(MANIFEST["assets"]))
@pytest.mark.parametrize("fault", ("missing", "truncated", "wrong-digest"))
def test_texture_bad_installed_asset_uses_policy(name, fault, tmp_path, monkeypatch):
    path = tmp_path / (name + ".gray")
    if fault == "truncated":
        path.write_bytes(b"corrupt")
    elif fault == "wrong-digest":
        height, width = MANIFEST["assets"][name]["representation"]["shape"]
        path.write_bytes(bytes(height * width))
    monkeypatch.setattr(_data_textures, "_PACKAGE", tmp_path)
    recovered = []

    def offline():
        recovered.append(name)
        raise ConnectionError("controlled offline resource provider")

    monkeypatch.setattr(original, name, offline)
    target = getattr(_data_textures, name)
    before = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            target()
        assert not recovered
        xx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match="skimage.data." + name):
            with pytest.raises(ConnectionError, match="controlled offline"):
                target()
        assert recovered == [name]
    finally:
        xx.set_fallback_enabled(before)


def test_texture_packaged_bytes_and_attribution():
    package = files("skimagexx").joinpath("_data_assets")
    rights = package.joinpath("textures.RIGHTS.txt").read_text(encoding="utf-8")
    for name, asset in MANIFEST["assets"].items():
        assert name + ".gray" in rights
        assert asset["rights"]["source_origin"] in rights
        resource = package.joinpath(name + ".gray").read_bytes()
        assert sha256(resource).hexdigest() == asset["representation"]["decoded_sha256"]
    for profile, pins in MANIFEST["source_pins"].items():
        path = ROOT / "tests/fixtures" / ("texture_assets_original_" + profile + "_v1.json")
        assert sha256(path.read_bytes()).hexdigest() == pins["observation_sha256"]
        observed = json.loads(path.read_bytes())
        assert observed["assets"].keys() == MANIFEST["assets"].keys()
        for name, item in observed["assets"].items():
            assert item["observed"] == MANIFEST["assets"][name]["representation"]
