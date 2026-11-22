"""Installed public camera and coffee companions against the frozen stock pixels."""

from hashlib import sha256
from importlib.resources import files
import inspect
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _data_camera_coffee


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "coverage/contracts/data-camera-coffee-assets-v1.json").read_bytes())


@pytest.mark.parametrize("name", ("camera", "coffee"))
def test_camera_coffee_installed_public(name):
    asset = CONTRACT["assets"][name]
    record = asset["representation"]
    target = getattr(xx.data, name)
    assert target is getattr(_data_camera_coffee, name)
    assert str(inspect.signature(target)) == "()"
    previous = xx.set_fallback_enabled(False)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            first, second = target(), target()
            first.flat[0] ^= 255
            third = target()
            with pytest.raises(TypeError):
                target(None)
            with pytest.raises(TypeError):
                target(image=None)
        assert not captured
    finally:
        xx.set_fallback_enabled(previous)
    for result in (first, second, third):
        assert type(result) is np.ndarray and result.dtype.str == record["dtype"]
        assert list(result.shape) == record["shape"] and list(result.strides) == record["strides"]
        assert result.base is None and result.flags.owndata and result.flags.writeable
        assert result.flags.c_contiguous and result.flags.aligned
    assert not np.shares_memory(first, second) and not np.shares_memory(first, third)
    assert not np.shares_memory(second, third)
    assert sha256(second.tobytes()).hexdigest() == sha256(third.tobytes()).hexdigest() == record["decoded_sha256"]
    assert sha256(first.tobytes()).hexdigest() != record["decoded_sha256"]


def test_camera_coffee_installed_resources():
    package = files("skimagexx").joinpath("_data_assets")
    notice = package.joinpath(CONTRACT["rights_notice"]).read_text(encoding="utf-8")
    for asset in CONTRACT["assets"].values():
        assert asset["member"] in notice and asset["upstream_source"] in notice
        assert all(value in notice for value in asset["rights"].values())
        raw = package.joinpath(asset["member"]).read_bytes()
        assert len(raw) == asset["representation"]["bytes"]
        assert sha256(raw).hexdigest() == asset["representation"]["decoded_sha256"]
