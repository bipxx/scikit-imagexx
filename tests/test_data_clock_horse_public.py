"""Clock and horse package resources and existing compatibility policy."""

from hashlib import sha256
from importlib.resources import files
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from skimage import data as original

import skimagexx as xx
from skimagexx import _data_clock_horse
from skimagexx._compat import FallbackWarning


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "coverage/contracts/data-clock-horse-v1.json").read_bytes())


@pytest.mark.parametrize("name", tuple(MANIFEST["assets"]))
def test_installed_clock_horse_exact(name):
    asset = MANIFEST["assets"][name]
    function = getattr(xx.data, name)
    assert function is getattr(_data_clock_horse, name)
    assert str(inspect.signature(function)) == str(inspect.signature(getattr(original, name))) == "()"
    previous = xx.set_fallback_enabled(False)
    try:
        first, second = function(), function()
    finally:
        xx.set_fallback_enabled(previous)
    description = asset["metadata"]
    for value in (first, second):
        assert type(value) is np.ndarray and value.dtype.str == description["dtype"]
        assert list(value.shape) == description["shape"] and list(value.strides) == description["strides"]
        assert value.flags.owndata and value.flags.writeable and value.flags.c_contiguous
        assert value.base is None and sha256(value.tobytes()).hexdigest() == description["decoded_sha256"]
    assert not np.shares_memory(first, second)
    first[0, 0] = not first[0, 0]
    assert sha256(second.tobytes()).hexdigest() == description["decoded_sha256"]
    if name == "horse":
        assert second.dtype == np.bool_ and set(second.tobytes()) == {0, 1}
        assert np.count_nonzero(second) == asset["polarity"]["true_count"]
        for location, expected in asset["polarity"]["pixels"].items():
            row, column = map(int, location.split(","))
            assert bool(second[row, column]) is expected


@pytest.mark.parametrize("name", tuple(MANIFEST["assets"]))
@pytest.mark.parametrize("fault", ("missing", "short", "wrong-digest"))
def test_clock_horse_resource_policy(name, fault, tmp_path, monkeypatch):
    asset = MANIFEST["assets"][name]
    resource = tmp_path / Path(asset["packaged_path"]).name
    if fault == "short":
        resource.write_bytes(b"corrupt")
    elif fault == "wrong-digest":
        resource.write_bytes(bytes(asset["resource_bytes"]))
    monkeypatch.setattr(_data_clock_horse, "_PACKAGE", tmp_path)
    reached = []

    def offline():
        reached.append(name)
        raise ConnectionError("controlled offline provider")
    monkeypatch.setattr(original, name, offline)
    function = getattr(_data_clock_horse, name)
    previous = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            function()
        assert not reached
        xx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match="skimage.data." + name):
            with pytest.raises(ConnectionError, match="controlled offline"):
                function()
        assert reached == [name]
    finally:
        xx.set_fallback_enabled(previous)


def test_clock_horse_bundled_files_and_attribution():
    assets = files("skimagexx").joinpath("_data_assets")
    rights = assets.joinpath("clock-horse.RIGHTS.txt").read_text(encoding="utf-8")
    assert "Stefan van der Walt" in rights and "Andreas Preuss" in rights
    for asset in MANIFEST["assets"].values():
        raw = assets.joinpath(Path(asset["packaged_path"]).name).read_bytes()
        assert len(raw) == asset["resource_bytes"]
        assert sha256(raw).hexdigest() == asset["metadata"]["decoded_sha256"]
    for name, report in MANIFEST["captures"].items():
        path = ROOT / "tests/fixtures" / f"clock_horse_original_{name}_v1.json"
        assert sha256(path.read_bytes()).hexdigest() == report["report_sha256"]
        frozen = json.loads(path.read_bytes())
        assert all(frozen["assets"][key]["observed"] == asset["metadata"]
                   for key, asset in MANIFEST["assets"].items())
