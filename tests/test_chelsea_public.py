"""Exact packaged Chelsea sample, alias identity and offline resource faults."""

from hashlib import sha256
from importlib.resources import files
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from skimage import data as original

import skimagexx as xx
from skimagexx import _data_chelsea
from skimagexx._compat import FallbackWarning


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "coverage/contracts/chelsea-cat-v1.json"


def test_chelsea_packaged_bytes_and_origin():
    expected = json.loads(MANIFEST.read_bytes())
    package = files("skimagexx").joinpath("_data_assets")
    raw = package.joinpath("chelsea.rgb").read_bytes()
    cc0 = package.joinpath("chelsea.CC0.txt").read_text(encoding="utf-8")
    assert "Stefan van der Walt, CC0 1.0 Universal" in cc0
    assert len(raw) == expected["resource"]["bytes"]
    assert sha256(raw).hexdigest() == expected["resource"]["sha256"]
    for operating_system in ("windows", "linux"):
        path = ROOT / "tests/fixtures" / f"chelsea_original_{operating_system}_v1.json"
        assert sha256(path.read_bytes()).hexdigest() == expected["original_captures"][operating_system]
        receipt = json.loads(path.read_bytes())
        assert receipt["observed"]["decoded_sha256"] == expected["resource"]["sha256"]
        assert [row["id"] for row in receipt["outcomes"]] == expected["original_captures"]["states"]


def test_chelsea_installed_strict_alias_and_layout():
    assert xx.data.chelsea is xx.data.cat is _data_chelsea.chelsea is _data_chelsea.cat
    assert original.chelsea is original.cat
    assert inspect.signature(xx.data.chelsea) == inspect.signature(original.chelsea)
    previous = xx.set_fallback_enabled(False)
    try:
        canonical, alias = xx.data.chelsea(), xx.data.cat()
        expected = original.chelsea()
        assert type(canonical) is type(alias) is type(expected) is np.ndarray
        assert canonical.dtype == alias.dtype == expected.dtype == np.dtype("uint8")
        assert canonical.shape == alias.shape == expected.shape == (300, 451, 3)
        assert canonical.strides == alias.strides == expected.strides == (1353, 3, 1)
        for image in (canonical, alias):
            assert image.flags.writeable and image.flags.owndata and image.flags.c_contiguous
            assert image.base is None and image.tobytes() == expected.tobytes()
        assert not np.shares_memory(canonical, alias)
        canonical[0, 0, 0] ^= 255
        assert alias.tobytes() == expected.tobytes()
    finally:
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize("fault", ("missing", "short", "wrong_digest"))
def test_chelsea_resource_faults_follow_existing_policy(fault, tmp_path, monkeypatch):
    path = tmp_path / "chelsea.rgb"
    if fault == "short":
        path.write_bytes(b"corrupt")
    elif fault == "wrong_digest":
        path.write_bytes(bytes(405900))
    monkeypatch.setattr(_data_chelsea, "_RESOURCE", path)
    calls = []

    def offline_stub():
        calls.append("chelsea")
        raise ConnectionError("controlled offline Chelsea provider")

    monkeypatch.setattr(original, "chelsea", offline_stub)
    previous = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            _data_chelsea.chelsea()
        assert not calls
        xx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match="skimage.data.chelsea"):
            with pytest.raises(ConnectionError, match="controlled offline Chelsea provider"):
                _data_chelsea.cat()
        assert calls == ["chelsea"]
    finally:
        xx.set_fallback_enabled(previous)
