"""Provider-owned sample routes alongside the existing data compatibility API."""

import numpy as np
import pytest
import skimage
import skimagexx as xx
from skimagexx._compat import FallbackWarning


OWNED = ("cat", "chelsea", "brick", "grass", "gravel", "text", "clock", "horse", "camera", "coffee")


def _sample(module, name):
    return getattr(module.data, name)()


def test_data_resource_hot_swap_is_independent_of_upstream(monkeypatch):
    upstream = {}
    for name in OWNED:
        marker = object()
        upstream[name] = marker
        monkeypatch.setattr(skimage.data, name, lambda value=marker: value)
    previous = xx.set_fallback_enabled(False)
    try:
        assert xx.data.cat is xx.data.chelsea
        for name in OWNED:
            assert _sample(skimage, name) is upstream[name]
            assert type(_sample(xx, name)) is np.ndarray
    finally:
        xx.set_fallback_enabled(previous)


def test_other_data_routes_keep_compatibility_policy(monkeypatch):
    marker = object()
    monkeypatch.setattr(skimage.data, "astronaut", lambda: marker)
    previous = xx.set_fallback_enabled(False)
    try:
        assert xx.data.__path__ == []
        with pytest.raises(NotImplementedError, match="skimage.data.astronaut"):
            _sample(xx, "astronaut")
        xx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match="skimage.data.astronaut"):
            assert _sample(xx, "astronaut") is marker
        with pytest.raises(AttributeError, match="_fetchers"):
            getattr(xx.data, "_fetchers")
    finally:
        xx.set_fallback_enabled(previous)
