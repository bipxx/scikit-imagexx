"""Compatibility availability/policy only; no future API is native yet."""
import pytest
from skimage import future as original
import skimagexx as xx
from tools.inventory import MODULES


def test_future_exports_and_strict_policy():
    assert set(xx.future.__all__)==set(original.__all__)
    assert set(xx.future.__all__)<=set(dir(xx.future))
    assert 'future' in MODULES
    previous=xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError):xx.future.fit_segmenter(None,None,None)
        with pytest.raises(NotImplementedError):xx.future.TrainableSegmenter
        xx.set_fallback_enabled(True)
        with pytest.warns(xx.FallbackWarning):assert xx.future.TrainableSegmenter is original.TrainableSegmenter
    finally:xx.set_fallback_enabled(previous)


def test_future_function_dispatch_preserves_arguments(monkeypatch):
    seen=[];model=object()
    def fit(labels,features,clf):
        seen.append((labels,features,clf));return clf
    monkeypatch.setattr(original,'fit_segmenter',fit)
    previous=xx.set_fallback_enabled(True)
    try:
        with pytest.warns(xx.FallbackWarning):actual=xx.future.fit_segmenter('labels',features='features',clf=model)
        assert actual is model and seen==[('labels','features',model)]
    finally:xx.set_fallback_enabled(previous)
