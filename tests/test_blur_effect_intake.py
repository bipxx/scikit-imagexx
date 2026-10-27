"""Declared safe intake refusals, separate from frozen original parity."""
import numpy as np
import pytest


@pytest.mark.parametrize('shape', [(0,), (0, 7), (7, 0), (5, 7, 0)])
def test_blur_zero_spatial_axes_refuse_before_filtering(shape, monkeypatch):
    import skimagexx as xx
    from skimagexx import _blur_effect
    def forbidden(*args, **kwargs):
        raise AssertionError('zero-axis input reached filtering')
    monkeypatch.setattr(_blur_effect.ndi, 'uniform_filter1d', forbidden)
    monkeypatch.setattr(_blur_effect.ndi, 'convolve', forbidden)
    monkeypatch.setattr(xx._core, 'uniform_filter1d', forbidden)
    monkeypatch.setattr(xx._core, 'correlate_sparse', forbidden)
    with pytest.raises(ValueError, match='positive spatial axes'):
        xx.measure.blur_effect(np.empty(shape, dtype=np.float64))
