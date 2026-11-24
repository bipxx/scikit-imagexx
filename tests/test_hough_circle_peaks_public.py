import inspect
import sys

import numpy as np
import pytest

from hough_circle_peaks_reference import (
    MODERN_ORDER_FIELDS, cases, circle_fields, observe, profile_rows, same_host_order,
)


if sys.platform == 'win32':
    PROFILE = 'windows'
elif np.__version__ == '1.24.4':
    PROFILE = 'linux-minimum'
else:
    PROFILE = 'linux-modern'


@pytest.fixture(scope='module')
def prepared_circle_fields():
    from skimage.transform import hough_circle

    return dict(circle_fields(np, hough_circle))


@pytest.mark.parametrize('case_id', [row['id'] for row in profile_rows(PROFILE)])
def test_hough_circle_peaks_exact_public(case_id, prepared_circle_fields, monkeypatch):
    from skimage import transform as original
    from skimage.transform import hough_transform
    import skimagexx as xx
    from skimagexx import _core
    from skimagexx._hough_circle_peaks import hough_circle_peaks as owned

    public = xx.transform.hough_circle_peaks
    stock = original.hough_circle_peaks
    entry = _core.hough_circle_peak_indices
    assert public is owned
    assert stock is hough_transform.hough_circle_peaks
    assert inspect.signature(public) == inspect.signature(stock)
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    args, row = next((args, row) for args, row in cases(np, prepared_circle_fields, PROFILE) if row['id'] == case_id)
    compat = row['scope'] in ('error', 'compatibility')
    options = row['options']
    if PROFILE == 'linux-modern' and case_id in MODERN_ORDER_FIELDS:
        row = same_host_order(stock, np, args, row)
    native_calls = []
    provider_calls = []

    def trace_native(volume, x, y, threshold):
        native_calls.append((volume is args[0], x, y, threshold))
        return entry(volume, x, y, threshold)

    def trace_provider(volume, radii, **controls):
        assert compat
        assert volume is args[0] and radii is args[1]
        assert controls == {
            'min_xdistance': 1, 'min_ydistance': 1, 'threshold': None, 'num_peaks': np.inf,
            'total_num_peaks': np.inf, 'normalize': False,
        } | options
        provider_calls.append(True)
        return stock(volume, radii, **controls)

    monkeypatch.setattr(_core, 'hough_circle_peak_indices', trace_native)
    monkeypatch.setattr(original, 'hough_circle_peaks', trace_provider)
    previous = xx.set_fallback_enabled(False)
    try:
        if compat:
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                public(*args, **options)
            assert not native_calls and not provider_calls
            xx.set_fallback_enabled(True)
        observe(public, np, args, row, fallback_warning=compat)
    finally:
        xx.set_fallback_enabled(previous)
    assert provider_calls == ([True] if compat else [])
    assert native_calls == ([] if compat else [
        (True, options.get('min_xdistance', 1), options.get('min_ydistance', 1), options.get('threshold')),
    ])
