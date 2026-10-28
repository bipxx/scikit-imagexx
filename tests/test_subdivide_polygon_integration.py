"""Registered B-spline, frozen observations and actual coordinate default adapters."""
import inspect
import json
from pathlib import Path
import sys

import pytest
if __package__:
    from .subdivide_polygon_contract import API, cases
    from .test_subdivide_polygon_public import observe
    from .test_subdivide_polygon_pipeline import NATIVE_IDS
else:
    from subdivide_polygon_contract import API, cases
    from test_subdivide_polygon_public import observe
    from test_subdivide_polygon_pipeline import NATIVE_IDS

ROOT = Path(__file__).parents[1]
CASES = cases()


@pytest.fixture(scope='module')
def product():
    import numpy as np, skimagexx as xx
    return np, xx, json.loads((ROOT / 'coverage/contracts/subdivide-polygon.json').read_bytes())


def test_polygon_registered_exports_and_coordinate_carrier(product):
    np, xx, matrix = product
    from skimagexx import _subdivide_polygon as implementation
    from skimagexx.coverage import native_functions, native_dtypes
    assert xx.measure.subdivide_polygon is implementation.subdivide_polygon
    assert 'subdivide_polygon' in xx.measure.__all__
    assert str(inspect.signature(xx.measure.subdivide_polygon)) == matrix['signature']
    assert API in native_functions() and native_dtypes(API) == ('float64',)
    assert inspect.isbuiltin(implementation._core.subdivide_polygon_apply)


@pytest.mark.parametrize('spec', CASES, ids=[spec['id'] for spec in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_subdivide_registered_public_frozen_contract(product, spec, workers, monkeypatch, tmp_path):
    # The private guard uses the same exported callable and actual builtin;
    # it traps whole-original forwarding and SciPy on every accepted path.
    from skimagexx import _subdivide_polygon as implementation
    assert product[1].measure.subdivide_polygon is implementation.subdivide_polygon
    if __package__:
        from .test_subdivide_polygon_pipeline import test_subdivide_pipeline
    else:
        from test_subdivide_polygon_pipeline import test_subdivide_pipeline
    test_subdivide_pipeline(spec, workers, monkeypatch, tmp_path)


DEFAULTS = (
    ('two', (2., -1.)), ('three-flat', (1., 1., 1.)),
    ('three-peak', (0., 2., 0.)), ('three-below', (0., .25, 0.)),
    ('seven-line', tuple(float(2*i+3) for i in range(7))),
    ('seven-peak', (0., 0., 0., 10., 0., 0., 0.)),
    ('nineteen-line', tuple(float(i) for i in range(19))),
    ('signed-values', (-4., -2., 0., 2.)))


@pytest.mark.parametrize('case', DEFAULTS, ids=[case[0] for case in DEFAULTS])
@pytest.mark.parametrize('workers', (1, 4))
def test_polygon_default_adapter_independent_masks(product, case, workers, monkeypatch):
    np, xx, _ = product
    from skimagexx import _subdivide_polygon as implementation
    from skimagexx.coverage import default_call, subdivide_polygon_default_error, assert_default_parity
    import skimage.measure as public
    import skimage.measure._polygon as original
    _, values = case
    image = np.zeros((len(values), 19), np.float64)
    image[:, 0] = values
    before = image.tobytes()
    args, options = default_call(API, image)
    coords, = args
    assert options == {'degree':2,'preserve_ends':False} and coords.dtype == np.dtype('float64')
    assert np.array_equal(coords[:, 0], np.arange(len(values)))
    assert np.array_equal(coords[:, 1], image[:, 0])
    changed = image.copy(); changed[:, 0] += 5.
    assert not np.array_equal(default_call(API, changed)[0][0], coords)
    # Source masks [3,1]/4 and [1,3]/4, increasing convolution taps.
    expected = np.empty((2*(len(values)-1),2), dtype=np.float64)
    expected[0::2] = .25*coords[1:] + .75*coords[:-1]
    expected[1::2] = .75*coords[1:] + .25*coords[:-1]
    input_before = coords.tobytes()
    call, entry = xx.measure.subdivide_polygon, implementation._core.subdivide_polygon_apply
    from scipy import signal
    code = original.subdivide_polygon.__code__
    calls, accepted = [], []
    previous, prior = xx.get_num_threads(), sys.getprofile()
    def reject(*args, **kwargs): raise AssertionError('default B-spline forwarded to original')
    def profile(frame, event, value):
        if event == 'call' and frame.f_code is code: reject()
        if event == 'c_return' and value is entry: calls.append('subdivide_polygon_apply')
        if event == 'return' and frame.f_code is call.__code__:
            accepted.append(frame.f_locals.get('native_result') is not None)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(signal, 'convolve2d', reject)
            patch.setattr(public, 'subdivide_polygon', reject)
            patch.setattr(original, 'subdivide_polygon', reject)
            sys.setprofile(profile)
            try: actual = call(*args, **options)
            finally: sys.setprofile(prior)
        assert calls == ['subdivide_polygon_apply'] and accepted == [True]
        assert coords.tobytes() == input_before and image.tobytes() == before
        assert not np.shares_memory(actual, coords)
        assert subdivide_polygon_default_error(image, actual, expected) == 0.
        assert_default_parity(API, np.dtype('float64'), actual, expected, image=image)
    finally:
        sys.setprofile(prior)
        xx.set_num_threads(previous)


@pytest.mark.parametrize('reason', ('dtype', 'word', 'storage', 'readonly', 'shape', 'metadata'))
def test_polygon_exact_default_checker_rejects_corruption(product, reason):
    np, _, _ = product
    from skimagexx.coverage import subdivide_polygon_default_error
    image = np.zeros((3, 19))
    expected = np.array([[.25, 0.], [.75, 0.], [1.25, 0.], [1.75, 0.]])
    actual = expected.copy()
    if reason == 'dtype': actual = actual.astype(np.float32)
    elif reason == 'word': actual[0, 0] = -0.
    elif reason == 'storage': actual = actual.view()
    elif reason == 'readonly': actual.flags.writeable = False
    elif reason == 'shape': actual = actual[:1].copy()
    else: actual = actual.astype(np.dtype('float64', metadata={'changed': True}))
    with pytest.raises(AssertionError): subdivide_polygon_default_error(image, actual, expected)
