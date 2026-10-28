"""Registered RDP, frozen observations and actual coordinate default adapters."""
import inspect
import json
from pathlib import Path
import sys

import pytest
if __package__:
    from .approximate_polygon_contract import API, cases
    from .test_approximate_polygon_public import observe
    from .approximate_polygon_runtime_reference import RUNTIME_REFERENCE_IDS, runtime_reference
    from .test_approximate_polygon_pipeline import NATIVE_IDS
else:
    from approximate_polygon_contract import API, cases
    from test_approximate_polygon_public import observe
    from approximate_polygon_runtime_reference import RUNTIME_REFERENCE_IDS, runtime_reference
    from test_approximate_polygon_pipeline import NATIVE_IDS

ROOT = Path(__file__).parents[1]
CASES = cases()


@pytest.fixture(scope='module')
def product():
    import numpy as np, skimagexx as xx
    return np, xx, json.loads((ROOT / 'coverage/contracts/approximate-polygon.json').read_bytes())


def test_polygon_registered_exports_and_coordinate_carrier(product):
    np, xx, matrix = product
    from skimagexx import _approximate_polygon as implementation
    from skimagexx.coverage import native_functions, native_dtypes
    assert xx.measure.approximate_polygon is implementation.approximate_polygon
    assert 'approximate_polygon' in xx.measure.__all__
    assert str(inspect.signature(xx.measure.approximate_polygon)) == matrix['signature']
    assert API in native_functions() and native_dtypes(API) == ('float64',)
    assert inspect.isbuiltin(implementation._core.approximate_polygon_apply)


@pytest.mark.parametrize('spec', CASES, ids=[spec['id'] for spec in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_polygon_registered_public_frozen_contract(product, spec, workers, monkeypatch, tmp_path):
    np, xx, matrix = product
    from skimagexx import _approximate_polygon as implementation
    import skimage.measure as public
    import skimage.measure._polygon as original
    expected = next(row for row in matrix['original_observations'][np.__version__] if row['id'] == spec['id'])
    if spec['id'] in RUNTIME_REFERENCE_IDS:
        expected = runtime_reference(spec['id'], np)
    call, entry = xx.measure.approximate_polygon, implementation._core.approximate_polygon_apply
    original_code = original.approximate_polygon.__code__
    calls, accepted = [], []
    previous, prior = xx.get_num_threads(), sys.getprofile()
    def reject(*args, **kwargs):
        raise AssertionError('registered RDP forwarded to original whole API')
    def profile(frame, event, value):
        if event == 'call' and frame.f_code is original_code: reject()
        if event == 'c_return' and value is entry: calls.append('approximate_polygon_apply')
        if event == 'return' and frame.f_code is call.__code__:
            accepted.append(frame.f_locals.get('native_result') is True)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(public, 'approximate_polygon', reject)
            patch.setattr(original, 'approximate_polygon', reject)
            sys.setprofile(profile)
            try: actual = observe(call, spec, np)
            finally: sys.setprofile(prior)
        assert actual == expected
        assert accepted == [spec['id'] in NATIVE_IDS] and len(calls) <= 1
        if spec['id'] in NATIVE_IDS: assert calls == ['approximate_polygon_apply']
        assert xx.get_num_threads() == workers
        (tmp_path / 'actual.json').write_text(json.dumps(dict(
            id=spec['id'], workers=workers, observation=actual, C_returns=calls,
            native_accepted=accepted)) + '\n')
    finally:
        sys.setprofile(prior)
        xx.set_num_threads(previous)


DEFAULTS = (
    ('singleton', (2.,), (0,)),
    ('two', (2., -1.), (0, 1)),
    ('three-flat', (1., 1., 1.), (0, 2)),
    ('three-peak', (0., 2., 0.), (0, 1, 2)),
    ('three-below', (0., .25, 0.), (0, 2)),
    ('seven-line', tuple(float(2 * i + 3) for i in range(7)), (0, 6)),
    ('seven-peak', (0., 0., 0., 10., 0., 0., 0.), (0, 2, 3, 4, 6)),
    ('nineteen-line', tuple(float(i) for i in range(19)), (0, 18)))


@pytest.mark.parametrize('case', DEFAULTS, ids=[case[0] for case in DEFAULTS])
@pytest.mark.parametrize('workers', (1, 4))
def test_polygon_default_adapter_independent_masks(product, case, workers, monkeypatch):
    np, xx, _ = product
    from skimagexx import _approximate_polygon as implementation
    from skimagexx.coverage import default_call, approximate_polygon_default_error, assert_default_parity
    import skimage.measure as public
    import skimage.measure._polygon as original
    _, values, indices = case
    image = np.zeros((len(values), 19), np.float64)
    image[:, 0] = values
    before = image.tobytes()
    args, options = default_call(API, image)
    coords, tolerance = args
    assert tolerance == .5 and options == {} and coords.dtype == np.dtype('float64')
    assert np.array_equal(coords[:, 0], np.arange(len(values)))
    assert np.array_equal(coords[:, 1], image[:, 0])
    changed = image.copy(); changed[:, 0] += 5.
    assert not np.array_equal(default_call(API, changed)[0][0], coords)
    expected = coords[list(indices), :].copy()
    input_before = coords.tobytes()
    call, entry = xx.measure.approximate_polygon, implementation._core.approximate_polygon_apply
    code = original.approximate_polygon.__code__
    calls, accepted = [], []
    previous, prior = xx.get_num_threads(), sys.getprofile()
    def reject(*args, **kwargs): raise AssertionError('default RDP forwarded to original')
    def profile(frame, event, value):
        if event == 'call' and frame.f_code is code: reject()
        if event == 'c_return' and value is entry: calls.append('approximate_polygon_apply')
        if event == 'return' and frame.f_code is call.__code__:
            accepted.append(frame.f_locals.get('native_result') is True)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(public, 'approximate_polygon', reject)
            patch.setattr(original, 'approximate_polygon', reject)
            sys.setprofile(profile)
            try: actual = call(*args, **options)
            finally: sys.setprofile(prior)
        assert calls == ['approximate_polygon_apply'] and accepted == [True]
        assert coords.tobytes() == input_before and image.tobytes() == before
        assert not np.shares_memory(actual, coords)
        assert approximate_polygon_default_error(image, actual, expected) == 0.
        assert_default_parity(API, np.dtype('float64'), actual, expected, image=image)
    finally:
        sys.setprofile(prior)
        xx.set_num_threads(previous)


@pytest.mark.parametrize('reason', ('dtype', 'word', 'storage', 'readonly', 'shape', 'metadata'))
def test_polygon_exact_default_checker_rejects_corruption(product, reason):
    np, _, _ = product
    from skimagexx.coverage import approximate_polygon_default_error
    image = np.zeros((3, 19))
    expected = np.array([[0., 0.], [2., 0.]])
    actual = expected.copy()
    if reason == 'dtype': actual = actual.astype(np.float32)
    elif reason == 'word': actual[0, 0] = -0.
    elif reason == 'storage': actual = actual.view()
    elif reason == 'readonly': actual.flags.writeable = False
    elif reason == 'shape': actual = actual[:1].copy()
    else: actual = actual.astype(np.dtype('float64', metadata={'changed': True}))
    with pytest.raises(AssertionError): approximate_polygon_default_error(image, actual, expected)
