"""Saved line-ND rows through an owned component; no original algorithm calls."""
from contextlib import contextmanager
import gc
import importlib
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .line_nd_contract import cases
    from .line_nd_observer import observe, SIGNATURE
    from .draw_coordinates_observer import encode
else:
    from line_nd_contract import cases
    from line_nd_observer import observe, SIGNATURE
    from draw_coordinates_observer import encode

ROOT = Path(__file__).parents[1]
CASES = cases()


@pytest.fixture(scope='module')
def reference():
    return json.loads((ROOT/'coverage/contracts/line-nd.json').read_bytes())


@contextmanager
def strict_owned(monkeypatch, workers):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _line_nd as owned
    old_workers, old_fallback = xx.get_num_threads(), xx.set_fallback_enabled(False)
    def denied(*args, **options):
        raise AssertionError('owned line-ND invoked an original draw/linspace algorithm')
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(importlib.import_module('skimage.draw'), 'line_nd', denied)
            original = importlib.import_module('skimage.draw.draw_nd')
            patch.setattr(original, 'line_nd', denied)
            patch.setattr(original, '_round_safe', denied)
            patch.setattr(np, 'linspace', denied)
            yield xx, owned
        assert xx.get_num_threads() == workers
    finally:
        xx.set_num_threads(old_workers)
        xx.set_fallback_enabled(old_fallback)
        assert xx.get_num_threads() == old_workers


def test_line_nd_signature_and_frozen_scope(reference):
    from skimagexx import _line_nd as owned
    assert str(inspect.signature(owned.line_nd)) == SIGNATURE
    assert len(CASES) == 54
    assert sum(r['error'] is None for r in reference['original_observations']['1.24.4']) == 45
    contract = json.loads((ROOT/'coverage/contracts/line-nd-frontend-v1.json').read_bytes())
    assert contract['saved_returns_per_profile'] == 45 and contract['saved_errors_per_profile'] == 9


@pytest.mark.parametrize('spec', CASES, ids=[s['id'] for s in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_line_nd_saved_public_pipeline(spec, workers, reference, monkeypatch):
    import numpy as np
    expected = next(r for r in reference['original_observations'][np.__version__] if r['id'] == spec['id'])
    with strict_owned(monkeypatch, workers) as (xx, owned):
        entry, getter = owned._core.draw_line_nd, owned._core.get_num_threads
        assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
        returns, snapshots, accepted = [], [], []
        def snapshot():
            value = getter()
            snapshots.append(value)
            return value
        def profile(frame, event, value):
            if event == 'c_return' and value is entry:
                returns.append('draw_line_nd')
            if event == 'return' and frame.f_code is owned.line_nd.__code__:
                accepted.append(frame.f_locals.get('native_result') is not None)
        previous = sys.getprofile()
        with monkeypatch.context() as patch:
            patch.setattr(owned._core, 'get_num_threads', snapshot)
            sys.setprofile(profile)
            try:
                actual = observe(owned.line_nd, spec, np)
            finally:
                sys.setprofile(previous)
        assert actual == expected
        if spec['kind'] in ('error-missing-stop', 'error-keyword'):
            assert not returns and not snapshots and not accepted
        else:
            assert returns == ['draw_line_nd'] and snapshots == [workers]
            assert accepted == [expected['error'] is None or spec['kind'] == 'scalar']


@pytest.mark.parametrize('integer', (False, True))
def test_line_nd_shared_base_lifetime_and_input_independence(integer, monkeypatch):
    import numpy as np
    start, stop = np.array([0., 1.]), np.array([4., 3.])
    before = start.tobytes(), stop.tobytes()
    with strict_owned(monkeypatch, 1) as (xx, owned):
        first, second = owned.line_nd(start, stop, integer=integer)
    owner = first.base
    assert owner is second.base and owner.flags.owndata and owner.base is None
    assert first is not second and not first.flags.owndata and not second.flags.owndata
    assert not np.shares_memory(first, second)
    assert not any(np.shares_memory(a, b) for a in (first, second) for b in (start, stop))
    assert (start.tobytes(), stop.tobytes()) == before
    start[:] = 99; stop[:] = 99
    assert first[0] == 0 and second[0] == 1
    first[0] = 7
    assert second[0] == 1
    del second, owner, start, stop
    gc.collect()
    assert first[0] == 7 and first[-1] == 3


def test_line_nd_underflow_callbacks_use_prepared_snapshots(reference, monkeypatch):
    import numpy as np
    start, stop = np.array([0., 0.]), np.array([4.2, 5e-324])
    calls = []
    def handler(message, flags):
        calls.append((message, flags))
        start[:] = 100
        stop[:] = 200
    expected = next(r['result'] for r in reference['original_observations'][np.__version__]
                    if r['id'] == 'line-nd/underflow-step')
    with strict_owned(monkeypatch, 4) as (xx, owned):
        with np.errstate(under='call', call=handler):
            actual = owned.line_nd(start, stop, integer=False)
        assert encode(actual, np) == expected
    assert calls == [('underflow', 4), ('underflow', 4)]
    assert np.all(start == 100) and np.all(stop == 200)


@pytest.mark.parametrize('workers', (1, 4))
def test_line_nd_large_allocation_has_no_capture_quota(workers, monkeypatch):
    import numpy as np
    points = 131073  # Two independent dimensions cross the existing262144 threshold.
    with strict_owned(monkeypatch, workers) as (xx, owned):
        first, second = owned.line_nd([0, 0], [points, points])
    expected = np.arange(points, dtype=np.intp)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert first.base is second.base and first.base.shape == (2, points)
    assert first.strides == second.strides == (2*np.dtype(np.intp).itemsize,)


@pytest.mark.parametrize('dtype', ('float16', 'float32', 'int16'))
def test_line_nd_uncaptured_scalar_dtype_frontier_is_explicit(dtype, monkeypatch):
    import numpy as np
    with strict_owned(monkeypatch, 1) as (xx, owned):
        with pytest.raises(NotImplementedError, match='outside the assessed native partition'):
            owned.line_nd(np.array(0, dtype=dtype), np.array(3, dtype=dtype), integer=False)


def test_line_nd_private_entry_refuses_borrowed_storage(monkeypatch):
    import numpy as np
    with strict_owned(monkeypatch, 1) as (xx, owned):
        source = np.array([0., 1., 2., 3.])
        assert owned._core.draw_line_nd(source[::2], source[1::2], False, False, 1, owned._SCALAR_MODULO_NATIVE) is None


@pytest.mark.parametrize('workers', (1, 4))
def test_line_nd_saved_scalar_modulo_promotion_holdout(workers, monkeypatch):
    import numpy as np
    supplement = json.loads((ROOT/'coverage/contracts/line-nd-rounding-promotion-v1.json').read_bytes())
    spec = supplement['cases'][0]
    expected = supplement['original_observations'][np.__version__][0]
    with strict_owned(monkeypatch, workers) as (xx, owned):
        entry, getter = owned._core.draw_line_nd, owned._core.get_num_threads
        returns, snapshots, accepted = [], [], []
        def snapshot():
            value = getter()
            snapshots.append(value)
            return value
        def profile(frame, event, value):
            if event == 'c_return' and value is entry:
                returns.append('draw_line_nd')
            if event == 'return' and frame.f_code is owned.line_nd.__code__:
                accepted.append(frame.f_locals.get('native_result') is not None)
        previous = sys.getprofile()
        with monkeypatch.context() as patch:
            patch.setattr(owned._core, 'get_num_threads', snapshot)
            sys.setprofile(profile)
            try:
                actual = observe(owned.line_nd, spec, np)
            finally:
                sys.setprofile(previous)
        assert actual == expected
        assert inspect.isbuiltin(entry) and returns == ['draw_line_nd']
        assert snapshots == [workers] and accepted == [True]


def test_line_nd_scalar_modulo_policy_is_actual_runtime_capability():
    import numpy as np
    from skimagexx import _line_nd as owned
    assert owned._SCALAR_MODULO_NATIVE is ((np.float32(0) % 1).dtype == np.dtype(np.float32))
