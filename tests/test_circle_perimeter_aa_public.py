"""Circle AA component against immutable original rows; no original recapture."""
from contextlib import contextmanager
import gc
import importlib
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .circle_perimeter_aa_contract import cases
    from .circle_perimeter_aa_observer import SIGNATURE, observe
    from .draw_coordinates_contract import ScalarProvider
    from .draw_coordinates_observer import encode
else:
    from circle_perimeter_aa_contract import cases
    from circle_perimeter_aa_observer import SIGNATURE, observe
    from draw_coordinates_contract import ScalarProvider
    from draw_coordinates_observer import encode

ROOT = Path(__file__).parents[1]
CASES = cases()


@pytest.fixture(scope='module')
def reference():
    return json.loads((ROOT / 'coverage/contracts/circle-perimeter-aa.json').read_bytes())


@contextmanager
def strict_owned(monkeypatch):
    import skimagexx as xx
    from skimagexx import _circle_perimeter_aa as owned
    previous = xx.get_num_threads()
    fallback = xx.set_fallback_enabled(False)

    def denied(*args, **kwargs):
        raise AssertionError('circle AA forwarded to an original algorithm')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(importlib.import_module('skimage.draw'), 'circle_perimeter_aa', denied)
            patch.setattr(importlib.import_module('skimage.draw.draw'), 'circle_perimeter_aa', denied)
            patch.setattr(importlib.import_module('skimage.draw._draw'), '_circle_perimeter_aa', denied)
            yield xx, owned
    finally:
        xx.set_num_threads(previous)
        xx.set_fallback_enabled(fallback)


def test_circle_aa_signature_and_scoped_frontier(reference):
    from skimagexx import _circle_perimeter_aa as owned
    assert str(inspect.signature(owned.circle_perimeter_aa)) == SIGNATURE
    assert len(CASES) == 48
    assert sum(r['error'] is None for r in reference['original_observations']['1.24.4']) == 41
    policy = json.loads((ROOT / 'coverage/contracts/circle-perimeter-aa-frontend-v1.json').read_bytes())
    assert policy['native_integer_clip_IDs'] == 40 and policy['native_recurrence_IDs'] == 41
    assert policy['provider_tail_IDs'] == ['circle-aa/float-shape']


@pytest.mark.parametrize('spec', CASES, ids=[s['id'] for s in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_circle_perimeter_aa_saved_public_pipeline(spec, workers, reference, monkeypatch):
    import numpy as np
    expected = next(r for r in reference['original_observations'][np.__version__] if r['id'] == spec['id'])
    with strict_owned(monkeypatch) as (xx, owned):
        xx.set_num_threads(workers)
        call, entry = owned.circle_perimeter_aa, owned._core.draw_circle_perimeter_aa
        assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
        returns, snapshots, admitted = [], [], []
        getter = owned._core.get_num_threads

        def snapshot():
            result = getter()
            snapshots.append(result)
            return result

        def profile(frame, event, value):
            if event == 'c_return' and value is entry:
                returns.append('draw_circle_perimeter_aa')
            if event == 'return' and frame.f_code is call.__code__:
                admitted.append(frame.f_locals.get('native_result') is not None)

        previous = sys.getprofile()
        with monkeypatch.context() as patch:
            patch.setattr(owned._core, 'get_num_threads', snapshot)
            sys.setprofile(profile)
            try:
                actual = observe(call, spec, np)
            finally:
                sys.setprofile(previous)
        assert actual == expected
        if expected['error'] is None:
            assert returns == ['draw_circle_perimeter_aa'] and snapshots == [workers] and admitted == [True]
        else:
            assert not returns and not snapshots and not any(admitted)


@pytest.mark.parametrize('layout', ('positive-gap', 'negative-gap', 'unaligned'))
def test_circle_aa_shape_storage_uses_saved_integer_clip(layout, reference, monkeypatch):
    import numpy as np
    if layout == 'positive-gap':
        shape = np.array([8, 99, 9], dtype=np.intp)[::2]
    elif layout == 'negative-gap':
        shape = np.array([9, 99, 8], dtype=np.intp)[::-2]
    else:
        shape = np.ndarray((2,), dtype=np.intp, buffer=bytearray(17), offset=1)
        shape[:] = [8, 9]
    before = (shape.tobytes(), shape.strides, shape.ctypes.data)
    expected = next(r['result'] for r in reference['original_observations'][np.__version__]
                    if r['id'] == 'circle-aa/array-shape')
    with strict_owned(monkeypatch) as (xx, owned):
        actual = owned.circle_perimeter_aa(4, 5, 5, shape=shape)
        assert encode(actual, np) == expected
    assert before == (shape.tobytes(), shape.strides, shape.ctypes.data)


def test_circle_aa_output_lifetime_and_no_reference_capture_quota(monkeypatch):
    import numpy as np
    with strict_owned(monkeypatch) as (xx, owned):
        rows, columns, weights = owned.circle_perimeter_aa(0, 0, 512)
        assert len(rows) > 4096 and len(rows) == len(columns) == len(weights)
        assert rows.dtype == columns.dtype == np.dtype(np.intp) and weights.dtype == np.dtype('float64')
        assert all(a.base is None and a.flags.owndata and a.flags.c_contiguous for a in (rows, columns, weights))
        assert not any(np.shares_memory(a, b) for a, b in ((rows, columns), (rows, weights), (columns, weights)))
        last, first_column = int(rows[-1]), int(columns[0])
        rows[0] = -99
        weights[0] = .25
        assert columns[0] == first_column
        del columns, weights
        gc.collect()
        assert rows[0] == -99 and rows[-1] == last


@pytest.mark.parametrize('shape', ((), (5,)))
def test_circle_aa_source_proven_short_tuple_quarantine_preserves_conversion(shape, monkeypatch):
    events = []
    args = [ScalarProvider(v, events, n) for v, n in zip((1, 2, 3), ('r', 'c', 'radius'))]
    with strict_owned(monkeypatch) as (xx, owned):
        with pytest.raises(NotImplementedError, match='builtin tuple shapes shorter than 2'):
            owned.circle_perimeter_aa(*args, shape=shape)
    assert events == ['r.__index__', 'c.__index__', 'radius.__index__']


@pytest.mark.parametrize('args', ((0, 0, -(1 << 63)), (0, 0, 1 << 32),
                                  ((1 << 63) - 1, 0, 2), (-(1 << 63), 0, -3)))
def test_circle_aa_source_unsafe_geometry_is_explicit_gap(args, monkeypatch):
    with strict_owned(monkeypatch) as (xx, owned):
        with pytest.raises(NotImplementedError, match='safe integer domain'):
            owned.circle_perimeter_aa(*args)


@pytest.mark.parametrize('bad', (0, 1, 2))
def test_circle_aa_index_overflow_stops_later_callbacks(bad, monkeypatch):
    events = []
    args = [ScalarProvider(1, events, name) for name in ('r', 'c', 'radius')]
    args[bad].value = 2**100
    with strict_owned(monkeypatch) as (xx, owned):
        with pytest.raises(OverflowError, match='Python int too large to convert to C ssize_t'):
            owned.circle_perimeter_aa(*args, shape=())
    assert events == [n + '.__index__' for n in ('r', 'c', 'radius')[:bad+1]]


def test_circle_aa_unknown_shape_protocol_is_unassessed_without_callbacks(monkeypatch):
    class Unknown:
        def __getitem__(self, index):
            raise AssertionError('unassessed shape protocol executed')
    with strict_owned(monkeypatch) as (xx, owned):
        with pytest.raises(NotImplementedError, match='shape carrier/protocol'):
            owned.circle_perimeter_aa(1, 2, 3, shape=Unknown())
