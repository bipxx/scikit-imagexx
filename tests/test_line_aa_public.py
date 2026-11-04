"""Actual owned public line_aa against saved originals; never recaptures them."""
from contextlib import contextmanager
import gc
import importlib
import inspect
import json
from pathlib import Path
import sys
import pytest
if __package__:
    from .line_aa_contract import cases, ScalarProvider
    from .line_aa_observer import SIGNATURE, observe
else:
    from line_aa_contract import cases, ScalarProvider
    from line_aa_observer import SIGNATURE, observe

ROOT = Path(__file__).parents[1]
CASES = cases()


@pytest.fixture(scope='module')
def reference():
    # Decode the immutable reference once per process, never once per case.
    matrix = json.loads((ROOT / 'coverage/contracts/line-aa.json').read_bytes())
    assert len(matrix['cases']) == 48
    return matrix


@contextmanager
def strict_owned(monkeypatch):
    import skimagexx as xx
    from skimagexx import _line_aa as owned
    previous = xx.get_num_threads()
    fallback = xx.set_fallback_enabled(False)
    def denied(*args, **kwargs):
        raise AssertionError('line_aa forwarded to an original algorithm')
    public = importlib.import_module('skimage.draw')
    original = importlib.import_module('skimage.draw.draw')
    compiled = importlib.import_module('skimage.draw._draw')
    try:
        with monkeypatch.context() as patch:
            patch.setattr(public, 'line_aa', denied)
            patch.setattr(original, 'line_aa', denied)
            patch.setattr(compiled, '_line_aa', denied)
            yield xx, owned
    finally:
        xx.set_num_threads(previous)
        xx.set_fallback_enabled(fallback)
        assert xx.get_num_threads() == previous


def test_line_aa_public_signature_and_native_frontier(reference):
    import skimagexx as xx
    from skimagexx import _line_aa as owned
    assert xx.draw.line_aa is owned.line_aa
    assert str(inspect.signature(xx.draw.line_aa)) == SIGNATURE
    rows = reference['original_observations']['1.24.4']
    successful = {row['id'] for row in rows if row['error'] is None}
    assert len(successful) == 39 and set(reference['native_expected_ids']) <= successful
    assert {'line-aa/scalar-array', 'line-aa/provider-first', 'line-aa/providers-all'} <= successful


@pytest.mark.parametrize('spec', CASES, ids=[spec['id'] for spec in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_line_aa_saved_public_pipeline(spec, workers, reference, monkeypatch):
    import numpy as np
    expected = next(row for row in reference['original_observations'][np.__version__] if row['id'] == spec['id'])
    admitted = expected['error'] is None
    with strict_owned(monkeypatch) as (xx, owned):
        xx.set_num_threads(workers)
        call, entry = xx.draw.line_aa, owned._core.draw_line_aa
        assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
        returns, snapshots, admission = [], [], []
        getter = owned._core.get_num_threads
        def snapshot():
            result = getter(); snapshots.append(result); return result
        def profile(frame, event, value):
            if event == 'c_return' and value is entry: returns.append('draw_line_aa')
            if event == 'return' and frame.f_code is call.__code__:
                admission.append(frame.f_locals.get('native_result') is not None)
        previous = sys.getprofile()
        with monkeypatch.context() as patch:
            patch.setattr(owned._core, 'get_num_threads', snapshot)
            sys.setprofile(profile)
            try:
                actual = observe(call, spec, np)
            finally:
                sys.setprofile(previous)
        assert actual == expected
        assert xx.get_num_threads() == workers
        if admitted:
            assert returns == ['draw_line_aa'] and admission == [True]
            assert snapshots == [workers]
        else:
            assert not returns and not snapshots and not any(admission)


def test_line_aa_independent_owning_storage_survives_return(monkeypatch):
    import numpy as np
    with strict_owned(monkeypatch) as (xx, owned):
        rows, columns, weights = xx.draw.line_aa(5, 6, 8, 14)
        assert rows.dtype == columns.dtype == np.dtype(np.intp)
        assert weights.dtype == np.dtype('float64')
        assert all(a.base is None and a.flags.owndata and a.flags.c_contiguous for a in (rows, columns, weights))
        assert not any(np.shares_memory(a, b) for a, b in ((rows, columns), (rows, weights), (columns, weights)))
        old_column, last_row = int(columns[0]), int(rows[-1])
        rows[0] = -987; weights[0] = .25
        assert columns[0] == old_column
        del columns, weights
        gc.collect()
        assert rows[0] == -987 and rows[-1] == last_row


def test_line_aa_product_has_no_reference_capture_quota(monkeypatch):
    import numpy as np
    with strict_owned(monkeypatch) as (xx, owned):
        rows, columns, weights = xx.draw.line_aa(0, 0, 0, 10000)
        assert len(rows) == len(columns) == len(weights) == 10001
        assert np.array_equal(rows, np.zeros(10001, dtype=np.intp))
        assert np.array_equal(columns, np.arange(10001, dtype=np.intp))
        assert np.array_equal(weights, np.ones(10001, dtype=np.float64))


@pytest.mark.parametrize('args', ((2**40, 0, 2**40+1, 0),
                                  (0, 0, 32768, 32768), (0, 0, 46341, 0),
                                  ((1<<31)-1, 0, (1<<31)-1, 0),
                                  (-(1<<31), 0, -(1<<31), 0),
                                  (sys.maxsize, 0, sys.maxsize, 0)))
def test_line_aa_unsupported_upstream_unsafe_geometry_never_calls_original(args, monkeypatch):
    with strict_owned(monkeypatch) as (xx, owned):
        with pytest.raises(NotImplementedError, match='assessed safe C-int'):
            xx.draw.line_aa(*args)


@pytest.mark.parametrize('bad_index', (0, 1, 2, 3))
def test_line_aa_overflow_stops_before_later_index_callbacks(bad_index, reference, monkeypatch):
    events = []
    args = [ScalarProvider(1, events, name) for name in ('r0', 'c0', 'r1', 'c1')]
    args[bad_index].value = 2**100
    expected = next(row['error'][2] for row in reference['original_observations']['1.24.4'] if row['id'] == 'line-aa/overflow-first')
    with strict_owned(monkeypatch) as (xx, owned):
        with pytest.raises(OverflowError) as error:
            xx.draw.line_aa(*args)
        assert str(error.value) == expected
    assert events == [name+'.__index__' for name in ('r0', 'c0', 'r1', 'c1')[:bad_index+1]]


def test_line_aa_index_protocol_unsafe_result_is_quarantined(monkeypatch):
    events = []
    args = [ScalarProvider(value, events, name) for value, name in zip((2**40, 0, 2**40+1, 0), ('r0', 'c0', 'r1', 'c1'))]
    with strict_owned(monkeypatch) as (xx, owned):
        with pytest.raises(NotImplementedError, match='assessed safe C-int'):
            xx.draw.line_aa(*args)
    assert events == [name+'.__index__' for name in ('r0', 'c0', 'r1', 'c1')]
