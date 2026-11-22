import inspect
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning

from hough_line_peaks_reference import cases, sha, snapshot


ROOT = Path(__file__).resolve().parents[1]
FROZEN_BYTES = (ROOT / 'coverage/contracts/hough-line-peaks-reference-v1.json').read_bytes()
assert sha(FROZEN_BYTES) == 'e0e9e5753977b710bce389db3d3297986fd84f0378a03a72be2a9067fb2ed00d'
FROZEN = json.loads(FROZEN_BYTES)
SAVED = {row['id']: row for row in FROZEN['common_cases']}


@pytest.mark.parametrize('name,args,options,scope,retained',
    list(cases(np, {'rows': FROZEN['parent_hough_rows']})), ids=list(SAVED))
def test_hough_peaks_frozen_and_installed(name, args, options, scope, retained, monkeypatch):
    from skimage import transform as original
    from skimagexx._hough_line_peaks import hough_line_peaks as owned

    assert xx.transform.hough_line_peaks is owned
    assert inspect.signature(owned) == inspect.signature(original.hough_line_peaks)
    entry = _core.hough_line_peak_indices
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    before = [snapshot(array, False) for array in args]
    expected = SAVED[name]
    tie = name in FROZEN['linux_minimum_capped_ties']
    if tie and (sys.platform, np.__version__) == ('linux', '1.24.4'):
        expected = FROZEN['linux_minimum_capped_ties'][name]
    assert before == expected['inputs'] and retained == expected['parent_hough_row']
    native = scope not in ('compatibility', 'error')
    calls = []

    def trace(*entry_args):
        calls.append(tuple(a is b for a, b in zip(args, entry_args[:3], strict=True)))
        return entry(*entry_args)

    monkeypatch.setattr(_core, 'hough_line_peak_indices', trace)
    previous = xx.set_fallback_enabled(False)
    try:
        if not native:
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                xx.transform.hough_line_peaks(*args, **options)
            xx.set_fallback_enabled(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if 'error' in expected:
                with pytest.raises(Exception) as err:
                    xx.transform.hough_line_peaks(*args, **options)
                assert [type(err.value).__module__, type(err.value).__name__, str(err.value)] == expected['error']
            else:
                result = xx.transform.hough_line_peaks(*args, **options)
                assert type(result).__name__ == expected['container']
                assert not any(np.shares_memory(out, source) for out in result for source in args)
                actual = [snapshot(a, False) for a in result]
                known_tie = (sys.platform, np.__version__) in (('win32', '2.5.4'), ('linux', '1.24.4'))
                if not tie or known_tie:
                    assert actual == expected['result']
                if scope == 'benchmark' or tie:
                    assert actual == [snapshot(a, False) for a in original.hough_line_peaks(*args, **options)]
        assert sum(item.category is FallbackWarning for item in caught) == int(not native)
        assert [[type(w.message).__module__, w.category.__name__, str(w.message)]
                for w in caught if w.category is not FallbackWarning] == expected['warnings']
    finally:
        xx.set_fallback_enabled(previous)
    assert calls == ([(True, True, True)] if native else [])
    assert [snapshot(array, False) for array in args] == before


def test_hough_peaks_checked_owner_and_vote_limit():
    votes = np.zeros((5, 5), dtype=np.uint64)
    angles = np.arange(5, dtype=np.float64)
    distances = np.arange(5, dtype=np.float64)
    exotic = np.lib.stride_tricks.as_strided(np.zeros(2, dtype=np.uint64), (5, 5), (40, 8))
    with pytest.raises(ValueError, match='ndarray-backed storage'):
        _core.hough_line_peak_indices(exotic, angles, distances, 1, 1, None)
    votes[2, 2] = 1048577
    state = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            xx.transform.hough_line_peaks(votes, angles, distances)
    finally:
        xx.set_fallback_enabled(state)
