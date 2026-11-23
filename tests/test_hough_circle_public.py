import inspect
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning

from hough_circle_reference import circle_cases, sha, snapshot


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (ROOT / 'coverage/contracts/hough-circle-reference-v1.json').read_bytes()
assert sha(REFERENCE) == '7c8f78e1ac86d316ca704f10d0da7722e026f5105d5e438ef37f589bc534a9a3'
SAVED = {row['id']: row for row in json.loads(REFERENCE)['cases']}


@pytest.mark.parametrize('name,args,options,scope', list(circle_cases(np)), ids=list(SAVED))
def test_frozen_hough_circle(name, args, options, scope, monkeypatch):
    from skimage import transform as original
    from skimagexx._hough_circle import hough_circle as owned

    assert xx.transform.hough_circle is owned
    assert inspect.signature(owned) == inspect.signature(original.hough_circle)
    builtin = _core.hough_circle_accumulate
    assert inspect.isbuiltin(builtin) and builtin.__module__ == 'skimagexx._core'
    native = scope in ('bounded', 'benchmark')
    expected = SAVED[name]
    def fingerprint():
        return [snapshot(a, False) if type(a) is np.ndarray else list(a) if type(a) is tuple else a for a in args]

    assert fingerprint() == expected['inputs'] and options == expected['options']
    calls = []

    def witness(image, radii, normalize, full):
        calls.append(image is args[0])
        return builtin(image, radii, normalize, full)

    monkeypatch.setattr(_core, 'hough_circle_accumulate', witness)
    previous = xx.set_fallback_enabled(False)
    try:
        if not native:
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                owned(*args, **options)
            xx.set_fallback_enabled(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if 'error' in expected:
                with pytest.raises(Exception) as err:
                    owned(*args, **options)
                assert [type(err.value).__module__, type(err.value).__name__, str(err.value)] == expected['error']
            else:
                result = owned(*args, **options)
                assert type(result).__name__ == expected['container']
                assert not any(np.shares_memory(a, result) for a in args if type(a) is np.ndarray)
                actual = snapshot(result, result.size <= 1024)
                assert actual == expected['result'][0]
                if scope == 'benchmark':
                    assert snapshot(original.hough_circle(*args, **options), False) == actual
        assert sum(n.category is FallbackWarning for n in caught) == int(not native)
        assert [[type(n.message).__module__, n.category.__name__, str(n.message)]
                for n in caught if n.category is not FallbackWarning] == expected['warnings']
    finally:
        xx.set_fallback_enabled(previous)
    assert calls == ([True] if native else [])
    assert fingerprint() == expected['inputs']


def test_hough_circle_bounded_storage_and_volume_controls():
    exotic = np.lib.stride_tricks.as_strided(np.zeros(1, dtype=np.uint8), (3, 4), (4, 1))
    with pytest.raises(ValueError, match='ndarray-backed storage'):
        _core.hough_circle_accumulate(exotic, [1], True, False)
    image = np.zeros((512, 512), dtype=np.uint8)
    assert _core.hough_circle_accumulate(image, [64] * 16, True, True) is None
