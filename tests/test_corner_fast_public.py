"""Public parity and policy tests for feature.corner_fast."""

import hashlib
import inspect
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _corner_fast
from skimagexx._compat import FallbackWarning
from corner_fast_cases import cases


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/corner_fast_originals_v1.json'
FROZEN = json.loads(FIXTURE.read_bytes())
ALL_CASES = [(name, options, native) for name, _, options, native in cases(np)]


@pytest.fixture(params=(1, 4))
def candidate(request):
    previous_threads = xx.get_num_threads()
    previous_fallback = xx.set_fallback_enabled(False)
    try:
        xx.set_num_threads(request.param)
        assert xx.get_num_threads() == request.param
        target = xx.feature.corner_fast
        assert target is _corner_fast.corner_fast
        assert inspect.isbuiltin(xx._core.corner_fast_apply)
        assert str(inspect.signature(target)) == '(image, n=12, threshold=0.15)'
        yield target
    finally:
        xx.set_num_threads(previous_threads)
        xx.set_fallback_enabled(previous_fallback)


@pytest.mark.parametrize('case_name', [name for name, _, _ in ALL_CASES])
def test_frozen_corner_fast_parity(candidate, case_name):
    matched = next((name, image, options, native) for name, image, options, native in cases(np)
                   if name == case_name)
    _, image, options, native = matched
    expected = FROZEN[case_name]
    assert native is expected['native']
    if not native:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            candidate(image, **options)
        xx.set_fallback_enabled(True)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            if 'error' in expected:
                err_name, err_msg = expected['error']
                with pytest.raises(Exception) as caught:
                    candidate(image, **options)
                assert type(caught.value).__name__ == err_name
                assert str(caught.value) == err_msg
                return
            actual = candidate(image, **options)
        assert any(w.category is FallbackWarning and 'skimage.feature.corner_fast' in str(w.message)
                   for w in notices)
    else:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            actual = candidate(image, **options)
        assert not notices

    out_meta = expected['output']
    assert type(actual) is np.ndarray
    assert list(actual.shape) == out_meta['shape']
    assert actual.dtype.str == out_meta['dtype']
    assert list(actual.strides) == out_meta['strides']
    assert actual.flags.c_contiguous
    assert actual.flags.writeable and not actual.flags.owndata
    assert not np.shares_memory(actual, image)
    assert type(actual.base) is memoryview
    assert type(actual.base.obj) is (
        np.ndarray if native else type(actual.base.obj)
    )
    if native:
        assert actual.base.obj.flags.owndata and actual.base.obj.base is None
    assert int((actual != 0).sum()) == out_meta['nonzero']
    assert hashlib.sha256(actual.tobytes()).hexdigest() == out_meta['digest']


def test_corner_fast_repeat_independence(candidate):
    _, image, options, _ = next(cases(np))
    first = candidate(image, **options)
    second = candidate(image, **options)
    assert first.base.obj is not second.base.obj
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, second)


def test_corner_fast_metadata_and_scalar_fallback(candidate, monkeypatch):
    from skimage import feature

    base = np.zeros((10, 10), dtype=np.uint8)
    for deferred, args in (
        (base.view(np.dtype('uint8', metadata={'x': 1})), (12, 0.15)),
        (base.astype(np.float32).view(np.dtype('float32', metadata={'x': 1})), (12, 0.15)),
        (base.astype(np.float64).view(np.dtype('float64', metadata={'x': 1})), (12, 0.15)),
        (base, (True, 0.15)),
        (base, (np.int64(12), 0.15)),
        (base, (12, 0)),
        (np.full((10, 10), np.nan, dtype=np.float64), (12, 0.15)),
        (np.full((10, 10), 300.0, dtype=np.float32), (12, 0.15)),
    ):
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            candidate(deferred, *args)

    seen = []
    marker = object()

    def provider(*args, **kwargs):
        seen.append((args, kwargs))
        return marker

    monkeypatch.setattr(feature, 'corner_fast', provider)
    xx.set_fallback_enabled(True)
    tagged = base.view(np.dtype('uint8', metadata={'x': 1}))
    n_val = np.int64(9)
    with pytest.warns(FallbackWarning, match='skimage.feature.corner_fast'):
        assert candidate(tagged, n_val, 0.25) is marker
    (args, kwargs), = seen
    assert not kwargs
    assert len(args) == 3 and args[0] is tagged and args[1] is n_val and args[2] == 0.25
