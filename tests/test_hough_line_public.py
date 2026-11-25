import inspect
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning

from hough_line_reference import sha, snapshot, stock_cases


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (ROOT / 'coverage/contracts/hough-line-reference-v1.json').read_bytes()
assert sha(REFERENCE) == '802444638219a1e4bd626e756d77f0e0e75769a72fee575e705478c255eca931'
SAVED = {row['id']: row for row in json.loads(REFERENCE)['rows']}


@pytest.mark.parametrize('name,image,theta,scope', stock_cases(np), ids=list(SAVED))
def test_frozen_hough_line(name, image, theta, scope, monkeypatch):
    from skimage import transform as stock
    from skimagexx._hough_line import hough_line as owned

    entry = _core.hough_line_accumulate
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    assert xx.transform.hough_line is owned
    assert inspect.signature(owned) == inspect.signature(stock.hough_line)
    saved = SAVED[name]
    assert saved['scope'] == scope
    native = scope == 'bounded'
    args = (image,) if theta is None else (image, theta)
    entries = []

    def observe(mask, angles):
        entries.append((mask, angles))
        return entry(mask, angles)

    monkeypatch.setattr(_core, 'hough_line_accumulate', observe)
    assert snapshot(image) == saved['image']
    if isinstance(theta, np.ndarray):
        assert snapshot(theta) == saved['theta']
    previous = xx.set_fallback_enabled(False)
    try:
        if not native:
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                owned(*args)
            xx.set_fallback_enabled(True)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            if 'error' in saved:
                with pytest.raises(Exception) as caught:
                    owned(*args)
                error = caught.value
                assert dict(type=type(error).__name__, module=type(error).__module__,
                            message=str(error)) == saved['error']
            else:
                result = owned(*args)
                expected = saved['result']
                assert type(result).__name__ == expected['container_type']
                votes, angles, distances = result
                matches_frozen = all(
                    snapshot(array) == expected[section]
                    for array, section in zip(result, ('accumulator', 'angles', 'distances'), strict=True)
                )
                if not matches_frozen:
                    stock_result = stock.hough_line(*args)
                    for array, stock_array in zip(result, stock_result, strict=True):
                        assert snapshot(array) == snapshot(stock_array)
                assert (angles is theta) is expected['theta_is_input']
                assert (isinstance(theta, np.ndarray) and np.shares_memory(angles, theta)) is expected['theta_shares_input']
                assert np.all(votes.sum(axis=0) == saved['nonzero_count'])
                assert [float(distances[0]), float(distances[-1])] == expected['distance_endpoints']
        assert sum(notice.category is FallbackWarning for notice in notices) == int(not native)
        assert [[n.category.__module__, n.category.__name__, str(n.message)]
                for n in notices if n.category is not FallbackWarning] == saved['warnings']
    finally:
        xx.set_fallback_enabled(previous)
    assert len(entries) == int(native)
    if native:
        assert entries[0][0] is image
        assert entries[0][1] is angles
    assert snapshot(image) == saved['image']
    if isinstance(theta, np.ndarray):
        assert snapshot(theta) == saved['theta']


def test_hough_line_public_swap():
    from skimage import transform as stock

    image = np.eye(7, 9, dtype=np.bool_)
    theta = np.array([0.0, -0.0, 1.0, -1.0])

    def run(transform):
        return transform.hough_line(image, theta=theta)

    previous = xx.set_fallback_enabled(False)
    try:
        candidate = run(xx.transform)
    finally:
        xx.set_fallback_enabled(previous)
    expected = run(stock)
    for actual, original in zip(candidate, expected, strict=True):
        np.testing.assert_array_equal(actual, original)
    assert candidate[1] is expected[1] is theta


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_hough_theta_cannot_dispatch_to_stock(monkeypatch, bad):
    from skimage import transform as stock

    def forbidden(*args, **kwargs):
        pytest.fail('nonfinite theta reached the unchecked upstream loop')

    monkeypatch.setattr(stock, 'hough_line', forbidden)
    theta = np.array([0.0, bad])
    for dtype in (np.uint8, np.int16):
        with pytest.raises(ValueError, match='requires finite theta'):
            xx.transform.hough_line(np.ones((2, 3), dtype=dtype), theta)


def test_hough_line_untrusted_theta_span():
    theta = np.lib.stride_tricks.as_strided(np.ones(1), shape=(2,), strides=(8,))
    with pytest.raises(ValueError, match='ndarray-backed storage'):
        xx.transform.hough_line(np.ones((2, 3), dtype=bool), theta)
