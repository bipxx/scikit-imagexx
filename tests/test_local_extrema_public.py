import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from skimage import morphology as upstream

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning


REFERENCE = Path(__file__).resolve().parents[1] / 'coverage/contracts/local-extrema-reference-v1.json'
REFERENCE_HASH = '245c45ebf046487b884f88f2024b207b3f2bd723a813bde847c013bc0c3e0400'
raw = REFERENCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == REFERENCE_HASH
CASES = json.loads(raw)['rows']
FUNCTIONS = ('local_maxima', 'local_minima')


def input_array(row):
    image = np.array(row['values'], dtype=row['dtype'], order='F' if row['layout'] == 'F' else 'C')
    return image[:, ::-1] if row['layout'] == 'reverse' else image


@pytest.mark.parametrize('name', FUNCTIONS)
@pytest.mark.parametrize('indices', [False, True])
@pytest.mark.parametrize('row', CASES, ids=[r['id'] for r in CASES])
def test_local_extrema_frozen_public(row, indices, name, monkeypatch):
    function = getattr(xx.morphology, name)
    assert inspect.signature(function) == inspect.signature(getattr(upstream, name))
    entry = _core.local_extrema_integer_mask
    assert inspect.isbuiltin(entry)
    assert entry.__module__ == _core.__name__
    entered = []

    def count_entry(*args):
        entered.append(args[1:])
        return entry(*args)

    monkeypatch.setattr(_core, 'local_extrema_integer_mask', count_entry)
    image = input_array(row)
    before = image.tobytes(order='A')
    footprint = None if row['footprint'] is None else np.array(row['footprint'], dtype=bool)
    native = footprint is None or bool(footprint[1, 1])
    prior = xx.set_fallback_enabled(False)
    try:
        options = dict(footprint=footprint, connectivity=row['connectivity'],
                       indices=indices, allow_borders=row['allow_borders'])
        if native:
            result = function(image, **options)
        else:
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                function(image, **options)
            xx.set_fallback_enabled(True)
            with pytest.warns(FallbackWarning):
                result = function(image, **options)
    finally:
        xx.set_fallback_enabled(prior)

    if native:
        assert len(entered) == 1 and entered[0][0] is (name == 'local_minima')
    else:
        assert entered == []
    assert image.tobytes(order='A') == before
    expected = row['results'][name]
    if indices:
        assert type(result) is tuple and len(result) == 2
        for coordinate, saved in zip(result, expected['indices'], strict=True):
            assert coordinate.dtype == np.dtype(np.intp)
            np.testing.assert_array_equal(coordinate, saved)
    else:
        assert result.dtype == np.dtype(bool)
        assert result.strides == tuple(expected['strides'])
        assert bool(result.flags.owndata) is expected['owns']
        assert result.base.shape == tuple(expected['base_shape'])
        assert not np.shares_memory(result, image)
        np.testing.assert_array_equal(result, expected['mask'])


@pytest.mark.parametrize('name', FUNCTIONS)
def test_local_extrema_cross_with_center_ignores_connectivity(name):
    row = next(row for row in CASES if row['id'] == 'cross-footprint')
    image = input_array(row)
    footprint = np.array(row['footprint'], dtype=bool)
    footprint[1, 1] = True
    expected = getattr(upstream, name)(image, footprint=footprint, connectivity=0)
    function = getattr(xx.morphology, name)
    prior = xx.set_fallback_enabled(False)
    try:
        actual = function(image, footprint=footprint, connectivity=0)
    finally:
        xx.set_fallback_enabled(prior)
    np.testing.assert_array_equal(actual, expected)


def test_local_extrema_exotic_owner_excluded_from_native(monkeypatch):
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    view = np.lib.stride_tricks.as_strided(image, shape=image.shape, strides=image.strides)

    def fail_entry(*args):
        pytest.fail('unproven buffer reached the native entry')

    monkeypatch.setattr(_core, 'local_extrema_integer_mask', fail_entry)
    prior = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            xx.morphology.local_maxima(view)
    finally:
        xx.set_fallback_enabled(prior)


@pytest.mark.parametrize('name', FUNCTIONS)
def test_local_extrema_float_fallback_preserves_policy(name):
    function = getattr(xx.morphology, name)
    image = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    prior = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            function(image)
        xx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning):
            result = function(image)
        np.testing.assert_array_equal(result, getattr(upstream, name)(image))
    finally:
        xx.set_fallback_enabled(prior)
