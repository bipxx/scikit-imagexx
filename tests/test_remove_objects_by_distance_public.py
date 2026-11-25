"""Compare the public distance filter and its fallback with stock scikit-image."""

import importlib
import inspect
import sys
import warnings

import numpy as np
import pytest
from skimage import morphology as original

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning
from skimagexx._remove_objects_by_distance import remove_objects_by_distance as owned


API = 'remove_objects_by_distance'


def test_remove_objects_by_distance_public_owner():
    exports = {}
    exec('from skimagexx.morphology import *', exports)
    assert exports[API] is xx.morphology.remove_objects_by_distance is owned
    assert inspect.signature(owned) == inspect.signature(original.remove_objects_by_distance)
    entry = _core.remove_objects_by_distance_integer
    assert inspect.isbuiltin(entry)
    if entry.__module__ == '_tmp_verify':
        assert sys.modules['_tmp_verify'].remove_objects_by_distance_integer is entry
    else:
        assert entry.__module__ == 'skimagexx._core'


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('dtype', ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'))
def test_remove_objects_by_distance_greedy_ties(dtype, workers, monkeypatch):
    labels = np.array([3, 0, 2, 0, 1, 0, 2, 2, 0, 5, 5, 5, 0, 6, 6, 6], dtype=dtype)
    scores = np.array([0, 8, -3, 8, 0, 9, 9], dtype=np.float64)
    cases = (
        (2, {}),
        (2, dict(priority=scores)),
        (np.nextafter(2., 0.).item(), dict(priority=scores, p_norm=1)),
        (2.5, dict(spacing=(1.25,), p_norm=np.inf)),
    )
    expected = [original.remove_objects_by_distance(labels, radius, **kw) for radius, kw in cases]
    before = labels.tobytes(), scores.tobytes()
    native_entry = _core.remove_objects_by_distance_integer
    calls = []

    def observed(*args):
        calls.append(args[0])
        output = native_entry(*args)
        assert type(output) is np.ndarray
        return output

    def denied(*args, **kw):
        raise AssertionError('in-domain call executed the provider')

    monkeypatch.setattr(_core, 'remove_objects_by_distance_integer', observed)
    monkeypatch.setattr(original, API, denied)
    monkeypatch.setattr(importlib.import_module('skimage.morphology.misc'), API, denied)
    old_threads = xx.get_num_threads()
    old_fallback = xx.set_fallback_enabled(False)
    try:
        xx.set_num_threads(workers)
        for (radius, kw), reference in zip(cases, expected):
            actual = xx.morphology.remove_objects_by_distance(labels, radius, **kw)
            np.testing.assert_array_equal(actual, reference)
            assert actual.dtype == reference.dtype
            assert actual.flags.owndata and actual.flags.c_contiguous and actual.flags.writeable
            assert actual.base is None and not np.shares_memory(actual, labels)
        assert len(calls) == len(cases) and all(argument is labels for argument in calls)
        assert (labels.tobytes(), scores.tobytes()) == before
        assert xx.get_num_threads() == workers
    finally:
        xx.set_num_threads(old_threads)
        xx.set_fallback_enabled(old_fallback)


def test_remove_objects_by_distance_boundaries_and_cutoffs():
    rng = np.random.default_rng(71279)
    prior = xx.set_fallback_enabled(False)
    try:
        for case in range(180):
            rank = 1 + case % 3
            shape = tuple(rng.integers(1, 10, rank))
            labels = rng.integers(0, 12, size=shape, dtype=np.int16)
            labels[rng.random(shape) < 0.45] = 0
            spacing = tuple(rng.choice((0.25, 0.75, 1.0, 1.5, 2.0), rank).tolist())
            if case % 2:
                spacing = (spacing[0],) * rank
            radius = (0., 0.25, 0.75, 1., 1.5, 2., 9.)[(case // 3) % 7]
            norm = (1, 2, np.inf)[(case // 9) % 3]
            priority = None if case % 2 else rng.integers(-3, 4, size=12).astype(np.float64)
            options = dict(priority=priority, spacing=spacing, p_norm=norm)
            expected = original.remove_objects_by_distance(labels, radius, **options)
            np.testing.assert_array_equal(owned(labels, radius, **options), expected)

        labels = np.array([[3, 0], [0, 2]], dtype=np.uint16)
        for radius in (np.sqrt(2.), np.nextafter(np.sqrt(2.), 0.),
                       np.nextafter(np.sqrt(2.), np.inf)):
            radius = float(radius)
            np.testing.assert_array_equal(
                owned(labels, radius), original.remove_objects_by_distance(labels, radius))
        for labels in (np.zeros((3, 5), dtype=np.int32), np.full((2, 5, 3), 9, dtype=np.uint64)):
            np.testing.assert_array_equal(owned(labels, 5), labels)
    finally:
        xx.set_fallback_enabled(prior)


def test_remove_objects_by_distance_checks_owner_before_read():
    backing = np.array([1], dtype=np.int16)
    invalid = np.lib.stride_tricks.as_strided(backing, (4,), (backing.itemsize,), writeable=True)
    with pytest.raises(ValueError, match='requires ndarray-backed storage'):
        _core.remove_objects_by_distance_integer(invalid, 1., 2., (1.,), None)


class Subclass(np.ndarray):
    pass


@pytest.mark.parametrize('case', ('priority-type', 'priority-short', 'priority-nan', 'out-c',
                             'out-f', 'out-in-place', 'tagged', 'subclass', 'negative',
                             'p-norm', 'spacing', 'noncontiguous', 'empty', 'swapped',
                             'readonly', 'priority-readonly', 'priority-long-double'))
def test_remove_objects_by_distance_tagged_fallback(case):
    def make():
        labels = np.array([[3, 0, 1], [0, 5, 5]], dtype=np.int32)
        options = {}
        if case == 'priority-type':
            options['priority'] = np.arange(6, dtype=np.int32)
        elif case == 'priority-short':
            options['priority'] = np.zeros(4)
        elif case == 'priority-nan':
            options['priority'] = np.array([0., 1., 2., np.nan, 4., 5.])
        elif case == 'out-c':
            options['out'] = np.empty_like(labels, dtype=np.int64)
        elif case == 'out-f':
            options['out'] = np.empty_like(labels, order='F')
        elif case == 'out-in-place':
            options['out'] = labels
        elif case == 'tagged':
            labels = labels.view(np.dtype(np.int32, metadata={'label': 'tag'}))
        elif case == 'subclass':
            labels = labels.view(Subclass)
        elif case == 'negative':
            labels[0, 0] = -3
        elif case == 'p-norm':
            options['p_norm'] = 3
        elif case == 'spacing':
            options['spacing'] = (1., 1.3)
        elif case == 'noncontiguous':
            labels = labels[:, ::-1]
        elif case == 'empty':
            labels = labels[:, :0]
        elif case == 'swapped':
            labels = labels.astype(labels.dtype.newbyteorder('S'))
        elif case == 'readonly':
            labels.flags.writeable = False
        elif case == 'priority-readonly':
            options['priority'] = np.zeros(6)
            options['priority'].flags.writeable = False
        elif case == 'priority-long-double':
            options['priority'] = np.zeros(6, dtype=np.longdouble)
        return labels, options

    labels, options = make()
    before = labels.tobytes()
    prior = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            owned(labels, 2, **options)
        assert labels.tobytes() == before
        xx.set_fallback_enabled(True)
        original_input, original_options = make()
        with warnings.catch_warnings(record=True) as stock_warnings:
            warnings.simplefilter('always')
            try:
                expected = original.remove_objects_by_distance(original_input, 2, **original_options)
                error = None
            except (TypeError, ValueError) as exc:
                error = exc
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            if error is not None:
                with pytest.raises(type(error)) as actual_error:
                    owned(labels, 2, **options)
                assert str(actual_error.value) == str(error)
            else:
                actual = owned(labels, 2, **options)
                np.testing.assert_array_equal(actual, expected)
                assert type(actual) is type(expected)
                assert actual.dtype == expected.dtype and actual.dtype.metadata == expected.dtype.metadata
                if 'out' in options:
                    assert actual is options['out']
        assert sum(item.category is FallbackWarning for item in notices) == 1
        assert [(item.category, str(item.message)) for item in notices if item.category is not FallbackWarning] == [
            (item.category, str(item.message)) for item in stock_warnings]
        np.testing.assert_array_equal(labels, original_input)
    finally:
        xx.set_fallback_enabled(prior)
