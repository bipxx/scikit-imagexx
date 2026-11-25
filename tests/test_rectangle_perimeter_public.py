import importlib
import inspect

import numpy as np
import pytest
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(('start', 'kwargs', 'saved'), [
    ((1, 1), {'end': (2, 2)},
     ([0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 0], [0, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0])),
    ((5, 3), {'end': (-1, 6)},
     ([-2] * 7 + list(range(-1, 6)) + [6] * 8 + list(range(5, -3, -1)),
      [2, 3, 4, 5, 6] + [7] * 11 + [6, 5, 4, 3] + [2] * 10)),
    ([0, 0], {'extent': (0, 0)}, ([-1, -1, -1, 0, 0, 0, 0, -1], [-1, 0, 0, 0, 0, -1, -1, -1])),
    ((1, 1), {'extent': (10, 0), 'shape': (5, 5)},
     ([0, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0], [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])),
    ((2, 3), {'extent': (-2, -2), 'shape': (3, 4), 'clip': True},
     ([0, 1, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0], [3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 1, 2, 3])),
    ((2, 3), {'end': (10, 10), 'shape': (5, 5), 'clip': True},
     ([1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 3, 2, 1], [2, 3, 4, 4, 4, 4, 4, 4, 3, 2, 2, 2, 2, 2])),
    ((2, 3), {'end': (10, 10), 'shape': (1, 1)}, ([], [])),
    (np.array([3, 4], dtype=np.intp), {'extent': np.array([-2, 0], dtype=np.intp)},
     ([0, 0, 0, 1, 2, 3, 3, 3, 3, 2, 1, 0], [3, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3])),
])
@pytest.mark.parametrize('workers', [1, 4])
def test_rectangle_perimeter_routes_and_preserves_order(start, kwargs, saved, workers, monkeypatch):
    import skimage.draw as original
    import skimagexx as xx
    from skimagexx._rectangle_perimeter import rectangle_perimeter

    assert xx.draw.rectangle_perimeter is rectangle_perimeter
    assert inspect.signature(rectangle_perimeter) == inspect.signature(original.rectangle_perimeter)
    entry = xx._core.rectangle_perimeter
    assert inspect.isbuiltin(entry)
    expected = tuple(np.array(v, dtype=np.intp) for v in saved)
    if importlib.util.find_spec('matplotlib') is None:
        with pytest.raises(ImportError, match='requires "matplotlib'):
            original.rectangle_perimeter(start, **kwargs)
    else:
        expected = original.rectangle_perimeter(start, **kwargs)
        assert_array_equal(expected, saved)
    before = np.asarray(start).copy()

    def denied(*args, **kwargs):
        pytest.fail("native perimeter called an upstream provider")

    module = importlib.import_module('skimage.draw.draw')
    for name in ('rectangle_perimeter', 'polygon_perimeter', 'polygon_clip'):
        monkeypatch.setattr(module, name, denied)
    monkeypatch.setattr(original, 'rectangle_perimeter', denied)
    calls = []

    def observe(*args):
        result = entry(*args)
        calls.append(result)
        return result

    monkeypatch.setattr(xx._core, 'rectangle_perimeter', observe)
    old_fallback = xx.set_fallback_enabled(False)
    old_workers = xx.get_num_threads()
    try:
        xx.set_num_threads(workers)
        result = rectangle_perimeter(start, **kwargs)
        assert xx.get_num_threads() == workers
    finally:
        xx.set_num_threads(old_workers)
        xx.set_fallback_enabled(old_fallback)
    assert len(calls) == 1 and calls[0] is result
    assert type(result) is tuple and len(result) == 2
    assert result[0] is not result[1] and not np.shares_memory(*result)
    for array, reference in zip(result, expected, strict=True):
        assert type(array) is np.ndarray and array.dtype == reference.dtype == np.dtype(np.intp)
        assert array.dtype.metadata is None and array.strides == reference.strides
        assert array.flags.owndata and array.flags.writeable and array.flags.aligned and array.base is None
        assert not np.shares_memory(array, reference)
        assert_array_equal(array, reference)
    assert_array_equal(start, before)


@pytest.mark.parametrize(('kwargs', 'error', 'message'), [
    ({}, ValueError, 'Either `end` or `extent` must be given.'),
    ({'end': (2, 2), 'extent': (1, 1)}, ValueError, 'Cannot provide both `end` and `extent`.'),
    ({'end': (2, 2), 'clip': True}, ValueError, 'Must specify clipping shape'),
    ({'end': (5, 5), 'shape': (1, 1), 'clip': True}, IndexError, 'list index out of range'),
])
def test_rectangle_perimeter_errors(kwargs, error, message):
    import skimage.draw as original
    import skimagexx as xx

    previous = xx.set_fallback_enabled(False)
    try:
        if importlib.util.find_spec('matplotlib') is None:
            with pytest.raises(ImportError, match='requires "matplotlib'):
                original.rectangle_perimeter((2, 2), **kwargs)
        else:
            with pytest.raises(error) as expected:
                original.rectangle_perimeter((2, 2), **kwargs)
            assert str(expected.value) == message
        with pytest.raises(error) as actual:
            xx.draw.rectangle_perimeter((2, 2), **kwargs)
        assert str(actual.value) == message
    finally:
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('layout', ['metadata', 'float', 'readonly', 'strided'])
def test_rectangle_perimeter_tags_fallback(layout):
    import skimage.draw as original
    import skimagexx as xx
    from skimagexx._compat import FallbackWarning

    start = np.array([1, 2], dtype=np.intp)
    if layout == 'metadata':
        start = start.astype(np.dtype(np.intp, metadata={'label': 'origin'}))
    elif layout == 'float':
        start = (1.5, 2.5)
    elif layout == 'readonly':
        start.flags.writeable = False
    else:
        start = np.array([1, 0, 2], dtype=np.intp)[::2]
    previous = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='rectangle_perimeter'):
            xx.draw.rectangle_perimeter(start, extent=(3, 4))
        xx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match='rectangle_perimeter'):
            if importlib.util.find_spec('matplotlib') is None:
                with pytest.raises(ImportError, match='requires "matplotlib'):
                    xx.draw.rectangle_perimeter(start, extent=(3, 4))
            else:
                assert_array_equal(xx.draw.rectangle_perimeter(start, extent=(3, 4)),
                                   original.rectangle_perimeter(start, extent=(3, 4)))
    finally:
        xx.set_fallback_enabled(previous)
