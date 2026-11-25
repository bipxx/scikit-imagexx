import importlib
import inspect

import numpy as np
import pytest
from numpy.testing import assert_array_equal


TRIANGLE = ([0, 1, 2, 2, 1, 0, 0, 0, 0], [0, 1, 1, 1, 2, 2, 2, 1, 0])


@pytest.mark.parametrize(('r', 'c', 'kwargs', 'saved'), [
    ([0, 2, 0], [0, 1, 2], {}, TRIANGLE),
    ([0, 2, 0, 0], [0, 1, 2, 0], {}, TRIANGLE),
    ([4, 4, 4], [2, 2, 2], {}, ([4, 4], [2, 2])),
    ((0.5, 2.5, -0.5, -1.5), (1.5, 0.5, 2.5, -0.5), {},
     ([0, 1, 2, 2, 1, 0, 0, -1, -2, -2, -1, 0], [2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2])),
    (np.array([9, 7, 5, 3], dtype=np.float32), np.array([2, 5, 8, 11], dtype=np.int16), {},
     ([9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9],
      [2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2])),
    ([-2, 1, 4, 1], [1, 4, 1, -2], {'shape': (3, 3, 4)}, ([], [])),
    ([-2, 1, 4, 1], [1, 4, 1, -2], {'shape': (3, 3), 'clip': True},
     ([0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 0], [0, 1, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0])),
    ([1, 1, 8, -2], [1, 8, 1, -2], {'shape': (5, 4), 'clip': True},
     ([0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0],
      [0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0])),
    ([-1, 1, -1, 1], [.5, .5, 1.5, 1.5], {'shape': (1, 4), 'clip': True},
     ([0] * 12, [1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 1])),
    ([-2, 2, 2, -2], [-2, -2, 2, 2], {'shape': (0, 0), 'clip': True}, ([], [])),
    ([0, 0, 6, 6], [0, 6, 6, 0], {'shape': (4, 4)}, ([0, 0, 0, 0, 3, 2, 1, 0], [0, 1, 2, 3, 0, 0, 0, 0])),
])
@pytest.mark.parametrize('workers', [1, 4])
def test_polygon_perimeter_routes_and_preserves_order(r, c, kwargs, saved, workers, monkeypatch):
    import skimage.draw as original
    import skimagexx as xx
    from skimagexx._polygon_perimeter import polygon_perimeter

    assert xx.draw.polygon_perimeter is polygon_perimeter
    assert inspect.signature(polygon_perimeter) == inspect.signature(original.polygon_perimeter)
    entry = xx._core.polygon_perimeter
    assert inspect.isbuiltin(entry)
    expected = tuple(np.array(v, dtype=np.intp) for v in saved)
    if importlib.util.find_spec('matplotlib') is None:
        with pytest.raises(ImportError, match='requires "matplotlib'):
            original.polygon_perimeter(r, c, **kwargs)
    else:
        expected = original.polygon_perimeter(r, c, **kwargs)
        assert_array_equal(expected, saved)
    before = [np.asarray(v).copy() for v in (r, c)]

    def denied(*args, **kwargs):
        pytest.fail("native perimeter called an upstream provider")

    module = importlib.import_module('skimage.draw.draw')
    for name in ('polygon_perimeter', 'polygon_clip', 'line'):
        monkeypatch.setattr(module, name, denied)
    monkeypatch.setattr(original, 'polygon_perimeter', denied)
    calls = []

    def observe(*args):
        result = entry(*args)
        calls.append(result)
        return result

    monkeypatch.setattr(xx._core, 'polygon_perimeter', observe)
    old_fallback = xx.set_fallback_enabled(False)
    old_workers = xx.get_num_threads()
    try:
        xx.set_num_threads(workers)
        result = polygon_perimeter(r, c, **kwargs)
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
    assert_array_equal(r, before[0])
    assert_array_equal(c, before[1])


@pytest.mark.parametrize(('r', 'c', 'kwargs', 'error', 'message'), [
    ([], [], {}, ValueError, 'zero-size array to reduction operation minimum which has no identity'),
    ([1], [2], {}, IndexError, 'list index out of range'),
    ([1, 2], [2, 3], {}, IndexError, 'list index out of range'),
    ([4, 4, 5], [4, 5, 4], {'shape': (2, 2), 'clip': True}, IndexError, 'list index out of range'),
    ([0, 0, 1], [0, 1, 0], {'clip': True}, ValueError, 'Must specify clipping shape'),
])
def test_polygon_perimeter_errors(r, c, kwargs, error, message):
    import skimage.draw as original
    import skimagexx as xx

    previous = xx.set_fallback_enabled(False)
    try:
        if importlib.util.find_spec('matplotlib') is None:
            with pytest.raises(ImportError, match='requires "matplotlib'):
                original.polygon_perimeter(r, c, **kwargs)
        else:
            with pytest.raises(error) as expected:
                original.polygon_perimeter(r, c, **kwargs)
            assert str(expected.value) == message
        with pytest.raises(error) as actual:
            xx.draw.polygon_perimeter(r, c, **kwargs)
        assert str(actual.value) == message
    finally:
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('layout', ['metadata', 'byteorder', 'float16', 'longdouble', 'readonly', 'strided', 'subclass'])
def test_polygon_perimeter_tags_fallback(layout):
    import skimage.draw as original
    import skimagexx as xx
    from skimagexx._compat import FallbackWarning

    dtype = {
        'metadata': np.dtype(np.int32, metadata={'label': 'rows'}),
        'byteorder': np.dtype(np.int32).newbyteorder('S'),
        'float16': np.float16,
        'longdouble': np.longdouble,
    }.get(layout, np.intp)
    r, c = np.array([0, 2, 0], dtype=dtype), [0, 1, 2]
    if layout == 'readonly':
        r.flags.writeable = False
    elif layout == 'strided':
        r = np.array([0, 0, 2, 0, 0])[::2]
    elif layout == 'subclass':
        class Subclass(np.ndarray):
            pass
        r = r.view(Subclass)
    previous = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='polygon_perimeter'):
            xx.draw.polygon_perimeter(r, c)
        xx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match='polygon_perimeter'):
            if importlib.util.find_spec('matplotlib') is None:
                with pytest.raises(ImportError, match='requires "matplotlib'):
                    xx.draw.polygon_perimeter(r, c)
            else:
                assert_array_equal(xx.draw.polygon_perimeter(r, c), original.polygon_perimeter(r, c))
    finally:
        xx.set_fallback_enabled(previous)
