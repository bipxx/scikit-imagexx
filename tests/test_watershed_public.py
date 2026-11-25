"""Public watershed parity, native boundary checks, and backend swapping."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from skimage import segmentation as original

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning
from skimagexx._watershed import watershed


@pytest.fixture
def strict():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


def assert_same(actual, expected, inputs):
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert type(actual) is type(expected) is np.ndarray
    assert actual.dtype.num == expected.dtype.num
    assert actual.dtype.metadata == expected.dtype.metadata
    assert actual.strides == expected.strides
    assert actual.flags.owndata and actual.flags.writeable and actual.base is None
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous
    assert all(not np.shares_memory(actual, a) for a in inputs if type(a) is np.ndarray)


def test_watershed_export_and_native_handoff(strict, monkeypatch):
    exports = {}
    exec('from skimagexx.segmentation import *', exports)
    assert exports['watershed'] is xx.segmentation.watershed is watershed
    assert inspect.signature(watershed) == inspect.signature(original.watershed)
    entry = _core.watershed
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    seen = []

    def observe(*args):
        seen.append(args)
        return entry(*args)

    monkeypatch.setattr(_core, 'watershed', observe)
    image = np.zeros((5, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int16)
    labels[0, 3], labels[-1, 4] = 251, -12

    def pipeline(backend):
        return backend.segmentation.watershed(image, labels, watershed_line=True)

    expected = pipeline(SimpleNamespace(segmentation=original))
    marker = object()
    backend = SimpleNamespace(segmentation=SimpleNamespace(watershed=lambda *a, **k: marker))
    assert pipeline(backend) is marker
    assert_same(pipeline(xx), expected, (image, labels))
    assert len(seen) == 1 and seen[0][0] is image and seen[0][1] is labels


@pytest.mark.parametrize('dtype', [np.bool_, np.int8, np.uint8, np.int16, np.uint16,
                                   np.int32, np.uint32, np.int64, np.uint64,
                                   np.longlong, np.ulonglong])
def test_markers_output_metadata_and_signedness(strict, dtype):
    image = np.arange(35, dtype=np.float16).reshape(5, 7) % 5
    markers = np.zeros(image.shape, dtype=dtype)
    if dtype is np.bool_:
        markers.view(np.uint8)[1, 0] = 2
        markers[3, 4] = True
    else:
        limits = np.iinfo(dtype)
        markers[1, 0] = limits.max
        markers[3, 4] = limits.min or 1
    mask = np.ones(image.shape, dtype=np.int16)
    mask[0, :] = 0
    mask[2:, 2] = -9
    before = tuple(a.tobytes() for a in (image, markers, mask))
    expected = original.watershed(image, markers, connectivity=2, mask=mask, compactness=0.225)
    assert_same(watershed(image, markers, 2, None, mask, 0.225), expected, (image, markers, mask))
    assert tuple(a.tobytes() for a in (image, markers, mask)) == before


@pytest.mark.parametrize('shape', [(0,), (2, 0), (1, 1), (4, 5), (4, 5, 6), (2, 3, 1, 5)])
@pytest.mark.parametrize('markers', [None, 4])
def test_automatic_and_count_markers(strict, shape, markers):
    image = (np.arange(np.prod(shape), dtype=np.uint8) % 11).reshape(shape)
    mask = np.ones(shape, dtype=bool)
    mask.flat[3::4] = False
    args = (image, markers)
    kwargs = {'mask': mask} if image.size else {}
    if markers is not None and not image.size:
        with pytest.raises(ZeroDivisionError):
            watershed(*args, **kwargs)
        return
    assert_same(watershed(*args, **kwargs), original.watershed(*args, **kwargs), (image, mask))


def test_ordered_priority_queue_and_neighbor_parity(strict):
    rng = np.random.default_rng(62534)
    for rank in (1, 2, 3):
        for case in range(50):
            shape = tuple(rng.integers(1, 9, size=rank))
            image = rng.integers(-3, 7, size=shape).astype(np.float64)
            if case % 7 == 0:
                image.flat[::9] = np.nan
                image.flat[1::9] = np.inf
            markers = rng.integers(-4, 5, size=shape, dtype=np.int32)
            markers[rng.random(size=shape) < 0.8] = 0
            mask = rng.random(size=shape) > 0.2
            connectivity = 1 + case % rank
            if case % 4 == 0:
                connectivity = rng.random(size=(3,) * rank) > 0.4
                connectivity[(1,) * rank] = True
            for compactness, line in ((0, False), (0, True), (0.2, False), (0.2, True)):
                args = (image, markers, connectivity, None, mask, compactness, line)
                assert_same(watershed(*args), original.watershed(*args), (image, markers, mask))


def test_custom_offset_and_broadcast_markers(strict):
    image = np.arange(35, dtype=np.uint8).reshape(5, 7) % 6
    markers = np.array([[0], [0], [-200], [0], [0]], np.int32)
    footprint = np.array([[True, True, False], [True, False, False]], dtype=bool)
    offset = np.array([1, 2], dtype=np.intp)
    args = (image, markers, footprint, offset)
    assert_same(watershed(*args), original.watershed(*args), (image, markers))


def test_validates_shape_and_pybind_storage(strict):
    image = np.zeros((3, 4))
    markers = np.zeros(image.shape, dtype=np.int32)
    with pytest.raises(ValueError, match='must have same shape'):
        watershed(image, markers, mask=np.ones((1, 4), bool))
    with pytest.raises(ValueError, match='offsets exceed padded storage'):
        _core.watershed(image, markers, padding=(1, 1), neighbors=(100,))
    with pytest.raises(ValueError, match='requires ndarray-backed storage'):
        extent = np.lib.stride_tricks.as_strided(np.zeros(1), shape=image.shape, strides=image.strides)
        watershed(extent, markers)


@pytest.mark.parametrize('storage', ['subclass', 'tagged', 'longdouble', 'strided', 'readonly', 'swapped'])
def test_fallback_policy_for_array_storage(storage):
    class Subclass(np.ndarray):
        pass

    image = np.zeros((5, 7))
    if storage == 'subclass':
        image = image.view(Subclass)
    elif storage == 'tagged':
        image = image.view(np.dtype('float64', metadata={'terrain': 'surface'}))
    elif storage == 'longdouble':
        image = image.astype(np.longdouble)
    elif storage == 'strided':
        image = image[:, ::-1]
    elif storage == 'readonly':
        image.flags.writeable = False
    else:
        image = image.astype(image.dtype.newbyteorder('S'))
    labels = np.zeros(image.shape, dtype=np.intc)
    labels[0, -1] = 9
    previous = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            watershed(image, labels)
        with pytest.raises((ValueError, TypeError), match='watershed requires'):
            _core.watershed(image, labels)
        xx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match='watershed executed'):
            actual = watershed(image, labels)
        assert_same(actual, original.watershed(image, labels), (image, labels))
    finally:
        xx.set_fallback_enabled(previous)
