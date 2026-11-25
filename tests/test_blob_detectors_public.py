"""Live stock 0.26.0 parity for installed public scale-space detectors."""

import inspect
import runpy
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np
import pytest
import skimage
from skimage import feature as original

import skimagexx as xx
from skimagexx import _blob_detectors, _core


NAMES = ('blob_dog', 'blob_log', 'blob_doh')
BENCHMARKS = tuple(runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                      'benchmarks/blob_detectors.py'))['make_cases']())


def _compare(actual, expected, source):
    assert type(actual) is np.ndarray and actual.flags.owndata and actual.flags.c_contiguous
    assert not np.shares_memory(actual, source)
    assert (actual.shape, actual.dtype) == (expected.shape, expected.dtype)
    np.testing.assert_array_equal(actual, expected)


@pytest.fixture
def strict():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('case', BENCHMARKS, ids=[row[0] for row in BENCHMARKS])
def test_blob_detector_public_hot_swap(case, strict, monkeypatch):
    _, name, (image,), options = case
    public = getattr(xx.feature, name)
    stock = getattr(original, name)
    entry = _core.blob_detect
    assert skimage.__version__ == '0.26.0'
    assert public is getattr(_blob_detectors, name) and name in xx.feature.__all__
    assert inspect.signature(public) == inspect.signature(stock)
    assert inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core'
    assert _core.__file__.endswith(tuple(EXTENSION_SUFFIXES))

    def pipeline(backend):
        return getattr(backend.feature, name)(image, **options)

    before = image.tobytes()
    expected = pipeline(skimage)
    assert len(expected) >= 20
    calls = []

    def trace(value, *args):
        assert value is image
        calls.append(value)
        return entry(value, *args)

    def reject(*args, **kwargs):
        pytest.fail('the installed native blob detector entered the stock provider')

    monkeypatch.setattr(_core, 'blob_detect', trace)
    monkeypatch.setattr(original, name, reject)
    _compare(pipeline(xx), expected, image)
    assert len(calls) == 1 and image.tobytes() == before


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', (np.uint8, np.float32, np.float64))
def test_blob_dtype_threshold_and_empty(name, dtype, strict):
    image = np.zeros((27, 31), dtype=dtype)
    image[8, 9] = 255 if dtype is np.uint8 else 1
    stock = getattr(original, name)
    public = getattr(xx.feature, name)
    for options in ({'max_sigma': 3, 'threshold': .001, 'threshold_rel': .15},
                    {'max_sigma': 3, 'threshold': 2}, {}):
        _compare(public(image, **options), stock(image, **options), image)


@pytest.mark.parametrize('name,options', [
    ('blob_dog', dict(min_sigma=(.8, 1.4), max_sigma=[2.5, 4], sigma_ratio=1.8,
                      exclude_border=(3, 5), threshold=.004, threshold_rel=.15)),
    ('blob_log', dict(min_sigma=(1, 1.4), max_sigma=(3, 4), log_scale=True,
                      exclude_border=True, threshold=.004, threshold_rel=.15)),
    ('blob_doh', dict(min_sigma=2.1, max_sigma=2.2, num_sigma=2, log_scale=True, threshold=.001)),
])
def test_blob_anisotropic_border_log_and_duplicate_scale(name, options, strict):
    image = np.zeros((31, 33), dtype=np.float32)
    image[2, 4] = .6
    image[12, 11] = 1
    image[23, 25] = .9
    result = getattr(xx.feature, name)(image, **options)
    expected = getattr(original, name)(image, **options)
    assert len(expected)
    _compare(result, expected, image)


@pytest.mark.parametrize('name', NAMES)
def test_blob_effective_fallback_and_guard(name, strict, monkeypatch):
    image = np.eye(10, dtype=np.float64)
    readonly = image.copy()
    readonly.flags.writeable = False
    subtype = image.view(type('Image', (np.ndarray,), {}))
    exotic = np.lib.stride_tricks.as_strided(image, image.shape, image.strides)
    decorated = image.view(np.dtype('f8', metadata={'source': 'tag'}))
    nonnative = image.astype(image.dtype.newbyteorder('S'))
    values = (image[:, ::-1], readonly, subtype, exotic, decorated, nonnative, image.astype(np.longdouble))
    direct = _core.blob_detect
    for value in values:
        with pytest.raises(ValueError):
            direct(value, 1, [(1., 1.)], True, 1., .1, None, .5, (0, 0))

    def reject(*args):
        pytest.fail('unsupported image reached native blob parsing')

    monkeypatch.setattr(_core, 'blob_detect', reject)
    public = getattr(xx.feature, name)
    for value in values:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            public(value)

    view = image[:, ::-1]
    expected = getattr(original, name)(view, max_sigma=3, threshold=.001)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning, match=rf'skimage.feature.{name}') as caught:
        result = public(view, max_sigma=3, threshold=.001)
    assert len(caught) == 1
    _compare(result, expected, view)


def test_blob_nonfinite_image_uses_explicit_fallback(strict):
    image = np.zeros((4, 7), dtype=np.float32)
    image[1, 2] = np.nan
    assert _core.blob_detect(image, 1, [(1., 1.)], True, 1., .1, None, .5, (0, 0)) is None
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        xx.feature.blob_log(image)


@pytest.mark.parametrize('name', NAMES)
def test_blob_worker_count_preserves_order_and_dtype(name, strict):
    rows, cols = np.ogrid[:513, :521]
    image = (np.exp(-((rows - 135) ** 2 + (cols - 202) ** 2) / 8) +
             .8 * np.exp(-((rows - 404) ** 2 + (cols - 416) ** 2) / 12)).astype(np.float32)
    public = getattr(xx.feature, name)
    options = dict(min_sigma=2, max_sigma=4, threshold=.004)
    if name != 'blob_dog':
        options['num_sigma'] = 3
    before = xx.get_num_threads()
    assert before == 1
    serial = public(image, **options)
    expected = getattr(original, name)(image, **options)
    assert len(expected) >= 2
    _compare(serial, expected, image)
    try:
        xx.set_num_threads(4)
    except ValueError as error:
        assert str(error) == 'this build has no OpenMP support' and xx.get_num_threads() == 1
    else:
        try:
            assert xx.get_num_threads() == 4
            _compare(public(image, **options), serial, image)
        finally:
            xx.set_num_threads(before)
