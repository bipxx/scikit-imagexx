"""BRIEF class, strict native dispatch and reference parity."""

import builtins
import importlib.util
import inspect
import sys
import warnings

import numpy as np
import pytest
import skimage
from skimage.feature import BRIEF as Original
from skimage.feature.util import DescriptorExtractor
import skimage.feature.brief as original_module
import skimagexx as xx
from skimagexx import _brief


@pytest.fixture
def public():
    assert skimage.__version__ == "0.26.0"
    assert xx.feature.BRIEF is _brief.BRIEF
    assert issubclass(xx.feature.BRIEF, DescriptorExtractor)
    assert inspect.signature(xx.feature.BRIEF) == inspect.signature(Original)
    assert inspect.signature(xx.feature.BRIEF.extract) == inspect.signature(Original.extract)
    assert inspect.isbuiltin(xx._core.brief_sample)
    previous = xx.set_fallback_enabled(False)
    workers = xx.get_num_threads()
    xx.set_num_threads(1)
    try:
        yield xx.feature.BRIEF
    finally:
        xx.set_num_threads(workers)
        xx.set_fallback_enabled(previous)


def assert_descriptors(actual, expected):
    for name in ("mask", "descriptors"):
        received = getattr(actual, name)
        reference = getattr(expected, name)
        assert type(received) is np.ndarray
        np.testing.assert_array_equal(received, reference)
        assert received.dtype == reference.dtype == np.dtype(bool)
        assert received.flags.owndata and received.flags.writeable and received.flags.c_contiguous
        assert received.strides == reference.strides


@pytest.mark.parametrize("mode", ["normal", "UNIFORM"])
@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int32, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("sigma", [0, (1, 0.5)])
def test_reference_parity(public, mode, dtype, sigma, monkeypatch):
    source = np.random.default_rng(7).integers(0, 128, (69, 77), dtype=np.uint8).astype(dtype)
    image = source.T[:, ::-1]
    keypoints = np.asarray([[34, 33], [6, 8], [21.9, 20.4], [34, 33], [0, 0],
                            [7, 9], [np.nan, 16], [np.inf, 25]])
    options = dict(descriptor_size=128, patch_size=15, mode=mode, sigma=sigma, rng=17)
    stock = Original(**options)
    assert stock.extract(image, keypoints) is None

    def forbidden(*args, **kwargs):
        pytest.fail("BRIEF used the original descriptor kernel")

    monkeypatch.setattr(original_module, "_brief_loop", forbidden)
    candidate = public(**options)
    assert candidate.descriptors is candidate.mask is None
    image_before, keypoints_before = image.copy(), keypoints.copy()
    calls = []
    entry = xx._core.brief_sample

    def profile(frame, event, value):
        if event == "c_return" and value is entry:
            calls.append(entry)

    previous = sys.getprofile()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sys.setprofile(profile)
        try:
            assert candidate.extract(image, keypoints) is None
        finally:
            sys.setprofile(previous)
    assert calls == [entry]
    assert_descriptors(candidate, stock)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(keypoints, keypoints_before)
    prior_descriptors = candidate.descriptors
    assert candidate.extract(image, keypoints) is None
    assert_descriptors(candidate, stock)
    assert prior_descriptors is not candidate.descriptors
    np.testing.assert_array_equal(prior_descriptors, candidate.descriptors)


@pytest.mark.parametrize("bit_generator", [np.random.PCG64, np.random.MT19937, np.random.Philox])
def test_generator_spawn_and_mutable_state(public, bit_generator):
    rng = np.random.Generator(bit_generator(13))
    ref_rng = np.random.Generator(bit_generator(13))
    image = np.random.default_rng(1).random((71, 80), dtype=np.float32)
    keypoints = np.array([[30, 30], [40, 50], [0, 0]])
    for _ in range(2):
        candidate = public(rng=rng)
        stock = Original(rng=ref_rng)
        assert candidate.rng is not rng
        assert candidate.rng.random() == stock.rng.random()
        candidate.extract(image, keypoints)
        stock.extract(image, keypoints)
        assert_descriptors(candidate, stock)
        assert candidate.rng.random() == stock.rng.random()
        candidate.descriptor_size = stock.descriptor_size = 48
        candidate.mode = stock.mode = "uniform"
        candidate.extract(image, keypoints)
        stock.extract(image, keypoints)
        assert_descriptors(candidate, stock)
    assert rng.random() == ref_rng.random()


@pytest.mark.parametrize("width", [0, 16])
@pytest.mark.parametrize("keypoints", [np.empty((0, 2)), np.array([[0, 0], [39, 39]]), np.array([[20, 20]])])
def test_empty_results(public, width, keypoints):
    image = np.ones((40, 40))
    candidate = public(width, 9)
    stock = Original(width, 9)
    candidate.extract(image, keypoints)
    stock.extract(image, keypoints)
    assert_descriptors(candidate, stock)


@pytest.mark.parametrize("image,options,error,message", [
    (np.ones((0, 5)), {}, ValueError, "cannot be an empty array"),
    (np.ones((5, 5, 3)), {}, ValueError, "must be a 2-dimensional array"),
    (np.ones((5, 1)), {}, ValueError, "must be a 2-dimensional array"),
    (np.ones((60, 60)), {"sigma": -1}, ValueError, "Sigma values less than zero"),
    (np.ones((60, 60)), {"patch_size": 2}, ValueError, "cannot reshape array"),
])
def test_invalid_extract_retains_state(public, image, options, error, message):
    candidate = public(**options)
    with pytest.raises(error, match=message):
        candidate.extract(image, np.array([[26, 26]]))
    assert candidate.mask is candidate.descriptors is None


def test_constructor_errors_and_global_worker_setting(public):
    with pytest.raises(ValueError, match="must be 'normal' or 'uniform'"):
        public(mode="unsupported")
    with pytest.raises(AttributeError, match="lower"):
        public(mode=None)
    assert isinstance(public(rng=None).rng, np.random.Generator)
    image = np.arange(76 * 80, dtype=np.float32).reshape(76, 80)
    points = np.array([[26, 26], [27, 30], [31, 33]])
    candidate = public()
    candidate.extract(image, points)
    expected = candidate.descriptors.copy()
    xx.set_num_threads(2)
    candidate.extract(image, points)
    np.testing.assert_array_equal(candidate.descriptors, expected)


def test_array_subclass_preparation_obeys_fallback_policy(public, monkeypatch):
    class Image(np.ndarray):
        pass

    image = np.random.default_rng(1).random((75, 82), dtype=np.float32).view(Image)
    points = np.array([[30, 35], [0, 0], [43, 49]])
    stock, candidate = Original(), public()
    stock.extract(image, points)

    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        candidate.extract(image, points)
    assert candidate.mask is candidate.descriptors is None

    prepare = original_module._prepare_grayscale_input_2D
    calls = []

    def tracked_prepare(value):
        calls.append(value)
        return prepare(value)

    def forbidden(*args, **kwargs):
        pytest.fail("BRIEF used the original descriptor kernel")

    monkeypatch.setattr(original_module, "_prepare_grayscale_input_2D", tracked_prepare)
    monkeypatch.setattr(original_module, "_brief_loop", forbidden)
    previous = xx.set_fallback_enabled(True)
    try:
        with pytest.warns(xx.FallbackWarning, match="skimage.feature.brief._prepare_grayscale_input_2D"):
            assert candidate.extract(image, points) is None
    finally:
        xx.set_fallback_enabled(previous)
    assert len(calls) == 1 and calls[0] is image
    assert_descriptors(candidate, stock)


def test_native_only_class_can_load_without_scikit_image(public, monkeypatch):
    original_import = builtins.__import__

    def without_scikit_image(name, *args, **kwargs):
        if name == "skimage" or name.startswith("skimage."):
            raise ModuleNotFoundError("No module named 'skimage'", name="skimage")
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("skimagexx._brief_native_only", _brief.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(builtins, "__import__", without_scikit_image)
    spec.loader.exec_module(module)
    assert issubclass(module.BRIEF, module.DescriptorExtractor)
    candidate = module.BRIEF(sigma=0)
    candidate.extract(np.ones((70, 70)), np.array([[30, 33], [0, 0]]))
    np.testing.assert_array_equal(candidate.mask, [True, False])
    np.testing.assert_array_equal(candidate.descriptors, np.zeros((1, 256), dtype=bool))
