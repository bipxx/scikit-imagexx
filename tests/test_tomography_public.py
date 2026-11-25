import inspect
import warnings

import numpy as np
import pytest
from skimage import transform as stock
from skimagexx import FallbackWarning, _core, set_fallback_enabled, transform


APIS = ("radon", "iradon", "iradon_sart")


@pytest.fixture(autouse=True)
def strict_native():
    previous = set_fallback_enabled(False)
    try:
        yield
    finally:
        set_fallback_enabled(previous)


def _phantom(rows, columns, dtype):
    r, c = np.ogrid[:rows, :columns]
    result = (((r - rows // 2 + 2) / (rows / 5)) ** 2 + ((c - columns // 2) / (columns / 4)) ** 2 < 1)
    return result.astype(dtype)


def _match(result, reference):
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape
    tolerance = 1.5e-6 if reference.dtype == np.float32 else 5e-12
    np.testing.assert_allclose(result, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("name", APIS)
def test_public_signature(name):
    assert inspect.signature(getattr(transform, name)) == inspect.signature(getattr(stock, name))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int16, np.uint8, np.bool_])
@pytest.mark.parametrize("circle", [True, False])
def test_radon_geometry_and_conversion(dtype, circle):
    source = _phantom(29, 32, dtype)
    if source.dtype.kind in "iu":
        source *= 193
    for angles in (None, np.array([-120, -0.5, 0, 30, 60, 90, 134.5, 180, 270], dtype=np.float32)):
        kwargs = dict(theta=angles, circle=circle, preserve_range=angles is not None)
        _match(transform.radon(source, **kwargs), stock.radon(source, **kwargs))


def test_radon_circle_warning_and_crop_asymmetry():
    x = np.arange(14 * 9, dtype=np.float64).reshape(14, 9)
    with pytest.warns(UserWarning, match="image must be zero outside"):
        expected = stock.radon(x, [0, 33, 90])
    with pytest.warns(UserWarning, match="image must be zero outside"):
        actual = transform.radon(x, [0, 33, 90])
    _match(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], x[3:12].sum(axis=0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("filter_name", ["ramp", "shepp-logan", "cosine", "hamming", "hann", None])
@pytest.mark.parametrize("interpolation", ["linear", "nearest", "cubic"])
def test_iradon_filter_and_interpolation(dtype, filter_name, interpolation):
    rng = np.random.default_rng(161)
    x = rng.normal(size=(27, 7)).astype(dtype)
    fast_angles = np.array([0, 30, 45, 60, 90, 120, 150], dtype=dtype)
    for kwargs in (
        dict(theta=fast_angles, output_size=np.int64(30), circle=False),
        dict(circle=True),
    ):
        kwargs.update(filter_name=filter_name, interpolation=interpolation)
        _match(transform.iradon(x, **kwargs), stock.iradon(x, **kwargs))


@pytest.mark.parametrize("count", [4, 5, 6, 63])
def test_iradon_short_cubic_and_power_of_two_padding(count):
    x = np.random.default_rng(732).random((count, 9))
    _match(transform.iradon(x, circle=False, output_size=count + 3, interpolation="cubic"),
           stock.iradon(x, circle=False, output_size=count + 3, interpolation="cubic"))


def test_iradon_integer_ranges_and_single_detector():
    x = np.arange(117, dtype=np.int16).reshape(13, 9) - 40
    for preserve in (True, False):
        _match(transform.iradon(x, preserve_range=preserve), stock.iradon(x, preserve_range=preserve))
    _match(transform.iradon(x[:1]), stock.iradon(x[:1]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count", [7, 48])
def test_sart_default(dtype, count):
    x = np.random.default_rng(26).random((35, count)).astype(dtype)
    _match(transform.iradon_sart(x), stock.iradon_sart(x))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("clip", [None, (-0.1, 0.14), (0.7, 0.2)])
def test_sart_shift_clip_and_initial_mutation(dtype, clip):
    rng = np.random.default_rng(15)
    x = rng.random((24, 8)).astype(dtype)
    image = rng.uniform(-0.05, 0.05, size=(24, 24)).astype(dtype)
    original_image = image.copy()
    kwargs = dict(theta=[100.2, 70, -8, 59, 2, 170, 105.5, 42],
                  projection_shifts=[0.125, -0.75, 0, 0.75, 0.25, 0, 0, -0.25],
                  clip=clip, relaxation=0.28, dtype=dtype)
    actual = transform.iradon_sart(x, image=image, **kwargs)
    expected = stock.iradon_sart(x, image=original_image, **kwargs)
    _match(actual, expected)
    np.testing.assert_allclose(image, original_image, atol=1e-6 if dtype == np.float32 else 5e-12)
    assert (actual is image) == (clip is None)


def test_sart_sinogram_precision_conversion():
    x = np.random.default_rng(319).random((31, 31), dtype=np.float32)
    _match(transform.iradon_sart(x, dtype="double"), stock.iradon_sart(x, dtype="double"))


@pytest.mark.parametrize("name", APIS)
def test_dtype_alias_is_native(name):
    descriptor = np.dtype("=d", copy=True)
    x = _phantom(21, 21, np.float64).astype(descriptor)
    assert x.dtype == np.dtype(np.float64) and x.dtype is not np.dtype(np.float64)
    _match(getattr(transform, name)(x), getattr(stock, name)(x))


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize("name", APIS)
@pytest.mark.parametrize("storage", ["metadata", "native_longdouble", "noncontiguous", "readonly",
                                     "subclass", "byte_swapped"])
def test_unsupported_storage_routes_to_stock(name, storage):
    x = _phantom(25, 25, np.float64)
    if storage == "metadata":
        x = x.view(np.dtype("d", metadata={"sample": "tagged"}))
    elif storage == "native_longdouble":
        x = x.astype(np.longdouble)
    elif storage == "noncontiguous":
        x = x[:, ::-1]
    elif storage == "readonly":
        x.flags.writeable = False
    elif storage == "subclass":
        x = x.view(ArraySubclass)
    else:
        x = x.astype(x.dtype.newbyteorder("S"))
    native, original = getattr(transform, name), getattr(stock, name)
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        native(x)

    set_fallback_enabled(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            expected = original(x)
        except (ValueError, TypeError) as exc:
            with pytest.warns(FallbackWarning), pytest.raises(type(exc)) as actual:
                native(x)
            assert str(actual.value) == str(exc)
        else:
            with pytest.warns(FallbackWarning):
                actual = native(x)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype
            assert actual.dtype.metadata == expected.dtype.metadata


def test_direct_bindings_reject_tagged_inputs_and_subclasses():
    x = np.zeros((8, 6))
    for source in (x.view(np.dtype("d", metadata={"tag": 1})), x.view(ArraySubclass), x.astype(np.longdouble)):
        assert _core._tomography_radon(source, np.zeros((4, 6)), True) is None
        assert _core._tomography_iradon(source, np.zeros((2, 6)), 8, 0, 0, True) is None
        assert _core._tomography_sart(source, np.arange(6.0), np.zeros(6), None, None, 0.15, True) is None


@pytest.mark.parametrize("name", APIS)
def test_tagged_theta_routes_to_stock(name):
    x = _phantom(21, 21, np.float64)
    theta = np.linspace(0, 180, 21, endpoint=False).view(np.dtype("d", metadata={"units": "degrees"}))
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        getattr(transform, name)(x, theta)
    set_fallback_enabled(True)
    with pytest.warns(FallbackWarning):
        actual = getattr(transform, name)(x, theta)
    np.testing.assert_array_equal(actual, getattr(stock, name)(x, theta))


@pytest.mark.parametrize(
    "name, kwargs, message",
    [
        ("iradon", {"filter_name": "unknown"}, "Unknown filter"),
        ("iradon", {"interpolation": "unknown"}, "Unknown interpolation"),
        ("iradon", {"theta": [0, 90]}, "does not match"),
        ("iradon_sart", {"projection_shifts": [0, 1]}, "does not match"),
        ("iradon_sart", {"clip": [0, 1, 2]}, "length-2"),
    ],
)
def test_invalid_arguments_keep_upstream_validation(name, kwargs, message):
    set_fallback_enabled(True)
    x = _phantom(21, 21, np.float64)
    with pytest.raises(ValueError, match=message) as expected:
        getattr(stock, name)(x, **kwargs)
    with pytest.warns(FallbackWarning), pytest.raises(ValueError, match=message) as actual:
        getattr(transform, name)(x, **kwargs)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("workers", [1, 4])
def test_worker_parity(workers):
    previous = _core.get_num_threads()
    try:
        _core.set_num_threads(workers)
    except ValueError as exc:
        # The SDK also supports builds compiled without OpenMP.
        assert workers == 4 and "OpenMP" in str(exc)
        return
    try:
        x = _phantom(78, 78, np.float64)
        sino = transform.radon(x)
        _match(sino, stock.radon(x))
        _match(transform.iradon(sino), stock.iradon(sino))
        _match(transform.iradon_sart(sino), stock.iradon_sart(sino))
    finally:
        _core.set_num_threads(previous)
