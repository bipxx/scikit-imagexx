"""Public contract and ordered output of the two marching-cubes methods."""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.measure import marching_cubes as reference
from skimagexx import FallbackWarning, set_fallback_enabled
from skimagexx.measure import marching_cubes


@pytest.fixture
def strict():
    previous = set_fallback_enabled(False)
    try:
        yield
    finally:
        set_fallback_enabled(previous)


def assert_mesh(actual, expected):
    assert len(actual) == len(expected) == 4
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype
        assert_array_equal(a, b)


def test_signature():
    assert inspect.signature(marching_cubes) == inspect.signature(reference)


@pytest.mark.parametrize('method', ['lewiner', 'lorensen'])
def test_all_cube_signs(method, strict):
    rng = np.random.default_rng(123)
    for index in range(1, 255):
        signs = np.array([1 if index & (1 << k) else -1 for k in range(8)])
        volume = (signs * rng.uniform(0.1, 2, size=8)).reshape(2, 2, 2).astype(np.float32)
        assert_mesh(marching_cubes(volume, 0, method=method), reference(volume, 0, method=method))


@pytest.mark.parametrize('method', ['lewiner', 'lorensen'])
@pytest.mark.parametrize('step', [1, 2, 3])
@pytest.mark.parametrize('allow_degenerate', [True, False])
def test_ordered_mesh(method, step, allow_degenerate, strict):
    z, y, x = np.ogrid[-6:7, -7:8, -8:9]
    volume = (z*z + 2*y*y + 3*x*x).astype(np.dtype('single')).transpose(2, 0, 1)
    mask = np.array(volume < 60, dtype='uint8', order='C')[:, :, ::-1]
    options = dict(
        method=method, step_size=step, allow_degenerate=allow_degenerate,
        gradient_direction='ascent', spacing=(0.5, 2, 1.25), mask=mask,
    )
    assert_mesh(marching_cubes(volume, 35, **options), reference(volume, 35, **options))


@pytest.mark.parametrize('dtype', ['float64', 'uint16', '>f4', 'bool'])
def test_default_level_dtypes(dtype, strict):
    volume = np.zeros((7, 8, 9), dtype=dtype)
    volume[2:5, 3:6, 3:7] = 1
    assert_mesh(marching_cubes(volume), reference(volume))


def test_metadata_boundary():
    volume = np.zeros((4, 4, 4), dtype=np.dtype('f4', metadata={'unit': 'voxel'}))
    volume[1:3, 1:3, 1:3] = 1
    with pytest.warns(FallbackWarning):
        assert_mesh(marching_cubes(volume), reference(volume))


@pytest.mark.parametrize('options', [
    {'method': 'unknown'}, {'spacing': (1, 2)}, {'level': 7},
    {'step_size': 0}, {'gradient_direction': 'sideways'},
    {'mask': np.zeros((2, 2, 2), bool)}, {'step_size': 4},
])
def test_errors(options, strict):
    volume = np.zeros((4, 4, 4), np.float32)
    volume[1, 1, 1] = 1
    with pytest.raises((ValueError, RuntimeError)) as upstream:
        reference(volume, **options)
    with pytest.raises(type(upstream.value)) as native:
        marching_cubes(volume, **options)
    assert str(native.value) == str(upstream.value)


def test_pipeline_hot_swap_and_storage_owners(monkeypatch, strict):
    from skimage import measure as upstream
    from skimagexx import _core
    from skimagexx._marching_cubes import marching_cubes as wrapper

    assert marching_cubes is wrapper
    assert inspect.isbuiltin(_core.measure_marching_cubes)
    volume = np.zeros((4, 4, 4), np.float32)
    volume[1:3, 1:3, 1:3] = 1
    expected = upstream.marching_cubes(volume)
    monkeypatch.setattr(upstream, 'marching_cubes', marching_cubes)
    output = upstream.marching_cubes(volume)
    assert_mesh(output, expected)
    assert not any(np.shares_memory(array, volume) for array in output)
    assert output[0].base.flags.owndata
    assert output[2].base.flags.owndata
