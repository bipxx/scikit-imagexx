"""NumPy-only fixture/checker intake: original and backend imports are refused."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def intake():
    import numpy as np
    assert np.__version__ in ('1.24.4', '2.5.4')
    attempts = []

    class Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('scipy', 'skimage', 'skimagexx'):
                attempts.append(fullname)
                raise AssertionError('Intake attempted original/backend import: ' + fullname)

    assert not any(n.split('.')[0] in ('scipy', 'skimage', 'skimagexx') for n in sys.modules)
    guard = Refuse()
    sys.meta_path.insert(0, guard)
    try:
        modules = []
        for name in ('draw_polygon', 'draw_coordinates'):
            spec = importlib.util.spec_from_file_location('intake_' + name, ROOT / ('benchmarks/' + name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        yield np, modules[0], modules[1].snapshot
        assert attempts == []
    finally:
        sys.meta_path.remove(guard)


@pytest.mark.parametrize('index', range(12))
def test_frozen_fixture_builds_only_safe_mutable_columns(intake, index):
    np, driver, snapshot = intake
    spec = json.loads(driver.CONTRACT.read_bytes())['cases'][index]
    args, options, buffers, bounds = driver.fixture(spec, np)
    if spec['api'] == 'draw.polygon':
        columns = [np.atleast_1d(v) for v in args]
        assert options['shape'] == (None if spec['shape'] is None else tuple(spec['shape']))
    else:
        assert args[0] == tuple(spec['shape']) and options == {}
        columns = list(np.asarray(args[1]).T)
    assert len(columns) == 2 and all(a.ndim == 1 and a.size == len(spec['vertices']) for a in columns)
    for dimension, a in enumerate(columns):
        normalized = np.ascontiguousarray(a, dtype=np.float64)
        assert normalized.flags.c_contiguous and normalized.flags.writeable
        assert normalized.tolist() == [v[dimension] for v in spec['vertices']]
    assert all(a.flags.writeable and a.flags.aligned and a.dtype.kind in 'if' for a in buffers)
    assert bounds['scan_pixels'] <= 512**2 and bounds['edge_tests'] <= 64 * 1024**2
    if spec['carrier'] == 'strided':
        assert buffers and any(not a.flags.c_contiguous for a in buffers)


@pytest.mark.parametrize('coordinates', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_distinct_owning_output_is_accepted(intake, coordinates, empty):
    np, driver, snapshot = intake
    expected = ((np.arange(0 if empty else 3, dtype=np.intp), np.arange(0 if empty else 3, dtype=np.intp))
                if coordinates else np.zeros((0 if empty else 2, 3), dtype=bool))
    actual = tuple(a.copy() for a in expected) if coordinates else expected.copy()
    driver.checked_output(actual, expected, [], np, snapshot, {'output_pairs_upper_bound': 3}, coordinates)


@pytest.mark.parametrize('coordinates', [False, True])
@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('alias', ['same-reference', 'same-input', 'shared-reference'])
def test_original_or_input_alias_is_rejected_even_when_empty(intake, coordinates, empty, alias):
    np, driver, snapshot = intake
    expected = ((np.arange(0 if empty else 3, dtype=np.intp), np.arange(0 if empty else 3, dtype=np.intp))
                if coordinates else np.zeros((0 if empty else 2, 3), dtype=bool))
    actual = tuple(a.copy() for a in expected) if coordinates else expected.copy()
    buffers = []
    if alias == 'same-reference':
        actual = expected
    elif alias == 'same-input':
        buffers = [actual[0] if coordinates else actual]
    else:
        actual = (expected[0].view(), actual[1]) if coordinates else expected.view()
    with pytest.raises(AssertionError):
        driver.checked_output(actual, expected, buffers, np, snapshot,
                              {'output_pairs_upper_bound': 3}, coordinates)


@pytest.mark.parametrize('empty', [False, True])
def test_coordinate_pair_cannot_reuse_one_array(intake, empty):
    np, driver, snapshot = intake
    expected = tuple(np.arange(0 if empty else 3, dtype=np.intp) for _ in range(2))
    value = expected[0].copy()
    with pytest.raises(AssertionError):
        driver.checked_output((value, value), expected, [], np, snapshot,
                              {'output_pairs_upper_bound': 3}, True)
