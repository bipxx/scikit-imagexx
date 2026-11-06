"""Prospective input-safety and source closure, with target imports forbidden."""
import copy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prospect():
    attempts = []

    class Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('numpy', 'scipy', 'skimage', 'skimagexx'):
                attempts.append(fullname)
                raise AssertionError('Prospective benchmark imported target: ' + fullname)

    guard = Refuse()
    sys.meta_path.insert(0, guard)
    try:
        spec = importlib.util.spec_from_file_location('prospective_draw_polygon', ROOT / 'benchmarks/draw_polygon.py')
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        contract = json.loads(driver.CONTRACT.read_bytes())
        yield driver, contract
        assert attempts == []
    finally:
        sys.meta_path.remove(guard)


def test_original_recipes_and_allocation_are_bounded_before_any_calls(prospect):
    driver, contract = prospect
    assert len(contract['cases']) == len({s['id'] for s in contract['cases']}) == 12
    assert [sum(s['api'] == api for s in contract['cases']) for api in contract['apis']] == [6, 6]
    bounds = [driver.source_safe(s) for s in contract['cases']]
    assert max(v['scan_pixels'] for v in bounds) <= 512**2
    assert max(v['edge_tests'] for v in bounds) <= 64 * 1024**2
    assert max(v['mask_pixels'] for v in bounds) <= 512**2
    assert sum(v['scan_pixels'] == 0 for v in bounds) == 2
    assert contract['worker_count'] == 1 and contract['repeat'] == 7 and contract['raw_samples'] == 168
    assert contract['target_ms'] == 100 and contract['warmup_calibration_min_ms'] == 250


@pytest.mark.parametrize('field,value', [
    ('api', 'draw.other'), ('shape', [1]), ('shape', [-1, 8]),
    ('shape', [513, 8]), ('dtype', 'object'), ('carrier', 'readonly'),
    ('vertices', []), ('vertices', [[1, 2, 3]]),
    ('vertices', [[float('nan'), 2]]), ('vertices', [[2**40, 1]]),
    ('vertices', [[.5, 1]]), ('vertices', [[1, 2]] * 65),
])
def test_unassessed_original_recipes_refused(prospect, field, value):
    driver, contract = prospect
    spec = copy.deepcopy(contract['cases'][0])
    spec[field] = value
    with pytest.raises(AssertionError):
        driver.source_safe(spec)


def test_mask_requires_an_ordinary_two_dimensional_shape(prospect):
    driver, contract = prospect
    spec = copy.deepcopy(next(s for s in contract['cases'] if s['api'] == 'draw.polygon2mask'))
    spec['shape'] = None
    with pytest.raises(AssertionError):
        driver.source_safe(spec)


def test_frozen_reference_sampler_and_public_native_witnesses(prospect):
    driver, contract = prospect
    digest = lambda body: hashlib.sha256(body).hexdigest()
    assert digest(driver.REFERENCE.read_bytes()) == contract['reference_contract_sha256']
    for name, pin in contract['source_closure'].items():
        assert digest((ROOT / name).read_bytes().replace(b'\r\n', b'\n')) == pin
    ellipse = json.loads((ROOT / 'coverage/contracts/ellipse-perimeter-benchmark.json').read_bytes())
    assert contract['method_sha256'] == ellipse['method_sha256']
    assert contract['native_entries'] == {
        'draw.polygon': ['draw_polygon_prepare', 'draw_polygon_coords_into'],
        'draw.polygon2mask': ['draw_polygon_prepare', 'draw_polygon_mask_into'],
    }
    assert contract['execution_before_separate_root_GO'] == {
        'original_calls': 0, 'native_calls': 0, 'timed_batches': 0, 'builds': 0,
    }


def test_current_polygon_context_preserves_historical_cases_and_driver(prospect):
    driver, current = prospect
    historical = ROOT / 'coverage/contracts/draw-polygon-runtime-v1.json'
    old = json.loads(historical.read_bytes())
    assert hashlib.sha256(historical.read_bytes()).hexdigest() == '6a40a1a7c82d534229f815296358c1e5b60fd5f5bf255b05c3d02ff7a426a636'
    assert {k:v for k,v in current.items() if k not in ('source_closure','scope')} == {k:v for k,v in old.items() if k not in ('source_closure','scope')}
    source = (ROOT / 'benchmarks/draw_polygon.py').read_text()
    restored = source.replace("CONTRACT = ROOT / 'coverage/contracts/draw-polygon-draw-batch-runtime-v1.json'", "CONTRACT = ROOT / 'coverage/contracts/draw-polygon-runtime-v1.json'", 1)
    assert hashlib.sha256(restored.encode()).hexdigest() == '009f684108559dbcc27286cd10a5b7227a59e463fb3e08af0271dca0781404fe'
