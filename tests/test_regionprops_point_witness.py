"""Saved single-pixel storage contract; original capture is never a test."""
import json
import os
from pathlib import Path
import sys
import pytest
if __package__:
    from .regionprops_point_contract import cases
    from .regionprops_state_contract import fixture
    from .regionprops_state_observer import observe
else:
    from regionprops_point_contract import cases
    from regionprops_state_contract import fixture
    from regionprops_state_observer import observe

ROOT = Path(__file__).parents[1]
MATRIX = ROOT / 'coverage/contracts/regionprops-point-witness-v1.json'
CASES = cases()


def test_regionprops_point_frozen_contract():
    matrix = json.loads(MATRIX.read_bytes())
    assert matrix['cases'] == CASES and len(CASES) == 4
    assert len({spec['id'] for spec in CASES}) == 4
    assert matrix['native_API_credit'] == 0


@pytest.mark.parametrize('spec', CASES, ids=[s['id'] for s in CASES])
def test_regionprops_point_actual_controller_intake_originals_refused(spec):
    import numpy as np
    image, options = fixture(spec, np)
    assert image.shape == tuple(spec['shape']) and int(image.sum()) == 1
    assert options['cache'] is spec['cache']
    calls = []

    def refused(actual_image, **actual_options):
        calls.append((actual_image.shape, actual_options))
        raise RuntimeError('point-intake-original-refused')

    row = observe(refused, spec, np)
    assert len(calls) == 1 and row['id'] == spec['id']
    assert row['error'] == ['builtins', 'RuntimeError', 'point-intake-original-refused']
    assert row['regions'] == [] and all(row['checks'].values())


@pytest.mark.parametrize('spec', CASES, ids=[s['id'] for s in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_regionprops_point_saved_candidate_exact(spec, workers, monkeypatch):
    import numpy as np
    import skimagexx as xx
    from skimagexx import _regionprops_state as owned
    matrix = json.loads(MATRIX.read_bytes())
    expected = next(r for r in matrix['original_observations'][np.__version__]
                    if r['id'] == spec['id'])
    policy = matrix['type_policy']
    assert owned.RegionProperties is not owned._Provider
    assert issubclass(owned.RegionProperties, owned._Provider)
    entries = {getattr(owned._core, 'regionprops_state_' + name): name
               for name in ('prepare', 'image', 'population', 'coords')}
    returns = []

    def profiler(frame, event, value):
        if event == 'c_return' and value in entries:
            returns.append(entries[value])

    def denied(*args, **kwargs):
        raise AssertionError('point candidate invoked original algorithm')

    old_workers = xx.get_num_threads()
    old_profiler = sys.getprofile()
    xx.set_num_threads(workers)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owned, '_provider_regionprops', denied)
            for name in ('image', 'num_pixels', 'coords'):
                patch.setattr(owned._Provider, name, property(denied))
            sys.setprofile(profiler)
            try:
                actual = observe(owned.regionprops, spec, np)
            finally:
                sys.setprofile(old_profiler)
        folder = Path(os.environ.get('SKIMAGEXX_POINT_ROWS_DIR', ROOT / 'build' / 'regionprops-point-rows'))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (spec['id'].split('/')[-1] + '-workers' + str(workers) + '.json')).write_text(
            json.dumps(actual, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        assert set(actual) == set(expected)
        for key in expected:
            if key != 'region_types':
                assert actual[key] == expected[key], (spec['id'], key)
        assert expected['region_types'] == [policy['original_qualified_type']]
        assert actual['region_types'] == [policy['backend_qualified_type']]
        assert returns.count('prepare') == 1
        assert set(returns) == {'prepare', 'image', 'population', 'coords'}
    finally:
        sys.setprofile(old_profiler)
        xx.set_num_threads(old_workers)
