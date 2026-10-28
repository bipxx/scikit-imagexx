"""Actual native grid-label admission against the frozen original records."""
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from skimagexx import _regular_seeds as implementation
if __package__:
    from .regular_seeds_contract import cases
    from .test_regular_seeds_public import observe
else:
    from regular_seeds_contract import cases
    from test_regular_seeds_public import observe

CASES = cases()
MATRIX = Path(__file__).parents[1] / 'coverage/contracts/regular-seeds.json'
NATIVE_IDS = frozenset((
    'ordinary2d', 'ordinary1d', 'ordinary3d', 'ordinary4d', 'thin-axis',
    'reordered-axes', 'sparse-large', 'one-seed', 'full-grid', 'excess-points',
    'approximate-count', 'shape-list', 'shape-array', 'empty-axis', 'scalar-shape',
    'numpy-count', 'fractional-count', 'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64', 'bool', 'float32', 'float64',
    'signed-wrap', 'unsigned-wrap', 'dtype-metadata'))


def test_regular_seeds_private_signature():
    assert str(inspect.signature(implementation.regular_seeds)) == json.loads(MATRIX.read_bytes())['signature']
    assert inspect.isbuiltin(implementation._core.regular_seeds_apply)


@pytest.mark.parametrize('spec', CASES, ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_regular_seeds_pipeline(spec, workers, monkeypatch, tmp_path):
    import skimage.util as public
    import skimage.util._regular_grid as original
    expected = next(row for row in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__]
                    if row['id'] == spec['id'])
    core = implementation._core
    native = core.regular_seeds_apply
    get_workers = core.get_num_threads
    previous, old_profile = get_workers(), sys.getprofile()
    calls, returns, accepted, snapshots, forbidden = [], [], [], [], []
    native_expected = spec['id'] in NATIVE_IDS
    def reject(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError('regular_seeds forwarded to original or native labels used np.arange')
    def snapshot():
        snapshots.append(get_workers())
        return snapshots[-1]
    def profile(frame, event, value):
        if value is native:
            if event == 'c_call': calls.append('regular_seeds_apply')
            elif event == 'c_return': returns.append('regular_seeds_apply')
        if event == 'return' and frame.f_code is implementation.regular_seeds.__code__:
            accepted.append(bool(frame.f_locals.get('native_result', False)))
    try:
        core.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(public, 'regular_seeds', reject)
            patch.setattr(original, 'regular_seeds', reject)
            if native_expected: patch.setattr(np, 'arange', reject)
            patch.setattr(core, 'get_num_threads', snapshot)
            sys.setprofile(profile)
            try: actual = observe(implementation.regular_seeds, spec, np)
            finally: sys.setprofile(old_profile)
        assert actual == expected and not forbidden
        assert accepted == [native_expected]
        if native_expected:
            assert calls == returns == ['regular_seeds_apply'] and snapshots == [workers]
        assert get_workers() == workers
        (tmp_path / 'candidate.json').write_text(json.dumps(dict(
            id=spec['id'], workers=workers, observation=actual, C_calls=calls,
            C_returns=returns, native_accepted=accepted, worker_snapshots=snapshots)) + '\n')
    finally:
        sys.setprofile(old_profile)
        core.set_num_threads(previous)


@pytest.mark.parametrize('workers', (0, 1025))
def test_regular_seeds_worker_boundary(workers):
    with pytest.raises(ValueError, match='worker snapshot'):
        implementation._core.regular_seeds_apply(np.zeros((1,), np.int64),
                                                (slice(None),), workers)
