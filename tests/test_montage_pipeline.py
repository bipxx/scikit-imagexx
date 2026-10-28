"""Prepared native fill/copy against once-only upstream observations."""
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from skimagexx import _montage as implementation
if __package__:
    from .montage_contract import cases
    from .test_montage_public import observe
else:
    from montage_contract import cases
    from test_montage_public import observe

CASES = cases()
MATRIX = Path(__file__).parents[1] / 'coverage/contracts/montage.json'
NATIVE_IDS = frozenset((
    'gray-square', 'gray-incomplete', 'gray-one', 'gray-strided', 'gray-fortran',
    'gray-readonly', 'grid-row', 'grid-column', 'grid-extra', 'grid-fractional',
    'padding-one', 'padding-numpy-int', 'color-last-mean', 'color-middle-mean',
    'color-first-mean', 'channel-axis-zero', 'channel-tuple-one',
    'fill-color-vector', 'fill-nan', 'fill-inf', 'rescale-float', 'rescale-uint8',
    'dtype-bool', 'dtype-int8', 'dtype-uint16', 'dtype-int64', 'dtype-float16',
    'dtype-complex64', 'big-endian', 'dtype-metadata', 'array-protocol',
    'grid-protocol', 'fill-protocol', 'zero-height'))


def test_montage_private_signature():
    assert str(inspect.signature(implementation.montage)) == json.loads(MATRIX.read_bytes())['signature']
    assert inspect.isbuiltin(implementation._core.montage_apply)


@pytest.mark.parametrize('spec', CASES, ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_montage_pipeline(spec, workers, monkeypatch, tmp_path):
    import skimage.util as public
    import skimage.util._montage as original
    expected = next(row for row in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__]
                    if row['id'] == spec['id'])
    core = implementation._core
    native = core.montage_apply
    get_workers = core.get_num_threads
    previous, old_profile = get_workers(), sys.getprofile()
    calls, returns, accepted, snapshots, forbidden = [], [], [], [], []
    native_expected = spec['id'] in NATIVE_IDS or (spec['id'] == 'fill-array' and np.__version__ == '1.24.4')
    def reject(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError('montage forwarded to original')
    def snapshot():
        snapshots.append(get_workers())
        return snapshots[-1]
    def profile(frame, event, value):
        if value is native:
            if event == 'c_call': calls.append('montage_apply')
            elif event == 'c_return': returns.append('montage_apply')
        if event == 'return' and frame.f_code is implementation.montage.__wrapped__.__code__:
            accepted.append(bool(frame.f_locals.get('native_result', False)))
    try:
        core.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(public, 'montage', reject)
            patch.setattr(original, 'montage', reject)
            patch.setattr(core, 'get_num_threads', snapshot)
            sys.setprofile(profile)
            try: actual = observe(implementation.montage, spec, np)
            finally: sys.setprofile(old_profile)
        assert actual == expected and not forbidden
        assert accepted == ([] if spec['id'] in ('channel-tuple-many', 'channel-out-of-range')
                            else [native_expected])
        assert calls == returns and len(calls) <= 1
        assert snapshots == [workers] * len(calls)
        if native_expected: assert calls == ['montage_apply']
        assert get_workers() == workers
        (tmp_path / 'candidate.json').write_text(json.dumps(dict(
            id=spec['id'], workers=workers, observation=actual, C_calls=calls,
            C_returns=returns, native_accepted=accepted, worker_snapshots=snapshots)) + '\n')
    finally:
        sys.setprofile(old_profile)
        core.set_num_threads(previous)


@pytest.mark.parametrize('workers', (0, 1025))
def test_montage_worker_boundary(workers):
    with pytest.raises(ValueError, match='worker snapshot'):
        implementation._core.montage_apply(np.zeros((1, 1, 1, 1)), np.zeros(1),
                                           np.zeros((1, 1, 1)), (1, 1, 0), workers)


@pytest.mark.parametrize('reason', ('short-fill', 'small-grid', 'object'))
def test_montage_refuses_before_writing(reason):
    dtype = object if reason == 'object' else np.int16
    source = np.arange(24).reshape(4, 2, 3, 1).astype(dtype)
    fill = np.empty(0, dtype) if reason == 'short-fill' else np.zeros(1, dtype)
    grid = (1, 2, 0) if reason == 'small-grid' else (2, 2, 0)
    output = np.full((2 * grid[0], 3 * grid[1], 1), 91, dtype)
    assert implementation._core.montage_apply(source, fill, output, grid, 1) is False
    assert np.all(output == 91)
