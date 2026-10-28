"""Actual native geometry and complete once-only original observations."""
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from skimagexx import _rolling_ball_kernels as implementation
if __package__:
    from .rolling_ball_kernels_contract import APIS, cases
    from .test_rolling_ball_kernels_public import observe
else:
    from rolling_ball_kernels_contract import APIS, cases
    from test_rolling_ball_kernels_public import observe

CASES = cases()
MATRIX = Path(__file__).parents[1] / 'coverage/contracts/rolling-ball-kernels.json'
LITERAL_IDS = frozenset(('ball-protocol-order', 'ellipsoid-intensity-protocol',
                         'ellipsoid-intensity-nan', 'ellipsoid-intensity-inf'))


@pytest.mark.parametrize('name', ('ball_kernel', 'ellipsoid_kernel'))
def test_geometry_private_signature(name):
    matrix = json.loads(MATRIX.read_bytes())
    assert str(inspect.signature(getattr(implementation, name))) == matrix['signatures']['restoration.' + name]
    assert inspect.isbuiltin(getattr(implementation._core, name + '_apply'))


@pytest.mark.parametrize('spec', CASES, ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_geometry_pipeline(spec, workers, monkeypatch, tmp_path):
    import skimage.restoration as public
    import skimage.restoration._rolling_ball as original
    expected = next(row for row in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__]
                    if row['id'] == spec['id'])
    name = spec['api'].split('.')[-1]
    call = getattr(implementation, name)
    core = implementation._core
    native = getattr(core, name + '_apply')
    get_workers = core.get_num_threads
    previous, old_profile = get_workers(), sys.getprofile()
    calls, returns, accepted, snapshots, forbidden = [], [], [], [], []
    native_expected = expected['error'] is None and spec['id'] not in LITERAL_IDS
    def reject(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError('rolling geometry forwarded to original whole API')
    def snapshot():
        snapshots.append(get_workers())
        return snapshots[-1]
    def profile(frame, event, value):
        if value is native:
            if event == 'c_call': calls.append(name + '_apply')
            elif event == 'c_return': returns.append(name + '_apply')
        if event == 'return' and frame.f_code is call.__code__:
            accepted.append(bool(frame.f_locals.get('native_result', False)))
    try:
        core.set_num_threads(workers)
        with monkeypatch.context() as patch:
            for target in APIS:
                patch.setattr(public, target.split('.')[-1], reject)
                patch.setattr(original, target.split('.')[-1], reject)
            patch.setattr(core, 'get_num_threads', snapshot)
            sys.setprofile(profile)
            try: actual = observe(call, spec, np)
            finally: sys.setprofile(old_profile)
        assert actual == expected and not forbidden
        assert accepted == [native_expected]
        assert calls == returns and len(calls) <= 1
        assert snapshots == [workers] * len(calls)
        if native_expected: assert calls == [name + '_apply']
        assert get_workers() == workers
        (tmp_path / 'candidate.json').write_text(json.dumps(dict(
            id=spec['id'], workers=workers, observation=actual, C_calls=calls,
            C_returns=returns, native_accepted=accepted, worker_snapshots=snapshots)) + '\n')
    finally:
        sys.setprofile(old_profile)
        core.set_num_threads(previous)


@pytest.mark.parametrize('name', ('ball_kernel', 'ellipsoid_kernel'))
@pytest.mark.parametrize('workers', (0, 1025))
def test_geometry_worker_boundary(name, workers):
    core = implementation._core
    axes = (np.array([-1., 0., 1.]),)
    output = np.empty(3)
    with pytest.raises(ValueError, match='worker snapshot'):
        if name == 'ball_kernel': core.ball_kernel_apply(axes, 1., 1., output, workers)
        else: core.ellipsoid_kernel_apply(axes, np.ones(1), 1., output, workers)


@pytest.mark.parametrize('name', ('ball_kernel', 'ellipsoid_kernel'))
@pytest.mark.parametrize('reason', ('alias', 'shape', 'dtype'))
def test_geometry_refuses_before_writing(name, reason):
    values = np.array([-1., 0., 1.], dtype=np.float32 if reason == 'dtype' else np.float64)
    output = values if reason == 'alias' else np.full(2 if reason == 'shape' else 3, 91.)
    before = output.tobytes()
    core = implementation._core
    if name == 'ball_kernel': accepted = core.ball_kernel_apply((values,), 1., 1., output, 1)
    else: accepted = core.ellipsoid_kernel_apply((values,), np.ones(1), 1., output, 1)
    assert accepted is False and output.tobytes() == before
