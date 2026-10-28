"""Actual RDP ownership and complete frozen original observations."""
import ast
import inspect
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest
from skimagexx import _approximate_polygon as implementation
if __package__:
    from .approximate_polygon_contract import cases
    from .test_approximate_polygon_public import observe
    from .approximate_polygon_runtime_reference import RUNTIME_REFERENCE_IDS, runtime_reference
else:
    from approximate_polygon_contract import cases
    from test_approximate_polygon_public import observe
    from approximate_polygon_runtime_reference import RUNTIME_REFERENCE_IDS, runtime_reference

CASES = cases()
MATRIX = Path(__file__).parents[1] / 'coverage/contracts/approximate-polygon.json'
NATIVE_IDS = frozenset((
    'open-zigzag', 'closed-square', 'closed-square-coarse', 'collinear',
    'outside-segment', 'repeated-endpoints', 'all-identical', 'duplicate-interior',
    'singleton', 'two-points', 'threshold-below', 'threshold-equal',
    'threshold-above', 'first-max-tie', 'fortran', 'reverse-rows',
    'reverse-columns', 'step-rows', 'step-columns', 'readonly', 'signed-zero',
    'dtype-metadata', 'float32-tolerance', 'moderate-257'))


def test_polygon_private_signature_and_literal_body():
    assert str(inspect.signature(implementation.approximate_polygon)) == '(coords, tolerance)'
    assert inspect.isbuiltin(implementation._core.approximate_polygon_apply)
    # The literal fallback keeps the frozen source arithmetic/control statements.
    source = ast.parse(inspect.getsource(implementation.approximate_polygon))
    call = source.body[0]
    first = call.body[0]
    assert isinstance(first, ast.If) and ast.unparse(first.test) == 'tolerance <= 0'
    assert ast.unparse(first.body[0]) == 'return coords'
    import skimage.measure._polygon as original
    frozen = ast.parse(inspect.getsource(original.approximate_polygon)).body[0]
    start = next(i for i, node in enumerate(call.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'chain')
    assert ast.dump(ast.Module(body=call.body[start:], type_ignores=[])) == ast.dump(
        ast.Module(body=frozen.body[2:], type_ignores=[]))



@pytest.mark.parametrize('spec', CASES, ids=[spec['id'] for spec in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_polygon_pipeline(spec, workers, monkeypatch, tmp_path):
    import skimage.measure as public
    import skimage.measure._polygon as original
    expected = next(row for row in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__]
                    if row['id'] == spec['id'])
    if spec['id'] in RUNTIME_REFERENCE_IDS:
        expected = runtime_reference(spec['id'], np)
    core = implementation._core
    native, call = core.approximate_polygon_apply, implementation.approximate_polygon
    get_workers = core.get_num_threads
    previous, old_profile = get_workers(), sys.getprofile()
    calls, returns, accepted, snapshots, forbidden = [], [], [], [], []
    def reject(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError('RDP forwarded to original whole API')
    def snapshot():
        snapshots.append(get_workers())
        return snapshots[-1]
    def profile(frame, event, value):
        if value is native:
            if event == 'c_call': calls.append('approximate_polygon_apply')
            elif event == 'c_return': returns.append('approximate_polygon_apply')
        if event == 'call' and frame.f_code is original_code: reject()
        if event == 'return' and frame.f_code is call.__code__:
            accepted.append(bool(frame.f_locals.get('native_result', False)))
    original_code = original.approximate_polygon.__code__
    try:
        core.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(public, 'approximate_polygon', reject)
            patch.setattr(original, 'approximate_polygon', reject)
            patch.setattr(core, 'get_num_threads', snapshot)
            sys.setprofile(profile)
            try: actual = observe(call, spec, np)
            finally: sys.setprofile(old_profile)
        assert actual == expected and not forbidden
        assert accepted == [spec['id'] in NATIVE_IDS]
        assert calls == returns and len(calls) <= 1
        assert snapshots == [workers] * len(calls)
        if spec['id'] in NATIVE_IDS: assert calls == ['approximate_polygon_apply']
        assert get_workers() == workers
        (tmp_path / 'candidate.json').write_text(json.dumps(dict(
            id=spec['id'], workers=workers, observation=actual, C_calls=calls,
            C_returns=returns, native_accepted=accepted, worker_snapshots=snapshots)) + '\n')
    finally:
        sys.setprofile(old_profile)
        core.set_num_threads(previous)


@pytest.mark.parametrize('workers', (0, 1025))
def test_polygon_worker_boundary(workers):
    with pytest.raises(ValueError, match='worker snapshot'):
        implementation._core.approximate_polygon_apply(
            np.ones((3, 2)), 1., np.empty(3, dtype=bool), np.empty(3), workers,
            implementation._scalar_trig)


@pytest.mark.parametrize('reason', ('shape', 'dtype', 'mask-readonly', 'scratch-readonly', 'alias'))
def test_polygon_storage_refuses_before_writing(reason):
    coords = np.arange(6, dtype=np.float32 if reason == 'dtype' else np.float64).reshape(3, 2)
    keep = np.full(2 if reason == 'shape' else 3, True)
    scratch = np.full(3, 91.)
    if reason == 'mask-readonly': keep.flags.writeable = False
    elif reason == 'scratch-readonly': scratch.flags.writeable = False
    elif reason == 'alias': scratch = coords[:, 0]
    before = (keep.tobytes(), scratch.tobytes(), coords.tobytes())
    assert implementation._core.approximate_polygon_apply(
        coords, 1., keep, scratch, 1, implementation._scalar_trig) is False
    assert (keep.tobytes(), scratch.tobytes(), coords.tobytes()) == before


def test_polygon_forged_storage_is_rejected_before_reading():
    coords = np.lib.stride_tricks.as_strided(np.ones(2), shape=(3, 2), strides=(16, 8))
    with pytest.raises(ValueError, match='FFT'):
        implementation._core.approximate_polygon_apply(
            coords, 1., np.empty(3, dtype=bool), np.empty(3), 1, implementation._scalar_trig)


@pytest.mark.parametrize('workers', (1, 4))
def test_polygon_trig_stays_on_caller_and_exceptions_restore_state(workers):
    core = implementation._core
    caller = threading.get_ident()
    seen = []
    def fail(dr, dc):
        seen.append(threading.get_ident())
        raise LookupError('polygon trig witness')
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    with pytest.raises(LookupError, match='polygon trig witness'):
        core.approximate_polygon_apply(np.array([[0., 0.], [1., 1.], [0., 2.]]),
                                      .5, np.empty(3, dtype=bool), np.empty(3), workers, fail)
    assert seen == [caller]
    assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer


def test_polygon_has_no_fixture_point_count_cap():
    coords = np.column_stack((np.arange(513, dtype=np.float64), np.zeros(513)))
    core = implementation._core
    keep, scratch = np.empty(513, dtype=bool), np.empty(513)
    assert core.approximate_polygon_apply(coords, .5, keep, scratch, 1, implementation._scalar_trig) is True
    assert np.flatnonzero(keep).tolist() == [0, 512]
