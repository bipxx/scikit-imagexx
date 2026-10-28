"""Private C admission and exact frozen public observations for label_points."""
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from skimagexx import _label_points as implementation
if __package__:
    from .label_points_contract import cases
    from .test_label_points_public import observe
else:
    from label_points_contract import cases
    from test_label_points_public import observe

CASES = cases()
MATRIX = Path(__file__).parents[1] / 'coverage/contracts/label-points.json'
STAGE = 'label_points_apply'


def test_label_points_private_signature():
    assert str(inspect.signature(implementation.label_points)) == json.loads(MATRIX.read_bytes())['signature']
    assert inspect.isbuiltin(getattr(implementation._core, STAGE))


@pytest.mark.parametrize('spec', CASES, ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_label_points_pipeline(spec, workers, monkeypatch, tmp_path):
    import skimage.util as public
    import skimage.util._label as original
    expected = next(row for row in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__] if row['id'] == spec['id'])
    core = implementation._core
    get_workers = core.get_num_threads
    previous = get_workers()
    old_profile = sys.getprofile()
    calls, returns, accepted, snapshots, forbidden = [], [], [], [], []
    provider_order = []
    zeros = np.zeros
    def zero_provider(*args, **kwargs):
        provider_order.append("zeros")
        return zeros(*args, **kwargs)
    native_expected = expected['error'] is None and spec['id'] not in (
        'rank0-onepoint', 'shape-numpy-integers', 'protocol-order')
    def reject(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError('original label_points forwarding or native output provider')
    def snapshot():
        provider_order.append("workers")
        snapshots.append(get_workers())
        return snapshots[-1]
    def profile(frame, event, value):
        if getattr(value, '__module__', None) == core.__name__ and getattr(value, '__name__', None) == STAGE:
            if event == 'c_call': calls.append(STAGE)
            elif event == 'c_return': returns.append(STAGE)
        if event == 'return' and frame.f_code is implementation.label_points.__code__:
            accepted.append(frame.f_locals.get('native_result') is not None)
    try:
        core.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(public, 'label_points', reject)
            patch.setattr(original, 'label_points', reject)
            patch.setattr(core, 'get_num_threads', snapshot)
            patch.setattr(np, 'zeros', zero_provider)
            if native_expected:
                patch.setattr(np, 'arange', reject)
            sys.setprofile(profile)
            try: actual = observe(implementation.label_points, spec, np)
            finally: sys.setprofile(old_profile)
        assert actual == expected and not forbidden
        assert accepted == [native_expected]
        if native_expected:
            assert calls == returns == [STAGE] and snapshots == [workers]
            assert provider_order == ["zeros", "workers"]
        else:
            assert len(calls) <= 1 and calls == returns
            assert snapshots == ([workers] if calls else [])
        assert get_workers() == workers
        (tmp_path / 'candidate.json').write_text(json.dumps(dict(id=spec['id'], workers=workers,
            observation=actual, C_calls=calls, C_returns=returns, native_accepted=accepted,
            worker_snapshots=snapshots)) + '\n')
    finally:
        sys.setprofile(old_profile)
        core.set_num_threads(previous)


@pytest.mark.parametrize('workers', (0, 1025))
def test_label_points_worker_boundary(workers):
    with pytest.raises(ValueError, match='worker snapshot'):
        implementation._core.label_points_apply((np.array([0], dtype=np.int64),), (1,), workers)


def test_label_points_object_callback_keeps_late_point_count(monkeypatch):
    # The source evaluates coords.shape[0] after round/cast callbacks. These
    # inputs stay on that provider tail, including its broadcasting error.
    coords = np.empty((3, 2), dtype=object)
    class Coordinate:
        def __init__(self, value): self.value = value
        def __lt__(self, other): return self.value < other
        def rint(self):
            coords.shape = (2, 3)
            return float(self.value)
    for index, value in enumerate((0, 1, 2, 3, 1, 2)):
        coords.flat[index] = Coordinate(value)
    def reject(*args, **kwargs):
        raise AssertionError('object callback input entered native label generation')
    with monkeypatch.context() as patch:
        patch.setattr(implementation._core, STAGE, reject)
        patch.setattr(implementation._core, 'get_num_threads', reject)
        with pytest.raises(ValueError, match='shape mismatch'):
            implementation.label_points(coords, (4, 5))
    assert coords.shape == (2, 3)


@pytest.mark.parametrize('workers', (1, 4))
def test_label_points_output_overload_assigns_without_clearing(workers):
    output = np.full((2, 3), 99, dtype=np.uint64)
    indices = (np.array([0, 1, 0], np.int64), np.array([1, 2, 1], np.int64))
    result = implementation._core.label_points_apply(indices, output, workers)
    assert result is output
    assert output.tolist() == [[99, 3, 99], [99, 99, 2]]


@pytest.mark.parametrize('kind', ('readonly', 'strided', 'wrong-dtype', 'big-endian'))
def test_label_points_output_overload_refuses_unsupported_storage(kind):
    output = np.zeros((2, 3), dtype=np.uint64)
    if kind == 'readonly': output.flags.writeable = False
    elif kind == 'strided': output = output[:, ::2]
    elif kind == 'wrong-dtype': output = output.astype(np.int64)
    elif kind == 'big-endian': output = output.astype('>u8')
    before = output.tobytes()
    assert implementation._core.label_points_apply((np.array([0], np.int64), np.array([1], np.int64)), output, 1) is None
    assert output.tobytes() == before
