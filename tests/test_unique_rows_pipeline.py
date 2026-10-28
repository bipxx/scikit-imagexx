"""Actual native byte-sort admission against immutable full public records."""
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from skimagexx import _unique_rows as implementation
if __package__:
    from .unique_rows_contract import cases
    from .test_unique_rows_public import observe
else:
    from unique_rows_contract import cases
    from test_unique_rows_public import observe

CASES = cases()
MATRIX = Path(__file__).parents[1] / 'coverage/contracts/unique-rows.json'
STAGE = 'unique_rows_apply'


def test_unique_rows_private_signature():
    assert str(inspect.signature(implementation.unique_rows)) == json.loads(MATRIX.read_bytes())['signature']
    assert inspect.isbuiltin(getattr(implementation._core, STAGE))


@pytest.mark.parametrize('spec', CASES, ids=[row['id'] for row in CASES])
@pytest.mark.parametrize('workers', (1, 4))
def test_unique_rows_pipeline(spec, workers, monkeypatch, tmp_path):
    import skimage.util as public
    import skimage.util.unique as original
    expected = next(row for row in json.loads(MATRIX.read_bytes())['original_observations'][np.__version__] if row['id'] == spec['id'])
    core = implementation._core
    get_workers = core.get_num_threads
    previous, old_profile = get_workers(), sys.getprofile()
    calls, returns, accepted, snapshots, forbidden = [], [], [], [], []
    native_expected = expected['error'] is None
    def reject(*args, **kwargs):
        forbidden.append(True)
        raise AssertionError('native unique_rows forwarded to original API or NumPy unique')
    def snapshot():
        snapshots.append(get_workers())
        return snapshots[-1]
    def profile(frame, event, value):
        if getattr(value, '__module__', None) == core.__name__ and getattr(value, '__name__', None) == STAGE:
            if event == 'c_call': calls.append(STAGE)
            elif event == 'c_return': returns.append(STAGE)
        if event == 'return' and frame.f_code is implementation.unique_rows.__code__:
            accepted.append(frame.f_locals.get('native_result') is not None)
    try:
        core.set_num_threads(workers)
        with monkeypatch.context() as patch:
            patch.setattr(public, 'unique_rows', reject)
            patch.setattr(original, 'unique_rows', reject)
            patch.setattr(np, 'unique', reject)
            patch.setattr(core, 'get_num_threads', snapshot)
            sys.setprofile(profile)
            try: actual = observe(implementation.unique_rows, spec, np)
            finally: sys.setprofile(old_profile)
        assert actual == expected and not forbidden
        assert accepted == [native_expected]
        if native_expected:
            assert calls == returns == [STAGE] and snapshots == [workers]
        else:
            assert calls == returns == snapshots == []
        assert get_workers() == workers
        (tmp_path / 'candidate.json').write_text(json.dumps(dict(id=spec['id'], workers=workers,
            observation=actual, C_calls=calls, C_returns=returns, native_accepted=accepted,
            worker_snapshots=snapshots)) + '\n')
    finally:
        sys.setprofile(old_profile)
        core.set_num_threads(previous)


@pytest.mark.parametrize('workers', (0, 1025))
def test_unique_rows_worker_boundary(workers):
    with pytest.raises(ValueError, match='worker snapshot'):
        implementation._core.unique_rows_apply(np.array([[0]], dtype=np.int64), workers)
