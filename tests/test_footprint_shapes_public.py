"""Public footprint parity on the 226 frozen cases, not a performance gate.

The public namespace must reach the shared compiled stages without
forwarding footprint, hull, or dilation algorithms. NumPy allocation, dtype,
mgrid and reduction providers remain explicit; this is not full C++ coverage.
Running against a component harness is not installed whole-product qualification.
"""
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGES = ('footprint_shapes_ellipse', 'footprint_shapes_hull', 'footprint_shapes_l1',
          'footprint_shapes_atoms', 'footprint_shapes_cross_plan', 'footprint_shapes_cross',
          'footprint_shapes_dilate', 'footprint_shapes_full_mask')


def _load_reference_helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FACTORY = _load_reference_helper('footprint_shapes_cases')
REFERENCE = _load_reference_helper('footprint_shapes_reference')
CASES = FACTORY.cases()
WORKER_FOUR_IDS = frozenset({
    'ellipse/1-1-None', 'octagon/1-1-None', 'octahedron/1-None',
    'star/1', 'footprint_from_sequence/cross-C',
})


@pytest.fixture(scope='module')
def runtime():
    import numpy as np
    import skimage
    import skimage.morphology as original
    import skimage.morphology.convex_hull as hull
    import skimage.morphology.gray as gray
    import skimagexx as xx
    from skimagexx import _footprint_shapes as owned
    from skimagexx.coverage import native_functions

    assert skimage.__version__ == '0.26.0'
    assert owned._core is xx._core
    assert xx.get_num_threads is xx._core.get_num_threads
    assert xx.set_num_threads is xx._core.set_num_threads
    assert xx._core.footprint_shapes_interface == 'footprint-shapes-v1'
    assert xx.get_num_threads() == 1
    try:
        xx.set_num_threads(4)
    except ValueError as error:
        assert str(error) == 'this build has no OpenMP support'
        assert xx.get_num_threads() == 1
        can_parallel = False
    else:
        assert xx.get_num_threads() == 4
        can_parallel = True
    finally:
        xx.set_num_threads(1)
    for name in STAGES:
        function = getattr(xx._core, name)
        assert inspect.isbuiltin(function) and function.__module__ == xx._core.__name__
    data = json.loads((ROOT / 'coverage/contracts/footprint-shapes.json').read_bytes())
    assert data['capture_source'] == 'b3c61fcc565c15860465eb374b395cd143d84414'
    assert data['case_specs'] == CASES and len(CASES) == 226
    profiles = {('win32', '2.5.4'): 'windows', ('linux', '2.5.4'): 'modern',
                ('linux', '1.24.4'): 'minimum'}
    assert (sys.platform, np.__version__) in profiles, 'no frozen oracle for this runtime'
    profile = profiles[sys.platform, np.__version__]
    observations = data['profiles'][profile]['observations']
    assert [row['id'] for row in observations] == [row['id'] for row in CASES]
    assert len({row['id'] for row in CASES}) == 226
    assert len([row for row in CASES if row['id'] in WORKER_FOUR_IDS]) == 5
    return SimpleNamespace(np=np, xx=xx, core=xx._core, owned=owned,
                           original=original, hull=hull, gray=gray,
                           registered=set(native_functions()), can_parallel=can_parallel,
                           expected={row['id']: row for row in observations})


def _observe_public(spec, runtime, returns):
    """Reuse the frozen observer's snapshots/counters without its private import gate."""
    np, backend = runtime.np, runtime.xx.morphology
    function = getattr(backend, spec['api'])
    args, kwargs = FACTORY.fixture(spec, np)
    before = REFERENCE.snapshot((args, kwargs), np)
    state = (dict(np.geterr()), np.geterrcall(), np.getbufsize())
    counts, entry_profile = REFERENCE.entry_profiler(function.__code__)
    assert sys.getprofile() is None

    def profile(frame, event, value):
        entry_profile(frame, event, value)
        if (event == 'c_return' and getattr(value, '__name__', None) in STAGES
                and getattr(value, '__module__', None) == runtime.core.__name__):
            returns.append(value.__name__)

    result, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sys.setprofile(profile)
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            # Errors are frozen public outcomes and are retained before comparison.
            error = [type(exc).__module__, type(exc).__name__, str(exc)]
        finally:
            sys.setprofile(None)
    returned = REFERENCE.arrays(result, np)
    inputs = REFERENCE.arrays((args, kwargs), np)
    return dict(id=spec['id'], phase='actual-public', before=before,
        after=REFERENCE.snapshot((args, kwargs), np), result=REFERENCE.snapshot(result, np),
        error=error, entries=counts,
        warnings=[[x.category.__module__, x.category.__name__, str(x.message)] for x in caught],
        aliases=[[r is a, bool(np.shares_memory(r, a))] for r in returned for a in inputs],
        output_aliases=[[r is a, bool(np.shares_memory(r, a))] for r in returned for a in returned],
        state_before=dict(policy=state[0], handler_none=state[1] is None, buffer=state[2]),
        restored=(state[0] == np.geterr() and state[1] is np.geterrcall() and state[2] == np.getbufsize()),
        entry_identity_preserved=getattr(backend, spec['api']) is function)


def _check_public_case(spec, runtime, workers, monkeypatch, tmp_path):
    xx = runtime.xx
    assert xx.get_num_threads() == 1
    previous_fallback = xx.set_fallback_enabled(False)
    returns, forwards = [], []

    def denied(*args, **kwargs):
        forwards.append(True)
        raise AssertionError('upstream footprint/hull/dilation algorithm forwarding')

    targets = [(runtime.original, name) for name in FACTORY.APIS]
    targets += [(runtime.hull, 'convex_hull_image'), (runtime.gray, 'dilation')]
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            for module, name in targets:
                patch.setattr(module, name, denied)
            actual = _observe_public(spec, runtime, returns)
        # Keep the complete actual observation and native witness before assertions.
        (tmp_path / 'actual-public.json').write_text(json.dumps(dict(
            workers=workers, observation=actual, compiled_returns=returns,
            forward_attempts=len(forwards)), sort_keys=True) + '\n', encoding='utf-8')
        assert not forwards
        assert xx.get_num_threads() == workers
        expected = dict(runtime.expected[spec['id']], phase='actual-public')
        assert actual == expected
        if actual['error'] is None:
            assert returns, 'successful public call without a real compiled stage return'
    finally:
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous_fallback)


@pytest.mark.parametrize('name', FACTORY.APIS)
def test_public_footprint_registration_and_signature(runtime, name):
    public = getattr(runtime.xx.morphology, name)
    original = getattr(runtime.original, name)
    assert 'morphology.' + name in runtime.registered
    assert name in runtime.xx.morphology.__all__
    assert public is getattr(runtime.owned, name) and public is not original
    assert inspect.signature(public) == inspect.signature(original)


@pytest.mark.parametrize('spec', CASES, ids=lambda spec: spec['id'])
def test_public_frozen_footprint_case(runtime, spec, monkeypatch, tmp_path):
    _check_public_case(spec, runtime, 1, monkeypatch, tmp_path)


@pytest.mark.parametrize('spec', [row for row in CASES if row['id'] in WORKER_FOUR_IDS],
                         ids=lambda spec: spec['id'])
def test_public_footprint_optional_four_workers(runtime, spec, monkeypatch, tmp_path):
    if not runtime.can_parallel:
        assert runtime.xx.get_num_threads() == 1
        with pytest.raises(ValueError, match='this build has no OpenMP support'):
            runtime.xx.set_num_threads(4)
        assert runtime.xx.get_num_threads() == 1
        (tmp_path / 'portable-refusal.json').write_text(json.dumps(dict(
            id=spec['id'], requested_workers=4, image_calls=0)) + '\n', encoding='utf-8')
        return
    _check_public_case(spec, runtime, 4, monkeypatch, tmp_path)


@pytest.mark.parametrize('workers', (1, 4))
def test_large_public_ellipse_above_parallel_fill_threshold(runtime, workers, monkeypatch):
    # 515*513 > execution.hpp's 256*1024 partition threshold. The small frozen
    # corpus checks worker policy but never reaches this actual fill branch.
    xx, np = runtime.xx, runtime.np
    assert xx.get_num_threads() == 1
    if workers == 4 and not runtime.can_parallel:
        with pytest.raises(ValueError, match='this build has no OpenMP support'):
            xx.set_num_threads(4)
        return
    expected = runtime.original.ellipse(256, 257, dtype=np.uint16)
    previous = xx.set_fallback_enabled(False)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context() as patch:
            def denied(*args, **kwargs):
                raise AssertionError('upstream ellipse forwarding')
            patch.setattr(runtime.original, 'ellipse', denied)
            actual = xx.morphology.ellipse(256, 257, dtype=np.uint16)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.size > 256*1024 and np.any(actual == 0) and np.any(actual == 1)
        np.testing.assert_array_equal(actual, expected)
    finally:
        xx.set_num_threads(1)
        xx.set_fallback_enabled(previous)


def test_cross_plan_rejects_owned_unaligned_before_typed_read(runtime):
    np, core = runtime.np, runtime.core
    item = np.dtype(np.intp).itemsize
    owner = np.zeros(1 + 3 * item, dtype=np.uint8)
    # Valid in-allocation view; no unaligned typed writes prepare this control.
    sums = np.ndarray((3,), dtype=np.intp, buffer=owner, offset=1)
    before = owner.tobytes()
    assert sums.flags.c_contiguous and not sums.flags.aligned
    with pytest.raises(ValueError, match='cross sums must be aligned'):
        core.footprint_shapes_cross_plan(sums, 1)
    assert owner.tobytes() == before


def test_metadata_check_accepts_owned_roots_views_and_empty_arrays(runtime):
    np, core = runtime.np, runtime.core
    owner = np.zeros(6, dtype=np.uint8)
    empty = np.zeros(0, dtype=np.uint8)
    arrays = (owner, owner[1:5], empty, owner[3:3])
    assert owner.flags.owndata and empty.flags.owndata
    assert arrays[1].base is owner and arrays[3].base is owner
    before = owner.tobytes()
    for array in arrays:
        metadata = (array.shape, array.strides, array.dtype, array.flags.owndata)
        assert core.footprint_shapes_check(array) is None
        assert (array.shape, array.strides, array.dtype, array.flags.owndata) == metadata
    assert owner.tobytes() == before
