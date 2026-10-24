"""Private real-binding prototype; no public namespace or SDK admission.

112 safe source observations execute C++; 23 rejection/blocked observations
retain their original boundary classification. The sweep remains serial.
"""
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SPEC = os.environ.get('SKIMAGEXX_TV_BREGMAN_CORE')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contracts = _load('_bregman_frozen_contracts', ROOT / 'tests/test_tv_bregman_contracts.py')
frozen = json.loads(contracts.MATRIX.read_text())


@pytest.fixture(scope='module')
def implementation():
    if SPEC is None:
        pytest.skip('explicit private Bregman binding path required; public admission pending')
    path = Path(SPEC).resolve(strict=True)
    core = _load('_tv_bregman_core', path)
    module = _load('skimagexx._tv_bregman_private_source', ROOT / 'python/skimagexx/_tv_bregman.py')
    module._core = core
    assert inspect.isbuiltin(module._require_kernel())
    assert Path(core.__file__).resolve() == path
    assert core.get_num_threads() == 1
    if os.environ.get('SKIMAGEXX_EXPECT_GIL_ASSERTIONS') == '1':
        assert core.gil_assertions is True
    return module, core


@pytest.fixture(params=(1, 4))
def workers(request, implementation):
    _, core = implementation
    previous = core.get_num_threads()
    if request.param == 4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF') == '1':
        with pytest.raises(ValueError, match='no OpenMP'):
            core.set_num_threads(4)
        assert core.get_num_threads() == previous == 1
        pytest.skip('portable explicitly rejects worker4; not parallel parity')
    core.set_num_threads(request.param)
    try:
        assert core.get_num_threads() == request.param
        yield request.param
    finally:
        core.set_num_threads(previous)


@pytest.mark.parametrize('spec', contracts.CASES, ids=lambda row: row['id'])
def test_frozen_observations_with_actual_private_cpp(spec, workers, implementation, monkeypatch):
    module, core = implementation
    calls = []
    original = core.tv_bregman_run
    # Admission validates authentic builtins; instrumentation belongs at the
    # helper boundary, leaving the authentic core entry unmodified.
    helper = module._denoise_tv_bregman
    def counted_helper(*args):
        calls.append((args[0].shape, args[5].__array_interface__['data'][0]))
        return helper(*args)
    monkeypatch.setattr(module, '_denoise_tv_bregman', counted_helper)
    def forbidden(*args, **kwargs):
        raise AssertionError('upstream Bregman algorithm forwarding forbidden')
    monkeypatch.setattr(contracts.original_module, '_denoise_tv_bregman', forbidden)
    observer = types.FunctionType(contracts.observe.__code__,
        dict(contracts.observe.__globals__, original_module=module))
    actual = observer(module.denoise_tv_bregman, spec)
    expected = frozen['observations'][frozen['profiles'][contracts.profile()]['snapshots'][spec['id']]]
    assert actual == expected
    assert len(calls) == (len(actual['entries']) if spec['mode'] == 'compiled' else 0)
    if spec['mode'] == 'compiled' and len(calls) > 1:
        assert len({pointer for _, pointer in calls}) == 1


@pytest.mark.parametrize('fault', ('absent', 'version', 'float-cap', 'bool-cap', 'partial', 'entry', 'sweep'))
def test_admission_before_image_access(implementation, monkeypatch, fault):
    module, core = implementation
    attrs = dict(tv_bregman_interface='tv-bregman-prepared-v1', tv_bregman_capabilities=3,
                 tv_bregman_parallel_sweep=False, tv_bregman_run=core.tv_bregman_run, __name__=core.__name__)
    if fault == 'absent': attrs.pop('tv_bregman_run')
    elif fault == 'version': attrs['tv_bregman_interface'] = 'other'
    elif fault == 'float-cap': attrs['tv_bregman_capabilities'] = 3.0
    elif fault == 'bool-cap': attrs['tv_bregman_capabilities'] = True
    elif fault == 'partial': attrs['tv_bregman_capabilities'] = 1
    elif fault == 'entry': attrs['tv_bregman_run'] = lambda *a: None
    else: attrs['tv_bregman_parallel_sweep'] = True
    monkeypatch.setattr(module, '_core', types.SimpleNamespace(**attrs))
    class Unreadable:
        def __array__(self, *args, **kwargs): raise AssertionError('image accessed')
    with pytest.raises(NotImplementedError): module.denoise_tv_bregman(Unreadable())


@pytest.mark.parametrize('dtype', ('f4', 'f8'))
@pytest.mark.parametrize('maximum', (0, -1, 1, 3))
def test_real_trace_and_setup(dtype, maximum, workers, implementation):
    _, core = implementation
    image = np.arange(20, dtype=dtype).reshape(4, 5, 1) / 32
    out = np.full((6, 7, 1), 0.125, dtype=dtype)
    untouched = out[[0, 0, -1, -1], [0, -1, 0, -1]].copy()
    traced = core.tv_bregman_run(image, np.dtype(dtype).type(.5), maximum, 1e-6, True, out, trace=True)
    assert traced['iterations'] == len(traced['trace'])
    assert traced['iterations'] <= max(0, maximum)
    assert np.array_equal(out[[0, 0, -1, -1], [0, -1, 0, -1]], untouched)
    again = np.full_like(out, .125)
    assert core.tv_bregman_run(image, np.dtype(dtype).type(.5), maximum, 1e-6, True, again) is None
    assert out.tobytes() == again.tobytes()


@pytest.mark.parametrize('fault', ('endian', 'readonly', 'unaligned', 'dtype', 'shape', 'overlap', 'extent', 'nonfinite'))
def test_prewrite_binding_guards(fault, implementation):
    _, core = implementation
    image = np.arange(20, dtype='f4').reshape(4, 5, 1) / 32
    out = np.zeros((6, 7, 1), dtype='f4')
    if fault == 'endian': image = image.astype('>f4')
    elif fault == 'readonly': image.flags.writeable = False
    elif fault == 'unaligned': image = np.ndarray(image.shape, 'f4', buffer=bytearray(81), offset=1)
    elif fault == 'dtype': image = image.astype('i4')
    elif fault == 'shape': out = np.zeros((6, 8, 1), dtype='f4')
    elif fault == 'overlap': image = out.reshape(-1)[:20].reshape(4, 5, 1)
    elif fault == 'extent': image = np.lib.stride_tricks.as_strided(image[:1], shape=(4, 5, 1), strides=(20, 4, 4))
    elif fault == 'nonfinite': image[0, 0, 0] = np.inf
    before = out.tobytes()
    with pytest.raises((ValueError, TypeError)): core.tv_bregman_run(image, .5, 2, 1e-6, True, out)
    assert out.tobytes() == before


def test_private_source_and_fixed_signature(implementation):
    module, core = implementation
    assert str(inspect.signature(module.denoise_tv_bregman)) == frozen['signature']
    assert hashlib.sha256(contracts.MATRIX.read_bytes()).hexdigest() == '638a6f569348a94f36ba11ef1dbbb8f1f08f4acbc1aec1cf893e9fe968cfa85c'
    assert core.tv_bregman_parallel_sweep is False


@pytest.mark.parametrize('dtype', ('f4', 'f8'))
def test_actual_initialization_worker_team(dtype, workers, implementation):
    _, core = implementation
    image = np.full((513, 519, 1), .25, dtype=dtype)
    out = np.zeros((515, 521, 1), dtype=dtype)
    result = core.tv_bregman_run(image, np.dtype(dtype).type(.5), 2, 1e-6, True, out, trace=True)
    assert result['initialization_workers'] == workers
    assert result['iterations'] == 1 and result['trace'][0] == (0, True)
    assert np.all(out[1:-1, 1:-1] == .25)


def test_binding_owners_released_on_success_and_guard_failure(implementation):
    _, core = implementation
    image = np.full((4, 5, 1), .25, dtype='f8')
    out = np.zeros((6, 7, 1), dtype='f8')
    counts = (sys.getrefcount(image), sys.getrefcount(out))
    for _ in range(128):
        assert core.tv_bregman_run(image, .5, 2, 1e-6, True, out) is None
        with pytest.raises(ValueError):
            core.tv_bregman_run(image, -1., 2, 1e-6, True, out)
    assert (sys.getrefcount(image), sys.getrefcount(out)) == counts


@pytest.mark.parametrize('parameter', ('weight', 'maximum', 'eps', 'isotropic'))
@pytest.mark.parametrize('mutation', ('rank', 'shape', 'dtype', 'strides', 'readonly', 'owner-resize'))
def test_scalar_hooks_cannot_invalidate_captured_input(parameter, mutation, implementation):
    """Never execute unchecked original Cython on deliberately invalidated storage."""
    _, core = implementation
    owner = np.arange(20, dtype='f4')
    image = owner.reshape(4, 5, 1)
    out = np.full((6, 7, 1), .125, dtype='f4')
    before = out.tobytes()
    events = []
    def change():
        events.append(parameter)
        if mutation == 'rank': image.shape = (4, 5)
        elif mutation == 'shape': image.shape = (5, 4, 1)
        elif mutation == 'dtype': image.dtype = np.uint32
        elif mutation == 'strides': image.strides = (4, 16, 4)
        elif mutation == 'readonly': image.flags.writeable = False
        else: owner.resize((1,), refcheck=False)
    class Hook:
        def __float__(self): change(); return .5
        def __index__(self): change(); return 2
        def __int__(self): change(); return 1
    arguments = [image, .5, 2, 1e-6, True, out]
    arguments[{'weight': 1, 'maximum': 2, 'eps': 3, 'isotropic': 4}[parameter]] = Hook()
    with pytest.raises(ValueError, match='image (metadata|allocation) changed|array exceeds its allocation extent'):
        core.tv_bregman_run(*arguments)
    assert events == [parameter]
    assert out.tobytes() == before


@pytest.mark.parametrize('parameter', ('weight', 'maximum', 'eps', 'isotropic'))
def test_scalar_hooks_preserve_actual_posthook_values(parameter, implementation):
    _, core = implementation
    image = np.arange(20, dtype='f8').reshape(4, 5, 1) / 32
    out = np.zeros((6, 7, 1), dtype='f8')
    events = []
    def change():
        events.append(parameter)
        image[0, 0, 0] = .25
    class Hook:
        def __float__(self): change(); return .5 if parameter == 'weight' else 1e-6
        def __index__(self): change(); return 2
        def __int__(self): change(); return 1
    arguments = [image, .5, 2, 1e-6, True, out]
    arguments[{'weight': 1, 'maximum': 2, 'eps': 3, 'isotropic': 4}[parameter]] = Hook()
    assert core.tv_bregman_run(*arguments) is None
    expected = np.zeros_like(out)
    assert core.tv_bregman_run(image, .5, 2, 1e-6, True, expected) is None
    assert events == [parameter] and out.tobytes() == expected.tobytes()
