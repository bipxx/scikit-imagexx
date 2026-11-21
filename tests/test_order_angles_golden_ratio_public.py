"""All45 saved generator contracts per profile, actual native entry ownership."""
import importlib,importlib.util,inspect,json,os,sys,types
from pathlib import Path
import pytest
from order_angles_golden_ratio_contract import cases


@pytest.fixture(scope='module')
def native():
    import numpy as np
    from util_views_fenv import backend
    from order_angles_golden_ratio_saved import load_saved
    profile=os.environ['GOLDEN_PROFILE'];assert(np.__version__,profile)in(('1.24.4','minimum'),('2.5.4','modern'))
    libc,helper=backend();root=Path(__file__).resolve().parents[1]
    if os.environ.get('GOLDEN_CORE_EXTENSION'):
        package=types.ModuleType('_golden_family');package.__path__=[str(root/'python/skimagexx')];sys.modules[package.__name__]=package
        spec=importlib.util.spec_from_file_location('_golden_family._core',os.environ['GOLDEN_CORE_EXTENSION']);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);package._core=core;sys.modules[spec.name]=core
        own=importlib.import_module('_golden_family._order_angles_golden_ratio')
    else:
        import skimagexx as xx
        xx.set_fallback_enabled(False);xx.set_num_threads(1);core=xx._core
        own=importlib.import_module('skimagexx._order_angles_golden_ratio')
        assert xx.transform.order_angles_golden_ratio is own.order_angles_golden_ratio
    assert core.get_num_threads()==1
    assert own._core is core and all(inspect.isbuiltin(getattr(core,name))for name in('_order_angles_begin','_order_angles_resume','_order_angles_next'))
    yield np,core,own,load_saved(profile),libc,helper
    core.set_num_threads(1);assert core.get_num_threads()==1


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('recipe',cases(),ids=lambda s:s['id'])
def test_saved_original_generator(recipe,workers,native,tmp_path):
    from order_angles_golden_ratio_saved import observe_saved
    np,core,own,gold,libc,helper=native;before_workers=core.get_num_threads();core.set_num_threads(workers)
    entries=tuple((getattr(core,name),name)for name in('_order_angles_begin','_order_angles_resume','_order_angles_next'));returns=[];old_profile=sys.getprofile()
    def profile(frame,event,value):
        if old_profile is not None:old_profile(frame,event,value)
        if event=='c_return':
            for entry,name in entries:
                if value is entry:returns.append(name)
    try:
        sys.setprofile(profile)
        actual,delta=observe_saved(own.order_angles_golden_ratio,recipe,gold[recipe['id']],np,libc,helper)
    finally:
        sys.setprofile(old_profile);core.set_num_threads(before_workers)
    (tmp_path/'comparison.json').write_text(json.dumps(dict(id=recipe['id'],workers=workers,differences=delta,actual=actual,native_C_returns=returns),indent=2)+'\n')
    assert not delta,delta
    if actual['yields']:assert '_order_angles_begin'in returns
    if actual['yields']>1:assert '_order_angles_next'in returns


def test_one_lazy_global_budget_snapshot_and_live_native_state(native):
    np,core,own,_,_,_=native
    core.set_num_threads(1);theta=np.asarray([0.,90.]);g=own.order_angles_golden_ratio(theta)
    assert inspect.getgeneratorstate(g)=='GEN_CREATED'and 'state'not in g.gi_frame.f_locals
    core.set_num_threads(4);first=next(g);state=g.gi_frame.f_locals['state'];assert state.worker_budget==4
    core.set_num_threads(1);second=next(g);assert state.worker_budget==4 and type(first)is type(second)is np.int64
    g.close();assert core.get_num_threads()==1
    if os.environ.get('SKIMAGEXX_DISABLE_SIMD')=='1':assert not core.has_avx2()


@pytest.mark.parametrize('kind', (
    'provider', 'long', 'matrix', 'complex', 'nan', 'signaling-nan', 'infinity',
    'subnormal', 'outside', 'integer-outside', 'foreign-buffer', 'noncanonical-bool',
))
def test_fallback_is_lazy_and_forwards_generator_protocol(kind, native, monkeypatch):
    np, _, own, _, _, _ = native
    events = []

    class Provider:
        def __array__(self, *args, **kwargs):
            raise AssertionError('provider was coerced before fallback')

    class Thrown(Exception):
        pass

    factories = {
        'provider': Provider,
        'long': lambda: np.arange(17),
        'matrix': lambda: np.zeros((2, 2)),
        'complex': lambda: np.zeros(3, dtype=np.complex64),
        'nan': lambda: np.array([0.0, np.nan]),
        'signaling-nan': lambda: np.array([0x7ff0000000000001], dtype=np.uint64).view(np.float64),
        'infinity': lambda: np.array([-np.inf], dtype=np.float16),
        'subnormal': lambda: np.array([1], dtype=np.uint32).view(np.float32),
        'outside': lambda: np.array([361.0]),
        'integer-outside': lambda: np.array([2**53 + 1], dtype=np.uint64),
        'foreign-buffer': lambda: np.frombuffer(bytearray(8), dtype=np.float64),
        'noncanonical-bool': lambda: np.array([2], dtype=np.uint8).view(np.bool_),
    }
    theta = factories[kind]()
    token, error = object(), Thrown()

    def upstream(value):
        assert value is theta
        events.append('upstream started')
        try:
            assert (yield 8) is token
            try:
                yield 3
            except Thrown as thrown:
                assert thrown is error
                yield 1
        finally:
            events.append('upstream closed')

    def fallback(module, name, value):
        assert (module, name) == ('transform', 'order_angles_golden_ratio')
        events.append('fallback')
        return upstream(value)

    monkeypatch.setattr(own, 'fallback', fallback)
    pending = own.order_angles_golden_ratio(theta)
    pending.close()
    assert events == []
    result = own.order_angles_golden_ratio(theta)
    assert inspect.getgeneratorstate(result) == 'GEN_CREATED' and events == []
    assert next(result) == 8
    assert events == ['fallback', 'upstream started']
    assert result.send(token) == 3
    assert result.throw(error) == 1
    result.close()
    assert events == ['fallback', 'upstream started', 'upstream closed']


def test_strict_refusal_happens_at_first_resume(native):
    np, _, own, _, _, _ = native
    compat = importlib.import_module(own.fallback.__module__)
    saved = compat.set_fallback_enabled(False)
    try:
        result = own.order_angles_golden_ratio(np.ones(17))
        assert inspect.getgeneratorstate(result) == 'GEN_CREATED'
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            next(result)
    finally:
        compat.set_fallback_enabled(saved)


def test_native_refusal_does_not_touch_fp_or_resume_upstream(native, monkeypatch):
    import ctypes
    from util_views_fenv import FEnv

    np, core, own, _, libc, helper = native
    signaling_nan = np.array([0x7ff0000000000001], dtype=np.uint64).view(np.float64)
    captured = FEnv()
    assert libc.fegetenv(ctypes.byref(captured)) == 0
    try:
        refused = core._order_angles_begin(signaling_nan, False)
        after = FEnv()
        assert libc.fegetenv(ctypes.byref(after)) == 0
        assert refused is None and bytes(after) == bytes(captured)
    finally:
        assert helper.reference_fenv_restore(ctypes.byref(captured), 32) == 0

    def refuse(*args):
        raise AssertionError('native iteration fell back after yielding')

    monkeypatch.setattr(own, 'fallback', refuse)
    theta = np.array([0.0, 45.0, 90.0])
    result = own.order_angles_golden_ratio(theta)
    assert next(result) == 0
    theta[1] = np.nan
    with pytest.raises(ValueError, match='live input left the native domain'):
        next(result)
    assert inspect.getgeneratorstate(result) == 'GEN_CLOSED'