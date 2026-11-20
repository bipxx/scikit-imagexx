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
