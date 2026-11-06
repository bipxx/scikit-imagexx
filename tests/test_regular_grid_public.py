"""Full49-node Linux captured-domain parity through installed or private core.

Raw Linux flags/rounding and captured scalar words are not Windows golds.
The qualifier selects this whole module only for Linux profiles; existing
Windows global/default/SDK tests remain required, with full platform gaps open.
"""
import importlib
import importlib.util
import inspect
import ctypes
import json
import os
import platform
from pathlib import Path
import sys
import types
import numpy as np
import pytest
from regular_grid_contract import cases
from regular_grid_observer import observe
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def target():
    directory=os.environ.get('REGULAR_GRID_COMPONENT_DIR')
    if directory:
        path=next(Path(directory).glob('_regular_grid_core*.so'))
        spec=importlib.util.spec_from_file_location('_regular_grid_core',path)
        core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        package=types.ModuleType('_regular_grid_test');package.__path__=[str(ROOT/'python/skimagexx')];package._core=core
        sys.modules[package.__name__]=package;sys.modules[package.__name__+'._core']=core
        compat=importlib.import_module(package.__name__+'._compat')
        module=importlib.import_module(package.__name__+'._regular_grid')
    else:
        import skimagexx as xx
        module=importlib.import_module('skimagexx._regular_grid');core=xx._core
        compat=importlib.import_module('skimagexx._compat')
        assert xx.util.regular_grid is module.regular_grid and module._core is core
        assert Path(core.__file__).resolve().parent==Path(xx.__file__).resolve().parent
        assert xx.get_num_threads is core.get_num_threads and xx.set_num_threads is core.set_num_threads
        for name in ('regular_grid_prepare','regular_grid_finish','regular_grid_suffix_product','get_num_threads','set_num_threads'):
            entry=getattr(core,name)
            assert inspect.isbuiltin(entry)and entry.__module__=='skimagexx._core'
    previous=compat.set_fallback_enabled(False)
    try:
        assert inspect.isbuiltin(core.regular_grid_prepare) and inspect.isbuiltin(core.regular_grid_finish)
        assert core.get_num_threads()==1
        yield module,core
    finally:core.set_num_threads(1);compat.set_fallback_enabled(previous)


@pytest.mark.parametrize('spec',cases(),ids=lambda row:row['id'])
def test_saved_rows(target,spec,tmp_path):
    module,_=target
    profile='minimum' if np.__version__.startswith('1.') else 'modern'
    data=json.loads((ROOT/'tests/fixtures/regular_grid_originals_v1.json').read_bytes())
    expected=next(r for r in data['profiles'][profile]['raw_rows'] if r['id']==spec['id'])
    # Reuse the captured Linux x86-64 controller's32byte fenv_t restore after
    # each row. Warn/call zero divisions intentionally leave their flags set.
    assert sys.platform=='linux'
    assert platform.machine()=='x86_64'
    libc=ctypes.CDLL(None);saved=(ctypes.c_ubyte*32)()
    assert libc.fegetenv(ctypes.byref(saved))==0
    try:actual=observe(module.regular_grid,spec,np)
    finally:assert libc.fesetenv(ctypes.byref(saved))==0
    directory=Path(os.environ.get('REGULAR_GRID_RESULT_DIR',str(tmp_path)));directory.mkdir(parents=True,exist_ok=True)
    (directory/(spec['id']+'.json')).write_text(json.dumps(actual,indent=2)+'\n')
    for key in ('before','after','result','error','events','warnings','state_during','checks','fenv_before','fenv_after'):
        assert actual[key]==expected[key],key


def test_ordinary_numeric_ownership(target,monkeypatch):
    module,_=target
    def refused(*args,**kwargs):raise RuntimeError('ARRAY_NUMERIC_PROVIDER_REFUSED')
    for name in ('sort','argsort','prod','full','round'):monkeypatch.setattr(np,name,refused)
    result=module.regular_grid((4093,13,997),6000)
    assert len(result)==3 and all(type(s.start) is np.dtype(int).type and s.stop is None for s in result)


def test_floor_before_round_and_workers_unchanged(target):
    module,core=target;old=core.get_num_threads()
    try:
        for budget in (1,4):
            core.set_num_threads(budget)
            assert module.regular_grid((25,),10)==(slice(np.dtype(int).type(1),None,np.dtype(int).type(2)),)
            assert module.regular_grid((35,),10)==(slice(np.dtype(int).type(1),None,np.dtype(int).type(4)),)
            assert core.get_num_threads()==budget
    finally:core.set_num_threads(old)


def test_strict_unsupported_without_originals(target):
    module,_=target
    with pytest.raises(TypeError):module.regular_grid((5,7))
    for shape,points in ((np.array([5,7],dtype=np.complex64),6),((5,7),np.float32(6)),((5,7),np.float64(-2)),((float('nan'),7),6)):
        with pytest.raises(NotImplementedError):module.regular_grid(shape,points)


def test_exact_early_comparison_precedes_integer_point_conversion(target):
    module,_=target
    # Literal source witness: exact float/int comparison is false, subsequent
    # Python floating division rounds the point operand and produces step1.
    result=module.regular_grid((2**53+4,),2**53+3)
    assert len(result)==1 and type(result[0].start) is np.dtype(int).type
    assert result[0].start==0 and result[0].step==1 and result[0].stop is None
