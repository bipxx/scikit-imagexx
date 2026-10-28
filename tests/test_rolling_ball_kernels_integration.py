"""Registered rolling geometry, exact saved observations and constructor adapters."""
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import pytest

if __package__:
    from .rolling_ball_kernels_contract import APIS,cases
    from .test_rolling_ball_kernels_public import observe
else:
    from rolling_ball_kernels_contract import APIS,cases
    from test_rolling_ball_kernels_public import observe

ROOT=Path(__file__).parents[1];CASES=cases()


@pytest.fixture(scope='module')
def product():
    import numpy as np,skimagexx as xx
    return np,xx,json.loads((ROOT/'coverage/contracts/rolling-ball-kernels.json').read_bytes())


@pytest.mark.parametrize('api',APIS)
def test_geometry_registered_exports_and_carrier(product,api):
    np,xx,matrix=product
    from skimagexx import _rolling_ball_kernels as implementation
    from skimagexx.coverage import native_functions,native_dtypes
    name=api.split('.')[-1]
    assert getattr(xx.restoration,name)is getattr(implementation,name)
    assert str(inspect.signature(getattr(xx.restoration,name)))==matrix['signatures'][api]
    assert api in native_functions()and native_dtypes(api)==('float64',)
    assert inspect.isbuiltin(getattr(implementation._core,name+'_apply'))


@pytest.mark.parametrize('spec',CASES,ids=[s['id']for s in CASES])
@pytest.mark.parametrize('workers',(1,4))
def test_geometry_registered_public_frozen_contract(product,spec,workers,monkeypatch,tmp_path):
    np,xx,matrix=product
    from skimagexx import _rolling_ball_kernels as implementation
    if __package__:
        from .test_rolling_ball_kernels_pipeline import LITERAL_IDS
    else:
        from test_rolling_ball_kernels_pipeline import LITERAL_IDS
    import skimage.restoration as public
    import skimage.restoration._rolling_ball as original
    expected=next(r for r in matrix['original_observations'][np.__version__]if r['id']==spec['id'])
    name=spec['api'].split('.')[-1];call=getattr(xx.restoration,name);entry=getattr(implementation._core,name+'_apply')
    codes={getattr(original,api.split('.')[-1]).__code__for api in APIS};calls=[];accepted=[]
    previous,prior=xx.get_num_threads(),sys.getprofile()
    def reject(*a,**kw):raise AssertionError('registered rolling constructor forwarded to original whole API')
    def profile(frame,event,value):
        if event=='call'and frame.f_code in codes:reject()
        if event=='c_return'and value is entry:calls.append(name+'_apply')
        if event=='return'and frame.f_code is call.__code__:accepted.append(frame.f_locals.get('native_result')is True)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            for api in APIS:
                key=api.split('.')[-1];patch.setattr(public,key,reject);patch.setattr(original,key,reject)
            sys.setprofile(profile)
            try:actual=observe(call,spec,np)
            finally:sys.setprofile(prior)
        assert actual==expected
        native_expected=expected['error']is None and spec['id']not in LITERAL_IDS
        assert accepted==[native_expected]and len(calls)<=1
        if native_expected:assert calls==[name+'_apply']
        assert xx.get_num_threads()==workers
        (tmp_path/'actual.json').write_text(json.dumps(dict(id=spec['id'],workers=workers,observation=actual,C_returns=calls,native_accepted=accepted))+'\n')
    finally:
        sys.setprofile(prior);xx.set_num_threads(previous)


@pytest.mark.parametrize('api',APIS)
@pytest.mark.parametrize('shape',((1,1),(3,19),(7,9)))
@pytest.mark.parametrize('workers',(1,4))
def test_geometry_default_adapter_exact_words_and_inf_masks(product,api,shape,workers,monkeypatch):
    np,xx,_=product
    from skimagexx import _rolling_ball_kernels as implementation
    from skimagexx.coverage import default_call,rolling_kernel_default_error,assert_default_parity
    module_spec=importlib.util.spec_from_file_location('rolling_benchmark_default_oracle',ROOT/'benchmarks/rolling_ball_kernels.py')
    benchmark=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(benchmark)
    image=np.zeros(shape,np.float64);args,options=default_call(api,image)
    other=np.full(shape,99.,np.float64);assert default_call(api,other)==(args,options)
    params=dict(radius=args[0],ndim=args[1])if api==APIS[0]else dict(shape=list(args[0]),intensity=args[1])
    raw,output_shape=benchmark.coordinate_oracle(dict(api=api,parameters=params))
    expected=np.empty(output_shape,np.float64);expected[...] = np.frombuffer(raw,np.float64).reshape(output_shape)
    name=api.split('.')[-1];call=getattr(xx.restoration,name);entry=getattr(implementation._core,name+'_apply')
    original=importlib.import_module('skimage.restoration._rolling_ball');public=importlib.import_module('skimage.restoration')
    codes={getattr(original,a.split('.')[-1]).__code__for a in APIS};calls=[];accepted=[];prior=sys.getprofile();previous=xx.get_num_threads()
    def reject(*a,**kw):raise AssertionError('default rolling constructor forwarded to original')
    def profile(frame,event,value):
        if event=='call'and frame.f_code in codes:reject()
        if event=='c_return'and value is entry:calls.append(name+'_apply')
        if event=='return'and frame.f_code is call.__code__:accepted.append(frame.f_locals.get('native_result')is True)
    try:
        xx.set_num_threads(workers)
        with monkeypatch.context()as patch:
            for a in APIS:
                key=a.split('.')[-1];patch.setattr(original,key,reject);patch.setattr(public,key,reject)
            sys.setprofile(profile)
            try:actual=call(*args,**options)
            finally:sys.setprofile(prior)
        assert calls==[name+'_apply']and accepted==[True]
        assert rolling_kernel_default_error(api,shape,actual,expected)==0.
        assert_default_parity(api,np.dtype('float64'),actual,expected,image=image)
    finally:
        sys.setprofile(prior);xx.set_num_threads(previous)


@pytest.mark.parametrize('reason',('inf-sign','dtype','word','storage'))
def test_geometry_exact_checker_rejects_corruption(product,reason):
    np,_,_=product
    from skimagexx.coverage import rolling_kernel_default_error
    expected=np.array([[np.inf,0.,np.inf],[0.,1.,0.],[np.inf,0.,np.inf]],np.float64)
    actual=expected.copy()
    if reason=='inf-sign':actual[0,0]=-np.inf
    elif reason=='dtype':actual=actual.astype(np.float32)
    elif reason=='word':actual[1,1]=np.nextafter(actual[1,1],2.)
    else:actual=actual.view()
    with pytest.raises(AssertionError):rolling_kernel_default_error(APIS[0],(3,3),actual,expected)
