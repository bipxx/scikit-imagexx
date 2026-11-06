"""Actual NumPy factories plus cached/cold target-request refusal only."""
import builtins
import json
import numpy as np
import pytest
from next_util_benchmark_support import ROOT,CONTEXT,load_driver,bound_request_refusal
DATA=json.loads(CONTEXT.read_bytes())


@pytest.mark.parametrize('route',tuple(DATA['routes']))
def test_actual_numpy_frozen_factories(route,monkeypatch):
    driver=load_driver(route);old=builtins.__import__;attempts=[]
    def refuse(name,*a,**k):
        if name.split('.')[0]in('skimage','skimagexx'):
            attempts.append(name);raise AssertionError('TARGET_REQUEST_FORBIDDEN')
        return old(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',refuse)
    for spec in driver.case_specs():
        result=driver.fixture(spec,np)
        if route=='crop':image,options=result;args=(image,spec['width'])
        else:args,options=result
        arrays=[v for v in args if type(v)is np.ndarray]
        assert all(v.size<=1048576 and v.flags.aligned for v in arrays)
        if route!='regular-grid':
            assert all(list(v.shape)==spec['shape']and v.dtype==np.dtype(spec['dtype'])for v in arrays)
            if spec['layout']=='F':assert all(v.flags.f_contiguous for v in arrays)
            if spec['layout']=='reverse':assert all(v.strides[-1]<0 for v in arrays)
            if spec['layout']=='readonly':assert all(not v.flags.writeable for v in arrays)
    assert not attempts


@pytest.mark.parametrize('route',tuple(DATA['routes']))
@pytest.mark.parametrize('fault',('none','none-cached'))
def test_actual_numpy_bound_CLI_stops_at_target_request(tmp_path,monkeypatch,route,fault):
    bound_request_refusal(tmp_path,monkeypatch,route,fault,target='skimage')


@pytest.mark.parametrize('fault',('none','none-cached'))
def test_actual_numpy_windows_API_selector_refuses_target(tmp_path,monkeypatch,fault):
    bound_request_refusal(tmp_path,monkeypatch,'views',fault,target='skimage',api='util.view_as_windows')
