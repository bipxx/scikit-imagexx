"""Actual NumPy-only setup/refused observer checks; never call either API."""
import sys
import pytest
from array_views_contract import cases,fixture
from array_views_observer import observe,validate_result_storage
from crop_observer import encode
BY_ID={s['id']:s for s in cases()}


@pytest.fixture(scope='module')
def np():
    import numpy as np
    assert not any(n.split('.')[0]in('skimage','scipy','skimagexx')for n in sys.modules)
    return np


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_actual_factory_and_codec_original_refused(spec,np):
    before=np.geterr(),np.geterrcall(),np.getbufsize();args,options,events,buffers,bound=fixture(spec,np)
    assert events==[]and len(args)==2
    assert set(options)==({'step'}if spec['api'].endswith('windows')else set())
    assert encode((args,options,buffers),np)==encode((args,options,buffers),np)
    assert before==(np.geterr(),np.geterrcall(),np.getbufsize())
    assert all(a.size<=4096 for a in buffers.values())


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_actual_refused_observer_all_recipes(spec,np):
    calls=[]
    def denied(*args,**options):calls.append(True);raise RuntimeError('ARRAY_VIEWS_ORIGINAL_REFUSED')
    row=observe(denied,spec,np)
    assert calls==[True]and row['error']==dict(module='builtins',type='RuntimeError',message='ARRAY_VIEWS_ORIGINAL_REFUSED')
    assert row['before']==row['after']and all(row['checks'].values())and not row['FP_events']


@pytest.mark.parametrize('name',('view_as_blocks/baseline','view_as_blocks/layout-F','view_as_windows/layout-reverse_all','view_as_windows/readonly','view_as_windows/dtype-packed','view_as_blocks/dtype-object'))
def test_observer_deterministic_view_success_no_original(name,np):
    def standin(array,*args,**options):return array.view()
    row=observe(standin,BY_ID[name],np)
    assert row['error']is None and row['before']==row['after']and all(row['checks'].values())
    assert row['storage']['distinct_input_slots']==row['storage']['logical_elements']
    if name.endswith('readonly'):assert not row['mutation']['performed']


def test_observer_writable_overlap_single_slot_restores(np):
    def safe_overlap(array,*args,**options):return np.ndarray((2,2),dtype=array.dtype,buffer=array.base,strides=(0,0))
    row=observe(safe_overlap,BY_ID['view_as_windows/baseline'],np)
    assert row['storage']['distinct_input_slots']==1 and row['storage']['overlapping']
    assert row['mutation']['performed']and row['mutation']['restored']and len(row['mutation']['same_slot_indices'])==4
    assert row['before']==row['after']


def test_result_storage_refuses_unrelated_array_before_payload_read(np):
    _,_,_,buffers,_=fixture(BY_ID['view_as_blocks/baseline'],np)
    with pytest.raises(AssertionError,match='outside known live input slots'):
        validate_result_storage(np.zeros((2,2),dtype=np.int16),buffers,np)


def test_observer_warn_handler_and_raise_branches(np):
    for mode in ('warn','call','raise'):
        def standin(array,*args,**options):
            with np.errstate(divide=mode):np.divide(np.float64(1),np.float64(0))
            return array.view()
        row=observe(standin,BY_ID['view_as_blocks/baseline'],np)
        assert all(row['checks'].values())
        if mode=='warn':assert len(row['warnings'])==1 and not row['FP_events']
        if mode=='call':assert len(row['FP_events'])==1 and not row['warnings']
        if mode=='raise':assert row['error']['type']=='FloatingPointError'
