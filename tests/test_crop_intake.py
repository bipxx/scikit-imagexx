"""Actual NumPy-only crop setup; every original/native import is refused."""
import sys
import pytest
from crop_contract import cases,fixture,source_safe,MAX_INPUT_ELEMENTS
from crop_observer import encode,observe,retain_object_payloads,object_payload_relations
BY_ID={s['id'].split('/',1)[1]:s for s in cases()}


@pytest.fixture(scope='module')
def numpy_only():
    import numpy as np
    assert not any(n.split('.')[0]in('scipy','skimage','skimagexx')for n in sys.modules)
    return np


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_crop_factory_codec_original_refused(spec,numpy_only):
    np=numpy_only;state=np.geterr(),np.geterrcall(),np.getbufsize()
    args,options,events,buffers=fixture(spec,np);bound=source_safe(spec)
    assert len(args)==2 and set(options)=={'copy','order'}and events==[]
    assert all(type(a)is np.ndarray or isinstance(a,np.ndarray)for a in buffers.values())
    assert all(a.size<=bound['physical_input_elements']for name,a in buffers.items()if name!='width_array')
    assert encode((args,options,buffers),np)==encode((args,options,buffers),np)
    assert np.geterr()==state[0]and np.geterrcall()is state[1]and np.getbufsize()==state[2]
    if spec['input_kind']=='distinct_objects':
        values=list(buffers['input_array'].flat)
        assert all(type(x)is str for x in values)
        if len(values)>2:assert values[0]==values[2]and values[0]is not values[2]


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_crop_observer_refused_body_all_recipes(spec,numpy_only):
    np=numpy_only;state=np.geterr(),np.geterrcall(),np.getbufsize();calls=[]
    def denied(*args,**options):calls.append(True);raise AssertionError('CROP_ORIGINAL_REFUSED')
    row=observe(denied,spec,np)
    assert calls==[True]and row['error']==dict(module='builtins',type='AssertionError',message='CROP_ORIGINAL_REFUSED')
    assert row['before']==row['after']and all(row['checks'].values())and row['FP_events']==[]
    assert np.geterr()==state[0]and np.geterrcall()is state[1]and np.getbufsize()==state[2]
    if row['object_payload_identity']is not None:assert row['object_payload_identity']['kind']=='error'


@pytest.mark.parametrize('name,mode',[
    ('width-zero','view'),('readonly-view','view'),('layout-reverse_all','view'),
    ('copy-True-order-K','copy'),('dtype-packed','copy'),('dtype-complex128','copy'),
    ('dtype-tagged_i16','copy'),('dtype-V3','view'),('dtype-S3','view'),('dtype-U2','copy'),
    ('zeroD-view','scalar'),('zeroD-packed-view','scalar'),('zeroD-object-view','scalar'),
    ('dtype-object','view'),('copy-dtype-object','copy'),('object-distinct-copy','copy'),
    ('zeroD-distinct-object','scalar'),('input-array_provider','asarray'),
    ('input-interface_provider','asarray'),('input-list','asarray'),('input-tuple','asarray'),
    ('input-memoryview','asarray'),('input-subclass','view'),('input-numpy_scalar','identity'),('input-python_scalar','identity'),
])
def test_crop_observer_deterministic_success_branches(name,mode,numpy_only):
    np=numpy_only;calls=[]
    def standin(ar,width,**options):
        calls.append(True)
        if mode=='view':return ar.view()
        if mode=='copy':return ar.copy(order='K')
        if mode=='scalar':return ar[()]
        if mode=='asarray':return np.asarray(ar)
        return ar
    row=observe(standin,BY_ID[name],np)
    assert calls==[True]and row['error']is None and all(row['checks'].values())
    assert row['before']==row['after']
    if name=='object-distinct-copy':
        relation=row['object_payload_identity'];assert relation['kind']=='array'
        assert all(sum(line)==2 for line in relation['is_retained']) # owner/input-array labels point to same slots.
    if name=='zeroD-distinct-object':
        assert row['result_type']==dict(module='builtins',name='str')
        assert row['object_payload_identity']['kind']=='scalar'and row['object_payload_identity']['is_retained']==[True,True]
    if name=='input-array_provider':assert any(e['event']=='array.__array__'for e in row['events'])
    if name=='input-interface_provider':assert any(e['event']=='array.__array_interface__'for e in row['events'])


@pytest.mark.parametrize('name',('object-distinct-copy','zeroD-distinct-object'))
def test_crop_identity_rejects_equal_reconstructed_payloads(name,numpy_only):
    np=numpy_only;args,_,_,buffers=fixture(BY_ID[name],np);ar=args[0];retained=retain_object_payloads(buffers,np)
    if ar.ndim:
        shallow=ar.copy();reconstructed=ar.copy()
        for index,value in enumerate(ar.flat):reconstructed.flat[index]=value.encode().decode()
        assert all(a==b and a is not b for a,b in zip(shallow.flat,reconstructed.flat))
    else:
        shallow=ar[()];reconstructed=shallow.encode().decode();assert shallow==reconstructed and shallow is not reconstructed
    assert encode(shallow,np)==encode(reconstructed,np)
    good=object_payload_relations(shallow,retained,np);bad=object_payload_relations(reconstructed,retained,np)
    assert good!=bad
    if ar.ndim:assert all(any(line)for line in good['is_retained'])and not any(any(line)for line in bad['is_retained'])
    else:assert any(good['is_retained'])and not any(bad['is_retained'])


def test_crop_object_identity_refuses_oversize_setup(numpy_only):
    np=numpy_only;large=np.empty(MAX_INPUT_ELEMENTS+1,dtype=object)
    with pytest.raises(AssertionError):retain_object_payloads({'controlled':large},np)


@pytest.mark.parametrize('mode',('warn','call','raise'))
def test_crop_observer_FP_warning_callback_error_and_restoration(mode,numpy_only):
    np=numpy_only;old=np.geterr(),np.geterrcall(),np.getbufsize();calls=[]
    def standin(ar,width,**options):
        calls.append(True)
        with np.errstate(divide=mode):np.divide(np.float64(1),np.float64(0))
        return ar.view()
    row=observe(standin,BY_ID['width-zero'],np)
    assert calls==[True]and all(row['checks'].values())
    if mode=='warn':assert len(row['warnings'])==1 and row['FP_events']==[]and row['error']is None
    elif mode=='call':assert len(row['FP_events'])==1 and row['warnings']==[]and row['error']is None
    else:assert row['error']['type']=='FloatingPointError'
    assert np.geterr()==old[0]and np.geterrcall()is old[1]and np.getbufsize()==old[2]


def test_crop_observer_detects_policy_change_and_restores_globals(numpy_only):
    np=numpy_only;old=np.geterr(),np.geterrcall(),np.getbufsize()
    def change(ar,width,**options):np.seterr(divide='raise');return ar.view()
    with pytest.raises(AssertionError):observe(change,BY_ID['width-zero'],np)
    assert np.geterr()==old[0]and np.geterrcall()is old[1]and np.getbufsize()==old[2]
