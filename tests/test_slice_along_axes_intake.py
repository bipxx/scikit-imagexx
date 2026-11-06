"""NumPy-only fixture/observer setup. No original or project native imports."""
import ctypes
import sys
import pytest
from slice_along_axes_contract import cases,fixture,source_safe
from slice_along_axes_observer import observe,encode
from crop_observer import retain_object_payloads,object_payload_relations
BY_ID={row['id'].split('/',1)[1]:row for row in cases()}


@pytest.fixture(scope='module')
def numpy_only():
    import numpy as np
    assert np.__version__ in ('1.24.4','2.5.4')
    assert not any(name.split('.')[0]in('scipy','skimage','skimagexx')for name in sys.modules)
    return np


@pytest.fixture(autouse=True)
def restore_controller_FENV():
    # Existing original controllers use the retained Linux x86-64 fenv_t32
    # layout. Preserve full caller state around EVERY setup case, including FP
    # warn/call/raise controls; never hide observed source-call FP changes.
    if sys.platform!='linux':yield;return
    libc=ctypes.CDLL(None);saved=(ctypes.c_ubyte*32)();assert libc.fegetenv(ctypes.byref(saved))==0
    try:yield
    finally:assert libc.fesetenv(ctypes.byref(saved))==0


@pytest.mark.parametrize('row',cases(),ids=lambda r:r['id'])
def test_factory_codec_bounded_without_any_target(row,numpy_only):
    np=numpy_only;state=np.geterr(),np.geterrcall(),np.getbufsize()
    args,options,events,buffers=fixture(row,np);bound=source_safe(row)
    assert len(args)==2 and set(options)=={'axes','copy'}and events==[]
    assert all(isinstance(a,np.ndarray)and a.size<=bound['physical_input_elements']for a in buffers.values())
    assert encode((args,options,buffers),np)==encode((args,options,buffers),np)
    assert events==[]and np.geterr()==state[0]and np.geterrcall()is state[1]and np.getbufsize()==state[2]


@pytest.mark.parametrize('row',cases(),ids=lambda r:r['id'])
def test_observer_refused_body_all_recipes(row,numpy_only):
    calls=[]
    def refused(*args,**options):calls.append(True);raise AssertionError('SLICE_ORIGINAL_REFUSED')
    result=observe(refused,row,numpy_only)
    assert calls==[True]and result['error']==dict(module='builtins',type='AssertionError',message='SLICE_ORIGINAL_REFUSED')
    assert result['before']==result['after']and result['events']==[]and result['FP_events']==[]and result['warnings']==[]
    assert all(result['checks'].values())and result['supplied_call_attempts']==1


@pytest.mark.parametrize('name,mode',[
    ('ordinary-default','view'),('layout-F-False','view'),('layout-reverse_all-False','view'),('readonly-False','view'),
    ('copy-dtype-object','copy'),('copy-dtype-packed','copy'),('copy-dtype-U2','copy'),('copy-dtype-complex128','copy'),
    ('input-subclass','copy'),('identity-array_provider','identity'),('identity-none-image','identity'),('identity-numpy_scalar','identity'),
    ('image-probe-view-False','image'),('image-probe-owned-True','image'),('image-probe-list-False','image'),
    ('mutation-False','view'),('mutation-True','copy'),('mutation-readonly','view'),
    ('input-distinct_objects','view'),('identity-distinct_objects','copy')])
def test_deterministic_success_standins_only(name,mode,numpy_only):
    np=numpy_only;calls=[]
    def standin(image,*args,**options):
        calls.append(True)
        if mode=='identity':return image
        if mode=='image':return image[(slice(None),)*image.array.ndim]
        return image.view()if mode=='view'else image.copy()
    row=observe(standin,BY_ID[name],np)
    assert calls==[True]and row['error']is None and all(row['checks'].values())
    assert row['before']==row['after']
    if row['mutation']is not None:assert row['mutation']['restored']
    if name=='mutation-readonly':assert row['mutation']['error']['type']=='ValueError'
    if mode=='identity':assert row['result_is_argument']


def test_object_identity_distinguishes_shallow_from_reconstruction(numpy_only):
    np=numpy_only;args,_,_,buffers=fixture(BY_ID['input-distinct_objects'],np);array=args[0]
    values=list(array.flat);assert values[0]==values[2]and values[0]is not values[2]
    retained=retain_object_payloads(buffers,np);shallow=array.copy();rebuilt=array.copy()
    for i,value in enumerate(array.flat):rebuilt.flat[i]=value.encode().decode()
    assert encode(shallow,np)==encode(rebuilt,np)
    good=object_payload_relations(shallow,retained,np);bad=object_payload_relations(rebuilt,retained,np)
    assert good!=bad and all(any(line)for line in good['is_retained'])and not any(any(line)for line in bad['is_retained'])


@pytest.mark.parametrize('mode',('warn','call','raise'))
def test_observer_FP_warning_callback_error_setup(mode,numpy_only):
    np=numpy_only;calls=[]
    def standin(image,*args,**options):
        calls.append(True)
        with np.errstate(divide=mode):np.divide(np.float64(1),np.float64(0))
        return image.view()
    row=observe(standin,BY_ID['ordinary-default'],np);assert calls==[True]and all(row['checks'].values())
    if mode=='warn':assert len(row['warnings'])==1 and row['FP_events']==[]and row['error']is None
    elif mode=='call':assert len(row['FP_events'])==1 and row['warnings']==[]and row['error']is None
    else:assert row['error']['type']=='FloatingPointError'


def test_observer_rejects_policy_mutation_and_restores_globals(numpy_only):
    np=numpy_only;old=np.geterr(),np.geterrcall(),np.getbufsize()
    def tamper(image,*args,**options):np.seterr(divide='raise');return image.view()
    with pytest.raises(AssertionError):observe(tamper,BY_ID['ordinary-default'],np)
    assert np.geterr()==old[0]and np.geterrcall()is old[1]and np.getbufsize()==old[2]
