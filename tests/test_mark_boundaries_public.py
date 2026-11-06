"""All250 admitted saved records; no original calls or golden rewrite."""
import base64,ctypes,hashlib,importlib,json,os,sys,types,zlib
from pathlib import Path
import numpy as np
import pytest
from mark_boundaries_contract import cases,fixture
from mark_boundaries_observer import observe as legacy_observe
from mark_boundaries_observer_v2 import observe as typed_observe
ROOT=Path(__file__).resolve().parents[1]
TRANSPORT=json.loads((ROOT/'tests/fixtures/mark_boundaries_originals_v1.json').read_bytes())
rows={}
for name,blob in TRANSPORT['blobs'].items():
    raw=zlib.decompress(base64.b64decode(blob['zlib_base64']));assert hashlib.sha256(raw).hexdigest()==blob['sha256']
    if name.endswith('journal.jsonl'):
        rows['1.24.4']=[dict(record=json.loads(line)['row'],format='legacy39bb')for line in raw.splitlines()if json.loads(line)['event']=='completed']
    else:
        version='1.24.4'if'/minimum/'in name else'2.5.4'
        rows.setdefault(version,[]).extend(dict(record=row,format='typed288')for row in json.loads(raw))
EXTENSION=json.loads((ROOT/'tests/fixtures/mark_boundaries_originals_v2.json').read_bytes())
assert hashlib.sha256((ROOT/EXTENSION['predecessor']['path']).read_bytes()).hexdigest()==EXTENSION['predecessor']['sha256']
admission_raw=zlib.decompress(base64.b64decode(EXTENSION['admission']['raw_zlib_base64']))
assert hashlib.sha256(admission_raw).hexdigest()==EXTENSION['admission']['raw_sha256']=='3eb0cdd022e24bda9a863219f2d2b7df7d67e1677ace896538f64b5cd1b0984f'
witness_raw=zlib.decompress(base64.b64decode(EXTENSION['witness']['raw_zlib_base64']))
assert hashlib.sha256(witness_raw).hexdigest()==EXTENSION['witness']['raw_sha256']==json.loads(admission_raw)['raw_files'][EXTENSION['witness']['path']]['sha256']
witness=json.loads(witness_raw);assert len(witness)==1 and witness[0]['id']==EXTENSION['case_id']=='mark_boundaries/background-none'
assert witness[0]['id']not in{r['record']['id']for r in rows['1.24.4']}
rows['1.24.4'].append(dict(record=witness[0],format='typed288'))
assert all(len(value)==len({r['record']['id']for r in value})==125 for value in rows.values())
SAVED={r['record']['id']:r for r in rows[np.__version__]}
SPECS=cases();assert len(SPECS)==len(SAVED)==125 and{r['id']for r in SPECS}==set(SAVED)


@pytest.fixture(scope='module')
def owned():
    directory=os.environ.get('MARK_COMPONENT_DIR')
    if not directory:return importlib.import_module('skimagexx._mark_boundaries')
    sys.path.insert(0,directory);core=importlib.import_module('_mark_boundaries_core')
    package=types.ModuleType('mark_component');package.__path__=[str(ROOT/'python/skimagexx')];sys.modules[package.__name__]=package;sys.modules[package.__name__+'._core']=core
    return importlib.import_module(package.__name__+'._mark_boundaries')


@pytest.fixture(scope='module')
def fp_backend():
    from util_views_fenv import backend
    if os.environ.get('MARK_COMPONENT_DIR'):
        assert os.environ.get('MARK_FENV_HELPER')
        os.environ['UTIL_VIEWS_FENV_MODE']='private-fixed-v1'
        os.environ['UTIL_VIEWS_FENV_HELPER']=os.environ['MARK_FENV_HELPER']
    else:
        assert os.environ.get('UTIL_VIEWS_FENV_MODE')=='source-built-v1','Genuine MARK requires admitted source-built FP receipt, no private fallback'
    return backend()  # Authenticate once at this new entry, outside case loop.


def comparable(row):
    # Only explicitly tagged raw process-address diagnostics are nonportable.
    if isinstance(row,dict):
        return {k:comparable(v)for k,v in row.items()if not(k=='pointer_delta'and row.get('pointer_delta_role')=='nonsharing-diagnostic-only')}
    if isinstance(row,list):return [comparable(v)for v in row]
    return row


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_saved_mark_public(spec,workers,owned,fp_backend):
    expected=SAVED[spec['id']];observe=legacy_observe if expected['format']=='legacy39bb'else typed_observe
    core=owned._core;old=core.get_num_threads();core.set_num_threads(workers);calls=[]
    def guard(frame,event,value):
        if event=='call':
            filename=frame.f_code.co_filename.replace('\\','/')
            if '/site-packages/skimage/'in filename or '/site-packages/scipy/ndimage/'in filename:raise AssertionError('ORIGINAL_PROVIDER_BODY_FORBIDDEN '+filename)
        if event=='c_return'and any(value is getattr(core,name)for name in ('mark_boundaries_prepare','mark_boundaries_mask','mark_boundaries_outline','mark_boundaries_paint')):calls.append(value.__name__)
    libc,helper=fp_backend
    def env():
        raw=(ctypes.c_ubyte*32)();assert libc.fegetenv(ctypes.byref(raw))==0;return raw
    def state(raw):
        b=bytes(raw);return dict(control=int.from_bytes(b[0:2],'little'),status=int.from_bytes(b[4:6],'little'),mxcsr=int.from_bytes(b[28:32],'little'))
    immediate=[];saved=env()
    def target(*args,**kwargs):
        # Establish the captured INPUT control/status/MXCSR on a current trusted
        # empty-stack fenv template. No production or AFTER flags are masked.
        incoming=env();expected_input=expected['record']['target_immediate']['before']
        for field,offset,width in [('control',0,2),('status',4,2),('mxcsr',28,4)]:incoming[offset:offset+width]=expected_input[field].to_bytes(width,'little')
        assert helper.reference_fenv_restore(ctypes.byref(incoming),32)==0
        before=env()
        try:return owned.mark_boundaries(*args,**kwargs)
        finally:immediate.append(dict(before=state(before),after=state(env())))
    prior=sys.getprofile();sys.setprofile(guard)
    try:actual=observe(target,spec,np)
    finally:
        sys.setprofile(prior);core.set_num_threads(old)
        assert helper.reference_fenv_restore(ctypes.byref(saved),32)==0 and bytes(env())==bytes(saved)
    expected_row={k:v for k,v in expected['record'].items()if k not in ('target_immediate','after_observer_fenv','caller_restore')}
    assert comparable(actual)==comparable(expected_row)
    if actual['error']is None:assert calls[0]=='mark_boundaries_prepare'and calls[-1]=='mark_boundaries_paint'
    assert core.get_num_threads()==old
    expected_fp=expected['record']['target_immediate']
    assert immediate==[{phase:{name:expected_fp[phase][name]for name in ('control','status','mxcsr')}for phase in ('before','after')}]


@pytest.mark.parametrize('dtype',('bool','uint8','int8','uint16','int16','uint32','int32','uint64','int64','float16','float32','float64','>i2'))
def test_binary_generic_dtype_native_owned(dtype,owned):
    import inspect
    image=np.arange(4,dtype=np.uint8).reshape(2,2).astype(dtype)
    labels=np.array([[0,1],[1,0]],dtype=np.uint8);labels.flags.writeable=False
    original=image.tobytes();out=owned.mark_boundaries(image,labels,mode='thick')
    assert out.dtype==np.dtype('float32'if np.dtype(dtype).kind=='f'and np.dtype(dtype).itemsize<=4 else'float64')
    assert out.shape==(2,2,3)and out.flags.owndata and not np.shares_memory(out,image)
    assert out.tolist()==[[[1,1,0],[1,1,0]],[[1,1,0],[1,1,0]]]and image.tobytes()==original
    assert inspect.isbuiltin(owned._core.mark_boundaries_prepare)and inspect.isbuiltin(owned._core.mark_boundaries_mask)


@pytest.mark.parametrize('kind',('extended-float','rank5','swapped-labels','nonexact-wide-labels'))
def test_explicit_unassessed_partitions_do_not_fallback(kind,owned):
    image=np.zeros((2,2),dtype=np.uint8);labels=np.array([[0,1],[1,0]],dtype=np.uint8)
    if kind=='extended-float':
        if np.dtype(np.longdouble).itemsize<=8:pytest.skip('Platform has no distinct extended-float dtype')
        image=image.astype(np.longdouble)
    elif kind=='rank5':image=np.zeros((1,1,1,1,3),dtype=np.float32)
    elif kind=='swapped-labels':labels=labels.astype('>u2')
    else:labels=np.array([[2**54,1],[1,0]],dtype=np.int64)
    with pytest.raises(NotImplementedError):owned.mark_boundaries(image,labels,mode='thick')
