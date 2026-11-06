"""All216 admitted rows, strict target FP and payload; no original calls."""
import base64,ctypes,hashlib,importlib,json,os,sys,types,zlib
from pathlib import Path
import numpy as np
import pytest
from finite_radon_contract import cases
from finite_radon_observer_v2 import observe
from finite_radon_saved_parity import portable_record
ROOT=Path(__file__).resolve().parents[1]
TRANSPORT=json.loads((ROOT/'tests/fixtures/finite_radon_originals_v1.json').read_bytes())
PROFILE='minimum'if np.__version__=='1.24.4'else'modern'
BLOB=TRANSPORT['profiles'][PROFILE];RAW=zlib.decompress(base64.b64decode(BLOB['zlib_base64']));assert hashlib.sha256(RAW).hexdigest()==BLOB['sha256']
GOLD={r['id']:r for r in json.loads(RAW)}

def differences(a,b,p=''):
    if type(a)!=type(b):return[p+':type']
    if isinstance(a,dict):
        if a.keys()!=b.keys():return[p+':keys']
        return[z for k in a for z in differences(a[k],b[k],p+'/'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return[p+':length']
        return[z for i,(x,y)in enumerate(zip(a,b))for z in differences(x,y,p+'/'+str(i))]
    return[]if a==b else[p]

@pytest.fixture(scope='module')
def runtime():
    from util_views_fenv import backend,FEnv
    directory=os.environ.get('FINITE_RADON_COMPONENT_DIR')
    if directory:
        os.environ['UTIL_VIEWS_FENV_MODE']='private-fixed-v1';os.environ['UTIL_VIEWS_FENV_HELPER']=os.environ['FINITE_RADON_FENV_HELPER']
        sys.path.insert(0,directory);core=importlib.import_module('_finite_radon_core');package=types.ModuleType('finite_radon_saved_component');package.__path__=[str(ROOT/'python/skimagexx')];sys.modules[package.__name__]=package;sys.modules[package.__name__+'._core']=core;owned=importlib.import_module(package.__name__+'._finite_radon')
    else:
        assert os.environ.get('UTIL_VIEWS_FENV_MODE')=='source-built-v1';owned=importlib.import_module('skimagexx._finite_radon')
    libc,helper=backend();records=[];out=Path(os.environ['FINITE_RADON_RECORDS'])
    yield owned,libc,helper,FEnv,records
    out.write_text(json.dumps(records,indent=2,allow_nan=False)+'\n',encoding='utf-8')

@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_saved_public_strict(spec,workers,runtime):
    owned,libc,helper,FEnv,records=runtime;core=owned._core;old_workers=core.get_num_threads();core.set_num_threads(workers);gold=GOLD[spec['id']];immediate=[];native=[];oldprofile=sys.getprofile()
    def env():
        f=FEnv();assert libc.fegetenv(ctypes.byref(f))==0;return f
    def restore(f):assert helper.reference_fenv_restore(ctypes.byref(f),32)==0
    def state(f):return dict(raw=bytes(f).hex(),control=f.control,status=f.status,mxcsr=f.mxcsr)
    def seed(expected):
        f=env();f.control=expected['control'];f.status=expected['status'];f.mxcsr=expected['mxcsr'];restore(f);return env()
    def nstate():return dict(policy=np.geterr(),handler_type=[type(np.geterrcall()).__module__,type(np.geterrcall()).__name__],handler_id=id(np.geterrcall()),buffer=np.getbufsize())
    def guard(frame,event,value):
        if event=='call':
            p=frame.f_code.co_filename.replace('\\','/')
            if '/site-packages/skimage/'in p or '/site-packages/scipy/ndimage/'in p:raise AssertionError('ORIGINAL_PROVIDER_BODY_FORBIDDEN '+p)
        if event=='c_call'and getattr(value,'__name__',None)=='sum'and isinstance(getattr(value,'__self__',None),np.ndarray):raise AssertionError('NUMPY_SUM_IMAGE_PROVIDER_FORBIDDEN')
        if event=='c_return'and any(value is getattr(core,n)for n in('frt2','ifrt2','_finite_radon_inverse_prefix')):native.append(value.__name__)
    external=env();saved=seed(gold['caller_restore']['saved']);old_np=np.geterr(),np.geterrcall(),np.getbufsize();printing=np.get_printoptions();row=None
    try:
        sys.setprofile(guard)
        def supplied(*args,**kwargs):
            seed(gold['target_immediate']['before']);before=nstate();hardware=env()
            try:return getattr(owned,spec['api'])(*args,**kwargs)
            finally:immediate.append(dict(before=state(hardware),after=state(env()),numpy_before=before,numpy_after=nstate()))
        row=observe(supplied,spec,np);assert len(immediate)==1
        row['target_immediate']=immediate[0];row['after_observer_fenv']=state(env())
    finally:
        sys.setprofile(oldprofile);np.seterr(**old_np[0]);np.seterrcall(old_np[1]);np.setbufsize(old_np[2]);np.set_printoptions(**printing);restore(saved);actual=env()
        caller=dict(saved=state(saved),actual=state(actual),exact32=bytes(saved)==bytes(actual));restore(external);assert bytes(env())==bytes(external);core.set_num_threads(old_workers)
    assert caller['exact32'];row['caller_restore']=caller
    ds=differences(portable_record(row),portable_record(gold));records.append(dict(id=spec['id'],workers=workers,actual=row,differences=ds,native_returns=native))
    assert not ds,';'.join(ds)
    if row['error']is None:assert native,'Expected actual owned builtin return'
