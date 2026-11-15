"""Linux admitted94-row public-target consumer; no original resolution/call."""
import base64,ctypes,hashlib,importlib,json,os,sys,zlib
from pathlib import Path
import numpy as np
import pytest
from draw_multiblock_lbp_observer_v2 import observe,cases
from draw_multiblock_lbp_saved_parity import portable_record
ROOT=Path(__file__).resolve().parents[1]
T=json.loads((ROOT/'tests/fixtures/draw_multiblock_lbp_originals_v1.json').read_bytes())
adraw=zlib.decompress(base64.b64decode(T['admission_raw_zlib_base64']))
assert hashlib.sha256(adraw).hexdigest()==T['admission_raw_sha256']=='1bcfff2be1b241b989a6142a756131474e89e756dd30047b0e653dde3d822726'
phase=(ROOT/'coverage/contracts/draw-multiblock-lbp-observer-phase-scope-v1.json').read_bytes()
assert hashlib.sha256(phase).hexdigest()=='4c21baa192a904a957435cb2178f361cffb36d74a84399912daa98e5ef5a26e7'
assert json.loads(phase)['status']=='PUBLIC_API_FP_BOUNDARY_STRICT_POST_OBSERVER_FORMATTER_DIAGNOSTIC'
AD=json.loads(adraw);PROFILE='minimum'if np.__version__=='1.24.4'else'modern'
P=T['profiles'][PROFILE];assert np.__version__==P['numpy']
RAW=zlib.decompress(base64.b64decode(P['raw_zlib_base64']))
assert hashlib.sha256(RAW).hexdigest()==P['raw_sha256']==AD['raw_files'][f'build/draw-multiblock-lbp-codec-capture-v1/capture-v1/{PROFILE}/observations.json']['sha256']
GOLD=json.loads(RAW);SPECS=cases();assert len(GOLD)==len(SPECS)==94 and[r['id']for r in GOLD]==[s['id']for s in SPECS]

@pytest.fixture(scope='module')
def installed():
    import inspect,skimagexx as xx
    module=importlib.import_module('skimagexx._draw_multiblock_lbp')
    assert module._core is xx._core and xx.feature.draw_multiblock_lbp is module.draw_multiblock_lbp
    assert all(inspect.isbuiltin(getattr(xx._core,n))and getattr(xx._core,n).__module__=='skimagexx._core'for n in ('_lbp_prepare','_lbp_binary','_lbp_assign'))
    assert xx.get_num_threads()==1;xx.set_fallback_enabled(False)
    return xx,module

@pytest.fixture(scope='module')
def fenv():
    assert sys.platform=='linux','Linux captured full32 envelope is not a Windows parity consumer'
    assert os.environ.get('UTIL_VIEWS_FENV_MODE')=='source-built-v1','Genuine LBP requires actual-host source-built helper/receipt, no private fallback'
    from util_views_fenv import backend
    return backend()

def differences(a,b,path=''):
    if type(a)is not type(b):return[path+':type']
    if isinstance(a,dict):
        if a.keys()!=b.keys():return[path+':keys']
        return[d for k in a for d in differences(a[k],b[k],path+'/'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return[path+':length']
        return[d for i,(x,y)in enumerate(zip(a,b))for d in differences(x,y,path+'/'+str(i))]
    return[]if a==b else[path]

@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('index',range(94),ids=[s['id']for s in SPECS])
def test_saved_lbp_public(index,workers,installed,fenv):
    xx,module=installed;libc,helper=fenv;spec,gold=SPECS[index],GOLD[index]
    def env():
        value=(ctypes.c_ubyte*32)();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
    def state(value):
        b=bytes(value);return dict(raw=b.hex(),control=int.from_bytes(b[:2],'little'),status=int.from_bytes(b[4:6],'little'),mxcsr=int.from_bytes(b[28:32],'little'))
    def nstate():return dict(policy=np.geterr(),handler_type=[type(np.geterrcall()).__module__,type(np.geterrcall()).__name__],handler_id=id(np.geterrcall()),buffer=np.getbufsize())
    process=env();software=np.geterr(),np.geterrcall(),np.getbufsize();printing=np.get_printoptions();old=xx.get_num_threads();prior=sys.gettrace();pairs=[]
    def guard(frame,event,value):
        if event=='call'and frame.f_globals.get('__name__','').split('.')[0]in ('skimage','scipy'):
            raise AssertionError('ORIGINAL_PROVIDER_BODY_FORBIDDEN')
        return None
    # Trusted current empty-stack template, source-captured caller INPUT fields.
    # No product/AFTER state is cleared, masked, normalized or rewritten.
    saved=env();source=gold['caller_restore']['saved']
    for name,offset,width in (('control',0,2),('status',4,2),('mxcsr',28,4)):saved[offset:offset+width]=source[name].to_bytes(width,'little')
    restore(saved);xx.set_num_threads(workers)
    def call(*args,**options):
        before=nstate();raw=env()
        try:return module.draw_multiblock_lbp(*args,**options)
        finally:pairs.append(dict(before=state(raw),after=state(env()),numpy_before=before,numpy_after=nstate()))
    try:
        sys.settrace(guard)
        try:actual=observe(call,spec,np)
        finally:sys.settrace(prior)
        assert len(pairs)==1;actual['target_immediate']=pairs[0];actual['after_observer_fenv']=state(env())
        restore(saved);restored=env();assert bytes(saved)==bytes(restored)
        actual['caller_restore']=dict(saved=state(saved),actual=state(restored),exact32=True)
        paths=differences(portable_record(actual),portable_record(gold))
        # Explicit admitted outside-API collector diagnostic, retained in output.
        assert all(p=='/after_observer_fenv/mxcsr'for p in paths),paths
        assert all(actual['checks'].values())and actual['supplied_call_attempts']==1
        if os.environ.get('LBP_RESULT_DIR'):
            out=Path(os.environ['LBP_RESULT_DIR']);out.mkdir(parents=True,exist_ok=True)
            (out/(str(index)+'-workers'+str(workers)+'.json')).write_text(json.dumps(dict(record=actual,strict_full_record_match=not paths,collector_diagnostic_paths=paths),indent=2)+'\n')
    finally:
        sys.settrace(prior);xx.set_num_threads(old);np.seterr(**software[0]);np.seterrcall(software[1]);np.setbufsize(software[2]);np.set_printoptions(**printing);restore(process)
        assert bytes(env())==bytes(process)and xx.get_num_threads()==old
