"""Pure hosted IGG3 source/inert controls; scientific requests are refused."""
import ast,hashlib,importlib.util,json,os,subprocess,sys,types,zipfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'tools/igg_host_native_diagnostic.py'
def tool():
    spec=importlib.util.spec_from_file_location('igg_host_diag',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_frozen_source_input_runtime_and_budget_closure():
    c=tool().contract()
    assert len(c['wheels'])==14 and sum(r['name']!='scikit-imagexx'for r in c['wheels'].values())==13
    numpy=next(r for r in c['wheels'].values()if r['name']=='numpy')
    assert numpy['version']=='1.24.4'and numpy['sha256']=='7215847ce88a85ce39baf9e89070cb860c98fdddacbaa6c0da3ffb31b3350bd5'
    assert c['bounds']['owned_topcalls']==3 and c['bounds']['image_core_calls']==15
    assert c['bounds']['originals']==c['bounds']['timings']==0
    assert c['case_ids']==['dtype-float64','dtype-complex128','image-list']
    assert c['qualified']['revision']=='852cc1882fbd757f55bc9f92eb783282e2f5addb'
    assert c['stager']['revision']=='dda9cc83d9434a10616f84cf1e1125d93f5f36fd'
    assert all(v is None for v in c['authority'].values())
    for name,pin in c['source_pins'].items():
        if 'fixtures/'not in name:assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin


def test_exact_binding_allocates_outputs_and_cannot_write_retained_input_buffers():
    binder=(ROOT/'src/bind_inverse_gaussian_gradient.cpp').read_text()
    header=(ROOT/'include/skimagexx/inverse_gaussian_gradient.hpp').read_text()
    assert binder.count('py::array output(')==2 and binder.count('output.mutable_data()')==2
    for name in('image','gradient','alpha'):assert name+'.mutable_data()'not in binder
    assert 'image.data(),output.mutable_data()'in binder and 'gradient.data(),alpha.data(),output.mutable_data()'in binder
    assert '{w.data(),w.data()+w.size()}'in binder
    assert 'magnitude(const void*,void*'in header and 'finish(const void*,const void*,void*'in header
    tree=ast.parse((ROOT/'python/skimagexx/_inverse_gaussian_gradient.py').read_bytes())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='inverse_gaussian_gradient')
    assert not any(isinstance(n,ast.AugAssign)or isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)for t in n.targets)for n in ast.walk(fn))
    calls=[n for n in ast.walk(fn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='_core']
    assert sorted(n.func.attr for n in calls)==['get_num_threads','inverse_gaussian_finish','inverse_gaussian_magnitude']


class Buffer:
    def __init__(self,name):self.name=name
    def __array__(self,*a,**k):raise AssertionError('NO_COERCION')
    def __bytes__(self):raise AssertionError('NO_BYTES_IN_TARGET')
    def copy(self):raise AssertionError('NO_COPY')
    def tobytes(self):raise AssertionError('NO_BYTES_IN_TARGET')


def test_actual_wrappers_retain_reference_identity_without_extra_native_or_encoding():
    m=tool();state=[bytes(32)];calls=[];born=[]
    image=Buffer('image');weights=Buffer('coefficients');plans=[[(0,weights)],[(1,weights)]]
    def magnitude(a,p,w):
        assert a is image and p is plans and w==1;calls.append('magnitude');out=Buffer('magnitude');born.append(out);state[0]=b'M'*32;return out
    def finish(g,a,shape,stage,w):
        assert g is born[-1]and a is alpha and shape==[3,4]and w==1
        calls.append(stage);out=Buffer('stage'+str(stage));born.append(out);state[0]=bytes([stage])*32;return out,stage
    alpha=Buffer('alpha');core=types.SimpleNamespace(inverse_gaussian_magnitude=magnitude,inverse_gaussian_finish=finish)
    refs=m.StageReferences(lambda:state[0]);encoded=[]
    with refs.installed(core):
        value=core.inverse_gaussian_magnitude(image,plans,1)
        for stage in range(4):value,flags=core.inverse_gaussian_finish(value,alpha,[3,4],stage,1)
        with pytest.raises(AssertionError,match='TARGET_IMMEDIATE_FP_NOT_SAVED'):refs.encode(encoded.append)
        assert not encoded
        with pytest.raises(AssertionError,match='UNEXPECTED_IMAGE_ENTRY'):core.inverse_gaussian_finish(value,alpha,[3,4],4,1)
    assert core.inverse_gaussian_magnitude is magnitude and core.inverse_gaussian_finish is finish
    assert calls==['magnitude',0,1,2,3]and state[0]==b'\x03'*32
    assert refs.refs[0][1]is image and refs.refs[0][2]is plans and refs.refs[0][3]is born[0]
    for i,row in enumerate(refs.refs[1:]):assert row[2]is born[i]and row[4][0]is born[i+1]
    refs.target_state_saved();refs.encode(encoded.append);assert len(encoded)==5
    assert [x.name for x in born]==['magnitude','stage0','stage1','stage2','stage3']


@pytest.mark.parametrize('fault',('native_error','observer_FP','extra_magnitude'))
def test_wrappers_restore_live_entries_and_refuse_errors_without_normalizing_FP(fault):
    m=tool();state=[bytes(32)];count=[]
    def magnitude(*a):
        count.append(1);state[0]=b'N'*32
        if fault=='native_error':raise RuntimeError('native-failure')
        return Buffer('result')
    finish=lambda *a:None;core=types.SimpleNamespace(inverse_gaussian_magnitude=magnitude,inverse_gaussian_finish=finish)
    refs=m.StageReferences(lambda:state[0])
    if fault=='observer_FP':
        class Changed(list):
            def append(self,value):super().append(value);state[0]=b'X'*32
        refs.refs=Changed()
    with pytest.raises((RuntimeError,AssertionError)):
        with refs.installed(core):
            core.inverse_gaussian_magnitude(None,None,1)
            if fault=='extra_magnitude':core.inverse_gaussian_magnitude(None,None,1)
    assert core.inverse_gaussian_magnitude is magnitude and core.inverse_gaussian_finish is finish
    assert len(count)==1 and state[0]==(b'X'*32 if fault=='observer_FP'else b'N'*32)


def test_actual_run_defers_stage_encoding_until_recorded_target_FP_and_restored_caller():
    tree=ast.parse(PATH.read_bytes());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');text=ast.get_source_segment(PATH.read_text(),run)
    assert text.index('validate_go(go)')<text.index('import numpy as np')
    assert text.index("immediate.append(")<text.index('actual=observe(')<text.index('stages.target_state_saved()')<text.index('stages.encode(')
    assert 'finally:restore(caller);assert bytes(env())==bytes(caller)'in text
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='StageReferences')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='installed')
    fragment=ast.get_source_segment(PATH.read_text(),fn)
    for forbidden in('np.','.copy(','.tobytes(','.astype(','.encode(','.restore('):assert forbidden not in fragment
    assert fragment.count('self.state()==after')==2
    assert 'partial-observation.json'in text and 'terminal-fallback.json'in text


@pytest.mark.parametrize('cached',(False,True))
@pytest.mark.parametrize('failure',('missing_GO','unbound','unknown_case'))
def test_actual_CLI_refuses_before_scientific_request_cold_cached(tmp_path,cached,failure):
    go=tmp_path/'go.json';go.write_text(json.dumps({'status':None}if failure!='unknown_case'else{'status':'ROOT_AUTHORIZED_HOSTED_IGG_THREE_ONCE','contract_sha256':hashlib.sha256((ROOT/'coverage/contracts/igg-host-native-diagnostic-v1.json').read_bytes()).hexdigest(),'source':'852cc1882fbd757f55bc9f92eb783282e2f5addb','case_ids':['unknown']}))
    args=['run','--go',str(go if failure!='missing_GO'else tmp_path/'absent'),'--inputs',str(tmp_path),'--runtime',str(tmp_path),'--output',str(tmp_path/'out')]
    code="""import builtins,sys,runpy,types
old=builtins.__import__
def refuse(n,*a,**k):
 if n.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise AssertionError('SCIENTIFIC_REQUEST')
 return old(n,*a,**k)
if sys.argv[2]=='True':sys.modules['numpy']=types.ModuleType('numpy')
builtins.__import__=refuse;sys.argv=[sys.argv[1],*sys.argv[3:]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-I','-c',code,str(PATH),str(cached),*args],capture_output=True,text=True)
    assert p.returncode!=0 and 'SCIENTIFIC_REQUEST'not in p.stderr
    assert not(tmp_path/'out').exists()


@pytest.mark.parametrize('fault',('none','digest','source','size','second_download'))
def test_actual_archive_receipt_boundary_is_exact_without_digest_replay(tmp_path,fault):
    m=tool();path=tmp_path/'small.zip'
    with zipfile.ZipFile(path,'w')as z:z.writestr('tiny.txt','no-science')
    expected=dict(id=1,run=2,revision='a'*40,bytes=path.stat().st_size,sha256='b'*64)
    r=dict(expected,actual_API_digest_exact=True,download_passes=1)
    if fault=='digest':r['sha256']='changed'
    if fault=='source':r['revision']='changed'
    if fault=='size':r['bytes']+=1
    if fault=='second_download':r['download_passes']=2
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(r))
    if fault=='none':
        with m.checked_archive(path,receipt,expected)as z:assert z.read('tiny.txt')==b'no-science'
    else:
        with pytest.raises(AssertionError):m.checked_archive(path,receipt,expected)


@pytest.mark.parametrize('fault',('local','wrong_run','wrong_source','wrong_digest'))
def test_actual_fetch_refuses_before_download_and_never_restarts(tmp_path,monkeypatch,fault):
    m=tool();c=m.contract();e=c['stager'];events=[]
    monkeypatch.setenv('GITHUB_ACTIONS','true'if fault!='local'else'false')
    api=dict(id=e['id'],name=e['name'],expired=False,size_in_bytes=e['bytes'],digest='sha256:'+e['sha256'],workflow_run=dict(id=e['run'],head_sha=e['revision']))
    if fault=='wrong_run':api['workflow_run']['id']+=1
    if fault=='wrong_source':api['workflow_run']['head_sha']='changed'
    if fault=='wrong_digest':api['digest']='changed'
    def metadata(argv):events.append(argv);return json.dumps(api).encode()
    monkeypatch.setattr(m.subprocess,'check_output',metadata)
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:pytest.fail('DOWNLOAD_FORBIDDEN'))
    with pytest.raises(AssertionError):m.fetch(types.SimpleNamespace(output=tmp_path/'new'))
    assert len(events)==(0 if fault=='local'else 1)


@pytest.mark.parametrize('fault',('none','threads','forced_CPU','not_Actions','wrong_OS','dirty','HEAD'))
def test_actual_GO_environment_boundary_positive_and_refusals(monkeypatch,fault):
    m=tool();c=m.contract();c['source_pins']={} # only metadata boundary, no fixture/gold hashing replay
    monkeypatch.setattr(m,'contract',lambda:c)
    go=dict(status='ROOT_AUTHORIZED_HOSTED_IGG_THREE_ONCE',source=c['qualified']['revision'],diagnostic_source='a'*40,case_ids=c['case_ids'],owned_topcalls=3,max_image_core_calls=15,original_calls=0,timing=0,helper_receipt='/absolute/new-host/receipt.json',helper_receipt_sha256='b'*64,contract_sha256=hashlib.sha256(m.CONTRACT.read_bytes()).hexdigest())
    for n in('LD_PRELOAD','LD_AUDIT','NPY_DISABLE_CPU_FEATURES','OPENBLAS_CORETYPE','SKIMAGEXX_DISABLE_SIMD','UTIL_VIEWS_FENV_HELPER'):monkeypatch.delenv(n,raising=False)
    for n in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'):monkeypatch.setenv(n,'1')
    for n,v in dict(GITHUB_ACTIONS='true',GITHUB_SHA='a'*40,UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=go['helper_receipt'],UTIL_VIEWS_FENV_RECEIPT_SHA256=go['helper_receipt_sha256']).items():monkeypatch.setenv(n,v)
    monkeypatch.setattr(m,'sys',types.SimpleNamespace(version_info=(3,11,17),platform='linux'))
    monkeypatch.setattr(m.platform,'freedesktop_os_release',lambda:dict(ID='ubuntu',VERSION_ID='22.04'if fault=='wrong_OS'else'24.04'))
    def git(argv,**kw):return ('b'*40 if fault=='HEAD'else'a'*40)if'rev-parse'in argv else('dirty'if fault=='dirty'else'')
    monkeypatch.setattr(m.subprocess,'check_output',git)
    if fault=='threads':monkeypatch.setenv('OPENBLAS_NUM_THREADS','4')
    if fault=='forced_CPU':monkeypatch.setenv('NPY_DISABLE_CPU_FEATURES','AVX2')
    if fault=='not_Actions':monkeypatch.setenv('GITHUB_ACTIONS','false')
    if fault=='none':m.validate_go(go)
    else:
        with pytest.raises(AssertionError):m.validate_go(go)


@pytest.mark.parametrize('writer_fails',(False,True))
def test_actual_main_preserves_early_primary_and_distinct_fallback(tmp_path,monkeypatch,capsys,writer_fails):
    m=tool();out=tmp_path/'result'
    def failed(args):out.mkdir();raise RuntimeError('early-runtime-import-or-helper-failure')
    monkeypatch.setattr(m,'run',failed)
    if writer_fails:
        def refused(*args):raise OSError('writer-failure')
        monkeypatch.setattr(m,'dump',refused)
    with pytest.raises(RuntimeError,match='early-runtime-import-or-helper-failure'):
        m.main(['run','--go','unbound','--inputs','unused','--runtime','unused','--output',str(out)])
    if writer_fails:
        text=capsys.readouterr().err;assert 'writer-failure'in text and 'early-runtime-import-or-helper-failure'in text
    else:
        d=json.loads((out/'outer-failure.json').read_bytes());assert d['actual_consumed_calls']is None and d['primary']['type']=='RuntimeError'


def test_actual_main_does_not_append_diagnostics_to_an_existing_output(tmp_path,monkeypatch):
    m=tool();out=tmp_path/'frozen';out.mkdir();(out/'terminal.json').write_text('old')
    def failed(args):raise RuntimeError('existing-output')
    monkeypatch.setattr(m,'run',failed)
    with pytest.raises(RuntimeError,match='existing-output'):m.main(['run','--go','unbound','--inputs','unused','--runtime','unused','--output',str(out)])
    assert sorted(p.name for p in out.iterdir())==['terminal.json']and(out/'terminal.json').read_text()=='old'
