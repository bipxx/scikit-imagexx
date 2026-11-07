"""Three-case hosted IGG diagnostic, never qualification or measurement."""
from pathlib import Path
import argparse,builtins,contextlib,ctypes,datetime,hashlib,importlib,importlib.util,json,os,platform,subprocess,sys,types,zipfile
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/igg-host-native-diagnostic-v1.json'
H=lambda b:hashlib.sha256(b).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def contract():return json.loads(CONTRACT.read_bytes())


class StageReferences:
    """Only references retained between native return and target FP capture."""
    def __init__(self,state):self.state=state;self.refs=[];self.sealed=False
    @contextlib.contextmanager
    def installed(self,core):
        magnitude,finish=core.inverse_gaussian_magnitude,core.inverse_gaussian_finish
        def observed_magnitude(image,plans,workers):
            assert not self.refs,'UNEXPECTED_IMAGE_ENTRY'
            result=magnitude(image,plans,workers);after=self.state()
            self.refs.append(('magnitude',image,plans,result))
            assert self.state()==after,'STAGE_OBSERVER_CHANGED_FP'
            return result
        def observed_finish(gradient,alpha,shape,stage,workers):
            assert type(stage)is int and 0<=stage<4 and len(self.refs)==stage+1,'UNEXPECTED_IMAGE_ENTRY'
            result=finish(gradient,alpha,shape,stage,workers);after=self.state()
            self.refs.append(('finish',stage,gradient,alpha,result))
            assert self.state()==after,'STAGE_OBSERVER_CHANGED_FP'
            return result
        core.inverse_gaussian_magnitude=observed_magnitude;core.inverse_gaussian_finish=observed_finish
        try:yield self
        finally:core.inverse_gaussian_magnitude=magnitude;core.inverse_gaussian_finish=finish
    def target_state_saved(self):
        assert len(self.refs)==5 and [r[1]for r in self.refs[1:]]==[0,1,2,3]
        self.sealed=True
    def encode(self,encode):
        assert self.sealed,'TARGET_IMMEDIATE_FP_NOT_SAVED'
        return [encode(row)for row in self.refs]


def checked_archive(path,receipt,expected):
    r=json.loads(Path(receipt).read_bytes());p=Path(path)
    assert r['actual_API_digest_exact']is True and r['download_passes']==1
    for key in('id','run','revision','bytes','sha256'):assert r[key]==expected[key],key
    assert p.stat().st_size==r['bytes']
    return zipfile.ZipFile(p)


def fetch(args):
    """Existing once-streamed retained-reader loop, fixed two-input scope."""
    assert os.environ.get('GITHUB_ACTIONS')=='true','Hosted input preparation only'
    c=contract();out=Path(args.output);out.mkdir(exist_ok=False)
    for kind in('stager','qualified'):
        expected=c[kind];endpoint='repos/bipxx/scikit-imagexx/actions/artifacts/'+str(expected['id'])
        meta=json.loads(subprocess.check_output(['gh','api',endpoint]))
        dump(out/(str(expected['id'])+'-API.json'),meta)
        assert meta['id']==expected['id']and meta['name']==expected['name']and not meta['expired']
        assert meta['workflow_run']['id']==expected['run']and meta['workflow_run']['head_sha']==expected['revision']
        assert meta['size_in_bytes']==expected['bytes']and meta['digest']=='sha256:'+expected['sha256']
        archive=out/(str(expected['id'])+'.zip');digest=hashlib.sha256();size=0
        with archive.open('xb')as destination:
            process=subprocess.Popen(['gh','api',endpoint+'/zip'],stdout=subprocess.PIPE)
            for data in iter(lambda:process.stdout.read(1024*1024),b''):
                destination.write(data);digest.update(data);size+=len(data)
            code=process.wait()
        assert code==0 and size==expected['bytes']and digest.hexdigest()==expected['sha256']
        dump(out/(str(expected['id'])+'-retention.json'),dict(expected,path=str(archive),actual_API_digest_exact=True,download_passes=1))


def intake(args):
    c=contract();out=Path(args.output);out.mkdir(exist_ok=False)
    with checked_archive(args.stager,args.stager_receipt,c['stager'])as z:
        raw=z.read('runtime.json');assert H(raw)==c['runtime_json_sha256'];runtime=json.loads(raw)
        assert runtime['source']['revision']==c['stager']['revision']and runtime['wheels']==c['wheels']
        dump(out/'runtime.json',runtime);wheels=out/'wheels';wheels.mkdir()
        for name,row in c['wheels'].items():
            body=z.read('wheels/'+name);assert H(body)==row['sha256'],name
            (wheels/name).write_bytes(body)
    with checked_archive(args.qualified,args.qualified_receipt,c['qualified'])as z:
        raw=z.read(c['qualified_prefix']+'manifest.json');assert H(raw)==c['qualified_manifest_sha256'];m=json.loads(raw)
        assert m['passed']is False and m['source']['revision']==c['qualified']['revision']
        core=out/'core';core.mkdir()
        for name,row in c['core_members'].items():
            info=z.getinfo(c['qualified_prefix']+name);assert info.file_size==row['bytes']and m['records'][name]==row
            body=z.read(info);assert len(body)==row['bytes']
            (core/Path(name).name).write_bytes(body) # ZIP CRC + once-streamed full identity, no secondcore seal
    dump(out/'intake.json',dict(status='DIAGNOSTIC_INPUTS_ONLY',source=c['qualified']['revision'],runtime_source=c['stager']['revision'],wheels=c['wheels'],core_members=c['core_members'],whole_product_admission=False))


def validate_go(go):
    c=contract()
    assert go['status']=='ROOT_AUTHORIZED_HOSTED_IGG_THREE_ONCE'
    assert go['contract_sha256']==H(CONTRACT.read_bytes())
    assert go['source']==c['qualified']['revision']and go['case_ids']==c['case_ids']
    assert go['owned_topcalls']==3 and go['max_image_core_calls']==15 and go['original_calls']==go['timing']==0
    assert go['helper_receipt']and go['helper_receipt_sha256']
    assert os.environ['UTIL_VIEWS_FENV_MODE']=='source-built-v1'and 'UTIL_VIEWS_FENV_HELPER'not in os.environ
    assert os.environ['UTIL_VIEWS_FENV_RECEIPT']==go['helper_receipt']and os.environ['UTIL_VIEWS_FENV_RECEIPT_SHA256']==go['helper_receipt_sha256']
    assert os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('GITHUB_SHA')==go['diagnostic_source']
    assert all(os.environ.get(n)=='1'for n in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'))
    assert not any(os.environ.get(n)for n in('LD_PRELOAD','LD_AUDIT','NPY_DISABLE_CPU_FEATURES','OPENBLAS_CORETYPE','SKIMAGEXX_DISABLE_SIMD'))
    assert sys.version_info[:3]==(3,11,17)and sys.platform=='linux'
    release=platform.freedesktop_os_release();assert release['ID']=='ubuntu'and release['VERSION_ID']=='24.04'
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==go['diagnostic_source']
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    for name,pin in c['source_pins'].items():assert H((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))==pin,name


def run(args):
    R=ROOT;B=Path(args.inputs);W=ROOT
    IDS=('dtype-float64','dtype-complex128','image-list')
    go=json.loads(Path(args.go).read_bytes());validate_go(go)
    assert go['status']=='ROOT_AUTHORIZED_HOSTED_IGG_THREE_ONCE'
    assert go['source']=='852cc1882fbd757f55bc9f92eb783282e2f5addb'
    assert go['case_ids']==list(IDS)and go['owned_topcalls']==3 and go['original_calls']==0
    assert go['script_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(go['deadline_utc'].replace('Z','+00:00'))
    assert go['core_sha256']=='ddcc9e141d3b6b9a2a5c4979ce6bc1877c7391b261cece3673bee4c7eae4732a'
    admission=json.loads((B/'intake.json').read_bytes());c=contract();assert admission['source']==c['qualified']['revision']and admission['wheels']==c['wheels']and admission['core_members']==c['core_members']
    O=Path(args.output);O.mkdir(exist_ok=False)
    old_import=builtins.__import__
    def refuse(name,*a,**k):
     if name.split('.')[0]in('skimage','scipy'):raise AssertionError('ORIGINAL_PROVIDER_REQUEST_FORBIDDEN '+name)
     return old_import(name,*a,**k)
    builtins.__import__=refuse
    sys.path.insert(0,str(Path(args.runtime).resolve()))
    import numpy as np
    assert np.__version__=='1.24.4'and Path(np.__file__).resolve().is_relative_to(Path(args.runtime).resolve())
    sys.path[:0]=[str(W/'tests')]
    from inverse_gaussian_gradient_contract import recipes
    from inverse_gaussian_gradient_observer import observe,encode
    from util_views_fenv import FEnv,backend
    libc,helper=backend()
    def env():
     v=FEnv();assert libc.fegetenv(ctypes.byref(v))==0;return v
    def restore(v):assert helper.reference_fenv_restore(ctypes.byref(v),32)==0
    initial=env();rows=[];calls=0;primary=None;process_restored=False
    try:
     package=types.ModuleType('_igg_qualified');package.__path__=[str(B/'core')];sys.modules[package.__name__]=package
     path=B/'core/_core.cpython-311-x86_64-linux-gnu.so'
     assert path.stat().st_size==8073648
     spec=importlib.util.spec_from_file_location(package.__name__+'._core',path)
     core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);package._core=core;sys.modules[spec.name]=core
     own=importlib.import_module(package.__name__+'._inverse_gaussian_gradient')
     assert core.get_num_threads()==1
     from numpy.core import _multiarray_umath
     dump(O/'host.json',dict(python=sys.version,numpy=np.__version__,numpy_origin=np.__file__,platform=platform.platform(),libc=platform.libc_ver(),cpu=Path('/proc/cpuinfo').read_text(),numpy_cpu_features=_multiarray_umath.__cpu_features__,blas_configuration={k:np.__config__.get_info(k)for k in('blas_opt_info','blas_ilp64_opt_info','openblas64__info')},loaded_libraries=sorted({line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if '.so'in line}),new_VM_not_retired_failure_VM=True))
     saved={r['id']:r for r in json.loads((W/'tests/fixtures/inverse_gaussian_gradient_original_v1.json').read_bytes())['profiles']['minimum']['rows']}
     chosen=[s for s in recipes()if s['id']in IDS];assert tuple(s['id']for s in chosen)==IDS
     for case in chosen:
      assert case['shape']==[3,4]and case['sigma']==1.0 and case['alpha']==100.0
      expected=saved[case['id']];caller=env();immediate=[];stages=StageReferences(lambda:bytes(env()))
      def target(*args,**options):
       nonlocal calls
       assert calls<3;calls+=1
       incoming=env()
       for key in('control','status','mxcsr'):setattr(incoming,key,expected['target_immediate']['before'][key])
       restore(incoming);before=env()
       try:
        with stages.installed(core):return own.inverse_gaussian_gradient(*args,**options)
       finally:immediate.append(dict(before={k:getattr(before,k)for k in('control','status','mxcsr')},after={k:getattr(env(),k)for k in('control','status','mxcsr')}))
      try:actual=observe(target,case,np)
      finally:restore(caller);assert bytes(env())==bytes(caller)
      try:stages.target_state_saved()
      except BaseException:
       dump(O/'partial-observation.json',dict(id=case['id'],actual=actual,expected=expected,immediate=immediate,stage_count=len(stages.refs)))
       raise
      try:encoded_stages=stages.encode(lambda row:encode(row,np))
      finally:restore(caller);assert bytes(env())==bytes(caller)
      differences=[k for k in actual if actual[k]!=expected[k]]
      if immediate!=[{phase:{k:expected['target_immediate'][phase][k]for k in('control','status','mxcsr')}for phase in('before','after')}]:differences.append('target_semantic_FP')
      rows.append(dict(id=case['id'],passed=not differences,differences=differences,actual=actual,immediate=immediate,expected=expected,caller32=True,stages=encoded_stages))
      (O/'rows.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
      assert actual['error']==expected['error'],'Unexpected native exception; stop remaining cases'
    except BaseException as e:
     primary=dict(type=type(e).__qualname__,message=str(e));raise
    finally:
     try:restore(initial);assert bytes(env())==bytes(initial);process_restored=True
     finally:
      builtins.__import__=old_import
      terminal=dict(source=go['source'],calls=calls,rows=len(rows),failures=[dict(id=r['id'],fields=r['differences'])for r in rows if not r['passed']],primary=primary,original_calls=0,timing=0,process32=process_restored,scope='Actual852 core, private diagnostic namespace on new hosted minimumVM; not oldVM/productqualification.')
      try:dump(O/'terminal.json',terminal)
      except BaseException as writer:
       dump(O/'terminal-fallback.json',dict(terminal=terminal,writer_error=dict(type=type(writer).__qualname__,message=str(writer))))
       raise
      print(json.dumps(terminal))
    return int(bool(terminal['failures']))


def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('plan')
    q=sub.add_parser('fetch');q.add_argument('--output',required=True)
    q=sub.add_parser('intake')
    for n in('stager','stager-receipt','qualified','qualified-receipt','output'):q.add_argument('--'+n,required=True)
    q=sub.add_parser('run')
    for n in('go','inputs','runtime','output'):q.add_argument('--'+n,required=True)
    a=p.parse_args(argv)
    if a.mode=='plan':print(json.dumps(contract(),indent=2));return 0
    if a.mode=='fetch':fetch(a);return 0
    if a.mode=='intake':intake(a);return 0
    existed=Path(a.output).exists()
    try:return run(a)
    except BaseException as error:
        out=Path(a.output)
        if not existed and out.is_dir():
            diagnostic=dict(primary=dict(type=type(error).__qualname__,message=str(error)),scope='Actual run failed; consult terminal/rows if present. Startup before helper snapshot has no process32 guarantee. No inferred consumed count or successful result.',maximum_owned_calls=3,actual_consumed_calls=None)
            try:dump(out/'outer-failure.json',diagnostic)
            except BaseException as writer:print(json.dumps(dict(primary=diagnostic['primary'],outer_writer_error=dict(type=type(writer).__qualname__,message=str(writer)))),file=sys.stderr)
        raise
if __name__=='__main__':raise SystemExit(main())
