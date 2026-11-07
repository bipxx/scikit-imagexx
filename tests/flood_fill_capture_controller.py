"""Fixed flood/fill one-shot capture; default prefix has zero original authority."""
import argparse,base64,ctypes,hashlib,importlib,importlib.metadata,inspect,json,os,platform,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
R=Path('/mnt/c/Users/HEQ/Projects/scikit-imagexx')
W=Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
LIMITS=('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')
class EntryRefused(BaseException):pass
class FEnv(ctypes.Structure):
    _fields_=[('control',ctypes.c_ushort),('reserved1',ctypes.c_ushort),('status',ctypes.c_ushort),('reserved2',ctypes.c_ushort),('tags',ctypes.c_ushort),('reserved3',ctypes.c_ushort),('eip',ctypes.c_uint32),('cs',ctypes.c_ushort),('opcode',ctypes.c_ushort),('data',ctypes.c_uint32),('ds',ctypes.c_ushort),('reserved5',ctypes.c_ushort),('mxcsr',ctypes.c_uint32)]

def dump(path,value):
    with path.open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def append(path,value):
    with path.open('a',encoding='utf-8',newline='\n')as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

def np_state(np):
    return dict(policy=np.geterr(),handler_type=[type(np.geterrcall()).__module__,type(np.geterrcall()).__name__],handler_id=id(np.geterrcall()),buffer=np.getbufsize())

def check_deadline(go):
    now=datetime.now(timezone.utc);start=datetime.fromisoformat(go['not_before_utc']);end=datetime.fromisoformat(go['deadline_utc'])
    assert start.utcoffset()is not None and end.utcoffset()is not None and start<=now<end,'GO_DEADLINE_EXPIRED_OR_NOT_ACTIVE'

def validate(path,digest,profile):
    assert path.is_file(),'MISSING_BINDING'
    assert H(path)==digest,'BINDING_DIGEST_MISMATCH'
    q=json.loads(path.read_bytes())
    assert q['schema']=='flood-fill-capture-v1'and q['status']=='UNBOUND_CAPTURE_CAPABLE_ZERO_BUDGET'
    assert all(type(q[n])is int and q[n]==0 for n in('original_budget','native_image_budget','sampler_budget','observation_budget'))
    assert q['root_original_GO']is None
    ids=q['case_ids'];assert len(ids)==len(set(ids))==96 and all(type(x)is str for x in ids),'CASE_POOL_INVALID'
    assert q['api_case_ids']=={'flood':ids[:45],'flood_fill':ids[45:]}
    assert all(x.startswith('flood:')for x in ids[:45])and all(x.startswith('flood_fill:')for x in ids[45:])
    assert q['profile_case_ids']=={'minimum':ids,'modern':ids},'FIXED_PROFILE_WHITELIST_CHANGED'
    assert q['controller_raw_sha256']==H(Path(__file__)),'CONTROLLER_PIN_MISMATCH'
    assert profile in q['profiles']and all(os.environ.get(n)=='1'for n in LIMITS)
    assert not os.environ.get('GIT_DIR')and not os.environ.get('GIT_WORK_TREE')
    assert subprocess.check_output([*q['git_argv'],'rev-parse','HEAD'],text=True).strip()==q['source']
    assert not subprocess.check_output([*q['git_argv'],'status','--porcelain'],text=True).strip()
    for name,pin in q['source_files_raw_sha256'].items():assert H(W/name)==pin,'SOURCE_PIN_MISMATCH '+name
    item=q['safety'];assert H(R/item['path'])==item['sha256'];s=json.loads((R/item['path']).read_bytes())
    assert s['status']=='PASS_FIXED_RECIPE_SOURCE_GEOMETRY_AND_STORAGE_NO_ORIGINAL_AUTHORITY'and s['source']==q['frozen_case_source']
    assert s['scope']['recipes']==96 and s['scope']['recipe_counts']=={'flood':45,'flood_fill':51}
    item=q['helper']['admission'];assert H(R/item['path'])==item['sha256']and json.loads((R/item['path']).read_bytes())['passed']
    for n in('source','compiler_receipt'):assert H(R/q['helper'][n]['path'])==q['helper'][n]['sha256']
    return q

def validate_go(args,q):
    assert args.root_go is not None and args.root_go_sha256 is not None,'NO_ROOT_GO'
    assert args.root_go.is_file()and H(args.root_go)==args.root_go_sha256,'ROOT_GO_DIGEST_MISMATCH'
    go=json.loads(args.root_go.read_bytes());assert go['status']=='ROOT_AUTHORIZED_FLOOD_FILL_REFERENCE_ONCE'
    assert go['source']==q['source']and go['binding_sha256']==args.binding_sha256 and go['controller_sha256']==q['controller_raw_sha256']
    assert go['source_files_raw_sha256']==q['source_files_raw_sha256']and go['cases_sha256']==q['cases_sha256']
    assert go['case_ids']==q['profile_case_ids'][args.profile]and go['api_case_ids']==q['api_case_ids']
    assert go['profiles']==['minimum','modern']and go['profile']==args.profile
    assert type(go['maximum_public_topcalls_per_profile'])is int and go['maximum_public_topcalls_per_profile']==96
    assert type(go['maximum_total_public_topcalls'])is int and go['maximum_total_public_topcalls']==192
    assert type(go['calls_per_recipe'])is int and go['calls_per_recipe']==1
    assert all(type(go[n])is int and go[n]==0 for n in('native_calls','sampler_bodies','timed_batches','builds'))
    assert Path(go['journal_directory']).resolve()==args.output.resolve()and go['no_retry']is True
    check_deadline(go)
    item=go['independent_readiness'];assert H(R/item['path'])==item['sha256'];review=json.loads((R/item['path']).read_bytes())
    assert review['passed']is True and review['source']==q['source']and review['controller_sha256']==q['controller_raw_sha256']and review['binding_sha256']==args.binding_sha256
    assert len(go['prefixes'])==4
    seen=set()
    for item in go['prefixes']:
        assert H(R/item['path'])==item['sha256'];p=json.loads((R/item['path']).read_bytes())
        assert p['status']=='PASS_ACTUAL_FLOOD_FILL_CONTROLLER_ZERO_BODY_PREFIX'and p['profile']==args.profile
        assert p['source']==q['source']and p['controller_raw_sha256']==q['controller_raw_sha256']and p['binding_sha256']==args.binding_sha256
        assert p['recipe_observations']==0 and not p['journal_created']and p['full_initial_state_restored']
        assert p['counts']['target_body_lines']==p['counts']['provider_body_lines']==0
        assert p['counts']['target_attempted']==p['counts']['target_refused']==2
        assert p['counts']['provider_attempted']==p['counts']['provider_refused']==2
        assert p['counts']['compiled_provider_attempted']==p['counts']['compiled_provider_refused']==1
        assert len(p['controls'])==6 and all(x['caller_restore']['exact32']for x in p['controls'])
        assert len(p['observer_target_immediate'])==1 and p['module_request']['was_cached']==(p['cache_state']=='cached')
        assert p['case_ids']==q['case_ids']and p['full_recipe_pool_count']==96
        assert p['runtime_identity']==go['runtime_identity']and p['helper']==q['helper']
        assert len(p['codec_metadata_controls'])==2 and all(x['caller_restore']['exact32']for x in p['codec_metadata_controls'])
        seen.add((p['prefix_api'],p['cache_state']))
    assert seen=={(api,cache)for api in('flood','flood_fill')for cache in('cold','cached')}
    return go

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('prefix','capture'),default='prefix');p.add_argument('--profile',choices=('minimum','modern'),required=True);p.add_argument('--prefix-api',choices=('flood','flood_fill'));p.add_argument('--cache-state',choices=('cold','cached'),default='cold');p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha256',required=True);p.add_argument('--root-go',type=Path);p.add_argument('--root-go-sha256');p.add_argument('--output',type=Path,required=True);p.add_argument('--expect-prerequisite-refusal',action='store_true');args=p.parse_args()
    assert not args.output.exists();args.output.mkdir(parents=True)
    try:
        q=validate(args.binding,args.binding_sha256,args.profile)
        if args.mode=='capture':
            assert args.prefix_api is None and args.cache_state=='cold','CAPTURE_HAS_FIXED_FULL_POOL_NO_FILTER'
            go=validate_go(args,q)
        else:
            assert args.prefix_api is not None and args.root_go is None and args.root_go_sha256 is None,'PREFIX_HAS_ZERO_AUTHORITY'
            go=None
    except (AssertionError,FileNotFoundError)as error:
        assert args.expect_prerequisite_refusal
        assert not any(n.split('.')[0]in('numpy','scipy','skimage','skimagexx')for n in sys.modules)
        dump(args.output/'terminal.json',dict(status='PASS_EXPECTED_PREREQUISITE_REFUSAL',error=dict(type=type(error).__name__,message=str(error)),controller_raw_sha256=H(Path(__file__)),original_budget=0,target_body_lines=0,scientific_imports=0,recipe_observations=0,journal_created=False));return
    assert not args.expect_prerequisite_refusal,'Invalid fault unexpectedly admitted'
    profile=q['profiles'][args.profile];assert sys.executable==profile['executable']and sys.version==profile['python']
    for name,expected in profile['packages'].items():
        dist=importlib.metadata.distribution(name);assert dist.version==expected['version']
        assert hashlib.sha256(dist.read_text('RECORD').encode()).hexdigest()==expected['RECORD_sha256']
        assert hashlib.sha256(dist.read_text('METADATA').encode()).hexdigest()==expected['METADATA_sha256']
    sources={}
    for name,item in q['upstream'][args.profile].items():
        path=Path(importlib.metadata.distribution(item['package']).locate_file(name)).resolve();assert H(path)==item['sha256'];sources[name]=str(path)
    assert sys.platform=='linux'and platform.machine()=='x86_64'and sys.byteorder=='little'and ctypes.sizeof(FEnv)==32 and FEnv.mxcsr.offset==28
    helper=ctypes.CDLL(str(R/q['helper']['library']['path']));libc=ctypes.CDLL(None)
    helper.reference_fenv_layout.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_size_t];helper.reference_fenv_layout.restype=ctypes.c_int
    helper.reference_fenv_restore.argtypes=[ctypes.c_void_p,ctypes.c_size_t];helper.reference_fenv_restore.restype=ctypes.c_int
    words=(ctypes.c_uint32*8)();assert helper.reference_fenv_layout(words,8)==0 and list(words)==q['helper']['ABI']==[32,0,4,8,12,20,28,8]
    libc.gnu_get_libc_version.restype=ctypes.c_char_p;assert libc.gnu_get_libc_version().decode()=='2.39'
    libc.fegetenv.argtypes=[ctypes.c_void_p];libc.fegetenv.restype=ctypes.c_int
    for name in('feclearexcept','feraiseexcept','fesetround'):getattr(libc,name).argtypes=[ctypes.c_int];getattr(libc,name).restype=ctypes.c_int
    def env():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):assert helper.reference_fenv_restore(ctypes.byref(value),32)==0
    def state(value):return dict(raw=bytes(value).hex(),control=value.control,status=value.status,mxcsr=value.mxcsr)
    initial=env();assert initial.tags==0xffff and initial.control&0x3f==0x3f and initial.mxcsr&0x1f80==0x1f80
    counts=dict(target_attempted=0,target_refused=0,provider_attempted=0,provider_refused=0,target_body_lines=0,provider_body_lines=0,compiled_provider_attempted=0,compiled_provider_refused=0,nested_flood_entries=0)
    targets={};protected={};entries=[];controls=[];codec_metadata_controls=[];capture_rows=[];last_case_diagnostics=None;failure=None;identity=None;module_request=None;inside=False;capture_allowed=False;current_case=None;call_count=0;public_attempts=0;total_public=0;immediate_calls=[];np=None
    oldtrace=sys.gettrace();oldprofile=sys.getprofile();assert oldtrace is None and oldprofile is None
    target_path=sources['skimage/morphology/_flood_fill.py']
    def kind(frame):
        if any(frame.f_code is fn.__code__ for fn in targets.values()):return 'target'
        if frame.f_code.co_filename==target_path and frame.f_code.co_name in('flood','flood_fill'):return 'target'
        if frame.f_code in protected:return 'provider'
        return None
    def trace(frame,event,value):
        nonlocal inside,call_count
        label=kind(frame)
        if label and args.mode=='capture':
            if event=='call':
                assert capture_allowed and current_case is not None,'UNAUTHORIZED_TARGET_OR_PROVIDER'
                if label=='target':
                    if inside:
                        assert active_api=='flood_fill' and frame.f_code is targets['flood'].__code__
                        counts['nested_flood_entries']+=1
                    else:
                        assert call_count==0;check_deadline(go);inside=True;call_count+=1;counts['target_attempted']+=1
                else:assert inside;counts['provider_attempted']+=1
            elif label=='target'and event=='return':
                if frame.f_code is targets[active_api].__code__:inside=False
            elif event=='line':assert inside;counts[label+'_body_lines']+=1
            return trace
        if label and event=='call':
            counts[label+'_attempted']+=1;counts[label+'_refused']+=1;entries.append(dict(kind=label,function=frame.f_code.co_name,module=frame.f_globals.get('__name__'),fenv_at_entry=state(env())))
            raise EntryRefused(label.upper()+'_REFUSED_BEFORE_BODY')
        if label and event=='line':counts[label+'_body_lines']+=1;raise AssertionError('BODY_LINE_FORBIDDEN')
        return trace
    def profiler(frame,event,value):
        if event=='c_call'and np is not None and getattr(value,'__name__',None)=='sum'and isinstance(getattr(value,'__self__',None),np.ndarray):
            if args.mode=='capture':assert inside and capture_allowed;counts['compiled_provider_attempted']+=1
            else:counts['compiled_provider_attempted']+=1;counts['compiled_provider_refused']+=1;raise EntryRefused('COMPILED_PROVIDER_REFUSED_BEFORE_BODY')
    sys.settrace(trace);sys.setprofile(profiler)
    try:
        import numpy as np
        assert np.__version__==profile['packages']['numpy']['version']
        assert Path(np.__file__).resolve()==Path(profile['packages']['numpy']['root'])/'numpy/__init__.py'
        name='skimage.morphology._flood_fill'
        if args.mode=='prefix'and args.cache_state=='cached':importlib.import_module(name)
        was_cached=name in sys.modules
        if args.mode=='prefix':assert was_cached==(args.cache_state=='cached')
        # Explicit Python import request is made even when the module is cached.
        module=__import__(name,fromlist=['flood','flood_fill']);public=importlib.import_module('skimage.morphology')
        module_request=dict(name=name,was_cached=was_cached,module=str(module.__file__))
        assert Path(module.__file__).resolve()==Path(target_path)
        for api in('flood','flood_fill'):
            targets[api]=getattr(module,api);assert getattr(public,api)is targets[api]and str(inspect.signature(targets[api]))==q['signatures'][api]
            assert inspect.isfunction(targets[api])and Path(targets[api].__code__.co_filename).resolve()==Path(target_path)
        aliases=importlib.import_module('skimage.segmentation')
        assert all(getattr(aliases,a)is targets[a] for a in targets)
        for fn in(module._resolve_neighborhood,np.pad):
            while True:
                if hasattr(fn,'__code__'):protected[fn.__code__]=getattr(fn,'__name__','provider')
                if not hasattr(fn,'__wrapped__'):break
                fn=fn.__wrapped__
        binaries=[]
        for expected in profile['packages'].values():
            for row in expected['compiled']:
                path=Path(row['path']);assert path.stat().st_size==row['bytes'];binaries.append(dict(row,scope='Retained admitted large compiled record/path/size, no repeated large binary hash'))
        cy=importlib.import_module('skimage.morphology._flood_fill_cy');assert Path(cy.__file__).resolve() in [Path(x['path']).resolve() for x in binaries]
        native=importlib.import_module('numpy.core._multiarray_umath'if np.__version__.split('.')[0]=='1'else'numpy._core._multiarray_umath')
        assert Path(native.__file__).resolve()in[Path(x['path']).resolve()for x in binaries]
        assert not any(n.split('.')[0]=='skimagexx'for n in sys.modules)
        sys.path.insert(0,str(W/'tests'));from flood_fill_readiness import cases
        from flood_fill_observer import observe
        specs=cases();assert [s['id']for s in specs]==q['case_ids']and len(specs)==96
        assert hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest()==q['cases_sha256']
        identity=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,scikit_image=profile['packages']['scikit-image']['version'],module=str(module.__file__),signatures={api:str(inspect.signature(fn))for api,fn in targets.items()},identity_aliases={'segmentation.'+api:'morphology.'+api for api in targets},sources=sources,packages=profile['packages'],compiled_dependencies=binaries,glibc='2.39',ABI=list(words))
        def target_call(*values,**options):
            nonlocal public_attempts,total_public
            if args.mode=='capture':
                assert capture_allowed and current_case is not None and public_attempts==0
                check_deadline(go);public_attempts+=1;total_public+=1
                assert total_public<=96
            before=np_state(np);hardware=env()
            try:return targets[active_api](*values,**options)
            finally:immediate_calls.append(dict(before=state(hardware),after=state(env()),numpy_before=before,numpy_after=np_state(np)))
        if args.mode=='capture':
            assert go['runtime_identity']==identity
            journal=args.output/'journal.jsonl'
            dump(args.output/'precall-freeze.json',dict(source=q['source'],binding_sha256=args.binding_sha256,controller_sha256=H(Path(__file__)),root_GO_sha256=args.root_go_sha256,runtime=identity,case_ids=q['case_ids']))
            for spec in specs:
                check_deadline(go);assert H(args.root_go)==args.root_go_sha256
                saved=env();old=np.geterr(),np.geterrcall(),np.getbufsize();printing=np.get_printoptions();current_case=spec['id'];active_api=spec['api'];call_count=0;public_attempts=0;immediate_calls=[];row=None;case_failure=None;stage='started'
                assert saved.tags==0xffff and saved.control&0x3f==0x3f and saved.mxcsr&0x1f80==0x1f80
                try:
                    append(journal,dict(event='started',id=current_case,utc=datetime.now(timezone.utc).isoformat(),consumed_before=total_public,retry=False))
                    try:
                        stage='observer';capture_allowed=True
                        try:row=observe(target_call,spec,np)
                        finally:capture_allowed=False;inside=False
                        assert public_attempts==call_count==len(immediate_calls)==1
                        assert old==(np.geterr(),np.geterrcall(),np.getbufsize())and printing==np.get_printoptions()
                        row['target_immediate']=immediate_calls[0];row['after_observer_fenv']=state(env())
                    except BaseException as error:
                        case_failure=dict(module=type(error).__module__,type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),stage=stage)
                        raise
                    finally:
                        np.seterr(**old[0]);np.seterrcall(old[1]);np.setbufsize(old[2]);np.set_printoptions(**printing);restore(saved);actual=env()
                        last_case_diagnostics=dict(id=current_case,immediate=list(immediate_calls),caller_restore=dict(saved=state(saved),actual=state(actual),exact32=bytes(actual)==bytes(saved)),target_entries=call_count,public_attempts=public_attempts,retry=False)
                        assert bytes(actual)==bytes(saved),'EXACT32_RESTORE_FAILED'
                        if row is not None:row['caller_restore']=last_case_diagnostics['caller_restore']
                    stage='postvalidation';assert H(args.binding)==args.binding_sha256 and H(args.root_go)==args.root_go_sha256;check_deadline(go)
                    stage='completed_write';append(journal,dict(event='completed',id=current_case,target_entries=call_count,public_attempts=public_attempts,row=row));capture_rows.append(row);current_case=None
                except BaseException as error:
                    # Preserve original error even if the failure journal write
                    # also fails. No row is fabricated and no call is retried.
                    propagated=dict(module=type(error).__module__,type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),stage=stage)
                    if case_failure is None:case_failure=propagated
                    if last_case_diagnostics is None or last_case_diagnostics['id']!=current_case:
                        last_case_diagnostics=dict(id=current_case,immediate=list(immediate_calls),caller_restore=None,target_entries=call_count,public_attempts=public_attempts,retry=False)
                    last_case_diagnostics['failure']=case_failure
                    if propagated!=case_failure:last_case_diagnostics['propagated_failure']=propagated
                    try:append(journal,dict(event='failed_case_diagnostics',**last_case_diagnostics))
                    except BaseException as writer_error:last_case_diagnostics['failure_writer_error']=dict(type=type(writer_error).__name__,message=str(writer_error),traceback=traceback.format_exc())
                    raise
            assert len(capture_rows)==total_public==96
            dump(args.output/'observations.json',capture_rows)
        else:
            active_api=args.prefix_api;spec=next(s for s in specs if s['api']==active_api)
            def refused_call(label,fn,values,options):
                saved=env();old=np.geterr(),np.geterrcall(),np.getbufsize();printing=np.get_printoptions();before_counts=dict(counts)
                previous_trace=sys.gettrace();previous_profile=sys.getprofile();sys.settrace(trace);sys.setprofile(profiler)
                try:
                    before=env()
                    try:fn(*values,**options)
                    except (EntryRefused,RuntimeError)as error:
                        message=str(error);assert message in('TARGET_REFUSED_BEFORE_BODY','PROVIDER_REFUSED_BEFORE_BODY','COMPILED_PROVIDER_REFUSED_BEFORE_BODY','HARDWARE_SETUP_ONLY')
                    else:raise AssertionError('Expected refusal not observed')
                    immediate=dict(before=state(before),after=state(env()))
                    assert old==(np.geterr(),np.geterrcall(),np.getbufsize())and printing==np.get_printoptions()
                finally:
                    sys.settrace(previous_trace);sys.setprofile(previous_profile)
                    np.seterr(**old[0]);np.seterrcall(old[1]);np.setbufsize(old[2]);np.set_printoptions(**printing);restore(saved);actual=env();assert bytes(actual)==bytes(saved)
                controls.append(dict(control=label,message=message,immediate=immediate,caller_restore=dict(saved=state(saved),actual=state(actual),exact32=True),delta={n:counts[n]-before_counts[n]for n in counts}))
            refused_call('target-direct-'+active_api,targets[active_api],(None,None)if active_api=='flood'else(None,None,None),{})
            refused_call('target-same-observer-'+active_api,lambda:observe(target_call,spec,np),(),{})
            assert len(immediate_calls)==1
            refused_call('neighborhood',module._resolve_neighborhood,(None,None,2),{})
            refused_call('numpy-pad',np.pad,(None,1),{})
            # Refuse the actual compiled helper ATTRIBUTE REQUEST before dispatch;
            # do not rely on Cython profiling to prevent its algorithm body.
            import types
            original_class=module.__class__
            class RequestGuard(types.ModuleType):
                def __getattribute__(self,name):
                    if name in ('_flood_fill_equal','_flood_fill_tolerance'):
                        counts['compiled_provider_attempted']+=1;counts['compiled_provider_refused']+=1
                        raise EntryRefused('COMPILED_PROVIDER_REFUSED_BEFORE_BODY')
                    return super().__getattribute__(name)
            module.__class__=RequestGuard
            try:refused_call('compiled-flood-helper-request',lambda:getattr(module,'_flood_fill_equal'),(),{})
            finally:module.__class__=original_class
            def hardware_only():
                assert libc.feclearexcept(0x3d)==0 and libc.feraiseexcept(4)==0 and libc.fesetround(0x400)==0
                raise RuntimeError('HARDWARE_SETUP_ONLY')
            refused_call('sticky-rounding-finally-full32',hardware_only,(),{})
            assert counts['target_attempted']==counts['target_refused']==2
            assert counts['provider_attempted']==counts['provider_refused']==2
            assert counts['compiled_provider_attempted']==counts['compiled_provider_refused']==1
            assert counts['target_body_lines']==counts['provider_body_lines']==0
            assert controls[-1]['immediate']['after']['mxcsr']&4 and controls[-1]['immediate']['after']['control']&0xc00==0x400
            class InertTypedError(Exception):pass
            for metadata in((np.add,'same_kind',np.dtype('int64'),np.dtype('uint32'),2),(np.add,[np.dtype('uint64'),np.dtype('uint64')])):
                saved=env()
                try:
                    def inert(*values,**options):raise InertTypedError(*metadata)
                    synthetic=observe(inert,spec,np);assert synthetic['error']['name']=='InertTypedError'and synthetic['supplied_call_attempts']==1
                finally:restore(saved);actual=env();assert bytes(saved)==bytes(actual)
                codec_metadata_controls.append(dict(scope='INERT_TYPED_EXCEPTION_NOT_ORIGINAL',error=synthetic['error'],caller_restore=dict(saved=state(saved),actual=state(actual),exact32=True)))
    except BaseException as error:failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        sys.settrace(oldtrace);sys.setprofile(oldprofile);restore(initial);final=env();assert bytes(final)==bytes(initial)
    # Stable source/runtime bindings are validated once at entry and once at
    # the final trust boundary, never whole-source/binary rehash per case.
    try:
        assert H(args.binding)==args.binding_sha256 and H(Path(__file__))==q['controller_raw_sha256']
        assert subprocess.check_output([*q['git_argv'],'rev-parse','HEAD'],text=True).strip()==q['source']
        assert not subprocess.check_output([*q['git_argv'],'status','--porcelain'],text=True).strip()
        for name,pin in q['source_files_raw_sha256'].items():assert H(W/name)==pin,name
        if args.mode=='capture':assert H(args.root_go)==args.root_go_sha256;check_deadline(go)
    except BaseException as error:
        final_failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        if failure is None:failure=final_failure
        else:failure['final_validation_failure']=final_failure
    if args.mode=='prefix':assert not any((args.output/n).exists()for n in('journal.jsonl','observations.json','precall-freeze.json'))
    dump(args.output/'terminal.json',dict(status=('PASS_ACTUAL_FLOOD_FILL_CONTROLLER_ZERO_BODY_PREFIX'if args.mode=='prefix'else'COMPLETE_PENDING_INDEPENDENT_RAW_ADMISSION')if failure is None else'FAILED_FLOOD_FILL_CONTROLLER',source=q['source'],controller_raw_sha256=H(Path(__file__)),binding_sha256=args.binding_sha256,profile=args.profile,mode=args.mode,prefix_api=args.prefix_api,cache_state=args.cache_state,module_request=module_request,case_ids=q['case_ids'],full_recipe_pool_count=96,counts=counts,entries=entries,controls=controls,codec_metadata_controls=codec_metadata_controls,last_case_diagnostics=last_case_diagnostics,failure=failure,runtime_identity=identity,helper=q['helper'],initial_fenv=state(initial),final_fenv=state(final),full_initial_state_restored=True,observer_target_immediate=immediate_calls if args.mode=='prefix'else None,public_attempts=total_public,recipe_observations=len(capture_rows),journal_created=(args.output/'journal.jsonl').exists(),original_computation=0 if args.mode=='prefix'else counts['target_attempted'],original_budget=0 if args.mode=='prefix'else 96,native_image_bodies=0,sampler_bodies=0,builds=0,root_original_GO=args.root_go_sha256,source_unchanged_after=failure is None,git_argv=q['git_argv'],environment={n:os.environ[n]for n in LIMITS}))
    print('FLOOD_FILL_CONTROLLER',args.profile,args.mode,'PASS'if failure is None else failure)
    if failure:raise SystemExit(1)
if __name__=='__main__':main()
