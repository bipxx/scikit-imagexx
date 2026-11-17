"""Strict saved8 wide-outer private whole-API comparison; root GO required.

Uses unchanged real find/MARK frontends and binders, never constructs overlay
from saved masks. No original call, result normalization, tolerance or FP mask.
"""
import argparse,base64,ctypes,hashlib,importlib,importlib.metadata,inspect,json,os,platform,struct,subprocess,sys,types,zlib
from pathlib import Path
from datetime import datetime,timezone
W=Path(__file__).resolve().parents[1]
R=Path('/mnt/c/Users/HEQ/Projects/scikit-imagexx')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIXTURE=W/'tests/fixtures/mark_wide_outer_originals_v1.json'


def load_saved():
    q=json.loads(FIXTURE.read_bytes());rows={}
    for p,item in q['profiles'].items():
        raw=zlib.decompress(base64.b64decode(item['observations_zlib_base64']))
        assert hashlib.sha256(raw).hexdigest()==item['observations_sha256'];rows[p]=json.loads(raw)
    return q,rows


def semantic(state):return {n:state[n]for n in('control','status','mxcsr')}


def seed_bytes(current,captured):
    # Current trusted empty-stack template retains its own instruction/data
    # addresses, tags/reserved bytes; only authentic incoming semantic words
    # are seeded. NEVER copy nonportable historical IP/data addresses.
    assert len(current)==32 and int.from_bytes(current[8:10],'little')==0xffff
    assert captured['control']&0x3f==0x3f and captured['control']&0xc00==0
    assert captured['mxcsr']&0x1f80==0x1f80 and captured['mxcsr']&(3<<13)==0
    out=bytearray(current)
    for name,offset,fmt in [('control',0,'<H'),('status',4,'<H'),('mxcsr',28,'<I')]:struct.pack_into(fmt,out,offset,captured[name])
    return bytes(out)


def assess(expected,actual,immediate):
    # Raw originals remain intact; controller-only process/address diagnostics
    # are displayed separately, not compared as portable API values.
    diagnostic={'target_immediate','after_observer_fenv','caller_restore'}
    a={k:v for k,v in actual.items()if k not in diagnostic};e={k:v for k,v in expected.items()if k not in diagnostic}
    differences=[]
    for key in sorted(set(a)|set(e)):
        if a.get(key)!=e.get(key):differences.append('software.'+key)
    for phase in('before','after'):
        if semantic(immediate[phase])!=semantic(expected['target_immediate'][phase]):differences.append('target_FP.'+phase)
        # Handler addresses are explicitly rawdiagnostics; type/policy/bufsize
        # remain strict. No result or hardware words are changed.
        left=immediate['numpy_'+phase];right=expected['target_immediate']['numpy_'+phase]
        if {k:v for k,v in left.items()if k!='handler_id'}!={k:v for k,v in right.items()if k!='handler_id'}:differences.append('target_NumPy.'+phase)
    return differences


def write(path,value):
    raw=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with path.open('x',encoding='utf-8',newline='\n')as f:f.write(raw);f.flush();os.fsync(f.fileno())


def validate(args):
    assert args.root_go is not None and args.root_go_sha256 is not None,'NO_ROOT_GO'
    assert H(args.root_go)==args.root_go_sha256
    q=json.loads(args.root_go.read_bytes());transport,rows=load_saved()
    assert q['status']=='ROOT_AUTHORIZED_MARK_WIDE_PRIVATE_SAVED4_ONCE'and q['profile']==args.profile
    assert q['script_sha256']==H(Path(__file__))and q['fixture_sha256']==H(FIXTURE)
    assert q['case_ids']==[r['id']for r in rows[args.profile]]and q['native_public_calls']==4 and q['workers']==1
    assert q['original_calls']==q['timing']==q['builds']==0
    assert q['runtime']==transport['profiles'][args.profile]['runtime']and q['helper']==transport['profiles'][args.profile]['helper']
    assert Path(q['output']).resolve()==args.output.resolve()and not args.output.exists()
    end=datetime.fromisoformat(q['deadline_utc']);assert end.utcoffset()is not None and datetime.now(timezone.utc)<end
    assert subprocess.check_output([*q['git_argv'],'rev-parse','HEAD'],text=True).strip()==q['source']
    assert not subprocess.check_output([*q['git_argv'],'status','--porcelain'],text=True).strip()
    for n,pin in q['source_files'].items():assert H(W/n)==pin
    # New private module is authenticated once at this actual import boundary;
    # no old archive/core/corpus pass is repeated.
    assert Path(q['component']['path']).stat().st_size==q['component']['bytes']and H(q['component']['path'])==q['component']['sha256']
    assert all(os.environ.get(n)=='1'for n in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS','PYTHONDONTWRITEBYTECODE'))
    return q,transport,rows[args.profile]


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=('minimum','modern'),required=True);p.add_argument('--root-go',type=Path);p.add_argument('--root-go-sha256');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    q,transport,expected=validate(args) # Fail before science/helper without GO.
    assert sys.platform=='linux'and platform.machine()=='x86_64'and sys.byteorder=='little'and sys.executable==q['runtime']['executable']and sys.version==q['runtime']['python']
    for name,expected_package in q['runtime']['packages'].items():
        package=importlib.metadata.distribution(name);assert package.version==expected_package['version']
        assert hashlib.sha256(package.read_text('RECORD').encode()).hexdigest()==expected_package['RECORD_sha256']
        assert hashlib.sha256(package.read_text('METADATA').encode()).hexdigest()==expected_package['METADATA_sha256']
    helper_path=R/q['helper']['library']['path'];assert H(helper_path)==q['helper']['library']['sha256']
    helper=ctypes.CDLL(str(helper_path));libc=ctypes.CDLL(None)
    libc.gnu_get_libc_version.restype=ctypes.c_char_p;assert libc.gnu_get_libc_version().decode()=='2.39'
    helper.reference_fenv_layout.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_size_t];helper.reference_fenv_layout.restype=ctypes.c_int
    helper.reference_fenv_restore.argtypes=[ctypes.c_void_p,ctypes.c_size_t];helper.reference_fenv_restore.restype=ctypes.c_int
    layout=(ctypes.c_uint32*8)();assert helper.reference_fenv_layout(layout,8)==0 and list(layout)==[32,0,4,8,12,20,28,8]
    libc.fegetenv.argtypes=[ctypes.c_void_p];libc.fegetenv.restype=ctypes.c_int
    def env():
        raw=(ctypes.c_ubyte*32)();assert libc.fegetenv(ctypes.byref(raw))==0;return bytes(raw)
    def restore(raw):
        value=(ctypes.c_ubyte*32).from_buffer_copy(raw);assert helper.reference_fenv_restore(value,32)==0
    def state(raw):return dict(raw=raw.hex(),control=int.from_bytes(raw[0:2],'little'),status=int.from_bytes(raw[4:6],'little'),mxcsr=int.from_bytes(raw[28:32],'little'))
    process=env();args.output.mkdir(parents=True);results=[];consumed=0;last=None;failure=None;oldtrace=sys.gettrace();oldprofile=sys.getprofile()
    try:
        import numpy as np
        assert np.__version__==q['runtime']['numpy']
        assert Path(np.__file__).resolve()==Path(q['runtime']['packages']['numpy']['root'])/'numpy/__init__.py'
        for package in q['runtime']['packages'].values():
            for binary in package['compiled']:assert Path(binary['path']).stat().st_size==binary['bytes']
        sys.path.insert(0,str(Path(q['component']['path']).parent));core=importlib.import_module('_mark_boundaries_core')
        assert Path(core.__file__).resolve()==Path(q['component']['path']).resolve()and core.get_num_threads()==1
        assert all(inspect.isbuiltin(getattr(core,n))for n in('find_boundaries_into','mark_boundaries_prepare','mark_boundaries_mask','mark_boundaries_color','mark_boundaries_count','mark_boundaries_paint','get_num_threads'))
        package=types.ModuleType('wide_outer_private');package.__path__=[str(W/'python/skimagexx')];sys.modules[package.__name__]=package;sys.modules[package.__name__+'._core']=core
        find=importlib.import_module(package.__name__+'._find_boundaries');mark=importlib.import_module(package.__name__+'._mark_boundaries')
        functions={'segmentation.find_boundaries':find.find_boundaries,'segmentation.mark_boundaries':mark.mark_boundaries}
        import mark_wide_outer_capture as observer
        specs=observer.case_specs();assert [s['id']for s in specs]==q['case_ids']
        denied=[];returned=[];inside=False
        def deny(frame,event,value):
            if event=='call'and frame.f_globals.get('__name__','').startswith(('skimage.','scipy.ndimage')):denied.append(frame.f_code.co_name);raise AssertionError('ORIGINAL_PROVIDER_FORBIDDEN')
            return deny
        def profile(frame,event,value):
            if inside and event=='c_return'and any(value is getattr(core,n)for n in('find_boundaries_into','mark_boundaries_prepare','mark_boundaries_mask','mark_boundaries_color','mark_boundaries_count','mark_boundaries_paint')):returned.append(value.__name__)
        for spec,expected_row in zip(specs,expected):
            assert H(args.root_go)==args.root_go_sha256
            assert datetime.now(timezone.utc)<datetime.fromisoformat(q['deadline_utc'])
            caller=env();immediate=[];returned=[];row=None;inside=False;last=dict(id=spec['id'],raw_target_immediate=None,actual=None,caller_restore=None)
            def target(*a,**k):
                nonlocal consumed,inside
                assert not inside;consumed+=1;assert consumed<=4
                restore(seed_bytes(env(),expected_row['target_immediate']['before']))
                before=env();policy=observer.np_state(np);inside=True
                try:return functions[spec['api']](*a,**k)
                finally:
                    after=env();inside=False;last['raw_target_immediate']=dict(before32=before.hex(),after32=after.hex());immediate.append(dict(before=state(before),after=state(after),numpy_before=policy,numpy_after=observer.np_state(np)))
            sys.settrace(deny);sys.setprofile(profile)
            try:
                row=observer.observe(target,spec,np);last['actual']=row
            finally:
                sys.settrace(oldtrace);sys.setprofile(oldprofile);restore(caller);actual=env();last['caller_restore']=dict(saved32=caller.hex(),actual32=actual.hex(),exact32=actual==caller);assert actual==caller
            assert len(immediate)==1 and not denied and core.get_num_threads()==1
            required={'find_boundaries_into'}if spec['api'].endswith('find_boundaries')else {'mark_boundaries_prepare','mark_boundaries_mask','mark_boundaries_color','mark_boundaries_count','mark_boundaries_paint'}
            differences=assess(expected_row,row,immediate[0])
            if not required<=set(returned):differences.append('native_pipeline.incomplete')
            last=dict(id=spec['id'],actual=row,target_immediate=immediate[0],expected_target_immediate=expected_row['target_immediate'],differences=differences,native_C_returns=returned,caller_restore=dict(saved32=caller.hex(),actual32=actual.hex(),exact32=True),private_component_only=True)
            write(args.output/(str(len(results))+'-case.json'),last);results.append(last)
        assert consumed==4
    except BaseException as error:failure=dict(type=type(error).__name__,message=str(error));raise
    finally:
        sys.settrace(oldtrace);sys.setprofile(oldprofile);restore(process);exact=env()==process
        summary=dict(status='PASS_PRIVATE_PROFILE'if failure is None and len(results)==4 and all(not r['differences']for r in results)else'FAILED_PRIVATE_COMPARISON',failure=failure,profile=args.profile,consumed=consumed,completed=len(results),last=last,difference_rows=sum(bool(r['differences'])for r in results),process32_before_collector_exact=exact,original_calls=0,timed_batches=0,genuine_core=False,installed_SDK=False)
        try:
            try:write(args.output/'terminal.json',summary)
            except BaseException as error:summary['terminal_writer_error']=dict(type=type(error).__name__,message=str(error));write(args.output/'terminal-fallback.json',summary);raise
        finally:restore(process);assert env()==process
    if any(r['differences']for r in results):raise SystemExit(1)

if __name__=='__main__':main()
