"""Artifact-only, NEW HOST metadata diagnosis; never image/native acceptance.

The frozen helper is traced, not patched. The probe makes no explicit image,
matmul/BLAS arithmetic, build, numerical calibration, thread-setter or CPU-force
requests. Original package imports retain their ordinary initialization.
"""
from __future__ import annotations
import argparse,ast,base64,csv,ctypes,hashlib,importlib,inspect,io,json,os
from pathlib import Path,PurePosixPath
import platform,stat,struct,subprocess,sys,traceback,urllib.error,urllib.request,zipfile

ROOT=Path(__file__).resolve().parents[1]
FAILED_RUN=40117331038
FAILED_SOURCE='a567c5d775bdc82f3b3aba7a1e8ba9a33fc50679'
ARTIFACT_ID=12569527308
ARTIFACT_SHA='494a920118521de40107300dc7748aa108647175f6fd10d439dee1b5406dd98d'
ARTIFACT_SIZE=400047839
WHEEL_SHA='22b08d134dadf9ee8970fb779df09c6f5d5481c38ee329fc35c1b063750ca917'
SOURCE_SHA='3a10c842e082111a3b1b4d5d1e526e2393fa16a48c18a199aa03936bb126eb94'
HELPER_RAW_SHA='8f9937b65467265d4a6f55921af19c163083f02b08d45988f7b76e8904f235b7'
HELPER_SHA='deb219b0cfb0f38d04e53d13e09ebfc40196f29de97c1e3c4c3d4273e416e9b8' # canonical LF
ORIGINAL_PYTHON_SHA='31fe10b4ed82a2f960af0ef9931f46a4b84ff286f8b6ffd9dee098cf4f10f17f'
PREFIX='public-qualification/windows/'
MAX_ZIP=512*1024*1024
FORCING=('OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES','NPY_ENABLE_CPU_FEATURES','MKL_DEBUG_CPU_TYPE','MKL_SERVICE_FORCE_INTEL')
POOLS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')
ENV_KEYS=(*FORCING,*POOLS,'OMP_WAIT_POLICY','LD_PRELOAD','LD_AUDIT','PYTHONPATH','PYTHONHOME',
          'PROCESSOR_IDENTIFIER','NUMBER_OF_PROCESSORS','GITHUB_RUN_ID')
METADATA_FUNCTIONS={'_unforced','_loaded_blas','_pe_imports','_authenticate_runtime','_runtime_profile','resolve','module_path'}
ADMISSION_IMPORTS=('_rank','_components','_component_consumers','_metrics','_geometric',
    '_footprint_dense','_footprint_decomposition','_footprint_util','_tv_chambolle','_tv_bregman','_max_tree','_matrix_transform')

def require(value,message):
    if not value:raise ValueError(message)
def H(raw):return hashlib.sha256(raw).hexdigest()
def file_record(path):
    path=Path(path).resolve()
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest)
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def safe_member(name):
    p=PurePosixPath(name)
    require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name,'archive path escape')
    return p


def verify_helper_bytes(raw):
    require(H(raw)==HELPER_RAW_SHA,'raw frozen Windows helper differs')
    require(H(raw.replace(b'\r\n',b'\n'))==HELPER_SHA,'canonical frozen helper differs')


def verify_admission_imports(qsource):
    """Authenticate the context from donor source, not the controller tree."""
    tree=ast.parse(qsource)
    admission=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='admission')
    tuples=[n.iter for n in ast.walk(admission) if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple)]
    candidates=[ast.literal_eval(n) for n in tuples
                if any(isinstance(e,ast.Constant) and e.value=='_matrix_transform' for e in n.elts)]
    require(candidates==[ADMISSION_IMPORTS],'frozen admission import order differs')


def wheel_records(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=[n for n in z.namelist() if not n.endswith('/')]
        require(len(names)==len(set(names)),'duplicate wheel entry')
        require(not any(stat.S_ISLNK(i.external_attr>>16) for i in z.infolist()),'wheel symlink refused')
        for n in names:safe_member(n)
        record,=[n for n in names if n.endswith('.dist-info/RECORD')]
        rows=list(csv.reader(io.StringIO(z.read(record).decode())))
        require(len(rows)==len(names) and {r[0] for r in rows}==set(names),'wheel RECORD closure differs')
        result={}
        for name,encoded,size in rows:
            body=z.read(name)
            if name==record:require(not encoded and not size,'RECORD self hash')
            else:
                want='sha256='+base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip('=')
                require(encoded==want and int(size)==len(body),'wheel payload differs')
            require('.data/' not in name,'unassessed wheel relocation')
            result[name]=dict(bytes=len(body),sha256=H(body))
        return result


def fetch_artifact(token):
    require(token,'read-only Actions token required')
    api=f'https://api.github.com/repos/bipxx/scikit-imagexx/actions/artifacts/{ARTIFACT_ID}'
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'}
    with urllib.request.urlopen(urllib.request.Request(api,headers=headers)) as response:meta=json.load(response)
    require(meta['workflow_run']['id']==FAILED_RUN and meta['workflow_run']['head_sha']==FAILED_SOURCE,'wrong donor run/source')
    require(meta['size_in_bytes']==ARTIFACT_SIZE and meta['digest']=='sha256:'+ARTIFACT_SHA and not meta['expired'],'wrong donor artifact')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    try:
        urllib.request.build_opener(NoRedirect).open(urllib.request.Request(api+'/zip',headers=headers))
        raise ValueError('signed artifact redirect required')
    except urllib.error.HTTPError as error:
        require(error.code in (301,302,303,307,308),'artifact redirect failed')
        url=error.headers['Location'];require(url.startswith('https://'),'unsafe redirect')
    # Deliberately no bearer header on the signed blob URL.
    with urllib.request.urlopen(url) as response:raw=response.read(MAX_ZIP+1)
    require(len(raw)==ARTIFACT_SIZE and len(raw)<=MAX_ZIP and H(raw)==ARTIFACT_SHA,'outer artifact bytes differ')
    return meta,raw


def extract_inputs(raw,out):
    """Only authenticate/extract exact original wheels; no pip/build/new env."""
    require(H(raw)==ARTIFACT_SHA,'unverified donor ZIP')
    site=out/'genuine-prefix';site.mkdir(exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos=[i for i in z.infolist() if not i.is_dir()]
        require(len(infos)==len({i.filename for i in infos}),'duplicate artifact entry')
        index={}
        for info in infos:
            safe_member(info.filename)
            require(not stat.S_ISLNK(info.external_attr>>16),'artifact symlink refused')
            with z.open(info) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest() # verifies CRC
            index[info.filename]=dict(bytes=info.file_size,sha256=digest,crc32=f'{info.CRC:08x}')
        state=json.loads(z.read(PREFIX+'state.json'));package=json.loads(z.read(PREFIX+'package.json'))
        require(state['source']['revision']==package['source_revision']==FAILED_SOURCE,'wrong product source')
        require(package['wheel_sha256']==WHEEL_SHA and package['source_archive_sha256']==SOURCE_SHA,'product/source pins differ')
        require(len(package['members_sha256'])==114 and len(package['SDK_sha256'])==65,'not the failed complete114/65 product')
        source=z.read(PREFIX+'source.zip');require(H(source)==SOURCE_SHA,'source ZIP differs')
        with zipfile.ZipFile(io.BytesIO(source)) as archive:
            require(archive.comment.decode()==FAILED_SOURCE,'source identity differs')
            frozen=archive.read('python/skimagexx/_matrix_transform.py')
            verify_helper_bytes(frozen)
            verify_admission_imports(archive.read('tools/public_qualification.py'))
        native=[n for n,r in index.items() if n.startswith(PREFIX+'product-wheels/') and n.endswith('.whl') and r['sha256']==WHEEL_SHA]
        require(len(native)==1,'exact built wheel absent')
        inputs={native[0]:dict(sha256=WHEEL_SHA,kind='failed-built-product')}
        expected_runtime=state['runtime_wheels']
        require(len(expected_runtime)==14,'original Windows runtime closure incomplete')
        for name,row in expected_runtime.items():
            member=PREFIX+'runtime-wheels/'+name
            require(index[member]['sha256']==row['sha256'],'runtime wheel differs')
            inputs[member]=dict(row,kind='original-runtime-wheel')
        files={}
        for name,pin in inputs.items():
            body=z.read(name);require(H(body)==pin['sha256'],'selected input differs')
            records=wheel_records(body)
            if pin['kind']=='failed-built-product':
                require({n:r['sha256'] for n,r in records.items()}==package['members_sha256'],'product RECORD/payload differs')
            with zipfile.ZipFile(io.BytesIO(body)) as wheel:
                for relative,row in records.items():
                    require(relative not in files,'wheel namespace collision')
                    target=site/safe_member(relative);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(wheel.read(relative))
                    files[relative]=row
        helper=(site/'skimagexx/_matrix_transform.py').read_bytes()
        verify_helper_bytes(helper)
        receipt=dict(donor_run=FAILED_RUN,donor_source=FAILED_SOURCE,donor_outer_sha256=ARTIFACT_SHA,all_members=index,
            source_zip_sha256=H(source),inputs=inputs,files=files,package=package,status='FAILED_ORIGINAL_PRODUCT_NOT_ACCEPTED')
        dump(out/'input-authentication.json',receipt)
        return site,receipt


def safe_value(value):
    if value is None or type(value) in (str,bool,int):return value
    if type(value) is bytes:
        try:text=value.decode('ascii') if len(value)<1024 else None
        except UnicodeError:text=None
        return dict(bytes=len(value),sha256=H(value),hex=value.hex() if len(value)<1024 else None,ascii=text)
    if isinstance(value,Path):return str(value)
    if type(value) in (tuple,list):return [safe_value(v) for v in value[:12]]
    return {'type':type(value).__name__}


def trace_natural_profile(wrapper):
    """Trace exact unmodified metadata code and its FIRST natural return."""
    source=Path(wrapper.__file__).resolve();lines=source.read_text().splitlines();events=[]
    wanted=('key','pin','names','path','origin','description','actual','corename','identities','version','digest','n','u','handle')
    previous=sys.gettrace();require(previous is None,'external tracer would change diagnostic context')
    def trace(frame,event,arg):
        if frame.f_code.co_name not in METADATA_FUNCTIONS:return None
        if Path(frame.f_code.co_filename).resolve()!=source:return None
        if event in ('return','exception'):
            row=dict(function=frame.f_code.co_name,event=event,line=frame.f_lineno,
                source_context=lines[max(0,frame.f_lineno-4):frame.f_lineno+1],
                locals={n:safe_value(frame.f_locals[n]) for n in wanted if n in frame.f_locals})
            if event=='return':row['return']=safe_value(arg)
            else:row['exception']=dict(type=arg[0].__name__,message=str(arg[1]))
            events.append(row)
        return trace
    sys.settrace(trace)
    try:result=wrapper._runtime_profile()
    finally:sys.settrace(previous)
    return result,events


def post_attempt_getters(wrapper,np):
    """Reusable READ-ONLY payload for a later admission recorder.

    It never retries/clears the natural certificate. Values are explicitly
    post-attempt observations, not invented return values from the first call.
    """
    umath=importlib.import_module('numpy._core._multiarray_umath');u=Path(umath.__file__).resolve()
    result=dict(UMATH=file_record(u),CPU_features=umath.__cpu_features__)
    try:
        loaded=wrapper._loaded_blas(u,u.read_bytes())
        if loaded is None:result['loaded_BLAS']=None
        else:
            library,path,resolve=loaded;pin=wrapper._RUNTIME_PINS[(np.__version__,file_record(u)['sha256'])];values={}
            for name in ('corename','num_threads','config'):
                f=getattr(library,pin[1].replace('corename',name));f.argtypes=[]
                f.restype=ctypes.c_int if name=='num_threads' else ctypes.c_char_p
                owner=resolve(f);values[name]=dict(symbol=f.__name__,owner=file_record(owner),value=safe_value(f()))
            result['loaded_BLAS']=values
    except Exception as error:result['metadata_exception']=dict(type=type(error).__name__,message=str(error))
    return result


def certificate_payload(wrapper,np):
    """Future recorder API: one natural attempt, then labeled metadata reads.

    This module does NOT wire that future hook into production qualification.
    The caller must consume natural_profile, never a post-query retry result.
    """
    result,events=trace_natural_profile(wrapper)
    refusals=[e for e in events if e['event']=='return' and e.get('return') is None
              and e['function'] in ('_loaded_blas','_authenticate_runtime','_runtime_profile')]
    predicate=refusals[0] if result is None and refusals else None
    detail='see unchanged source line/context'
    if predicate and predicate['function']=='_runtime_profile' and 'corename' in predicate['locals']:
        local=predicate['locals'];allowed=getattr(wrapper,'_RUNTIME_CORES',{}).get(local.get('digest'),())
        detail=('unassessed corename' if local['corename'] not in allowed
                else 'known corename; first return guard requires threads()!=1 (post-read value is separately labeled)')
    return dict(natural_profile=result,first_natural_trace=events,
                first_refusal=predicate,predicate_detail=detail,
                post_attempt=post_attempt_getters(wrapper,np),certificate_retried=False)


def probe(out):
    """NEW HOST; no explicit image/matmul requests, ordinary imports unchanged."""
    require(os.name=='nt' and sys.version_info[:3]==(3,13,16) and struct.calcsize('P')==8,'exact Windows CP313.16 host required')
    env={k:os.environ.get(k) for k in ENV_KEYS}
    dump(out/'environment-before-import.json',dict(environment=env,new_host=True,original_failed_host=False))
    require(not any(k in os.environ for k in FORCING) and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),'forced/interposed environment refused, never sanitized')
    require(all(os.environ.get(k)=='1' for k in POOLS),'original thread policy must be applied before import')
    auth=json.loads((out/'input-authentication.json').read_text());site=(out/'genuine-prefix').resolve()
    require(not any(n.split('.')[0] in ('numpy','scipy','skimage','skimagexx') for n in sys.modules),'preimported numerical context refused')
    require(site in {Path(p).resolve() for p in sys.path if p}, 'genuine prefix must be configured before interpreter startup')
    # Preserve the failed admission's public-core-before-explicit-NumPy order.
    # Importing NumPy first could hide a real whole-core initialization issue.
    import skimagexx as xx
    xx.set_fallback_enabled(False)
    require(xx._core.get_num_threads()==1,'public core initial worker policy differs')
    for name in ADMISSION_IMPORTS:
        module=importlib.import_module('skimagexx.'+name)
        require(module._core is xx._core,'alternate worker core in admission context')
    import numpy as np
    wrapper=importlib.import_module('skimagexx._matrix_transform')
    require(Path(xx.__file__).resolve()==site/'skimagexx/__init__.py' and wrapper._core is xx._core,'not genuine whole-core context')
    require(Path(np.__file__).resolve()==site/'numpy/__init__.py','wrong NumPy origin')
    require(file_record(xx._core.__file__)['sha256']==auth['package']['members_sha256'][auth['package']['sole_core']],'wrong core bytes')
    helper=Path(wrapper.__file__).read_bytes()
    verify_helper_bytes(helper)
    before=(xx._core.matrix_transform_completed(),xx._core.matrix_transform_v2_completed())
    payload=certificate_payload(wrapper,np)
    observation=dict(scope='NEW HOST METADATA ONLY; NEVER original failed-host provenance or native acceptance',
        public_acceptance=False,donor_source=FAILED_SOURCE,donor_run=FAILED_RUN,new_host=True,
        interpreter=file_record(sys.executable),original_interpreter_sha256=ORIGINAL_PYTHON_SHA,
        same_interpreter_binary=file_record(sys.executable)['sha256']==ORIGINAL_PYTHON_SHA,
        python=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,sys_path=sys.path,platform=platform.uname()._asdict(),
        admission_import_order=['skimagexx',*ADMISSION_IMPORTS,'numpy-explicit'],fallback_enabled=False,
        environment=env,package=file_record(xx.__file__),core=file_record(xx._core.__file__),helper=file_record(wrapper.__file__),
        certificate=payload,counter_before=before,
        workers=xx._core.get_num_threads(),has_openmp=xx._core.has_openmp,
        legacy_interface=xx._core.matrix_transform_interface,legacy_capabilities=xx._core.matrix_transform_capabilities,
        v2_interface=xx._core.matrix_transform_v2_interface,v2_capabilities=xx._core.matrix_transform_v2_capabilities)
    after=(xx._core.matrix_transform_completed(),xx._core.matrix_transform_v2_completed())
    require(before==after==(0,0),'matrix completion counter changed unexpectedly')
    observation['counter_after']=after
    # Rehash all input wheel files after imports; new caches are neither inputs
    # nor silently substituted source. No fixture/native algorithm is executed.
    for name,row in auth['files'].items():
        require(file_record(site/name)['sha256']==row['sha256'],'genuine extracted input mutated')
    dump(out/'observations.json',observation)
    return observation


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('plan','all','probe'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv);out=args.output.resolve()
    require(out.is_relative_to((ROOT/'build').resolve()),'owned build output required')
    if args.stage=='plan':
        print(json.dumps(dict(artifact=ARTIFACT_ID,outer_sha256=ARTIFACT_SHA,wheel_sha256=WHEEL_SHA,
            source_sha256=SOURCE_SHA,source_revision=FAILED_SOURCE,host='NEW Windows3.13.16',public_acceptance=False,images=0,timers=0)));return
    try:
        if args.stage=='all':
            require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('GITHUB_RUN_ID','').isdigit(), 'fresh hosted diagnostic only; no local preparation')
            out.mkdir(parents=True,exist_ok=False);require(os.name=='nt','Windows artifact host only')
            revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
            controller={}
            for name in ('tools/matrix_runtime_certificate_probe.py','tests/test_matrix_runtime_certificate_probe.py',
                         '.github/workflows/matrix-runtime-certificate-probe.yml'):
                blob=subprocess.check_output(['git','-C',str(ROOT),'show',revision+':'+name])
                require((ROOT/name).read_bytes().replace(b'\r\n',b'\n')==blob.replace(b'\r\n',b'\n'),'controller working source differs')
                controller[name]=dict(raw_git_sha256=H(blob),source_utf8=blob.decode())
            dump(out/'controller.json',dict(revision=revision,members=controller,new_host=True,public_acceptance=False))
            meta,raw=fetch_artifact(os.environ.get('GH_TOKEN'));dump(out/'artifact.json',meta);extract_inputs(raw,out)
            environment=dict(os.environ);environment['PYTHONPATH']=str(out/'genuine-prefix');environment['PYTHONNOUSERSITE']='1';environment['PYTHONDONTWRITEBYTECODE']='1'
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'probe','--output',str(out)],env=environment,capture_output=True)
            (out/'probe.stdout').write_bytes(result.stdout);(out/'probe.stderr').write_bytes(result.stderr)
            dump(out/'probe.exit.json',dict(returncode=result.returncode));require(result.returncode==0,'metadata child failed; preserve raw report')
        else:probe(out)
    except Exception:
        out.mkdir(parents=True,exist_ok=True);(out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
