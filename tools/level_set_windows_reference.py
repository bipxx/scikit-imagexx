"""NEW HOST original-only Windows witness for two existing LevelSet cases.
No project import, candidate/core call, build, environment creation or timing.
Linux frozen observations remain immutable; this does not select supplements.
"""
import argparse, ctypes, hashlib, importlib.util, io, json, os, platform
from pathlib import Path
import subprocess, sys, traceback, zipfile

ROOT=Path(__file__).resolve().parents[1]
RUN=40686116976
SOURCE='75bc34e5bb978361baeea8340dbeae2491c42086'
ARTIFACT=12818350704
ARTIFACT_SIZE=1150273123
ARTIFACT_SHA='191a94b3ec6c76541897b2433ce5b4fb1c542a6ed7f12c2854a269f3fd3aae8c'
SOURCE_ZIP_SHA='e8896a58ffb8707b588ba47e92cfa89c047a17cca4107cf25d388b72afc61921'
ORIGINAL_SHA='71ced09bbab3918523c38abeda1235a79ca34eb7467d76875e0ef5b13d2092d6'
IDS=('checkerboard_level_set/square-wide','disk_level_set/radius-longdouble')
CONTRACT_PINS={
 'tests/level_set_contract.py':'2d993c41d01d125bbee11d5c61df57bad0d4cf39d15d4997f141fa4fcc55d40a',
 'tests/test_level_set_public.py':'cac50419b1c1ea5789ad20a3142a0e45617f4916e2f1123e076ace0e604fca30',
 'coverage/contracts/level-set.json':'7336286ba886aa2895a7b5df6f69f10be49803090371eaaa1259061f00a75be9'}
PREFIX='public-qualification/windows/'
H=lambda raw:hashlib.sha256(raw).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def transport():
    helper=load('original_windows_transport',ROOT/'tools/matrix_runtime_certificate_probe.py')
    for name,value in dict(FAILED_RUN=RUN,FAILED_SOURCE=SOURCE,ARTIFACT_ID=ARTIFACT,
        ARTIFACT_SHA=ARTIFACT_SHA,ARTIFACT_SIZE=ARTIFACT_SIZE,MAX_ZIP=ARTIFACT_SIZE).items():setattr(helper,name,value)
    return helper


def prepare(raw,out,helper):
    site=out/'original-prefix';site.mkdir();files={};wheels={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        state=json.loads(z.read(PREFIX+'state.json'));package=json.loads(z.read(PREFIX+'package.json'))
        require(state['source']['revision']==package['source_revision']==SOURCE,'donor source differs')
        require(package['source_archive_sha256']==SOURCE_ZIP_SHA and len(state['runtime_wheels'])==14,'donor source/runtime closure differs')
        # Runtime wheels only. Deliberately exclude the entire project wheel/SDK.
        for name,record in state['runtime_wheels'].items():
            body=z.read(PREFIX+'runtime-wheels/'+name);require(H(body)==record['sha256'],'runtime wheel differs')
            records=helper.wheel_records(body);wheels[name]=record['sha256']
            with zipfile.ZipFile(io.BytesIO(body)) as wheel:
                for relative,entry in records.items():
                    require(relative.split('/')[0]!='skimagexx' and relative not in files,'project package or wheel collision')
                    target=site/helper.safe_member(relative);target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(wheel.read(relative));files[relative]=entry
        source=z.read(PREFIX+'source.zip');require(H(source)==SOURCE_ZIP_SHA,'frozen source ZIP differs')
        with zipfile.ZipFile(io.BytesIO(source)) as frozen:
            require(frozen.comment.decode()==SOURCE,'source identity differs')
            for name,pin in CONTRACT_PINS.items():
                body=frozen.read(name)
                require(H(body.replace(b'\r\n',b'\n'))==pin==state['source']['files_sha256'][name],'frozen factory/observer differs')
                target=out/'reference-source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
    dump(out/'inputs.json',dict(new_host=True,original_only=True,product_imported=False,donor_run=RUN,donor_source=SOURCE,
        artifact=ARTIFACT,artifact_sha256=ARTIFACT_SHA,wheels=wheels,files=files,runtime_pins=state['runtime_pins'],contract_pins=CONTRACT_PINS))


def capture_original(harness,np,functions):
    specs=tuple(s for s in harness.CASES if s['id'] in IDS)
    require(len(harness.CASES)==79 and tuple(s['id'] for s in specs)==IDS,'only the two frozen79-spec IDs permitted')
    rows=[];calls=[];arguments=[]
    for spec in specs:
        original=functions[spec['api']]
        def once(*args,**kwargs):
            require(spec['id'] not in calls,'original case called twice')
            calls.append(spec['id'])
            item=dict(id=spec['id'])
            if 'radius' in kwargs:
                value=kwargs['radius'];cls=type(value)
                item.update(radius_type_module=cls.__module__,radius_type_name=cls.__name__,
                    radius_type_is_float64=cls is np.float64,radius_type_is_longdouble=cls is np.longdouble,
                    radius_dtype=str(value.dtype),radius_dtype_type_module=value.dtype.type.__module__,
                    radius_dtype_type_name=value.dtype.type.__name__,radius_itemsize=value.dtype.itemsize)
            arguments.append(item)
            return original(*args,**kwargs)
        rows.append(harness.observe(once,spec,np))
    require(tuple(calls)==IDS,'original must execute once per case')
    return rows,calls,arguments


def child(out):
    require(os.name=='nt' and sys.version_info[:3]==(3,13,16) and sys.flags.optimize==0,'exact unoptimized Windows CPython3.13.16 required')
    require(not any(n=='skimagexx' or n.startswith('skimagexx.') for n in sys.modules),'project imported before witness')
    forcing=('OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES','NPY_ENABLE_CPU_FEATURES','MKL_DEBUG_CPU_TYPE','MKL_SERVICE_FORCE_INTEL','LD_PRELOAD','LD_AUDIT')
    require(not any(os.environ.get(n) for n in forcing),'CPU/runtime forcing forbidden')
    require(all(os.environ.get(n)=='1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')),'reference pools must be1')
    auth=json.loads((out/'inputs.json').read_text());site=out/'original-prefix'
    for name,entry in auth['files'].items():require(H((site/name).read_bytes())==entry['sha256'],'original runtime changed')
    sys.path.insert(0,str(out/'reference-source/tests'))
    harness=load('original_only_level_set',out/'reference-source/tests/test_level_set_public.py')
    require(harness.SOURCE_SHA==ORIGINAL_SHA,'original source pin differs')
    for name,pin in CONTRACT_PINS.items():require(H((out/'reference-source'/name).read_bytes().replace(b'\r\n',b'\n'))==pin,'frozen contract changed')
    specs=[s for s in harness.CASES if s['id'] in IDS]
    require(len(harness.CASES)==79 and tuple(s['id'] for s in specs)==IDS,'unapproved witness cases')
    dump(out/'before-original-freeze.json',dict(ids=IDS,specifications=specs,contract_pins=CONTRACT_PINS,original_source_sha256=ORIGINAL_SHA,original_calls=0))
    np,functions,runtime=harness.runtime()
    for name in ('numpy','scipy','skimage','pytest'):
        require(Path(__import__(name).__file__).resolve().is_relative_to(site),'foreign runtime '+name)
    import importlib.metadata
    for name,version in auth['runtime_pins'].items():require(importlib.metadata.version(name)==version,'runtime version differs')
    rows,calls,arguments=capture_original(harness,np,functions)
    dump(out/'observed-rows.json',dict(original_calls=calls,actual_arguments=arguments,observations=rows))
    dtype=np.dtype(np.longdouble)
    abi=dict(c_long_bytes=ctypes.sizeof(ctypes.c_long),longdouble_is_float64=np.longdouble is np.float64,
        longdouble_dtype=str(dtype),longdouble_dtype_type_module=dtype.type.__module__,longdouble_dtype_type_name=dtype.type.__name__,
        longdouble_itemsize=dtype.itemsize,longdouble_nmant=int(np.finfo(np.longdouble).nmant))
    require(not any(n=='skimagexx' or n.startswith('skimagexx.') for n in sys.modules),'project import occurred')
    for name,entry in auth['files'].items():require(H((site/name).read_bytes())==entry['sha256'],'original runtime mutated')
    for name,pin in CONTRACT_PINS.items():require(H((out/'reference-source'/name).read_bytes().replace(b'\r\n',b'\n'))==pin,'Linux frozen contract mutated')
    dump(out/'original-observations.json',dict(scope='NEW HOST ORIGINAL ONLY; no candidate or qualification credit; Linux79 records unchanged',
        original_calls=calls,native_calls=0,timings=False,runtime=runtime,abi=abi,actual_arguments=arguments,observations=rows,
        platform=platform.uname()._asdict(),python=sys.version,executable=sys.executable,executable_sha256=H(Path(sys.executable).read_bytes()),
        optimization=sys.flags.optimize,project_imported=False,all_inputs_post_unchanged=True))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('plan','all','_child'));parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv);out=args.output.resolve()
    require(out.is_relative_to((ROOT/'build').resolve()),'ignored owned output required')
    if args.stage=='plan':
        print(json.dumps(dict(donor_run=RUN,donor_source=SOURCE,artifact=ARTIFACT,artifact_sha256=ARTIFACT_SHA,
            ids=IDS,original_calls=2,native_calls=0,product_imported=False,timings=False,builds=0,original_source_sha256=ORIGINAL_SHA,contract_pins=CONTRACT_PINS)));return 0
    require(os.environ.get('GITHUB_ACTIONS')=='true','fresh hosted original witness only; no local scientific calls')
    try:
        if args.stage=='_child':return child(out)
        require(os.name=='nt','Windows donor only');out.mkdir(parents=True)
        helper=transport();metadata,raw=helper.fetch_artifact(os.environ.get('GH_TOKEN'));dump(out/'artifact.json',metadata);prepare(raw,out,helper);del raw
        dump(out/'controller.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=H(Path(__file__).read_bytes()),new_host=True,original_only=True))
        env=dict(os.environ,PYTHONPATH=str(out/'original-prefix'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_child','--output',str(out)],env=env,capture_output=True)
        (out/'original.stdout').write_bytes(result.stdout);(out/'original.stderr').write_bytes(result.stderr);dump(out/'child-exit.json',dict(exit_code=result.returncode));return result.returncode
    except Exception:
        out.mkdir(parents=True,exist_ok=True);(out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':raise SystemExit(main())
