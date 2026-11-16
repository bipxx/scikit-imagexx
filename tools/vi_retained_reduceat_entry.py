"""Prospective failed438 retained-row primitive entry; default UNBOUND."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
from datetime import datetime, timedelta, timezone
import zipfile

STATUS = 'ROOT_AUTHORIZED_FAILED438_ROW_REDUCEAT_ENTRY_V1'
PURPOSE = 'FAILED438_RETAINED_ROW_REDUCEAT_ONLY'
MASKS = ('NPY_DISABLE_CPU_FEATURES', 'OPENBLAS_CORETYPE', 'SKIMAGEXX_DISABLE_SIMD', 'LD_PRELOAD', 'LD_AUDIT')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def safe_parts(name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and '\\' not in name and ':' not in name, 'Unsafe member')
    require(all(p not in ('', '.', '..') and p.rstrip(' .') == p for p in name.split('/')), 'Unsafe member component')
    require(all(p.split('.')[0].upper() not in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(1,10)), *('LPT'+str(i) for i in range(1,10))} for p in path.parts), 'Reserved member')
    return path.parts


def check_authority(go, inputs, entry_pin, inputs_pin, now):
    require(isinstance(go,dict) and go.get('status')==STATUS, 'UNBOUND: distinct primitive authority required')
    require(go.get('purpose')==PURPOSE and go.get('allow_failed438')is True and go.get('allow_F1_build_once')is True,'Wrong primitive scope')
    p=inputs['producer']
    require((go.get('producer_run'),go.get('source_revision'),go.get('job_id'),go.get('artifact_id'),go.get('artifact_sha256'),go.get('artifact_bytes'))==(p['run'],p['revision'],p['job'],p['artifact'],p['sha256'],p['bytes']),'Wrong failed producer')
    require(go.get('entry_sha256')==entry_pin and go.get('inputs_sha256')==inputs_pin,'Entry/input bytes changed')
    require(go.get('root_intake_sha256')==inputs['root_intake']['sha256'],'Failed intake changed')
    require(go.get('retained_data_sha256')==inputs['retained_study']['input_sha256'],'Retained input changed')
    require(go.get('captured_CPU')==inputs['retained_study']['captured_CPU'],'Captured CPU scope changed')
    require(go.get('case')==inputs['case'] and go.get('workers')==1 and go.get('caps')==inputs['caps'],'Primitive operation/budget changed')
    require(isinstance(go.get('controller_revision'),str)and len(go['controller_revision'])==40,'Controller unbound')
    primitive=go.get('primitive',{})
    require(primitive.get('path')and len(primitive.get('sha256')or'')==64 and primitive.get('status')=='ROOT_GO_VI_RETAINED_ROW_REDUCEAT_V1','Primitive runner unbound')
    safe_parts(primitive['path'])
    require(isinstance(go.get('nonce'),str)and len(go['nonce'])>=16,'Fresh nonce unbound')
    issued=datetime.fromisoformat(go['issued_utc']);deadline=datetime.fromisoformat(go['deadline_utc'])
    require(issued.utcoffset()==deadline.utcoffset()==timedelta(0)and issued<=now<deadline and timedelta(0)<deadline-issued<=timedelta(hours=1),'Expired/unbounded UTC')
    require(go.get('software_policy')=={'errors':{'divide':'warn','over':'warn','under':'ignore','invalid':'warn'},'buffer':8192},'Software policy differs')


def reject_masks(environment):
    require(not any(environment.get(k) for k in MASKS), 'CPU/preload intervention forbidden')


def verify_api(run, job, artifacts, inputs, repository):
    p = inputs['producer']
    require(run['id'] == p['run'] and run['head_sha'] == p['revision'], 'Run/source differs')
    require(run['head_repository']['full_name'] == repository and run['repository']['full_name'] == repository, 'Wrong repository')
    require(run['path'] == '.github/workflows/public-qualification.yml' and run['status'] == 'completed' and run['conclusion'] == 'failure' and run['run_attempt'] == 1, 'Only exact failed campaign')
    require(job['id'] == p['job'] and job['run_id'] == p['run'] and job['head_sha'] == p['revision'] and job['name'] == p['job_name'], 'Wrong minimum job')
    require(job['status'] == 'completed' and job['conclusion'] == 'failure', 'Only failed minimum job')
    require(artifacts['total_count'] == len(artifacts['artifacts']), 'Artifact pagination incomplete')
    matches = [a for a in artifacts['artifacts'] if a['name'] == p['artifact_name']]
    require(len(matches) == 1, 'Artifact role ambiguous')
    a = matches[0]
    require(a['id'] == p['artifact'] and a['size_in_bytes'] == p['bytes'] and a['digest'] == 'sha256:'+p['sha256'] and not a['expired'], 'Artifact identity differs')
    require(a['workflow_run']['id'] == p['run'] and a['workflow_run']['head_sha'] == p['revision'], 'Artifact source differs')
    return a


def validate_selected_index(infos, selected):
    require(len(infos) <= 100000, 'Archive entry ceiling')
    names = [i.filename for i in infos]
    require(len(names) == len(set(names)) and len(names) == len({n.casefold() for n in names}), 'Duplicate/aliased ZIP member')
    by_name = {}
    for info in infos:
        name = info.filename[:-1] if info.is_dir() else info.filename
        safe_parts(name)
        mode = info.external_attr >> 16
        require(not stat.S_ISLNK(mode) and not (mode & 0o170000 and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))), 'Nonregular ZIP member')
        by_name[info.filename] = info
    require(set(selected) <= set(by_name), 'Selected package/wheels/metadata missing')
    require(len(selected) == 247 and sum(v['bytes'] for v in selected.values()) < 256*1024**2, 'Selected extraction ceiling')
    for name, row in selected.items():
        safe_parts(row['destination'])
        require(by_name[name].file_size == row['bytes'] and not by_name[name].is_dir(), 'Selected member size/type differs')
    require(len({v['destination'].casefold() for v in selected.values()}) == len(selected), 'Destination alias')
    return by_name


def extract_selected(archive, destination, selected, factory=zipfile.ZipFile):
    destination = Path(destination)
    require(not destination.exists(), 'Fresh selected destination required')
    with factory(archive) as source:
        by_name = validate_selected_index(source.infolist(), selected)
        destination.mkdir(parents=True)
        for name, row in selected.items():
            target = destination.joinpath(*safe_parts(row['destination']))
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(); count = 0
            with source.open(by_name[name]) as incoming, target.open('xb') as output:
                for block in iter(lambda: incoming.read(1024*1024), b''):
                    count += len(block); require(count <= row['bytes'], 'Selected stream exceeds bound')
                    digest.update(block); output.write(block)
            require(count == row['bytes'] and digest.hexdigest() == row['sha256'], 'Selected payload hash differs')


def tree_manifest(root):
    root = Path(root); result = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink() and not (getattr(path.lstat(),'st_file_attributes',0) & 0x400), 'Linked installed member')
        if path.is_file(): result[path.relative_to(root).as_posix()] = sha(path)
    return result


def validate_manifests(state, package, sdk, inputs):
    require(state['schema'] == 'public-qualification-v2' and state['profile'] == 'minimum' and state['source']['revision'] == inputs['producer']['revision'] and state['no_timing'] is True, 'State/source differs')
    require(package['source_revision'] == state['source']['revision'] and package['record_verified'] is True, 'Package provenance differs')
    require(package['members_sha256'] == inputs['package_members_sha256'] and len(package['members_sha256']) == 228, 'Package map differs')
    require(package['SDK_sha256'] == sdk['SDK_sha256'] == inputs['SDK_sha256'] and len(inputs['SDK_sha256']) == 108, 'SDK map differs')
    require(package['sole_core'] == inputs['core']['member'] and package['members_sha256'][package['sole_core']] == inputs['core']['sha256'], 'Wrong core')
    require(state['runtime_wheels'] == inputs['runtime_wheels'] and state['runtime_pins'] == inputs['runtime_pins'] and len(state['runtime_wheels']) == 13, 'Dependency wheel map differs')
    for name, pin in inputs['source_files_canonical_sha256'].items(): require(state['source']['files_sha256'][name] == pin, 'Prepared source pin differs')


def verify_package(stage, inputs):
    stage = Path(stage); meta = stage/'metadata'
    state = json.loads((meta/'state.json').read_bytes()); package = json.loads((meta/'package.json').read_bytes()); sdk = json.loads((meta/'sdk.json').read_bytes())
    validate_manifests(state, package, sdk, inputs)
    require(tree_manifest(stage/'package') == package['members_sha256'], 'Actual package members differ')
    return state

def verify_sources(controller, snapshot, inputs):
    for name, pin in inputs['source_files_canonical_sha256'].items():
        require(hashlib.sha256((Path(snapshot)/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == pin, 'Snapshot source changed: '+name)
    for name in inputs['production4']:
        require((Path(controller)/name).read_bytes().replace(b'\r\n',b'\n') == (Path(snapshot)/name).read_bytes().replace(b'\r\n',b'\n'), 'Controller production changed')
    for name, pin in inputs['FENV7_raw_sha256'].items(): require(sha(Path(snapshot)/name) == pin, 'FENV source changed')


def check_helper(receipt, inputs):
    receipt = Path(receipt); data = json.loads(receipt.read_bytes())
    require(data['schema'] == 'util-views-fenv-build-v1' and data['status'] == 'PASS_SOURCE_BUILT_FENV32', 'Fresh helper controls failed')
    require(data['source_context_sha256'] == inputs['FENV7_raw_sha256']['coverage/contracts/util-views-fenv-v1.json'], 'Wrong helper context')
    require(data['controls'] == ['controls-0.json'], 'Exactly one runtime helper-control process')
    for name, pin in data['files'].items(): require(sha(receipt.parent.joinpath(*safe_parts(name))) == pin, 'Helper bundle changed')
    return receipt.parent/'libutil_views_fenv_restore.so'


def host_metadata():
    reject_masks(os.environ)
    text = Path('/proc/cpuinfo').read_text().split('\n\n',1)[0]
    values = {k.strip():v.strip() for k,v in (line.split(':',1) for line in text.splitlines() if ':' in line)}
    return dict(system=os.uname().sysname,machine=os.uname().machine,python=sys.version,executable=sys.executable,
        CPU={k:values.get(k) for k in ('vendor_id','model name','cpu family','model','stepping')},flags=sorted(values.get('flags','').split()),
        environment={k:os.environ.get(k) for k in (*MASKS,'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','LD_LIBRARY_PATH')})


def mapped_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(None,5)
        if len(parts)==6 and parts[5].startswith('/') and ('.so' in parts[5] or Path(parts[5]).name.startswith('ld-')):
            require(not parts[5].endswith(' (deleted)'), 'Loaded deleted library')
            paths.add(str(Path(parts[5]).resolve()))
    return {name:dict(bytes=Path(name).stat().st_size,sha256=sha(name)) for name in sorted(paths)}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def captured_eligibility(study,host,features,normalized_libraries):
    comparison={name:record==study['captured_loaded_libraries'].get(name)for name,record in normalized_libraries.items()}
    required=[name for name in study['captured_loaded_libraries']if name.startswith('runtime/numpy/core/_multiarray_umath')or name.startswith('runtime/numpy.libs/libopenblas')]
    gates=dict(CPU=host['CPU']==study['captured_CPU'],CPU_flags=host['flags']==study['captured_CPU_flags'],features=features==study['captured_numpy_features'],libraries=bool(required)and set(required)<=set(normalized_libraries)and all(comparison.values()),platform=host['system']=='Linux'and host['machine']=='x86_64')
    return dict(eligible=all(gates.values()),gates=gates,library_comparison=comparison)


def prepare(go, runtime):
    require(isinstance(runtime,dict)and runtime.get('status')=='ROOT_BOUND_FAILED438_REDUCEAT_RUNTIME_V1','UNBOUND: primitive runtime')
    authorization=Path(runtime['authorization_path']);require(sha(authorization)==runtime['authorization_sha256'],'Parent authority changed')
    parent=json.loads(authorization.read_bytes());inputs_path=Path(runtime['inputs_path']);require(sha(inputs_path)==parent['inputs_sha256'],'Inputs changed')
    inputs=json.loads(inputs_path.read_bytes());check_authority(parent,inputs,sha(__file__),sha(inputs_path),datetime.now(timezone.utc))
    require(go['hosted_entry']['sha256']==parent['entry_sha256']and go['runner_sha256']==parent['primitive']['sha256'],'Runner/entry binding differs')
    require(go['retained_data_sha256']==inputs['retained_study']['input_sha256'],'Child data binding differs')
    reject_masks(os.environ)
    require(sys.flags.isolated and sys.flags.no_site and not sys.flags.optimize and sys.version_info[:3]==(3,11,17),'Wrong isolated interpreter')
    stage=Path(runtime['stage']);snapshot=Path(runtime['snapshot']);controller=Path(runtime['controller'])
    verify_sources(controller,snapshot,inputs);verify_package(stage,inputs)
    current_runtime=tree_manifest(stage/'runtime');require(current_runtime==runtime['installed_runtime_sha256'],'Installed runtime changed')
    study=inputs['retained_study']
    require(all(current_runtime.get(name)==pin for name,pin in study['scientific_runtime_sha256'].items()),'Captured NumPy/SciPy source/binary identity differs')
    require(current_runtime['scipy/sparse/_compressed.py']==study['scipy_compressed_sha256'],'Captured sparse reduction source differs')
    helper=check_helper(runtime['helper_receipt'],inputs);require(sha(runtime['helper_receipt'])==runtime['helper_receipt_sha256'],'Helper receipt changed')
    prior_path=list(sys.path);prior_env={k:os.environ.get(k)for k in('UTIL_VIEWS_FENV_MODE','UTIL_VIEWS_FENV_RECEIPT','UTIL_VIEWS_FENV_RECEIPT_SHA256')}
    try:
        sys.path[:0]=[str(stage/'runtime')]
        os.environ.update(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=runtime['helper_receipt'],UTIL_VIEWS_FENV_RECEIPT_SHA256=runtime['helper_receipt_sha256'])
        import ctypes
        import importlib
        import struct
        np=importlib.import_module('numpy')
        require(Path(np.__file__).resolve().is_relative_to((stage/'runtime').resolve())and np.__version__=='1.24.4','Wrong NumPy prefix/version')
        fenv=_load(snapshot/'tests/util_views_fenv.py','_retained_reduceat_fenv')
        before_host=host_metadata();before_libs=mapped_libraries();features=dict(importlib.import_module('numpy.core._multiarray_umath').__cpu_features__)
        def normalized(name):
            prefix=str((stage/'runtime').resolve())+'/'
            return 'runtime/'+name[len(prefix):]if name.startswith(prefix)else name
        eligibility=captured_eligibility(study,before_host,features,{normalized(name):value for name,value in before_libs.items()})
        gates=eligibility['gates'];eligible=eligibility['eligible'];library_comparison=eligibility['library_comparison']
        input_raw=bytes.fromhex(study['input_raw_hex']);require(hashlib.sha256(input_raw).hexdigest()==study['input_sha256']and len(input_raw)==448,'Retained56words changed')
        starts_raw=struct.pack('<7q',*study['starts'])
        # Exactly two frombuffer transports before observer caller FP scope.
        values=np.frombuffer(input_raw,dtype='<f8');starts=np.frombuffer(starts_raw,dtype='<i8')
        require(values.shape==(56,)and values.strides==(8,)and values.flags.aligned and starts.shape==(7,)and starts.flags.aligned,'Primitive transport differs')
        identity=dict(scope=PURPOSE,host=before_host,features=features,library_comparison=library_comparison,eligibility_gates=gates,eligible=eligible,new_host_separate_witness=True,capture_run=study['capture_run'],capture_pair_sha256=study['rawpair_sha256'],core_authenticated_not_loaded=inputs['core'],scientific_runtime_files_compared=len(study['scientific_runtime_sha256']),entry_FP_calls=0,caller32_scope=inputs['entry_scope']['caller32_scope'])
        ctx=dict(status='ROOT_AUTHENTICATED_RETAINED_ROW_REDUCEAT_ENTRY_V1',eligible=eligible,np=np,ctypes=ctypes,fenv=fenv,values=values,starts=starts,input_raw=input_raw,starts_raw=starts_raw,model_raw_hex=study['model_seven_entropy_raw_hex'],identity=identity,entry_transport_counts=dict(frombuffer_transport=2))
        def post_identity():
            verify_sources(controller,snapshot,inputs);verify_package(stage,inputs)
            require(tree_manifest(stage/'runtime')==runtime['installed_runtime_sha256'],'Runtime bytes changed')
            require(sha(authorization)==runtime['authorization_sha256']and sha(inputs_path)==parent['inputs_sha256'],'Authority changed')
            require(check_helper(runtime['helper_receipt'],inputs).resolve()==helper.resolve(),'Helper changed')
            require(host_metadata()==before_host,'Host/environment changed')
            after=mapped_libraries();expected=dict(before_libs);expected[str(helper.resolve())]=dict(bytes=helper.stat().st_size,sha256=sha(helper))
            require(after==expected or after==before_libs,'Loaded library closure changed')
            write_new(runtime['identity_after_path'],dict(status='PASS_STDLIB_PRIMITIVE_POST_IDENTITY',host=before_host,loaded_libraries=after,FP_calls=0,science_calls=0))
        ctx.update(post_identity=post_identity,_entry_prior_path=prior_path,_entry_prior_env=prior_env,_entry_terminal_path=runtime['terminal_path'])
        return ctx
    except BaseException:
        sys.path[:]=prior_path
        for key,value in prior_env.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        raise


def finish(ctx):
    # Fresh-child imports are outside the primitive caller32 scope. No FP/backend
    # or scientific function is queried by this final stdlib writer/restoration.
    try:
        terminal = ctx.get('_primitive_terminal')
        if terminal is None:
            terminal = dict(status='REDUCEAT_TERMINAL_MISSING_PARTIAL_SCOPE',completed_primitive=False,final_restore_verified=False)
        write_new(ctx['_entry_terminal_path'],terminal)
        if ctx.get('_primitive_terminal') is not None:
            require(terminal.get('schema') == 'vi-retained-row-reduceat-terminal-v1', 'Primitive terminal schema differs')
    finally:
        sys.path[:] = ctx['_entry_prior_path']
        for key,value in ctx['_entry_prior_env'].items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def _command(argv, log):
    import subprocess
    write_new(str(log)+'.command.json',argv)
    with Path(log).open('xb') as stream:
        done = subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
    write_new(str(log)+'.exit.json',dict(exit=done.returncode))
    require(done.returncode==0,'Command failed; no retry: '+str(log))


def _git_identity(path, expected):
    import subprocess
    head = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    clean = subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()
    require(head==expected and not clean,'Git HEAD/clean entry differs')
    return dict(path=str(path),HEAD=head,clean=True,Git=subprocess.check_output(['git','--version'],text=True).strip())


def finish_child(exit_code, secondary):
    if exit_code:
        raise SystemExit(exit_code)
    require(secondary is None, 'Post-identity failed')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('authenticate','intake','check','bind','run'))
    parser.add_argument('--authorization',type=Path);parser.add_argument('--inputs',type=Path)
    parser.add_argument('--controller',type=Path);parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--stage',type=Path);parser.add_argument('--results',type=Path)
    args=parser.parse_args(argv)
    require(args.authorization is not None and args.inputs is not None and args.phase,'UNBOUND: no auth/network/archive/import/build/target authority')
    go=json.loads(args.authorization.read_bytes()); require(go.get('status')==STATUS,'UNBOUND: explicit failed438 authority required')
    inputs=json.loads(args.inputs.read_bytes());check_authority(go,inputs,sha(__file__),sha(args.inputs),datetime.now(timezone.utc))
    require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('GITHUB_RUN_ATTEMPT')=='1' and os.environ.get('GITHUB_SHA')==go['controller_revision'],'Hosted first attempt only')
    reject_masks(os.environ);results=args.results.resolve();stage=args.stage.resolve();controller=args.controller.resolve();snapshot=args.snapshot.resolve()
    results.mkdir(parents=True,exist_ok=True)
    if args.phase=='authenticate':
        import subprocess
        require(sha(controller/inputs['root_intake']['path'])==inputs['root_intake']['sha256'],'Root failed intake absent/changed')
        receipt=json.loads((controller/inputs['root_intake']['path']).read_bytes());require(receipt['status']==inputs['root_intake']['status'] and receipt['source']==inputs['producer']['revision'] and receipt['run']==inputs['producer']['run'],'Not root-admitted failed438')
        write_new(results/'Git-entry.json',dict(controller=_git_identity(controller,go['controller_revision']),snapshot=_git_identity(snapshot,go['source_revision'])))
        verify_sources(controller,snapshot,inputs)
        records=[]
        for endpoint,name in [(f"runs/{go['producer_run']}",'producer-run'),(f"jobs/{go['job_id']}",'producer-job'),(f"runs/{go['producer_run']}/artifacts?per_page=100",'producer-artifacts')]:
            data=subprocess.check_output(['gh','api','repos/'+os.environ['GITHUB_REPOSITORY']+'/actions/'+endpoint]);write_new(results/(name+'.json'),json.loads(data));records.append(json.loads(data))
        selected=verify_api(*records,inputs,os.environ['GITHUB_REPOSITORY']);write_new(results/'selection.json',selected)
        if not (results/'root-authorization.json').exists():shutil.copyfile(args.authorization,results/'root-authorization.json')
        else:require(sha(results/'root-authorization.json')==sha(args.authorization),'Initial authority differs')
        shutil.copyfile(args.inputs,results/'frozen-inputs.json');shutil.copyfile(__file__,results/'hosted-entry-source.py')
    elif args.phase=='intake':
        import subprocess
        selected=json.loads((results/'selection.json').read_bytes());require(selected['id']==inputs['producer']['artifact'],'Auth selection missing')
        archive=stage.parent/'failed438-row-reduceat-minimum.zip';require(not archive.exists() and not stage.exists(),'Fresh transport/stage required')
        with archive.open('xb') as stream:done=subprocess.run(['gh','api',f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/artifacts/{inputs['producer']['artifact']}/zip"],stdout=stream)
        write_new(results/'download-exit.json',dict(exit=done.returncode));require(done.returncode==0,'Download failed; no retry')
        actual=dict(bytes=archive.stat().st_size,sha256=sha(archive));write_new(results/'transport.json',actual)
        require(actual==dict(bytes=inputs['producer']['bytes'],sha256=inputs['producer']['sha256']),'Transport identity differs')
        extract_selected(archive,stage,inputs['selection']);verify_package(stage,inputs)
        shutil.copyfile(stage/'metadata/pinned-runtime-inputs.txt',stage/'requirements.txt')
        archive.unlink();write_new(results/'intake.json',dict(status='PASS_AUTHENTICATED_FAILED438_SELECTED_ONLY',selected_files=247,selected_bytes=inputs['selected_bytes'],ZIP_removed=True,qualification_credit=False,VI4_credit=False))
    elif args.phase=='check':
        verify_sources(controller,snapshot,inputs)
        require((results/'intake.json').is_file(), 'Actual intake prerequisite absent')
    elif args.phase=='bind':
        verify_sources(controller,snapshot,inputs);verify_package(stage,inputs)
        require((results/'intake.json').is_file() and (results/'preparation-capacity.json').is_file(),'Predecessor receipts absent')
        capacity=json.loads((results/'preparation-capacity.json').read_bytes());require(capacity['passed']is True and capacity['estimated_peak_bytes']==8*1024**3,'Capacity unbound')
        runtime_tree=tree_manifest(stage/'runtime');require(runtime_tree,'Offline runtime absent')
        import email.parser
        distributions={}
        for meta in (stage/'runtime').glob('*.dist-info/METADATA'):
            message=email.parser.BytesParser().parsebytes(meta.read_bytes());distributions[message['Name'].lower().replace('_','-')]=message['Version']
        require(distributions=={k.lower().replace('_','-'):v for k,v in inputs['runtime_pins'].items()},'Exactly13 offline nonproduct distributions required')
        helper=results/'fenv/receipt.json';check_helper(helper,inputs)
        primitive=controller.joinpath(*safe_parts(go['primitive']['path']));require(sha(primitive)==go['primitive']['sha256'],'Primitive source differs')
        runtime=dict(status='ROOT_BOUND_FAILED438_REDUCEAT_RUNTIME_V1',authorization_path=str(args.authorization.resolve()),authorization_sha256=sha(args.authorization),inputs_path=str(args.inputs.resolve()),stage=str(stage),controller=str(controller),snapshot=str(snapshot),helper_receipt=str(helper.resolve()),helper_receipt_sha256=sha(helper),installed_runtime_sha256=runtime_tree,identity_after_path=str(results/'entry-identity-after.json'),terminal_path=str(results/'primitive-final-terminal.json'))
        write_new(results/'runtime-binding.json',runtime)
        child=dict(status=go['primitive']['status'],runner_sha256=go['primitive']['sha256'],case=inputs['case'],workers=1,caps=inputs['caps'],source_revision=go['source_revision'],producer_run=go['producer_run'],primitive_only=True,retained_data_sha256=inputs['retained_study']['input_sha256'],hosted_entry=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),runtime_binding=dict(path=str(results/'runtime-binding.json'),sha256=sha(results/'runtime-binding.json')),software_policy=go['software_policy'],output=str(results/'reduceat-output'),issued_utc=go['issued_utc'],deadline_utc=go['deadline_utc'])
        write_new(results/'primitive-GO.json',child)
    elif args.phase=='run':
        import subprocess
        child_path=results/'primitive-GO.json';child=json.loads(child_path.read_bytes());require(child['runner_sha256']==go['primitive']['sha256'],'Child authority differs')
        primitive=controller.joinpath(*safe_parts(go['primitive']['path']))
        command=[sys.executable,'-B','-X','utf8','-I','-S',str(primitive),'--go',str(child_path),'--go-sha256',sha(child_path)]
        write_new(results/'primitive-command.json',command)
        with (results/'primitive.log').open('xb')as log:done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        write_new(results/'primitive-exit.json',dict(exit=done.returncode))
        secondary=None
        try:verify_sources(controller,snapshot,inputs);verify_package(stage,inputs)
        except BaseException as error:secondary=dict(type=type(error).__name__,error=str(error))
        write_new(results/'outer-post-identity.json',dict(passed=secondary is None,secondary=secondary,child_primary_exit=done.returncode,FP_calls=0))
        finish_child(done.returncode, secondary)


if __name__=='__main__':main()
