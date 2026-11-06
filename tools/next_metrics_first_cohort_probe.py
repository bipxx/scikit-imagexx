"""Prospective actual-CLI probe; requires a root-bound genuine product and GO."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver_argv(root,route,binding,output,deadline):
    argv=[str(Path(root)/route['driver']),*route['argv'],'--output',str(output)]
    if route['kind']=='util-executor':
        assert '--first-cohort'in argv and '--api'in argv and '--case'in argv
        assert '--deadline-utc'not in argv
    else:argv+=['--deadline-utc',deadline]
    if route.get('root_binding_flag'):argv += [route['root_binding_flag'],str(binding)]
    return argv


def validate_util_entry(frame,route,root,counts,accepted,returns,getters,input_state,inputs_before):
    """Inspect the real execute caller after all driver parity checks, before measure body."""
    assert Path(frame.f_back.f_code.co_filename).resolve()==Path(root)/'benchmarks/next_util_execution.py'
    caller=frame.f_back.f_locals
    assert caller['cli'].first_cohort is True and caller['cli'].measurement_go is None
    assert caller['spec']==route['first_case'] and caller['api']==route['api']
    assert counts==dict(original=1,native=1)and accepted==[True]
    assert caller['accepted']==[True]and not caller['rows']
    # execute installs its admitted native-denial/C-return observer during native.
    # Its authenticated frame is the real marker evidence; outer profile is paused.
    assert returns==[]
    actual_returns=caller['returns']
    assert {k:actual_returns.count(k)for k in caller['requirements']}==caller['requirements']==route['native_return_counts']
    assert len(actual_returns)==sum(route['native_return_counts'].values())and len(getters)==route['native_worker_getters']
    assert caller['expected_state']==caller['actual_state']
    assert caller['reference_FP']==caller['native_FP']
    assert bytes(caller['save']())==bytes(caller['caller'])and len(bytes(caller['caller']))==32
    assert inputs_before==[input_state((caller['args'],caller['options']))]
    assert [frame.f_back.f_globals['array_state'](v,caller['np'])for v in caller['inputs']]==caller['before']
    if route['api']=='segmentation.join_segmentations':
        # Exact ArrayMap class identity remains open; record both real classes.
        assert route['ArrayMap_class_frontier']=='OPEN_EXACT_CLASS_IDENTITY'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--root-go', type=Path, required=True)
    parser.add_argument('--route', required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_bytes())
    assert binding.get('status') == 'ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS', 'UNBOUND: no scientific calls permitted'
    go = json.loads(args.root_go.read_bytes())
    assert go['binding_sha256'] == sha(args.binding)
    assert go['probe_sha256'] == sha(__file__)
    assert go['timed_batches'] == 0
    route = binding['routes'][args.route]
    assert go['routes'] == [route.get('execution_route',args.route)]
    assert set(binding['routes']) == {args.route}
    assert datetime.now(timezone.utc) < datetime.fromisoformat(go['deadline_utc'])
    assert go['maximum_original_topcalls'] == go['maximum_native_topcalls'] == 1
    root = Path(binding['source_root']).resolve()
    git = binding['git_argv']
    assert not os.environ.get('GIT_DIR') and not os.environ.get('GIT_WORK_TREE')
    assert subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip() == binding['source_revision']
    assert not subprocess.check_output([*git, 'status', '--porcelain'], text=True).strip()
    assert sha(binding['genuine_root_admission_path']) == binding['genuine_root_admission_sha256']
    admission = json.loads(Path(binding['genuine_root_admission_path']).read_bytes())
    assert admission['source'] == binding['source_revision']
    assert admission['passed'] is True and admission['root_accepted'] is True and admission['profiles'] == 4
    assert binding['product_kind'] == 'GENUINE_INSTALLED_WHOLE_PRODUCT'
    assert Path(sys.executable).resolve() == Path(binding['interpreter']).resolve()
    for name, pin in binding['source_files_sha256'].items():
        assert sha(root / name) == pin
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
        assert os.environ[key] == '1'
    assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
    assert not os.environ.get('SKIMAGEXX_DISABLE_SIMD')
    route = binding['routes'][args.route]
    helper_binding=Path(binding['output_root']).parent/'helper-binding.json'
    assert sha(helper_binding)==go['helper_binding_sha256']
    helper=json.loads(helper_binding.read_bytes())
    assert all(os.environ.get(k)==v for k,v in helper['env'].items())
    assert set(helper['env'])=={'UTIL_VIEWS_FENV_MODE','UTIL_VIEWS_FENV_RECEIPT','UTIL_VIEWS_FENV_RECEIPT_SHA256'}
    helper_root=Path(helper['env']['UTIL_VIEWS_FENV_RECEIPT']).parent
    for name,pin in helper['files'].items():
        path=(helper_root/name).resolve();assert path.is_relative_to(helper_root.resolve())and sha(path)==pin
    assert not os.environ.get('UTIL_VIEWS_FENV_HELPER')
    folder = Path(binding['output_root']) / args.route
    folder.mkdir(parents=True, exist_ok=False)  # Never replay a consumed route.
    output = folder / 'never-written.json'
    def save(name, value):
        (folder / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    def journal(event, role):
        with (folder / 'calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, role=role, counts=dict(counts))) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    sys.path[:0] = [binding['installed_site'], str(root / 'benchmarks'), str(root / 'tools'), str(root)]
    import numpy as np
    import skimage
    import skimagexx as xx
    from skimagexx import _compat
    assert np.__version__ == binding['numpy'] and skimage.__version__ == '0.26.0'
    assert Path(xx._core.__file__).resolve() == Path(binding['core_path']).resolve()
    assert sha(xx._core.__file__) == binding['core_sha256']
    assert Path(xx.__file__).resolve().is_relative_to(Path(binding['installed_site']).resolve())
    namespace = route['api'].split('.')[0]
    public = importlib.import_module('skimage.' + namespace)
    native_public = importlib.import_module('skimagexx.' + namespace)
    assert getattr(xx, namespace) is native_public
    original = getattr(public, route['api'].split('.')[1])
    native = getattr(native_public, route['api'].split('.')[1])
    owned = importlib.import_module(route['owned_module'])
    assert native is getattr(owned, route['api'].split('.')[1])
    acceptance = getattr(owned, route['acceptance_function'])
    assert inspect.signature(native) == inspect.signature(original)
    expected_entries = route['native_entries']
    entries = {getattr(xx._core, name): name for name in expected_entries}
    assert all(inspect.isbuiltin(entry) for entry in entries)
    getter = xx._core.get_num_threads
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    workers, fallback = xx.get_num_threads(), _compat._fallback_enabled
    assert workers == 1
    oldtrace, oldprofile, oldargv = sys.gettrace(), sys.getprofile(), sys.argv
    counts = dict(original=0, native=0)
    native_active = False
    accepted, c_returns, getters = [], [], []
    inputs_before = []
    def input_state(value):
        if isinstance(value, np.ndarray):
            return (id(value), value.dtype.str, repr(value.dtype.metadata), value.shape,
                    value.strides, value.ctypes.data, value.flags.writeable, value.tobytes())
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(input_state(item) for item in value))
        if isinstance(value, dict):
            return tuple((key, input_state(item)) for key, item in sorted(value.items()))
        return value
    stopped = {}
    sampler = root / route['sampler']
    driver = root / route['driver']
    argv = driver_argv(root,route,args.binding,output,go['deadline_utc'])
    class StopBeforeBody(BaseException):
        pass
    def profile(frame, event, value):
        if native_active and event == 'c_call':
            assert not str(getattr(value, '__module__', '')).startswith(('skimage.draw', 'skimage.metrics', 'skimage.segmentation'))
        if native_active and event == 'c_return':
            if value in entries:
                c_returns.append(entries[value])
            if value is getter:
                getters.append('get_num_threads')
    def trace(frame, event, value):
        nonlocal native_active
        role = 'original' if frame.f_code is original.__code__ else 'native' if frame.f_code is native.__code__ else None
        if event == 'call' and role:
            if role == 'original':
                assert not native_active
            assert counts[role] == 0
            counts[role] += 1
            journal('before', role)
            if role == 'original':
                parent = frame.f_back
                assert Path(parent.f_code.co_filename).resolve() == (root/'benchmarks/next_util_execution.py' if route['kind']=='util-executor' else driver)
                arguments = parent.f_locals['args'] if route['kind']=='util-executor' else (parent.f_locals['image'],) if namespace == 'segmentation' else parent.f_locals.get('arguments', parent.f_locals.get('positional'))
                inputs_before.append(input_state((arguments, parent.f_locals['options'])))
            if role == 'native':
                native_active = True
        if event == 'return' and role:
            if role == 'native':
                native_active = False
            journal('after', role)
        if event == 'return' and frame.f_code is acceptance.__code__:
            result = frame.f_locals.get('native_result')
            accepted.append(value is not None if route['kind']=='util-executor' else result is True if route.get('native_result_kind') == 'true' else result is not None)
        if event == 'call' and frame.f_code.co_name == 'measure' and Path(frame.f_code.co_filename).resolve() == sampler:
            assert counts == dict(original=1, native=1) and accepted == [True]
            assert not {'samples', 'functions', 'events', 'loops'}.intersection(frame.f_locals)
            caller = frame.f_back.f_locals
            if route['kind']=='util-executor':
                validate_util_entry(frame,route,root,counts,accepted,c_returns,getters,input_state,inputs_before)
                assert (np.geterr(),np.geterrcall(),np.getbufsize())==(policy,handler,buffer)
                assert xx.get_num_threads()==1 and not _compat._fallback_enabled
                stopped.update(source=binding['source_revision'],route=args.route,api=route['api'],counts=dict(counts),native_result_accepted=accepted,real_sampler=str(sampler),sampler_body_calls=0,timed_batches=0,public_driver_checks_completed=True,caller_full32_restored=True,ArrayMap_class_frontier=route.get('ArrayMap_class_frontier'),marker_evidence='Authenticated execute native observer returns; outer profile paused during that call')
                save('sampler-entry.json',stopped)
                raise StopBeforeBody()
            assert Path(frame.f_back.f_code.co_filename).resolve() == driver
            assert caller['spec'] == route['first_case']
            assert caller['calls'] == expected_entries and caller['accepted'] == [True]
            if namespace != 'segmentation':
                assert caller['snapshots'] == ['get_num_threads']
            assert sys.getprofile() is profile and not caller['rows']
            arguments = (caller['image'],) if namespace == 'segmentation' else caller.get('arguments', caller.get('positional'))
            assert inputs_before == [input_state((arguments, caller['options']))]
            if namespace == 'metrics':
                assert caller['error'] == 0
                assert frame.f_back.f_globals['input_state'](arguments) == caller['state']
                if route['api'] == 'metrics.contingency_table':
                    assert caller['actual'].format == caller['expected'].format == 'csr'
                    assert frame.f_back.f_globals['output_snapshot'](caller['actual']) == frame.f_back.f_globals['output_snapshot'](caller['expected'])
                else:
                    assert type(caller['actual']) is type(caller['expected']) is tuple
                    assert len(caller['actual']) == len(caller['expected']) == 3
                    assert frame.f_back.f_globals['output_snapshot'](caller['actual']) == frame.f_back.f_globals['output_snapshot'](caller['expected'])
            elif namespace == 'segmentation':
                assert frame.f_back.f_globals['snapshot'](caller['image']) == caller['frozen']
                assert caller['image'].ctypes.data == caller['address']
                assert frame.f_back.f_globals['snapshot'](caller['actual']) == frame.f_back.f_globals['snapshot'](caller['expected'])
            else:
                assert (arguments, caller['options']) == caller['input_state']
                assert caller['snapshot'](caller['actual']) == caller['snapshot'](caller['expected'])
            assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
            assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
            stopped.update(source=binding['source_revision'], route=args.route, api=route['api'],
                           counts=dict(counts), native_result_accepted=accepted,
                           real_sampler=str(sampler), sampler_body_calls=0, timed_batches=0,
                           public_driver_checks_completed=True)
            save('sampler-entry.json', stopped)
            raise StopBeforeBody()
        return trace
    save('precall.json', dict(source=binding['source_revision'], argv=argv, route=route,
                             core_path=binding['core_path'], core_sha256=binding['core_sha256'],
                             product_kind=binding['product_kind'], binding_sha256=sha(args.binding)))
    try:
        try:
            sys.settrace(trace)
            sys.setprofile(profile)
            sys.argv = argv
            runpy.run_path(str(driver), run_name='__main__')
        except StopBeforeBody:
            pass
        assert stopped and not output.exists() and not output.with_suffix('.batches.jsonl').exists()
    except BaseException as error:
        save('failure.json', dict(type=type(error).__name__, error=str(error), counts=counts,
                                 traceback=traceback.format_exc(), retry=False))
        raise
    finally:
        sys.settrace(oldtrace)
        sys.setprofile(oldprofile)
        sys.argv = oldargv
        xx.set_num_threads(workers)
        xx.set_fallback_enabled(fallback)
    save('terminal.json', dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY', **stopped,
                               outer_state_restored=xx.get_num_threads() == workers and _compat._fallback_enabled == fallback))


def read_manifest(controller):
    path=Path(controller)/'coverage/contracts/next-metrics-measurement-v2.json'
    manifest=json.loads(path.read_bytes())
    assert manifest['schema']=='next-metrics-measurement-v1'
    assert manifest['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON','UNBOUND: no helper/target/sampler execution'
    assert manifest['first_cohort']['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS'
    return path,manifest


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def prepare_host_helper(snapshot,results,python,manifest):
    # Called once outside targets/cohorts/sampler after actual producer admission.
    assert manifest['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'and manifest['first_cohort']['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS'
    source=manifest['helper']['source_sha256']
    for name,pin in source.items():assert sha(Path(snapshot)/name)==pin
    assert manifest['helper']['mode']=='source-built-v1'and manifest['helper']['prepare_once_on_actual_host']is True
    assert not os.environ.get('UTIL_VIEWS_FENV_HELPER')
    bundle=Path(results)/'util-views-fenv'
    assert not bundle.exists(),'Helper preparation cannot be replayed'
    command=[str(python),str(Path(snapshot)/'tests/util_views_fenv_build.py'),'--output',str(bundle.resolve()),'--control-python',str(python)]
    write_json(Path(results)/'helper-command.json',command)
    with (Path(results)/'helper-build.log').open('w')as log:
        done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    write_json(Path(results)/'helper-exit.json',dict(exit=done.returncode))
    assert done.returncode==0,'Helper setup failed; no cohorts authorized'
    receipt=bundle/'receipt.json';assert receipt.is_file()
    # The approved builder ran full32 positive/negative controls on this host.
    # Drivers independently validate this external digest+receipt before targets.
    env=dict(UTIL_VIEWS_FENV_MODE='source-built-v1',UTIL_VIEWS_FENV_RECEIPT=str(receipt.resolve()),UTIL_VIEWS_FENV_RECEIPT_SHA256=sha(receipt))
    write_json(Path(results)/'helper-binding.json',dict(env=env,files={f.relative_to(bundle).as_posix():sha(f)for f in bundle.rglob('*')if f.is_file()}))
    return env


def scoped_binding(common,cohort,route,go_path,contract_sha256):
    value=dict(common)
    value.update(routes={cohort:route},root_go_path=str(go_path),base_contract_sha256=contract_sha256)
    return value


def scoped_measurement(manifest,route,parent_sha256):
    # Existing manifest schema, exact fixed parent cases; no gate/method override.
    value=json.loads(json.dumps(manifest))
    if route['kind']=='util-executor':
        family=Path(route['driver']).stem
        admitted=manifest['families'][family]
        ids=[s['id']for s in route['cases']]
        assert set(ids)<=set(admitted['case_ids'])
        value['families'][family]=dict(contract=route['contract'],cases=len(ids),case_ids=ids)
    value['scoped_parent_sha256']=parent_sha256
    value['scoped_api']=route['api']
    return value


def prepare_cohorts(controller,snapshot,results,manifest,python,site,deadline):
    controller=Path(controller);snapshot=Path(snapshot);results=Path(results)
    first=manifest['first_cohort'];routes=first['routes']
    assert len(routes)==9 and manifest['execution_scope']['untimed_topcalls']==dict(original=9,native=9)
    assert first['source_revision']==manifest['producer']['revision']==manifest['prepared_source']['revision']
    probe=controller/first['probe_path'];assert sha(probe)==first['probe_sha256']
    admission=controller/first['root_admission_path'];stager=controller/first['root_stager_path']
    assert sha(admission)==first['root_admission_sha256']and sha(stager)==first['root_stager_sha256']
    runtime=controller.parent/'stage/runtime.json';assert sha(runtime)==manifest['runtime_sha256']
    core=Path(site)/first['core_member'];assert core.resolve().is_relative_to(Path(site).resolve())and sha(core)==manifest['core_sha256']
    common=dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',prepared_source=manifest['prepared_source'],producer=manifest['producer'],runtime_path=str(runtime.resolve()),runtime_sha256=manifest['runtime_sha256'],shared_CMake_sha256=manifest['prepared_source']['files_sha256']['CMakeLists.txt'],genuine_root_stager_path=str(stager.resolve()),genuine_root_stager_sha256=first['root_stager_sha256'],source_root=str(snapshot.resolve()),source_revision=first['source_revision'],git_argv=['git','-C',str(snapshot.resolve())],genuine_root_admission_path=str(admission.resolve()),genuine_root_admission_sha256=first['root_admission_sha256'],product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',interpreter=str(python),source_files_sha256=first['source_files_raw_sha256'],installed_site=str(site),numpy=json.loads(runtime.read_bytes())['runtime']['versions']['numpy'],core_path=str(core.resolve()),core_sha256=manifest['core_sha256'],output_root=str((results/'first-cohort').resolve()))
    for name,pin in common['source_files_sha256'].items():assert sha(snapshot/name)==pin
    manifest_path=controller/'coverage/contracts/next-metrics-measurement-v2.json'
    index={}
    for cohort,route in routes.items():
        folder=results/('authority-'+cohort);folder.mkdir(exist_ok=False)
        go_path=folder/'root-go.json';binding_path=folder/'binding.json'
        write_json(binding_path,scoped_binding(common,cohort,route,go_path.resolve(),sha(snapshot/route['contract'])))
        write_json(go_path,dict(binding_sha256=sha(binding_path),probe_sha256=first['probe_sha256'],routes=[route.get('execution_route',cohort)],timed_batches=0,maximum_original_topcalls=1,maximum_native_topcalls=1,deadline_utc=deadline,helper_binding_sha256=sha(results/'helper-binding.json')))
        measurement=folder/'measurement.json';write_json(measurement,scoped_measurement(manifest,route,sha(manifest_path)))
        index[cohort]=dict(binding=str(binding_path.resolve()),go=str(go_path.resolve()),measurement=str(measurement.resolve()))
    write_json(results/'cohort-authorities.json',index)
    return index


def run_cohorts(controller,results,manifest,python,authorities):
    results=Path(results);terminals=[]
    for cohort in manifest['first_cohort']['routes']:
        authority=authorities[cohort]
        command=[str(python),str(Path(controller)/manifest['first_cohort']['probe_path']),'--binding',authority['binding'],'--root-go',authority['go'],'--route',cohort]
        write_json(results/('first-cohort-'+cohort+'-command.json'),command)
        with (results/('first-cohort-'+cohort+'.log')).open('w')as log:done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        write_json(results/('first-cohort-'+cohort+'-exit.json'),dict(exit=done.returncode))
        assert done.returncode==0,'Stop remaining cohorts/timing; no retry: '+cohort
        terminal=json.loads((results/'first-cohort'/cohort/'terminal.json').read_bytes())
        assert terminal['status']=='PASS_REAL_CLI_STOP_BEFORE_BODY'and terminal['route']==cohort
        assert terminal['source']==manifest['first_cohort']['source_revision']and terminal['counts']==dict(original=1,native=1)
        assert terminal['timed_batches']==terminal['sampler_body_calls']==0 and terminal['outer_state_restored']is True
        terminals.append(terminal)
    assert len(terminals)==9
    write_json(results/'first-cohort-summary.json',dict(status='PASS',original_public_topcalls=9,native_public_topcalls=9,timed_batches=0,sampler_body_calls=0,routes=terminals))


def measurement_commands(snapshot,results,manifest,python,authorities,deadline):
    commands={}
    for cohort,route in manifest['first_cohort']['routes'].items():
        # ARE+contingency share exactly the original8case driver invocation.
        if cohort=='metrics-contingency':continue
        output=Path(results)/(cohort+'.json')
        command=[str(python),str(Path(snapshot)/route['driver']),'--threads','1','--repeat','7','--output',str(output)]
        if route['kind']=='util-executor':command+=['--api',route['api'],'--root-binding',authorities[cohort]['binding'],'--measurement-go',authorities[cohort]['measurement']]
        else:
            command+=['--deadline-utc',deadline]
            if route.get('root_binding_flag'):command+=[route['root_binding_flag'],authorities[cohort]['binding']]
        commands[cohort]=command
    return commands


def run_measurements(snapshot,results,manifest,python,authorities,deadline):
    results=Path(results)
    summary=json.loads((results/'first-cohort-summary.json').read_bytes());assert summary['status']=='PASS'and len(summary['routes'])==9
    verifier=[str(python),str(Path(snapshot)/'tools/ci_measurement_artifacts.py'),'verify','--wheel-dir',str(Path(snapshot).parent/'stage/wheels'),'--artifact',str(Path(snapshot).parent/'stage/runtime.json')]
    for cohort,command in measurement_commands(snapshot,results,manifest,python,authorities,deadline).items():
        with (results/(cohort+'-identity-before.log')).open('w')as log:subprocess.run(verifier,stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(results/(cohort+'-command.json'),command)
        with (results/(cohort+'.log')).open('w')as log:done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        write_json(results/(cohort+'-exit.json'),dict(exit=done.returncode))
        assert done.returncode==0,'Measurement failed; no automatic retry: '+cohort
        with (results/(cohort+'-identity-after.log')).open('w')as log:subprocess.run(verifier,stdout=log,stderr=subprocess.STDOUT,check=True)


def validate_reports(snapshot,results,manifest,authorities):
    import math,statistics
    snapshot=Path(snapshot);results=Path(results);summaries={}
    context=json.loads((snapshot/'coverage/contracts/next-util-benchmark-execution-v2.json').read_bytes())
    for cohort,route in manifest['first_cohort']['routes'].items():
        if cohort=='metrics-contingency':continue
        if route['kind']=='util-executor':specs=route['cases']
        else:specs=json.loads((snapshot/route['contract']).read_bytes())['cases']
        count=len(specs)
        report=json.loads((results/(cohort+'.json')).read_bytes());meta=report['metadata']
        assert {k:meta[k]for k in manifest['benchmark_fingerprint']}==manifest['benchmark_fingerprint']
        assert meta['core_sha256']==manifest['core_sha256']
        assert meta['threads']==1 and meta['repeat']==7 and meta['target_ms']==100
        assert meta['contract_sha256']==sha(snapshot/route['contract'])
        assert len(report['results'])==count
        if route['kind']=='util-executor':
            assert meta['root_binding_sha256']==sha(authorities[cohort]['binding'])
            assert meta['measurement_sha256']==sha(authorities[cohort]['measurement'])
            assert meta['runtime_sha256']==manifest['runtime_sha256']and meta['producer']==manifest['producer']
            assert meta['method_sha256']==context['method']['method_sha256']and meta['retain_all_losses']is True
        else:
            assert meta['raw_samples']==count*14
            assert set(meta['reference_pool_limits'].values())=={'1'}
        assert [row['case_id']for row in report['results']]==[spec['id']for spec in specs]
        journal=[json.loads(line)for line in (results/(cohort+'.batches.jsonl')).read_text().splitlines()]
        assert {event['case_id']for event in journal}=={spec['id']for spec in specs}
        for case_index,(row,spec) in enumerate(zip(report['results'],specs)):
            assert row['native_accepted'] is True
            if route['kind']!='util-executor':assert row['parameters']==spec
            else:
                assert row['api']==route['api']
                assert row['FP'].keys()=={'flags','rounding'}
                assert type(row['FP']['flags'])is int and type(row['FP']['rounding'])is int
            assert list(map(len,row['samples_ms']))==[7,7]
            assert all(math.isfinite(x)and x>0 for arm in row['samples_ms']for x in arm)
            assert list(map(len,row['loops']))==[7,7] and all(type(x)is int and x>0 for arm in row['loops']for x in arm)
            assert row['reference_ms']==statistics.median(row['samples_ms'][0])
            assert row['cpp_ms']==statistics.median(row['samples_ms'][1])
            assert row['speedup']==row['reference_ms']/row['cpp_ms']
            events=[event for event in journal if event['case_id']==row['case_id']]
            batches=row['batch_records']
            assert len(events)==2*len(batches)+1
            assert all(event['case_index']==case_index for event in events)
            timing_keys=('reference_ms','cpp_ms','speedup','samples_ms','loops','sample_labels','warmup_elapsed_ns','sample_order','batch_records')
            assert events[-1]['event']=='measure_returned' and events[-1]['timing']=={key:row[key]for key in timing_keys}
            warm=[0,0];accepted=[[],[]];counts=[[],[]];order=[]
            for before,after,batch in zip(events[0:-1:2],events[1:-1:2],batches):
                assert before['event']=='batch_before' and after==batch and after['event']=='batch_after'
                assert all(before[key]==after[key]for key in ('case_index','case_id','function','round','ordinal','loops'))
                assert after['complete'] is True and after['failure'] is None
                arm=after['function'];elapsed=after['elapsed_ns'];n=after['loops']
                assert arm in (0,1) and type(elapsed)is int and elapsed>0 and type(n)is int and n>0
                if after['phase']=='warmup':
                    assert before['requested_phase']=='warmup' and not after['accepted'] and after['round']==-1
                    warm[arm]+=elapsed
                elif after['phase']=='underlength-calibration':
                    assert before['requested_phase']=='sample' and not after['accepted'] and elapsed<100000000
                else:
                    assert before['requested_phase']=='sample' and after['phase']=='sample' and after['accepted'] and elapsed>=100000000
                    accepted[arm].append(elapsed/n/1e6);counts[arm].append(n)
                    order.append((after['round'],after['ordinal'],arm))
            assert warm==row['warmup_elapsed_ns'] and min(warm)>=250000000
            assert accepted==row['samples_ms'] and counts==row['loops']
            expected_order=[[0,1]if(round_id+case_index+1)%2==0 else[1,0]for round_id in range(7)]
            assert row['sample_order']==expected_order
            assert order==[(round_id,ordinal,arm)for round_id,arms in enumerate(expected_order)for ordinal,arm in enumerate(arms)]
        summaries[cohort]=dict(raw_samples=count*14,
            wins=[row['case_id']for row in report['results']if row['speedup']>1],
            losses=[row['case_id']for row in report['results']if row['speedup']<1],
            ties=[row['case_id']for row in report['results']if row['speedup']==1])
    assert sum(v['raw_samples']for v in summaries.values())==manifest['execution_scope']['canonical_raw_samples']
    write_json(results/'summary.json',dict(families=summaries,scope='Current one-thread public original comparison only; all losses retained; no whole-product qualification or prior-C++ speed claim.'))


if __name__ == '__main__':
    main()
