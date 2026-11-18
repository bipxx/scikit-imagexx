"""Unbound VI4 cold launcher and failure-only value capture; no timing hook."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

CONTEXT='coverage/contracts/vi4-public-matmul-capture-ci-h-extrema-v1.json'
PROVIDER='coverage/contracts/vi-public-matmul-provider-benchmark-ci-h-extrema-v1.json'
PROBE='tools/vi_first_cohort_probe.py'
CODEC='tools/vi_cohort_value_diagnostic.py'
EXECUTE='benchmarks/variation_of_information_execution.py'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n')as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())


def context(root):
    root=Path(root);data=json.loads((root/CONTEXT).read_bytes());capture=data['capture']
    assert capture['schema']=='vi4-public-matmul-capture-v1'
    assert sha(root/'tools/vi4_public_matmul_capture.py')==capture['runner_sha256']
    for name,pin in capture['dependencies'].items():assert sha(root/name)==pin,name
    assert capture['provider_context']==PROVIDER
    return data


def bound_probe(controller):
    """Isolated source-pinned namespace; never mutate a live old driver/module."""
    controller=Path(controller);context(controller)
    spec=importlib.util.spec_from_file_location('_vi4_fresh_bound_probe',controller/PROBE)
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    probe.CONTRACT=CONTEXT;probe.DRIVER_CONTEXT=PROVIDER
    def driver_adapter(root):
        path=Path(root)/'benchmarks/vi_public_matmul_provider_execution_v2.py'
        entry=importlib.util.spec_from_file_location('_vi4_provider_entry',path)
        module=importlib.util.module_from_spec(entry);entry.loader.exec_module(module)
        return module.executor()
    probe.driver_adapter=driver_adapter
    return probe


def read_manifest(controller):
    data=context(controller)
    assert data['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON','UNBOUND: fresh current genuine campaign required'
    assert data['producer']['revision']!='16a5b50193125199ad26cf5a89acec5babf5290c','Old16 authority is historical'
    return bound_probe(controller).read_manifest(controller)


def prepare_host_helper(snapshot,results,python,manifest):
    return bound_probe(Path(snapshot).parent/'controller').prepare_host_helper(snapshot,results,python,manifest)


def prepare_cohorts(controller,snapshot,results,manifest,python,site,deadline):
    return bound_probe(controller).prepare_cohorts(controller,snapshot,results,manifest,python,site,deadline)


def failure_path(binding,phase,output):
    assert phase in('cohort','measurement')
    root=Path(binding['output_root']);assert root.is_absolute()
    relative='first-cohort/vi/value-failure-v1.json'if phase=='cohort'else'vi.value-failure-v1.json'
    expected=root/relative;output=Path(output)
    assert output.is_absolute()and output.resolve()==expected.resolve()
    assert output.resolve().is_relative_to(root.resolve())and not output.exists()
    return output


def descriptor(value):
    result=dict(type=[type(value).__module__,type(value).__name__])
    for key in('shape','nbytes'):
        try:
            field=getattr(value,key);result[key]=list(field)if key=='shape'else int(field)
        except BaseException as error:result[key+'_unavailable']=type(error).__name__
    try:result['dtype']=value.dtype.str
    except BaseException as error:result['dtype_unavailable']=type(error).__name__
    return result


def _note(error):
    return dict(type=type(error).__name__,error=str(error))


def capture_failure(error,root,phase,output,configuration):
    """Only observed traceback locals; no target/factory/helper initialization."""
    root=Path(root);rows=[];tb=error.__traceback__
    while tb is not None:
        rows.append((tb.tb_frame,tb.tb_lineno));tb=tb.tb_next
    matches=[(f,line)for f,line in rows if f.f_code.co_name=='execute'and Path(f.f_code.co_filename).resolve()==(root/EXECUTE).resolve()]
    controller=Path(configuration.get('controller_root',root))
    probes=[f for f,_ in rows if f.f_code.co_name=='main'and Path(f.f_code.co_filename).resolve()==(controller/PROBE).resolve()]
    record=dict(schema='vi4-value-failure-v1',primary=_note(error),phase=phase,retry=False,
        capture_context_sha256=configuration['controller_context_sha256']if'controller_context_sha256'in configuration else sha(root/CONTEXT),value_frame_found=bool(matches),notes=[],
        NPY_DISABLE_CPU_FEATURES=os.environ.get('NPY_DISABLE_CPU_FEATURES'),
        added_target_calls=0,added_fixture_calls=0,added_sampler_calls=0,
        success_path_extra_FP_calls=0,FP=dict(before_codecs='unavailable',after_writer='see stderr final-restore marker'))
    save=restore=saved=None
    if len(matches)==1:
        frame,line=matches[0];values=dict(frame.f_locals);marks=configuration['phase_lines']
        record.update(execute_line=line,output_scope='earlier-untimed-pair'if line>=marks['sample_state']else'current-untimed-preflight')
        fresh=lambda name:name in values and line>marks.get(name,10**9)
        if fresh('spec'):
            spec=values['spec'];assert spec in configuration['cases']
            record['case']=dict(spec);record['index']=values['index']
        for key in('reference_before','native_before','reference_FP','native_FP','returns','requirements','accepted','suffix','before'):
            if fresh(key):record[key]=values[key]
        record['completed_case_ids']=[r['case_id']for r in values.get('rows',[])if isinstance(r,dict)and'case_id'in r]
        record['identity']={k:values[k]for k in('api','binding_pin','go_pin','source_before')if k in values}
        if'b'in values:
            record['binding_identity']={k:values['b'][k]for k in('source_revision','producer','runtime_sha256','core_sha256','interpreter','numpy','product_kind')if k in values['b']}
        for name in('reference_notices','native_notices'):
            if fresh(name):record[name]=[dict(category=n.category.__name__,message=str(n.message))for n in values[name]]
        if fresh('caller'):
            save,restore=values.get('save'),values.get('restore')
            saved=values.get('sample_state')if line>marks['sample_state']else values['caller']
            record['execute_caller_raw32']=bytes(values['caller']).hex()
            if line>marks['sample_state']:record['execute_sample_raw32']=bytes(values['sample_state']).hex()
            record['FP']['boundary']='sample'if line>marks['sample_state']else'caller'
    elif matches:
        record['notes'].append(dict(type='AmbiguousExecuteFrame',error='No output attribution'))
        values={};fresh=lambda name:False
    else:values={};fresh=lambda name:False
    if phase=='cohort'and probes:
        p=dict(probes[-1].f_locals)
        record['probe_counts']=dict(p.get('counts',{}));record['probe_restored']=p.get('restored',False)
        # The old main finally has restored its OUTER envelope after execute.
        save,restore,saved=p.get('save'),p.get('restore'),p.get('outer')
        record['FP']['boundary']='cohort-outer'
    usable=callable(save)and callable(restore)and saved is not None
    if usable:
        try:
            raw=bytes(saved)
            if len(raw)!=32:
                usable=False;raise AssertionError('Unavailable non32 envelope')
            record['FP']['saved_raw32']=raw.hex();record['FP']['before_codecs']=bytes(save())==raw
        except BaseException as secondary:record['notes'].append(dict(stage='FP-before',**_note(secondary)))
    try:
        codec=runpy.run_path(str(root/CODEC))
        # Each output is independent and precedes every input hash/CPU read.
        for name in('expected','actual'):
            if not fresh(name):record[name]=dict(unavailable='not freshly assigned at failing statement');continue
            field=record[name]=dict(descriptor=descriptor(values[name]))
            try:field['words']=codec['array_words'](values[name],output=True)
            except BaseException as secondary:field['unavailable']=_note(secondary)
        if fresh('args')and'case'in record:
            try:
                spec=record['case'];n=spec['shape'][0]*spec['shape'][1];assert n in(16384,65536)
                args=values['args'];assert len(args)==2
                for value in args:
                    assert type(value).__module__=='numpy'and type(value).__name__=='ndarray'
                    assert tuple(value.shape)==tuple(spec['shape'])and value.dtype.kind=='i'and value.dtype.itemsize==8
                    assert value.nbytes==n*8
                common=values['common'];record['input_after']=[common.array_state(a,values['np'])for a in args]
            except BaseException as secondary:record['notes'].append(dict(stage='input-hash',**_note(secondary)))
        try:record['runtime_cpu']=codec['runtime_cpu_evidence']()
        except BaseException as secondary:record['notes'].append(dict(stage='runtime-metadata',**_note(secondary)))
    except BaseException as secondary:record['notes'].append(dict(stage='codec-load',**_note(secondary)))
    finally:
        if usable:
            try:restore(saved);record['FP']['after_codecs']=True
            except BaseException as secondary:record['notes'].append(dict(stage='FP-after-codecs',**_note(secondary)))
    try:
        Path(output).parent.mkdir(parents=True,exist_ok=True);write_json(output,record)
    finally:
        if usable:
            try:restore(saved)
            except BaseException as secondary:sys.stderr.write('VI4_CAPTURE_FINAL_RESTORE_FAIL '+json.dumps(_note(secondary))+'\n')
            else:sys.stderr.write('VI4_CAPTURE_FINAL_RESTORE_OK\n')


def run_catching(script,argv,root,phase,output,configuration):
    oldargv=sys.argv;oldpath=list(sys.path)
    try:
        sys.argv=list(argv);sys.path[0:1]=[str(Path(script).parent)]
        try:
            if phase=='cohort':
                # Fresh module namespace binds the new route; old main body and
                # its source/role/helper/envelope checks remain unchanged.
                return bound_probe(configuration.get('controller_root',root)).main()
            return runpy.run_path(str(script),run_name='__main__')
        except BaseException as error:
            try:capture_failure(error,root,phase,output,configuration)
            except BaseException as secondary:sys.stderr.write('VI4_CAPTURE_SECONDARY '+json.dumps(_note(secondary))+'\n')
            raise
    finally:sys.argv=oldargv;sys.path[:]=oldpath


def _post_identity(controller,snapshot,results,manifest,python,label,pin):
    parent=Path(controller)/CONTEXT;runtime=Path(snapshot).parent/'stage/runtime.json'
    assert sha(parent)==pin and json.loads(parent.read_bytes())==manifest and sha(runtime)==manifest['runtime_sha256']
    command=[str(python),str(Path(snapshot)/'tools/ci_measurement_artifacts.py'),'verify','--wheel-dir',str(Path(snapshot).parent/'stage/wheels'),'--artifact',str(runtime)]
    with(Path(results)/(label+'-identity-after.log')).open('w')as log:checked=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    write_json(Path(results)/(label+'-identity-after-exit.json'),dict(exit=checked.returncode))
    assert checked.returncode==0,'Postbatch installed core/runtime/SDK verifier failed'
    assert sha(runtime)==manifest['runtime_sha256']and sha(parent)==pin


def _child(controller,snapshot,results,manifest,python,label,phase,command):
    parent_pin=sha(Path(controller)/CONTEXT)
    results=Path(results);binding=json.loads(Path(command[command.index('--binding')+1]if phase=='cohort'else command[command.index('--root-binding')+1]).read_bytes())
    output=Path(binding['output_root'])/('first-cohort/vi/value-failure-v1.json'if phase=='cohort'else'vi.value-failure-v1.json')
    failure_path(binding,phase,output)
    actual=[str(python),str(Path(controller)/'tools/vi4_public_matmul_capture.py'),'child','--root',str(Path(controller)),'--phase',phase,'--failure-output',str(output),'--',*command[1:]]
    write_json(results/(label+'-command.json'),command);write_json(results/(label+'-capture-command.json'),actual)
    with(results/(label+'.log')).open('w')as log:done=subprocess.run(actual,stdout=log,stderr=subprocess.STDOUT)
    primary=None
    try:write_json(results/(label+'-exit.json'),dict(exit=done.returncode))
    except BaseException as error:primary=error
    secondary=None
    try:_post_identity(controller,snapshot,results,manifest,python,'first-cohort'if phase=='cohort'else label,parent_pin)
    except BaseException as error:
        secondary=error
        try:write_json(results/(label+'-identity-failure.json'),dict(**_note(error),retry=False,child_exit=done.returncode))
        except BaseException as note:sys.stderr.write('VI4_IDENTITY_SECONDARY '+str(note)+'\n')
    if done.returncode:raise SystemExit(done.returncode)
    if primary is not None:raise primary
    if secondary is not None:raise secondary


def run_cohorts(controller,results,manifest,python,authorities):
    probe=bound_probe(controller);results=Path(results);terminals=[]
    assert set(manifest['first_cohort']['routes'])=={'vi'}
    for cohort in manifest['first_cohort']['routes']:
        authority=authorities[cohort]
        command=[str(python),str(Path(controller)/manifest['first_cohort']['probe_path']),'--binding',authority['binding'],'--root-go',authority['go'],'--route',cohort]
        _child(controller,Path(controller).parent/'snapshot',results,manifest,python,'first-cohort-'+cohort,'cohort',command)
        terminal=json.loads((results/'first-cohort'/cohort/'terminal.json').read_bytes())
        assert terminal['status']=='PASS_REAL_CLI_STOP_BEFORE_BODY'and terminal['route']==cohort
        assert terminal['source']==manifest['first_cohort']['source_revision']and terminal['counts']==dict(original=1,native=1)
        assert terminal['timed_batches']==terminal['sampler_body_calls']==0 and terminal['outer_state_restored']is True
        assert terminal['poststop_source_authority_unchanged']is True
        assert terminal['root_binding_sha256']==sha(authority['binding'])
        terminals.append(terminal)
    assert len(terminals)==1
    write_json(results/'first-cohort-summary.json',dict(status='PASS',original_public_topcalls=1,native_public_topcalls=1,timed_batches=0,sampler_body_calls=0,poststop_source_authority_unchanged=True,postbatch_offline_identity_verified=True,routes=terminals))


def scoped_measurement(manifest,route,parent_sha256,authority,terminal_path,summary):
    # The root-bound parent measurement GO preauthorizes only this fixed scope,
    # conditional on the real VI cohort and the postbatch verifier. Never create
    # a timing envelope from an unbound manifest or a cohort-only GO.
    assert manifest['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'
    assert summary['status']=='PASS'and summary['postbatch_offline_identity_verified']is True and len(summary['routes'])==1
    raw=Path(terminal_path).read_bytes();done=json.loads(raw);binding_pin=sha(authority['binding'])
    assert done in summary['routes']and done['status']=='PASS_REAL_CLI_STOP_BEFORE_BODY'
    assert done['api']==route['api']and done['root_binding_sha256']==binding_pin
    assert done['counts']==dict(original=1,native=1)and done['timed_batches']==done['sampler_body_calls']==0
    assert done['outer_state_restored']is done['poststop_source_authority_unchanged']is True
    family=manifest['capture']['report_family'];admitted=manifest['families'][family]
    assert admitted==dict(contract=route['contract'],cases=len(route['cases']),case_ids=[s['id']for s in route['cases']],untimed_calls_per_case=1)
    value=json.loads(json.dumps(manifest));value['completed_first_cohort']=dict(path=str(Path(terminal_path).resolve()),sha256=hashlib.sha256(raw).hexdigest(),binding_sha256=binding_pin)
    value['scoped_parent_sha256']=parent_sha256;value['scoped_api']=route['api'];return value


def run_measurements(snapshot,results,manifest,python,authorities,deadline):
    probe=bound_probe(Path(snapshot).parent/'controller')
    assert manifest['deadline_utc']is None and manifest['execution_clock']=='ROOT_DISPATCH_INPUT_BOUND'
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    results=Path(results);controller=Path(snapshot).parent/'controller';parent=controller/CONTEXT;parent_pin=sha(parent)
    assert json.loads(parent.read_bytes())==manifest and manifest['status']=='ROOT_AUTHORIZED_ONE_INITIAL_COMPARISON'
    summary=json.loads((results/'first-cohort-summary.json').read_bytes())
    for name,route in manifest['first_cohort']['routes'].items():
        target=Path(authorities[name]['measurement']);assert not target.exists()
        scoped=scoped_measurement(manifest,route,parent_pin,authorities[name],results/'first-cohort'/name/'terminal.json',summary)
        scoped['deadline_utc']=deadline;write_json(target,scoped)
    verifier=[str(python),str(Path(snapshot)/'tools/ci_measurement_artifacts.py'),'verify','--wheel-dir',str(Path(snapshot).parent/'stage/wheels'),'--artifact',str(Path(snapshot).parent/'stage/runtime.json')]
    for cohort,command in probe.measurement_commands(snapshot,results,manifest,python,authorities,deadline).items():
        assert sha(parent)==parent_pin
        with(results/(cohort+'-identity-before.log')).open('w')as log:subprocess.run(verifier,stdout=log,stderr=subprocess.STDOUT,check=True)
        _child(controller,snapshot,results,manifest,python,cohort,'measurement',command)
        assert sha(parent)==parent_pin


def validate_reports(snapshot,results,manifest,authorities):
    return bound_probe(Path(snapshot).parent/'controller').validate_reports(snapshot,results,manifest,authorities)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('child',));parser.add_argument('--root',type=Path,required=True);parser.add_argument('--phase',choices=('cohort','measurement'),required=True);parser.add_argument('--failure-output',type=Path,required=True)
    args,command=parser.parse_known_args();_,manifest=read_manifest(args.root)
    assert command[0]=='--';command=command[1:];assert command
    flag='--binding'if args.phase=='cohort'else'--root-binding'
    binding=json.loads(Path(command[command.index(flag)+1]).read_bytes())
    output=failure_path(binding,args.phase,args.failure_output)
    assert Path(command[0]).resolve()==(args.root/PROBE).resolve()if args.phase=='cohort'else Path(command[0]).resolve()==(Path(binding['source_root'])/'benchmarks/vi_public_matmul_provider_consumer_v2.py').resolve()
    configuration=dict(manifest['capture'],controller_context_sha256=sha(args.root/CONTEXT),controller_root=str(args.root))
    run_catching(command[0],command,Path(binding['source_root']),args.phase,output,configuration)


if __name__=='__main__':main()
