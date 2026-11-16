"""One retained56word add.reduceat study; default UNBOUND, no VI or log/dot."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

STATUS='ROOT_GO_VI_RETAINED_ROW_REDUCEAT_V1'
CASE=dict(id='vi-retained-first-xlogx-row-reduceat',operation='numpy.add.reduceat',dtype='<f8',input_shape=[56],output_shape=[7],starts=[0,8,16,24,32,40,48])
CAPS=dict(reduceat=1,returned=1,frombuffer_transport=2,original_VI=0,native_VI=0,log2=0,dot=0,fixtures=0,sampler=0,timing=0,warmup=0,CPU_masks=0,retries=0,FP_save=5,FP_restore=2,F1_backend=1,F1_ABI_query=1,F1_glibc_query=1,software_getters=6,software_setters=3)


def require(value,message):
    if not value:raise RuntimeError(message)


def check_go(go,pin,now):
    require(isinstance(go,dict)and go.get('status')==STATUS,'UNBOUND: distinct one-primitive GO required')
    require(go.get('runner_sha256')==pin and go.get('case')==CASE and go.get('caps')==CAPS and go.get('workers')==1,'Runner/case/budget differs')
    require(go.get('source_revision')=='8ac153c6659a3056e3e6ac18063213af4642a768'and go.get('producer_run')==43817382620 and go.get('primitive_only')is True,'Wrong product/purpose')
    require(isinstance(go.get('retained_data_sha256'),str)and len(go['retained_data_sha256'])==64,'Retained input unbound')
    for key in ['hosted_entry','runtime_binding']:
        require(go.get(key,{}).get('path')and len(go[key].get('sha256')or'')==64,'Entry/runtime unbound')
    issued=datetime.fromisoformat(go['issued_utc']);deadline=datetime.fromisoformat(go['deadline_utc'])
    require(issued.utcoffset()==deadline.utcoffset()==timedelta(0)and issued<=now<deadline and timedelta(0)<deadline-issued<=timedelta(hours=1),'Expired/nonUTC GO')


def load_pinned(reference,name):
    path=Path(reference['path']);require(path.is_absolute()and hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256'],'Pinned source changed')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run_study(ctx,go):
    require(ctx['status']=='ROOT_AUTHENTICATED_RETAINED_ROW_REDUCEAT_ENTRY_V1','Primitive entry unbound')
    counts={key:0 for key in CAPS if key not in ('F1_ABI_query','F1_glibc_query')}
    counts.update(ctx['entry_transport_counts']);primary=None;primary_tb=None;secondary=[];output=None;raw_output=None;outer=None;policy=None;libc=helper=None
    record=dict(schema='vi-retained-row-reduceat-terminal-v1',identity=ctx['identity'],status='STARTED_PRIMITIVE_ONLY',counts=counts,controlled_counts_only=True,model_comparison_is_not_gold=True,primary=None,secondary=secondary,outputs={},FP_saves=[],FP_restores=[],completed_primitive=False,final_restore_attempted=False,final_restore_verified=False,source_bounds_not_live_counters=dict(F1_ABI_query=1,F1_glibc_query=1),scope='After entry prepare/import/transport through final bodyrestore; before entry.finish; no across-finish or pre-import FP claim')
    def remember(error,stage):
        nonlocal primary,primary_tb
        note=dict(stage=stage,type=type(error).__name__,error=str(error))
        if primary is None:primary=error;primary_tb=error.__traceback__;record['primary']=note
        else:secondary.append(note)
    def bump(key):
        require(counts[key]<CAPS[key],'Controlled call budget exceeded: '+key);counts[key]+=1
    def save(label):
        bump('FP_save');state=ctx['fenv'].FEnv();rc=libc.fegetenv(ctx['ctypes'].byref(state))
        row=dict(label=label,result=rc,raw32=bytes(state).hex()if rc==0 else None);record['FP_saves'].append(row)
        require(rc==0,'FP snapshot failed: '+label);return state
    def restore(label):
        bump('FP_restore');rc=helper.reference_fenv_restore(ctx['ctypes'].byref(outer),32)
        row=dict(label=label,result=rc,expected_raw32=bytes(outer).hex(),actual_raw32=None,exact=False);record['FP_restores'].append(row)
        state=save(label+'-readback');row['actual_raw32']=bytes(state).hex();row['exact']=rc==0 and bytes(state)==bytes(outer)
        require(row['exact'],'Caller32 restoration failed: '+label)
    np=ctx['np']
    def software():
        bump('software_getters');errors=np.geterr();bump('software_getters');handler=np.geterrcall();bump('software_getters');buffer=np.getbufsize();return errors,handler,buffer
    if not ctx['eligible']:
        record.update(status='INELIGIBLE_NEW_HOST_ZERO_PRIMITIVES',pending_reason='Captured CPU/features/source/binary eligibility differs',FP_scope_entered=False,bounds_passed=True)
        ctx['_primitive_terminal']=record
        try:ctx['post_identity']();record['post_identity']=True
        except BaseException as error:
            remember(error,'ineligible-post-identity');record['status']='PRIMITIVE_PROTOCOL_FAILURE_RETAINED';raise
        return record
    try:
        bump('F1_backend');libc,helper=ctx['fenv'].backend()
        outer=save('outer-natural')
        record['FP_scope_entered']=True
        policy=software();require(policy[0]==go['software_policy']['errors']and policy[1]is None and policy[2]==go['software_policy']['buffer'],'Natural software context differs')
        # Check numeric controls only AFTER the first actual FP read; no seeding.
        require((outer.control&0xc00)==0 and (outer.mxcsr&0xe040)==0,'Natural nearest/gradual FP policy required')
        true_add=np.add;operation=true_add.reduceat
        require(type(np.add)is np.ufunc and operation.__self__ is np.add and operation.__name__=='reduceat','Current true add.reduceat required')
        before=save('before-primitive');record['before_raw32']=bytes(before).hex()
        notices=[]
        try:
            with warnings.catch_warnings(record=True)as notices:
                warnings.simplefilter('always')
                try:
                    require(datetime.now(timezone.utc)<datetime.fromisoformat(go['deadline_utc']),'Primitive GO expired')
                    bump('reduceat');output=operation(ctx['values'],ctx['starts']);ctx['_primitive_return']=output;bump('returned')
                    # Preserve value AND rawwords before after-snapshot/context exit.
                    raw_output=output.tobytes();record['outputs']['actual']=dict(raw_hex=raw_output.hex(),sha256=hashlib.sha256(raw_output).hexdigest())
                except BaseException as error:remember(error,'primitive-or-immediate-output')
        except BaseException as error:remember(error,'warning-context')
        finally:
            try:after=save('after-primitive');record['after_raw32']=bytes(after).hex()
            except BaseException as error:remember(error,'post-primitive-FP')
            try:restore('first-restore')
            except BaseException as error:remember(error,'first-restore')
        if output is not None:
            require(type(output)is np.ndarray and output.shape==(7,)and output.dtype.str=='<f8'and output.dtype.metadata is None and output.strides==(8,)and output.nbytes==56 and output.base is None and output.flags.c_contiguous and output.flags.owndata and output.flags.writeable and output.flags.aligned,'Output metadata differs')
            require(raw_output is not None and len(raw_output)==56,'Seven output words required')
            address=output.__array_interface__['data'][0]
            source_ranges=[(value.__array_interface__['data'][0],value.nbytes)for value in(ctx['values'],ctx['starts'])]
            require(all(address+56<=start or start+size<=address for start,size in source_ranges),'Output aliases retained input')
            record['outputs']['actual'].update(type=[type(output).__module__,type(output).__name__],dtype=output.dtype.str,shape=list(output.shape),strides=list(output.strides),nbytes=output.nbytes,owned=True,writeable=True,aligned=True,no_input_alias=True)
            require(ctx['values'].tobytes()==ctx['input_raw']and ctx['starts'].tobytes()==ctx['starts_raw'],'Readonly retained inputs changed')
            model=bytes.fromhex(ctx['model_raw_hex']);require(len(model)==56,'Model comparison size differs')
            differences=[i for i in range(7)if raw_output[8*i:8*i+8]!=model[8*i:8*i+8]]
            record.update(completed_primitive=True,model_raw_hex=model.hex(),different_rows=differences,observation='MATCHES_CPP_MODEL'if not differences else'DIFFERS_FROM_CPP_MODEL',new_host_separate_witness=True)
        record['warnings']=[dict(type=type(w.message).__name__,message=str(w.message))for w in notices];require(not notices,'Unexpected primitive warning')
        require(np.add is true_add,'Primitive provider identity changed')
        after_policy=software();require(after_policy[0]==policy[0]and after_policy[1]is policy[1]and after_policy[2]==policy[2],'Primitive changed software policy')
    except BaseException as error:remember(error,'protocol')
    finally:
        if policy is not None:
            try:
                bump('software_setters');np.seterr(**policy[0]);bump('software_setters');np.seterrcall(policy[1]);bump('software_setters');np.setbufsize(policy[2])
            except BaseException as error:remember(error,'software-restore')
        try:ctx['post_identity']();record['post_identity']=True
        except BaseException as error:remember(error,'post-identity')
        if outer is not None:
            record['final_restore_attempted']=True
            try:restore('final-restore');record['final_restore_verified']=True
            except BaseException as error:remember(error,'final-restore')
        bounds=all(counts.get(k,0)<=v for k,v in CAPS.items()if k not in('F1_ABI_query','F1_glibc_query'))
        full=counts['FP_save']==5 and counts['FP_restore']==2
        record.update(bounds_passed=bounds,full_FP_counts_passed=full if counts['returned']==1 else None,pending_reason=None if record['completed_primitive']and full and record['final_restore_verified']else'Partial or failed primitive protocol')
        try:
            require(bounds,'Primitive bounds failed')
            if counts['returned']==1:require(full and record['final_restore_verified'],'Completed primitive final FP protocol failed')
        except BaseException as error:remember(error,'final-budget-gate')
        record['status']='OBSERVATION_COMPLETE_'+record['observation']if primary is None and record['completed_primitive']else'PRIMITIVE_PROTOCOL_FAILURE_RETAINED'
        ctx['_primitive_terminal']=record
    if primary is not None:raise primary.with_traceback(primary_tb)
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--go',type=Path);parser.add_argument('--go-sha256');cli=parser.parse_args(argv)
    require(cli.go is not None and cli.go_sha256 is not None,'UNBOUND: no imports/helper/primitive authority')
    raw=cli.go.read_bytes();require(hashlib.sha256(raw).hexdigest()==cli.go_sha256,'GO bytes changed');go=json.loads(raw)
    check_go(go,hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),datetime.now(timezone.utc))
    require(sys.flags.isolated and sys.flags.no_site and not sys.flags.optimize,'Use -B -Xutf8 -I -S')
    runtime_raw=Path(go['runtime_binding']['path']).read_bytes();require(hashlib.sha256(runtime_raw).hexdigest()==go['runtime_binding']['sha256'],'Runtime binding changed')
    entry=load_pinned(go['hosted_entry'],'_retained_reduceat_entry');ctx=entry.prepare(go,json.loads(runtime_raw));primary=primary_tb=None;terminal=None
    try:terminal=run_study(ctx,go)
    except BaseException as error:primary,primary_tb=error,error.__traceback__
    finally:
        if '_primitive_terminal'not in ctx:ctx['_primitive_terminal']=dict(schema='vi-retained-row-reduceat-terminal-v1',status='PRIMITIVE_PROTOCOL_EARLY_FAILURE',completed_primitive=False,final_restore_verified=False,pending_reason='Body did not publish a final FP observation')
        try:entry.finish(ctx)
        except BaseException as error:
            if primary is None:primary,primary_tb=error,error.__traceback__
            else:primary.add_note('Primitive entry finish also failed: '+str(error))
    if primary is not None:raise primary.with_traceback(primary_tb)
    print(json.dumps(terminal,indent=2))


if __name__=='__main__':main()
