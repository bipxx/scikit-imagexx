"""Inert exception/codec/authority controls; no NumPy or target bodies."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('vi_value_diagnostic', ROOT / 'tools/vi_cohort_value_diagnostic.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


class Dtype:
    hasobject = False
    metadata = None
    def __init__(self, value='<f8'): self.str=value; self.descr=[('',value)]
    def __eq__(self, other): return isinstance(other,Dtype)and self.str==other.str


class ndarray:
    __module__ = 'numpy'  # Inert supplied stand-in, never an actual NumPy object.
    def __init__(self, raw, dtype='<f8'):
        self.raw=raw; self.dtype=Dtype(dtype); self.shape=(2,); self.strides=(len(raw)//2,); self.nbytes=len(raw)
        self.flags=SimpleNamespace(owndata=True,writeable=True,c_contiguous=True,f_contiguous=False)
        self.base=None
    def tobytes(self): return self.raw


class FP:
    def __init__(self): self.raw=b'A'*32; self.restores=0
    def save(self): return self.raw
    def restore(self, saved): self.raw=bytes(saved); self.restores+=1


def fake_probe(tmp_path, fp, mismatch, *, codec=False, successful=False):
    # Reuse the exact failing assertion AST, not an invented comparison interface.
    path=ROOT/'benchmarks/variation_of_information_execution.py'
    source=path.read_text();tree=ast.parse(source)
    assertion=next(n for n in ast.walk(tree)if isinstance(n,ast.Assert)
                   and 'expected.dtype' in ast.get_source_segment(source,n))
    statement=ast.get_source_segment(source,assertion)
    counts=dict(original=0,native=0);errors=[]
    expected=ndarray(struct.pack('<dd',1.,2.))
    actual=ndarray(struct.pack('<dd',1.,3.)if mismatch=='word'else struct.pack('<ff',1.,2.)if mismatch=='dtype'else expected.raw,
                   '<f4'if mismatch=='dtype'else'<f8')
    if codec:actual.dtype.hasobject=True
    def original(): counts['original']+=1;fp.raw=b'R'*32;return expected
    def native(): counts['native']+=1;fp.raw=b'N'*32;return actual
    body='''def execute():
    args=(INPUT,INPUT);spec={'id':'INERT_FIRST_CASE'};before=[{'payload':'saved-source-fingerprint'}]*2
    caller=FP.save();reference_before=native_before={'control':895,'status':32,'mxcsr':8096}
    expected=ORIGINAL();reference_FP={'control':895,'status':32,'mxcsr':8096}
    actual=NATIVE();native_FP={'control':895,'status':32,'mxcsr':8096}
    returns=['label_pair_counts_preweighted','variation_of_information_csr'];requirements={n:1 for n in returns}
    accepted=suffix=[True]
    try:
        ASSERTION
    finally:FP.restore(caller)
    VALIDATE(locals())
    SAMPLER()
'''.replace('ASSERTION',statement)
    holder=SimpleNamespace()
    class StopBeforeBody(BaseException):pass
    sampler_calls=[]
    def validate_and_stop(values):
        assert holder.probe.validate_native_entry(values)is values
        raise StopBeforeBody()
    def sampler():sampler_calls.append(True);raise AssertionError('INERT_SAMPLER_BODY_FORBIDDEN')
    ns=dict(INPUT=ndarray(struct.pack('<qq',0,1),'<i8'),FP=fp,ORIGINAL=original,NATIVE=native,np=SimpleNamespace(dtype=lambda name:Dtype()),
            VALIDATE=validate_and_stop,SAMPLER=sampler)
    exec(compile(body,str(path),'exec'),ns)
    terminal=tmp_path/'terminal.json'
    probe_path=ROOT/'tools/vi_first_cohort_probe.py'
    main='''def main():
    counts=COUNTS;restored=False;outer=FP.save()
    try:
        try:EXECUTE()
        except STOP:pass
    except BaseException as error:ERRORS.append(error);raise
    finally:FP.restore(outer);restored=True
    TERMINAL.write_text(JSON.dumps(dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',counts=counts,
        sampler_body_calls=0,timed_batches=0,outer_state_restored=True,poststop_source_authority_unchanged=True)))
'''
    pns=dict(COUNTS=counts,FP=fp,EXECUTE=ns['execute'],ERRORS=errors,TERMINAL=terminal,JSON=json,STOP=StopBeforeBody)
    exec(compile(main,str(probe_path),'exec'),pns)
    validations=[]
    def validate(values):validations.append(values);return values
    holder.probe=SimpleNamespace(main=pns['main'],__file__=str(probe_path),validate_native_entry=validate,validations=validations,sampler_calls=sampler_calls)
    return holder.probe,path,terminal,counts,errors,expected,actual


@pytest.mark.parametrize('mismatch',('dtype','word'))
def test_actual_wrapper_after_old_finally_retains_words_FP_counts_and_primary(tmp_path,mismatch):
    fp=FP();probe,path,terminal,counts,errors,expected,actual=fake_probe(tmp_path,fp,mismatch)
    output=tmp_path/'diagnostic.json';oldargv=sys.argv;oldpath=list(sys.path)
    with pytest.raises(AssertionError)as caught:
        D.run_diagnostic(probe,['inert-probe'],fp.save,fp.restore,output,path,terminal)
    record=json.loads(output.read_bytes())
    assert caught.value is errors[0]and counts==dict(original=1,native=1)
    assert record['probe_restored']is True and record['probe_counts']==counts
    assert record['expected']['raw_hex']==expected.raw.hex()and record['actual']['raw_hex']==actual.raw.hex()
    assert record['expected']['dtype']=='<f8'and record['actual']['dtype']==actual.dtype.str
    assert record['strict_trace']['reference_FP']==record['strict_trace']['native_FP']
    assert record['strict_trace']['returns']==['label_pair_counts_preweighted','variation_of_information_csr']
    assert record['own_before_raw32']==record['after_old_probe_raw32']==record['own_restored_raw32']==(b'A'*32).hex()
    assert fp.raw==b'A'*32 and sys.argv is oldargv and sys.path==oldpath
    assert record['target_calls_added_by_wrapper']==record['sampler_body_calls_authorized']==0


@pytest.mark.parametrize('fault',('codec','writer','final-authority','restore'))
def test_failure_diagnostics_preserve_primary_and_no_extra_calls(tmp_path,monkeypatch,fault):
    fp=FP();probe,path,terminal,counts,errors,_,_=fake_probe(tmp_path,fp,'word',codec=fault=='codec')
    output=tmp_path/'diagnostic.json';writer=D.write_record
    if fault=='writer':
        def fail_primary(path,value):
            if Path(path)==output:
                output.write_text('{"partial":');raise OSError('INERT_WRITER')
            writer(path,value)
        monkeypatch.setattr(D,'write_record',fail_primary)
    restore=fp.restore
    if fault=='restore':
        def once(saved):
            if fp.restores==2:fp.restores+=1;raise ValueError('INERT_RESTORE')
            restore(saved)
        # Probe uses fp.restore directly; wrapper's first restore is call3.
        restore_for_wrapper=once
    else:restore_for_wrapper=restore
    def final_check():
        if fault=='final-authority':raise ValueError('INERT_AUTHORITY_CHANGED')
    with pytest.raises(AssertionError)as caught:
        D.run_diagnostic(probe,['inert'],fp.save,restore_for_wrapper,output,path,terminal,final_check)
    record=json.loads((output.with_suffix('.failure.json')if fault=='writer'else output).read_bytes())
    assert caught.value is errors[0]and counts==dict(original=1,native=1)and fp.raw==b'A'*32
    assert record['probe_counts']==counts and record['strict_trace']['native_FP']['status']==32
    assert record['after_old_probe_raw32']==(b'A'*32).hex()
    if fault=='writer':assert record['writer_error']['type']=='OSError'and output.read_text()=='{"partial":'
    else:assert record['secondary_errors']


def test_successful_sampler_stop_retains_validated_values_without_extra_calls(tmp_path):
    fp=FP();probe,path,terminal,counts,_,expected,actual=fake_probe(tmp_path,fp,'none',successful=True)
    validator=probe.validate_native_entry
    record=D.run_diagnostic(probe,['inert'],fp.save,fp.restore,tmp_path/'diagnostic.json',path,terminal)
    assert record['status']=='PASS_OLD_PROBE_SAMPLER_STOP_NO_TIMING'
    assert record['terminal']['counts']==dict(original=1,native=1)and counts==dict(original=1,native=1)
    assert record['value_frame_found']is True and record['value_frame_source']=='validated_sampler_entry'
    assert record['expected']['raw_hex']==expected.raw.hex()==record['actual']['raw_hex']
    assert record['actual']['flags']['owndata']and record['actual']['base_type']is None
    assert len(record['input_before'])==len(record['input_after'])==2
    assert record['input_after'][0]['dtype']=='<i8'and record['input_after'][0]['payload_sha256']==hashlib.sha256(struct.pack('<qq',0,1)).hexdigest()
    assert record['strict_trace']['reference_FP']==record['strict_trace']['native_FP']
    assert record['probe_restored']and len(probe.validations)==1 and probe.validate_native_entry is validator and not probe.sampler_calls
    assert record['target_calls_added_by_wrapper']==record['sampler_body_calls_authorized']==0 and fp.raw==b'A'*32


def test_codecs_follow_old_restoration_and_cannot_rewrite_FP_snapshots(tmp_path,monkeypatch):
    fp=FP();probe,path,terminal,counts,_,_,_=fake_probe(tmp_path,fp,'none')
    original=D.array_words;calls=[]
    def codec(value,*,output):
        assert fp.restores>=2 and fp.raw==b'A'*32
        calls.append(output)
        # The retained strict state must be independent of later codec effects.
        probe.validations[0]['native_FP']['status']=999
        return original(value,output=output)
    monkeypatch.setattr(D,'array_words',codec)
    record=D.run_diagnostic(probe,['inert'],fp.save,fp.restore,tmp_path/'diagnostic.json',path,terminal)
    assert calls==[False,False,True,True]and counts==dict(original=1,native=1)
    assert record['strict_trace']['native_FP']['status']==32 and fp.raw==b'A'*32


def test_old_probe_raw32_is_saved_before_terminal_JSON_codec(tmp_path,monkeypatch):
    fp=FP();probe,path,terminal,counts,_,_,_=fake_probe(tmp_path,fp,'none')
    loads=D.json.loads
    def codec(value):fp.raw=b'C'*32;return loads(value)
    monkeypatch.setattr(D.json,'loads',codec)
    record=D.run_diagnostic(probe,['inert'],fp.save,fp.restore,tmp_path/'diagnostic.json',path,terminal)
    assert record['after_old_probe_raw32']==(b'A'*32).hex()
    assert fp.raw==b'A'*32 and counts==dict(original=1,native=1)and not probe.sampler_calls


def test_terminal_codec_failure_preserves_primary_and_already_validated_values(tmp_path,monkeypatch):
    fp=FP();probe,path,terminal,counts,_,expected,_=fake_probe(tmp_path,fp,'none')
    loads=D.json.loads;primary=ValueError('INERT_TERMINAL_CODEC')
    def codec(value):
        if value==terminal.read_bytes():raise primary
        return loads(value)
    monkeypatch.setattr(D.json,'loads',codec)
    output=tmp_path/'diagnostic.json'
    with pytest.raises(ValueError)as caught:
        D.run_diagnostic(probe,['inert'],fp.save,fp.restore,output,path,terminal)
    record=loads(output.read_bytes())
    assert caught.value is primary and record['expected']['raw_hex']==expected.raw.hex()
    assert record['after_old_probe_raw32']==(b'A'*32).hex()and record['own_full32_restored']
    assert counts==dict(original=1,native=1)and not probe.sampler_calls


@pytest.mark.parametrize('fault',('codec','writer','final-authority','restore','fallback','CPU-metadata'))
def test_success_tail_failure_is_nonzero_without_repeating_pair(tmp_path,monkeypatch,fault):
    fp=FP();probe,path,terminal,counts,_,_,_=fake_probe(tmp_path,fp,'none',codec=fault=='codec')
    output=tmp_path/'diagnostic.json';writer=D.write_record;restore=fp.restore
    if fault in('writer','fallback'):
        def failing_writer(path,value):
            if Path(path)==output or fault=='fallback':raise OSError('INERT_WRITER')
            writer(path,value)
        monkeypatch.setattr(D,'write_record',failing_writer)
    if fault=='CPU-metadata':
        def unavailable():raise OSError('INERT_CPU_METADATA')
        monkeypatch.setattr(D,'runtime_cpu_evidence',unavailable)
    if fault=='restore':
        def once(saved):
            if fp.restores==2:fp.restores+=1;raise ValueError('INERT_RESTORE')
            restore(saved)
        restoring=once
    else:restoring=restore
    def final_check():
        if fault=='final-authority':raise ValueError('INERT_AUTHORITY')
    with pytest.raises((AssertionError,OSError,ValueError)):
        D.run_diagnostic(probe,['inert'],fp.save,restoring,output,path,terminal,final_check)
    assert counts==dict(original=1,native=1)and len(probe.validations)==1 and fp.raw==b'A'*32


def test_late_probe_identity_failure_keeps_success_values_and_primary(tmp_path):
    fp=FP();probe,path,terminal,counts,_,expected,_=fake_probe(tmp_path,fp,'none')
    old=probe.main;primary=ValueError('INERT_C1_FAILURE')
    def main():old();raise primary
    probe.main=main
    with pytest.raises(ValueError)as caught:
        D.run_diagnostic(probe,['inert'],fp.save,fp.restore,tmp_path/'diagnostic.json',path,terminal)
    record=json.loads((tmp_path/'diagnostic.json').read_bytes())
    assert caught.value is primary and record['status']=='FAILED_OLD_PROBE_NO_RETRY'
    assert record['value_frame_source']=='validated_sampler_entry'and record['expected']['raw_hex']==expected.raw.hex()
    assert counts==dict(original=1,native=1)and fp.raw==b'A'*32


def test_missing_execute_fields_are_not_invented(tmp_path):
    fp=FP();failure=ValueError('INERT_EARLY_FAILURE')
    def main():raise failure
    probe=SimpleNamespace(main=main,__file__=str(tmp_path/'probe.py'),validate_native_entry=lambda value:value)
    with pytest.raises(ValueError)as caught:
        D.run_diagnostic(probe,['inert'],fp.save,fp.restore,tmp_path/'diagnostic.json',tmp_path/'execute.py',tmp_path/'terminal.json')
    record=json.loads((tmp_path/'diagnostic.json').read_bytes())
    assert caught.value is failure and not record['value_frame_found']
    assert not{'expected','actual','input_before','input_after','driver_caller_raw32'}.intersection(record)


def test_dispatch_evidence_reads_loaded_metadata_without_import_or_kernel_calls(monkeypatch):
    features={'AVX2':True,'AVX512F':False}
    module=SimpleNamespace(__cpu_features__=features,__cpu_baseline__=('SSE2',),__cpu_dispatch__=['AVX2'])
    monkeypatch.setitem(sys.modules,'numpy.core._multiarray_umath',module)
    record=D.runtime_cpu_evidence()
    assert record['numpy_dispatch']['numpy.core._multiarray_umath']==dict(__cpu_features__=features,__cpu_baseline__=['SSE2'],__cpu_dispatch__=['AVX2'])
    assert 'not selected per-call kernel'in record['dispatch_scope']


def test_primary_survives_both_writers_failing(tmp_path,monkeypatch):
    fp=FP();probe,path,terminal,counts,errors,_,_=fake_probe(tmp_path,fp,'word')
    def fail(*args):raise OSError('INERT_BOTH_WRITERS')
    monkeypatch.setattr(D,'write_record',fail)
    with pytest.raises(AssertionError)as caught:
        D.run_diagnostic(probe,['inert'],fp.save,fp.restore,tmp_path/'diagnostic.json',path,terminal)
    assert caught.value is errors[0]and counts==dict(original=1,native=1)and fp.raw==b'A'*32
    assert any('writer and fallback failed'in note for note in caught.value.__notes__)


@pytest.mark.parametrize('fault',('none','UNBOUND','wrapper','binding','expired'))
def test_external_GO_one_way_binding_and_early_refusal(tmp_path,fault):
    root=tmp_path/'snapshot';root.mkdir();controller=tmp_path/'controller';(controller/'coverage/contracts').mkdir(parents=True)
    dependency=root/'metadata.txt';dependency.write_text('INERT_SOURCE')
    context=dict(controller_dependencies={},prepared_dependencies={'metadata.txt':D.digest(dependency)},qualified_source='a'*40)
    cp=controller/D.CONTEXT;cp.write_text(json.dumps(context))
    binding=tmp_path/'binding.json';binding.write_text(json.dumps(dict(source_root=str(root),source_revision='a'*40)))
    oldgo=tmp_path/'oldgo.json';oldgo.write_text('{}');newgo=tmp_path/'diagnostic-go.json';output=tmp_path/'output.json'
    go=dict(status='ROOT_AUTHORIZED_ONE_VI_VALUE_DIAGNOSTIC',wrapper_sha256=D.digest(D.__file__),
        api='metrics.variation_of_information',
        binding_sha256=D.digest(binding),cohort_GO_sha256=D.digest(oldgo),context_sha256=D.digest(cp),
        maximum_original_topcalls=1,maximum_native_topcalls=1,sampler_body_calls=0,timed_batches=0,
        deadline_utc='2999-01-01T00:00:00+00:00',output_path=str(output),source_revision='a'*40)
    if fault=='UNBOUND':go['status']='UNBOUND'
    elif fault in('wrapper','binding'):go[fault+'_sha256']='0'*64
    elif fault=='expired':go['deadline_utc']='2000-01-01T00:00:00+00:00'
    newgo.write_text(json.dumps(go));args=argparse.Namespace(binding=binding,root_go=oldgo,diagnostic_go=newgo,diagnostic_go_sha256=D.digest(newgo),output=output)
    if fault=='none':assert D.validate_authority(args,controller)[1]==root
    else:
        with pytest.raises(AssertionError):D.validate_authority(args,controller)
    # GO SHA is external: there is no reciprocal binding.root_GO_sha256 cycle.
    assert 'diagnostic_go_sha256'not in json.loads(binding.read_bytes())and not output.exists()


def test_frozen_probe_driver_assertion_and_dependencies_exact():
    context=json.loads((ROOT/D.CONTEXT).read_bytes())
    for name,pin in context['controller_dependencies'].items():assert D.digest(ROOT/name)==pin
    for name,pin in context['prepared_dependencies'].items():assert D.digest(ROOT/name)==pin
    tree=ast.parse((ROOT/'tools/vi_cohort_value_diagnostic.py').read_bytes())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run_diagnostic')
    assert sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='main'for n in ast.walk(run))==1
    assert context['actual_authority']is None and context['original_budget_now']==0


def test_value_capture_body_is_exact_admitted_v2_with_only_context_and_terminal_metadata():
    raw=(ROOT/'tools/vi_cohort_value_diagnostic.py').read_bytes()
    current=b'coverage/contracts/vi-cohort-value-diagnostic-v4.json'
    terminal=b"Path(binding['output_root']) / 'first-cohort/vi/terminal.json'"
    assert raw.count(current)==raw.count(terminal)==1
    raw=raw.replace(current,b'coverage/contracts/vi-cohort-value-diagnostic-v2.json',1)
    raw=raw.replace(terminal,b"Path(binding['output_root']) / 'vi/terminal.json'",1)
    assert hashlib.sha256(raw).hexdigest()=='ff7a2a9f99aa87c639e61a6be0558ecfc80668d819b0a2b2d6e564794eec6f3a'
