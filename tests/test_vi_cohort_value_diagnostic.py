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
'''.replace('ASSERTION',statement)
    ns=dict(INPUT=ndarray(struct.pack('<qq',0,1),'<i8'),FP=fp,ORIGINAL=original,NATIVE=native,np=SimpleNamespace(dtype=lambda name:Dtype()))
    exec(compile(body,str(path),'exec'),ns)
    terminal=tmp_path/'terminal.json'
    probe_path=ROOT/'tools/vi_first_cohort_probe.py'
    main='''def main():
    counts=COUNTS;restored=False;outer=FP.save()
    try:EXECUTE()
    except BaseException as error:ERRORS.append(error);raise
    finally:FP.restore(outer);restored=True
    TERMINAL.write_text(JSON.dumps(dict(status='PASS_REAL_CLI_STOP_BEFORE_BODY',counts=counts,
        sampler_body_calls=0,timed_batches=0,outer_state_restored=True,poststop_source_authority_unchanged=True)))
'''
    pns=dict(COUNTS=counts,FP=fp,EXECUTE=ns['execute'],ERRORS=errors,TERMINAL=terminal,JSON=json)
    exec(compile(main,str(probe_path),'exec'),pns)
    return SimpleNamespace(main=pns['main'],__file__=str(probe_path)),path,terminal,counts,errors,expected,actual


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


def test_successful_sampler_stop_has_no_timing_and_no_invented_failure_words(tmp_path):
    fp=FP();probe,path,terminal,counts,_,_,_=fake_probe(tmp_path,fp,'none',successful=True)
    record=D.run_diagnostic(probe,['inert'],fp.save,fp.restore,tmp_path/'diagnostic.json',path,terminal)
    assert record['status']=='PASS_OLD_PROBE_SAMPLER_STOP_NO_TIMING'
    assert record['terminal']['counts']==dict(original=1,native=1)and counts==dict(original=1,native=1)
    assert record['value_frame_found']is False and 'expected'not in record and fp.raw==b'A'*32


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
