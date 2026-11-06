"""Pure source/binding controls; synthetic receipts grant no runtime credit."""
import ast
import copy
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import pytest
import util_views_fenv as envelope


def test_exact_source_context_public_inverse_and_FP_body():
    data=json.loads(envelope.CONTEXT.read_bytes())
    for name,pin in data['source_pins_raw_sha256'].items():assert envelope.sha(envelope.ROOT/name)==pin,name
    assert envelope.sha(envelope.ROOT/envelope.SOURCES[0])==envelope.SOURCE_SHA256
    old=envelope.project_public_source((envelope.ROOT/'tests/test_util_views_public.py').read_bytes())
    manifest=json.loads((envelope.ROOT/'coverage/contracts/util-view-operations-v1.json').read_bytes())
    assert envelope.hashlib.sha256(old).hexdigest()==manifest['source_pins_raw_sha256']['tests/test_util_views_public.py']
    previous=ast.parse(data['public_FP_inverse']['baseline']);current=ast.parse(Path(envelope.__file__).read_bytes())
    outer=next(n for n in current.body if isinstance(n,ast.FunctionDef)and n.name=='captured_call')
    before=next(n for n in previous.body[0].body if isinstance(n,ast.FunctionDef)and n.name=='run')
    after=next(n for n in outer.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
    assert data['original_budget']==data['product_algorithm_changes']==data['gold_case_method_changes']==0


def synthetic_bundle(tmp_path):
    # These known byte snapshots exercise only validator structure; they are
    # explicitly synthetic, never executed or offered as emitted helper proof.
    def dump(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return p
    initial=bytes.fromhex('7f03ffff0000ffffffffffff0000000000000000000000000000ffffa01f0000')
    rows=[]
    for rounding in(0,0x400,0x800,0xc00):
        for sticky in(0,0x20,0x05,0x3d):
            for c2 in(0,0x400):
                v=envelope.FEnv.from_buffer_copy(initial);v.control=(v.control&~0xc00)|rounding;v.status=sticky|c2;v.mxcsr=(v.mxcsr&~(0x3f|0x6000))|sticky|(rounding<<3)
                rows.append(dict(rounding=rounding,sticky=sticky,C2=c2,expected=bytes(v).hex(),restored=bytes(v).hex()))
    library=tmp_path/'libutil_views_fenv_restore.so';library.write_bytes(b'SYNTHETIC_METADATA_ONLY_NEVER_LOAD')
    control=dict(status='PASS_FULL32_CURRENT_EMITTED_HELPER',library_sha256=envelope.sha(library),ABI=envelope.ABI,original_calls=0,scientific_imports=0,failure=None,initial=initial.hex(),final=initial.hex(),positive=rows,negative=[dict(kind=k,returncode=c,exact_state_unchanged=True)for k,c in(('null',-1),('length',-1),('nonempty-x87',-2),('unmasked-x87',-3),('unmasked-SSE',-3),('reserved-MXCSR',-4))])
    dump('controls.json',control)
    toolchain=dict(path='SYNTHETIC_COMPILER',sha256='0'*64,backend='SYNTHETIC_BACKEND',backend_sha256='0'*64)
    header=tmp_path/'fenv.h';header.write_text('synthetic header')
    headers=[dict(original='/synthetic/bits/fenv.h',captured='fenv.h')]
    pins={n:envelope.sha(envelope.ROOT/n)for n in envelope.SOURCES}
    dump('build-command.json',dict(argv=[toolchain['path'],'-fno-fast-math'],compiler=toolchain,headers=headers,source_pins=pins))
    record=dict(schema='util-views-fenv-build-v1',status='PASS_SOURCE_BUILT_FENV32',ABI=envelope.ABI,source_context_sha256=envelope.sha(envelope.CONTEXT),source_pins=pins,compiler=toolchain,headers=headers,controls=['controls.json'],files={p.name:envelope.sha(p)for p in tmp_path.iterdir()if p.is_file()})
    return dump('receipt.json',record),record


def test_synthetic_metadata_validation_is_not_runtime_admission(tmp_path):
    path,_=synthetic_bundle(tmp_path)
    assert envelope.validate_receipt(path,envelope.sha(path),toolchain=False)['status']=='PASS_SOURCE_BUILT_FENV32'


@pytest.mark.parametrize('fault',('missing','digest','status','ABI','source','context','library','control-hash','control-cells','control-bytes','controls-absent','escape','toolchain'))
def test_missing_tampered_or_unqualified_receipt_refused_before_target(tmp_path,monkeypatch,fault):
    path,record=synthetic_bundle(tmp_path);digest=envelope.sha(path)
    if fault=='missing':path.unlink()
    elif fault=='digest':digest='0'*64
    elif fault=='library':(tmp_path/'libutil_views_fenv_restore.so').write_bytes(b'TAMPERED')
    elif fault=='control-hash':(tmp_path/'controls.json').write_text('{}')
    else:
        if fault=='status':record['status']='UNBOUND'
        elif fault=='ABI':record['ABI']=[0]*8
        elif fault=='source':record['source_pins'][envelope.SOURCES[0]]='0'*64
        elif fault=='context':record['source_context_sha256']='0'*64
        elif fault=='controls-absent':record['controls']=[]
        elif fault=='escape':record['files']['../outside']='0'*64
        elif fault.startswith('control-'):
            p=tmp_path/'controls.json';control=json.loads(p.read_bytes())
            if fault=='control-cells':control['positive'].pop()
            else:control['positive'][0]['restored']='00'*32
            p.write_text(json.dumps(control));record['files']['controls.json']=envelope.sha(p)
        path.write_text(json.dumps(record));digest=envelope.sha(path)
    monkeypatch.delenv('UTIL_VIEWS_FENV_HELPER',raising=False);monkeypatch.setenv('UTIL_VIEWS_FENV_MODE','source-built-v1');monkeypatch.setenv('UTIL_VIEWS_FENV_RECEIPT',str(path));monkeypatch.setenv('UTIL_VIEWS_FENV_RECEIPT_SHA256',digest)
    monkeypatch.setattr(envelope,'configure',lambda *a:pytest.fail('bad receipt reached native library load'))
    calls=[]
    with pytest.raises((AssertionError,FileNotFoundError)):
        call=envelope.captured_call(lambda:calls.append('TARGET'),{})
        call()
    assert calls==[]


def test_source_built_never_falls_back_to_old_private_library(monkeypatch):
    monkeypatch.setenv('UTIL_VIEWS_FENV_MODE','source-built-v1');monkeypatch.setenv('UTIL_VIEWS_FENV_HELPER','old-local.so')
    with pytest.raises(AssertionError,match='legacy helper forbidden'):envelope.captured_call(lambda:pytest.fail('target'),{})


@pytest.mark.parametrize('mode',('source-built-v1','unknown'))
def test_unbound_modes_fail_before_target(monkeypatch,mode):
    for name in('UTIL_VIEWS_FENV_HELPER','UTIL_VIEWS_FENV_RECEIPT','UTIL_VIEWS_FENV_RECEIPT_SHA256'):monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv('UTIL_VIEWS_FENV_MODE',mode)
    with pytest.raises(AssertionError):envelope.captured_call(lambda:pytest.fail('target'),{})


def test_nonLinux_builder_refuses_before_output_or_compile(tmp_path):
    if sys.platform=='linux':return
    result=subprocess.run([sys.executable,str(envelope.ROOT/'tests/util_views_fenv_build.py'),'--output',str(tmp_path/'never-created')],capture_output=True,text=True)
    assert result.returncode!=0 and 'Only Linux x86_64/glibc32'in result.stderr and not(tmp_path/'never-created').exists()
