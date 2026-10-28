"""Prospective workload/word/observer tests with stdlib array stubs only."""
import ast
import builtins
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'benchmarks/map_array.py'


def load():
    spec=importlib.util.spec_from_file_location('map_array_benchmark_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_frozen_nine_cases_import_without_scientific_dependencies(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0] not in ('numpy','scipy','skimage','skimagexx')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    module=load();rows=module.case_specs();contract=json.loads(module.CONTRACT.read_bytes())
    assert len(rows)==len({r['case_id'] for r in rows})==9
    assert {kind:sum(r['kind']==kind for r in rows) for kind in ('dense','signed-sparse','wide','strided-input','strided-output')}=={
        'dense':3,'signed-sparse':2,'wide':2,'strided-input':1,'strided-output':1}
    assert [(tuple(r['shape']),r['input_dtype'],r['output_dtype']) for r in rows[:3]]==[(tuple([s,s]),'uint8','uint16') for s in (64,256,1024)]
    assert all(math.prod(r['shape'])<=1048576 for r in rows)
    assert sum(r['preallocated_out'] for r in rows)==1 and rows[-1]['output_stride_bytes']==8
    assert (contract['seed'],contract['repeat'],contract['target_ms'],contract['threads'],contract['raw_samples'])==(2026,7,19,1,126)
    assert contract['original_python_source_sha256']=='aaca3e8ec753ab84d793a39e7aa71db991d5d39dd457cba96d11b4aba1f2cfe2'


class Array:
    def __init__(self,raw,kind='u',width=1,shape=None,strides=None):
        self.raw=raw;self.dtype=SimpleNamespace(kind=kind,itemsize=width,str=('|' if width==1 else '<')+kind+str(width),metadata=None)
        self.shape=shape or (len(raw)//width,);self.strides=strides or (width,);self.base=None
        self.flags=SimpleNamespace(owndata=True,c_contiguous=True,f_contiguous=True,writeable=True)
    def tobytes(self):return self.raw


@pytest.mark.parametrize('width',(1,2,4,8,16))
def test_oracle_last_duplicate_missing_signed_and_opaque_words(width):
    image=Array(struct.pack('=bbbb',-1,3,9,0),'i',1)
    keys=Array(struct.pack('=bbbb',3,-1,0,-1),'i',1)
    raw=bytes(range(width*4));values=Array(raw,'u',width)
    actual=load().oracle_words(image,keys,values)
    assert actual==raw[3*width:4*width]+raw[:width]+bytes(width)+raw[2*width:3*width]


def test_oracle_uint64_highbit_max_and_missing_zero_word():
    maximum=(1<<64)-1
    image=Array(struct.pack('=QQQQ',maximum,(1<<63)+3,0,17),'u',8)
    keys=Array(struct.pack('=QQQ',(1<<63)+3,maximum,maximum),'u',8)
    values=Array(b'AAAABBBBCCCC','u',4)
    assert load().oracle_words(image,keys,values)==b'CCCCAAAA'+bytes(8)


def test_exact_output_check_retains_layout_identity_and_hashes_only():
    a,b=Array(b'\x01\x00'),Array(b'\x01\x00')
    row=load().checked_outputs(a,b,a.raw,b,a)
    assert row['result_is_out'] and row['reference_result_is_out']
    assert row['output_sha256']==row['reference_sha256']==row['oracle_sha256']==hashlib.sha256(a.raw).hexdigest()
    assert not any(isinstance(v,bytes) for v in row.values())
    with pytest.raises(AssertionError):load().checked_outputs(a,b,a.raw,b,Array(a.raw))
    with pytest.raises(AssertionError):load().checked_outputs(a,b,b'wrong',b,a)
    b.strides=(2,)
    with pytest.raises(AssertionError):load().checked_outputs(a,b,a.raw,None,None)


def test_forwarding_and_profile_guards_restore_before_timing():
    tree=ast.parse(SOURCE.read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    timer=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='measure')
    restore=next(n for n in ast.walk(main) if isinstance(n,ast.Try) and any('sys.setprofile(prior)' in ast.unparse(f) for f in n.finalbody))
    precheck=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checked_outputs')
    assert restore.end_lineno<precheck.lineno<timer.lineno
    assert 'primitive._map_array = saved' in ast.unparse(restore)  # Chained saved-target restoration.
    text=ast.unparse(main)
    assert "returns == ['map_array_apply'] and accepted == [True]" in text
    assert "frame.f_locals.get('native_result') is not None" in text
    assert 'arg is primitive_call' in text and '(reference_code, primitive_code)' in text
    assert [ast.unparse(a) for a in timer.args[2:]]==['args.repeat','19']


def test_whole_public_timing_contract_and_all_samples_losses_preserved():
    text=SOURCE.read_text();tree=ast.parse(text)
    keywords={n.arg for n in ast.walk(tree) if isinstance(n,ast.keyword)}
    assert {'inputs','output_metadata','output_sha256','reference_sha256','oracle_sha256','result_is_out',
        'actual_C_returns','native_accepted','contract_sha256','harness_sha256','core_sha256','primitive_sha256',
        'source_before','reference_pool_limits','provider_frontier'}<=keywords
    assert 'raw_samples=126' in text and 'repeat=7,target_ms=19' in text and "choices=[1]" in text
    assert '[snapshot(a) for a in arrays]==before' in text and "[a[1::2].tobytes() for a in output_backings]" in text
    assert 'source_before==source_after' in text and 'CONTRACT.read_bytes()==contract_raw' in text
    assert 'for row in rows:' in text and 'raw_hex' not in text
    scope=json.loads((ROOT/'coverage/contracts/map-array-benchmark.json').read_bytes())['scope']
    assert 'allocation is outside timing' in scope and 'No input/value alias benchmark' in scope
    assert 'all original/current raw samples and losses retained' in scope
