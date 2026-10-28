"""Frozen row workloads and byte/first-index oracle; no scientific imports."""
import ast, builtins, hashlib, importlib.util, json, math
from pathlib import Path
import struct
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'benchmarks/unique_rows.py'


def load():
    spec=importlib.util.spec_from_file_location('unique_rows_benchmark_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_nine_frozen_real_row_cases_import_without_scientific_dependencies(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0] not in ('numpy','scipy','skimage','skimagexx')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    module=load();raw=module.CONTRACT.read_bytes();c=json.loads(raw);rows=module.case_specs()
    assert len(rows)==len({r['case_id'] for r in rows})==9
    assert [r['shape'] for r in rows]==[[256,2],[65536,2],[65536,3],[65536,3],[32768,32],[16384,8],[16384,4],[8192,8],[65536,4]]
    assert [r['dtype'] for r in rows]==['uint8','uint8','float32','float64','uint8','float64','float32','float64','uint8']
    assert all(r['shape'][0]<=65536 and math.prod(r['shape'])<=1048576 for r in rows)
    assert [r['layout'] for r in rows[-3:]]==['row-and-column-step2','F','readonly']
    assert rows[4]['row_width_bytes']==32 and rows[4]['kind']=='prefix'
    assert rows[5]['row_width_bytes']==64 and rows[5]['unique_pool']==1
    assert (c['seed'],c['threads'],c['repeat'],c['target_ms'],c['raw_samples'])==(2026,1,7,19,126)
    assert c['original_python_source_sha256']=='72d88a45b800ed9adae7ec80e7c80aba9d1f92c3ee3d21d8731fcefc753bd751'
    assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()=='1af9719b21d980ce68a34fd93e47eb537547caa89fa4df93a7e7ed658ac42674'


class Array:
    def __init__(self,raw,shape,name='uint8',strides=None):
        size={'uint8':1,'float32':4,'float64':8}[name]
        self.raw=raw;self.shape=shape;self.strides=strides or (shape[1]*size,size);self.base=None
        self.dtype=SimpleNamespace(name=name,itemsize=size,str={'uint8':'|u1','float32':'<f4','float64':'<f8'}[name],metadata=None)
        self.flags=SimpleNamespace(owndata=True,c_contiguous=True,f_contiguous=False,writeable=True)
    def tobytes(self):return self.raw


def test_opaque_unsigned_byte_sort_preserves_first_duplicate_index():
    ar=Array(bytes([2,0,1,255,2,0,1,0]),(4,2))
    result=load().oracle_rows(ar)
    assert result==dict(raw=bytes([1,0,1,255,2,0]),shape=(3,2),first_indices=[3,1,0])


@pytest.mark.parametrize('prefix_width',(30,62))
def test_long_equal_prefixes_and_embedded_zero_bytes(prefix_width):
    prefix=b'A'*prefix_width;words=[prefix+tail for tail in (b'\x02\0',b'\x01\xff',b'\x02\0',b'\x01\0')]
    result=load().oracle_rows(Array(b''.join(words),(4,prefix_width+2)))
    assert result['raw']==words[3]+words[1]+words[0] and result['first_indices']==[3,1,0]


def test_float_rows_are_raw_words_not_numeric_sort_or_nan_equality():
    raw=struct.pack('<5I',0x80000000,0x7fc00002,0,0x7fc00001,0x80000000)
    result=load().oracle_rows(Array(raw,(5,1),'float32'))
    assert result['raw']==struct.pack('<4I',0,0x80000000,0x7fc00001,0x7fc00002)
    assert result['first_indices']==[2,0,3,1] and result['shape']==(4,1)


def test_all_equal_wide_rows_choose_first_and_keep_complete_word():
    word=bytes(range(64));result=load().oracle_rows(Array(word*17,(17,8),'float64'))
    assert result==dict(raw=word,shape=(1,8),first_indices=[0])


def test_oracle_decodes_logical_words_from_strided_readonly_array():
    ar=Array(bytes([2,0,1,255,2,0,1,0]),(4,2),strides=(8,2));ar.flags.writeable=False
    assert load().oracle_rows(ar)['first_indices']==[3,1,0]


def test_complete_output_metadata_and_bytes_are_required_before_export():
    a,b=Array(bytes([1,0,2,0]),(2,2)),Array(bytes([1,0,2,0]),(2,2))
    oracle=dict(raw=a.raw,shape=(2,2),first_indices=[3,1]);row=load().checked_outputs(a,b,oracle)
    assert row['output_sha256']==row['reference_sha256']==row['oracle_sha256']==hashlib.sha256(a.raw).hexdigest()
    assert 'raw' not in row['output_metadata']
    with pytest.raises(AssertionError):load().checked_outputs(a,b,dict(oracle,raw=bytes(4)))
    b.flags.owndata=False
    with pytest.raises(AssertionError):load().checked_outputs(a,b,oracle)
    b.flags.owndata=True;b.strides=(8,2)
    with pytest.raises(AssertionError):load().checked_outputs(a,b,oracle)


def test_provider_traps_and_genuine_first_indices_precede_public_timing():
    tree=ast.parse(SOURCE.read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    timer=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='measure')
    restoration=next(n for n in ast.walk(main) if isinstance(n,ast.Try) and any('sys.setprofile(prior)' in ast.unparse(f) for f in n.finalbody))
    check=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checked_outputs')
    assert restoration.end_lineno<check.lineno<timer.lineno
    assert [ast.unparse(a) for a in timer.args]==['lambda: reference(ar)','lambda: native(ar)','args.repeat','19']
    text=ast.unparse(main)
    assert "returns == ['unique_rows_apply'] and accepted == [True]" in text
    assert "frame.f_locals.get('native_result')" in text and "indices['raw'] == struct.pack" in text
    assert 'arg is unique_call' in text and '(reference_code, unique_code)' in text
    assert 'original.unique_rows, np.unique = saved' in ast.unparse(restoration)


def test_all_nine_results_raw_samples_and_losses_are_preserved():
    source=SOURCE.read_text();tree=ast.parse(source)
    keywords={n.arg for n in ast.walk(tree) if isinstance(n,ast.keyword)}
    assert {'output_metadata','output_sha256','reference_sha256','oracle_sha256','native_indices','actual_C_returns','native_accepted',
        'contract_sha256','harness_sha256','core_sha256','source_before','reference_pool_limits','provider_frontier'}<=keywords
    assert 'raw_samples=126' in source and 'repeat=7,target_ms=19' in source and 'choices=[1]' in source
    assert source.count('[snapshot(a) for a in arrays]==before')==2
    assert 'source_before==source_after' in source and 'CONTRACT.read_bytes()==contract_raw' in source
    assert 'for row in rows:' in source and "if row['speedup']" not in source
    c=json.loads((ROOT/'coverage/contracts/unique-rows-benchmark.json').read_bytes())
    assert 'all gains and losses retained' in c['scope'] and 'output destruction' in c['timed_boundary']
    assert 'gather' in c['provider_frontier'] and 'first occurrence index' in c['independent_oracle']
