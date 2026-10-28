"""Prospective benchmark guards and independent word checks using inert arrays."""
import ast, builtins, hashlib, importlib.util, json, math
from pathlib import Path
import struct
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'benchmarks/label_points.py'


def load():
    spec=importlib.util.spec_from_file_location('label_points_benchmark_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_frozen_nine_coordinate_cases_import_without_scientific_dependencies(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0] not in ('numpy','scipy','skimage','skimagexx')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    module=load();contract=json.loads(module.CONTRACT.read_bytes());rows=module.case_specs()
    assert len(rows)==len({r['case_id'] for r in rows})==9
    assert [(r['kind'],r['output_shape'],r['coordinate_dtype'],r['point_count']) for r in rows]==[
        ('sparse',[64,64],'int64',16),('dense',[64,64],'float64',4096),
        ('sparse',[1024,1024],'int64',16),('dense',[1024,1024],'float64',1048576),
        ('sparse',[64,64,64],'float32',32),('dense',[64,64,64],'int64',262144),
        ('duplicates',[1024,1024],'float64',65536),('strided',[1024,1024],'float32',65536),
        ('strided',[64,64,64],'int64',32768)]
    assert all(math.prod(r['output_shape'])<=1048576 for r in rows)
    assert (contract['seed'],contract['threads'],contract['repeat'],contract['target_ms'],contract['raw_samples'])==(2026,1,7,19,126)
    assert contract['original_python_source_sha256']=='8c409baaf3527b3466c8d8e973e4482182bdf172e8486c33d91eb469403742b2'
    assert hashlib.sha256(module.CONTRACT.read_bytes()).hexdigest()=='3fceb3b66c254f1330f30f779a34a042162c08cf59921bf1343b06376f450221'


class Array:
    def __init__(self,raw,name='int64',shape=None,strides=None):
        width=4 if name=='float32' else 8
        self.raw=raw;self.dtype=SimpleNamespace(name=name,str=('<f4' if name=='float32' else '<f8' if name=='float64' else '<i8'),metadata=None)
        self.shape=shape or (len(raw)//width,);self.strides=strides or (width,);self.base=None
        self.flags=SimpleNamespace(owndata=True,c_contiguous=True,f_contiguous=False,writeable=True)
    def tobytes(self):return self.raw


@pytest.mark.parametrize('dtype,fmt',(('float32','f'),('float64','d')))
def test_reverse_oracle_rounds_ties_even_and_last_rounded_duplicate_wins(dtype,fmt):
    coords=Array(struct.pack('='+fmt*8,.5,1.5,1.5,2.5,.5,1.5,2.5,3.5),dtype,(4,2))
    words=struct.unpack('=20Q',load().oracle_words(coords,(4,5)))
    assert words==tuple({2:3,12:2,14:4}.get(i,0) for i in range(20))


def test_oracle_uses_logical_coordinate_words_and_c_order_3d_offsets():
    coords=Array(struct.pack('=9q',1,2,3,0,0,0,1,2,3),'int64',(3,3),(-48,-8))
    words=struct.unpack('=24Q',load().oracle_words(coords,(2,3,4)))
    assert words==(2,)+tuple(0 for _ in range(22))+(3,)


def test_empty_independent_oracle_zeroes_complete_output():
    assert load().oracle_words(Array(b'','int64',(0,2)),(2,3))==bytes(48)


@pytest.mark.parametrize('coords',((-1,0),(2,0),(0,3)))
def test_oracle_rejects_out_of_domain_coordinate_instead_of_clipping(coords):
    with pytest.raises(AssertionError):load().oracle_words(Array(struct.pack('=2q',*coords),'int64',(1,2)),(2,3))


def test_output_precheck_requires_full_words_layout_and_exports_hashes_only():
    a,b=Array(struct.pack('=2Q',0,2)),Array(struct.pack('=2Q',0,2))
    report=load().checked_outputs(a,b,a.raw)
    assert report['output_sha256']==report['reference_sha256']==report['oracle_sha256']==hashlib.sha256(a.raw).hexdigest()
    assert 'raw' not in report['output_metadata']
    with pytest.raises(AssertionError):load().checked_outputs(a,b,bytes(16))
    b.strides=(16,)
    with pytest.raises(AssertionError):load().checked_outputs(a,b,a.raw)


def test_native_and_forwarding_prechecks_restore_before_whole_public_timing():
    tree=ast.parse(SOURCE.read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    timer=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='measure')
    restoration=next(n for n in ast.walk(main) if isinstance(n,ast.Try) and any('sys.setprofile(prior)' in ast.unparse(f) for f in n.finalbody))
    precheck=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checked_outputs')
    assert restoration.end_lineno<precheck.lineno<timer.lineno
    assert [ast.unparse(a) for a in timer.args]==['lambda: reference(coords, shape)','lambda: native(coords, shape)','args.repeat','19']
    source=ast.unparse(main)
    assert "returns == ['label_points_apply'] and accepted == [True]" in source
    assert "frame.f_locals.get('native_result') is not None" in source
    assert 'frame.f_code is reference_code' in source and 'original.label_points = saved' in ast.unparse(restoration)


def test_all_cases_samples_mutations_and_losses_remain_in_report():
    source=SOURCE.read_text();tree=ast.parse(source)
    keywords={n.arg for n in ast.walk(tree) if isinstance(n,ast.keyword)}
    assert {'output_metadata','output_sha256','reference_sha256','oracle_sha256','actual_C_returns','native_accepted',
        'contract_sha256','harness_sha256','core_sha256','source_before','reference_pool_limits','provider_frontier'}<=keywords
    assert 'raw_samples=126' in source and 'repeat=7,target_ms=19' in source and 'choices=[1]' in source
    assert source.count('[snapshot(a) for a in arrays]==before')==2
    assert 'source_before==source_after' in source and 'CONTRACT.read_bytes()==contract_raw' in source
    assert 'for row in rows:' in source and 'if row[\'speedup\']' not in source
    contract=json.loads((ROOT/'coverage/contracts/label-points-benchmark.json').read_bytes())
    assert 'all gains and losses retained' in contract['scope']
    assert 'output destruction' in contract['timed_boundary'] and 'sparse allocation-cost losses' in contract['frontier']
