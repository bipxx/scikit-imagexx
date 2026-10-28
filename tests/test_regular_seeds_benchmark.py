"""Inert frozen-workload/oracle checks; imports no scientific dependencies."""
import ast,builtins,hashlib,importlib.util,json,math,struct
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'benchmarks/regular_seeds.py'


def load():
    spec=importlib.util.spec_from_file_location('regular_seeds_benchmark_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_frozen_nine_real_constructor_parameters_import_without_scientific_packages(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0]not in('numpy','scipy','skimage','skimagexx')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    module=load();rows=module.case_specs();contract=json.loads(module.CONTRACT.read_bytes())
    assert len(rows)==len({r['case_id']for r in rows})==9
    assert [r['shape']for r in rows]==[[16,16],[512,512],[1000,1000],[64,64,64],[100,100,100],[512,512],[768,768],[96,96,96],[768,1024]]
    assert [r['dtype']for r in rows]==['int64','int32','int64','float32','float64','uint8','float64','int32','float32']
    assert [r['n_points']for r in rows]==[16,262144,100,262144,125,262144,589824,4096,1024]
    assert all(math.prod(r['shape'])<=1000000 for r in rows)
    assert (contract['threads'],contract['repeat'],contract['target_ms'],contract['raw_samples'])==(1,7,19,126)
    assert contract['original_python_source_sha256']=='64fcce3556f0cbf8cafd05a54ea3545b65b18bdcabbee25b953a21ba06875581'


def test_cartesian_labels_follow_selected_c_order_not_requested_count():
    result=load().oracle_labels((3,5),(slice(0,None,2),slice(1,None,2)),'int32')
    assert result==dict(raw=struct.pack('=15i',0,1,0,2,0,0,0,0,0,0,0,3,0,4,0),shape=(3,5),selected_count=4)


def test_unsigned_wrap_and_zero_unselected_words():
    result=load().oracle_labels((600,),(slice(0,None,2),),'uint8')
    assert result['selected_count']==300
    assert result['raw'][::2]==bytes(i%256 for i in range(1,301))
    assert result['raw'][1::2]==bytes(300)


@pytest.mark.parametrize('dtype,fmt',(('int64','q'),('float32','f'),('float64','d')))
def test_oracle_full_grid_exact_typed_labels(dtype,fmt):
    result=load().oracle_labels((2,3),(slice(None),slice(None)),dtype)
    assert result['raw']==struct.pack('=6'+fmt,*range(1,7)) and result['selected_count']==6


def test_empty_grid_and_refusal_of_negative_steps():
    assert load().oracle_labels((0,7),(slice(None),slice(None)),'int64')['raw']==b''
    with pytest.raises(AssertionError):load().oracle_labels((3,),(slice(None,None,-1),),'int32')


class Array:
    def __init__(self,raw,shape,dtype='int32'):
        self.raw=raw;self.shape=shape;self.strides=(shape[-1]*4,4);self.base=None
        self.dtype=SimpleNamespace(name=dtype,str='<i4',metadata=None)
        self.flags=SimpleNamespace(owndata=True,writeable=True,c_contiguous=True,f_contiguous=False)
    def tobytes(self):return self.raw


def test_exact_dtype_shape_layout_words_checked_before_export():
    a=Array(struct.pack('=4i',0,1,2,0),(2,2));b=Array(a.raw,a.shape);oracle=dict(raw=a.raw,shape=a.shape,selected_count=2)
    result=load().checked_outputs(a,b,oracle,'int32')
    assert result['output_sha256']==result['reference_sha256']==result['oracle_sha256']==hashlib.sha256(a.raw).hexdigest()
    for field,value in(('raw',bytes(16)),('shape',(1,4)),('strides',(16,4))):
        old=getattr(b,field);setattr(b,field,value)
        with pytest.raises(AssertionError):load().checked_outputs(a,b,oracle,'int32')
        setattr(b,field,old)
    with pytest.raises(AssertionError):load().checked_outputs(a,b,oracle,'float32')


def test_actual_native_acceptance_whole_api_trap_and_parity_precede_timer():
    tree=ast.parse(SOURCE.read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    timer=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure')
    check=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='checked_outputs')
    restoration=next(n for n in ast.walk(main)if isinstance(n,ast.Try))
    assert restoration.end_lineno<check.lineno<timer.lineno
    assert [ast.unparse(a)for a in timer.args]==['lambda: reference(shape, points, dtype=dtype)','lambda: native(shape, points, dtype=dtype)','args.repeat','19']
    text=ast.unparse(main)
    assert "returns == ['regular_seeds_apply'] and accepted == [True]"in text
    assert "frame.f_locals.get('native_result') is True"in text
    assert 'frame.f_code is reference_code'in text and 'original.regular_seeds = saved'in text
    assert 'original.regular_grid(shape, points)'in text


def test_all_results_and_losses_retained_and_provider_frontier_explicit():
    text=SOURCE.read_text();contract=json.loads(load().CONTRACT.read_bytes())
    assert 'raw_samples=126'in text and 'repeat=7,target_ms=19'in text
    assert 'source_before==source_after'in text and 'CONTRACT.read_bytes()==contract_raw'in text
    assert "if row['speedup']"not in text and 'for row in rows:'in text
    assert 'all gains and losses retained'in contract['scope']
    assert 'output destruction'in contract['timed_boundary'] and 'np.zeros'in contract['timed_boundary']
    assert 'only label generation and scatter are owned'in contract['provider_frontier']
