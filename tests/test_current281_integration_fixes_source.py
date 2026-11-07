"""Exact regression scope for the three actual current280 integration failures."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import builtins
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/current281-integration-fixes-v1.json'
CONTEXT_SHA256='84716fa3a2f8bfab0a912b3f4b397311eb5ed95e9c6c1f5c640598d170543556'
H=lambda raw:hashlib.sha256(raw).hexdigest()


def test_exact_fix_projection_and_unchanged_checker_engines():
    assert H(CONTEXT.read_bytes())==CONTEXT_SHA256
    c=json.loads(CONTEXT.read_bytes())
    from tools.join_source_context import integration_fix_predecessor_bytes
    for name,row in c['sources'].items():assert H(integration_fix_predecessor_bytes(ROOT,name,(ROOT/name).read_bytes()))==row['predecessor_canonical_lf_sha256'],name
    for name,pin in c['unchanged_computation'].items():assert H((ROOT/name).read_bytes())==pin,name
    for key,pin in c['immutable_checker_functions'].items():
        path,name=key.split(':');text=(ROOT/path).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
        assert H(ast.get_source_segment(text,node).encode())==pin
    assert c['actual_genuine_authority']is c['actual_benchmark_authority']is None
    assert c['reference_capture_calls_remaining']==0


@pytest.mark.parametrize('fault',('source','inverse'))
def test_correction_rejects_unreviewed_source_or_inverse(tmp_path,fault):
    from tools.join_source_context import integration_fix_predecessor_bytes
    c=json.loads(CONTEXT.read_bytes());name='python/skimagexx/coverage.py';raw=(ROOT/name).read_bytes()
    if fault=='source':raw+=b'changed'
    else:c['sources'][name]['inverse_fragments'][0]['predecessor']+='changed'
    target=tmp_path/'coverage/contracts';target.mkdir(parents=True);(target/CONTEXT.name).write_text(json.dumps(c))
    with pytest.raises(AssertionError):integration_fix_predecessor_bytes(tmp_path,name,raw)


@pytest.mark.parametrize('dtype',('int8','uint8','int16','uint16','int32','uint32','int64','uint64','float32','float64','bool'))
@pytest.mark.parametrize('cached',(False,True))
def test_actual_default_factory_builds_valid_labels_without_algorithm_requests(dtype,cached,monkeypatch):
    coverage=runpy.run_path(str(ROOT/'python/skimagexx/coverage.py'))
    descriptor=SimpleNamespace(name=dtype,kind='u'if dtype.startswith('uint')else'i'if dtype.startswith('int')else'f'if dtype.startswith('float')else'b',metadata={'retained':'descriptor'})
    fixed=SimpleNamespace(name='int16',kind='i',metadata=None)
    class Array:
        def __init__(self,values,dtype):self.values=copy.deepcopy(values);self.dtype=dtype
        def copy(self):return Array(self.values,self.dtype)
    requests=[]
    numpy=SimpleNamespace(array=lambda values,*,dtype:Array(values,dtype),dtype=lambda name:fixed if name=='int16'else pytest.fail('Unexpected dtype coercion'))
    old=builtins.__import__
    def request(name,*a,**k):
        if name=='numpy':requests.append(name);return numpy
        if name.split('.')[0]in('scipy','skimage','skimagexx'):pytest.fail('Algorithm import request: '+name)
        return old(name,*a,**k)
    with monkeypatch.context()as patch:
        if cached:patch.setitem(sys.modules,'numpy',SimpleNamespace(cached_must_not_be_used=True))
        patch.setattr(builtins,'__import__',request)
        args,options=coverage['default_call']('metrics.variation_of_information',SimpleNamespace(dtype=descriptor))
    assert requests==['numpy']and options=={}and len(args)==2
    assert args[0]is not args[1]and args[0].values==args[1].values==[[0,1,1],[2,2,2]]
    expected=descriptor if descriptor.kind in'iu'else fixed
    assert args[0].dtype is args[1].dtype is expected
    assert expected.kind in'iu'
    assert coverage['native_dtypes']('metrics.variation_of_information')==('int8','uint8','int16','uint16','int32','uint32','int64','uint64')


def test_VI_fixture_three_results_and_state_tail_are_used():
    source=(ROOT/'tests/test_variation_of_information_integration.py').read_text();tree=ast.parse(source)
    test=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='test_vi_swappable_public_saved_perfect_and_strict_checker')
    assignments=[n for n in ast.walk(test)if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='fixture']
    assert len(assignments)==1 and [n.id for n in assignments[0].targets[0].elts]==['args','options','buffers']
    assert 'before=observation((args,options,buffers),np)'in source
    assert 'assert observation((args,options,buffers),np)==before'in source
    assert 'assert policy==(np.geterr(),np.geterrcall(),np.getbufsize())'in source
    assert 'actual.tobytes()==expected_raw'in source and "expected[0]=1."not in source
    fixture=(ROOT/'tests/label_pair_metrics_contract.py').read_text();node=next(n for n in ast.parse(fixture).body if isinstance(n,ast.FunctionDef)and n.name=='fixture')
    returns=[n for n in ast.walk(node)if isinstance(n,ast.Return)and isinstance(n.value,ast.Tuple)];assert len(returns)==1 and len(returns[0].value.elts)==3


def test_invalid_float_labels_remain_distinct_from_nominal_carrier_success():
    source=(ROOT/'tests/test_variation_of_information_integration.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='test_vi_invalid_float_labels_are_provider_errors_not_native_counts')
    body=ast.get_source_segment(source,node)
    assert 'dtype=dtype'in body and 'labels.copy()'in body and 'pytest.raises(TypeError)'in body
    assert "'label_pair_counts_preweighted'"in body and "'variation_of_information_csr'"in body
    assert 'observation(args,np)==before'in body and 'getattr(original'not in body
    generic=(ROOT/'tests/test_api.py').read_text();begin=generic.index("        if api=='metrics.variation_of_information':");end=generic.index("        if api=='draw.set_color':",begin);branch=generic[begin:end]
    assert "all(value.dtype.kind in 'iu' for value in args)"in branch and 'assert calls==[1]'in branch
    assert "pytest.fail('float VI labels reached"not in branch


def test_flood_expected_is_padded_crop_and_strict_checker_is_kept():
    text=(ROOT/'tests/test_flood_fill_integration.py').read_text()
    assert 'flags=np.zeros((4,4),dtype=np.uint8)'in text and 'flags[1:3,1:3]=[[1,1],[1,0]]'in text
    assert 'expected=flags[1:3,1:3].view(bool)'in text and 'expected.strides==(4,1)'in text
    assert 'np.shares_memory(expected,flags)'in text and 'not np.shares_memory(expected,image)'in text
    assert 'assert_default_parity(api,image.dtype,actual,expected,image=image)'in text


def test_actual_metadata_and_collector_close_the_new_source_guard():
    row=next(r for r in json.loads((ROOT/'coverage/api.json').read_bytes())['entries']if r['api']=='metrics.variation_of_information')
    coverage=runpy.run_path(str(ROOT/'python/skimagexx/coverage.py'));assert row['native_default_case_dtypes']==list(coverage['native_dtypes'](row['api']))
    assert 'not floating-label support'in row['default_input_kind']
    tree=ast.parse((ROOT/'tools/inventory.py').read_bytes());assert any(isinstance(n,ast.Constant)and n.value==row['default_input_kind']for n in ast.walk(tree))
    q=runpy.run_path(str(ROOT/'tools/public_qualification.py'))
    assert CONTEXT.relative_to(ROOT).as_posix()in q['DRAW_SOURCE_PATHS']and Path(__file__).relative_to(ROOT).as_posix()in q['DRAW_SOURCE_PATHS']
    command=next(l.strip()for l in(ROOT/'.github/workflows/public-qualification.yml').read_text().splitlines()if l.strip().startswith('python -m pytest -q --capture=sys'))
    assert command.index('tests/test_current281_integration_fixes_source.py')<command.index('2>&1 | tee')
