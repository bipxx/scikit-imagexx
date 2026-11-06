"""Source-only crop planning and refused recipe intake; no original or NumPy calls."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from crop_contract import API,SIGNATURE,cases,source_safe,MAX_INPUT_ELEMENTS,MAX_PHYSICAL_ELEMENTS
ROOT=Path(__file__).parents[1]
DATA=json.loads((ROOT/'coverage/contracts/crop-v1.json').read_bytes())


def test_crop_source_signature_order_and_provider_boundary():
    raw=(ROOT/DATA['upstream']['fixture']).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==DATA['upstream']['raw_and_canonical_lf_sha256']
    tree=ast.parse(raw);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    assert function.name=='crop' and [a.arg for a in function.args.args]==['ar','crop_width','copy','order']
    assert [ast.literal_eval(n)for n in function.args.defaults]==[False,'K']
    assert not function.args.posonlyargs and not function.args.kwonlyargs and function.args.vararg is function.args.kwarg is None
    text=raw.decode();coercion=text.index('ar = np.array(ar, copy=False)');width=text.index('if isinstance(crop_width, Integral)');slices=text.index('slices = tuple(');truth=text.index('if copy:');copycall=text.index('cropped = np.array(ar[slices], order=order, copy=True)')
    assert coercion<width<slices<truth<copycall
    assert 'cropped = ar[slices]'in text and DATA['signature']==SIGNATURE
    assert DATA['upstream']['profiles']['minimum']['numpy']=='1.24.4' and DATA['upstream']['profiles']['modern']['numpy']=='2.5.4'
    assert len({r['upstream_raw_sha256']for r in DATA['upstream']['profiles'].values()})==1


def test_crop_frozen_cases_cover_intended_interface_without_parity_claim():
    specs=cases();assert specs==DATA['cases']and len(specs)==len({s['id']for s in specs})
    assert DATA['api']==API and DATA['original_observations']=={}
    assert DATA['capture_plan']['status']=='NOT_AUTHORIZED_PENDING_INDEPENDENT_SOURCE_SAFETY_AND_ROOT_GO'
    assert DATA['capture_plan']['proposed_cases_per_profile']==len(specs)
    assert DATA['capture_plan']['proposed_two_profile_total']==2*len(specs)
    assert DATA['source_safety']['original_calls']==0
    assert {s['order']for s in specs if type(s['order'])is str}>={'K','C','F','A','invalid'}
    assert {len(s['shape'])for s in specs}=={0,1,2,3,4}
    assert any(0 in s['shape']for s in specs)and any(s['readonly']for s in specs)
    assert {'F','reverse_all','step_all','permute3'}<={s['layout']for s in specs}
    assert {'subclass','array_provider','array_provider_error','interface_provider','list','tuple','numpy_scalar','python_scalar','memoryview'}<={s['input_kind']for s in specs}
    assert any(s['dtype']=='object'and s['copy']is True for s in specs)
    assert any(s['width']==2**70 for s in specs)and any(s['width']==-1 for s in specs)


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_crop_each_recipe_is_source_bounded_without_materialization(spec):
    bound=source_safe(spec)
    assert bound['logical_input_elements']<=MAX_INPUT_ELEMENTS
    assert bound['physical_input_elements']<=MAX_PHYSICAL_ELEMENTS
    assert bound['max_result_elements']<=bound['logical_input_elements']
    assert bound['original_calls']==0


@pytest.mark.parametrize('fault',('unknown-field','missing-field','negative-shape','bool-shape','rank','axis','elements','dtype','input-protocol','layout','readonly','width-magnitude','width-depth','width-vector','np-width-overflow','sequence-protocol','copy-protocol','order-object','order-string','identifier'))
def test_crop_refuses_unreviewed_intake_before_any_target_import(fault,monkeypatch):
    import builtins
    previous=builtins.__import__;attempts=[]
    def refuse(name,*args,**kwargs):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):attempts.append(name);raise AssertionError('Target import forbidden: '+name)
        return previous(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',refuse)
    row=copy.deepcopy(cases()[0])
    if fault=='unknown-field':row['callable']='external'
    elif fault=='missing-field':row.pop('width')
    elif fault=='negative-shape':row['shape']=[-1,5]
    elif fault=='bool-shape':row['shape']=[True,5]
    elif fault=='rank':row['shape']=[1]*5
    elif fault=='axis':row['shape']=[9,1]
    elif fault=='elements':row['shape']=[8,8,8]
    elif fault=='dtype':row['dtype']='longdouble'
    elif fault=='input-protocol':row['input_kind']='forged_array_interface'
    elif fault=='layout':row['layout']='as_strided_external'
    elif fault=='readonly':row['readonly']=1
    elif fault=='width-magnitude':row['width']=2**71
    elif fault=='width-depth':
        row['width']=0
        for _ in range(6):row['width']={'container':'tuple','items':[row['width']]}
    elif fault=='width-vector':row['width']={'width_array':list(range(9))}
    elif fault=='np-width-overflow':row['width']={'np_scalar':'uint64','value':2**64}
    elif fault=='sequence-protocol':row['width']={'sequence_probe':[1,2,3]}
    elif fault=='copy-protocol':row['copy']={'bool_probe':'unknown'}
    elif fault=='order-object':row['order']=object()
    elif fault=='order-string':row['order']='x'*32
    else:row['id']='outside/recipe'
    with pytest.raises(AssertionError):source_safe(row)
    assert attempts==[]


def test_crop_source_freeze_and_science_free_import_graph():
    for path,pin in DATA['source_canonical_lf_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,path
    for path in ('tests/crop_contract.py','tests/crop_observer.py'):
        tree=ast.parse((ROOT/path).read_bytes())
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names]if isinstance(node,ast.Import)else[node.module or'']
                assert all(n.split('.')[0]not in('numpy','scipy','skimage','skimagexx')for n in names)
    # The prospective observer has one callsite; only a separately authorized
    # external journal controller may invoke it, never these source guards.
    tree=ast.parse((ROOT/'tests/crop_observer.py').read_bytes());observer=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='observe')
    assert len([n for n in ast.walk(observer)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='call'])==1


def test_crop_imports_without_scientific_or_original_modules(monkeypatch):
    import builtins
    previous=builtins.__import__;attempts=[]
    def refuse(name,*args,**kwargs):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):attempts.append(name);raise AssertionError('Forbidden '+name)
        return previous(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',refuse)
    for path in ('crop_contract.py','crop_observer.py'):
        spec=importlib.util.spec_from_file_location('crop_source_import_'+path,ROOT/'tests'/path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert attempts==[]


def test_crop_requires_public_performance_and_preserves_477_scope():
    assert DATA['performance_plan']['required']is True
    assert len(DATA['performance_plan']['proposed_cases'])==8
    assert DATA['route']['inventory_entries']==477 and DATA['route']['current_implementation']=='upstream-only'
    assert DATA['route']['prerequisites']==['array']
    entries=json.loads((ROOT/'coverage/api.json').read_bytes())['entries']
    assert len(entries)==477 and next(r for r in entries if r['api']==API)['implementation']=='upstream-only'
    assert DATA['gates']['implementation'].startswith('NOT_AUTHORIZED')
    assert DATA['unassessed']and DATA['reuse_and_implementation_direction']['SIMD'].startswith('No SIMD')


def test_crop_future_capture_requires_actual_source_locator_before_original_budget():
    gate=DATA['capture_plan']['actual_setup_gate']
    assert 'REAL source locator' in gate and 'actual retained Linux runtime' in gate
    assert 'stubbed command is not proof' in gate and 'no new preflight framework' in gate
    assert DATA['source_safety']['original_calls']==0
