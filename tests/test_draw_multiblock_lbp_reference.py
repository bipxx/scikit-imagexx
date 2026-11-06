"""Pure source/safety contract; no algorithm/scientific imports."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest
from draw_multiblock_lbp_contract import API,SIGNATURE,OFFSETS,UNASSESSED,cases,source_safe
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/draw-multiblock-lbp-v1.json'


def test_frozen_signature_cases_sources_and_budget():
    data=json.loads(CONTRACT.read_bytes());assert data['api']==API and data['signature']==SIGNATURE
    assert data['cases']==cases()and data['bounds_by_case']=={s['id']:source_safe(s)for s in cases()}
    assert data['case_count']==len(cases())==len({s['id']for s in cases()})
    assert data['unassessed']==list(UNASSESSED)
    assert data['original_calls']==data['native_calls']==data['helper_calls']==data['timed_batches']==0
    for name,pin in data['source_freeze'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    assert data['state']=='SOURCE_ONLY_PENDING_INDEPENDENT_SAFETY_AND_ROOT_GO'


def test_exact_source_orientation_and_operation_order():
    data=json.loads(CONTRACT.read_bytes());text=data['source_fragments']['feature/texture.py:draw_multiblock_lbp'];tree=ast.parse(text);function=tree.body[0]
    assignment=next(n for n in function.body if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='neighbor_rect_offsets'for v in n.targets))
    assert ast.literal_eval(assignment.value)==OFFSETS
    assert data['bit_orientation']==[dict(bit=7-i,row_offset=r,column_offset=c)for i,(r,c)in enumerate(OFFSETS)]
    assert text.index('np.asarray(color_greater_block')<text.index('np.asarray(color_less_block')<text.index('output = np.copy(image)')<text.index('if len(image.shape) < 3:')<text.index('gray2rgb(image)')<text.index('output = img_as_float(output)')<text.index('neighbor_rect_offsets[:, 0] *= height')<text.index('central_rect_r = r + height')<text.index('has_greater_value = lbp_code & (1 << (7 - element_num))')
    assert text.count('(1 - alpha) * output[')==2 and text.count('] + alpha * color_')==2
    assert '(0, 0)'not in repr(OFFSETS)
    calls={n.func.id for n in ast.walk(function)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)}
    assert '_multiblock_lbp'not in calls and 'rectangle'not in calls
    assert 'clip('not in text and 'np.clip'not in text
    assert 'return np.stack(3 * (image,), axis=channel_axis)'in data['source_fragments']['color/colorconv.py:gray2rgb']
    for name,body in data['source_fragments'].items():assert hashlib.sha256(body.encode()).hexdigest()==data['source_fragment_sha256'][name]


@pytest.mark.parametrize('spec',cases(),ids=lambda s:s['id'])
def test_factory_and_original_work_bounds(spec):
    b=source_safe(spec);assert b['factory_elements']<=4096 and b['max_output_elements']<=12288
    assert b['source_neighbor_updates']==8 and b['max_broadcast_temporary_elements']==10000
    assert b['original_call_budget']==0


@pytest.mark.parametrize('change',({'shape':[100000,100000]},{'width':2**60},{'code':2**100},{'dtype':'longdouble'},{'alpha':object()}))
def test_unreviewed_unsafe_recipe_refused(change):
    with pytest.raises(AssertionError):source_safe({**cases()[0],**change})


@pytest.mark.parametrize('cached',(False,True))
def test_real_unbound_CLI_refuses_cold_cached_science(cached):
    code="""import builtins,runpy,sys,types
old=builtins.__import__
def refuse(name,*a,**k):
 if name.split('.')[0]in('numpy','skimage','skimagexx'):raise RuntimeError('SCIENCE_REQUEST_REACHED')
 return old(name,*a,**k)
if sys.argv[2]=='True':sys.modules['numpy']=types.ModuleType('numpy')
builtins.__import__=refuse;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'tests/draw_multiblock_lbp_contract.py'),str(cached)],capture_output=True,text=True)
    assert p.returncode!=0 and 'DRAW_MULTIBLOCK_LBP_UNBOUND'in p.stderr and 'SCIENCE_REQUEST_REACHED'not in p.stderr and 'Traceback'not in p.stderr


def test_sampler_cases_SDK_reuse_and_still_pending():
    data=json.loads(CONTRACT.read_bytes());module=runpy.run_path(str(ROOT/'benchmarks/draw_multiblock_lbp.py'))
    assert len(module['case_specs']())==4 and all(module['case_bounds'](r)['source_updates']==8 for r in module['case_specs']())
    source=(ROOT/'tools/montage_sustained.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='measure')
    assert hashlib.sha256((ast.get_source_segment(source,node)+'\n').encode()).hexdigest()==data['benchmark_method']['method_sha256']
    p=subprocess.run([sys.executable,'-I',str(ROOT/'benchmarks/draw_multiblock_lbp.py')],capture_output=True,text=True)
    assert p.returncode!=0 and 'DRAW_MULTIBLOCK_LBP_UNBOUND'in p.stderr
    assert data['SDK_proposal']['named_complete_operation']=='draw_multiblock_lbp'and data['SDK_proposal']['implemented']is False
    assert data['reuse_limitations']['whole_set_color_is_equivalent']is False
    for name,pin in data['reuse_source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
