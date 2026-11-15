"""Narrow source/inverse/boundary guards, no scientific/native execution."""
import ast,copy,hashlib,json,runpy,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'coverage/contracts/util-view-boundary-optimization-v1.json').read_bytes())
H=lambda b:hashlib.sha256(b).hexdigest()


def test_exact_two_source_delta_and_immutable_donors_golds_methods():
    assert set(C['sources'])=={'src/crop.cpp','python/skimagexx/_array_views.py'}
    for name,row in C['sources'].items():
        current=(ROOT/name).read_bytes();assert H(current)==row['current_raw_sha256']and current.decode()==row['current_source']
        assert H(row['predecessor_source'].encode())==row['predecessor_raw_sha256']
    active=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))
    context=json.loads((ROOT/active['CONTEXT']).read_bytes())
    for name,pin in C['unchanged_source_raw_sha256'].items():
        raw=active['historical_bytes'](ROOT,name,section='successor_source_predecessors')if name in context['successor_source_predecessors']else(ROOT/name).read_bytes()
        assert H(raw)==pin,name
    assert C['method_cases_golds_unchanged']and not C['performance_claim']and C['original_calls_authorized']==0


def test_cached_intp_metadata_preserves_every_function_body_and_order():
    row=C['sources']['python/skimagexx/_array_views.py'];before=ast.parse(row['predecessor_source']);after=ast.parse(row['current_source'])
    assignments=[n for n in after.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_INTP_MAX'for t in n.targets)];assert len(assignments)==1
    assert ast.unparse(assignments[0].value)=='np.iinfo(np.intp).max'
    after.body.remove(assignments[0])
    class Restore(ast.NodeTransformer):
        def visit_Name(self,n):
            return ast.parse('np.iinfo(np.intp).max',mode='eval').body if n.id=='_INTP_MAX'and isinstance(n.ctx,ast.Load)else n
    assert ast.dump(Restore().visit(after))==ast.dump(before)


def test_cpp_keeps_complete_view_geometry_and_disjoint_pod_serial_contract():
    row=C['sources']['src/crop.cpp'];old=row['predecessor_source'];new=row['current_source']
    a='CropPlan crop_view(';b='void crop_copy_into('
    assert old[old.index(a):old.index(b)]==new[new.index(a):new.index(b)]
    assert 'if(contiguous)std::memcpy(target,source,width*view.itemsize)'in new
    assert 'add(from,source_last);add(to,target_last);'in new
    assert 'if(i+1<count){target+=target_step;source+=source_step;}'in new
    assert all('copy_line<'+str(n)+'>'in new for n in(1,2,4,8,16,32))
    assert 'get_num_threads'not in new and 'omp 'not in new and 'std::atomic'not in new
    assert 'std::memcpy(target,source,bytes)'in new  # Odd/structured POD widths retained.


def test_actual_frozen_executor_still_refuses_new_production_bytes():
    # Execute only the real source-predicate AST, never binding/target bodies.
    context=json.loads((ROOT/'coverage/contracts/next-util-benchmark-execution-v2.json').read_bytes())
    tree=ast.parse((ROOT/'benchmarks/next_util_execution.py').read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='load_binding')
    loop=next(n for n in ast.walk(fn)if isinstance(n,ast.For)and ast.unparse(n.iter)=="context['production_source_canonical_sha256'].items()")
    code=compile(ast.Module(body=[loop],type_ignores=[]),'<actual frozen executor source check>','exec')
    for name,row in C['sources'].items():
        assert context['production_source_canonical_sha256'][name]==row['predecessor_raw_sha256']
        with pytest.raises(AssertionError,match=name):exec(code,dict(context={'production_source_canonical_sha256':{name:context['production_source_canonical_sha256'][name]}},digest=H,source_bytes=lambda p:Path(p).read_bytes().replace(b'\r\n',b'\n'),root=ROOT))


@pytest.mark.parametrize('driver',('crop.py','slice_along_axes.py','array_views.py'))
@pytest.mark.parametrize('cached',(False,True))
def test_real_unbound_cli_still_stops_before_cold_cached_requests(driver,cached):
    code="""import builtins,runpy,sys,types
old=builtins.__import__
def refuse(n,*a,**k):
 if n.split('.')[0]in('numpy','scipy','skimage','skimagexx'):raise AssertionError('REQUEST_FORBIDDEN')
 return old(n,*a,**k)
if sys.argv[2]=='True':sys.modules['numpy']=types.ModuleType('numpy')
builtins.__import__=refuse;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'benchmarks'/driver),str(cached)],capture_output=True,text=True)
    assert p.returncode!=0 and 'PROSPECTIVE_UNBOUND'in p.stderr and 'REQUEST_FORBIDDEN'not in p.stderr
