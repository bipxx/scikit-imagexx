"""Pure dtype-construction/source-context guards; no scientific bodies."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from draw_multiblock_lbp_metadata_context import historical_bytes
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-metadata-v1.json').read_bytes())
H=lambda b:hashlib.sha256(b).hexdigest()


def test_exact_binder_inverse_and_all_other_source_unchanged():
    assert H(historical_bytes(ROOT,D['source']))==D['predecessor_sha256']
    for name,pin in D['unchanged_source_sha256'].items():assert H((ROOT/name).read_bytes())==pin,name
    assert D['original_budget']==0 and D['API_completion']is False


def test_only_gray_constructor_changes_metadata_and_endianness():
    binder=(ROOT/D['source']).read_text()
    assert "a.ndim()<3?py::dtype::from_args(a.dtype().attr(\"type\")):a.dtype()"in binder
    assert "const auto dtype=k=='f'?"in binder and 'py::array result(dtype,out.shape)'in binder
    assert 'result.dtype'not in binder and 'PyArray_Cast'not in binder
    assert 'dtype'not in (ROOT/'include/skimagexx/draw_multiblock_lbp.hpp').read_text()


def test_actual_old_benchmark_binding_predicate_refuses_new_binder():
    context=json.loads((ROOT/'coverage/contracts/draw-multiblock-lbp-benchmark-execution-v2.json').read_bytes())
    pin=context['source_files_canonical_sha256'][D['source']]
    assert pin==D['predecessor_sha256']and H((ROOT/D['source']).read_bytes())!=pin
    tree=ast.parse((ROOT/'benchmarks/mark_boundaries_execution.py').read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_binding')
    loop=next(n for n in ast.walk(fn)if isinstance(n,ast.For)and ast.unparse(n.iter)=="c['source_files_canonical_sha256'].items()")
    code=compile(ast.Module(body=[loop],type_ignores=[]),'<actual source predicate>','exec')
    with pytest.raises(AssertionError,match=D['source']):exec(code,dict(c={'source_files_canonical_sha256':{D['source']:pin}},root=ROOT,snapshot=ROOT,Path=Path,digest=H,source_bytes=lambda p:Path(p).read_bytes().replace(b'\r\n',b'\n'),prepared={'files_sha256':{D['source']:pin}}))


@pytest.mark.parametrize('cached',(False,True))
def test_actual_UNBOUND_benchmark_cold_cached_no_requests(cached):
    code="""import builtins,sys,runpy,types
old=builtins.__import__
def refuse(n,*a,**kw):
 if n.split('.')[0]in('numpy','skimage','scipy','skimagexx'):raise AssertionError('BODY_REQUEST_FORBIDDEN')
 return old(n,*a,**kw)
if sys.argv[2]=='True':sys.modules['numpy']=types.ModuleType('numpy')
builtins.__import__=refuse;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'benchmarks/draw_multiblock_lbp_native.py'),str(cached)],capture_output=True,text=True)
    assert p.returncode!=0 and 'UNBOUND'in p.stderr and 'BODY_REQUEST_FORBIDDEN'not in p.stderr
