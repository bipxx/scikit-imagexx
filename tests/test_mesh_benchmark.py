"""Mesh workload/CI metadata only; no scientific calls or timing."""
import ast
import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'benchmarks/mesh_surface_area.py'


def test_fixed_six_meshes_import_without_scientific_dependencies(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0]not in('numpy','scipy','skimage','skimagexx'),name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    spec=importlib.util.spec_from_file_location('mesh_benchmark_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=module.case_specs()
    assert len(rows)==len({r['case_id']for r in rows})==6
    assert {(r['cells'],r['dtype'])for r in rows}=={(c,d)for c in(64,128,256)for d in('float32','float64')}
    assert {r['vertex_count']for r in rows}=={4225,16641,66049}
    assert {r['face_count']for r in rows}=={8192,32768,131072}
    assert all(r['api']=='measure.mesh_surface_area'for r in rows)


def test_quadratic_dyadic_heightfield_is_frozen_outside_timer():
    tree=ast.parse(SOURCE.read_bytes())
    fixture=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='mesh_inputs')
    z=next(n for n in fixture.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='z')
    assert ast.dump(z)==ast.dump(ast.parse('z=.125*(x*x+2*y*y+x*y)').body[0])
    names={n.attr for n in ast.walk(fixture)if isinstance(n,ast.Attribute)}
    assert not {'sin','cos','marching_cubes'}&names
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    calls=[n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)]
    make=next(n for n in calls if n.func.id=='mesh_inputs')
    check=next(n for n in calls if n.func.id=='mesh_default_error')
    timer=next(n for n in calls if n.func.id=='measure')
    assert make.lineno<check.lineno<timer.lineno
    assert [ast.unparse(a)for a in check.args]==['call_args','actual','expected']
    assert [ast.unparse(a)for a in timer.args[2:]]==['args.repeat','15']
    assert 'reference(*call_args)'in ast.unparse(main)and 'native(*call_args)'in ast.unparse(main)
    assert 'default_call'not in{n.id for n in ast.walk(tree)if isinstance(n,ast.Name)}


def test_parity_requires_actual_native_completion_and_records_real_storage():
    text=SOURCE.read_text(encoding='utf-8')
    assert "returns==['mesh_surface_area'] and accepted==[True]"in text
    assert "frame.f_locals.get('total')is not None"in text
    assert 'finally:sys.setprofile(prior)'in text
    assert text.index('finally:sys.setprofile(prior)')<text.index('timing=measure(')
    tree=ast.parse(text);keywords={n.arg for n in ast.walk(tree)if isinstance(n,ast.keyword)}
    assert {'vertex_shape','face_shape','vertex_dtype_str','face_dtype_str','input_sha256',
            'result_type','reference_result_type','output_dtype_str','reference_dtype_str',
            'output_raw_hex','reference_raw_hex','native_accepted','actual_C_returns',
            'height_formula','height_formula_sha256','harness_sha256','core_sha256',
            'source_before','raw_samples','numpy_buffer_elements'}<=keywords
    assert 'raw_samples=84'in text and '[7,7]'in text


def test_generic_mesh_checker_receives_full_actual_call_context():
    tree=ast.parse((ROOT/'benchmarks/coverage.py').read_bytes())
    branch=next(n for n in ast.walk(tree)if isinstance(n,ast.If)and ast.unparse(n.test)=="api == 'measure.mesh_surface_area'")
    calls=[];args=(object(),object());a,e=object(),object()
    def checker(*values):calls.append(values);return 0.
    namespace=dict(call_args=args,actual=a,expected=e,mesh_default_error=checker)
    exec(compile(ast.Module(branch.body,type_ignores=[]),'<mesh call context>','exec'),namespace)
    assert calls==[(args,a,e)]and namespace['error']==0.
    provider=next(n for n in ast.walk(tree)if isinstance(n,ast.If)and "dtype.char not in ('f', 'd')"in ast.unparse(n.test))
    namespace=dict(api='measure.mesh_surface_area',call_args=(SimpleNamespace(dtype=SimpleNamespace(char='B')),),backend='native')
    exec(compile(ast.Module([provider],type_ignores=[]),'<mesh provider classification>','exec'),namespace)
    assert namespace['backend']=='primitive-provider'


def test_both_mesh_modes_and_phase3_counts_are_in_preparation():
    workflow=(ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8')
    producer=workflow.split('  numpy-minimum:',1)[1].split('  measure-minimum:',1)[0]
    for mode in('normal','scalar'):
        target='mesh-surface-area-'+mode+'.xml'
        assert producer.count(target)==1 and producer.index(target)<producer.index('ci_measurement_artifacts.py freeze')
    phase3=(ROOT/'.github/workflows/phase3-public-benchmark.yml').read_text(encoding='utf-8')
    assert phase3.count('tests/test_mesh_surface_area_public.py')==2
    assert 'len(cases) == 2924'in phase3 and 'len(native) == 237'in phase3 and 'expected == 1409'in phase3
