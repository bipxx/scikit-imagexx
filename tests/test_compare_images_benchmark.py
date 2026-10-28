"""Fixed image-comparison workloads and CI isolation; no scientific calls."""
import ast
import builtins
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'benchmarks/compare_images.py'


def test_cases_import_without_scientific_dependencies(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0]not in ('numpy','scipy','skimage','skimagexx')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    spec=importlib.util.spec_from_file_location('comparison_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=module.case_specs()
    assert len(rows)==len({r['case_id']for r in rows})==18
    assert {(tuple(r['shape']),r['dtype'],r['method'])for r in rows}=={
        ((s,s),d,m)for s in (256,1024)for d in ('uint8','float32','float64')
        for m in ('diff','blend','checkerboard')}
    assert all(r['n_tiles']==[8,8]for r in rows)


def test_exact_checks_and_observer_cleanup_precede_measurement():
    tree=ast.parse(SOURCE.read_bytes());text=ast.unparse(tree)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    timer=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure')
    checks=[n for n in ast.walk(main)if isinstance(n,ast.Assert)and 'actual.tobytes() == expected.tobytes()'in ast.unparse(n)]
    assert len(checks)==1 and checks[0].lineno<timer.lineno
    restore=next(n for n in ast.walk(main)if isinstance(n,ast.Try)and any('sys.setprofile(prior)'in ast.unparse(f)for f in n.finalbody))
    assert restore.end_lineno<timer.lineno
    assert 'captured[original_code] == captured[native_code]'in text
    assert "returns == ['compare_images'] and accepted == [True]"in text
    assert "frame.f_locals.get('native_comparison') is not None"in text
    assert [ast.unparse(a)for a in timer.args[2:]]==['args.repeat','19']


def test_report_keeps_storage_fingerprints_and_all_timing_samples():
    tree=ast.parse(SOURCE.read_bytes());keywords={n.arg for n in ast.walk(tree)if isinstance(n,ast.keyword)}
    assert {'converted_inputs','output_metadata','output_sha256','reference_sha256','actual_C_returns',
            'native_accepted','harness_sha256','input_sha256','source_before','core_sha256',
            'reference_pool_limits','provider_frontier'}<=keywords
    text=SOURCE.read_text(encoding='utf-8')
    assert 'raw_samples=252'in text and 'for seed in (2026,2027)'in text
    assert 'output destruction'in text and 'Literal img_as_float'in text
    assert 'output_raw_hex'not in text  # Large image words are checked, then hashed.


def test_ci_gates_precede_freeze_and_only_newest_timing_runs():
    workflow=(ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8')
    producer=workflow.split('  numpy-minimum:',1)[1].split('  measure-minimum:',1)[0]
    for mode in ('normal','scalar'):
        target='compare-images-'+mode+'.xml'
        assert producer.count(target)==1 and producer.index(target)<producer.index('ci_measurement_artifacts.py freeze')
    measure=workflow.split('  measure-minimum:',1)[1].split('  debug-gil:',1)[0]
    lines=measure.splitlines()
    timings=[(line,lines[i+1])for i,line in enumerate(lines)if line.strip().startswith('- run: python benchmarks/')]
    assert len(timings)==20
    assert all('!inputs.latest_only'in guard for command,guard in timings if 'benchmarks/compare_images.py'not in command)
    newest=[guard for command,guard in timings if 'benchmarks/compare_images.py'in command]
    assert len(newest)==1 and '!inputs.prepare_only'in newest[0]and 'latest_only'not in newest[0]
    assert 'build/ci-compare-images.md'in measure
    phase3=(ROOT/'.github/workflows/phase3-public-benchmark.yml').read_text(encoding='utf-8')
    assert phase3.count('tests/test_compare_images_public.py')==2
    assert 'len(cases) == 3205'in phase3 and 'len(native) == 239'in phase3 and 'expected == 1417'in phase3
