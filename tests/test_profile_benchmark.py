"""Profile workload metadata; never imports scientific packages or times code."""
import ast
import builtins
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'benchmarks/profile_line.py'


def test_fixed_eighteen_profiles_import_without_scientific_dependencies(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0]not in('numpy','scipy','skimage','skimagexx'),name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    spec=importlib.util.spec_from_file_location('profile_benchmark_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=module.case_specs()
    assert len(rows)==len({r['case_id']for r in rows})==18
    assert {(r['shape'][0],r['dtype'],r['linewidth'])for r in rows}=={
        (s,d,w)for s in(256,1024)for d in('uint8','float32','float64')for w in(1,9,33)}
    assert {r['profile_length']for r in rows}=={360,1446}
    assert all(r['sampled_values']==r['profile_length']*r['linewidth']for r in rows)
    assert all(r['src']==(.25,1.5)and r['dst']==(r['shape'][0]-1.25,r['shape'][0]-1.75)for r in rows)


def test_exact_public_and_pre_reducer_checks_precede_timing():
    tree=ast.parse(SOURCE.read_bytes());text=ast.unparse(tree)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    timer=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure')
    exact=[n for n in ast.walk(main)if isinstance(n,ast.Assert)and 'actual.tobytes() == expected.tobytes()'in ast.unparse(n)]
    assert len(exact)==1 and exact[0].lineno<timer.lineno
    assert 'captured[0] == captured[1]'in text
    assert 'accepted == [True]'in text and "frame.f_locals.get('native_pixels') is not None"in text
    restore=next(n for n in ast.walk(main)if isinstance(n,ast.Try)and any('sys.setprofile(prior)'in ast.unparse(f)for f in n.finalbody))
    assert restore.end_lineno<timer.lineno
    assert [ast.unparse(a)for a in timer.args[2:]]==['args.repeat','19']
    assert "xx.set_num_threads(args.threads)"in text and 'raw_samples=252'in text


def test_harness_reports_actual_sample_work_storage_and_provider_scope():
    tree=ast.parse(SOURCE.read_bytes());keywords={n.arg for n in ast.walk(tree)if isinstance(n,ast.keyword)}
    assert {'api','sampled_values','profile_length','input_dtype_str','sample_sha256','pre_reducer_metadata',
            'output_shape','output_strides','output_raw_hex','reference_raw_hex','actual_C_returns','native_accepted',
            'harness_sha256','source_before','core_sha256','reference_pool_limits','provider_frontier'}<=keywords
    text=SOURCE.read_text(encoding='utf-8')
    assert 'not image pixels'in text and 'literal NumPy providers'in text
    assert 'np.random.default_rng(2026)'in text and '(gray/255).astype'in text


def test_new_profiles_keep_correctness_gates_and_skip_only_old_timings():
    workflow=(ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8')
    producer=workflow.split('  numpy-minimum:',1)[1].split('  measure-minimum:',1)[0]
    for mode in('normal','scalar'):
        target='profile-line-'+mode+'.xml'
        assert producer.count(target)==1 and producer.index(target)<producer.index('ci_measurement_artifacts.py freeze')
    measure=workflow.split('  measure-minimum:',1)[1].split('  debug-gil:',1)[0]
    lines=measure.splitlines()
    timings=[(line,lines[i+1])for i,line in enumerate(lines)if line.strip().startswith('- run: python benchmarks/')]
    assert len(timings)==19
    assert all('!inputs.latest_only'in guard for command,guard in timings if 'benchmarks/blur_effect.py'not in command)
    newest=[guard for command,guard in timings if 'benchmarks/blur_effect.py'in command]
    assert len(newest)==1 and '!inputs.prepare_only'in newest[0]and 'latest_only'not in newest[0]
    assert 'build/ci-profile-line.md'in measure
    phase3=(ROOT/'.github/workflows/phase3-public-benchmark.yml').read_text(encoding='utf-8')
    assert phase3.count('tests/test_profile_line_public.py')==2
    assert 'len(cases) == 3045'in phase3 and 'len(native) == 238'in phase3 and 'expected == 1413'in phase3
