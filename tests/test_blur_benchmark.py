"""Blur benchmark metadata; no scientific imports or timing."""
import ast
import shlex
import builtins
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'benchmarks/blur_effect.py'


def test_fixed_twenty_four_cases_import_without_scientific_dependencies(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0]not in('numpy','scipy','skimage','skimagexx'),name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    spec=importlib.util.spec_from_file_location('blur_benchmark_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=module.case_specs()
    assert len(rows)==len({r['case_id']for r in rows})==24
    shapes=((4096,),(65536,),(256,256),(1024,1024),(32,32,32),(64,64,64))
    assert {(tuple(r['shape']),r['dtype'],r['h_size'])for r in rows}=={
        (shape,d,h)for shape,h in [*((s,11)for s in shapes),((256,256),1),((1024,1024),21)]
        for d in('uint8','float32','float64')}
    assert sum(r['axes']for r in rows)==48
    assert all(r['axes']==len(r['shape'])and r['pixels']>0 for r in rows)


def test_exact_six_stage_and_public_checks_precede_timing():
    tree=ast.parse(SOURCE.read_bytes());text=ast.unparse(tree)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    timer=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure')
    exact=[n for n in ast.walk(main)if isinstance(n,ast.Assert)and 'actual.tobytes() == expected.tobytes()'in ast.unparse(n)]
    assert len(exact)==1 and exact[0].lineno<timer.lineno
    assert "stages = ('filt_im', 'im_sharp', 'im_blur', 'T', 'M1', 'M2')"in text
    assert 'actual_stages == expected_stages'in text and 'actual_record == expected_record'in text
    assert "accepted == [True] * spec['axes']"in text
    assert "returns == ['uniform_filter1d', 'correlate_sparse', 'correlate_sparse'] * spec['axes']"in text
    restore=next(n for n in ast.walk(main)if isinstance(n,ast.Try)and any('sys.setprofile(prior_profile)'in ast.unparse(f)for f in n.finalbody))
    assert restore.end_lineno<timer.lineno
    assert any('sys.settrace(prior_trace)'in ast.unparse(f)for f in restore.finalbody)
    assert [ast.unparse(a)for a in timer.args[2:]]==['args.repeat','19']
    assert 'raw_samples=336'in text and 'total_uniform == 48'in text and 'total_sparse == 96'in text


def test_harness_reports_actual_stage_storage_and_provider_scope():
    tree=ast.parse(SOURCE.read_bytes());keywords={n.arg for n in ast.walk(tree)if isinstance(n,ast.keyword)}
    assert {'api','axes','pixels','input_dtype_str','intermediates','output_metadata',
            'output_raw_hex','reference_raw_hex','actual_C_returns','native_accepted',
            'harness_sha256','source_before','core_sha256','reference_pool_limits','provider_frontier',
            'accepted_uniform_stages','accepted_sparse_stages'}<=keywords
    text=SOURCE.read_text(encoding='utf-8')
    assert 'ordered rolling recurrence'in text and 'cropped reductions'in text
    assert 'np.random.default_rng(2026)'in text and '(gray/255).astype'in text
    assert 'EPSILON promotion rules'in text


def test_new_blur_keeps_all_gates_and_skips_only_old_timings():
    workflow=(ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8')
    producer=workflow.split('  numpy-minimum:',1)[1].split('  measure-minimum:',1)[0]
    for mode in('normal','scalar','intake'):
        target='blur-effect-'+mode+'.xml'
        assert producer.count(target)==1 and producer.index(target)<producer.index('ci_measurement_artifacts.py freeze')
    measure=workflow.split('  measure-minimum:',1)[1].split('  debug-gil:',1)[0]
    lines=measure.splitlines()
    # Reuse the central complete40/prepare_only/latest_only policy; this family is historical.
    timings=[(shlex.split(line.strip().removeprefix('- run: ')),lines[i+1].strip())
             for i,line in enumerate(lines)if line.strip().startswith('- run: python benchmarks/')]
    assert len(timings)==len({tuple(command)for command,guard in timings})==40
    prepare="(github.event_name != 'workflow_dispatch' || !inputs.prepare_only)"
    skip="(github.event_name != 'workflow_dispatch' || !inputs.latest_only)"
    older_guards={'if: '+prepare+' && '+skip, 'if: '+prepare[1:-1]+' && '+skip}
    assert all(guard in older_guards for command,guard in timings[:-1])
    assert timings[-1][1] in {'if: '+prepare, 'if: '+prepare[1:-1]}
    assert timings[-1][0]==['python','benchmarks/label_pair_metrics.py','--threads','1','--repeat','7',
                            '--output','build/ci-label-pair-metrics.json']
    family=[command for command,guard in timings if command[1]=='benchmarks/blur_effect.py']
    assert family==[['python','benchmarks/blur_effect.py','--threads','1','--repeat','7',
                     '--output','build/ci-blur-effect.json']]
    assert 'build/ci-blur-effect.md'in measure
    phase3=(ROOT/'.github/workflows/phase3-public-benchmark.yml').read_text(encoding='utf-8')
    # Retained manual recipe only; current-product eligibility remains a separate gate.
    public=[shlex.split(line.strip().split('2>&1',1)[0].split('|',1)[0])
            for line in phase3.splitlines()if line.strip().startswith('python -m pytest -q ')
            and '--junitxml build/phase3/preparation/public-'in line]
    assert len(public)==2
    for mode,command in zip(('normal','scalar'),public):
        assert command[:4]==['python','-m','pytest','-q']
        assert command.count('tests/test_blur_effect_public.py')==1
        assert command[command.index('--junitxml')+1]=='build/phase3/preparation/public-'+mode+'.xml'
    assert 'len(cases) == 5472'in phase3
    assert 'len(entries) == 477 and len(native) == 250'in phase3 and 'expected == 1447'in phase3
