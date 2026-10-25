"""Installed H203 gate: exact 886 original cases through the sole public core.

A fresh child authenticates original source before importing skimagexx. The
scientific bodies are extracted from the unchanged reviewed 829 private gate;
only unsafe-source skips become explicit blocked/no-call rows. No private core
selector, wrapper bootstrap, original H/reconstruction/clipping forwarding, or
timing path is admitted. Provider stages remain separately visible.
"""
import ast
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH = 'tests/test_h_extrema_floating_pipeline.py'
PRIVATE_SHA256 = '86d4f0491ea8fc89230a300b4b45d85f8a8ce92af1d381e7abe6e52b2f59f148'
FUNCTIONS = ('expected_row', 'final_provider_case', 'expects_real_mask', 'stage_contract', 'retain',
             'test_exact_owned_pipeline', 'test_safe_storage_guards_and_unchanged_integer_intake',
             'test_height_admission_does_not_invoke_protocols', 'test_array_height_storage_precedes_first_range_read')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def public_assertion_nodes():
    raw = (ROOT / PRIVATE_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PRIVATE_SHA256
    nodes = [copy.deepcopy(n) for n in ast.parse(raw).body
             if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    assert tuple(n.name for n in nodes) == FUNCTIONS
    numerical = next(n for n in nodes if n.name == 'test_exact_owned_pipeline')
    before = copy.deepcopy(numerical)
    edits = []
    for node in ast.walk(numerical):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'name in UNSAFE_IDS':
            last = node.body[-1]
            assert ast.unparse(last) == "pytest.skip('unsafe original intake is not an upstream exception contract')"
            node.body[-1] = ast.copy_location(ast.Return(value=None), last)
            edits.append(last)
    assert len(edits) == 1
    inverse = copy.deepcopy(numerical)
    for node in ast.walk(inverse):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'name in UNSAFE_IDS':
            node.body[-1] = edits[0]
    assert ast.dump(inverse, include_attributes=False) == ast.dump(before, include_attributes=False)
    return nodes


def audit_installed(output):
    if sys.flags.optimize != 0:
        raise RuntimeError('H public assertions require sys.flags.optimize=0')
    assert not any(n == 'skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    assert not any(n in os.environ for n in ('SKIMAGEXX_H_EXTREMA_CORE', 'SKIMAGEXX_EXPECT_OPENMP_OFF',
                                           'SKIMAGEXX_RECONSTRUCTION_CORE', 'SKIMAGEXX_H_EXTREMA_FLOAT_REFERENCE'))
    q = load('h_public_source_metadata', ROOT / 'tools/public_qualification.py')
    evolution = q.h_extrema_source_metadata(lambda name: (ROOT / name).read_bytes())
    for name, expected in q.RECONSTRUCTION_RUNTIME_SOURCE_PINS.items():
        assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
    profile = 'windows' if os.name == 'nt' else 'minimum' if importlib.metadata.version('numpy') == '1.24.4' else 'modern'
    reference = load('h_public_original_authentication', ROOT / 'tests/h_extrema_public_authentication.py')
    float_plan, authentication = reference.authenticate_floating(profile)
    cross = json.loads((ROOT / reference.RECORD).read_bytes())
    authentication['reconstruction_reference_metadata'] = {key:cross[key] for key in
        ('schema','previous_public_H_source','reviewed_policy_source','current_raw_sha256',
         'current_canonical_lf_sha256','reviewed_policy_migration_raw_sha256',
         'reviewed_validator_raw_sha256','allowed_assertion_bodies')}
    legacy = load('h_public_original_legacy', ROOT / 'tests/test_h_extrema_contracts.py')
    boundaries = load('h_public_original_boundaries', ROOT / 'tests/test_h_extrema_boundaries.py')
    floating = load('h_public_original_floating', ROOT / 'tests/test_h_extrema_floating_cases.py')
    import numpy as np
    assert legacy.R.profile() == profile
    policy = json.loads((ROOT / 'coverage/contracts/h-extrema-integer.json').read_bytes())
    matrix = json.loads((ROOT / 'coverage/contracts/h-extrema.json').read_bytes())
    float_matrix = json.loads((ROOT / 'coverage/contracts/h-extrema-floating.json').read_bytes())
    old_specs, float_specs = legacy.cases() + boundaries.cases(), floating.cases()
    specs = old_specs + float_specs
    assert len(specs) == len({s['id'] for s in specs}) == 886
    assert [s['id'] for s in old_specs] == policy['all_case_ids']
    assert float_specs == float_plan['cases']
    integer_ids = set(policy['initial_native_ids'])
    pending_ids = set(policy['mandatory_followup_ids'])
    unsafe_ids = set(policy['source_unsafe_blocked_ids'])
    assert (len(integer_ids), len(pending_ids), len(unsafe_ids)) == (424, 378, 14)
    assert not (integer_ids & pending_ids or integer_ids & unsafe_ids or pending_ids & unsafe_ids)
    import skimagexx as xx  # First candidate import; all original authentication is complete.
    core = xx._core
    owned = importlib.import_module('skimagexx._h_extrema')
    reconstruction = importlib.import_module('skimagexx._reconstruction')
    morphology = importlib.import_module('skimagexx.morphology')
    from skimagexx.coverage import native_functions
    assert core.__name__ == 'skimagexx._core' and owned._core is reconstruction._core is core
    assert owned.reconstruction is reconstruction.reconstruction
    package = Path(xx.__file__).resolve().parent
    assert all(Path(m.__file__).resolve().parent == package for m in (core, owned, reconstruction, morphology))
    assert core.get_num_threads() == xx.get_num_threads() == 1
    q.h_extrema_admission(core, morphology, owned, native_functions())
    q.reconstruction_admission(core, morphology, reconstruction, native_functions())
    for module, name in ((owned, 'python/skimagexx/_h_extrema.py'), (reconstruction, 'python/skimagexx/_reconstruction.py')):
        assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == {
            **q.H_EXTREMA_RUNTIME_SOURCE_PINS, **q.RECONSTRUCTION_RUNTIME_SOURCE_PINS}[name]
    for name in ('h_maxima', 'h_minima'):
        assert inspect.signature(vars(morphology)[name]) == inspect.signature(legacy.ORIGINAL[name])
    entries = ('h_extrema_integer_range', 'h_extrema_integer_seed', 'h_extrema_integer_mask',
               'h_extrema_real_mask', 'reconstruction_run')
    returns = {}
    for name in entries:
        entry = getattr(core, name)
        assert inspect.isbuiltin(entry) and entry.__name__ == name and entry.__module__ == core.__name__
        returns[entry] = name
    if os.environ.get('SKIMAGEXX_DISABLE_SIMD'):
        assert not core.has_avx2()
    forbidden = {f.__code__ for f in legacy.ORIGINAL.values()}
    forbidden.update((legacy.ORIGINAL_REC.__code__, legacy.R.SOURCE.rank_order.__code__,
                      legacy.SOURCE._add_constant_clip.__code__, legacy.SOURCE._subtract_constant_clip.__code__))
    summary = {}
    for fallback in (False, True):
        label = 'default' if fallback else 'strict'
        report_path = output.with_name(output.stem + '-' + label + '.json')
        rows, guards = [], []
        journal = report_path.with_suffix('.jsonl').open('x', encoding='utf-8', newline='\n')
        namespace = dict(np=np, pytest=pytest, sys=sys, os=os, json=json, hashlib=hashlib,
            PROFILE=profile, MATRIX=matrix, FLOAT_MATRIX=float_matrix, SPECS=specs,
            INTEGER_IDS=integer_ids, PENDING_IDS=pending_ids, UNSAFE_IDS=unsafe_ids,
            FLOAT_IDS={s['id'] for s in float_specs}, CORE=core, OWNED=owned, LEGACY=legacy,
            BOUNDARIES=boundaries, FLOATING=floating, RETURNS=returns, ENTRY_NAMES=entries,
            FORBIDDEN=forbidden, ROWS=rows, GUARDS=guards, JOURNAL=journal)
        exec(compile(ast.Module(body=public_assertion_nodes(), type_ignores=[]),
                     '<authenticated-installed-H829-assertions>', 'exec'), namespace)
        assert sum(namespace['expects_real_mask'](s) for s in old_specs) == (284 if profile == 'minimum' else 280)
        assert sum(namespace['expects_real_mask'](s) for s in float_specs) == (44 if profile == 'minimum' else 40)
        previous = xx.set_fallback_enabled(fallback)
        failure = None
        try:
            assert xx.set_fallback_enabled(fallback) is fallback
            for workers in (1, 4):
                xx.set_num_threads(workers)
                assert core.get_num_threads() == xx.get_num_threads() == workers
                for spec in specs:
                    namespace['test_exact_owned_pipeline'](spec, workers)
            namespace['test_safe_storage_guards_and_unchanged_integer_intake']()
            namespace['test_height_admission_does_not_invoke_protocols']()
            with pytest.MonkeyPatch.context() as monkeypatch:
                namespace['test_array_height_storage_precedes_first_range_read'](monkeypatch)
        except BaseException as error:
            failure = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            journal.close()
            xx.set_num_threads(1)
            xx.set_fallback_enabled(previous)
            report = dict(schema='h-extrema-installed-public886-v1', profile=profile, fallback_enabled=fallback,
                source_authentication=authentication, source_evolution=evolution,
                core=dict(path=str(Path(core.__file__).resolve()), sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()),
                default_workers=1, final_workers=core.get_num_threads(), has_avx2=core.has_avx2(),
                disable_simd=os.environ.get('SKIMAGEXX_DISABLE_SIMD'), observations=rows, guards=guards,
                failure=failure, full_API_complete=False, private_adapter=False, operation_timings=0)
            with report_path.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write('\n')
        assert len(rows) == 1772 and len(guards) == 14 and all(g['passed'] for g in guards)
        assert [(r['id'], r['workers']) for r in rows] == [(s['id'], w) for w in (1,4) for s in specs]
        blocked = [r for r in rows if r.get('status') == 'source-unsafe-blocked']
        assert len(blocked) == 28 and all(r['actual_calls'] == 0 for r in blocked)
        safe = [r for r in rows if r['id'] not in unsafe_ids]
        assert len(safe) == 1744 and all(r['public_observation_exact'] and r['intermediate_stages_exact'] for r in safe)
        assert not any(r.get('forwarding_attempts') for r in rows)
        summary[label] = dict(rows=len(rows), safe=len(safe), blocked=len(blocked), guards=len(guards), fallback_enabled=fallback)
    result = dict(logical_ids=886, policies=summary, rows=3544, private_adapter=False, full_API_complete=False, operation_timings=0)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


def test_installed_h_extrema_exact_886_partition(tmp_path):
    output = tmp_path / 'installed-h-extrema.json'
    env = dict(os.environ)
    for key in ('PYTHONOPTIMIZE', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()),
                             '--audit-installed', str(output)], env=env, capture_output=True,
                            text=True, encoding='utf-8', timeout=1200)
    (tmp_path / 'child.stdout.txt').write_text(result.stdout, encoding='utf-8')
    (tmp_path / 'child.stderr.txt').write_text(result.stderr, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_bytes())
    assert report['logical_ids'] == 886 and report['rows'] == 3544
    assert report['policies'] == {name:dict(rows=1772, safe=1744, blocked=28, guards=14, fallback_enabled=value)
                                  for name,value in (('strict',False),('default',True))}
    assert report['private_adapter'] is False and report['full_API_complete'] is False


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] == '--audit-installed'
    audit_installed(Path(sys.argv[2]))
