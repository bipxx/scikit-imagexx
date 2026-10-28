"""Inert checks of the fixed fresh-host witness; no target or scientific calls."""
import ast
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import textwrap
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / '.github/workflows/approximate-polygon-protocol-witness.yml').read_text()
SCRIPTS = [textwrap.dedent(s) for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n", WORKFLOW, re.S)]
# The runtime uses an isolated interpreter with additional flags.
RUNTIME = textwrap.dedent(WORKFLOW.split('" -I -B - <<\'PY\'\n', 1)[1].split('          PY\n', 1)[0])
PREPARE = SCRIPTS[0]


def helper(source, name, **namespace):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-witness-helper', 'exec'), namespace)
    return namespace[name]


def constant(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)
    # CAPSULES contains literal dict(...) calls, not executable source hooks.
    if name == 'CAPSULES':
        return {ast.literal_eval(k): {kw.arg: ast.literal_eval(kw.value) for kw in value.keywords}
                for k, value in zip(node.value.keys, node.value.values)}
    return ast.literal_eval(node.value)


def test_fixed_sources_artifacts_offline_boundary_and_no_dispatch_or_build():
    compile(PREPARE, 'prepare', 'exec'); compile(RUNTIME, 'runtime', 'exec')
    capsules = constant(PREPARE, 'CAPSULES')
    assert capsules['failed'] == dict(run=40765084067, revision='707e7ccb542d145f83a1f25ac4f6616068e35a51', artifact=12849353092, name='numpy-minimum-ciede2000', conclusion='failure', sha256='8831c6a818c1f75a4d621f9d1c25b2c0d9bb005015dc182ffe7e56f026c2ef42')
    assert capsules['dependencies'] == dict(run=40758210302, revision='390dff892ad13c1f9b8cee0e76dce9a016f14c01', artifact=12846458475, name='ci-performance-minimum', conclusion='success', sha256='d064c77793dbadd81c8de20e108f49467d19ffa59910fbccdd8609061cd42b89')
    assert "python-version: '3.11.17'" in WORKFLOW and 'runs-on: ubuntu-24.04' in WORKFLOW
    assert 'inputs:' not in WORKFLOW and 'if: always()' in WORKFLOW
    assert '--no-index --no-deps --only-binary=:all: --require-hashes' in WORKFLOW
    assert not re.search(r'pip wheel|cmake|ctest|python -m pytest|gh workflow.*dispatch|--method POST|gh run rerun', WORKFLOW)
    assert "r['name']!='scikit-imagexx'" in PREPARE and 'len(dependencies)==13' in PREPARE
    assert "(layout['product_count'],layout['SDK_count'])==(174,82)" in PREPARE
    assert "donor dependencies are separately authenticated, not failed-host binary identity" in PREPARE


def test_frozen_factory_observer_gold_and_exact_original_source_pins():
    assert constant(RUNTIME, 'IDS') == ('tolerance-provider-positive', 'tolerance-mutates-coordinates')
    assert constant(RUNTIME, 'PINS') == {
        'tests/approximate_polygon_contract.py':'6f79e5c91f761bd67b5499abc6842460ee132f347edce337c3168ea0252f821f',
        'tests/test_approximate_polygon_public.py':'e08bfdfe6cdef0a0ad24a5b2ed3660fcaf02ea9d2fa678b429eb8221a2d58c4e',
        'coverage/contracts/approximate-polygon.json':'101bb58c8ceae38d0a5dd2d66a8cfa628d069e41e2ef4d30740bb5e5c59ee758'}
    assert '26a27d89545d158f6b542857297995639b251ed3af0428c8c86a56c84928aabe' in RUNTIME
    assert 'sum(map(len,matrix[\'original_observations\'].values()))==120' in RUNTIME
    assert "report['gold_raw_unchanged']=sha(gold_path)==gold_raw_sha256" in RUNTIME
    assert "'NEW HOST, not the original failure host." in RUNTIME
    assert "assert 'skimagexx' not in sys.modules" in RUNTIME


@pytest.mark.parametrize('changed', (False, True))
def test_actual_orchestration_two_originals_before_import_and_four_frontends(changed):
    capture = helper(RUNTIME, 'capture_rows')
    ids = constant(RUNTIME, 'IDS'); specs = [dict(id=name) for name in ids]
    timeline, journal = [], []
    def save(row): journal.append(deepcopy(row))
    def reference(spec):
        assert journal[-1] == dict(stage='before', target='original', id=spec['id'])
        timeline.append(('original', spec['id']))
        return dict(id=spec['id'], events=[.5])
    def load(originals):
        assert timeline == [('original', name) for name in ids]
        assert list(originals) == list(ids)
        timeline.append(('import-product',))
        def frontend(spec, workers):
            assert journal[-1] == dict(stage='before', target='frontend', id=spec['id'], workers=workers)
            timeline.append(('frontend', spec['id'], workers))
            return dict(id=spec['id'], workers=workers, observation=dict(id=spec['id'], events=[.6 if changed else .5]))
        return frontend
    originals, rows = capture(specs, reference, load, save)
    assert timeline == [('original', name) for name in ids] + [('import-product',)] + [
        ('frontend', name, workers) for workers in (1, 4) for name in ids]
    assert len(originals) == 2 and len(rows) == 4 and len(journal) == 12
    assert len([r for r in journal if r['target'] == 'original' and r['stage'] == 'after']) == 2
    assert all('record' in r for r in journal if r['target'] == 'frontend' and r['stage'] == 'after')
    assert "journal.flush()" in RUNTIME


def test_actual_difference_paths_preserve_changed_callback_values():
    diff = helper(RUNTIME, 'differences')
    old = dict(events=dict(items=[dict(event='tolerance.__lt__', other=.3535533905932735)]), result='unchanged')
    new = deepcopy(old); new['events']['items'][0]['other'] = .35355339059327395
    assert diff(old, new) == [dict(path='/events/items/0/other', expected=.3535533905932735, observed=.35355339059327395)]
    assert diff(old, old) == []


def test_reports_are_written_before_comparison_failure_and_provider_calls_trapped():
    assert RUNTIME.index("(out/'original-records.json').write_text") < RUNTIME.index('import skimagexx as xx')
    assert RUNTIME.index("(out/'report.json').write_text") < RUNTIME.index("assert report['runtime_original_equal_all']")
    assert "if event=='call' and frame.f_code is reference.__code__: reject()" in RUNTIME
    assert "if event=='c_call': calls.append(entry.__name__)" in RUNTIME
    assert "elif event=='c_return': returns.append(entry.__name__)" in RUNTIME
    assert "not r['C_calls'] and not r['C_returns'] and not r['forwarded']" in RUNTIME
    assert 'sys.setprofile(prior)' in RUNTIME
    assert 'before_product_import' in RUNTIME and 'after_product_import' in RUNTIME
    for field in ('identity_before', 'identity_after', 'numpy_cpu_features', 'ImageVersion',
                  'libc', 'numpy_core', 'policy', 'handler_type', 'buffer_elements'):
        assert field in RUNTIME
    assert 'lscpu > results/cpu.txt' in WORKFLOW


@pytest.mark.parametrize('label', ('failed', 'dependencies'))
def test_actual_fetch_accepts_only_exact_terminal_capsule(tmp_path, monkeypatch, label):
    exercise_fetch(tmp_path, monkeypatch, label, None)


@pytest.mark.parametrize('fault', ('source', 'run', 'repository', 'workflow', 'conclusion', 'artifact',
                                   'expired', 'digest', 'foreign-run', 'traversal', 'symlink', 'duplicate'))
def test_actual_fetch_rejects_wrong_identity_or_unsafe_archive(tmp_path, monkeypatch, fault):
    exercise_fetch(tmp_path, monkeypatch, 'failed', fault)


def exercise_fetch(tmp_path, monkeypatch, label, fault):
    spec = deepcopy(constant(PREPARE, 'CAPSULES')[label])
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        info = zipfile.ZipInfo('../escape' if fault == 'traversal' else 'wheels/product.whl')
        if fault == 'symlink': info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, b'bounded inert bytes')
        if fault == 'duplicate':
            with pytest.warns(UserWarning, match='Duplicate name'): z.writestr(info, b'duplicate')
    raw = stream.getvalue(); spec['sha256'] = hashlib.sha256(raw).hexdigest()
    record = dict(id=spec['run'], head_sha=spec['revision'], path='.github/workflows/ci.yml',
                  head_repository=dict(full_name='bipxx/scikit-imagexx'), status='completed', conclusion=spec['conclusion'])
    item = dict(id=spec['artifact'], name=spec['name'], expired=False, digest='sha256:'+spec['sha256'],
                size_in_bytes=len(raw), workflow_run=dict(id=spec['run'], head_sha=spec['revision']))
    if fault == 'source': record['head_sha'] = 'a'*40
    elif fault == 'run': record['id'] += 1
    elif fault == 'repository': record['head_repository']['full_name'] = 'foreign/repo'
    elif fault == 'workflow': record['path'] = 'other.yml'
    elif fault == 'conclusion': record['conclusion'] = 'success'
    elif fault == 'artifact': item['id'] += 1
    elif fault == 'expired': item['expired'] = True
    elif fault == 'digest': item['digest'] = 'sha256:'+'a'*64
    elif fault == 'foreign-run': item['workflow_run']['id'] += 1
    replies = [record, dict(total_count=1, artifacts=[item])]
    def download(args, stdout, check): stdout.write(raw)
    subprocess = SimpleNamespace(check_output=lambda args: json.dumps(replies.pop(0)).encode(), run=download)
    result = tmp_path/'results'; result.mkdir()
    monkeypatch.setenv('GITHUB_REPOSITORY', 'bipxx/scikit-imagexx'); monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    fetch = helper(PREPARE, 'fetch', json=json, subprocess=subprocess, out=result,
                   base='repos/bipxx/scikit-imagexx/actions/', os=os, Path=Path, PurePosixPath=__import__('pathlib').PurePosixPath,
                   hashlib=hashlib, zipfile=zipfile, stat=stat)
    if fault:
        with pytest.raises(AssertionError): fetch(label, spec)
    else:
        path = fetch(label, spec)
        assert path.read_bytes() == raw
        assert (result/(label+'-run.json')).exists()
