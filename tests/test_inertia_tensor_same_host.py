"""Metadata-only guards for the inertia artifact comparison; no timing calls."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import re
import textwrap
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / '.github/workflows/inertia-tensor-same-host.yml').read_text()
SCRIPTS = [textwrap.dedent(s) for s in re.findall(
    r"python - <<'PY'\n(.*?)          PY\n", WORKFLOW, re.S)]
PASSES = ('1-baseline', '2-candidate', '3-candidate', '4-baseline')
COMMON = ('.github/workflows/ci.yml', 'benchmarks/coverage.py', 'benchmarks/run.py',
          'benchmarks/provenance.py', 'tools/ci_measurement_artifacts.py',
          'tools/run_feature_tests.py', 'tests/moments_cases.py', 'tests/test_moments_public.py',
          'tests/inertia_tensor_cases.py', 'tests/test_inertia_tensor_public.py',
          'tests/moments_derived_cases.py', 'tests/test_moments_derived_public.py',
          'tests/window_cases.py', 'tests/test_window_public.py',
          'tests/lpi_filter_cases.py', 'tests/test_lpi_filter_public.py',
          'tests/butterworth_cases.py', 'tests/test_butterworth_public.py',
          'benchmarks/normalized_layouts_cases.json', 'tests/test_normalized_layouts_public.py',
          'tests/test_immutable_buffer_roots_public.py')


def test_fresh_host_fixed_command_and_retention():
    assert len(SCRIPTS) == 3
    for source in SCRIPTS:
        compile(source, 'workflow-inline', 'exec')
    assert "default: '40378736391'" in WORKFLOW
    assert "default: '7456bfd5c03c32dcb617ff05f8ef1b8d9b21e3aa'" in WORKFLOW
    assert WORKFLOW.count('actions/checkout@v4') == 2
    block = WORKFLOW.split('      - name: Measure the fixed6', 1)[1].split('      - name:', 1)[0]
    command = ('benchmarks/coverage.py --api measure.inertia_tensor '
               '--sizes 256 1024 --threads 1 --repeat 7')
    assert command in block and 'for pass in ' + ' '.join(PASSES) + '; do' in block
    assert block.count('ci_measurement_artifacts.py verify') == 2
    assert block.index('verify') < block.index(command) < block.rindex('verify')
    assert block.index('benchmark-exit.txt') < block.index('identity-after.log')
    assert 'unset PYTHONPATH PYTHONHOME SKIMAGEXX_DISABLE_SIMD' in block
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel', block)
    assert WORKFLOW.index('--no-index --no-deps --only-binary=:all: --require-hashes') < WORKFLOW.index(command)
    assert '"$RUNNER_TEMP/$role-prepared.zip"' in WORKFLOW and 'path: results/' in WORKFLOW
    assert 'if: always()' in WORKFLOW and 'name: inertia-tensor-same-host-minimum' in WORKFLOW


@pytest.mark.parametrize('bad', [None, 'revision', 'repository', 'workflow', 'unfinished',
                                 'duplicate', 'artifact-run', 'digest'])
def test_api_identity_rejects_substitution(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path); (tmp_path / 'results').mkdir()
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    for role, number, revision in [('BASELINE', '11', 'b' * 40), ('CANDIDATE', '22', 'c' * 40)]:
        monkeypatch.setenv(role + '_RUN', number); monkeypatch.setenv(role + '_REVISION', revision)
    def response(argv):
        endpoint = argv[-1]; number = 11 if '/runs/11' in endpoint else 22
        revision = ('b' if number == 11 else 'c') * 40
        record = dict(id=number, head_sha=revision, head_repository={'full_name': 'owner/repo'},
                      path='.github/workflows/ci.yml', status='completed', conclusion='success')
        item = dict(id=number + 100, name='ci-performance-minimum', expired=False, size_in_bytes=123,
                    workflow_run={'id': number, 'head_sha': revision}, digest='sha256:' + 'a' * 64)
        if bad == 'revision': record['head_sha'] = 'd' * 40
        if bad == 'repository': record['head_repository']['full_name'] = 'other/repo'
        if bad == 'workflow': record['path'] = '.github/workflows/other.yml'
        if bad == 'unfinished': record['status'] = 'in_progress'
        if bad == 'artifact-run': item['workflow_run']['id'] += 1
        if bad == 'digest': item['digest'] = 'missing'
        items = [item, item] if bad == 'duplicate' else [item]
        return json.dumps({'total_count': len(items), 'artifacts': items} if '/artifacts?' in endpoint else record).encode()
    monkeypatch.setattr('subprocess.check_output', response)
    if bad:
        with pytest.raises(AssertionError): exec(compile(SCRIPTS[0], 'api-selection', 'exec'), {})
    else:
        exec(compile(SCRIPTS[0], 'api-selection', 'exec'), {})
        assert len(list((tmp_path / 'results').glob('*-selection.json'))) == 2


def pair_check(old, new):
    branch = next(n for n in ast.parse(SCRIPTS[1]).body
                  if isinstance(n, ast.If) and ast.unparse(n.test) == "role == 'candidate'")
    # Supply metadata directly instead of opening ZIPs or installed packages.
    exec(compile(ast.Module(branch.body[1:], type_ignores=[]), 'pair-equality', 'exec'),
         {'old': old, 'new': new})


def manifest():
    runtime = {k: 'same' for k in ('python', 'implementation', 'cache_tag', 'machine', 'system')}
    runtime.update(dependency_libraries_sha256={'numpy': {'blas.so': 'a'}},
                   versions={'numpy': '1.24.4', 'scikit-imagexx': '0.1.0'},
                   packages_sha256={k: {'source.py': 'a'} for k in ('numpy', 'scipy', 'skimage')})
    paths = (*COMMON, 'CMakeLists.txt', 'pyproject.toml', 'src/moments.cpp',
             'src/simd_moments_avx2.cpp', 'include/skimagexx/moments.hpp',
             'python/skimagexx/_moments.py', 'python/skimagexx/_inertia_checks.py',
             'src/inertia.cpp',
             'tests/test_moments_native.cpp')
    wheels = {f'dep{i}.whl': {'name': f'dep{i}', 'sha256': 'a'} for i in range(13)}
    wheels['native.whl'] = {'name': 'scikit-imagexx', 'sha256': 'a'}
    return dict(runtime=runtime, wheels=wheels, source={'files_sha256': dict.fromkeys(paths, 'a')})


def test_leaf_and_additive_native_controls_are_allowed():
    old = manifest(); new = deepcopy(old)
    for name in ('src/simd_moments_avx2.cpp', 'tests/test_moments_native.cpp'):
        new['source']['files_sha256'][name] = 'b'
    new['source']['files_sha256']['.github/workflows/inertia-tensor-same-host.yml'] = 'b'
    new['wheels']['native.whl']['sha256'] = 'b'
    pair_check(old, new)


@pytest.mark.parametrize('bad', ['frontend', 'policy', 'benchmark', 'other-kernel', 'build',
                                 'public-cases', 'missing-common', 'new-product-file', 'wheel', 'runtime'])
def test_other_product_policy_and_runtime_changes_reject(bad):
    old = manifest(); new = deepcopy(old); paths = new['source']['files_sha256']
    names = {'frontend': 'python/skimagexx/_moments.py', 'policy': 'python/skimagexx/_inertia_checks.py',
             'benchmark': 'benchmarks/coverage.py', 'other-kernel': 'src/inertia.cpp',
             'build': 'CMakeLists.txt', 'public-cases': 'tests/inertia_tensor_cases.py',
             'new-product-file': 'src/new.cpp'}
    if bad in names: paths[names[bad]] = 'changed'
    elif bad == 'missing-common': del paths['benchmarks/run.py']
    elif bad == 'wheel': new['wheels']['dep0.whl']['sha256'] = 'changed'
    else: new['runtime']['dependency_libraries_sha256']['numpy']['blas.so'] = 'changed'
    with pytest.raises(AssertionError): pair_check(old, new)


@pytest.mark.parametrize('bad', [None, 'missing', 'duplicate', 'worker', 'sample', 'median', 'fallback', 'scalar', 'nonfinite-error'])
def test_raw_reports_must_keep_every_case_and_sample(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    rows = [dict(api=api, size=n, dtype=d, threads=1, backend='native',
                 sample_labels=['scikit-image', 'current'], samples_ms=[[1.] * 7, [.5] * 7],
                 loops=[1, 1], reference_ms=1., cpp_ms=.5, speedup=2., max_abs_error=0.)
            for api in ('measure.inertia_tensor',)
            for n in (256, 1024) for d in ('uint8', 'float32', 'float64')]
    meta = dict(repeat=7, fallback_enabled=False, avx2=True, reference='0.26.0', numpy='1.24.4',
                selected_apis=['measure.inertia_tensor'], reference_fft_workers=None,
                provider_frontiers={'measure.inertia_tensor': 'explicit providers'})
    if bad == 'missing': rows.pop()
    elif bad == 'duplicate': rows[-1] = deepcopy(rows[0])
    elif bad == 'worker': rows[0]['threads'] = 4
    elif bad == 'sample': rows[0]['samples_ms'][0].pop()
    elif bad == 'median': rows[0]['cpp_ms'] = .6
    elif bad == 'fallback': meta['fallback_enabled'] = True
    elif bad == 'scalar': meta['avx2'] = False
    elif bad == 'nonfinite-error': rows[0]['max_abs_error'] = float('nan')
    for name in PASSES:
        folder = tmp_path / 'results' / name; folder.mkdir(parents=True)
        (folder / 'inertia-tensor.json').write_text(json.dumps({'metadata': meta, 'results': rows}))
    if bad:
        with pytest.raises(AssertionError): exec(compile(SCRIPTS[2], 'report-check', 'exec'), {})
    else: exec(compile(SCRIPTS[2], 'report-check', 'exec'), {})


@pytest.mark.parametrize('bad', [None, 'missing-family', 'short-family', 'failure', 'skip', 'smoke', 'registry', 'wheel-count'])
def test_preparation_only_product_requires_all_eight_cohorts(tmp_path, monkeypatch, bad):
    # No standalone benchmark JSON is created: prepare_only products are valid inputs.
    for family, count in (('inertia-tensor', 115), ('moments', 203), ('window', 179),
                          ('lpi', 161), ('butterworth', 175), ('moments-derived', 155),
                          ('normalized-layouts', 22), ('immutable-bytes', 134)):
        for mode in ('normal', 'scalar'):
            root = ET.Element('testsuite')
            for index in range(count): ET.SubElement(root, 'testcase', name=str(index))
            if family == 'inertia-tensor' and mode == 'scalar':
                if bad == 'missing-family': continue
                if bad == 'short-family': root.remove(root[-1])
                if bad in ('failure', 'skip'): ET.SubElement(root[0], 'failure' if bad == 'failure' else 'skipped')
            (tmp_path / (family + '-' + mode + '.xml')).write_bytes(ET.tostring(root))
    smoke = dict(status='passed', exit_code=0, outcomes={'passed': 1361}, results=[{}] * 1361,
                 implementations={'native-partial': 222, 'metadata': 3})
    if bad == 'smoke': smoke['results'].pop()
    if bad == 'registry': smoke['implementations']['native-partial'] = 221
    (tmp_path / 'feature-smoke.json').write_text(json.dumps(smoke))
    value = manifest(); value.update(schema='ci-offline-measurement-v1')
    value['source']['revision'] = 'b' * 40
    if bad == 'wheel-count': value['wheels'].pop('dep0.whl')
    (tmp_path / 'measurement.json').write_text(json.dumps(value))
    monkeypatch.setenv('BASELINE_REVISION', 'b' * 40)
    body = ast.parse(SCRIPTS[1]).body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'manifest')
    stop = next(i for i, n in enumerate(body) if isinstance(n, ast.If) and ast.unparse(n.test) == "role == 'candidate'")
    import os
    namespace = dict(prepared=tmp_path, role='baseline', json=json, ET=ET, os=os)
    code = compile(ast.Module(body[start:stop], type_ignores=[]), 'preparation-check', 'exec')
    if bad:
        with pytest.raises((AssertionError, FileNotFoundError)): exec(code, namespace)
    else: exec(code, namespace)
