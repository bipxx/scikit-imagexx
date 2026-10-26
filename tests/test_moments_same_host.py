"""Metadata-only guards for the moments artifact comparison; no timing calls."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import re
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / '.github/workflows/moments-same-host.yml').read_text()
SCRIPTS = [textwrap.dedent(s) for s in re.findall(
    r"python - <<'PY'\n(.*?)          PY\n", WORKFLOW, re.S)]
PASSES = ('1-baseline', '2-candidate', '3-candidate', '4-baseline')
COMMON = ('.github/workflows/ci.yml', 'benchmarks/coverage.py', 'benchmarks/run.py',
          'benchmarks/provenance.py', 'tools/ci_measurement_artifacts.py',
          'tools/run_feature_tests.py', 'tests/moments_cases.py', 'tests/test_moments_public.py')


def test_fresh_host_fixed_command_and_retention():
    assert len(SCRIPTS) == 3
    for source in SCRIPTS:
        compile(source, 'workflow-inline', 'exec')
    assert "default: '40333236658'" in WORKFLOW
    assert "default: '07a3074bf0e5194d6e96a25ebe4aad4998b04706'" in WORKFLOW
    assert WORKFLOW.count('actions/checkout@v4') == 2
    block = WORKFLOW.split('      - name: Measure the fixed18', 1)[1].split('      - name:', 1)[0]
    command = ('benchmarks/coverage.py --api measure.moments --api measure.moments_central '
               '--api measure.centroid --sizes 256 1024 --threads 1 --repeat 7')
    assert command in block and 'for pass in ' + ' '.join(PASSES) + '; do' in block
    assert block.count('ci_measurement_artifacts.py verify') == 2
    assert block.index('verify') < block.index(command) < block.rindex('verify')
    assert block.index('benchmark-exit.txt') < block.index('identity-after.log')
    assert 'unset PYTHONPATH PYTHONHOME SKIMAGEXX_DISABLE_SIMD' in block
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel', block)
    assert WORKFLOW.index('--no-index --no-deps --only-binary=:all: --require-hashes') < WORKFLOW.index(command)
    assert '"$RUNNER_TEMP/$role-prepared.zip"' in WORKFLOW and 'path: results/' in WORKFLOW
    assert 'if: always()' in WORKFLOW and 'name: moments-same-host-minimum' in WORKFLOW


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
             'python/skimagexx/_moments.py', 'python/skimagexx/_moments_checks.py',
             'tests/test_moments_native.cpp')
    return dict(runtime=runtime, wheels={'numpy.whl': {'name': 'numpy', 'sha256': 'a'},
                                        'native.whl': {'name': 'scikit-imagexx', 'sha256': 'a'}},
                source={'files_sha256': dict.fromkeys(paths, 'a')})


def test_leaf_and_additive_native_controls_are_allowed():
    old = manifest(); new = deepcopy(old)
    for name in ('src/simd_moments_avx2.cpp', 'tests/test_moments_native.cpp'):
        new['source']['files_sha256'][name] = 'b'
    new['source']['files_sha256']['.github/workflows/moments-same-host.yml'] = 'b'
    new['wheels']['native.whl']['sha256'] = 'b'
    pair_check(old, new)


@pytest.mark.parametrize('bad', ['frontend', 'policy', 'benchmark', 'other-kernel', 'build',
                                 'public-cases', 'missing-common', 'new-product-file', 'wheel', 'runtime'])
def test_other_product_policy_and_runtime_changes_reject(bad):
    old = manifest(); new = deepcopy(old); paths = new['source']['files_sha256']
    names = {'frontend': 'python/skimagexx/_moments.py', 'policy': 'python/skimagexx/_moments_checks.py',
             'benchmark': 'benchmarks/coverage.py', 'other-kernel': 'src/moments.cpp',
             'build': 'CMakeLists.txt', 'public-cases': 'tests/moments_cases.py',
             'new-product-file': 'src/new.cpp'}
    if bad in names: paths[names[bad]] = 'changed'
    elif bad == 'missing-common': del paths['benchmarks/run.py']
    elif bad == 'wheel': new['wheels']['numpy.whl']['sha256'] = 'changed'
    else: new['runtime']['dependency_libraries_sha256']['numpy']['blas.so'] = 'changed'
    with pytest.raises(AssertionError): pair_check(old, new)


@pytest.mark.parametrize('bad', [None, 'missing', 'duplicate', 'worker', 'sample', 'median', 'fallback', 'scalar'])
def test_raw_reports_must_keep_every_case_and_sample(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    rows = [dict(api=api, size=n, dtype=d, threads=1, backend='native',
                 sample_labels=['scikit-image', 'current'], samples_ms=[[1.] * 7, [.5] * 7],
                 loops=[1, 1], reference_ms=1., cpp_ms=.5, speedup=2.)
            for api in ('measure.moments', 'measure.moments_central', 'measure.centroid')
            for n in (256, 1024) for d in ('uint8', 'float32', 'float64')]
    meta = dict(repeat=7, fallback_enabled=False, avx2=True, reference='0.26.0', numpy='1.24.4')
    if bad == 'missing': rows.pop()
    elif bad == 'duplicate': rows[-1] = deepcopy(rows[0])
    elif bad == 'worker': rows[0]['threads'] = 4
    elif bad == 'sample': rows[0]['samples_ms'][0].pop()
    elif bad == 'median': rows[0]['cpp_ms'] = .6
    elif bad == 'fallback': meta['fallback_enabled'] = True
    elif bad == 'scalar': meta['avx2'] = False
    for name in PASSES:
        folder = tmp_path / 'results' / name; folder.mkdir(parents=True)
        (folder / 'moments.json').write_text(json.dumps({'metadata': meta, 'results': rows}))
    if bad:
        with pytest.raises(AssertionError): exec(compile(SCRIPTS[2], 'report-check', 'exec'), {})
    else: exec(compile(SCRIPTS[2], 'report-check', 'exec'), {})
