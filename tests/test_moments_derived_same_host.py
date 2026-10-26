"""Metadata-only guards for the derived-moments artifact comparison; no timing calls."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import textwrap
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / '.github/workflows/moments-derived-same-host.yml').read_text()
SCRIPTS = [textwrap.dedent(s) for s in re.findall(
    r"python - <<'PY'\n(.*?)          PY\n", WORKFLOW, re.S)]
PASSES = ('1-baseline', '2-candidate', '3-candidate', '4-baseline')
COMMON = ('.github/workflows/ci.yml', 'benchmarks/coverage.py', 'benchmarks/run.py',
          'benchmarks/provenance.py', 'tools/ci_measurement_artifacts.py',
          'tools/run_feature_tests.py', 'tests/moments_cases.py', 'tests/test_moments_public.py',
          'tests/moments_derived_cases.py', 'tests/test_moments_derived_public.py')
BENCHMARK = "def main():\n    derived_scope = 'original provider description'\n    return measure(prepare(4), repeat=7)\n"


def test_fresh_host_fixed_command_and_retention():
    assert len(SCRIPTS) == 3
    for source in SCRIPTS:
        compile(source, 'workflow-inline', 'exec')
    assert "default: '40350415600'" in WORKFLOW
    assert "default: '76cdd70ac85c889a68985c25964a1e0646fa7579'" in WORKFLOW
    assert WORKFLOW.count('actions/checkout@v4') == 2
    block = WORKFLOW.split('      - name: Measure the fixed5', 1)[1].split('      - name:', 1)[0]
    command = ('benchmarks/coverage.py --api measure.moments_normalized --api measure.moments_hu '
               '--sizes 256 1024 --threads 1 --repeat 7')
    assert command in block and 'for pass in ' + ' '.join(PASSES) + '; do' in block
    assert block.count('ci_measurement_artifacts.py verify') == 2
    assert block.index('verify') < block.index(command) < block.rindex('verify')
    assert block.index('benchmark-exit.txt') < block.index('identity-after.log')
    assert 'unset PYTHONPATH PYTHONHOME SKIMAGEXX_DISABLE_SIMD' in block
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel', block)
    assert WORKFLOW.index('--no-index --no-deps --only-binary=:all: --require-hashes') < WORKFLOW.index(command)
    assert '"$RUNNER_TEMP/$role-prepared.zip"' in WORKFLOW and 'path: results/' in WORKFLOW
    assert 'if: always()' in WORKFLOW and 'name: moments-derived-same-host-minimum' in WORKFLOW


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


def pair_check(old, new, tmp_path, sources=(BENCHMARK, BENCHMARK), stale_pin=False):
    for role, value, source in zip(('baseline', 'candidate'), (old, new), sources):
        path = tmp_path / 'snapshots' / role / 'benchmarks/coverage.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8', newline='\n')
        if not stale_pin:
            value['source']['files_sha256']['benchmarks/coverage.py'] = hashlib.sha256(source.encode()).hexdigest()
    branch = next(n for n in ast.parse(SCRIPTS[1]).body
                  if isinstance(n, ast.If) and ast.unparse(n.test) == "role == 'candidate'")
    # Supply metadata directly instead of opening ZIPs or installed packages.
    exec(compile(ast.Module(branch.body[1:], type_ignores=[]), 'pair-equality', 'exec'),
         {'old': old, 'new': new, 'ast': ast, 'hashlib': hashlib, 'Path': lambda *p: tmp_path.joinpath(*p)})


def manifest():
    runtime = {k: 'same' for k in ('python', 'implementation', 'cache_tag', 'machine', 'system')}
    runtime.update(dependency_libraries_sha256={'numpy': {'blas.so': 'a'}},
                   versions={'numpy': '1.24.4', 'scikit-imagexx': '0.1.0'},
                   packages_sha256={k: {'source.py': 'a'} for k in ('numpy', 'scipy', 'skimage')})
    paths = (*COMMON, 'CMakeLists.txt', 'pyproject.toml', 'src/moments.cpp',
             'src/simd_moments_avx2.cpp', 'include/skimagexx/moments.hpp',
             'python/skimagexx/_moments.py', 'python/skimagexx/_moments_checks.py',
             'python/skimagexx/_moments_derived.py', 'python/skimagexx/_moments_derived_checks.py',
             'src/bind_moments_derived.cpp',
             'tests/test_moments_native.cpp')
    wheels = {f'dep{i}.whl': {'name': f'dep{i}', 'sha256': 'a'} for i in range(13)}
    wheels['native.whl'] = {'name': 'scikit-imagexx', 'sha256': 'a'}
    return dict(runtime=runtime, wheels=wheels, source={'files_sha256': dict.fromkeys(paths, 'a')})



def candidate(old):
    new = deepcopy(old)
    for name in ('CMakeLists.txt', 'src/moments_normalized.cpp', 'src/moments_normalized.hpp',
                 'src/bind_moments_derived.cpp', 'src/simd_moments_avx2.cpp',
                 'python/skimagexx/_moments_derived.py', 'tests/test_moments_normalized_native.cpp'):
        new['source']['files_sha256'][name] = 'b'
    return new


def test_declared_batch_surface_and_provider_literal_are_allowed(tmp_path):
    old = manifest(); new = candidate(old)
    new['source']['files_sha256']['.github/workflows/moments-derived-same-host.yml'] = 'b'
    new['wheels']['native.whl']['sha256'] = 'b'
    pair_check(old, new, tmp_path, (BENCHMARK, BENCHMARK.replace('original provider', 'batch provider')))


@pytest.mark.parametrize('bad', ['frontend', 'policy', 'derived-policy', 'other-kernel', 'public-cases',
                                 'derived-cases', 'derived-public', 'missing-common', 'missing-batch',
                                 'new-product-file', 'wheel', 'wheel-count', 'runtime'])
def test_other_product_policy_and_runtime_changes_reject(tmp_path, bad):
    old = manifest(); new = candidate(old); paths = new['source']['files_sha256']
    names = {'frontend': 'python/skimagexx/_moments.py', 'policy': 'python/skimagexx/_moments_checks.py',
             'derived-policy': 'python/skimagexx/_moments_derived_checks.py', 'other-kernel': 'src/moments.cpp',
             'public-cases': 'tests/moments_cases.py', 'derived-cases': 'tests/moments_derived_cases.py',
             'derived-public': 'tests/test_moments_derived_public.py', 'new-product-file': 'src/new.cpp'}
    if bad in names: paths[names[bad]] = 'changed'
    elif bad == 'missing-common': del paths['benchmarks/run.py']
    elif bad == 'missing-batch': del paths['src/moments_normalized.cpp']
    elif bad == 'wheel': new['wheels']['dep0.whl']['sha256'] = 'changed'
    elif bad == 'wheel-count':
        del old['wheels']['dep0.whl']; del new['wheels']['dep0.whl']
    else: new['runtime']['dependency_libraries_sha256']['numpy']['blas.so'] = 'changed'
    with pytest.raises(AssertionError): pair_check(old, new, tmp_path)


@pytest.mark.parametrize('source', [
    BENCHMARK.replace('prepare(4)', 'prepare(5)'),
    BENCHMARK.replace('repeat=7', 'repeat=5'),
    BENCHMARK.replace('measure(', 'other_timer('),
    BENCHMARK.replace("'original provider description'", 'description()'),
    BENCHMARK.replace('    return', "    derived_scope = 'second'\n    return"),
    BENCHMARK.replace('    return', "    derived_scope += 'changed'\n    return"),
    BENCHMARK.replace('derived_scope =', 'derived_scope = another ='),
    BENCHMARK.replace('derived_scope =', 'other_scope ='),
])
def test_only_one_literal_description_can_be_normalized(tmp_path, source):
    old = manifest(); new = candidate(old)
    with pytest.raises(AssertionError): pair_check(old, new, tmp_path, (BENCHMARK, source))


def test_ast_source_must_match_the_producer_manifest(tmp_path):
    old = manifest(); new = candidate(old)
    with pytest.raises(AssertionError): pair_check(old, new, tmp_path, stale_pin=True)


@pytest.mark.parametrize('bad', [None, 'missing-family', 'short-family', 'failure', 'skip', 'smoke'])
def test_all_six_normal_scalar_cohorts_and_smoke_are_required(tmp_path, bad):
    for family, count in (('moments-derived', 155), ('moments', 203), ('window', 179),
                          ('lpi', 161), ('butterworth', 175), ('immutable-bytes', 134)):
        for mode in ('normal', 'scalar'):
            root = ET.Element('testsuite')
            for index in range(count): ET.SubElement(root, 'testcase', name=str(index))
            if family == 'moments-derived' and mode == 'scalar':
                if bad == 'missing-family': continue
                if bad == 'short-family': root.remove(root[-1])
                if bad in ('failure', 'skip'): ET.SubElement(root[0], 'failure' if bad == 'failure' else 'skipped')
            (tmp_path / (family + '-' + mode + '.xml')).write_bytes(ET.tostring(root))
    smoke = dict(status='passed', exit_code=0, outcomes={'passed': 1357}, results=[{}] * 1357)
    if bad == 'smoke': smoke['results'].pop()
    (tmp_path / 'feature-smoke.json').write_text(json.dumps(smoke))
    body = ast.parse(SCRIPTS[1]).body
    first = next(i for i, n in enumerate(body) if isinstance(n, ast.For)
                 and ast.unparse(n.target) == '(family, count)')
    last = next(i for i, n in enumerate(body) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "role == 'candidate'")
    code = compile(ast.Module(body[first:last], type_ignores=[]), 'producer-counts', 'exec')
    if bad:
        with pytest.raises((AssertionError, FileNotFoundError)): exec(code, {'prepared': tmp_path, 'ET': ET, 'json': json})
    else: exec(code, {'prepared': tmp_path, 'ET': ET, 'json': json})


@pytest.mark.parametrize('bad', [None, 'missing', 'duplicate', 'worker', 'sample', 'median', 'fallback', 'scalar', 'shape', 'layout'])
def test_raw_reports_must_keep_every_case_and_sample(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    rows = [dict(api=api, size=4, dtype=d, threads=1, backend='native', input_shape=[4, 4], layout='standard',
                 sample_labels=['scikit-image', 'current'], samples_ms=[[1.] * 7, [.5] * 7],
                 loops=[1, 1], reference_ms=1., cpp_ms=.5, speedup=2.)
            for api in ('measure.moments_normalized', 'measure.moments_hu')
            for d in ('uint8', 'float32', 'float64')
            if (api, d) != ('measure.moments_normalized', 'uint8')]
    meta = dict(repeat=7, fallback_enabled=False, avx2=True, reference='0.26.0', numpy='1.24.4')
    if bad == 'missing': rows.pop()
    elif bad == 'duplicate': rows[-1] = deepcopy(rows[0])
    elif bad == 'worker': rows[0]['threads'] = 4
    elif bad == 'sample': rows[0]['samples_ms'][0].pop()
    elif bad == 'median': rows[0]['cpp_ms'] = .6
    elif bad == 'fallback': meta['fallback_enabled'] = True
    elif bad == 'scalar': meta['avx2'] = False
    elif bad == 'shape': rows[0]['input_shape'] = [256, 256]
    elif bad == 'layout': rows[0]['layout'] = 'stepped'
    for name in PASSES:
        folder = tmp_path / 'results' / name; folder.mkdir(parents=True)
        (folder / 'moments-derived.json').write_text(json.dumps({'metadata': meta, 'results': rows}))
    if bad:
        with pytest.raises(AssertionError): exec(compile(SCRIPTS[2], 'report-check', 'exec'), {})
    else: exec(compile(SCRIPTS[2], 'report-check', 'exec'), {})
