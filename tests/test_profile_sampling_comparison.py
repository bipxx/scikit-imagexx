"""Metadata-only guards for the fixed18 Profile artifact comparison; no timing calls."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import re
import textwrap
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / '.github/workflows/profile-sampling-same-host.yml').read_text()
SCRIPTS = [textwrap.dedent(s) for s in re.findall(
    r"python - <<'PY'\n(.*?)          PY\n", WORKFLOW, re.S)]
PASSES = ('1-baseline', '2-candidate', '3-candidate', '4-baseline')
COMMON = ('.github/workflows/ci.yml', 'benchmarks/coverage.py', 'benchmarks/rgb2yuv_storage.py', 'benchmarks/run.py',
          'benchmarks/provenance.py', 'tools/ci_measurement_artifacts.py',
          'tools/run_feature_tests.py', 'tests/moments_cases.py', 'tests/test_moments_public.py',
          'tests/inertia_tensor_cases.py', 'tests/test_inertia_tensor_public.py',
          'tests/inertia_eigvals_cases.py', 'tests/test_inertia_eigvals_public.py',
          'tests/moments_derived_cases.py', 'tests/test_moments_derived_public.py',
          'tests/window_cases.py', 'tests/test_window_public.py',
          'tests/lpi_filter_cases.py', 'tests/test_lpi_filter_public.py',
          'tests/butterworth_cases.py', 'tests/test_butterworth_public.py',
          'benchmarks/normalized_layouts_cases.json', 'tests/test_normalized_layouts_public.py',
          'tests/test_immutable_buffer_roots_public.py',
          'tests/moments_coords_cases.py', 'tests/test_moments_coords_public.py',
          'tests/shannon_entropy_cases.py', 'tests/test_shannon_entropy_public.py',
          'tests/rgb2yuv_storage_cases.py', 'tests/test_rgb2yuv_storage_public.py', 'tests/test_color_transforms.py',
          'tests/block_reduce_cases.py', 'tests/test_block_reduce_public.py',
          'tests/perimeter_crofton_cases.py', 'tests/test_perimeter_crofton_public.py',
          'benchmarks/profile_line.py', 'tests/profile_line_cases.py', 'tests/test_profile_line_public.py')


def test_fresh_host_fixed_command_and_retention():
    assert len(SCRIPTS) == 4
    for source in SCRIPTS:
        compile(source, 'workflow-inline', 'exec')
    assert "default: '40448470579'" in WORKFLOW
    assert "default: '9ead9b326c8880958b60341921d5ef4091e7b57e'" in WORKFLOW
    assert WORKFLOW.count('actions/checkout@v4') == 2
    block = WORKFLOW.split('      - name: Measure the fixed18', 1)[1].split('      - name:', 1)[0]
    command = 'benchmarks/profile_line.py --threads 1 --repeat 7'
    assert command in block and 'for pass in ' + ' '.join(PASSES) + '; do' in block
    assert block.count('ci_measurement_artifacts.py verify') == 2
    assert block.index('verify') < block.index(command) < block.rindex('verify')
    assert block.index('benchmark-exit.txt') < block.index('identity-after.log')
    assert 'unset PYTHONPATH PYTHONHOME SKIMAGEXX_DISABLE_SIMD' in block
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel', block)
    assert WORKFLOW.index('--no-index --no-deps --only-binary=:all: --require-hashes') < WORKFLOW.index(command)
    assert '"$RUNNER_TEMP/$role-prepared.zip"' in WORKFLOW and 'path: results/' in WORKFLOW
    assert 'if: always()' in WORKFLOW and 'name: profile-sampling-same-host-minimum' in WORKFLOW


@pytest.mark.parametrize('bad', [None, 'revision', 'repository', 'workflow', 'unfinished',
                                 'duplicate', 'artifact-run', 'digest'])
def test_api_identity_rejects_substitution(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path); (tmp_path / 'results').mkdir()
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    for role, number, revision in [('BASELINE', '40448470579', '9ead9b326c8880958b60341921d5ef4091e7b57e'), ('CANDIDATE', '22', '99fdb2abc669828cad1a14499be496200b7e4e8a')]:
        monkeypatch.setenv(role + '_RUN', number); monkeypatch.setenv(role + '_REVISION', revision)
    def response(argv):
        endpoint = argv[-1]; number = 40448470579 if '/runs/40448470579' in endpoint else 22
        revision = '9ead9b326c8880958b60341921d5ef4091e7b57e' if number == 40448470579 else '99fdb2abc669828cad1a14499be496200b7e4e8a'
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


def manifest(candidate=False):
    runtime = {k: 'same' for k in ('python', 'implementation', 'cache_tag', 'machine', 'system')}
    runtime.update(dependency_libraries_sha256={'numpy': {'blas.so': 'a'}},
                   versions={'numpy': '1.24.4', 'scikit-imagexx': '0.1.0', 'pytest': '9.2.0' if candidate else '9.1.1'},
                   packages_sha256={k: {'source.py': 'a'} for k in ('numpy', 'scipy', 'skimage')})
    paths = (*COMMON, 'CMakeLists.txt', 'pyproject.toml', 'src/block_reduce.cpp',
             'src/block_reduce_internal.hpp', 'src/simd_block_reduce_avx2.cpp', 'src/resample.cpp', 'src/resample_detail.hpp', 'src/simd_resample_avx2.cpp', 'src/convert.cpp',
             'python/skimagexx/_block_reduce.py', 'python/skimagexx/_block_reduce_checks.py',
             'src/bind_block_reduce.cpp', 'tests/test_block_reduce_native.cpp')
    wheels = {f'dep{i}.whl': {'name': f'dep{i}', 'version': '1', 'sha256': 'a'} for i in range(12)}
    version, digest = ('9.2.0', '07acdcf88ec174f907dc01af70bf52d49b014bbab03faeeade31c96b18c2e588') if candidate else ('9.1.1', '37a86b45efb9a47a61a36449063e8e18d0cab3161329fc099eb21783169c4f0c')
    wheels['pytest-' + version + '-py3-none-any.whl'] = dict(name='pytest', version=version, sha256=digest)
    wheels['native.whl'] = {'name': 'scikit-imagexx', 'version': '0.1.0', 'sha256': 'a'}
    return dict(runtime=runtime, wheels=wheels, source={'files_sha256': dict.fromkeys(paths, 'a')})


def test_only_declared_simd_leaf_and_product_wheel_may_change():
    old = manifest(); new = manifest(candidate=True)
    new['source']['files_sha256']['src/resample.cpp'] = 'b'
    new['wheels']['native.whl']['sha256'] = 'b'
    pair_check(old, new)


@pytest.mark.parametrize('bad', ['frontend', 'policy', 'benchmark', 'other-kernel', 'build',
                                 'public-cases', 'public-test', 'binder', 'ci', 'missing-mutable', 'missing-common', 'new-product-file', 'wheel', 'runtime', 'pytest-version', 'pytest-hash'])
def test_other_product_policy_and_runtime_changes_reject(bad):
    old = manifest(); new = manifest(candidate=True); paths = new['source']['files_sha256']
    names = {'frontend': 'python/skimagexx/_block_reduce.py', 'policy': 'python/skimagexx/_block_reduce_checks.py',
             'benchmark': 'benchmarks/coverage.py', 'other-kernel': 'src/convert.cpp',
             'build': 'CMakeLists.txt', 'public-cases': 'tests/block_reduce_cases.py',
             'public-test': 'tests/test_block_reduce_public.py', 'binder': 'src/bind_block_reduce.cpp',
             'ci': '.github/workflows/ci.yml', 'new-product-file': 'src/new.cpp'}
    if bad == 'pytest-version': new['runtime']['versions']['pytest'] = '9.3.0'
    if bad == 'pytest-hash': new['wheels']['pytest-9.2.0-py3-none-any.whl']['sha256'] = 'changed'
    if bad in names: paths[names[bad]] = 'changed'
    elif bad == 'missing-mutable': del paths['src/resample.cpp']
    elif bad == 'missing-common': del paths['benchmarks/run.py']
    elif bad == 'wheel': new['wheels']['dep0.whl']['sha256'] = 'changed'
    elif bad == 'runtime': new['runtime']['dependency_libraries_sha256']['numpy']['blas.so'] = 'changed'
    with pytest.raises(AssertionError): pair_check(old, new)


@pytest.mark.parametrize('bad', [None, 'missing', 'duplicate', 'worker', 'sample', 'median', 'fallback',
    'scalar', 'nonfinite-error', 'fingerprint', 'not-accepted', 'sample-pixels', 'output', 'harness',
    'target-not-repeat', 'aggregate', 'original', 'expired'])
def test_raw_reports_must_keep_every_case_sample_and_exact_native_result(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    import datetime
    real_datetime = datetime.datetime
    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 10, 28, 9 if bad == 'expired' else 7, tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(datetime, 'datetime', FixedDatetime)
    decision_loss = bad in ('target-not-repeat', 'aggregate', 'original', 'expired')
    specs = []
    for side in (256, 1024):
        for dtype in ('uint8', 'float32', 'float64'):
            for width in (1, 9, 33):
                length = 360 if side == 256 else 1446
                specs.append(dict(case_id=f'side{side}:{dtype}:width{width}:length{length}',
                    api='measure.profile_line', shape=[side, side], src=[.25, 1.5], dst=[side-1.25, side-1.75],
                    dtype=dtype, linewidth=width, profile_length=length, sampled_values=length*width,
                    threads=1, backend='native', native_accepted=True, actual_C_returns=['profile_sample'],
                    max_abs_error=0., sample_labels=['scikit-image', 'current'], loops=[1, 1],
                    input_sha256='a', pre_reducer_metadata={'shape': [length, width]}, sample_sha256='a',
                    output_dtype_str='<f8', output_shape=[length], output_strides=[8],
                    output_raw_hex='00', reference_raw_hex='00'))
    fingerprint = dict(source_sha256='b'*64, source_hash_schema='cpp-python-build-posix-lf-v2', source_hash_scope='fixed')
    monkeypatch.setattr('runpy.run_path', lambda path: {'source_fingerprint': lambda source: fingerprint})
    for name in PASSES:
        candidate = 'candidate' in name
        rows = deepcopy(specs)
        cpp = .25 if candidate else .5
        if bad == 'aggregate' and candidate: cpp = .6
        if bad == 'original': cpp = 2. if candidate else 3.
        for row in rows:
            row.update(samples_ms=[[1.] * 7, [cpp] * 7], reference_ms=1., cpp_ms=cpp, speedup=1/cpp)
        if bad == 'target-not-repeat' and name == '2-candidate':
            row = next(r for r in rows if r['linewidth'] == 33)
            row.update(samples_ms=[[1.] * 7, [.6] * 7], reference_ms=1., cpp_ms=.6, speedup=1/.6)
        meta = dict(**fingerprint, source_before=fingerprint, repeat=7, threads=1, raw_samples=252,
            fallback_enabled=False, avx2=True, reference='0.26.0', numpy='1.24.4',
            selected_apis=['measure.profile_line'], reference_pool_limits={'OPENBLAS_NUM_THREADS': '1'},
            harness_sha256='9618c9efe2c716bcd0897084c88daefaf9c707243fa2ca6704c714515b45511c')
        if name == '2-candidate':
            if bad == 'missing': rows.pop()
            elif bad == 'duplicate': rows[-1] = deepcopy(rows[0])
            elif bad == 'worker': rows[0]['threads'] = 4
            elif bad == 'sample': rows[0]['samples_ms'][0].pop()
            elif bad == 'median': rows[0]['cpp_ms'] = .6
            elif bad == 'fallback': meta['fallback_enabled'] = True
            elif bad == 'scalar': meta['avx2'] = False
            elif bad == 'nonfinite-error': rows[0]['max_abs_error'] = float('nan')
            elif bad == 'fingerprint': meta['source_sha256'] = 'c'*64
            elif bad == 'not-accepted': rows[0]['native_accepted'] = False
            elif bad == 'sample-pixels': rows[0]['sample_sha256'] = 'b'
            elif bad == 'output': rows[0]['output_raw_hex'] = '01'
            elif bad == 'harness': meta['harness_sha256'] = 'c'*64
        folder = tmp_path / 'results' / name; folder.mkdir(parents=True)
        (folder / 'profile-line.json').write_text(json.dumps({'metadata': meta, 'results': rows}))
    if bad and not decision_loss:
        with pytest.raises(AssertionError): exec(compile(SCRIPTS[3], 'report-check', 'exec'), {})
    else:
        exec(compile(SCRIPTS[3], 'report-check', 'exec'), {})
        summary = json.loads((tmp_path / 'results/comparison.json').read_text())
        assert summary['status'] == ('DISCARD' if decision_loss else 'KEEP')
        if decision_loss:
            if bad == 'target-not-repeat':
                assert summary['direct_gain'] > 1 and summary['normalized_gain'] > 1
                assert summary['target_repeat_wins'] == 5
            elif bad == 'aggregate': assert summary['direct_gain'] < 1
            elif bad == 'original': assert summary['candidate_original'] < 1
            else: assert summary['completed_before_deadline'] is False
            return
        assert summary['direct_gain'] == 2 and summary['normalized_gain'] == 2
        assert summary['candidate_original'] == 4 and summary['target_repeat_wins'] == 6
        assert summary['direct_losses'] == summary['normalized_losses'] == []
        assert summary['raw_samples'] == 1008 and len(summary['rows']) == 18
        assert summary['groups']['controls_width1_9']['cases'] == 12
        assert summary['groups']['uint8']['cases'] == 6


@pytest.mark.parametrize('bad', [None, 'missing-family', 'short-family', 'failure', 'skip', 'smoke', 'registry', 'wheel-count', 'color-count'])
def test_preparation_only_product_requires_all_fourteen_cohorts_and_color(tmp_path, monkeypatch, bad):
    # No standalone benchmark JSON is created: prepare_only products are valid inputs.
    for family, count in (('inertia-tensor', 115), ('inertia-eigvals', 69), ('moments', 203), ('window', 179),
                          ('lpi', 161), ('butterworth', 175), ('moments-derived', 155),
                          ('normalized-layouts', 22), ('immutable-bytes', 134),
                          ('moments-coords', 98), ('shannon-entropy', 61), ('rgb2yuv-storage', 37),
                          ('block-reduce', 145), ('perimeter-crofton', 101), ('perimeter', 111), ('euler-number', 91),
                          ('polygon', 86), ('mesh-surface-area', 89), ('profile-line', 93), ('color-transforms-regression', 1848)):
        for mode in ('normal', 'scalar'):
            root = ET.Element('testsuite')
            for index in range(count): ET.SubElement(root, 'testcase', name=str(index))
            if family == 'perimeter-crofton' and mode == 'scalar':
                if bad == 'missing-family': continue
                if bad == 'short-family': root.remove(root[-1])
                if bad in ('failure', 'skip'): ET.SubElement(root[0], 'failure' if bad == 'failure' else 'skipped')
            if bad == 'color-count' and family == 'color-transforms-regression': root.remove(root[-1])
            (tmp_path / (family + '-' + mode + '.xml')).write_bytes(ET.tostring(root))
    for family, count in (('polygon-intake', 4), ('euler-number-intake', 4), ('block-reduce-integer-width', 9)):
        root = ET.Element('testsuite')
        for index in range(count): ET.SubElement(root, 'testcase', name=str(index))
        (tmp_path / (family + '.xml')).write_bytes(ET.tostring(root))
    smoke = dict(status='passed', exit_code=0, outcomes={'passed': 1409}, results=[{}] * 1409,
                 implementations={'native-partial': 234, 'metadata': 3})
    if bad == 'smoke': smoke['results'].pop()
    if bad == 'registry': smoke['implementations']['native-partial'] = 226
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


@pytest.mark.parametrize('corrupt', [False, True])
def test_normalization_copies_only_pytest_and_derived_lock(tmp_path, monkeypatch, corrupt):
    import hashlib
    import shutil
    monkeypatch.chdir(tmp_path); (tmp_path / 'results').mkdir()
    originals = {}
    for role in ('baseline', 'candidate'):
        prepared = tmp_path / 'snapshots' / role / 'build/prepared'
        (prepared / 'wheels').mkdir(parents=True); (prepared / 'ci-artifacts').mkdir()
        value = manifest(candidate=role == 'candidate')
        for name, item in value['wheels'].items():
            data = name.encode(); (prepared / 'wheels' / name).write_bytes(data)
            item['sha256'] = hashlib.sha256(data).hexdigest()
        value['source']['revision'] = role
        value['lock_sha256'] = 'original-lock'
        raw = json.dumps(value).encode(); (prepared / 'ci-artifacts/measurement.json').write_bytes(raw)
        originals[role] = raw
    baseline = tmp_path / 'snapshots/baseline/build/prepared'
    old = json.loads(originals['baseline']); filename, record = next((n, v) for n, v in old['wheels'].items() if v['name'] == 'pytest')
    if corrupt: (tmp_path / 'snapshots/candidate/build/prepared/wheels/dep0.whl').write_bytes(b'corrupt')
    body = ast.parse(SCRIPTS[2]).body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'normalized')
    code = compile(ast.Module(body[start:], type_ignores=[]), 'normalize-pytest', 'exec')
    namespace = dict(baseline=baseline, filename=filename, record=record, deepcopy=deepcopy, hashlib=hashlib, json=json, shutil=shutil, Path=Path)
    if corrupt:
        with pytest.raises(AssertionError): exec(code, namespace)
        return
    exec(code, namespace)
    for role in ('baseline', 'candidate'):
        prepared = tmp_path / 'snapshots' / role / 'build/prepared'; target = prepared.parent / 'measurement-v2'
        before = json.loads(originals[role]); after = json.loads((target / 'measurement.json').read_bytes())
        assert (prepared / 'ci-artifacts/measurement.json').read_bytes() == originals[role]
        assert after['source'] == before['source']
        before['wheels'] = {n: v for n, v in before['wheels'].items() if v['name'] != 'pytest'}
        before['wheels'][filename] = record; before['runtime']['versions']['pytest'] = '9.1.1'
        before['lock_sha256'] = hashlib.sha256((target / 'runtime-lock.txt').read_bytes()).hexdigest()
        assert after == before
        assert {p.name for p in (target / 'wheels').iterdir()} == set(after['wheels'])
        assert not (target / 'wheels/pytest-9.2.0-py3-none-any.whl').exists()
        assert (target / 'wheels' / filename).read_bytes() == filename.encode()
        assert (tmp_path / 'results' / (role + '-normalized-measurement.json')).read_bytes() == (target / 'measurement.json').read_bytes()
    block = WORKFLOW.split('      - name: Normalize only', 1)[1].split('      - name: Measure', 1)[0]
    assert '--no-index --no-deps --only-binary=:all: --require-hashes' in block
    assert block.index('pip install') < block.index('ci_measurement_artifacts.py')
    timed = WORKFLOW.split('      - name: Measure', 1)[1].split('      - name:', 1)[0]
    assert timed.count('--artifact build/measurement-v2/measurement.json') == 2
