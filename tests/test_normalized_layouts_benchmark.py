"""Metadata-only behavior checks for the fixed13 external benchmark."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('normalized_layouts', ROOT / 'benchmarks/normalized_layouts.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
WORKFLOW = (ROOT / '.github/workflows/normalized-layouts-same-host.yml').read_text()
SCRIPTS = [textwrap.dedent(s) for s in re.findall(r"python - <<'PY'\n(.*?)          PY\n", WORKFLOW, re.S)]
PASSES = ('1-baseline', '2-candidate', '3-candidate', '4-baseline')


def fixture():
    return dict(schema_version=1, cases=[dict(id=name, shape=[4, 4], dtype='<f4' if i == 1 else '<f8',
                logical_c_words_hex=['00' * (4 if i == 1 else 8)] * 16) for i, name in enumerate(harness.REAL_IDS)])


@pytest.mark.parametrize('bad', [None, 'schema', 'missing', 'duplicate', 'shape', 'dtype', 'words', 'width', 'hex'])
def test_fixture_is_closed_before_array_preparation(tmp_path, bad):
    value = fixture(); row = value['cases'][0]
    if bad == 'schema': value['schema_version'] = 2
    elif bad == 'missing': value['cases'].pop()
    elif bad == 'duplicate': value['cases'][-1] = row
    elif bad == 'shape': row['shape'] = [3, 4]
    elif bad == 'dtype': row['dtype'] = '<i8'
    elif bad == 'words': row['logical_c_words_hex'].pop()
    elif bad == 'width': row['logical_c_words_hex'][0] = '00'
    elif bad == 'hex': row['logical_c_words_hex'][0] = 'zz' * 8
    path = tmp_path / 'cases.json'; path.write_text(json.dumps(value))
    if bad:
        with pytest.raises((AssertionError, ValueError)): harness.saved_cases(path)
    else:
        rows = harness.saved_cases(path); cases = harness.case_specs(rows)
        assert len(cases) == len({r['case_id'] for r in cases}) == 13
        assert [(r['api'], r['dtype'], r['layout']) for r in cases[:5]] == [
            ('measure.moments_hu', 'uint8', 'C'), ('measure.moments_normalized', 'float32', 'C'),
            ('measure.moments_hu', 'float32', 'C'), ('measure.moments_normalized', 'float64', 'C'),
            ('measure.moments_hu', 'float64', 'C')]
        assert [r['case_id'] for r in cases[5:]] == [f'real/{name}/{layout}' for name in harness.REAL_IDS for layout in ('C', 'F')]


def test_harness_is_external_and_measurement_is_artifact_only():
    assert len(SCRIPTS) == 3
    for source in SCRIPTS: compile(source, 'workflow-inline', 'exec')
    assert WORKFLOW.count('actions/checkout@v4') == 3 and 'path: common' in WORKFLOW
    assert "default: '40361265234'" in WORKFLOW and "default: 'aecf5218563b09981bf329b6975e4e43cbf4334a'" in WORKFLOW
    block = WORKFLOW.split('      - name: Measure the fixed13', 1)[1].split('      - name:', 1)[0]
    command = '"$GITHUB_WORKSPACE/common/benchmarks/normalized_layouts.py" --product-root "$PWD" --threads 1 --repeat 7'
    assert command in block and 'for pass in ' + ' '.join(PASSES) + '; do' in block
    assert block.count('ci_measurement_artifacts.py verify') == 2
    assert block.index('verify') < block.index(command) < block.rindex('verify')
    assert block.index('benchmark-exit.txt') < block.index('identity-after.log')
    assert not re.search(r'pytest|cmake|ctest|pip install|pip wheel', block)
    assert WORKFLOW.index('--no-index --no-deps --only-binary=:all: --require-hashes') < WORKFLOW.index(command)


def pair_gate(old, new):
    branch = next(n for n in ast.parse(SCRIPTS[1]).body if isinstance(n, ast.If) and ast.unparse(n.test) == "role == 'candidate'")
    start = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.For) and ast.unparse(n.target) == 'key')
    stop = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.FunctionDef) and n.name == 'module')
    exec(compile(ast.Module(branch.body[start:stop], type_ignores=[]), 'pair-source-gate', 'exec'), {'old': old, 'new': new, 'hashlib': hashlib,
         'Path': lambda name: SimpleNamespace(read_bytes=lambda: b'fixture\r\n')})


def manifest():
    common = ['benchmarks/coverage.py', 'benchmarks/run.py', 'benchmarks/provenance.py', 'tools/ci_measurement_artifacts.py',
              'tools/run_feature_tests.py', 'tests/moments_cases.py', 'tests/test_moments_public.py',
              'tests/moments_derived_cases.py', 'tests/test_moments_derived_public.py', 'src/bind_moments_derived.cpp',
              'python/skimagexx/_moments_derived.py', 'CMakeLists.txt', 'src/moments_normalized.cpp',
              'python/skimagexx/_moments_derived_checks.py', 'python/skimagexx/coverage.py']
    runtime = dict.fromkeys(('python', 'implementation', 'cache_tag', 'machine', 'system'), 'same')
    runtime.update(versions={'numpy': '1.24.4'}, packages_sha256={n: {'x': 'same'} for n in ('numpy', 'scipy', 'skimage')},
                   dependency_libraries_sha256={'blas': 'same'})
    wheels = {f'dep{i}.whl': {'name': f'dep{i}', 'sha256': 'same'} for i in range(13)}
    wheels['product.whl'] = {'name': 'scikit-imagexx', 'sha256': 'old'}
    return dict(runtime=runtime, wheels=wheels, source={'files_sha256': dict.fromkeys(common, 'same')})


def candidate(old):
    value = deepcopy(old)
    value['source']['files_sha256'].update({
        'benchmarks/normalized_layouts_cases.json': hashlib.sha256(b'fixture\n').hexdigest(),
        'tests/test_normalized_layouts_public.py': 'df1786d5663dbd824f09dc8bdb0f9c15d335eb905459bcd9e6109317c8389c47'})
    return value


def test_only_frontend_and_binder_production_may_change():
    old = manifest(); new = candidate(old)
    for path in ('src/bind_moments_derived.cpp', 'python/skimagexx/_moments_derived.py',
                 '.github/workflows/ci.yml'):
        new['source']['files_sha256'][path] = 'new'
    new['wheels']['product.whl']['sha256'] = 'new'
    pair_gate(old, new)


@pytest.mark.parametrize('path', ['CMakeLists.txt', 'src/moments_normalized.cpp', 'src/new.cpp', 'benchmarks/run.py',
                                  'benchmarks/coverage.py', 'python/skimagexx/_moments_derived_checks.py',
                                  'python/skimagexx/coverage.py', 'tests/moments_derived_cases.py',
                                  'benchmarks/normalized_layouts_cases.json', 'tests/test_normalized_layouts_public.py'])
def test_other_source_changes_reject(path):
    old = manifest(); new = candidate(old); new['source']['files_sha256'][path] = 'changed'
    with pytest.raises(AssertionError): pair_gate(old, new)


@pytest.mark.parametrize('bad', [None, 'missing', 'duplicate', 'shape', 'layout', 'sample', 'median', 'product', 'common', 'mutation', 'worker', 'pool', 'output'])
def test_all_four_reports_preserve_cases_inputs_and_identities(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    common = tmp_path / 'common'; common.mkdir()
    (common / 'fixture.json').write_bytes(b'fixed input')
    hashes = {'fixture.json': hashlib.sha256(b'fixed input').hexdigest()}
    cases = harness.case_specs(fixture()['cases'])
    shared = dict(cases=cases, files_sha256=hashes, product_fingerprints={'baseline': {'source_sha256': 'base'}, 'candidate': {'source_sha256': 'candidate'}})
    (tmp_path / 'results').mkdir(); (tmp_path / 'results/common-inputs.json').write_text(json.dumps(shared))
    for index, name in enumerate(PASSES):
        role = name.split('-')[1]
        rows = [dict(c, threads=1, size=4, backend='native-partial', input_strides=[16, 4] if c['layout'] == 'C' else [4, 16],
                     input_c_sha256=c['case_id'], max_abs_error=0., output_dtype='<f8', output_strides=[32, 8], output_flags={'c_contiguous': True}, sample_labels=['scikit-image', 'current'],
                     samples_ms=[[1.] * 7, [.5] * 7], loops=[1, 1], reference_ms=1., cpp_ms=.5, speedup=2.) for c in cases]
        meta = dict(repeat=7, fallback_enabled=False, avx2=True, reference='0.26.0', numpy='1.24.4',
                    workers_before=1, workers_after=1, reference_pool_caps=dict.fromkeys(harness.POOL_VARIABLES, '1'),
                    product_root=str((tmp_path / 'snapshots' / role).resolve()), common_inputs_sha256=hashes,
                    source_sha256=shared['product_fingerprints'][role]['source_sha256'])
        if index == 2:
            if bad == 'missing': rows.pop()
            elif bad == 'duplicate': rows[-1] = deepcopy(rows[0])
            elif bad == 'shape': rows[0]['input_shape'] = [256, 256]
            elif bad == 'layout': rows[-1]['layout'] = 'C'
            elif bad == 'sample': rows[0]['samples_ms'][0].pop()
            elif bad == 'median': rows[0]['cpp_ms'] = .6
            elif bad == 'product': meta['source_sha256'] = 'common-root'
            elif bad == 'common': meta['common_inputs_sha256'] = {}
            elif bad == 'mutation': rows[0]['input_c_sha256'] = 'different bytes'
            elif bad == 'worker': meta['workers_after'] = 4
            elif bad == 'pool': meta['reference_pool_caps'] = {}
            elif bad == 'output': rows[-1]['output_strides'] = [8, 32]
        folder = tmp_path / 'results' / name; folder.mkdir()
        (folder / 'normalized-layouts.json').write_text(json.dumps(dict(metadata=meta, results=rows)))
    if bad:
        with pytest.raises(AssertionError): exec(compile(SCRIPTS[2], 'report-gate', 'exec'), {})
    else: exec(compile(SCRIPTS[2], 'report-gate', 'exec'), {})


@pytest.mark.parametrize('field', [None, 'dtype', 'strides', 'c_contiguous', 'f_contiguous', 'aligned', 'owndata', 'writeable'])
def test_output_layout_check_preserves_the_original_contract(field):
    expected = SimpleNamespace(dtype=SimpleNamespace(str='<f8'), strides=(32, 8),
                               flags=SimpleNamespace(c_contiguous=True, f_contiguous=False,
                                                     aligned=True, owndata=True, writeable=True))
    actual = deepcopy(expected)
    if field == 'dtype': actual.dtype.str = '<f4'
    elif field == 'strides': actual.strides = (8, 32)
    elif field: setattr(actual.flags, field, not getattr(actual.flags, field))
    if field:
        with pytest.raises(AssertionError): harness.check_output_layout(actual, expected)
    else:
        result = harness.check_output_layout(actual, expected)
        assert result['output_dtype'] == '<f8' and result['output_strides'] == [32, 8]
        assert result['output_flags'] == vars(expected.flags)


@pytest.mark.parametrize('path', ['benchmarks/normalized_layouts_cases.json', 'tests/test_normalized_layouts_public.py'])
def test_missing_candidate_cohort_source_rejects(path):
    old = manifest(); new = candidate(old)
    assert path not in old['source']['files_sha256']  # Baseline intentionally predates the cohort.
    del new['source']['files_sha256'][path]
    with pytest.raises(KeyError): pair_gate(old, new)
