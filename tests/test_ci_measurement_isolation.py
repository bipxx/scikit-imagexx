"""CI artifact/data-flow guards; no native or measurement acceptance claim."""
import ast
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ci_artifacts', ROOT / 'tools/ci_measurement_artifacts.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def jobs(text):
    parts = re.split(r'^  ([\w-]+):\s*$', text.split('\njobs:', 1)[1], flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def timing_commands(block):
    return re.findall(r'^      - run: (python benchmarks/[^\n]+)', block, flags=re.M)


def test_timing_commands_only_on_fresh_artifact_hosts():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    blocks = jobs(workflow)
    assert set(blocks) == {'check', 'numpy-minimum', 'measure', 'measure-minimum', 'debug-gil'}
    for name in ('check', 'numpy-minimum', 'debug-gil'):
        assert not re.search(r'python\s+benchmarks/', blocks[name]), name
    for name, dependency in [('measure', 'check'), ('measure-minimum', 'numpy-minimum')]:
        block = blocks[name]
        assert 'needs: ' + dependency in block and 'runs-on:' in block
        assert 'actions/download-artifact@v4' in block
        assert '--no-index --no-deps --require-hashes --find-links build/prepared/wheels' in block
        commands = '\n'.join(l for l in block.splitlines() if not l.lstrip().startswith('#'))
        assert not re.search(r'pytest|pip wheel|cmake|ctest|pip install \.|--no-build-isolation', commands)
        assert block.count('ci_measurement_artifacts.py verify') == 2
        assert block.index('ci_measurement_artifacts.py verify') < block.index('python benchmarks/')
        assert block.rindex('ci_measurement_artifacts.py verify') > block.rindex('python benchmarks/')
        assert 'if: always()' in block and 'build/ci-*.json' in block
    for name in ('check', 'numpy-minimum'):
        block = blocks[name]
        assert 'ci_measurement_artifacts.py freeze' in block
        assert 'runtime-freeze.txt' in block and '--wheel-dir build/wheels -r' in block
        assert 'build/wheels/*.whl' in block and 'build/ci-artifacts/' in block


def test_minimum_only_and_debug_assertions_are_enforced():
    blocks = jobs((ROOT / '.github/workflows/ci.yml').read_text())
    condition = "if: github.event_name != 'workflow_dispatch' || !inputs.minimum_only"
    assert all(condition in blocks[n] for n in ('check', 'measure', 'debug-gil'))
    assert all('inputs.minimum_only' not in blocks[n] for n in ('numpy-minimum', 'measure-minimum'))
    debug = blocks['debug-gil']
    assert '--config-settings=cmake.build-type=Debug' in debug
    assert '-UNDEBUG -UPYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF -DPYBIND11_ASSERT_GIL_HELD_INCREF_DECREF' in debug
    assert debug.index('ci_measurement_artifacts.py debug-assertions') < debug.index('python -m pytest')
    for name in ('test_api.py', 'test_rank_pipeline.py', 'test_rank_callbacks.py',
                 'test_component_consumers_pipeline.py', 'test_transform_classes_public.py',
                 'test_footprints_public.py', 'test_footprint_dense_pipeline.py',
                 'test_resample_core.py', 'test_resample_typed.py', 'test_alpha_color.py', 'test_frangi.py'):
        assert debug.count('tests/' + name) == 2
    assert "SKIMAGEXX_DISABLE_SIMD: '1'" in debug and 'normal.xml' in debug and 'scalar.xml' in debug


def test_debug_runtime_proof_rejects_release(monkeypatch):
    import skimagexx
    monkeypatch.setattr(skimagexx._core, '_consumer_gil_assertions', False, raising=False)
    with pytest.raises(AssertionError, match='must be active'):
        helper.main(['debug-assertions'])
    monkeypatch.setattr(skimagexx._core, '_consumer_gil_assertions', True)
    helper.main(['debug-assertions'])


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    wheels = tmp_path / 'wheels'; wheels.mkdir()
    site = tmp_path / 'site'; site.mkdir()
    distributions = {'numpy': 'numpy', 'scipy': 'scipy', 'scikit-image': 'skimage', 'scikit-imagexx': 'skimagexx'}
    for dist, package in distributions.items():
        with zipfile.ZipFile(wheels / (package + '-1-py3-none-any.whl'), 'w') as archive:
            archive.writestr(package + '-1.dist-info/METADATA', f'Name: {dist}\nVersion: 1\n')
            archive.writestr(package + '/__init__.py', b'# frozen fixture')
            if package == 'skimagexx':
                for name, value in {'include/skimagexx/image.hpp': b'header', 'lib/libskimagexx.a': b'archive',
                                    'lib/cmake/skimagexx/config.cmake': b'cmake', 'share/licenses/scikit-imagexx/LICENSE': b'license'}.items():
                    archive.writestr(name, value)
        with zipfile.ZipFile(wheels / (package + '-1-py3-none-any.whl')) as archive:
            archive.extractall(site)
    (site / 'numpy.libs').mkdir()
    (site / 'numpy.libs/blas.so').write_bytes(b'frozen dependency library')
    monkeypatch.setattr(helper.metadata, 'version', lambda name: '1')
    monkeypatch.setattr(helper.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(site / name / '__init__.py')))
    monkeypatch.setattr(helper, 'source_state', lambda root: {'revision': 'unit-fixture', 'files_sha256': {'test': 'frozen'}})
    artifact = tmp_path / 'artifacts/measurement.json'
    helper.freeze(tmp_path, wheels, artifact)
    return tmp_path, wheels, site, artifact


def test_exact_offline_seal_and_hash_lock(sealed):
    root, wheels, site, artifact = sealed
    result = helper.verify(root, wheels, artifact)
    assert len(result['wheels']) == 4 and len(result['runtime']['sdk_sha256']) == 4
    assert artifact.with_name('runtime-lock.txt').read_text().count('--hash=sha256:') == 4
    assert set(result['runtime']['packages_sha256']) == {'numpy', 'scipy', 'skimage', 'skimagexx'}


@pytest.mark.parametrize('target', ['wheel', 'lock', 'package', 'reference', 'dependency-library', 'sdk', 'extra-package', 'extra-sdk', 'source', 'runtime'])
def test_artifact_or_runtime_tampering_rejected(sealed, monkeypatch, target):
    root, wheels, site, artifact = sealed
    if target == 'wheel':
        path = next(wheels.glob('*.whl')); path.write_bytes(path.read_bytes() + b'changed')
    elif target == 'lock':
        artifact.with_name('runtime-lock.txt').write_text('changed')
    elif target in ('package', 'reference'):
        (site / ('skimagexx' if target == 'package' else 'numpy') / '__init__.py').write_bytes(b'changed')
    elif target == 'dependency-library':
        (site / 'numpy.libs/blas.so').write_bytes(b'changed')
    elif target == 'sdk':
        (site / 'include/skimagexx/image.hpp').write_bytes(b'changed')
    elif target == 'extra-package':
        (site / 'skimagexx/extra.py').write_bytes(b'extra')
    elif target == 'extra-sdk':
        (site / 'include/skimagexx/extra.hpp').write_bytes(b'extra')
    elif target == 'source':
        monkeypatch.setattr(helper, 'source_state', lambda root: {'revision': 'changed'})
    elif target == 'runtime':
        monkeypatch.setattr(helper.metadata, 'version', lambda name: '2')
    with pytest.raises(AssertionError):
        helper.verify(root, wheels, artifact)


def test_source_seal_rejects_dirty_source_and_tracks_all_files(tmp_path):
    helper.subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    source = tmp_path / 'source.py'; source.write_text('# source\n')
    helper.subprocess.run(['git', '-C', str(tmp_path), 'add', 'source.py'], check=True)
    helper.subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=CI test', '-c',
                           'user.email=ci-test@example.invalid', 'commit', '-m', 'fixture'], check=True, capture_output=True)
    before = helper.source_state(tmp_path)
    assert set(before['files_sha256']) == {'source.py'}
    source.write_text('# changed\n')
    with pytest.raises(AssertionError, match='must be clean'):
        helper.source_state(tmp_path)


def test_generated_inventory_is_retained_and_source_restored(tmp_path, monkeypatch):
    coverage = tmp_path / 'coverage'; coverage.mkdir()
    for name in ('api.json', 'README.md'):
        (coverage / name).write_bytes(b'original')
    def generate(*args, **kwargs):
        for name in ('api.json', 'README.md'):
            (coverage / name).write_bytes(b'generated')
    monkeypatch.setattr(helper.subprocess, 'run', generate)
    helper.inventory_snapshot(tmp_path)
    for name in ('api.json', 'README.md'):
        assert (coverage / name).read_bytes() == b'original'
        assert (tmp_path / 'build/ci-artifacts/inventory' / name).read_bytes() == b'generated'


def test_generated_inventory_restores_source_after_failure(tmp_path, monkeypatch):
    coverage = tmp_path / 'coverage'; coverage.mkdir()
    for name in ('api.json', 'README.md'):
        (coverage / name).write_bytes(b'original')
    def fail(*args, **kwargs):
        (coverage / 'api.json').write_bytes(b'partial')
        raise RuntimeError('unit failure')
    monkeypatch.setattr(helper.subprocess, 'run', fail)
    with pytest.raises(RuntimeError):
        helper.inventory_snapshot(tmp_path)
    assert all(p.read_bytes() == b'original' for p in coverage.iterdir())


# Frozen original commands: arguments, one-worker inputs and output paths stay exact.
ORIGINAL_COMMANDS = {'modern': ['python benchmarks/coverage.py --sizes 64 --threads 1 --repeat 1 --output build/ci-performance.json', 'python benchmarks/integral.py --size 32 --threads 1 --repeat 1 --output build/ci-integral.json', 'python benchmarks/integrate.py --size 32 --threads 1 --repeat 1 --output build/ci-integrate.json', 'python benchmarks/downscale.py --size 32 --threads 1 --repeat 1 --output build/ci-downscale.json', 'python benchmarks/contrast.py --size 32 --threads 1 --repeat 1 --output build/ci-contrast.json', 'python benchmarks/gaussian.py --size 32 --threads 1 --repeat 1 --output build/ci-gaussian.json', 'python benchmarks/metadata.py --sizes 32 --threads 1 --repeat 1 --output build/ci-metadata.json', 'python benchmarks/conversions.py --size 64 --threads 1 --dtypes uint16 int16 --output build/ci-conversions.json', 'python benchmarks/color.py --size 32 --threads 1 --output build/ci-color-layouts.json', 'python benchmarks/perceptual.py --size 32 --threads 1 --output build/ci-perceptual.json', 'python benchmarks/stains.py --size 32 --threads 1 --repeat 1 --output build/ci-stains.json', 'python benchmarks/farid.py --size 32 --threads 1 --repeat 1 --output build/ci-farid.json', 'python benchmarks/edges.py --size 32 --threads 1 --repeat 1 --output build/ci-edges.json', 'python benchmarks/alpha.py --size 32 --threads 1 --repeat 1 --output build/ci-alpha.json', 'python benchmarks/local_thresholds.py --size 32 --threads 1 --repeat 1 --output build/ci-local-thresholds.json', 'python benchmarks/thresholds.py --size 32 --threads 1 --repeat 1 --output build/ci-thresholds.json', 'python benchmarks/minimum.py --size 32 --threads 1 --repeat 1 --output build/ci-minimum.json', 'python benchmarks/li.py --size 32 --threads 1 --repeat 1 --output build/ci-li.json', 'python benchmarks/adaptive.py --size 32 --threads 1 --repeat 1 --output build/ci-adaptive.json', 'python benchmarks/rank_order.py --size 32 --threads 1 --repeat 1 --output build/ci-rank-order.json', 'python benchmarks/hysteresis.py --size 32 --threads 1 --repeat 1 --output build/ci-hysteresis.json', 'python benchmarks/sparse.py --size 32 --threads 1 --repeat 1 --output build/ci-sparse.json', 'python benchmarks/median.py --size 32 --threads 1 --repeat 1 --output build/ci-median.json', 'python benchmarks/multiotsu.py --size 32 --threads 1 --repeat 1 --output build/ci-multiotsu.json', 'python benchmarks/clahe.py --size 32 --threads 1 --repeat 1 --output build/ci-clahe.json', 'python benchmarks/label_colors.py --size 32 --threads 1 --repeat 1 --output build/ci-label-colors.json', 'python benchmarks/distance.py --size 32 --threads 1 --repeat 1 --output build/ci-distance.json', 'python benchmarks/morphology.py --size 32 --threads 1 --repeat 1 --output build/ci-morphology.json', 'python benchmarks/match_histograms.py --size 32 --threads 1 --repeat 1 --output build/ci-match-histograms.json'], 'minimum': ['python benchmarks/ciede2000.py --threads 1 --output build/ci-ciede2000.json']}


def test_original_timing_commands_and_corpora_preserved():
    blocks = jobs((ROOT / '.github/workflows/ci.yml').read_text())
    assert timing_commands(blocks['measure']) == ORIGINAL_COMMANDS['modern']
    assert timing_commands(blocks['measure-minimum']) == ORIGINAL_COMMANDS['minimum'] + [
        'python benchmarks/coverage.py --api filters.butterworth --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-butterworth.json',
        'python benchmarks/coverage.py --api filters.filter_forward --api filters.filter_inverse --api filters.wiener --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-lpi.json',
        'python benchmarks/coverage.py --api filters.window --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-window.json']
    assert len(ORIGINAL_COMMANDS['modern']) == 29 and len(ORIGINAL_COMMANDS['minimum']) == 1
    assert all('--threads 1' in c for cs in ORIGINAL_COMMANDS.values() for c in cs)


def test_lpi_contracts_precede_measurement_artifact_freeze():
    blocks = jobs((ROOT / '.github/workflows/ci.yml').read_text())
    preparation = blocks['numpy-minimum']
    required = """      - name: LPI owned FFT parity
        run: python -m pytest -q tests/test_lpi_filter_public.py --junitxml=build/ci-artifacts/lpi-normal.xml
      - name: LPI owned FFT parity with SIMD disabled
        run: python -m pytest -q tests/test_lpi_filter_public.py --junitxml=build/ci-artifacts/lpi-scalar.xml
        env:
          SKIMAGEXX_DISABLE_SIMD: '1'
"""
    assert required in preparation
    assert preparation.index(required) < preparation.index('ci_measurement_artifacts.py freeze')
    assert preparation.count('tests/test_lpi_filter_public.py') == 2
    assert 'build/ci-artifacts/' in preparation
    assert 'build/ci-lpi.md' in blocks['measure-minimum']


def test_window_contracts_precede_measurement_artifact_freeze():
    blocks = jobs((ROOT / '.github/workflows/ci.yml').read_text())
    preparation = blocks['numpy-minimum']
    required = """      - name: Window owned sampling parity
        run: python -m pytest -q tests/test_window_public.py --junitxml=build/ci-artifacts/window-normal.xml
      - name: Window owned sampling parity with SIMD disabled
        run: python -m pytest -q tests/test_window_public.py --junitxml=build/ci-artifacts/window-scalar.xml
        env:
          SKIMAGEXX_DISABLE_SIMD: '1'
"""
    assert required in preparation
    assert preparation.index('tests/test_lpi_filter_public.py') < preparation.index(required)
    assert preparation.index(required) < preparation.index('ci_measurement_artifacts.py freeze')
    assert preparation.count('tests/test_window_public.py') == 2
    assert 'build/ci-artifacts/' in preparation
    assert 'build/ci-window.md' in blocks['measure-minimum']


def test_window_shape_only_benchmark_has_two_cases_without_changing_other_apis():
    tree = ast.parse((ROOT / 'benchmarks/coverage.py').read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    guards = [n for n in ast.walk(main) if isinstance(n, ast.If)
              and any(isinstance(v, ast.Constant) and v.value == 'filters.window' for v in ast.walk(n.test))
              and len(n.body) == 1 and isinstance(n.body[0], ast.Continue)]
    assert len(guards) == 1 and not guards[0].orelse
    predicate = compile(ast.Expression(guards[0].test), '<window fixture predicate>', 'eval')
    entries = json.loads((ROOT / 'coverage/api.json').read_bytes())['entries']
    assert len(entries) == 477 and any(e['api'] == 'filters.window' for e in entries)
    for entry in entries:
        selected = [(size, dtype) for size in (256, 1024) for dtype in ('uint8', 'float32', 'float64')
                    if not eval(predicate, {'np': SimpleNamespace(float64='float64'),
                                            'api': entry['api'], 'dtype': dtype})]
        expected = [(256, 'float64'), (1024, 'float64')] if entry['api'] == 'filters.window' else [
            (size, dtype) for size in (256, 1024) for dtype in ('uint8', 'float32', 'float64')]
        assert selected == expected, entry['api']


def test_window_provider_metadata_does_not_claim_fft_workers():
    tree = ast.parse((ROOT / 'benchmarks/coverage.py').read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    scope_node = next(n for n in main.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'window_scope' for t in n.targets))
    scope = ast.literal_eval(scope_node.value)
    for text in ('SciPy signal.get_window', 'NumPy radial coordinates and clipping',
                 'owned C++ order-0/1 sampling', 'Default Hann/order1', 'no pixel input', 'float64 output'):
        assert text in scope
    frontier = next(n for n in main.body if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Constant)
                    and n.test.left.value == 'filters.window')
    assert len(frontier.body) == 1 and isinstance(frontier.body[0], ast.Assign)
    values = {'apis': ['filters.window'], 'provider_frontiers': {}, 'window_scope': scope}
    exec(compile(ast.Module(body=[frontier], type_ignores=[]), '<window provider metadata>', 'exec'), values)
    assert values['provider_frontiers'] == {'filters.window': scope}
    fft_guard = next(n for n in main.body if isinstance(n, ast.If)
                     and any(isinstance(v, ast.Import) and any(a.name == 'scipy.fft' for a in v.names)
                             for v in n.body))
    condition = compile(ast.Expression(fft_guard.test), '<reference FFT selection>', 'eval')
    lpi = {'filters.filter_forward', 'filters.filter_inverse', 'filters.wiener'}
    assert not eval(condition, {'lpi_apis': lpi, 'apis': ['filters.window']})
    for api in lpi | {'filters.butterworth'}:
        assert eval(condition, {'lpi_apis': lpi, 'apis': [api]})
