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


def level_set_family(tree):
    """Load the actual adapter family for isolated AST execution."""
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id=='LEVEL_SET_FUNCTIONS' for target in node.targets))


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
    # Offline identity includes the full interpreter patch version, not only ABI.
    versions = [re.findall(r"python-version: '([^']+)'", blocks[name])
                for name in ('numpy-minimum', 'measure-minimum')]
    assert versions == [['3.11.17'], ['3.11.17']]
    # Strict Windows LevelSet lookup is pinned to this witnessed interpreter.
    witnessed = json.loads((ROOT / 'coverage/contracts/level-set-windows.json').read_bytes())['abi']['python_version']
    assert re.fullmatch(r'3\.13\.\d+', witnessed)
    latest = [re.findall(r"python-version: '([^']+)'", blocks[name])
              for name in ('check', 'measure', 'debug-gil')]
    assert latest == [[witnessed], [witnessed], [witnessed]]
    condition = "if: github.event_name != 'workflow_dispatch' || !inputs.minimum_only"
    assert all(condition in blocks[n] for n in ('check', 'debug-gil'))
    assert "if: github.event_name != 'workflow_dispatch' || (!inputs.minimum_only && !inputs.prepare_only && !inputs.latest_only)" in blocks['measure']
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


@pytest.mark.parametrize('job,replacement',(
    ('check', '3.13'), ('measure', '3.13.17'), ('debug-gil', '3.13'),
    ('numpy-minimum', '3.11'), ('measure-minimum', '3.11.18')))
def test_runtime_python_pins_reject_floating_or_mismatched_interpreters(monkeypatch,job,replacement):
    path = ROOT / '.github/workflows/ci.yml'
    text = path.read_text()
    block = jobs(text)[job]
    changed, count = re.subn(r"python-version: '[^']+'", "python-version: '" + replacement + "'", block)
    assert count == 1
    mutated = text.replace(block, changed, 1)
    original = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self,*a,**kw: mutated if self == path else original(self,*a,**kw))
    with pytest.raises(AssertionError):
        test_minimum_only_and_debug_assertions_are_enforced()


def test_debug_runtime_proof_rejects_release(monkeypatch):
    import sys
    from types import ModuleType
    skimagexx = ModuleType('skimagexx')
    skimagexx._core = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'skimagexx', skimagexx)
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
        'python benchmarks/coverage.py --api filters.window --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-window.json',
        'python benchmarks/coverage.py --api measure.moments --api measure.moments_central --api measure.centroid --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-moments.json',
        'python benchmarks/coverage.py --api measure.moments_normalized --api measure.moments_hu --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-moments-derived.json',
        'python benchmarks/coverage.py --api measure.inertia_tensor --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-inertia-tensor.json',
        'python benchmarks/coverage.py --api measure.inertia_tensor_eigvals --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-inertia-eigvals.json',
        'python benchmarks/coverage.py --api measure.moments_coords --api measure.moments_coords_central --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-moments-coords.json',
        'python benchmarks/coverage.py --api measure.shannon_entropy --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-shannon-entropy.json',
        'python benchmarks/rgb2yuv_storage.py --threads 1 --repeat 7 --output build/ci-rgb2yuv-storage.json',
        'python benchmarks/coverage.py --api measure.block_reduce --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-block-reduce.json',
        'python benchmarks/coverage.py --api measure.perimeter_crofton --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-perimeter-crofton.json',
        'python benchmarks/coverage.py --api measure.perimeter --sizes 256 1024 --threads 1 --repeat 7 --output build/ci-perimeter.json',
        'python benchmarks/euler_number.py --threads 1 --repeat 7 --output build/ci-euler-number.json',
        'python benchmarks/polygon.py --threads 1 --repeat 7 --output build/ci-polygon.json',
        'python benchmarks/mesh_surface_area.py --threads 1 --repeat 7 --output build/ci-mesh-surface-area.json',
        'python benchmarks/profile_line.py --threads 1 --repeat 7 --output build/ci-profile-line.json',
        'python benchmarks/blur_effect.py --threads 1 --repeat 7 --output build/ci-blur-effect.json',
        'python benchmarks/compare_images.py --threads 1 --repeat 7 --output build/ci-compare-images.json',
        'python benchmarks/level_set.py --threads 1 --repeat 7 --output build/ci-level-set.json',
        'python benchmarks/map_array.py --threads 1 --repeat 7 --output build/ci-map-array.json',
        'python benchmarks/label_points.py --threads 1 --repeat 7 --output build/ci-label-points.json',
        'python benchmarks/unique_rows.py --threads 1 --repeat 7 --output build/ci-unique-rows.json',
        'python benchmarks/regular_seeds.py --threads 1 --repeat 7 --output build/ci-regular-seeds.json',
        'python benchmarks/montage.py --threads 1 --repeat 7 --output build/ci-montage.json',
        'python benchmarks/rolling_ball_kernels.py --threads 1 --repeat 7 --output build/ci-rolling-ball-kernels.json',
        'python benchmarks/approximate_polygon.py --threads 1 --repeat 7 --output build/ci-approximate-polygon.json',
        'python benchmarks/subdivide_polygon.py --threads 1 --repeat 7 --output build/ci-subdivide-polygon.json',
        'python benchmarks/circle_perimeter.py --threads 1 --repeat 7 --output build/ci-circle-perimeter.json',
        'python benchmarks/line_aa.py --threads 1 --repeat 7 --output build/ci-line-aa.json',
        'python benchmarks/regionprops_state.py --threads 1 --repeat 7 --output build/ci-regionprops-state.json',
        'python benchmarks/circle_perimeter_aa.py --threads 1 --repeat 7 --output build/ci-circle-perimeter-aa.json',
        'python benchmarks/stains.py --half-only --size 512 --threads 1 --repeat 7 --output build/ci-combine-stains-half.json',
        'python benchmarks/line_nd.py --threads 1 --repeat 7 --output build/ci-line-nd.json',
        'python benchmarks/draw_ellipsoid.py --threads 1 --repeat 7 --output build/ci-ellipsoid.json',
        'python benchmarks/ellipse_perimeter.py --threads 1 --repeat 7 --output build/ci-ellipse-perimeter.json',
        'python benchmarks/regionprops_scalars.py --threads 1 --repeat 7 --output build/ci-regionprops-scalars.json']
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


def test_moments_contracts_precede_measurement_artifact_freeze():
    blocks = jobs((ROOT / '.github/workflows/ci.yml').read_text())
    preparation = blocks['numpy-minimum']
    required = """      - name: Moments owned projection parity
        run: python -m pytest -q tests/test_moments_public.py --junitxml=build/ci-artifacts/moments-normal.xml
      - name: Moments owned projection parity with SIMD disabled
        run: python -m pytest -q tests/test_moments_public.py --junitxml=build/ci-artifacts/moments-scalar.xml
        env:
          SKIMAGEXX_DISABLE_SIMD: '1'
"""
    assert required in preparation
    assert preparation.index('tests/test_window_public.py') < preparation.index(required)
    assert preparation.index(required) < preparation.index('ci_measurement_artifacts.py freeze')
    assert preparation.count('tests/test_moments_public.py') == 2
    assert 'build/ci-artifacts/' in preparation
    assert 'build/ci-moments.md' in blocks['measure-minimum']


def test_moments_benchmark_routes_projection_and_derived_cellwise_checks():
    source = ast.parse((ROOT / 'python/skimagexx/coverage.py').read_bytes())
    declaration = next(n for n in source.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'MOMENTS_FUNCTIONS' for t in n.targets))
    family = ast.literal_eval(declaration.value)
    assert set(family) == {'measure.moments', 'measure.moments_central', 'measure.centroid'}
    tree = ast.parse((ROOT / 'benchmarks/coverage.py').read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    route = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                 and n.test.left.id == 'api' and any(isinstance(v, ast.Name) and v.id == 'MOMENTS_FUNCTIONS'
                                                   for v in ast.walk(n.test)))
    old = ast.parse('rtol,atol = parity_tolerance(api,np.dtype(dtype),image=data,expected=expected)\n'
                    'error=comparison_error(actual,expected,rtol,atol)').body
    derived = ('measure.moments_normalized', 'measure.moments_hu')
    assert len(route.orelse) == 1 and isinstance(route.orelse[0], ast.If)
    assert [ast.dump(n) for n in route.orelse[0].orelse] == [ast.dump(n) for n in old]
    code = compile(ast.Module(body=[route], type_ignores=[]), '<moments parity selection>', 'exec')
    for entry in json.loads((ROOT / 'coverage/api.json').read_bytes())['entries']:
        calls = []
        def moments(api, image, actual, expected):
            calls.append(('moments', api, image, actual, expected)); return 17
        def tolerance(api, dtype, *, image, expected):
            calls.append(('old tolerance', api, dtype, image, expected)); return 1, 2
        def compare(actual, expected, rtol, atol):
            calls.append(('old comparison', actual, expected, rtol, atol)); return 23
        values = dict(api=entry['api'], MOMENTS_FUNCTIONS=family, DERIVED_MOMENTS_FUNCTIONS=derived,
                      call_args=('prepared moments',), dtype='dtype', data='image',
                      actual='actual', expected='expected', np=SimpleNamespace(dtype=lambda x: x),
                      moments_default_error=moments, parity_tolerance=tolerance, comparison_error=compare)
        exec(code, values)
        if entry['api'] in family:
            assert calls == [('moments', entry['api'], 'image', 'actual', 'expected')] and values['error'] == 17
        elif entry['api'] in derived:
            assert calls == [('moments', entry['api'], 'prepared moments', 'actual', 'expected')] and values['error'] == 17
        else:
            assert calls == [('old tolerance', entry['api'], 'dtype', 'image', 'expected'),
                             ('old comparison', 'actual', 'expected', 1, 2)] and values['error'] == 23


def test_moments_provider_metadata_does_not_claim_fft_workers():
    tree = ast.parse((ROOT / 'benchmarks/coverage.py').read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    scope_node = next(n for n in main.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'moments_scope' for t in n.targets))
    scope = ast.literal_eval(scope_node.value)
    for text in ('NumPy dtype promotion', 'coordinate powers', 'layout preparation',
                 'analytical central-moment', 'centroid-ratio', 'owned typed C++ per-axis projections',
                 'floating-point flags', 'shared global workers'):
        assert text in scope
    family = ('measure.moments', 'measure.moments_central', 'measure.centroid')
    update = next(n for n in main.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                  and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'update'
                  and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'provider_frontiers')
    values = dict(apis=[*family, 'filters.window'], MOMENTS_FUNCTIONS=family,
                  provider_frontiers={'filters.window': 'old scope'}, moments_scope=scope)
    exec(compile(ast.Module(body=[update], type_ignores=[]), '<moments provider metadata>', 'exec'), values)
    assert values['provider_frontiers'] == {'filters.window': 'old scope', **dict.fromkeys(family, scope)}
    fft_guard = next(n for n in main.body if isinstance(n, ast.If)
                     and any(isinstance(v, ast.Import) and any(a.name == 'scipy.fft' for a in v.names)
                             for v in n.body))
    condition = compile(ast.Expression(fft_guard.test), '<reference FFT selection>', 'eval')
    lpi = {'filters.filter_forward', 'filters.filter_inverse', 'filters.wiener'}
    assert not eval(condition, {'lpi_apis': lpi, 'apis': family})
    for api in lpi | {'filters.butterworth'}:
        assert eval(condition, {'lpi_apis': lpi, 'apis': [api]})


def test_derived_benchmark_has_five_actual_tensor_cases():
    source = ast.parse((ROOT / 'python/skimagexx/coverage.py').read_bytes())
    coords_family = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'COORDS_MOMENTS_FUNCTIONS' for t in n.targets))
    family = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'DERIVED_MOMENTS_FUNCTIONS' for t in n.targets))
    dtype_function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'native_dtypes')
    namespace = {'DRAW_COORDINATE_FUNCTIONS':next(ast.literal_eval(n.value) for n in source.body
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                     and t.id == 'DRAW_COORDINATE_FUNCTIONS' for t in n.targets)),
                 'LEVEL_SET_FUNCTIONS':level_set_family(source),
                 'ROLLING_KERNEL_FUNCTIONS':next(ast.literal_eval(n.value) for n in source.body
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                     and t.id == 'ROLLING_KERNEL_FUNCTIONS' for t in n.targets))}
    exec(compile(ast.Module(body=[dtype_function], type_ignores=[]), '<supported default dtypes>', 'exec'), namespace)
    main = next(n for n in ast.parse((ROOT / 'benchmarks/coverage.py').read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    guard = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                 and any(isinstance(v, ast.Name) and v.id == 'DERIVED_MOMENTS_FUNCTIONS' for v in ast.walk(n.test))
                 and len(n.body) == 1 and isinstance(n.body[0], ast.Continue))
    predicate = compile(ast.Expression(guard.test), '<fixed tensor deduplication>', 'eval')
    selected = [(api, dtype) for api in family for size_index in range(3)
                for dtype in ('uint8', 'float32', 'float64')
                if dtype in namespace['native_dtypes'](api)
                and not eval(predicate, dict(api=api, size_index=size_index, DERIVED_MOMENTS_FUNCTIONS=family))]
    assert selected == [('measure.moments_normalized', 'float32'), ('measure.moments_normalized', 'float64'),
                        ('measure.moments_hu', 'uint8'), ('measure.moments_hu', 'float32'), ('measure.moments_hu', 'float64')]
    for entry in json.loads((ROOT / 'coverage/api.json').read_bytes())['entries']:
        if entry['api'] not in family:
            assert not eval(predicate, dict(api=entry['api'], size_index=2, DERIVED_MOMENTS_FUNCTIONS=family))
    labels = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id in ('case_size', 'case_dtype') for t in n.targets)]
    code = compile(ast.Module(body=labels, type_ignores=[]), '<actual input labels>', 'exec')
    for api in (*family, 'measure.moments'):
        values = dict(api=api, LEVEL_SET_FUNCTIONS=level_set_family(source), DERIVED_MOMENTS_FUNCTIONS=family, COORDS_MOMENTS_FUNCTIONS=coords_family, size=1024, dtype='fixture dtype',
                      call_args=(SimpleNamespace(dtype=SimpleNamespace(name='actual dtype')),),
                      np=SimpleNamespace(dtype=lambda x: SimpleNamespace(name=x)))
        exec(code, values)
        assert (values['case_size'], values['case_dtype']) == ((4, 'actual dtype') if api in family else (1024, 'fixture dtype'))


@pytest.mark.parametrize('api', ['measure.moments_normalized', 'measure.moments_hu'])
@pytest.mark.parametrize('shape', [(1, 1), (3, 19), (519, 521)])
def test_derived_default_preparation_metadata_and_copy_boundary(api, shape):
    # Execute the real preparer's control/data flow with a small storage spy;
    # this metadata test neither imports NumPy nor claims numerical parity.
    tree = ast.parse((ROOT / 'python/skimagexx/coverage.py').read_bytes())
    coords_family = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'COORDS_MOMENTS_FUNCTIONS' for t in n.targets))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'default_call')
    class Tensor:
        def __init__(self, shape, dtype):
            self.shape, self.dtype, self.flat = shape, dtype, self
            self.values = [None] * 16
        def __setitem__(self, index, value):
            if isinstance(index, slice):self.values[index] = value
            else:self.values[4 * index[0] + index[1]] = value
    descriptor = SimpleNamespace(name='float32', metadata={'unit': 'moment'})
    image = SimpleNamespace(shape=shape, size=shape[0] * shape[1], dtype=descriptor,
                            flat=range(shape[0] * shape[1]))
    def import_numpy(name, *args, **kwargs):
        assert name == 'numpy'
        return SimpleNamespace(empty=lambda shape, dtype: Tensor(shape, dtype))
    namespace = dict(DRAW_COORDINATE_FUNCTIONS=next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                     and t.id == 'DRAW_COORDINATE_FUNCTIONS' for t in n.targets)),
                     LEVEL_SET_FUNCTIONS=level_set_family(tree), DERIVED_MOMENTS_FUNCTIONS=('measure.moments_normalized', 'measure.moments_hu'),
                     COORDS_MOMENTS_FUNCTIONS=coords_family,
                     __builtins__={**vars(__import__('builtins')), '__import__': import_numpy})
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<derived preparer>', 'exec'), namespace)
    (prepared,), options = namespace['default_call'](api, image)
    expected = [i % image.size for i in range(16)]
    if api.endswith('normalized'):expected[0], expected[1], expected[4] = 4, 0, 0
    assert prepared is not image and prepared.shape == (4, 4) and prepared.dtype is descriptor
    assert prepared.values == expected and options == {}
    assert namespace['default_call']('measure.moments', image) == ((image,), {})


@pytest.mark.parametrize('api', ['measure.moments_coords', 'measure.moments_coords_central'])
@pytest.mark.parametrize('shape', [(1, 1), (3, 19), (256, 256)])
def test_coordinate_default_preparation_uses_actual_points(api, shape):
    tree = ast.parse((ROOT / 'python/skimagexx/coverage.py').read_bytes())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'default_call')
    family = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'COORDS_MOMENTS_FUNCTIONS' for t in n.targets))
    class Points:
        def __init__(self, shape, dtype):
            self.shape, self.dtype, self.flat = shape, dtype, self
            self.size = shape[0]*shape[1]
        def __setitem__(self, index, values):
            assert index == slice(None)
            self.values = values
    descriptor = SimpleNamespace(name='float32', metadata={'unit': 'coordinate'})
    image = SimpleNamespace(shape=shape, size=shape[0]*shape[1], dtype=descriptor,
                            flat=range(shape[0]*shape[1]))
    def import_numpy(name, *args, **kwargs):
        assert name == 'numpy'
        return SimpleNamespace(empty=lambda shape, dtype: Points(shape, dtype))
    namespace = dict(DRAW_COORDINATE_FUNCTIONS=next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                     and t.id == 'DRAW_COORDINATE_FUNCTIONS' for t in n.targets)),
                     LEVEL_SET_FUNCTIONS=level_set_family(tree), __builtins__={**vars(__import__('builtins')), '__import__': import_numpy})
    exec(compile(ast.Module(body=[family, function], type_ignores=[]), '<coordinate preparer>', 'exec'), namespace)
    (points,), options = namespace['default_call'](api, image)
    assert points is not image and points.shape == (shape[0], 2) and points.dtype is descriptor
    assert points.values == [i % image.size for i in range(2*shape[0])] and options == {}


def test_prepare_only_retains_verified_minimum_artifact_without_timing():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    assert 'prepare_only:\n        description: Prepare verified offline artifacts without standalone timing\n        type: boolean\n        default: false' in workflow
    blocks = jobs(workflow)
    minimum = blocks['measure-minimum']
    condition = "github.event_name != 'workflow_dispatch' || !inputs.prepare_only"
    assert not minimum.lstrip().startswith('if:')
    steps = re.split(r'^      - ', minimum, flags=re.M)[1:]
    timed = [step for step in steps if step.startswith('run: python benchmarks/')]
    assert len(timed) == 38 and all(condition in step for step in timed)
    assert all(condition not in step for step in steps if step not in timed)
    assert minimum.count('ci_measurement_artifacts.py verify') == 2
    assert 'name: ci-performance-minimum' in minimum
    assert 'build/prepared/ci-artifacts/' in minimum and 'build/prepared/wheels/' in minimum
    assert 'prepare_only' not in blocks['numpy-minimum']
    assert 'prepare_only' not in blocks['check'] and 'prepare_only' not in blocks['debug-gil']


def test_latest_only_keeps_all_gates_and_times_only_newest_batch():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    assert "latest_only:\n        description: Time only the newest minimum-Linux API batch; retain all build and parity gates\n        type: boolean\n        default: false" in workflow
    blocks = jobs(workflow)
    minimum = blocks['measure-minimum']
    timed = [s for s in re.split(r'^      - ', minimum, flags=re.M)[1:] if s.startswith('run: python benchmarks/')]
    skip = "(github.event_name != 'workflow_dispatch' || !inputs.latest_only)"
    assert len(timed) == 38 and all(skip in s for s in timed[:-1])
    assert skip not in timed[-1]
    assert 'benchmarks/regionprops_state.py ' in timed[-7]
    assert 'benchmarks/circle_perimeter_aa.py ' in timed[-6]
    assert 'benchmarks/stains.py --half-only ' in timed[-5]
    assert 'benchmarks/line_nd.py ' in timed[-4]
    assert 'benchmarks/draw_ellipsoid.py ' in timed[-3]
    assert 'benchmarks/ellipse_perimeter.py ' in timed[-2]
    assert 'benchmarks/regionprops_scalars.py ' in timed[-1]
    assert '!inputs.latest_only' in blocks['measure'].lstrip().splitlines()[0]
    assert 'latest_only' not in blocks['numpy-minimum']
    assert 'latest_only' not in blocks['check'] and 'latest_only' not in blocks['debug-gil']


@pytest.mark.parametrize('stem', ('perimeter-crofton', 'perimeter', 'euler-number', 'mesh-surface-area', 'level-set', 'map-array', 'label-points', 'unique-rows', 'regular-seeds'))
def test_measure_contracts_precede_freeze(stem):
    blocks = jobs((ROOT / '.github/workflows/ci.yml').read_text())
    preparation = blocks['numpy-minimum']
    for mode in ('normal', 'scalar'):
        command = ('python -m pytest -q tests/test_' + stem.replace('-', '_') + '_public.py '
                   '--junitxml=build/ci-artifacts/' + stem + '-' + mode + '.xml')
        assert preparation.count(command) == 1
        assert preparation.index(command) < preparation.index('ci_measurement_artifacts.py freeze')
    scalar = preparation.split(stem + '-scalar.xml', 1)[1]
    assert scalar.startswith("\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")
    assert 'build/ci-' + stem + '.md' in blocks['measure-minimum']


@pytest.mark.parametrize('api,checker_name', (('measure.perimeter_crofton', 'crofton_default_error'),
                                             ('measure.approximate_polygon', 'approximate_polygon_default_error'),
                                             ('measure.subdivide_polygon', 'subdivide_polygon_default_error'),
                                             ('measure.perimeter', 'perimeter_default_error'),
                                             ('measure.euler_number', 'euler_default_error'),
                                             ('measure.blur_effect', 'blur_default_error'),
                                             ('util.compare_images', 'compare_default_error'),
                                             ('util.map_array', 'map_array_default_error'), ('util.label_points', 'label_points_default_error'), ('util.unique_rows', 'unique_rows_default_error'),
                                             ('util.montage', 'montage_default_error'),
                                             ('measure.regionprops', 'regionprops_default_error'),
                                             ('draw.line_nd', 'line_nd_default_error')))
def test_measure_benchmark_uses_shared_scalar_policy_before_timing(api, checker_name):
    tree = ast.parse((ROOT / 'benchmarks/coverage.py').read_bytes())
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'skimagexx.coverage']
    assert any(a.name == checker_name for n in imports for a in n.names)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "api == " + repr(api))
    calls = []
    image, actual, expected = object(), object(), object(); shape = object()
    def checker(*args):
        calls.append(args)
        return 17.
    code = compile(ast.Module(body=branch.body, type_ignores=[]), '<Crofton checker route>', 'exec')
    namespace = dict(call_args=(image, shape), data=image, actual=actual, expected=expected)
    namespace[checker_name] = checker
    exec(code, namespace)
    assert calls == [(shape if api == 'util.label_points' else image, actual, expected)] and namespace['error'] == 17.
    timing = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'measure')
    assert branch.lineno < timing.lineno


@pytest.mark.parametrize('stem', ('level-set', 'map-array', 'label-points', 'unique-rows', 'regular-seeds'))
def test_owned_native_pipeline_gates_precede_freeze(stem):
    preparation=jobs((ROOT / '.github/workflows/ci.yml').read_text())['numpy-minimum']
    for mode in ('normal','scalar'):
        command=('python -m pytest -q tests/test_'+stem.replace('-','_')+'_pipeline.py '
                 '--junitxml=build/ci-artifacts/'+stem+'-pipeline-'+mode+'.xml')
        assert preparation.count(command)==1
        assert preparation.index(command)<preparation.index('ci_measurement_artifacts.py freeze')
    assert preparation.split(stem+'-pipeline-scalar.xml',1)[1].startswith(
        "\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")


def test_montage_frozen_rows_public_exports_and_native_entry_precede_freeze():
    blocks = jobs((ROOT / '.github/workflows/ci.yml').read_text())
    preparation = blocks['numpy-minimum']
    for mode in ('normal', 'scalar'):
        command = ('python -m pytest -q tests/test_montage_public.py tests/test_montage_integration.py '
                   'tests/test_montage_pipeline.py --junitxml=build/ci-artifacts/montage-' + mode + '.xml')
        assert preparation.count(command) == 1
        assert preparation.index(command) < preparation.index('ci_measurement_artifacts.py freeze')
    assert preparation.split('montage-scalar.xml', 1)[1].startswith(
        "\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")
    assert 'build/ci-montage.md' in blocks['measure-minimum']


def test_regular_seeds_default_checker_uses_actual_shape_and_requested_output_dtype():
    tree=ast.parse((ROOT/'benchmarks/coverage.py').read_bytes())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    branch=next(n for n in ast.walk(main)if isinstance(n,ast.If)and ast.unparse(n.test)=="api == 'util.regular_seeds'")
    calls=[];shape,dtype,actual,expected=object(),object(),object(),object()
    def checker(*args):calls.append(args);return 23.
    namespace=dict(call_args=(shape,64),options={'dtype':dtype},actual=actual,expected=expected,regular_seeds_default_error=checker)
    exec(compile(ast.Module(body=branch.body,type_ignores=[]),'<regular seeds checker>','exec'),namespace)
    assert calls==[(shape,dtype,actual,expected)] and namespace['error']==23.
    timer=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure')
    assert branch.lineno<timer.lineno
    assert 'shape-count-output-dtype-adapter' in ast.unparse(main)


def test_map_array_phase3_preparation_keeps_both_selectors_and_exact_censuses():
    workflow=(ROOT/'.github/workflows/phase3-public-benchmark.yml').read_text()
    commands=[line.strip() for line in workflow.splitlines()
              if line.strip().startswith('python -m pytest -q tests/test_footprint_shapes_public.py')]
    assert len(commands)==2
    for mode,command in zip(('normal','scalar'),commands):
        assert command.count('tests/test_map_array_public.py')==1
        assert command.count('tests/test_map_array_pipeline.py')==1
        assert 'public-'+mode+'.xml' in command
    assert workflow.count('tests/test_regular_seeds_public.py tests/test_regular_seeds_pipeline.py') == 2
    assert 'assert len(cases) == 5472' in workflow
    assert 'len(native) == 250' in workflow and 'assert expected == 1447' in workflow
    contract=json.loads((ROOT/'coverage/contracts/map-array-benchmark.json').read_bytes())
    assert (contract['threads'],contract['repeat'],contract['target_ms'],contract['raw_samples'])==(1,7,19,126)


@pytest.mark.parametrize('api', ('segmentation.checkerboard_level_set','segmentation.disk_level_set'))
def test_level_set_benchmark_dispatches_exact_shape_checker(api):
    tree=ast.parse((ROOT/'benchmarks/coverage.py').read_bytes())
    imported={alias.name for node in tree.body if isinstance(node,ast.ImportFrom)
              and node.module=='skimagexx.coverage' for alias in node.names}
    assert {'LEVEL_SET_FUNCTIONS','level_set_default_error'}<=imported
    coverage=ast.parse((ROOT/'python/skimagexx/coverage.py').read_bytes())
    assignment=next(node for node in coverage.body if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id=='LEVEL_SET_FUNCTIONS' for target in node.targets))
    names=ast.literal_eval(assignment.value)
    assert set(names)=={'segmentation.checkerboard_level_set','segmentation.disk_level_set'}
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    branch=next(node for node in ast.walk(main) if isinstance(node,ast.If)
                and ast.unparse(node.test)=='api in LEVEL_SET_FUNCTIONS'
                and any(isinstance(call,ast.Call) and isinstance(call.func,ast.Name)
                        and call.func.id=='level_set_default_error' for call in ast.walk(node)))
    assert eval(compile(ast.Expression(branch.test),'<level-set selector>','eval'),dict(api=api,LEVEL_SET_FUNCTIONS=names))
    calls=[];shape=(3,19);actual=object();expected=object()
    def checker(*args):
        calls.append(args)
        return 17.
    namespace=dict(call_args=(shape,),actual=actual,expected=expected,level_set_default_error=checker)
    exec(compile(ast.Module(body=branch.body,type_ignores=[]),'<level-set exact check>','exec'),namespace)
    assert calls==[(shape,actual,expected)] and namespace['error']==17.


def test_rolling_kernel_public_and_owned_pipeline_gates_precede_freeze():
    preparation = jobs((ROOT / '.github/workflows/ci.yml').read_text())['numpy-minimum']
    for mode in ('normal', 'scalar'):
        command = ('python -m pytest -q tests/test_rolling_ball_kernels_integration.py '
                   'tests/test_rolling_ball_kernels_pipeline.py --junitxml=build/ci-artifacts/'
                   'rolling-ball-kernels-' + mode + '.xml')
        assert preparation.count(command) == 1
        assert preparation.index(command) < preparation.index('ci_measurement_artifacts.py freeze')
    assert preparation.split('rolling-ball-kernels-scalar.xml', 1)[1].startswith(
        "\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")


@pytest.mark.parametrize('api', ('restoration.ball_kernel', 'restoration.ellipsoid_kernel'))
def test_rolling_kernel_benchmark_uses_exact_inf_aware_checker(api):
    tree = ast.parse((ROOT / 'benchmarks/coverage.py').read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'api in ROLLING_KERNEL_FUNCTIONS'
                  and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                          and c.func.id == 'rolling_kernel_default_error' for c in ast.walk(n)))
    calls = []
    class Carrier:
        shape = (3, 19)
    actual, expected = object(), object()
    def checker(*args):
        calls.append(args)
        return 17.
    namespace = dict(api=api, data=Carrier(), actual=actual, expected=expected,
                     rolling_kernel_default_error=checker)
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), '<rolling exact check>', 'exec'), namespace)
    assert calls == [(api, Carrier.shape, actual, expected)] and namespace['error'] == 17.
    timer = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'measure')
    assert branch.lineno < timer.lineno


def test_polygon_registered_and_owned_pipeline_gates_precede_freeze():
    preparation = jobs((ROOT / '.github/workflows/ci.yml').read_text())['numpy-minimum']
    for mode in ('normal', 'scalar'):
        command = ('python -m pytest -q tests/test_approximate_polygon_integration.py '
                   'tests/test_approximate_polygon_pipeline.py --junitxml=build/ci-artifacts/'
                   'approximate-polygon-' + mode + '.xml')
        assert preparation.count(command) == 1
        assert preparation.index(command) < preparation.index('ci_measurement_artifacts.py freeze')
    assert preparation.split('approximate-polygon-scalar.xml', 1)[1].startswith(
        "\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")


def test_subdivide_registered_gates_and_metadata_preflight_are_separate():
    preparation = jobs((ROOT / '.github/workflows/ci.yml').read_text())['numpy-minimum']
    for mode in ('normal', 'scalar'):
        command = ('python -m pytest -q tests/test_subdivide_polygon_integration.py '
                   'tests/test_subdivide_polygon_pipeline.py --junitxml=build/ci-artifacts/'
                   'subdivide-polygon-' + mode + '.xml')
        assert preparation.count(command) == 1
        assert preparation.index(command) < preparation.index('ci_measurement_artifacts.py freeze')
    assert preparation.split('subdivide-polygon-scalar.xml', 1)[1].startswith(
        "\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")
    assert 'tests/test_subdivide_polygon_public.py::test_subdivide_polygon_prospective_manifest' in preparation
    assert 'tests/test_subdivide_polygon_public.py::test_subdivide_polygon_saved_original_rows' in preparation
    assert 'tests/test_subdivide_polygon_benchmark.py' in preparation
    assert 'tests/test_subdivide_polygon_public.py --' not in preparation


def test_historical_subdivision_harness_is_available_for_actual_inverse_guard_jobs():
    blocks=jobs((ROOT / '.github/workflows/ci.yml').read_text())
    target='4f69eba77a262056459eaab1e0a39e53610e9305:benchmarks/subdivide_polygon.py'
    probe="if ! git cat-file -e '"+target+"' 2>/dev/null; then"
    fetch='git fetch --no-tags --depth=1 origin 4f69eba77a262056459eaab1e0a39e53610e9305'
    for name in ('check','numpy-minimum'):
        block=blocks[name]
        assert block.count(probe)==1 and block.count(fetch)==1
        assert block.index(probe)<block.index(fetch)<block.index('python -m pytest')
        assert "git cat-file -e '"+target+"'\n" in block
    assert '- run: python -m pytest -q\n' in blocks['check']
    assert 'tests/test_subdivide_polygon_benchmark.py' in blocks['numpy-minimum']
    # These jobs use selected numerical files or artifact-only measurements,
    # and never collect the historical benchmark inverse guard.
    for name in ('debug-gil','measure','measure-minimum'):
        assert fetch not in blocks[name]



def test_drawing_public_and_pipeline_gates_precede_source_freeze_and_registry_smoke():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    scalar_circle = workflow.split('      - name: Circle perimeter contracts with SIMD disabled', 1)[1].split('      - name:', 1)[0]
    assert 'tests/test_circle_perimeter_public.py' in scalar_circle
    assert "SKIMAGEXX_DISABLE_SIMD: '1'" in scalar_circle
    preparation=jobs((ROOT/'.github/workflows/ci.yml').read_text())['numpy-minimum']
    for mode in ('normal','scalar'):
        command=('python -m pytest -q tests/test_draw_coordinates_public.py '
                 'tests/test_draw_coordinates_integration.py --junitxml=build/ci-artifacts/'
                 'draw-coordinates-'+mode+'.xml')
        assert preparation.count(command)==1
        assert preparation.index(command)<preparation.index('ci_measurement_artifacts.py freeze')
        assert preparation.index(command)<preparation.index('tools/run_feature_tests.py --phase smoke')
    assert preparation.split('draw-coordinates-scalar.xml',1)[1].startswith(
        "\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")
    assert 'tests/test_draw_coordinates_public.py::' not in preparation
    assert '--phase smoke --report build/ci-artifacts/feature-smoke.json' in preparation
    # The shared runner uses the actual inventoried routes: retain test_api's
    # per-function cases and all four dedicated native-pipeline entries.
    inventory=json.loads((ROOT/'coverage/api.json').read_bytes())
    entries={entry['api']:entry for entry in inventory['entries']}
    for api in ('draw.line','draw.rectangle','draw.disk','draw.ellipse'):
        assert entries[api]['implementation']=='native-partial'
        assert entries[api]['parity_test']=='tests/test_draw_coordinates_integration.py::test_draw_captured_public_pipeline'
    feature_source=(ROOT/'tools/run_feature_tests.py').read_text()
    assert "str(ROOT / 'tests/test_api.py')" in feature_source
    contract=json.loads((ROOT/'coverage/contracts/current-public-qualification.json').read_bytes())
    assert contract['scope']=='current258'
    assert len(contract['image_paths'])==258 and len(contract['metadata_paths'])==3
    assert {'tests/test_draw_coordinates_public.py','tests/test_draw_coordinates_integration.py','tests/test_circle_perimeter_contract.py','tests/test_circle_perimeter_public.py','tests/test_line_aa_reference.py','tests/test_line_aa_public.py','tests/test_line_aa_default.py'}<=set(contract['required_selectors'])


def test_equivalent_descriptor_fixture_accepts_every_registered_dtype_before_build():
    # Construction-domain intake only; actual NumPy equivalence stays in the
    # scientific test. This catches a stale three-dtype whitelist pre-build.
    tree = ast.parse((ROOT / 'tests/test_dtype_aliases.py').read_bytes())
    alias = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'alias')

    class Descriptor:
        def newbyteorder(self, order):
            assert order in ('>', '=')
            return self

    namespace = {'np': SimpleNamespace(dtype=lambda name: Descriptor())}
    exec(compile(ast.Module(body=[alias], type_ignores=[]), '<dtype alias intake>', 'exec'), namespace)
    inventory = json.loads((ROOT / 'coverage/api.json').read_bytes())
    names = {dtype for entry in inventory['entries'] for dtype in entry['native_default_case_dtypes']}
    assert names
    for name in names:
        assert isinstance(namespace['alias'](name), Descriptor)



def test_line_nd_saved_defaults_and_typed_policy_are_preparation_gates():
    workflow = (ROOT/'.github/workflows/ci.yml').read_text()
    minimum = jobs(workflow)['numpy-minimum']
    for mode in ('normal','scalar'):
        path = 'build/ci-artifacts/line-nd-'+mode+'.xml'
        assert minimum.count(path) == 1 and minimum.index(path) < minimum.index('ci_measurement_artifacts.py freeze')
    assert minimum.split('line-nd-scalar.xml',1)[1].startswith("\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")
    qualification = (ROOT/'.github/workflows/public-qualification.yml').read_text()
    for selector in ('test_line_nd_reference.py','test_line_nd_rounding_promotion_reference.py','test_line_nd_benchmark.py'):
        assert qualification.index(selector) < qualification.index('tools/public_qualification.py all')
    cmake = (ROOT/'CMakeLists.txt').read_text()
    assert 'set_source_files_properties(src/line_nd.cpp src/bind_line_nd.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")' in cmake
    assert 'src/line_nd.cpp' in cmake and 'src/bind_line_nd.cpp' in cmake
    wrapper = (ROOT/'python/skimagexx/_line_nd.py').read_text()
    assert wrapper.count('(np.float32(0) % 1).dtype') == 1



def test_prebuild_line_nd_selects_only_dependency_free_nodes_and_runtime_keeps_intakes():
    import shlex
    workflow = (ROOT/'.github/workflows/public-qualification.yml').read_text()
    command = next(line.strip() for line in workflow.splitlines()
                   if line.strip().startswith('python -m pytest -q --capture=sys'))
    selected = set(shlex.split(command.split(' 2>&1 | tee ')[0]))
    modules = ('test_regionprops_state_reference.py','test_line_nd_reference.py',
               'test_line_nd_rounding_promotion_reference.py','test_line_nd_benchmark.py')
    assert all('tests/'+module not in selected for module in modules)
    assert {'tests/test_line_nd_reference.py::test_line_nd_prospective_matrix_and_source_safety',
            'tests/test_line_nd_reference.py::test_line_nd_saved_original_rows_without_replay',
            'tests/test_line_nd_rounding_promotion_reference.py::test_line_nd_rounding_promotion_source_safety_without_science',
            'tests/test_line_nd_benchmark.py::test_line_nd_benchmark_import_refuses_scientific_modules'} <= selected
    tree = ast.parse((ROOT/'tools/public_qualification.py').read_bytes())
    runtime = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                       and target.id == 'POST203_TESTS' for target in node.targets))
    assert set(modules) <= set(runtime)


def test_ellipsoid_preparation_and_dependency_free_source_gates():
    minimum=jobs((ROOT/'.github/workflows/ci.yml').read_text())['numpy-minimum']
    for mode in ('normal','scalar'):
        path='build/ci-artifacts/ellipsoid-'+mode+'.xml'
        assert minimum.count(path)==1 and minimum.index(path)<minimum.index('ci_measurement_artifacts.py freeze')
    assert minimum.split('ellipsoid-scalar.xml',1)[1].startswith("\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")
    text=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    assert 'tests/test_ellipsoid_reference.py::test_ellipsoid_prospective_matrix_and_source_safety' in text
    assert 'tests/test_ellipsoid_reference.py::test_ellipsoid_saved_observations_without_replay' in text
    assert 'tests/test_draw_ellipsoid_default.py::test_ellipsoid_saved_sdk_oracle_metadata_without_science' in text
    assert 'tests/test_ellipsoid_reference.py ' not in text
    assert 'tests/test_ellipsoid_benchmark.py ' not in text
    assert 'tests/test_ellipsoid_benchmark.py::test_ellipsoid_benchmark_import_refuses_science' in text
    cmake=(ROOT/'CMakeLists.txt').read_text()
    assert 'set_source_files_properties(src/draw_ellipsoid.cpp src/bind_draw_ellipsoid.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")' in cmake


def test_ellipsoid_generic_benchmark_uses_exact_storage_checker():
    tree=ast.parse((ROOT/'benchmarks/coverage.py').read_bytes())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    branch=next(n for n in ast.walk(main) if isinstance(n,ast.If) and ast.unparse(n.test)=="api == 'draw.ellipsoid'")
    actual,expected,data=object(),object(),object();calls=[]
    def checker(*args):calls.append(args);return 0.
    namespace=dict(actual=actual,expected=expected,data=data,ellipsoid_default_error=checker)
    exec(compile(ast.Module(body=branch.body,type_ignores=[]),'<ellipsoid generic storage>','exec'),namespace)
    assert calls==[(data,actual,expected)] and namespace['error']==0.


def test_ellipse_perimeter_shipping_preparation_and_source_gates():
    blocks=jobs((ROOT/'.github/workflows/ci.yml').read_text())
    minimum=blocks['numpy-minimum']
    assert all('ellipse-perimeter-'+mode+'.xml' in minimum for mode in ('normal','scalar'))
    assert minimum.split('ellipse-perimeter-scalar.xml',1)[1].startswith("\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")
    workflow=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    assert 'tests/test_ellipse_perimeter_reference.py::test_ellipse_perimeter_prospective_source_safety' in workflow
    assert 'tests/test_draw_ellipse_perimeter_default.py::test_ellipse_perimeter_saved_sdk_oracle_metadata_without_science' in workflow
    assert 'tests/test_ellipse_perimeter_benchmark.py::test_ellipse_perimeter_benchmark_factory_original_refused' in workflow
    assert 'tests/test_ellipse_perimeter_benchmark.py ' not in workflow
    assert 'checker_corruption_controls' not in workflow
    cmake=(ROOT/'CMakeLists.txt').read_text()
    assert 'set_source_files_properties(src/draw_ellipse_perimeter.cpp src/bind_draw_ellipse_perimeter.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")' in cmake


def test_region_scalar_installed_preparation_source_and_fixture_gates():
    import hashlib
    blocks=jobs((ROOT/'.github/workflows/ci.yml').read_text())
    for name in ('check','numpy-minimum'):
        job=blocks[name]
        assert 'tests/test_regionprops_bbox_extent_reference.py tests/test_regionprops_bbox_extent_public.py tests/test_regionprops_scalar_boundary_v2.py tests/test_regionprops_scalars_benchmark.py' in job
        assert 'regionprops-scalars-normal.xml' in job and 'regionprops-scalars-scalar.xml' in job
        assert job.split('regionprops-scalars-scalar.xml',1)[1].startswith("\n        env:\n          SKIMAGEXX_DISABLE_SIMD: '1'")
        if name=='numpy-minimum':assert job.index('regionprops-scalars-normal.xml')<job.index('ci_measurement_artifacts.py freeze')
    text=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    for node in ('test_bbox_extent_prospective_source_and_scope','test_bbox_extent_saved_rows_and_declared_pixel_mutation','test_bbox_extent_saved_scalar_types_errors_and_FP_phases'):
        assert 'tests/test_regionprops_bbox_extent_reference.py::'+node in text
    assert 'tests/test_regionprops_scalars_benchmark.py ' in text
    assert 'tests/test_regionprops_bbox_extent_reference.py ' not in text
    cmake=(ROOT/'CMakeLists.txt').read_text()
    assert 'src/regionprops_state.cpp src/bind_regionprops_state.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math;-frounding-math"' in cmake
    assert hashlib.sha256((ROOT/'tests/fixtures/regionprops_scalar_native.bin').read_bytes()).hexdigest()=='51c97708e670151caa78abcfefd64a4dc8145d739645f371f783fbf5327e04d8'


def test_coords_scaled_prebuild_keeps_numpy_intakes_installed():
    import shlex
    workflow=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    command=next(line.strip()for line in workflow.splitlines()
                 if line.strip().startswith('python -m pytest -q --capture=sys'))
    selected=set(shlex.split(command.split(' 2>&1 | tee ')[0]))
    assert 'tests/test_regionprops_coords_scaled_reference.py' not in selected
    assert 'tests/test_regionprops_coords_scaled_public.py' not in selected
    assert not any('test_coords_scaled_controller_intake_originals_refused' in item for item in selected)
    assert {'tests/test_regionprops_coords_scaled_reference.py::test_coords_scaled_prospective_scope_and_source',
            'tests/test_regionprops_coords_scaled_reference.py::test_coords_scaled_saved_sequences_and_accounting',
            'tests/test_regionprops_coords_scaled_reference.py::test_coords_scaled_saved_payload_words_and_declared_mutations',
            'tests/test_regionprops_coords_scaled_sdk_scope.py::test_coords_scaled_sdk_default_and_focused_route',
            'tests/test_regionprops_coords_scaled_sdk_scope.py::test_coords_scaled_reuses_installed_component_family'} <= selected
    assert workflow.index(command)<workflow.index('tools/public_qualification.py all')


def test_coords_scaled_installed_gates_keep_rows_and_dispatch_modes_separate():
    source=(ROOT/'.github/workflows/ci.yml').read_text()
    for job in ('check','numpy-minimum'):
        block=jobs(source)[job]
        steps={step.splitlines()[0].removeprefix('      - name: '):step
               for step in re.split(r'(?=^      - name: )',block,flags=re.M)
               if step.startswith('      - name: ')}
        normal=steps['Region coords scaled saved public and installed SDK boundary normal']
        scalar=steps['Region coords scaled public and installed SDK boundary disabled']
        assert 'tests/test_regionprops_coords_scaled_reference.py ' in normal
        for mode,step in (('normal',normal),('scalar',scalar)):
            assert 'tests/test_regionprops_coords_scaled_public.py ' in step
            assert 'tests/test_regionprops_coords_scaled_sdk_scope.py ' in step
            assert '--junitxml=build/ci-artifacts/regionprops-coords-scaled-'+mode+'.xml' in step
            assert 'SKIMAGEXX_COORDS_SCALED_ROWS_DIR: build/ci-artifacts/regionprops-coords-scaled-'+mode+'-rows' in step
            assert block.index(step)>block.index('pip install')
        assert "SKIMAGEXX_DISABLE_SIMD: '1'" in scalar
        assert 'SKIMAGEXX_DISABLE_SIMD:' not in normal
        if job=='numpy-minimum':
            assert block.index(scalar)<block.index('ci_measurement_artifacts.py freeze')
