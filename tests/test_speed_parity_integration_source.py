"""Literal current speed-parity wiring; source-only, never an epoch ROOT."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 'coverage/contracts/speed-parity-current-source-v38.json'
CONTEXT_SHA256 = '4a8669a042c967b8fe7338b46a00b2e3226637d006a03a20d36343334ca8b725'
RUNTIME = ('tests/test_mark_boundaries_fast_paths.py', 'tests/test_draw_multiblock_lbp_batched.py',
           'tests/test_flood_assignment_protocol.py', 'tests/test_flood_copy_protocol.py',
           'tests/test_flood_disjoint_assignment.py')
SOURCE = ('tests/test_speed_parity_integration_source.py', 'tests/test_flood_speed_source.py',
          'tests/test_ellipsoid_stats_source.py')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def data():
    raw = (ROOT / CONTEXT).read_bytes()
    assert digest(raw) == CONTEXT_SHA256
    value = json.loads(raw)
    assert value['execution_authority'] is None
    return value


def load(name):
    spec = importlib.util.spec_from_file_location('_current_speed_' + Path(name).stem, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selectors(raw):
    lines = [line.strip() for line in raw.decode().splitlines()
             if line.strip().startswith('python -m pytest -q --capture=sys')]
    assert len(lines) == 1 and lines[0].count(' 2>&1 | tee ') == 1
    return [word for word in shlex.split(lines[0].split(' 2>&1 | tee ', 1)[0])
            if word.startswith('tests/')]


@pytest.mark.parametrize('canonical', (False, True))
def test_current_production_and_early_history_are_exact_and_reject_tampering(canonical):
    value = data()
    from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
    helper = load('tools/vi_public_matmul_context.py')
    context = helper._context(ROOT, VI_PUBLIC_MATMUL_CONTEXT_SHA256)
    assert context['historical_source_inputs']['count'] == 2064
    assert context['predecessor_source'] == '2270b7fefc9a8e5cb3ba7fe51d2db5759c17634d'
    for name, pin in value['production_raw_sha256'].items():
        raw = (ROOT / name).read_bytes()
        assert digest(raw) == pin and name in context['sources'], name
        old = helper.predecessor_bytes(ROOT, name, raw, VI_PUBLIC_MATMUL_CONTEXT_SHA256, canonical)
        domain = 'canonical' if canonical else 'raw'
        assert digest(old) == context['sources'][name]['predecessor_' + domain + '_sha256']
        with pytest.raises(AssertionError):
            helper.predecessor_bytes(ROOT, name, raw + b'\n// damaged', VI_PUBLIC_MATMUL_CONTEXT_SHA256, canonical)
        with pytest.raises(AssertionError):
            helper.predecessor_bytes(ROOT, name, old, VI_PUBLIC_MATMUL_CONTEXT_SHA256, canonical)


def test_current_source_and_predecessor_contexts_are_bound_without_authority():
    value = data()
    for name, pin in value['source_raw_sha256'].items():
        assert digest((ROOT / name).read_bytes()) == pin, name
    for name, pin in value['immutable_predecessors'].items():
        assert digest((ROOT / name).read_bytes()) == pin, name
    assert ROOT == Path(__file__).resolve().parents[1]
    assert not any(isinstance(node, ast.ImportFrom) and node.module == 'vi_public_matmul_source_view'
                   for node in ast.walk(ast.parse(Path(__file__).read_bytes())))


def test_actual_source_command_preserves181_and_adds_only_current_source_guards():
    value = data()
    raw = (ROOT / '.github/workflows/public-qualification.yml').read_bytes()
    selected = selectors(raw)
    assert len(selected) == len(set(selected)) == 184
    assert selected == value['previous_source_selectors'] + list(SOURCE)
    assert not set(RUNTIME) & set(selected)


@pytest.mark.parametrize('profile', ('windows', 'minimum', 'modern', 'debug'))
def test_actual_qualification_adds_exact_runtime_groups_and_preserves_old(profile):
    value = data()
    q = load('tools/public_qualification.py')
    contract = q.current_contract(lambda name: (ROOT / name).read_bytes())
    assert q.CURRENT_SCOPE_CONTRACT == 'coverage/contracts/current-public-qualification-v15.json'
    prior = json.loads((ROOT / 'coverage/contracts/current-public-qualification-v8.json').read_bytes())
    assert contract['required_selectors'] == prior['required_selectors'] + list(RUNTIME) + ['tests/test_structural_similarity.py', 'tests/test_local_extrema_public.py',
        'tests/test_order_angles_golden_ratio_integration.py', 'tests/test_pearson_public.py',
        'tests/test_order_angles_golden_ratio_public.py', 'tests/test_hausdorff_public.py',
        'tests/test_ellipsoid_stats_public.py', 'tests/test_expand_labels_public.py', 'tests/test_binary_blobs_public.py',
        'tests/test_gaussian_nd.py', 'tests/test_gaussian_family.py', 'tests/test_resize_gaussian_raw.py']
    groups = q.applicable_public_groups({'public-post203': ('retained',)}, profile)
    assert groups['public-post203'] == ('retained',)
    assert groups['public-mark-fast-paths'] == ('test_mark_boundaries_fast_paths.py',)
    linux = {'public-lbp-batched-linux': ('test_draw_multiblock_lbp_batched.py',),
             'public-flood-assignment-linux': ('test_flood_assignment_protocol.py',),
             'public-flood-copy-linux': ('test_flood_copy_protocol.py',),
             'public-flood-disjoint-linux': ('test_flood_disjoint_assignment.py',)}
    for name, modules in linux.items():
        assert (name not in groups) if profile == 'windows' else groups[name] == modules
    assert contract['speed_parity_runtime'] == value['runtime_policy']
    assert contract['execution_authority'] is None


def test_collector_covers_every_current_input_and_installed_LBP_header():
    value = data()
    q = load('tools/public_qualification.py')
    required = set(value['source_raw_sha256']) | set(RUNTIME) | set(SOURCE) | {CONTEXT}
    assert required <= set(q.DRAW_SOURCE_PATHS)
    cmake = (ROOT / 'tests/cmake_consumer/CMakeLists.txt').read_text()
    assert 'set(_boundary_source draw_multiblock_lbp_enabled_consumer.cpp)' in cmake
    assert 'draw_multiblock_lbp_consumer.hpp' in (ROOT / 'tests/cmake_consumer/draw_multiblock_lbp_consumer.cpp').read_text()


@pytest.mark.parametrize('name', ('mark-boundaries', 'draw-multiblock-lbp'))
def test_current_benchmark_literals_refuse_old_pins_and_preserve_historical_reads(name):
    value = data()
    path = 'coverage/contracts/' + name + '-benchmark-execution-v16.json'
    boundary = load('benchmarks/mark_boundaries_execution.py')
    current = boundary.current_context(ROOT, path)
    predecessor = current['speed_source_successor']['predecessor_context']
    old = json.loads((ROOT / predecessor['path']).read_bytes())
    for key in ('cases', 'bounds', 'method', 'unbound', 'authority_scope', 'FENV_dependency'):
        assert current[key] == old[key], key
    assert all(item is None for item in current['unbound'].values())
    for source, pin in current['source_files_canonical_sha256'].items():
        assert digest((ROOT / source).read_bytes().replace(b'\r\n', b'\n')) == pin, source
    for source, row in current['speed_source_successor']['source_predecessors'].items():
        raw = (ROOT / source).read_bytes()
        with pytest.raises(AssertionError):
            boundary.validation_predecessor_bytes(ROOT, source, raw + b' ', path)
        # Reconstruct the exact prior historical projection without binding it
        # as a product, runtime, source revision or timing authority.
        historical = boundary.validation_predecessor_bytes(ROOT, source, raw, path)
        old_input = row['predecessor_source'].encode()
        assert digest(old_input) == row['predecessor_canonical_sha256']
        expected = boundary.validation_predecessor_bytes(ROOT, source, old_input, predecessor['path'])
        assert historical == expected


@pytest.mark.parametrize('family', ('mark-boundaries', 'draw-multiblock-lbp'))
@pytest.mark.parametrize('fault', (None, 'current', 'snapshot', 'prepared', 'old-context'))
def test_current_benchmark_binding_requires_actual_current_bytes(tmp_path, monkeypatch, family, fault):
    import test_mark_boundaries_benchmark_execution as factory
    monkeypatch.setattr(factory, 'source_fixture_bytes', lambda name, context_path: (ROOT / name).read_bytes())
    path = ROOT / ('coverage/contracts/' + family + '-benchmark-execution-v16.json')
    binding_path, binding, contract = factory.synthetic(tmp_path, monkeypatch, context_path=path)
    boundary = load('benchmarks/mark_boundaries_execution.py')
    relative = path.relative_to(ROOT).as_posix()
    accepted = boundary.validate_binding(binding_path, root=ROOT, context_path=relative)
    assert accepted[0] == binding and accepted[1] == contract
    if fault is None:
        return  # Inert metadata only; no imports of a scientific runtime.
    name = 'src/mark_boundaries.cpp' if family == 'mark-boundaries' else 'src/draw_multiblock_lbp.cpp'
    if fault == 'current':
        read = boundary.source_bytes
        monkeypatch.setattr(boundary, 'source_bytes', lambda p: read(p) + b'bad' if Path(p) == ROOT / name else read(p))
    elif fault == 'snapshot':
        target = Path(binding['source_root']) / name
        target.write_bytes(target.read_bytes() + b'bad')
    elif fault == 'prepared':
        binding['prepared_source']['files_sha256'][name] = '0' * 64
    else:
        binding['base_contract_sha256'] = digest((ROOT / contract['speed_source_successor']['predecessor_context']['path']).read_bytes())
    runtime_path = Path(binding['runtime_path'])
    runtime = json.loads(runtime_path.read_bytes())
    runtime['source'] = binding['prepared_source']
    runtime_path.write_text(json.dumps(runtime), encoding='utf-8')
    binding['runtime_sha256'] = digest(runtime_path.read_bytes())
    binding_path.write_text(json.dumps(binding), encoding='utf-8')
    go_path = Path(binding['root_go_path'])
    go = json.loads(go_path.read_bytes()); go['binding_sha256'] = digest(binding_path.read_bytes())
    go_path.write_text(json.dumps(go), encoding='utf-8')
    with pytest.raises(AssertionError):
        boundary.validate_binding(binding_path, root=ROOT, context_path=relative)


def test_new_groups_use_sourcebuilt_F1_and_distinct_FLOOD_record_pools():
    text = (ROOT / 'tools/public_qualification.py').read_text()
    assert "'public-lbp-batched-linux'" in text
    assert "mode+'-'+group+'-caller-rows'" in text
    for count in (107, 35, 59, 52, 13):
        assert str(count) in text
    q = load('tools/public_qualification.py')
    normal, disabled = q.environment(), q.environment(scalar=True)
    assert 'SKIMAGEXX_DISABLE_SIMD' not in normal and disabled['SKIMAGEXX_DISABLE_SIMD'] == '1'
    cpp = (ROOT / 'src/draw_multiblock_lbp.cpp').read_text()
    assert '!defined(SKIMAGEXX_LBP_SCALAR)' in cpp
    assert data()['mode_scope'] == 'hosted scalar is runtime-disabled; true compiler scalar additionally requires SKIMAGEXX_LBP_SCALAR=1, no reused old VI-only credit'


@pytest.mark.parametrize('previous,current,key', (
    ('find-boundaries-benchmark-binding-v8.json', 'find-boundaries-benchmark-binding-v9.json', 'source_closure'),
    ('find-boundaries-frontend-v7.json', 'find-boundaries-frontend-v8.json', 'source_canonical_pins'),
    ('find-boundaries-sdk-v5.json', 'find-boundaries-sdk-v6.json', 'source_canonical_pins'),
))
def test_find_current_literals_preserve_all_old_recipes_and_refuse_tampering(previous, current, key):
    prior_path = 'coverage/contracts/' + previous
    value = json.loads((ROOT / ('coverage/contracts/' + current)).read_bytes())
    prior = json.loads((ROOT / prior_path).read_bytes())
    assert value['speed_source_successor']['contract'] == prior_path
    assert value['speed_source_successor']['sha256'] == digest((ROOT / prior_path).read_bytes())
    assert value['speed_source_successor']['execution_authority'] is None
    assert {k: v for k, v in value.items() if k not in (key, 'speed_source_successor')} == {k: v for k, v in prior.items() if k not in (key, 'speed_source_successor')}
    assert set(value[key]) == set(prior[key])
    if key == 'source_closure':
        driver = load('benchmarks/find_boundaries.py')
        for name, pin in value[key].items():
            assert digest(driver.binding_source_bytes(name)) == pin, name
        driver.validate_source_closure(value)
        value[key]['src/find_boundaries.cpp'] = '0' * 64
        with pytest.raises(AssertionError):
            driver.validate_source_closure(value)
    else:
        for name, pin in value[key].items():
            assert digest((ROOT / name).read_bytes().replace(b'\r\n', b'\n')) == pin, name


def test_historical_operational_inputs_are_frozen_without_current_authority():
    from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
    helper = load('tools/vi_public_matmul_context.py')
    context = helper._context(ROOT, VI_PUBLIC_MATMUL_CONTEXT_SHA256)
    frozen = context['frozen_operational_inputs']
    assert set(frozen) == {'coverage/work_items.json'}
    for name, text in frozen.items():
        assert name not in context['sources'] and name not in data()['source_raw_sha256']
        raw = text.encode('utf-8'); row = context['historical_source_inputs']['files'][name]
        assert len(raw) == row['bytes']
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == row['git_blob_sha1']


def test_new13_runtime_module_is_current_collected_and_not_a_source_science_selector():
    value=data();name='tests/test_flood_disjoint_assignment.py'
    assert digest((ROOT/name).read_bytes())==value['source_raw_sha256'][name]
    q=load('tools/public_qualification.py')
    assert name in q.DRAW_SOURCE_PATHS
    assert name not in selectors((ROOT/'.github/workflows/public-qualification.yml').read_bytes())
    assert value['runtime_policy']['public-flood-disjoint-linux']==dict(modules=[name],profiles=['minimum','modern','debug'],nodes_per_mode=13,FENV='source-built-full32',caller_records_per_mode=13)
    previous=json.loads((ROOT/'coverage/contracts/current-public-qualification-v9.json').read_bytes())
    for group,policy in previous['speed_parity_runtime'].items():assert value['runtime_policy'][group]==policy


@pytest.mark.parametrize('canonical', (False, True))
def test_CI_runtime_successor_authenticates_exact_current_and_frozen_bytes(canonical):
    from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
    helper = load('tools/vi_public_matmul_context.py')
    context = helper._context(ROOT, VI_PUBLIC_MATMUL_CONTEXT_SHA256)
    assert context['predecessor_history_context'] == {
        'path': 'coverage/contracts/vi-public-matmul-source-history-v14.json',
        'sha256': '16e3ab9c003881f18c072f88f4b2fd940d8915b74e11b59b87b5a80ee31e9715',
        'source': 'e27633bf9186bb7cab5c1d0b9fbf1584c94c6b81',
    }
    for name in ('.github/workflows/ci.yml', 'tests/test_ci_measurement_isolation.py', 'tools/check_cmake_install.py'):
        raw = (ROOT / name).read_bytes()
        previous = helper.predecessor_bytes(ROOT, name, raw, VI_PUBLIC_MATMUL_CONTEXT_SHA256, canonical)
        domain = 'canonical' if canonical else 'raw'
        assert digest(previous) == context['sources'][name]['predecessor_' + domain + '_sha256']
        if name.endswith('ci.yml'):
            assert digest(previous) == '3b194f7e78c2906ef633d6991ba2f573818169eb6d93e81118860395968725f1'
            assert previous.count(b'python -m pip wheel ".[test]" cmake --wheel-dir build/wheels') == 1
            assert previous.count(b'python -m pip install --no-index --find-links build/wheels "scikit-imagexx[test]" cmake') == 1
        with pytest.raises(AssertionError):
            helper.predecessor_bytes(ROOT, name, raw + b'\n# changed', VI_PUBLIC_MATMUL_CONTEXT_SHA256, canonical)
        with pytest.raises(AssertionError):
            helper.predecessor_bytes(ROOT, name, previous, VI_PUBLIC_MATMUL_CONTEXT_SHA256, canonical)


def test_CI_runtime_successor_rejects_changed_predecessor_context(monkeypatch):
    from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
    helper = load('tools/vi_public_matmul_context.py')
    parent = ROOT / 'coverage/contracts/vi-public-matmul-source-history-v14.json'
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda self: original(self) + b' ' if self == parent else original(self))
    with pytest.raises(AssertionError):
        helper._context(ROOT, VI_PUBLIC_MATMUL_CONTEXT_SHA256)


def test_distance_sources_use_the_full_current_source_collector_without_scientific_selection(monkeypatch):
    value = data()
    names = ('src/distance_metrics.hpp', 'src/simd_color_avx2.cpp')
    q = load('tools/public_qualification.py')
    replies = {('status', '--porcelain'): '', ('rev-parse', 'HEAD'): 'a' * 40,
               ('ls-files',): '\n'.join(names)}
    monkeypatch.setattr(q.subprocess, 'check_output',
                        lambda argv, **kwargs: replies[tuple(argv[3:])])
    actual = q.source_state(ROOT)
    assert actual == dict(revision='a' * 40, files_sha256={
        name: digest((ROOT / name).read_bytes().replace(b'\r\n', b'\n')) for name in names})
    for name in names:
        assert digest((ROOT / name).read_bytes()) == value['production_raw_sha256'][name]
    regression = 'tests/test_distance_optimized_layouts.py'
    assert digest((ROOT / regression).read_bytes()) == value['source_raw_sha256'][regression]
    assert regression in q.DRAW_SOURCE_PATHS
    assert regression not in selectors((ROOT / '.github/workflows/public-qualification.yml').read_bytes())


@pytest.mark.parametrize('name,current_sha256', (
    ('tests/array_views_observer.py', 'f963f030efdcfebade82a1af25eab770a3156a3ccaef15416c7861d6df6485e2'),
    ('tests/slice_along_axes_observer.py', '70a6abfd834cc78ea887527ff16f8f74a545957e1578279f74697ce85a310f7d'),
))
def test_current_util_observer_is_exact_and_projects_frozen_reference(name, current_sha256):
    from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
    helper = load('tools/vi_public_matmul_context.py')
    contract = json.loads((ROOT / 'coverage/contracts/util-view-operations-v1.json').read_bytes())
    raw = (ROOT / name).read_bytes()
    assert digest(raw) == current_sha256
    historical = helper.predecessor_bytes(ROOT, name, raw, VI_PUBLIC_MATMUL_CONTEXT_SHA256)
    assert digest(historical) == contract['source_pins_raw_sha256'][name]
    assert historical != raw
    for invalid in (historical, raw + b'\n# changed observer source'):
        with pytest.raises(AssertionError):
            helper.predecessor_bytes(ROOT, name, invalid, VI_PUBLIC_MATMUL_CONTEXT_SHA256)
