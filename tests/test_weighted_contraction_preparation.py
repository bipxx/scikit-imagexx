"""Inert preparation and workflow controls: no compiler, helper or scientific calls."""
import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/diagnostics/prepare_weighted_contraction.py'
spec = importlib.util.spec_from_file_location('weighted_preparation_under_test', SOURCE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def host():
    return dict(system='Linux', machine='x86_64', os={'ID': 'ubuntu', 'VERSION_ID': '26.04'},
                glibc='glibc 2.43', python='3.11.17', byteorder='little')


def command(root, **overrides):
    row = dict(directory=str(root / 'build/native'), file=str(root / 'src/distance.cpp'),
               arguments=['/usr/bin/c++', '-DSKIMAGEXX_AVX2', '-I' + str(root / 'include'),
                          '-O3', '-DNDEBUG', '-fPIC', '-Wall', '-Wextra', '-Wpedantic',
                          '-fopenmp', '-std=gnu++17', '-o', 'distance.cpp.o', '-c',
                          str(root / 'src/distance.cpp')])
    row.update(overrides)
    return row


def test_unbound_refuses_before_path_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('UNBOUND must not inspect or allocate paths')
    monkeypatch.setattr(prep.Path, 'resolve', forbidden)
    with pytest.raises(RuntimeError, match='UNBOUND'):
        prep.main([])


@pytest.mark.parametrize('revision', [None, 'main', '0' * 39, 'A' * 40])
def test_exact_revision_required_before_setup(revision):
    args = ['--authority', prep.AUTHORITY]
    if revision is not None:
        args += ['--expected-revision', revision]
    with pytest.raises(RuntimeError, match='Exact revision'):
        prep.main(args)


def test_exact_host_accepted_without_any_runtime_query():
    prep.check_host(host())


@pytest.mark.parametrize('field,value', [('system', 'Windows'), ('machine', 'aarch64'),
    ('glibc', 'glibc 2.39'), ('python', '3.11.14'), ('byteorder', 'big'),
    ('os', {'ID': 'ubuntu', 'VERSION_ID': '24.04'})])
def test_wrong_host_refuses(field, value):
    facts = host(); facts[field] = value
    with pytest.raises(RuntimeError, match='Required Ubuntu26'):
        prep.check_host(facts)


def test_setup_python_loader_preserved_exactly(tmp_path):
    (tmp_path / 'lib').mkdir()
    loader = str(tmp_path.resolve() / 'lib')
    result = prep.check_environment({'LD_LIBRARY_PATH': loader, 'PYTHONOPTIMIZE': '0'}, tmp_path)
    assert result['LD_LIBRARY_PATH'] == result['allowed_loader_path'] == loader


@pytest.mark.parametrize('loader', ['lib', '/wrong/interpreter/lib', '/good/lib:/extra/lib'])
def test_wrong_loader_refuses(tmp_path, loader):
    with pytest.raises(RuntimeError, match='exact interpreter'):
        prep.check_environment({'LD_LIBRARY_PATH': loader}, tmp_path)


@pytest.mark.parametrize('name', ['CXXFLAGS', 'CMAKE_ARGS', 'LD_PRELOAD', 'GLIBC_TUNABLES',
                                'NPY_DISABLE_CPU_FEATURES', 'SKIMAGEXX_DISABLE_SIMD'])
def test_flag_and_feature_injection_refuses(tmp_path, name):
    with pytest.raises(RuntimeError, match='injection'):
        prep.check_environment({name: 'injected'}, tmp_path)


def test_capacity_exact_boundary_and_shortfall():
    required = 30 * 1024**3
    record = prep.check_capacity({'workspace': required, 'temp': required})
    assert record['required_free_bytes'] == required
    assert record['estimated_peak_bytes'] == 4 * 1024**3 and record['concurrent_bytes'] == 0
    with pytest.raises(RuntimeError, match='capacity'):
        prep.check_capacity({'workspace': required, 'temp': required - 1})


def test_actual_minimum_full_runtime_tuple_literal_only():
    wheels, versions, lock = prep.runtime_lock((ROOT / 'tools/public_qualification.py').read_text(encoding='utf-8'))
    assert len(wheels) == len(versions) == len(lock.splitlines()) == 13
    assert versions['numpy'] == '1.24.4' and versions['scipy'] == '1.15.3'
    assert all('--hash=sha256:' in line for line in lock.splitlines())


def test_nonliteral_runtime_is_never_executed():
    with pytest.raises((ValueError, RuntimeError)):
        prep.runtime_lock('RUNTIME_WHEELS = forbidden_scientific_call()')
    with pytest.raises(RuntimeError, match='tuple'):
        prep.runtime_lock("RUNTIME_WHEELS = {'minimum': {}}")


def test_compiler_commands_preserve_scalar_recipe_and_sole_delta(tmp_path):
    row = command(tmp_path)
    cpp, output = tmp_path / 'wrapper.cpp', tmp_path / 'products'
    actual_row, original, variants = prep.variant_commands([row], tmp_path, cpp, output)
    assert actual_row == row and original == row['arguments']
    baseline, strict = variants['baseline'], variants['strict']
    expected = list(row['arguments'])
    expected[expected.index('-o') + 1] = str(output / 'baseline.so')
    expected[expected.index('-c') + 1] = str(cpp)
    expected.remove('-c')
    expected += ['-iquote', str(tmp_path / 'src'), '-shared']
    assert baseline == expected
    assert '-c' not in baseline and '-shared' in baseline
    assert baseline[baseline.index('-o') + 1] == str(output / 'baseline.so')
    assert str(cpp) in baseline and baseline[-3:] == ['-iquote', str(tmp_path / 'src'), '-shared']
    normalized = [x.replace(str(output / 'strict'), str(output / 'baseline')) for x in strict[:-1]]
    assert normalized == baseline and strict[-1] == '-ffp-contract=off'
    for token in row['arguments'][1:12]:
        assert token in baseline
    assert not any(x.startswith(('-march', '-mfma', '-ffast-math')) for x in baseline)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'off', 'response', 'wrong-source'])
def test_ambiguous_or_changed_recipe_refuses(tmp_path, fault):
    row = command(tmp_path)
    rows = [row]
    if fault == 'missing':
        rows = []
    elif fault == 'duplicate':
        rows += [dict(row)]
    elif fault == 'off':
        row['arguments'].append('-ffp-contract=off')
    elif fault == 'response':
        row['arguments'].append('@uncaptured.rsp')
    else:
        row['arguments'][-1] = str(tmp_path / 'other.cpp')
    with pytest.raises(RuntimeError):
        prep.variant_commands(rows, tmp_path, tmp_path / 'wrapper.cpp', tmp_path / 'products')


def test_dependency_outputs_are_separate_from_compile_topology(tmp_path):
    row = command(tmp_path)
    row['arguments'] += ['-MD', '-MF', 'old.d', '-MT', 'old.o']
    _, _, variants = prep.variant_commands([row], tmp_path, tmp_path / 'wrapper.cpp', tmp_path / 'products')
    headers = prep.header_command(variants['baseline'])
    assert '-o' not in headers and '-shared' not in headers and '-MD' not in headers and '-MF' not in headers
    assert headers[-3:] == ['-M', '-MT', 'weighted_hue_graph_v1']
    assert '-O3' in headers and '-fopenmp' in headers


def test_codegen_absence_and_both_boolean_results():
    assert prep.contraction('0000 <weighted_hue_graph_v1>:\n  0: mulsd %xmm1,%xmm0') is False
    assert prep.contraction('0000 <weighted_hue_graph_v1>:\n  0: vfmadd132sd %xmm1,%xmm2,%xmm0') is True
    with pytest.raises(RuntimeError, match='disassembly absent'):
        prep.contraction('unrelated symbol')


def workflow_invariants(text):
    assert 'matrix:' not in text and text.count('runs-on: ubuntu-latest') == 1
    assert "python-version: '3.11.17'" in text
    assert text.index('Check actual host and scratch capacity before allocation') < text.index('actions/checkout@')
    assert "host.get('VERSION_ID') == '26.04'" in text and "record['glibc'] == '2.43'" in text
    assert '4*1024**3' in text and '20*1024**3' in text and '30*1024**3' in text
    assert text.index('prepare_weighted_contraction.py') < text.index("command = json.loads")
    assert prep.AUTHORITY in text and "result = subprocess.run(command['argv']" in text
    assert 'if: always()' in text and 'body-terminal.json' in text
    assert 'python -m pytest' not in text and 'benchmarks/' not in text and 'public-qualification' not in text


def test_actual_one_host_workflow_and_order_mutations():
    text = (ROOT / '.github/workflows/distance-overlap-diagnostics.yml').read_text(encoding='utf-8')
    workflow_invariants(text)
    with pytest.raises(AssertionError):
        workflow_invariants(text.replace('runs-on: ubuntu-latest', 'runs-on: ubuntu-latest\n    matrix: {}'))
    checkout = '      - uses: actions/checkout@v4\n'
    moved = text.replace(checkout, '').replace(
        '      - name: Check actual host and scratch capacity before allocation',
        checkout + '      - name: Check actual host and scratch capacity before allocation')
    with pytest.raises(AssertionError):
        workflow_invariants(moved)
    with pytest.raises(AssertionError):
        workflow_invariants(text.replace('prepare_weighted_contraction.py', 'PREP_PLACEHOLDER')
                            + '\nprepare_weighted_contraction.py\n')


def test_preparation_has_no_scientific_import_or_body_dispatch():
    source = SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(source)
    imported = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name.split('.')[0] in {'numpy', 'scipy', 'skimage', 'skimagexx'} for name in imported)
    assert all(flag in source for flag in ('--require-hashes', '--only-binary=:all:', '--no-deps', '--no-cache-dir'))
    assert "literal_assignment(source, 'RUNTIME_WHEELS')" in source
    assert "fenv.validate_receipt(receipt, sha(receipt))" in source
    assert 'fenv.backend(' not in source and 'ctypes.CDLL' not in source
    assert "status='PREPARED_BODY_NOT_EXECUTED'" in source
    assert "status='FAILED_PREPARATION_ZERO_BODY'" in source
    assert "'--binding-sha256', digest" in source and 'seconds=1800' in source
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {'get_dH2', 'weighted_hue_graph_v1', 'fegetenv', 'reference_fenv_restore'}
                   for node in ast.walk(tree))
