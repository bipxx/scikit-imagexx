"""Source and integer-descriptor guards; no scientific or native execution."""
import ast
import builtins
import copy
import hashlib
import itertools
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / 'coverage/contracts/flood-speed-parity-v2.json'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def predecessor(raw, row, canonical=False):
    if canonical:
        raw = raw.replace(b'\r\n', b'\n')
    domain = 'canonical_LF_sha256' if canonical else 'raw_sha256'
    assert digest(raw) == row['current_' + domain]
    for fragment in row['inverse']:
        current, old = (fragment[key].encode() for key in ('current', 'predecessor'))
        assert current and raw.count(current) == 1
        raw = raw.replace(current, old, 1)
    assert digest(raw) == row['predecessor_' + domain]
    return raw


@pytest.mark.parametrize('name', ('src/flood_fill.cpp', 'src/bind_flood_fill.cpp'))
@pytest.mark.parametrize('canonical', (False, True))
def test_exact_main_inverse_refuses_changed_current_old_and_context(name, canonical):
    row = json.loads(CONTEXT.read_bytes())['changed_sources'][name]
    raw = (ROOT / name).read_bytes()
    old = predecessor(raw, row, canonical)
    with pytest.raises(AssertionError):
        predecessor(old, row, canonical)
    with pytest.raises(AssertionError):
        predecessor(raw + b'\n// changed', row, canonical)
    damaged = copy.deepcopy(row)
    damaged['predecessor_raw_sha256'] = damaged['predecessor_canonical_LF_sha256'] = '0' * 64
    with pytest.raises(AssertionError):
        predecessor(raw, damaged, canonical)
    if canonical:
        assert predecessor(raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'), row, True) == old


def test_copy_only_changes_private_initialization_and_keeps_generic_and_queue():
    c = json.loads(CONTEXT.read_bytes())
    raw = (ROOT / 'src/flood_fill.cpp').read_bytes()
    old = predecessor(raw, c['changed_sources']['src/flood_fill.cpp'])
    start = b'    // Upstream stable squared-distance order'
    assert raw[raw.index(start):] == old[old.index(start):]
    assert raw.count(c['generic_initializer'].encode()) == 1
    assert raw[:raw.index(b'    std::vector<T> working')] == old[:old.index(b'    std::vector<T> working')]
    assert c['generic_initializer'].encode() in raw
    block = c['changed_sources']['src/flood_fill.cpp']['inverse'][0]['current']
    assert 'g.strides[axis]==I(sizeof(T))&&out.strides[axis]==1' in block
    assert 'if(a==axis)continue;' in block and 'if(line+1<lines)' in block
    assert 'std::size_t(width)*sizeof(T)' in block
    assert 'std::fill_n(' in block and 'std::memcpy(' in block
    assert all(token not in raw.decode() for token in (
        '#pragma omp', 'get_num_threads', 'set_num_threads', 'PyObject', 'simd'))


def copied_addresses(shape, strides, fortran, itemsize):
    """Integer-only line transport model; not flood or numeric parity."""
    rank = len(shape)
    axis = 0 if fortran else rank - 1
    padded = tuple(n + 2 for n in shape)
    mask_strides = [0] * rank
    count = 1
    for a in (range(rank) if fortran else range(rank - 1, -1, -1)):
        mask_strides[a] = count
        count *= padded[a]
    origin = sum(mask_strides)
    logical = 1
    for n in shape:
        logical *= n
    assert strides[axis] == itemsize and mask_strides[axis] == 1
    index = [0] * rank
    source, target = 0, origin
    result = []
    for line in range(logical // shape[axis]):
        result.extend((source + x * itemsize, target + x) for x in range(shape[axis]))
        if line + 1 < logical // shape[axis]:
            for a in range(rank - 1, -1, -1):
                if a == axis:
                    continue
                index[a] += 1
                if index[a] < shape[a]:
                    source += strides[a]
                    target += mask_strides[a]
                    break
                index[a] = 0
                source -= (shape[a] - 1) * strides[a]
                target -= (shape[a] - 1) * mask_strides[a]
    return result, mask_strides, origin


@pytest.mark.parametrize('shape,strides,fortran,itemsize', (
    ((5,), (1,), False, 1), ((5,), (4,), True, 4),
    ((3, 4), (16, 4), False, 4), ((3, 4), (4, 12), True, 4),
    ((2, 3, 4), (48, 16, 4), False, 4),
    ((2, 3, 4), (4, 8, 24), True, 4),
    ((2, 3, 4), (-48, 16, 4), False, 4),
    ((2, 3, 4), (4, 32, 96), True, 4),
    ((2, 3, 4), (0, -16, 4), False, 4),
    ((1, 4, 1), (16, 4, 4), False, 4),
    ((1, 4, 1), (4, 4, 16), True, 4),
))
def test_line_transport_matches_independent_coordinate_storage(shape, strides, fortran, itemsize):
    actual, mask_strides, origin = copied_addresses(shape, strides, fortran, itemsize)
    expected = [(sum(i * s for i, s in zip(index, strides)),
                 origin + sum(i * s for i, s in zip(index, mask_strides)))
                for index in itertools.product(*(range(n) for n in shape))]
    assert sorted(actual) == sorted(expected)
    assert len({target for _, target in actual}) == len(actual)


def test_assignment_order_and_validation_remain_exact_reviewed_e96():
    c = json.loads(CONTEXT.read_bytes())
    raw = (ROOT / 'src/bind_flood_fill.cpp').read_bytes()
    reviewed = predecessor(raw, c['assignment_widths'])
    assert digest(reviewed) == c['retained_assignment']['binder_raw_sha256']
    assert digest((ROOT / 'tests/test_flood_assignment_protocol.py').read_bytes()) == c['retained_assignment']['protocol_raw_sha256']
    start, end = b'    py::ssize_t selected=0;', b'    if(selected!=values.size())'
    old = predecessor(raw, c['changed_sources']['src/bind_flood_fill.cpp'])
    assert reviewed[:reviewed.index(start)] == old[:old.index(start)]
    assert reviewed[reviewed.index(end):] == old[old.index(end):]


def test_fixed_copy_sizes_keep_exact_ordered_body_and_generic_fallback():
    c = json.loads(CONTEXT.read_bytes())
    raw = (ROOT / 'src/bind_flood_fill.cpp').read_text()
    reviewed = predecessor(raw.encode(), c['assignment_widths']).decode()
    helper, dispatch = c['assignment_widths']['inverse']
    assert raw.count(helper['current']) == raw.count(dispatch['current']) == 1
    body = dispatch['predecessor']
    assert helper['current'] == (
        'template<int Bytes>void assign_values(py::array& output,const py::array& mask,const py::array& values){\n'
        + body.replace('      const auto bytes=output.itemsize();',
                       '      const py::ssize_t bytes=Bytes?Bytes:output.itemsize();') + '}\n')
    for size in (1, 2, 4, 8, 16):
        assert 'case ' + str(size) + ':return assign_values<' + str(size) + '>(output,mask,values);' in dispatch['current']
    assert 'default:return assign_values<0>(output,mask,values);' in dispatch['current']
    anchor = ' m.def("flood_assign",'
    assert raw[raw.index(anchor):raw.index(dispatch['current'])] == reviewed[reviewed.index(anchor):reviewed.index(body)]
    assert all(token not in helper['current'] for token in ('restrict', 'gil_scoped_release', '#pragma omp'))


def test_fixed_cases_sampler_frontend_contract_and_prepared_tests():
    c = json.loads(CONTEXT.read_bytes())
    benchmark = json.loads((ROOT / 'coverage/contracts/flood-fill-benchmark-v3.json').read_bytes())
    assert c['fixed_cases'] == benchmark['routes']['flood-fill']['cases']
    assert c['method'] == benchmark['method'] and len(c['fixed_cases']) == 8
    for name, pin in c['unchanged_raw_sha256'].items():
        raw = (ROOT / name).read_bytes()
        if name == 'benchmarks/flood_fill_execution.py':
            active = json.loads((ROOT / 'coverage/contracts/flood-fill-benchmark-v4.json').read_bytes())
            row = active['sources'][name]
            assert digest(raw.replace(b'\r\n', b'\n')) == row['current_canonical_sha256']
            fragment = row['inverse_fragments'][0]
            assert fragment == dict(current="CONTEXT='coverage/contracts/flood-fill-benchmark-v4.json'",
                                    predecessor="CONTEXT='coverage/contracts/flood-fill-benchmark-v3.json'")
            assert raw.count(fragment['current'].encode()) == 1
            raw = raw.replace(fragment['current'].encode(), fragment['predecessor'].encode(), 1)
        assert digest(raw) == pin, name
    for name in ('tests/test_flood_assignment_protocol.py', 'tests/test_flood_copy_protocol.py'):
        tree = ast.parse((ROOT / name).read_bytes())
        compile(tree, str(ROOT / name), 'exec')
    assert c['authority'] == dict(source_only=True, compile=0, native=0,
                                original=0, NumPy=0, helper=0, timing=0, hosted=0)


def test_prepared_copy_storage_ceiling_covers_actual_frozen_case_descriptors():
    c = json.loads(CONTEXT.read_bytes())
    tree = ast.parse((ROOT / 'tests/test_flood_copy_protocol.py').read_bytes())
    cases = ast.literal_eval(next(node.value for node in tree.body
                                 if isinstance(node, ast.Assign)
                                 and any(isinstance(target, ast.Name) and target.id == 'CASES'
                                         for target in node.targets)))
    maximum = 0
    for shape, layout, _ in cases:
        physical = (tuple(2 * n for n in shape) if layout == 'step' else
                    tuple(n if axis == 0 else 2 * n for axis, n in enumerate(shape))
                    if layout == 'outer-step' else shape)
        size = 4
        for n in physical:
            size *= n
        maximum = max(maximum, size)
    assert maximum == c['prepared_copy']['maximum_input_storage_bytes'] == 384
    assert c['prepared_copy']['nodes'] == 2 * 2 * len(cases) + 4 == 52


def literal_current_production(context, root):
    """The prospective current data uses actual bytes, never a source inverse."""
    for name, pin in context['production_source_canonical_sha256'].items():
        assert digest((root / name).read_bytes().replace(b'\r\n', b'\n')) == pin, name


def test_prospective_V4_has_current_literals_frozen_cases_and_no_authority():
    old_path = ROOT / 'coverage/contracts/flood-fill-benchmark-v3.json'
    old = json.loads(old_path.read_bytes())
    current = json.loads((ROOT / 'coverage/contracts/flood-fill-benchmark-v4.json').read_bytes())
    literal_current_production(current, ROOT)
    for key in ('routes', 'apis', 'method', 'supported_numpy', 'FENV_dependency', 'unbound'):
        assert current[key] == old[key], key
    assert current['schema'] == 'flood-fill-benchmark-v4' and current['state'] == 'UNBOUND'
    assert all(value is None for value in current['unbound'].values())
    assert current['predecessor_contract']['sha256'] == digest(old_path.read_bytes())
    assert current['predecessor_context']['sha256'] == digest(CONTEXT.read_bytes())
    for name, row in current['sources'].items():
        assert digest((ROOT / name).read_bytes().replace(b'\r\n', b'\n')) == row['current_canonical_sha256'], name
    expected_sources = copy.deepcopy(old['sources'])
    row = expected_sources['benchmarks/flood_fill_execution.py']
    row['current_canonical_sha256'] = digest((ROOT / 'benchmarks/flood_fill_execution.py').read_bytes().replace(b'\r\n', b'\n'))
    row['inverse_fragments'].insert(0, dict(current="CONTEXT='coverage/contracts/flood-fill-benchmark-v4.json'",
                                          predecessor="CONTEXT='coverage/contracts/flood-fill-benchmark-v3.json'"))
    assert current['sources'] == expected_sources
    for name, pin in current['FENV_dependency']['source_files_raw_sha256'].items():
        assert digest((ROOT / name).read_bytes()) == pin, name
    activation = current['activation']
    assert activation['status'] == 'CURRENT_EXECUTOR_LOCATOR_ACTIVE_UNBOUND_PENDING_ROOT_RELEASE'
    assert all(activation[key] is None for key in ('genuine_producer', 'current_runtime', 'current_core', 'root_GO'))
    tree = ast.parse((ROOT / activation['executor']).read_bytes())
    locator = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'CONTEXT' for target in node.targets))
    assert locator == activation['current_context'] == 'coverage/contracts/flood-fill-benchmark-v4.json'
    assert activation['required_context'] == 'coverage/contracts/flood-fill-benchmark-v4.json'
    assert activation['source_row_refresh_required_after_locator_change'] is False


@pytest.mark.parametrize('name', ('src/flood_fill.cpp', 'src/bind_flood_fill.cpp'))
def test_prospective_current_literals_refuse_historical_pins_and_changed_source(name, tmp_path):
    current = json.loads((ROOT / 'coverage/contracts/flood-fill-benchmark-v4.json').read_bytes())
    old = json.loads((ROOT / 'coverage/contracts/flood-fill-benchmark-v3.json').read_bytes())
    bad = copy.deepcopy(current)
    bad['production_source_canonical_sha256'][name] = old['production_source_canonical_sha256'][name]
    with pytest.raises(AssertionError):
        literal_current_production(bad, ROOT)
    for source in current['production_source_canonical_sha256']:
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    literal_current_production(current, tmp_path)
    target = tmp_path / name
    target.write_bytes(target.read_bytes() + b'\n// changed source\n')
    with pytest.raises(AssertionError):
        literal_current_production(current, tmp_path)


def test_actual_active_V4_entry_refuses_unbound_and_old_authority_before_science(monkeypatch, tmp_path):
    requests = []
    original_import = builtins.__import__

    def refused(name, *args, **kwargs):
        if name.split('.')[0] in ('numpy', 'scipy', 'skimage', 'skimagexx'):
            requests.append(name)
            raise AssertionError('science request before authority: ' + name)
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as scope:
        scope.setattr(builtins, '__import__', refused)
        active = runpy.run_path(str(ROOT / 'benchmarks/flood_fill_execution.py'))
        assert active['CONTEXT'] == 'coverage/contracts/flood-fill-benchmark-v4.json'
        with pytest.raises(SystemExit, match='PROSPECTIVE_UNBOUND'):
            active['load_binding'](None, ROOT, 'flood-fill')
        path = tmp_path / 'old-binding.json'
        path.write_text(json.dumps(dict(status='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS',
                                       product_kind='GENUINE_INSTALLED_WHOLE_PRODUCT',
                                       base_contract_sha256=digest((ROOT / 'coverage/contracts/flood-fill-benchmark-v3.json').read_bytes()))))
        with pytest.raises(AssertionError):
            active['load_binding'](path, ROOT, 'flood-fill')
    assert requests == []
