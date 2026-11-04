"""Source-discovery metadata only; no image algorithm imports or execution."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/regionprops-runtime-surface.json'


def test_regionprops_witness_integrity_and_no_execution_claim():
    record = json.loads(CONTRACT.read_bytes())
    assert record['status'] == 'SOURCE_DISCOVERY_ONLY_NO_NUMERICAL_OR_NATIVE_ACCEPTANCE'
    assert record['reference_version'] == '0.26.0'
    assert all(value == 0 for value in record['execution'].values())
    for witness in record['source_witnesses'].values():
        source = witness['source']
        assert hashlib.sha256(source.replace('\r\n', '\n').encode()).hexdigest() == witness['canonical_lf_sha256']
        ast.parse('def witness():\n' + source if source.startswith('    ') else source)


def test_regionprops_alias_and_availability_match_literal_source():
    record = json.loads(CONTRACT.read_bytes())
    witnesses = record['source_witnesses']
    aliases = ast.literal_eval(ast.parse(witnesses['PROPS']['source']).body[0].value)
    required = ast.literal_eval(ast.parse(witnesses['_require_intensity_image']['source']).body[0].value)
    node = ast.parse(witnesses['COL_DTYPES']['source']).body[0].value
    dtypes = {ast.literal_eval(k): v.id for k, v in zip(node.keys, node.values)}
    properties = {prop['name']: prop for prop in record['properties']}
    assert len(aliases) == 61 and len(properties) == 44
    assert record['aliases'] == aliases and set(properties) == set(dtypes)
    assert set(aliases.values()) <= set(properties)
    assert record['intensity_required'] == sorted(required)
    assert record['noniterated_canonical'] == ['coords_scaled', 'intensity_median', 'num_pixels']
    for name, prop in properties.items():
        assert prop['table_dtype'] == dtypes[name]
        assert prop['requires_intensity'] == (name in required)
        assert prop['iterated_with_intensity'] == (name in aliases.values())
        assert prop['iterated_without_intensity'] == (name in aliases.values() and name not in required)


def test_regionprops_descriptors_state_and_cache_match_class_source():
    record = json.loads(CONTRACT.read_bytes())
    cls = ast.parse(record['source_witnesses']['RegionProperties']['source']).body[0]
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    descriptors = {name: n for name, n in methods.items()
                   if any(isinstance(d, ast.Name) and d.id == 'property' for d in n.decorator_list)}
    fields = {n.attr for n in ast.walk(methods['__init__'])
              if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
              and isinstance(n.value, ast.Name) and n.value.id == 'self' and not n.attr.startswith('_')}
    assert fields == {'label', 'slice'} and len(descriptors) == 42
    assert {p['name'] for p in record['properties']} == set(descriptors) | fields
    assert set(record['custom_data_model_methods']) == {name for name in methods if name.startswith('__')}
    for prop in record['properties']:
        node = descriptors.get(prop['name'])
        names = set() if node is None else {d.id for d in node.decorator_list if isinstance(d, ast.Name)}
        assert prop['cached'] == ('_cached' in names)
        assert prop['only_2d'] == ('only2d' in names)
        assert prop['kind'] == ('instance_field' if node is None else 'property')
    assert sum(p['cached'] for p in record['properties']) == record['summary']['cached_descriptors']
    assert sum(p['only_2d'] for p in record['properties']) == record['summary']['only_2d_descriptors']


def test_regionprops_routes_close_discovery_without_duplicate_class_inventory():
    record = json.loads(CONTRACT.read_bytes())
    inventory = json.loads((ROOT / 'coverage/api.json').read_bytes())
    classes = [entry for entry in inventory['entries'] if entry['kind'] == 'class']
    assert len(classes) == 29 and sum(len(entry['members']) for entry in classes) == 284
    assert not any(entry['api'].endswith('RegionProperties') for entry in classes)
    names = [name for route in record['native_routes'].values() for name in route['properties']]
    assert len(names) == len(set(names)) == 44
    assert set(names) == {p['name'] for p in record['properties']}
    assert all(route['prerequisites'] and route['route'] for route in record['native_routes'].values())
    partitions = record['proposed_contract_partitions']
    assert len({part['id'] for part in partitions}) == 10
    assert all(part['status'] == 'proposed-not-executed' and part['cases'] and part['gate'] for part in partitions)
    assert record['unassessed'] and record['next_gate']['owner'] == 'root'
