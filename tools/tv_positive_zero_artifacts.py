"""Closed af7-to-741 TV comparison admission; standard-library metadata only.

The historical campaign remains immutable. Full products are authenticated by
its unchanged generic verifier; only the explicitly declared TV source/build
slice is compared. Different complete cores and links are never called equal.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = 'coverage/contracts/tv-positive-zero-comparison-plan.json'
PLAN_SHA256 = 'd36b7f63e3b7afeb832fd77c07ffabebaed9c349c4cc04dc802496909c101184'
BEFORE_PRODUCT = 'af7b2afb8f6cd0f30f408e3318631d5c9ffafbee'
AFTER_PRODUCT = '7417676bbb628e75feac1be49aac900bc5af7b15'
EXPERIMENT_SCHEMA = 'tv-positive-zero-experiment-v1'
REPORT_SCHEMA = 'tv-positive-zero-controlled-v1'
RECEIPT_SCHEMA = 'tv-positive-zero-preparation-v1'
CERTIFICATE_LIMIT = 64 * 1024


def require(value, message):
    if not value:
        raise RuntimeError(message)


def execution_environment():
    # Inherited factory/public guards contain assertions; -O removes them.
    require(sys.flags.optimize == 0, 'optimized Python cannot execute comparison guards')
    require(not any(name in sys.modules for name in ('sitecustomize', 'usercustomize')),
            'custom Python startup cannot authenticate this comparison')
    return dict(optimize=sys.flags.optimize, custom_startup=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


execution_environment()
require(sha(ROOT / PLAN_PATH) == PLAN_SHA256, 'prospective source plan changed')
PLAN = read_json(ROOT / PLAN_PATH)
# Authenticate before executing inherited standard-library controller code.
for _name, _record in PLAN['dependency_pins'].items():
    require(text_sha(ROOT / _name) == _record['canonical_lf_sha256'],
            'frozen comparison dependency changed: ' + _name)
_spec = importlib.util.spec_from_file_location('_tv_zero_historical_artifacts',
    ROOT / 'tools/tv_chambolle_admission_artifacts.py')
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

# These functions are role independent. No historical module global is changed.
PROFILES = _legacy.PROFILES
ROLES = _legacy.ROLES
RANK_VALIDATOR_SHA256 = _legacy.RANK_VALIDATOR_SHA256
FROZEN_FACTORY_SHA256 = _legacy.FROZEN_FACTORY_SHA256
DEFAULT_SCHEMA = _legacy.DEFAULT_SCHEMA
DEFAULT_NP_POLICY = _legacy.DEFAULT_NP_POLICY
DEFAULT_BUFFER_ELEMENTS = _legacy.DEFAULT_BUFFER_ELEMENTS
digest = _legacy.digest
save_new = _legacy.save_new
revision = _legacy.revision
rank_controller = _legacy.rank_controller
default_rule = _legacy.default_rule
validate_default_state = _legacy.validate_default_state
pack_default_observation = _legacy.pack_default_observation
unpack_default_observation = _legacy.unpack_default_observation
validate_default_dispatch = _legacy.validate_default_dispatch
ingest_default_reply = _legacy.ingest_default_reply
validate_default_checks = _legacy.validate_default_checks
fetch_product = _legacy.fetch_product
verify_controller_checkout = _legacy.verify_controller_checkout


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _scope(role):
    scope = PLAN['scopes'][role]
    return {name: copy.deepcopy(scope[name]) for name in
            ('image_paths', 'metadata_paths', 'sdk_families', 'sdk_members', 'sdk_family_names')}


def validate_experiment(value, controller_revision):
    fields = {'schema', 'status', 'controller_revision', 'before_product_source',
        'after_product_source', 'original_version', 'rounds', 'workers',
        'case_count', 'timed_count', 'factory_sha256', 'manifest_sha256',
        'plan_sha256', 'root_before_admission', 'root_after_admission', 'products'}
    require(type(value) is dict and set(value) == fields, 'closed experiment fields differ')
    require(value['schema'] == EXPERIMENT_SCHEMA and value['status'] == 'root-reviewed',
            'root-reviewed successor experiment required')
    require(type(controller_revision) is str and re.fullmatch('[0-9a-f]{40}', controller_revision)
            and value['controller_revision'] == controller_revision, 'wrong frozen controller')
    require(value['before_product_source'] == BEFORE_PRODUCT
            and value['after_product_source'] == AFTER_PRODUCT, 'literal product roles differ')
    require(value['plan_sha256'] == PLAN_SHA256, 'different source policy')
    require(type(value['root_before_admission']) is dict
            and value['root_before_admission'].get('passed') is True
            and value['root_before_admission'] == PLAN['old_certificate']['root_admission'],
            'old AFTER admission cannot be relabeled or replaced')
    require(value['original_version'] == '0.26.0', 'fixed original required')
    for name, expected in {'rounds': 7, 'workers': 1, 'case_count': 400, 'timed_count': 70}.items():
        require(type(value[name]) is int and value[name] == expected, 'fixed method differs: ' + name)
    require(value['factory_sha256'] == FROZEN_FACTORY_SHA256
            and value['manifest_sha256'] == sha(ROOT / 'coverage/contracts/tv-chambolle-benchmark.json'),
            'frozen400/70 corpus changed')
    admission = value['root_after_admission']
    require(type(admission) is dict and set(admission) == {'source', 'passed', 'sha256', 'run', 'profiles'},
            'AFTER genuine four-profile root admission pending')
    require(admission['source'] == AFTER_PRODUCT and admission['passed'] is True
            and _hash(admission['sha256']), 'AFTER root admission pending')
    require(type(admission['run']) is str and admission['run'].isdigit()
            and admission['run'] != PLAN['failed_after_run'], 'failed original741 run cannot admit AFTER')
    require(type(admission['profiles']) is dict
            and set(admission['profiles']) == {*PROFILES, 'debug'}, 'all four qualified profiles required')
    for record in admission['profiles'].values():
        require(type(record) is dict and set(record) == {'passed', 'qualification_sha256'}
                and record['passed'] is True and _hash(record['qualification_sha256']),
                'failed/incomplete qualification profile')
    products = value['products']
    require(type(products) is dict and set(products) == {'before', 'after'}, 'both actual products required')
    require(products['before'] == PLAN['old_certificate']['products'],
            'BEFORE must be byte-equivalent old-AFTER frames; historical records remain unchanged')
    c = rank_controller()
    for role, source in (('before', BEFORE_PRODUCT), ('after', AFTER_PRODUCT)):
        require(type(products[role]) is dict and set(products[role]) == set(PROFILES),
                'all three product profiles required')
        ids = []
        for profile, frame in products[role].items():
            require(type(frame) is dict and set(frame) == {'pin', 'artifact', 'core_sha256'},
                    'complete qualified frame required')
            pin = c.accepted_pin({'accepted_products': {role: {profile: frame['pin']}}}, role, profile)
            require(pin['revision'] == source and pin['artifact_name'] == 'rank-statistics-product-' + profile,
                    'qualified source/artifact differs')
            require(pin['scope'] == _scope(role)
                    and all(type(pin['scope'][k]) is int for k in
                        ('image_paths', 'metadata_paths', 'sdk_families', 'sdk_members')),
                    'exact role-specific scope required')
            require(_hash(frame['core_sha256']), 'actual owned core pin missing')
            artifact = frame['artifact']
            require(type(artifact) is dict and set(artifact) == {'id', 'digest', 'run', 'name'}
                    and type(artifact['id']) in (int, str) and str(artifact['id']).isdigit()
                    and int(artifact['id']) > 0 and _hash(artifact['digest'])
                    and artifact['run'] == pin['artifact_run'] and artifact['name'] == pin['artifact_name'],
                    'raw outer artifact provenance differs')
            ids.append(str(artifact['id']))
            if role == 'after':
                require(pin['artifact_run'] == admission['run']
                        and pin['qualification_sha256'] == admission['profiles'][profile]['qualification_sha256'],
                        'AFTER frame is not the root-admitted qualification')
        require(len(set(ids)) == len(PROFILES), 'distinct profile artifacts required')
    return value


def load_experiment(path, expected_sha, controller_revision=None):
    execution_environment()
    path = Path(path)
    require(_hash(expected_sha) and 0 < path.stat().st_size <= CERTIFICATE_LIMIT,
            'bounded exact external experiment required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'external experiment raw SHA differs')
    selected = controller_revision or revision()
    verify_controller_checkout(selected)
    return validate_experiment(json.loads(raw), selected)


def product_config(experiment):
    c = rank_controller()
    cfg = c.config()
    cfg['accepted_products'] = {r: {p: experiment['products'][r][p]['pin'] for p in PROFILES}
                               for r in ('before', 'after')}
    return cfg


def verify_product(path, role, profile, experiment):
    """New process/product boundary: retain the entire old per-product verifier."""
    c = rank_controller()
    result = c.verify_product(path, role, profile, product_config(experiment))
    path = Path(path)
    package = read_json(path / 'qualification-evidence/package.json')
    qualified = read_json(path / 'qualification.json')
    record = qualified['records']['package.json']
    require(sha(path / 'qualification-evidence/package.json') == record['sha256']
            and (path / 'qualification-evidence/package.json').stat().st_size == record['bytes'],
            'qualified physical package receipt differs')
    scope = PLAN['scopes'][role]
    require(package['wheel_sha256'] == experiment['products'][role][profile]['pin']['wheel_sha256']
            and len(package['members_sha256']) == scope['wheel_members'], 'physical wheel census differs')
    cores = {n: h for n, h in package['members_sha256'].items()
             if n.startswith('skimagexx/_core.') and n.endswith(('.so', '.pyd'))}
    require(len(cores) == 1 and list(cores.values()) == [experiment['products'][role][profile]['core_sha256']],
            'sole qualified core differs')
    require(qualified['expected_registry']['registered_count'] == scope['registered_count']
            and qualified['SDK_checks'] == scope['sdk_checks'], 'role registry/SDK check census differs')
    return result


def verify_source_slice(before, after):
    c = rank_controller()
    sources = {'before': Path(before), 'after': Path(after)}
    raw_indices = {r: c.tree(path) for r, path in sources.items()}
    canonical = {r: {name: text_sha(path / name) for name in raw_indices[r]}
                 for r, path in sources.items()}
    changed = {n for n in canonical['before'].keys() | canonical['after'].keys()
               if canonical['before'].get(n) != canonical['after'].get(n)}
    expected = {n for n, row in PLAN['full_source_delta'].items()
                if (row['before'] or {}).get('canonical_lf_sha256') != (row['after'] or {}).get('canonical_lf_sha256')}
    require(changed == expected, 'complete qualified source delta differs from reviewed revisions')
    for name, row in PLAN['full_source_delta'].items():
        for role, pin in row.items():
            require(canonical[role].get(name) == (pin or {}).get('canonical_lf_sha256'),
                    'unreviewed source delta bytes: ' + name)
    for name, row in PLAN['tv_source_slice'].items():
        for role, pin in row.items():
            require(canonical[role][name] == pin['canonical_lf_sha256'], 'TV source slice differs: ' + name)
    left = (sources['before'] / 'src/tv_denoise.cpp').read_bytes().replace(b'\r\n', b'\n')
    right = (sources['after'] / 'src/tv_denoise.cpp').read_bytes().replace(b'\r\n', b'\n')
    old = PLAN['helper_inverse']['before_body'].encode()
    new = PLAN['helper_inverse']['after_body'].encode()
    require(right.count(new) == left.count(old) == 1 and right.replace(new, old, 1) == left,
            'whole TV file inverse differs outside zero_bytes')
    return dict(complete_canonical_changed=sorted(changed),
                complete_raw_indices=raw_indices,
                tv_slice=list(PLAN['tv_source_slice']), helper_only_inverse=True)


def build_slice(identity):
    """Read authenticated actual TUs; retain third-party raw and project LF pins."""
    c = rank_controller()
    facts = identity['raw']
    roots = c._build_roots(facts)
    names = PLAN['relevant_translation_units']
    units = {}
    for unit in facts['translation_units']:
        name = c._below_build_root(unit['source'], facts['source_root'])
        if name not in names:
            continue
        require(name not in units, 'duplicate selected TV translation unit')
        dependencies = {}
        for path in unit['dependencies']:
            if c._absolute_build_path(path) == c._absolute_build_path(unit['source']):
                continue
            record = identity['file_records']['headers'][path]
            project = c._below_build_root(path, facts['source_root'])
            key = c._header_name(path, facts['header_roots'])
            require(key not in dependencies, 'duplicate selected header identity')
            dependencies[key] = ({'canonical_lf_sha256': record['canonical_lf_sha256']}
                if project is not None and project != '' else
                {'sha256': record['sha256'], 'bytes': record['bytes']})
        units[name] = dict(target=unit['target'], argv=[c._normalize_build_argument(v, roots) for v in unit['argv']],
                           dependencies=dependencies)
    require(set(units) == set(names), 'complete actual TV translation-unit slice required')
    return dict(configuration=facts['configuration'], binaries=identity['common']['binaries'], units=units)


def reference_runtime(runtime):
    state = copy.deepcopy(runtime['runtime'])
    state['versions'].pop('scikit-imagexx')
    state['packages_sha256'].pop('skimagexx')
    state.pop('sdk_sha256')
    wheels = {name: row for name, row in runtime['wheels'].items() if row['name'] != 'scikit-imagexx'}
    return dict(state=state, wheels=wheels)


def verify_pair(before, after, profile, experiment):
    products = {'before': Path(before), 'after': Path(after)}
    verified = {role: verify_product(path, role, profile, experiment) for role, path in products.items()}
    identities = {role: read_json(path / 'build-identity.json') for role, path in products.items()}
    slices = {role: build_slice(value) for role, value in identities.items()}
    require(slices['before'] == slices['after'], 'TV tools/actual headers/compile flags/configuration differ')
    runtimes = {role: reference_runtime(read_json(path / 'runtime.json')) for role, path in products.items()}
    require(runtimes['before'] == runtimes['after'], 'actual original/provider runtime differs')
    source = verify_source_slice(products['before'] / 'source', products['after'] / 'source')
    return dict(schema='tv-positive-zero-source-comparison-v1', products=verified,
                common_sha256=digest(slices['before']), tv_build_slice=slices['before'], source=source,
                full_build_identities=identities,
                attribution=PLAN['attribution'], whole_core_equivalent=False,
                helper_causal_attribution=False)


def validate_receipt(receipt, experiment_sha, generation, attempt):
    require(receipt.get('schema') == RECEIPT_SCHEMA, 'successor preparation schema required')
    # The old validator's complete method/default/barrier checks stay exact;
    # only a local schema view is supplied, never a modified historical record.
    view = dict(receipt, schema='tv-admission-preparation-v2')
    _legacy.validate_receipt(view, experiment_sha, generation, attempt)
    return receipt


def barrier(paths, experiment_sha, generation, attempt):
    require(len(paths) == 3, 'all three fresh preparation receipts required')
    rows = {}
    for path in paths:
        receipt = validate_receipt(read_json(path), experiment_sha, generation, attempt)
        require(receipt['profile'] not in rows, 'duplicate profile')
        rows[receipt['profile']] = receipt
    require(set(rows) == set(PROFILES), 'missing preparation profile')
    return rows
