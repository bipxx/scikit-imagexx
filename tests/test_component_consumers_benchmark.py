"""Harness-only delegated-original stubs never constitute public native proof."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile
import numpy as np
import pytest
from benchmarks import component_consumers as h


def forbidden(*args, **kwargs): raise AssertionError('untimed/failing consumer run reached forbidden work')


def unit_cases():
    for index, api in enumerate(h.APIS):
        yield h.Case(api, 'unit-only', dict(dtype='bool' if index == 1 else 'i2'), True, 'raster', 'unit-only')


def install_unit_stubs(monkeypatch, tmp_path, bad=False):
    originals = {api: h.resolve(api, True) for api in h.APIS}
    cases = list(unit_cases()); original_observe = h.observe_case
    rows = []
    for c in cases:
        observed = original_observe(originals[c.api], c)
        rows.append(dict(id=[c.api, c.id], classification=c.kind, observation_sha256=h.digest(observed)))
    matrix = dict(cases=[dict(api=c.api, case=c.id) for c in cases], reference_modules=[], profiles={h.profile(): dict(
        numpy=np.__version__, scipy=h.scipy.__version__, case_spec_sha256=h.digest([h.case_spec(c) for c in cases]), cases=rows)})
    path = tmp_path / 'unit-matrix.json'; path.write_text(json.dumps(matrix))
    monkeypatch.setattr(h, 'MANIFEST', path); monkeypatch.setattr(h, 'corpus', unit_cases)
    def resolve(api, original=False):
        if original: return originals[api]
        def stub(*args, **kwargs):
            result = originals[api](*args, **kwargs)
            if bad and api == h.APIS[-1]: result[0].flat[0] = 99
            return result
        stub.__module__ = 'skimagexx.unit_harness_stub'
        return stub
    monkeypatch.setattr(h, 'resolve', resolve)
    monkeypatch.setattr(h, 'native_functions', lambda: h.APIS)
    monkeypatch.setattr(h.xx._core, 'component_consumers_interface_version', 1, raising=False)
    implementation = SimpleNamespace(_core=h.xx._core)
    monkeypatch.setitem(sys.modules, 'skimagexx._component_consumers', implementation)
    monkeypatch.setattr(h.xx, '_component_consumers', implementation, raising=False)
    # A semantic harness simulation, deliberately bypassing native class provenance.
    monkeypatch.setattr(h, 'observe_case', lambda fn, case, native=False: original_observe(fn, case, False))
    monkeypatch.setattr(h, 'verify', lambda *args: {'unit_harness_stub_only': True})
    monkeypatch.setattr(h, 'seal', lambda *args: {'unit_harness_stub_only': True})


def test_reference_only_cannot_call_native_or_timer(monkeypatch, tmp_path):
    real = h.resolve
    monkeypatch.setattr(h, 'resolve', lambda api, original=False: real(api, True) if original else forbidden())
    monkeypatch.setattr(h, 'measure', forbidden); monkeypatch.setattr(time, 'perf_counter_ns', forbidden)
    monkeypatch.setattr(h, 'corpus', unit_cases)
    path = tmp_path / 'matrix.json'; path.write_text(json.dumps(dict(cases=[dict(api=c.api, case=c.id) for c in unit_cases()], profiles={}, reference_modules=[])))
    monkeypatch.setattr(h, 'MANIFEST', path)
    output = tmp_path / 'reference.json'
    assert h.main(['--reference-only', '--check-only', '--output', str(output)]) == 0
    r = json.loads(output.read_text()); assert r['status'] == 'complete-reference-check' and not r['results']


def test_missing_registry_precedes_capsule_fixtures_and_timer(monkeypatch, tmp_path):
    monkeypatch.setattr(h, 'native_functions', lambda: ())
    monkeypatch.setattr(h, 'corpus', forbidden); monkeypatch.setattr(h, 'verify', forbidden); monkeypatch.setattr(h, 'measure', forbidden)
    out = tmp_path / 'missing.json'
    assert h.main(['--capsule', str(tmp_path), '--check-only', '--output', str(out)]) == 1
    r = json.loads(out.read_text()); assert not r['checks'] and not r['results'] and 'not registered' in r['failures'][0]['message']


def test_forwarding_rejected_before_fixtures(monkeypatch, tmp_path):
    install_unit_stubs(monkeypatch, tmp_path)
    real = h.resolve; monkeypatch.setattr(h, 'resolve', lambda api, original=False: real(api, True))
    monkeypatch.setattr(h, 'corpus', forbidden); monkeypatch.setattr(h, 'measure', forbidden)
    out = tmp_path / 'forward.json'
    assert h.main(['--capsule', str(tmp_path), '--check-only', '--output', str(out)]) == 1
    assert 'upstream-forwarded' in json.loads(out.read_text())['failures'][0]['message']


def test_original_arraymap_is_never_counted_as_native():
    case = h.Case(h.APIS[3], 'map-provenance', dict(dtype='i2'))
    with pytest.raises(AssertionError, match='forwarded original ArrayMap'):
        h.observe_case(h.resolve(case.api, True), case, native=True)


def test_no_dense_coercion_when_describing_returned_maps(monkeypatch):
    case = h.Case(h.APIS[3], 'sparse-metadata', dict(dtype='u8', options={'offset': 256}))
    image, options, tracked = h.make_case(case)
    result = h.resolve(case.api, True)(image, **options)
    monkeypatch.setattr(type(result[1]), '__array__', forbidden)
    observation = h.describe(result, tracked)
    assert observation['map_vector_aliases'] == [True, True]
    assert observation['tuple'][1]['lazy'] and observation['tuple'][2]['lazy']


def test_unit_workers_and_policy_restored_without_timers(monkeypatch, tmp_path):
    install_unit_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(h, 'measure', forbidden); monkeypatch.setattr(time, 'perf_counter_ns', forbidden)
    old = h.xx.get_num_threads(); fallback = h.xx.set_fallback_enabled(True)
    try:
        out = tmp_path / 'unit-check.json'
        assert h.main(['--capsule', str(tmp_path), '--check-only', '--threads', '1', '4', '--output', str(out)]) == 0
        r = json.loads(out.read_text()); assert len(r['checks']) == 4 and not r['results']
        assert r['metadata']['artifacts']['unit_harness_stub_only'] is True
        assert all(set(c['by_workers']) == {'1', '4'} for c in r['checks'])
        assert h.xx.get_num_threads() == old and h.xx.set_fallback_enabled(False) is True
    finally: h.xx.set_fallback_enabled(fallback)


def test_last_bad_case_blocks_all_timers(monkeypatch, tmp_path):
    install_unit_stubs(monkeypatch, tmp_path, bad=True)
    monkeypatch.setattr(h, 'measure', forbidden); monkeypatch.setattr(time, 'perf_counter_ns', forbidden); monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    out = tmp_path / 'bad.json'
    assert h.main(['--capsule', str(tmp_path), '--output', str(out)]) == 1
    r = json.loads(out.read_text()); assert r['status'] == 'failed-parity' and len(r['checks']) == 4 and not r['results']


def test_unit_samples_and_regressions_retained(monkeypatch, tmp_path):
    install_unit_stubs(monkeypatch, tmp_path); monkeypatch.setenv('GITHUB_ACTIONS', 'true'); monkeypatch.setattr(time, 'perf_counter_ns', forbidden)
    def unit_measure(left, right, repeat, target):
        left(); right()
        return dict(reference_ms=1., cpp_ms=2., speedup=.5, samples_ms=[[1.]*repeat, [2.]*repeat], loops=[1, 1], sample_labels=['scikit-image', 'current'])
    monkeypatch.setattr(h, 'measure', unit_measure)
    out = tmp_path / 'unit-measure.json'
    assert h.main(['--capsule', str(tmp_path), '--repeat', '2', '--output', str(out)]) == 0
    r = json.loads(out.read_text()); assert len(r['results']) == 4
    assert all(row['samples_ms'] == [[1., 1.], [2., 2.]] and row['slower_than_original'] for row in r['results'])
    assert all(value['1']['regressions'] for value in r['summary_by_api'].values())
    assert r['metadata']['artifacts']['unit_harness_stub_only'] is True


@pytest.mark.parametrize('out', ['self', 'copy', 'F', 'strided'])
def test_mutating_callback_resets_inside_each_call_and_preserves_pristine_owners(out):
    case = h.Case(h.APIS[0], 'reset', dict(dtype='i2', out=out))
    call, arrays = h.timing_callback(h.resolve(case.api, True), case)
    before = {k: h.array_state(v) for k, v in arrays.items()}
    first, owners = call(); second, other_owners = call()
    np.testing.assert_array_equal(first, second)
    assert owners and other_owners and all(a is not b for a, b in zip(owners, other_owners))
    assert {k: h.array_state(v) for k, v in arrays.items()} == before


def test_fixed_case_ids_safety_and_separate_metadata_classes():
    cases = list(h.corpus()); matrix = json.loads(h.MANIFEST.read_text())
    assert len(cases) == 878 and sum(c.timed for c in cases) == 216
    assert len({(c.api, c.id) for c in cases}) == 878
    assert matrix['source_contract_sha256'] == h.artifacts.sha(h.ROOT / 'coverage/contracts/component-consumers.json')
    assert matrix['source_test_sha256'] == h.artifacts.sha(h.ROOT / 'tests/test_component_consumer_contracts.py')
    for frozen in matrix['profiles'].values():
        assert frozen['timing_eligible_case_count'] == 205
        assert all(row['classification'].startswith('raster') for row in frozen['cases'] if row['timing_eligible'])
    assert all(c.kind in ('raster', 'raster-mutation') for c in cases if c.timed)
    assert matrix['safety_exclusions'] and 'no dense' in matrix['returned_class_scope'].lower()


@pytest.mark.parametrize('args', [[], ['--reference-only'], ['--check-only'], ['--capsule', 'unused'],
    ['--capsule', 'unused', '--check-only', '--threads', '4'], ['--reference-only', '--check-only', '--capsule', 'unused']])
def test_invalid_configuration_never_times(monkeypatch, args):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False); monkeypatch.setattr(h, 'measure', forbidden)
    with pytest.raises(SystemExit): h.main(args)


def test_capsule_or_corpus_tamper_rejected_before_fixture_work(monkeypatch, tmp_path):
    install_unit_stubs(monkeypatch, tmp_path)
    def reject(*args): raise AssertionError('tampered exact capsule')
    monkeypatch.setattr(h, 'verify', reject); monkeypatch.setattr(h, 'corpus', forbidden); monkeypatch.setattr(h, 'measure', forbidden)
    out = tmp_path / 'tamper.json'
    assert h.main(['--capsule', str(tmp_path), '--check-only', '--output', str(out)]) == 1
    assert not json.loads(out.read_text())['checks']


@pytest.mark.parametrize('target', ['runtime', 'corpus', 'source'])
def test_actual_seal_checks_reject_tampering_before_runtime_or_pixels(monkeypatch, tmp_path, target):
    corpus = tmp_path / 'corpus.json'; corpus.write_text('{}'); monkeypatch.setattr(h, 'MANIFEST', corpus)
    capsule = tmp_path / 'capsule'; capsule.mkdir(); runtime = capsule / 'consumer-runtime.json'; runtime.write_text('{}')
    seal = dict(schema='consumer-exact-offline-v1', runtime_manifest_sha256=h.artifacts.sha(runtime),
        corpus_manifest_sha256=h.artifacts.sha(corpus), product_source=h.source_fingerprint(h.ROOT))
    (capsule / 'consumer-artifacts.json').write_text(json.dumps(seal))
    def runtime_must_not_run(*args): raise RuntimeError('tamper was not rejected before runtime verification')
    monkeypatch.setattr(h.artifacts, 'verify', runtime_must_not_run)
    if target == 'runtime': runtime.write_text('changed')
    elif target == 'corpus': corpus.write_text('changed')
    else: monkeypatch.setattr(h, 'source_fingerprint', lambda *args: {'unit': 'changed'})
    with pytest.raises(AssertionError): h.verify(capsule)


def test_source_header_omission_from_both_wheel_and_install_fails_before_fixtures(monkeypatch, tmp_path):
    real_verify = h.verify; install_unit_stubs(monkeypatch, tmp_path); monkeypatch.setattr(h, 'verify', real_verify)
    capsule = tmp_path / 'capsule'; (capsule / 'wheels').mkdir(parents=True)
    runtime = capsule / 'consumer-runtime.json'; runtime.write_text('{}')
    headers = list((h.ROOT / 'include/skimagexx').rglob('*.hpp'))
    omitted = next(p for p in headers if p.name not in ('component_consumers.hpp', 'components.hpp', 'image.hpp'))
    installed = tmp_path / 'installed'; sdk = {}
    with zipfile.ZipFile(capsule / 'wheels/unit.whl', 'w') as archive:
        archive.writestr('skimagexx/__init__.py', b'# unit fixture')
        for source in headers:
            if source == omitted: continue
            name = source.relative_to(h.ROOT).as_posix(); archive.writestr(name, source.read_bytes())
            path = installed / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(source.read_bytes()); sdk[name] = h.artifacts.sha(path)
    monkeypatch.setattr(h.artifacts, 'verify', lambda *args: {'runtime': {'sdk_sha256': sdk}})
    seal = dict(schema='consumer-exact-offline-v1', runtime_manifest_sha256=h.artifacts.sha(runtime),
        corpus_manifest_sha256=h.artifacts.sha(h.MANIFEST), product_source=h.source_fingerprint(h.ROOT))
    (capsule / 'consumer-artifacts.json').write_text(json.dumps(seal))
    monkeypatch.setattr(h, 'corpus', forbidden); monkeypatch.setattr(h, 'measure', forbidden)
    out = tmp_path / 'omitted.json'
    assert h.main(['--capsule', str(capsule), '--check-only', '--output', str(out)]) == 1
    result = json.loads(out.read_text()); assert not result['checks'] and not result['results']
    assert 'complete SDK header set' in result['failures'][0]['message']


def test_workflow_all_preparation_precedes_fresh_offline_measurement():
    text = (h.ROOT / '.github/workflows/component-consumers-benchmark.yml').read_text()
    preparation, measurement = text.split('\n  measure:', 1)
    assert 'needs: prepare' in measurement and '--prepare-capsule' in preparation
    commands = '\n'.join(l for l in measurement.splitlines() if not l.lstrip().startswith('#'))
    assert 'pip wheel' not in commands and 'pytest' not in commands and 'cmake' not in commands
    assert '--no-index --no-deps --require-hashes' in measurement and '--repeat 7' in measurement
    assert 'runtime-lock.txt' in measurement and 'build/prepared/capsule/' in measurement
