"""Fixed source-authenticated original corpus and private NMI parity tests."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/normalized-mutual-information.json'


def source_authentication():
    """Authenticate the audited installed Python sources before their import."""
    contract = json.loads(CONTRACT.read_text())
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-image')}
    assert versions['scikit-image'] == '0.26.0'
    profile = 'windows' if sys.platform == 'win32' else 'minimum' if versions['numpy'] == '1.24.4' else 'modern'
    assert versions['numpy'] == ('1.24.4' if profile == 'minimum' else '2.5.4')
    assert versions['scipy'] == ('1.15.3' if profile == 'minimum' else '1.18.1')
    records = []
    for relative, expected in contract['source_audit'][profile].items():
        distribution = 'scikit-image' if relative.startswith('skimage/') else relative.split('/')[0]
        path = Path(importlib.metadata.distribution(distribution).locate_file(relative))
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        assert actual == expected, 'Audited original source changed: ' + str(path)
        records.append(dict(path=str(path), sha256=actual, bytes=len(raw)))
    return dict(profile=profile, versions=versions, python=sys.version,
                executable=sys.executable, files=records)


def load_cases():
    spec = importlib.util.spec_from_file_location('nmi_fixed_cases', ROOT / 'tests/test_normalized_mutual_information_cases.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cases()


def observe(function, case):
    import numpy as np
    callbacks = []
    class Log:
        def write(self, value): callbacks.append(value)
    previous = np.geterrcall()
    trace = case['trace']
    if trace is not None: trace.clear()
    result = {}
    policy = case['policy']
    try:
        if policy == 'call': np.seterrcall(lambda text, flag: callbacks.append([text, flag]))
        elif policy == 'log': np.seterrcall(Log())
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            with np.errstate(**({'all': policy} if policy else {})):
                try:
                    value = function(case['image0'], case['image1'], bins=case['bins'])
                    scalar = np.asarray(value)
                    result['return'] = dict(type=type(value).__module__ + '.' + type(value).__name__,
                        dtype=scalar.dtype.str, shape=list(scalar.shape), bits=scalar.tobytes().hex(),
                        value=float(value) if np.isfinite(value) else 'nan' if np.isnan(value) else 'inf' if value > 0 else '-inf')
                except Exception as error:
                    result['exception'] = dict(type=type(error).__module__ + '.' + type(error).__name__, message=str(error))
        result['warnings'] = [dict(category=type(w.message).__name__, message=str(w.message)) for w in seen]
    finally:
        np.seterrcall(previous)
    result['callbacks'] = callbacks
    result['protocol'] = list(trace) if trace is not None else []
    return result


def record_reference(destination):
    auth = source_authentication()
    from skimage.metrics import normalized_mutual_information
    factory = ROOT / 'tests/test_normalized_mutual_information_cases.py'
    corpus = load_cases()
    result = dict(schema='normalized-mutual-information-original-v1', source_authentication=auth,
        contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        factory_sha256=hashlib.sha256(factory.read_bytes()).hexdigest(),
        candidate_calls=0, timings=0,
        cases=[dict(id=case['id'], expected_native=case['native'], observation=observe(normalized_mutual_information, case)) for case in corpus])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(auth['profile'], len(corpus), 'original cases; candidate calls 0; timings 0')


if __name__ != '__main__':
    import types
    import pytest

    AUTH = source_authentication()
    import numpy as np
    CASES = load_cases()
    POLICY = json.loads(CONTRACT.read_text())['comparison_policy']['floating']

    @pytest.fixture(scope='module')
    def native():
        path = os.environ.get('SKIMAGEXX_NMI_CORE')
        if not path: pytest.skip('Private NMI module requires an explicit authenticated fixture')
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == os.environ['SKIMAGEXX_NMI_CORE_SHA256']
        assert 'skimagexx._core' not in sys.modules, 'Private proof must have one SDK worker state'
        spec = importlib.util.spec_from_file_location('_nmi_core', path)
        core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
        assert core.get_num_threads() == 1
        if os.environ.get('SKIMAGEXX_DISABLE_SIMD') == '1': assert not core.has_avx2()
        package = types.ModuleType('_nmi_test_package'); package.__path__ = []
        sys.modules[package.__name__] = package
        sys.modules[package.__name__ + '._core'] = core
        loaded = {}
        for name in ('_compat', '_normalized_mutual_information'):
            spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, ROOT / 'python/skimagexx' / (name + '.py'))
            module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
            spec.loader.exec_module(module); loaded[name] = module
        yield core, loaded['_normalized_mutual_information'], loaded['_compat']
        core.set_num_threads(1)
        for name in list(sys.modules):
            if name == package.__name__ or name.startswith(package.__name__ + '.'):
                del sys.modules[name]

    def compare_observations(expected, actual):
        assert expected.keys() == actual.keys()
        for field in ('warnings', 'exception', 'callbacks', 'protocol'):
            if field in expected: assert expected[field] == actual[field]
        if 'return' not in expected: return
        before, after = expected['return'], actual['return']
        for field in ('type', 'dtype', 'shape'): assert before[field] == after[field]
        if isinstance(before['value'], str): assert before['value'] == after['value']
        else:
            assert np.isclose(before['value'], after['value'], rtol=POLICY['rtol'], atol=POLICY['atol'])

    @pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
    def test_original_contract_and_explicit_native_frontier(native, case):
        from skimage.metrics import normalized_mutual_information as original
        _, shell, _ = native
        snapshots = [x.tobytes() if isinstance(x, (np.ndarray, np.generic)) else None for x in (case['image0'], case['image1'])]
        expected = observe(original, case)
        actual = observe(shell.normalized_mutual_information, case)
        fallback = [x for x in actual['warnings'] if x['category'] == 'FallbackWarning']
        actual['warnings'] = [x for x in actual['warnings'] if x['category'] != 'FallbackWarning']
        if case['native'] is True: assert not fallback
        elif case['native'] is False: assert len(fallback) == 1
        compare_observations(expected, actual)
        for snapshot, image in zip(snapshots, (case['image0'], case['image1'])):
            if snapshot is not None: assert snapshot == image.tobytes()

    @pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
    def test_histogram_membership_edges_and_entropy(native, case):
        from scipy.stats import entropy
        from skimage.metrics.simple_metrics import _pad_to
        core, shell, _ = native
        if case['policy'] or case['id'] == 'rank-mismatch': pytest.skip('Separate exceptional contract case')
        prepared = shell._prepare(case['image0'], case['image1'], case['bins'])
        if prepared is None: pytest.skip('Explicit provider frontier')
        a, b = case['image0'], case['image1']
        if a.shape != b.shape:
            shape = np.maximum(a.shape, b.shape); a, b = _pad_to(a, shape), _pad_to(b, shape)
        expected, edges = np.histogramdd([np.reshape(a, -1), np.reshape(b, -1)], bins=case['bins'], density=False)
        actual = core.nmi_statistics(*prepared, True)
        np.testing.assert_array_equal(actual['counts'], expected)
        for i in range(2):
            np.testing.assert_array_equal(prepared[2 + i], edges[i])
            np.testing.assert_array_equal(prepared[4 + i], np.diff(edges[i]))
        if actual['admitted']:
            density, _ = np.histogramdd([np.reshape(a, -1), np.reshape(b, -1)], bins=case['bins'], density=True)
            expected_entropy = [entropy(np.sum(density, axis=0)), entropy(np.sum(density, axis=1)), entropy(density.reshape(-1))]
            np.testing.assert_allclose(actual['entropy'], expected_entropy, rtol=POLICY['rtol'], atol=POLICY['atol'])

    def test_worker_counts_and_entropy_are_deterministic(native):
        core, shell, _ = native
        case = next(c for c in CASES if c['id'] == 'worker-sized')
        prepared = shell._prepare(case['image0'], case['image1'], case['bins'])
        reference = core.nmi_statistics(*prepared, True)
        try:
            for workers in (2, 4):
                core.set_num_threads(workers)
                assert core.get_num_threads() == workers
                value = core.nmi_statistics(*prepared, True)
                np.testing.assert_array_equal(value['counts'], reference['counts'])
                assert value['entropy'] == reference['entropy']
        finally: core.set_num_threads(1)

    def test_ordinary_path_does_not_call_original_algorithms(native, monkeypatch):
        import skimage.metrics
        import skimage.metrics.simple_metrics as metrics
        import scipy.stats
        _, shell, _ = native
        def forbidden(*args, **kwargs): raise AssertionError('Original numerical algorithm reached')
        for module, name in ((np, 'histogramdd'), (np, 'bincount'), (np, 'searchsorted'),
                (scipy.stats, 'entropy'), (metrics, 'entropy'),
                (metrics, 'normalized_mutual_information'), (skimage.metrics, 'normalized_mutual_information')):
            monkeypatch.setattr(module, name, forbidden)
        case = next(c for c in CASES if c['id'] == 'float64-reverse')
        assert np.isfinite(shell.normalized_mutual_information(case['image0'], case['image1']))

    def test_provider_policy_is_visible_and_can_be_disabled(native):
        _, shell, compat = native
        a = np.arange(16.)
        compat.set_fallback_enabled(False)
        try:
            assert np.isfinite(shell.normalized_mutual_information(a, a[::-1]))
            with pytest.raises(NotImplementedError, match='outside native coverage'):
                shell.normalized_mutual_information(a, a, bins=np.int64(7))
        finally: compat.set_fallback_enabled(True)

    @pytest.mark.parametrize('edges,widths', [([], []), ([0., 1.], []), ([0., 0.], [1.]),
        ([0., np.inf], [1.]), ([0., 1.], [0.]), ([0., 1.], [np.nan]), ([0., 1.], [1e101])])
    def test_native_binding_rejects_invalid_histogram_metadata(native, edges, widths):
        core, _, _ = native
        with pytest.raises(ValueError):
            core.nmi_statistics(np.arange(4.), np.arange(4.), edges, [0., 4.], widths, [4.])

    def test_signature_and_single_worker_default(native):
        import inspect
        core, shell, _ = native
        assert str(inspect.signature(shell.normalized_mutual_information)) == '(image0, image1, *, bins=100)'
        assert core.get_num_threads() == 1

    def test_metadata_only_guard_rejects_overreaching_strided_owner(native):
        core, _, _ = native
        # Only the metadata-only entry is called: no oracle or kernel reads this
        # deliberately out-of-allocation array, even if an admission test fails.
        root = np.arange(4.)
        unsafe = np.lib.stride_tricks.as_strided(root, shape=(4096,), strides=(8,))
        with pytest.raises(ValueError, match='bounded ndarray'):
            core.nmi_validate_inputs(unsafe, root)

    def test_metadata_only_guard_checks_current_owner_after_conversion(native):
        core, _, _ = native
        root = np.arange(64.)
        stale_view = root[::2]
        class Resize:
            def __float__(self):
                root.resize(1, refcheck=False)
                return 1.
        with pytest.raises(ValueError, match='current owner allocation'):
            core.nmi_validate_inputs(stale_view, np.zeros(1), [Resize()])

    def test_owner_base_is_read_from_numpy_c_metadata(native):
        core, _, _ = native
        class Owner(np.ndarray):
            @property
            def base(self): raise AssertionError('Overridable Python base accessed')
        root = np.ndarray.__new__(Owner, (8,), dtype=np.float64)
        root[...] = np.arange(8.)
        image = root.view(np.ndarray)[::-1]
        assert core.nmi_validate_inputs(image, image)

    def test_bounded_builtin_buffer_and_memmap_owners(native, tmp_path):
        core, shell, _ = native
        buffer = bytearray(65)
        unaligned = np.ndarray((8,), dtype=np.float64, buffer=buffer, offset=1)
        unaligned[...] = np.arange(8.)
        readonly = np.frombuffer(np.arange(8.).tobytes(), dtype=np.float64)
        mapped = np.memmap(tmp_path / 'nmi-input.bin', dtype=np.float64, mode='w+', shape=(8,))
        mapped[...] = np.arange(8.)
        for image in (unaligned, readonly, mapped):
            assert core.nmi_validate_inputs(image, readonly)
            assert shell.normalized_mutual_information(image, readonly, bins=8) == 2.

    def test_unknown_owner_is_an_explicit_private_boundary(native):
        core, _, _ = native
        original = np.arange(8.)
        class Foreign:
            __array_interface__ = original.__array_interface__
        foreign = np.asarray(Foreign())
        with pytest.raises(ValueError, match='bounded ndarray'):
            core.nmi_validate_inputs(foreign, original)
        with pytest.raises(TypeError): core.nmi_validate_inputs([1., 2.], original)

    def test_direct_native_underflowed_probability_contributes_zero(native):
        import math
        core, _, _ = native
        edges = [0., 1e-100, 2e-100, 1e100]
        widths = [1e-100, 1e-100, 1e100]
        value = core.nmi_statistics(np.array([0., 0., 1.]), np.array([0., 1e-100, 1.]),
                                    edges, edges, widths, widths, True)
        assert value['admitted']
        np.testing.assert_allclose(value['entropy'], [math.log(2), 0., math.log(2)], rtol=2e-15, atol=0.)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-reference', required=True)
    record_reference(parser.parse_args().record_reference)
