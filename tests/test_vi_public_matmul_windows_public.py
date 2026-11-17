"""Windows current VI words/software contract; no Linux FEnv32 claim."""
from contextlib import nullcontext
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import pytest
from test_variation_of_information_public import target, SPECS
from test_variation_of_information_public import (
    test_owned_entropy_uses_only_explicit_log2_and_matmul,
    test_current_provider_primary_stops_later_callbacks,
    test_current_matmul_rejects_nonplain_scalar_transport,
)
from test_vi_public_log2_provider_large import call_once, ownership_profile, retain_record

ROOT = Path(__file__).resolve().parents[1]
FP_SCOPE = 'Windows software/word contract only; no Linux FEnv32 or hardware-FP claim'


@pytest.fixture(autouse=True, params=(1, 4))
def workers(target, request):
    assert sys.platform == 'win32' and not os.environ.get('VI_COMPONENT_DIR')
    directory = Path(os.environ['VI_PUBLIC_MATMUL_WINDOWS_WORKER_RECORD_DIR'])
    assert directory.is_absolute()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(request.node.nodeid.encode()).hexdigest() + '.json')
    assert not path.exists()
    core = target[1]
    before = core.get_num_threads(); assert before == 1
    core.set_num_threads(request.param)
    primary = None
    try:
        yield request.param
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            core.set_num_threads(before)
            after = core.get_num_threads()
            row = dict(node=request.node.nodeid, test=request.node.originalname, selected=request.param, before=before,
                       after=after, exact_default1=before == after == 1, FP_scope=FP_SCOPE)
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(row, indent=2) + '\n')
            assert row['exact_default1']
        except BaseException as error:
            if primary is None:
                raise
            sys.stderr.write('VI_WINDOWS_WORKER_SECONDARY ' + str(error) + '\n')


def software_record(state):
    policy, handler, buffer = state
    code = getattr(handler, '__code__', None)
    source = Path(code.co_filename) if code is not None else None
    witness = None if code is None else dict(identity=str(id(code)), first_line=code.co_firstlineno,
        source_path=str(source.resolve()), source_canonical_sha256=(hashlib.sha256(source.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if source.is_file() else None))
    return dict(policy=policy, handler_code=witness, handler_type=type(handler).__name__,
                handler_module=getattr(handler, '__module__', None),
                handler_name=getattr(handler, '__qualname__', None),
                handler_identity=str(id(handler)), buffer=buffer)


def retain_target_software(record, states):
    record['target_software'] = {role: [software_record(value) for value in values]
                                 for role, values in states.items()}


def retain_windows_record(path, record, outputs, encode, states, restored):
    failure = None
    try:
        retain_target_software(record, states)
        record['restored_software'] = {role: software_record(value) for role, value in restored.items()}
    except BaseException as error:
        failure = error
        record['diagnostics'].append(dict(stage='target software encoding', type=type(error).__name__, error=str(error)))
    try:
        retain_record(path, record, outputs, encode)
    except BaseException as error:
        if failure is None:
            raise
        failure.add_note('Secondary raw writer failure: ' + str(error))
    if failure is not None:
        raise failure


@pytest.mark.parametrize('spec', SPECS, ids=lambda value: value['id'])
def test_windows_current_public_original_pair(target, workers, spec):
    assert sys.platform == 'win32' and not os.environ.get('VI_COMPONENT_DIR')
    from label_pair_metrics_observer import observe, reference_runtime
    providers = runpy.run_path(str(ROOT / 'benchmarks/vi_public_matmul_provider_execution.py'))['candidate_providers']
    array_words = runpy.run_path(str(ROOT / 'tools/vi_cohort_value_diagnostic.py'))['array_words']
    np, functions, runtime = reference_runtime()
    import scipy.sparse as sparse
    import skimage
    import skimagexx as xx
    owned = importlib.import_module('skimagexx._variation_of_information')
    compat = importlib.import_module('skimagexx._compat')
    reference = functions['metrics.variation_of_information']
    native = target[0].variation_of_information; core = target[1]
    assert native is xx.metrics.variation_of_information is owned.variation_of_information
    assert reference is skimage.metrics.variation_of_information and owned._core is core is xx._core
    assert str(inspect.signature(native)) == str(inspect.signature(reference)) == '(image0=None, image1=None, *, table=None, ignore_labels=())'
    assert isinstance(np.log2, np.ufunc) and isinstance(np.matmul, np.ufunc)
    true_providers = np.log2, np.matmul
    directory = Path(os.environ['VI_PUBLIC_MATMUL_WINDOWS_RECORD_DIR']); assert directory.is_absolute()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (spec['id'].split('/')[-1] + '-workers' + str(workers) + '.json')
    assert not path.exists()
    def software():
        return np.geterr(), np.geterrcall(), np.getbufsize()
    policy = software(); fallback = compat._fallback_enabled; prior = sys.getprofile()
    outputs = {}; diagnostics = []; states = {}; restored = {}; events = []; primary = None
    counts = dict(original=0, native=0); returned = []; accepted = []; suffix = []
    record = dict(case=spec, workers=workers, runtime=runtime, timing=False, primary=None,
                  diagnostics=diagnostics, FP_scope=FP_SCOPE, natural_software=software_record(policy),
                  native_core_path=str(Path(core.__file__).resolve()), native_frontend_path=str(Path(owned.__file__).resolve()),
                  public_signature=str(inspect.signature(native)),
                  provider_events=events, counts=counts, returned=returned, accepted=accepted, suffix=suffix,
                  provider_types=dict(log2=type(np.log2).__name__, matmul=type(np.matmul).__name__))
    entries = {key: getattr(core, key) for key in ('get_num_threads', 'label_pair_check_storage', 'label_pair_counts_preweighted', 'variation_of_information_csr')}
    assert all(inspect.isbuiltin(value) and value.__module__ == core.__name__ for value in entries.values())
    phase = ['setup']
    guard = ownership_profile(reference, native, owned.contingency_table, compat.fallback,
                              entries, phase, counts, returned, accepted, suffix)
    def checked(call, role):
        def once(*args, **options):
            phase[0] = role; observations = []
            def snapshot():
                value = software(); observations.append(value); return value
            try:
                with providers(np, sparse) if role == 'native' else nullcontext() as current:
                    if role == 'native': events.append(current)
                    return call_once(call, args, options, snapshot, guard, outputs,
                                     'actual' if role == 'native' else 'expected', diagnostics)[0]
            finally:
                states[role] = observations; phase[0] = 'setup'
        return once
    try:
        xx.set_fallback_enabled(False)
        expected = observe(checked(reference, 'reference'), spec, np)
        record['reference_observation'] = expected
        restored['reference'] = software()
        assert restored['reference'][0] == policy[0] and restored['reference'][1] is policy[1] and restored['reference'][2] == policy[2]
        actual = observe(checked(native, 'native'), spec, np)
        record['native_observation'] = actual
        restored['native'] = software()
        assert restored['native'][0] == policy[0] and restored['native'][1] is policy[1] and restored['native'][2] == policy[2]
        assert counts == dict(original=1, native=1) and actual == expected
        for values in states.values():
            assert len(values) == 2 and values[0][0] == values[1][0]
            assert values[0][1] is values[1][1] and values[0][2] == values[1][2]
        assert set(states) == {'reference', 'native'}
        for index in (0, 1):
            reference_state, native_state = states['reference'][index], states['native'][index]
            assert reference_state[0] == native_state[0] and reference_state[2] == native_state[2]
            assert reference_state[1].__code__ is native_state[1].__code__
        if spec.get('images_none'):
            assert events == [[]] and returned == suffix == [] and accepted == [False]
            assert actual['error'] == dict(type='AttributeError', module='builtins', message="'NoneType' object has no attribute 'shape'")
        else:
            requirements = (dict(get_num_threads=1, label_pair_check_storage=3, variation_of_information_csr=1)
                if 'table' in spec else dict(get_num_threads=2, label_pair_check_storage=7, label_pair_counts_preweighted=1, variation_of_information_csr=1))
            assert {key: returned.count(key) for key in requirements} == requirements
            assert len(returned) == sum(requirements.values()) and accepted == [True]
            assert suffix == ([] if 'table' in spec else [True])
            assert [value['provider'] for value in events[0]] == ['numpy.log2', 'numpy.matmul', 'numpy.log2', 'numpy.matmul']
            assert type(outputs['expected']) is type(outputs['actual']) is np.ndarray
            assert outputs['expected'].dtype == outputs['actual'].dtype == np.dtype('float64')
            assert outputs['expected'].shape == outputs['actual'].shape == (2,)
            assert outputs['expected'].tobytes() == outputs['actual'].tobytes()
            assert outputs['expected'].flags.owndata and outputs['actual'].flags.owndata
            assert not np.shares_memory(outputs['expected'], outputs['actual'])
        assert core.get_num_threads() == workers and np.log2 is true_providers[0] and np.matmul is true_providers[1]
        assert np.geterr() == policy[0] and np.geterrcall() is policy[1] and np.getbufsize() == policy[2]
        record.update(software_context_unchanged=True, namespace_identity_unchanged=True)
    except BaseException as error:
        primary = error; record['primary'] = dict(type=type(error).__name__, error=str(error))
        raise
    finally:
        sys.setprofile(prior)
        try:
            xx.set_fallback_enabled(fallback)
            np.seterr(**policy[0]); np.seterrcall(policy[1]); np.setbufsize(policy[2])
            assert np.geterr() == policy[0] and np.geterrcall() is policy[1] and np.getbufsize() == policy[2]
            assert np.log2 is true_providers[0] and np.matmul is true_providers[1]
            assert xx.metrics.variation_of_information is native and skimage.metrics.variation_of_information is reference
            record['software_restored'] = True
        except BaseException as error:
            diagnostics.append(dict(stage='software restoration', type=type(error).__name__, error=str(error)))
            if primary is None:
                primary = error; record['primary'] = dict(type=type(error).__name__, error=str(error))
                raise
            sys.stderr.write('VI_WINDOWS_SOFTWARE_SECONDARY ' + str(error) + '\n')
        finally:
            try:
                retain_windows_record(path, record, outputs, array_words, states, restored)
            except BaseException as error:
                if primary is None:
                    raise
                sys.stderr.write('VI_WINDOWS_DIAGNOSTIC_SECONDARY ' + str(error) + '\n')
