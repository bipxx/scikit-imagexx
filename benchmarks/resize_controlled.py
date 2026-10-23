"""Same-process original/initial-C++/current-C++ resize comparison.

The 373 cases and their numerical contracts come directly from resize.py.
Local execution is untimed. A sealed artifact-fed Actions host is mandatory
for measurements; the existing two-backend benchmark is unchanged.
"""
try:
    from . import resize as h
except ImportError:
    import resize as h
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
import uuid
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import resize_controlled_artifacts as artifacts


def policies(module):
    return importlib.import_module(module.__name__ + '._compat')


def verify_backends(current, baseline):
    """Real modules, authenticated native extensions and independently mutable state."""
    assert current is not baseline and current._core is not baseline._core, 'native modules must be distinct'
    paths = [Path(module._core.__file__).resolve() for module in (current, baseline)]
    assert paths[0] != paths[1] and artifacts.sha(paths[0]) != artifacts.sha(paths[1]), 'initial/current binaries must be distinct'
    for module in (current, baseline):
        registry = importlib.import_module(module.__name__ + '.coverage').native_functions()
        assert set(h.APIS) <= set(registry), 'resize APIs must be publicly registered in both packages'
        assert module.get_num_threads() == 1, 'both packages must begin at default worker1'
        assert module.get_num_threads is module._core.get_num_threads and module.set_num_threads is module._core.set_num_threads
        for name in ('resize', 'rescale'):
            function = getattr(module.transform, name)
            assert function is not getattr(h.reference, name) and function.__module__.startswith(module.__name__ + '.'), 'upstream-forwarded resize path'
            assert function.__globals__.get('_core') is module._core, 'wrapper uses a different native extension'
    state = [(m.get_num_threads(), policies(m)._fallback_enabled) for m in (current, baseline)]
    try:
        current.set_num_threads(4); assert baseline.get_num_threads() == 1
        current.set_num_threads(1); baseline.set_num_threads(4); assert current.get_num_threads() == 1
        current.set_fallback_enabled(False); baseline.set_fallback_enabled(True)
        assert policies(current)._fallback_enabled is False
        baseline.set_fallback_enabled(False); current.set_fallback_enabled(True)
        assert policies(baseline)._fallback_enabled is False
    finally:
        for module, (workers, fallback) in zip((current, baseline), state):
            module.set_num_threads(workers); module.set_fallback_enabled(fallback)
    return {'module_names': [current.__name__, baseline.__name__], 'extension_paths': list(map(str, paths)),
            'extension_sha256': list(map(artifacts.sha, paths)), 'independent_workers': True,
            'independent_fallback': True, 'default_workers': [1, 1]}


def grouped_summary(rows, workers, eligible):
    result = {}
    for worker in workers:
        selected = [r for r in rows if r['threads'] == worker]
        assert len(selected) == eligible
        groups = {}
        for field in ('api', 'case', 'dtype'):
            groups[field] = {}
            for key in sorted({row[field] for row in selected}):
                group = [row for row in selected if row[field] == key]
                groups[field][key] = {
                    'cases': len(group),
                    'geometric_mean_current_vs_initial': math.exp(math.fsum(math.log(r['improvement']) for r in group) / len(group)),
                    'regressions': [r['case_index'] for r in group if r['improvement'] < 1],
                }
        result[str(worker)] = {'cases': len(selected), 'primary': worker == 1,
            'geometric_mean_current_vs_initial': math.exp(math.fsum(math.log(r['improvement']) for r in selected) / len(selected)),
            'geometric_mean_current_vs_original': math.exp(math.fsum(math.log(r['speedup']) for r in selected) / len(selected)),
            'geometric_mean_initial_vs_original': math.exp(math.fsum(math.log(r['baseline_speedup']) for r in selected) / len(selected)),
            'regressions': [dict(case_index=r['case_index'], improvement=r['improvement']) for r in selected if r['improvement'] < 1],
            'groups': groups}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', type=Path)
    parser.add_argument('--threads', type=int, nargs='+', default=[1])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=15.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args(argv)
    if not args.threads or min(args.threads) < 1 or len(set(args.threads)) != len(args.threads) or 1 not in args.threads or args.repeat < 1 or not math.isfinite(args.target_ms) or args.target_ms <= 0:
        parser.error('positive distinct workers/repeats/target-ms required; include worker1')
    if args.reference_only:
        if not args.check_only or args.capsule is not None:
            parser.error('--reference-only requires --check-only and no capsule; it makes no old/current claim')
        forwarded = ['--reference-only', '--check-only', '--threads', *map(str, args.threads)]
        if args.output is not None:
            forwarded += ['--output', str(args.output)]
        return h.main(forwarded)
    if args.capsule is None:
        parser.error('controlled comparison requires an authenticated --capsule')
    if not args.check_only and (args.output is None or os.environ.get('GITHUB_ACTIONS') != 'true'):
        parser.error('timings require an isolated Actions host and --output; use --check-only locally')
    session = str(uuid.uuid4())
    cpu = dict(h.cpu_metadata(), logical_cpus=os.cpu_count(), cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER', ''))
    report = {'metadata': {'session_id': session, 'process_id': os.getpid(), 'cpu_session': cpu,
        'backend_cpu_sessions': {name: session for name in ('scikit-image', 'current', 'baseline')},
        'python': sys.version, 'numpy': h.np.__version__, 'scipy': h.scipy.__version__, 'scikit_image': h.skimage.__version__,
        'same_process_cpu_for_all_backends': True, 'command': sys.argv if argv is None else list(argv),
        'requested_workers': args.threads, 'primary_workers': 1, 'repeat': args.repeat, 'target_ms': args.target_ms,
        'check_only': args.check_only, 'reference_only': False, 'corpus_version': h.CORPUS_VERSION,
        'scope': 'Same fixtures and one process; warmed public calls include normalization, allocations and result destruction. Initial-run CPU is historical context only.',
        'warning_timing_policy': 'Warnings/errors compared for all3 backends; timing warning display suppressed symmetrically.',
        'sample_labels': ['scikit-image', 'current', 'baseline'], 'sample_order_seed': 81,
        'initial_run': artifacts.INITIAL_RUN, 'source_revision': h.source_revision(ROOT), **h.source_fingerprint(ROOT)},
        'status': 'preflight', 'checks': [], 'results': [], 'untimed_cases': [], 'failures': []}
    h.save(report, args.output)
    baseline = None
    saved_state = []
    try:
        assert h.skimage.__version__ == '0.26.0', 'reference must be exactly scikit-image0.26.0'
        assert set(h.APIS) <= set(h.native_functions()), 'current native resize APIs are not registered'
        seal = artifacts.verify(args.capsule)
        baseline = artifacts.load_baseline(seal['initial'])
        report['metadata']['backend_identity'] = verify_backends(h.xx, baseline)
        initial_data = seal['initial']['initial_measurement']
        report['metadata']['artifacts'] = {k: v for k, v in seal.items() if k != 'initial'}
        report['metadata']['initial_identity'] = {
            'authenticated_sha256': seal['initial']['authenticated_sha256'],
            'source': seal['initial']['manifest'],
            'runtime': {k: initial_data[k] for k in ('python', 'numpy', 'scipy', 'scikit_image')},
            'historical_cpu': {k: initial_data.get(k) for k in ('cpu_model', 'cpu_identifier', 'logical_cpus')},
            'case_spec_sha256': initial_data['case_spec_sha256']}
        for module in (h.xx, baseline):
            saved_state.append((module, module.get_num_threads(), module.set_fallback_enabled(False)))
            assert policies(module)._fallback_enabled is False
        cases = list(h.corpus())
        specs = [h.spec(case) for case in cases]
        contracts = [{k: v for k, v in item.items() if k != 'input_sha256'} for item in specs]
        contract_hash = hashlib.sha256(json.dumps(contracts, sort_keys=True).encode()).hexdigest()
        assert len(cases) == 373 and sum(not c.expect_error for c in cases) == 372
        assert contract_hash == initial_data['case_contract_sha256'], '373-case workload contract differs from initial run'
        report['metadata'].update(case_count=len(cases), timing_eligible_case_count=372,
            case_contract_sha256=contract_hash,
            case_spec_sha256=hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest(), case_specs=specs)
        report['status'] = 'checking'
        for index, case in enumerate(cases):
            row = dict(case_index=index, **specs[index])
            snapshot = case.image.tobytes()
            try:
                ref = getattr(h.reference, case.api)
                expected, notices, expected_error = h.observe(lambda: ref(case.image, case.target, **case.options))
                assert case.image.tobytes() == snapshot, 'original mutated input'
                assert bool(expected_error is not None) == case.expect_error
                row['warnings'] = notices
                row['backends'] = {}
                if expected_error is not None:
                    row['expected_exception'] = {'type': type(expected_error).__name__, 'message': str(expected_error)}
                    report['untimed_cases'].append({'case_index': index, 'reason': 'fixed expected-error parity; never timed'})
                else:
                    assert type(expected) is h.np.ndarray and not h.np.shares_memory(expected, case.image)
                for label, module in [('current', h.xx), ('baseline', baseline)]:
                    function = getattr(module.transform, case.api)
                    first_bits = None
                    errors = {}
                    for worker in args.threads:
                        module.set_num_threads(worker)
                        assert module.get_num_threads() == worker and policies(module)._fallback_enabled is False
                        actual, actual_notices, error = h.observe(lambda: function(case.image, case.target, **case.options))
                        assert actual_notices == notices and case.image.tobytes() == snapshot
                        if expected_error is not None:
                            assert type(error) is type(expected_error) and str(error) == str(expected_error)
                            errors[str(worker)] = 'expected-error'
                        else:
                            if error is not None:
                                raise error
                            errors[str(worker)] = h.assert_result(actual, expected, case.image, case.options)
                            bits = actual.tobytes()
                            assert first_bits is None or bits == first_bits, label + ' worker outputs differ in bytes'
                            first_bits = bits
                    row['backends'][label] = errors
                row['status'] = 'passed'
            except Exception as error:
                row.update(status='failed', exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                report['failures'].append(row)
            report['checks'].append(row)
            h.save(report, args.output)
            print(f'CHECK {index+1}/{len(cases)} {case.api} {case.label}: {row["status"]}', flush=True)
        if report['failures']:
            report['status'] = 'failed-parity'
            return 1
        if not args.check_only:
            orders = []
            rng = random.Random(81)
            for _ in range(args.repeat):
                order = ['scikit-image', 'current', 'baseline']; rng.shuffle(order); orders.append(order)
            report['metadata']['paired_sample_order'] = orders
            report['status'] = 'measuring'
            for index, case in enumerate(cases):
                if case.expect_error:
                    continue
                ref, current, old = (getattr(m, case.api) for m in (h.reference, h.xx.transform, baseline.transform))
                snapshot = case.image.tobytes()
                for worker in args.threads:
                    h.xx.set_num_threads(worker); baseline.set_num_threads(worker)
                    assert h.xx.get_num_threads() == baseline.get_num_threads() == worker
                    assert not policies(h.xx)._fallback_enabled and not policies(baseline)._fallback_enabled
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        timing = h.measure(lambda: ref(case.image, case.target, **case.options),
                            lambda: current(case.image, case.target, **case.options), args.repeat, args.target_ms,
                            baseline=lambda: old(case.image, case.target, **case.options))
                    assert case.image.tobytes() == snapshot, 'timed backend mutated fixture'
                    row = dict(case_index=index, api='transform.'+case.api, case=case.label, dtype=case.image.dtype.str,
                        threads=worker, session_id=session, baseline_speedup=timing['reference_ms']/timing['baseline_ms'],
                        regression_vs_initial=timing['improvement'] < 1, slower_than_original=timing['speedup'] < 1, **timing)
                    report['results'].append(row); h.save(report, args.output)
            report['summary_by_workers'] = grouped_summary(report['results'], args.threads, 372)
        assert dict(h.cpu_metadata(), logical_cpus=os.cpu_count(), cpu_identifier=os.environ.get('PROCESSOR_IDENTIFIER', '')) == cpu, 'CPU session identity changed'
        assert artifacts.verify(args.capsule) == seal, 'source/runtime/artifact inputs changed'
        report['metadata']['seals_verified_before_and_after'] = True
        report['status'] = 'complete-parity-check' if args.check_only else 'complete-controlled-measurement'
        return 0
    except Exception as error:
        report['failures'].append({'phase': report['status'], 'exception_type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        report['status'] = 'failed'; print(f'FAILED {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        for module, workers, fallback in saved_state:
            module.set_num_threads(workers); module.set_fallback_enabled(fallback)
        h.save(report, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
