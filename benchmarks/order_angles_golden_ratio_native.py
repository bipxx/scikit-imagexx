"""Allocation-inclusive fresh generator construction AND full consumption.

Reuses the admitted util binding/F1/real-entry boundary and Montage sampler.
All actual authority is UNBOUND until root admits an installed product.
"""
import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 'coverage/contracts/order-angles-golden-ratio-benchmark-v2.json'


def common_module():
    context = json.loads((ROOT / CONTEXT).read_bytes())
    path = ROOT / 'benchmarks/next_util_execution.py'
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == context['common_canonical_sha256']
    spec = importlib.util.spec_from_file_location('_golden_benchmark_common', path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.CONTEXT = CONTEXT
    return common, context


def public_call(target, theta):
    # Construction alone executes no generator body and is not this benchmark.
    return list(target(theta))


def case_specs():
    context = json.loads((ROOT / CONTEXT).read_bytes())
    spec = importlib.util.spec_from_file_location('_golden_benchmark_cases', ROOT / 'tests/order_angles_golden_ratio_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    by_id = {row['id']: row for row in module.cases()}
    return [by_id[name] for name in context['case_ids']], module


def execute():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--root-binding', type=Path)
    parser.add_argument('--first-cohort', action='store_true')
    parser.add_argument('--measurement-go', type=Path)
    parser.add_argument('--case')
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, choices=[7], default=7)
    parser.add_argument('--output', type=Path, default=Path('build/golden-angle-initial.json'))
    cli = parser.parse_args()
    common, context = common_module()
    specs, factory = case_specs()
    assert specs == context['routes']['golden']['cases']
    if cli.case is not None:
        specs = [row for row in specs if row['id'] == cli.case]
    assert specs
    if cli.plan:
        print(json.dumps(dict(state=context['state'], cases=specs, method=context['method'], unbound=context['unbound']), indent=2))
        return
    binding, context, go, binding_sha256 = common.load_binding(cli.root_binding, ROOT, 'golden')
    go_sha256 = common.digest(Path(binding['root_go_path']).read_bytes())
    assert not (cli.first_cohort and cli.measurement_go)
    measurement_sha256 = None if cli.first_cohort else common.load_measurement(cli.measurement_go, binding, context, 'golden', specs)
    assert not cli.output.exists() and not cli.output.with_suffix('.batches.jsonl').exists()
    runtime = json.loads(Path(binding['runtime_path']).read_bytes())
    sdk = runtime['runtime']['sdk_sha256']
    assert 'include/skimagexx/order_angles_golden_ratio.hpp' in sdk
    assert any(name in sdk for name in ('lib/libskimagexx.a', 'lib/skimagexx.lib'))
    old_path = list(sys.path)
    try:
        sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'benchmarks')]
        from montage_sustained import measure
        from provenance import source_fingerprint
        assert common.digest(inspect.getsource(measure).encode()) == context['method']['method_sha256']
        import numpy as np
        import skimage
        import skimagexx as xx
        original_module = importlib.import_module('skimage.transform.radon_transform')
        own_module = importlib.import_module('skimagexx._order_angles_golden_ratio')
        reference = skimage.transform.order_angles_golden_ratio
        native = xx.transform.order_angles_golden_ratio
        assert reference is original_module.order_angles_golden_ratio and native is own_module.order_angles_golden_ratio
        assert inspect.isgeneratorfunction(reference) and inspect.isgeneratorfunction(native)
        assert str(inspect.signature(reference)) == str(inspect.signature(native)) == '(theta)'
        assert common.digest(common.source_bytes(original_module.__file__)) == context['original_source_canonical_sha256']
        assert skimage.__version__ == '0.26.0' and np.__version__ == binding['numpy']
        assert Path(xx._core.__file__).resolve() == Path(binding['core_path']).resolve()
        assert Path(xx._core.__file__).resolve().is_relative_to(Path(binding['installed_site']).resolve())
        assert all(os.environ[name] == '1' for name in common.LIMITS) and xx.get_num_threads() == 1
        xx.set_fallback_enabled(False)
        entries = {name: getattr(xx._core, name) for name in context['native_entries']}
        assert all(inspect.isbuiltin(entry) and entry.__module__ == 'skimagexx._core' for entry in entries.values())
        save, restore, flags = common.fp_controller(Path(binding['source_root']), context)
        source_before = source_fingerprint(Path(binding['source_root']))
        policy = np.geterr(), np.geterrcall(), np.getbufsize()
        def verify_authority():
            assert source_fingerprint(Path(binding['source_root'])) == source_before
            assert common.digest(Path(binding['core_path']).read_bytes()) == binding['core_sha256']
            assert common.digest(cli.root_binding.read_bytes()) == binding_sha256
            assert common.digest((ROOT / CONTEXT).read_bytes()) == binding['base_contract_sha256']
            assert common.digest(Path(binding['root_go_path']).read_bytes()) == go_sha256
            if cli.measurement_go:
                assert common.digest(cli.measurement_go.read_bytes()) == measurement_sha256
            assert xx.get_num_threads() == 1 and (np.geterr(), np.geterrcall(), np.getbufsize()) == policy
        rows = []
        try:
            for index, spec in enumerate(specs):
                args, options = factory.fixture(spec, np)
                assert not options and type(args[0]) is np.ndarray
                theta = args[0]
                before = common.array_state(theta, np)
                caller = save()
                prior = sys.getprofile()
                returns = []
                denied = {reference.__code__, importlib.import_module('skimagexx._compat').fallback.__code__}
                def profile(frame, event, value):
                    if event == 'call' and frame.f_code in denied:
                        raise AssertionError('Owned generator entered upstream/fallback')
                    if event == 'c_return':
                        returns.extend(name for name, entry in entries.items() if value is entry)
                try:
                    with warnings.catch_warnings(record=True) as notices:
                        warnings.simplefilter('always')
                        restore(caller)
                        expected = public_call(reference, theta)
                        expected_FP = flags()
                        restore(caller)
                        try:
                            sys.setprofile(profile)
                            actual = public_call(native, theta)
                            actual_FP = flags()
                        finally:
                            sys.setprofile(prior)
                    assert not notices and actual_FP == expected_FP
                    assert all(type(value) is np.int64 for value in expected + actual)
                    assert [value.tobytes() for value in actual] == [value.tobytes() for value in expected]
                    assert returns.count('_order_angles_begin') == 1
                    assert returns.count('_order_angles_next') == len(theta) - 1
                    assert returns.count('_order_angles_resume') == len(theta)
                    assert common.array_state(theta, np) == before
                finally:
                    sys.setprofile(prior)
                    restore(caller)
                sampler_previous = sys.getprofile()
                batch_caller = save()
                try:
                    if cli.first_cohort:
                        sys.setprofile(common.first_cohort_profile(measure, sampler_previous))
                    timing = measure(lambda: public_call(reference, theta), lambda: public_call(native, theta), 7, 100,
                                     journal=cli.output.with_suffix('.batches.jsonl'), case_index=index, case_id=spec['id'],
                                     pass_index=0, deadline=go['deadline_utc'])
                finally:
                    if cli.first_cohort:
                        sys.setprofile(sampler_previous)
                    restore(batch_caller)
                assert common.array_state(theta, np) == before
                rows.append(dict(case_id=spec['id'], api='transform.order_angles_golden_ratio', actual_C_returns=returns, FP=actual_FP, **timing))
        finally:
            # C1 runs after a real-entry observer stop as well as a normal batch.
            verify_authority()
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        with cli.output.open('x') as stream:
            json.dump(dict(metadata=dict(source_before=source_before, source_revision=binding['source_revision'], producer=binding['producer'],
                          root_binding_sha256=binding_sha256, measurement_sha256=measurement_sha256, core_sha256=binding['core_sha256'],
                          runtime_sha256=binding['runtime_sha256'], SDK_members=sdk, threads=1, repeat=7, target_ms=100,
                          warmup_min_ms=250, pass_index=0, retain_all_losses=True), results=rows), stream, indent=2)
    finally:
        sys.path[:] = old_path


if __name__ == '__main__':
    execute()
