"""Six prospective coords_scaled workloads; imports and pure intake are inert."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/regionprops-coords-scaled-benchmark-v1.json'
STAGES = ('local', 'origin_into', 'multiply_f64', 'add_f64_into')


def case_specs():
    return json.loads(CONTRACT.read_bytes())['cases']


def case_bounds(spec):
    # A closed benchmark recipe boundary, not an expanded public API contract.
    assert spec in case_specs()
    shape = spec['shape']; ndim = len(shape); count = math.prod(shape)
    assert ndim in (2, 3) and all(type(n) is int and 0 < n <= 192 for n in shape)
    assert count <= 24576 and spec['dtype'] in ('int16', 'int32', 'int64')
    assert spec['layout'] in ('C', 'F') and type(spec['cache']) is bool
    assert spec['operation'] in ('getters', 'constructor') and spec['max_label'] in (2, 8)
    assert spec['pattern'] in ('saved2d', 'saved3d', 'checkerboard', 'sparse')
    for name in ('spacing', 'offset'):
        if spec[name] is not None:
            assert len(spec[name]) == ndim
            assert all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 4 for v in spec[name])
    assert spec['native_partition'] in ('f64', 'integer-provider')
    return dict(input_elements=count, max_regions=spec['max_label'], max_output_elements=count*ndim,
                max_coordinate=192, max_integer_origin_plus_index=384,
                max_absolute_scaled_plus_offset=1540)


def fixture(spec, np):
    case_bounds(spec)
    shape = tuple(spec['shape'])
    if spec['pattern'] in ('saved2d', 'saved3d'):
        image = np.zeros(shape, dtype=spec['dtype'], order=spec['layout'])
        points = [(1, 1), (1, 2), (2, 1)] if len(shape) == 2 else [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        for point in points:
            image[point] = 1
        image[tuple(n-1 for n in shape)] = 2
    elif spec['pattern'] == 'sparse':
        # Reuse the exact existing constructor-sparse recipe.
        from regionprops_state import fixture as state_fixture
        return state_fixture(dict(spec, operation='constructor'), np)
    else:
        image = (np.indices(shape, dtype=np.int32).sum(axis=0) % 2 + 1).astype(spec['dtype'])
        if spec['layout'] == 'F':
            image = np.asfortranarray(image)
    options = dict(cache=spec['cache'], spacing=None if spec['spacing'] is None else tuple(spec['spacing']),
                   offset=None if spec['offset'] is None else tuple(spec['offset']))
    return image, options


def make_operation(call, image, options, spec):
    if spec['operation'] == 'constructor':
        def operation():
            regions = call(image, **options)
            return regions[0].coords_scaled
        return operation
    regions = call(image, **options)
    region = regions[0]
    if spec['cache']:
        region.image
    return lambda: region.coords_scaled


def checked_output(actual, expected, buffers, np):
    # Use the existing raw/storage snapshot, not a second geometry oracle.
    from draw_coordinates import snapshot
    assert type(actual) is type(expected) is np.ndarray
    assert actual is not expected and not np.shares_memory(actual, expected)
    assert actual.flags.owndata and actual.base is None
    assert actual.dtype.kind in ('f', 'i') and actual.dtype.itemsize == 8
    assert actual.ndim == 2 and actual.shape[1] in (2, 3)
    for buffer in buffers:
        assert actual is not buffer and not np.shares_memory(actual, buffer)
    row = snapshot((actual,))
    assert row == snapshot((expected,))
    # Preserve raw equality in the checker; hex makes the retained report JSON.
    row['arrays'][0]['raw'] = row['arrays'][0]['raw'].hex()
    return row


def require_prepared_source(manifest, root, revision, read_source):
    # Rebind at the child-process boundary; the controller also verifies inputs.
    assert set(manifest['source']) == {'revision', 'files_sha256'}
    assert manifest['source']['revision'] == revision
    assert read_source(root) == manifest['source']


def main():
    raw_contract = CONTRACT.read_bytes(); contract = json.loads(raw_contract)
    assert contract['state'] == 'FROZEN_READY_FOR_ROOT_FIRST_COHORT', 'prospective: genuine source/input gate has not closed'
    assert not contract['pending_integrated_source_paths']
    for name, pin in contract['source_closure'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin, name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--repeat', type=int, choices=[7], default=7)
    parser.add_argument('--backend-role', choices=['native', 'prior-cpp'], default='native')
    parser.add_argument('--pass-index', type=int, choices=range(1, 5), required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--input-manifest-sha256', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('build/ci-regionprops-coords-scaled.json'))
    args = parser.parse_args()
    if args.backend_role == 'native':
        assert set(contract['native_source_closure']) == set(contract['required_native_source_paths'])
        for name, pin in contract['native_source_closure'].items():
            assert hashlib.sha256((args.source_root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin, name
    assert args.backend_role == contract['comparison']['process_roles'][args.pass_index-1]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(args.deadline_utc)
    journal = args.output.with_suffix('.batches.jsonl')
    assert not args.output.exists() and not journal.exists()
    manifest_raw = args.input_manifest.read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == args.input_manifest_sha256
    manifest = json.loads(manifest_raw)
    assert manifest['schema'] == 'ci-offline-measurement-v1'
    assert len(manifest['wheels']) == len(manifest['runtime']['versions']) == 14
    manifest_core = [pin for name,pin in manifest['runtime']['packages_sha256']['skimagexx'].items()
                     if name.startswith('_core.') and name.endswith('.so')]
    assert len(manifest_core) == 1
    if args.backend_role == 'prior-cpp':
        assert args.source_revision == contract['baseline']['source']
        assert manifest_core[0] == contract['baseline']['core_sha256']
    sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'benchmarks')]
    from montage_sustained import measure
    from provenance import source_fingerprint
    from ci_measurement_artifacts import source_state
    require_prepared_source(manifest, args.source_root, args.source_revision, source_state)
    from regionprops_state import input_state, fixture as state_fixture, case_bounds as state_bounds
    for function in (input_state, state_fixture, state_bounds):
        assert hashlib.sha256(inspect.getsource(function).encode()).hexdigest() == contract['reused_state_method_sha256'][function.__name__]
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx import _compat
    original = importlib.import_module('skimage.measure._regionprops')
    public = importlib.import_module('skimage.measure')
    owned = importlib.import_module('skimagexx._regionprops_state')
    assert sys.version == manifest['runtime']['python']
    for name, version in [('numpy', np.__version__), ('scipy', scipy.__version__), ('scikit-image', skimage.__version__)]:
        assert manifest['runtime']['versions'][name] == version
    assert skimage.__version__ == '0.26.0'
    for name, pin in contract['original_source_sha256'].items():
        assert hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == pin
    original_binary = Path(importlib.import_module('scipy.ndimage._nd_image').__file__)
    binary_sha = hashlib.sha256(original_binary.read_bytes()).hexdigest()
    assert binary_sha == contract['original_binary_sha256'][np.__version__]
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest() == contract['method_sha256']
    before = source_fingerprint(args.source_root)
    core = Path(xx._core.__file__); core_sha = hashlib.sha256(core.read_bytes()).hexdigest()
    assert core_sha == manifest_core[0]
    pools = {name: os.environ[name] for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')}
    assert set(pools.values()) == {'1'} and xx.get_num_threads() == 1
    xx.set_fallback_enabled(False)
    native, upstream = xx.measure.regionprops, public.regionprops
    assert native is owned.regionprops and inspect.signature(native) == inspect.signature(upstream)
    provider = original.RegionProperties
    assert issubclass(owned.RegionProperties, provider)
    assert [owned.RegionProperties.__module__, owned.RegionProperties.__qualname__] == contract['type_policy']['backend_qualified_type']
    assert [provider.__module__, provider.__qualname__] == contract['type_policy']['original_qualified_type']
    native_fget = owned.RegionProperties.coords_scaled.fget
    entries = {} if args.backend_role == 'prior-cpp' else {
        getattr(xx._core, 'regionprops_state_coords_scaled_'+stage): stage for stage in STAGES}
    assert all(inspect.isbuiltin(entry) for entry in entries)
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    cpu_info = Path('/proc/cpuinfo')
    cpu = next((line.partition(':')[2].strip() for line in cpu_info.read_text().splitlines() if line.startswith('model name')), platform.processor()) if cpu_info.exists() else platform.processor()
    rows = []
    for index, spec in enumerate(case_specs()):
        image, options = fixture(spec, np); state = input_state(image); pixels = image.tobytes(order='C')
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            reference_operation = make_operation(upstream, image, options, spec)
            expected, expected_again = reference_operation(), reference_operation()
        assert not notices
        checked_output(expected_again, expected, [image], np)
        calls, phase_rows, worker_snapshots, worker_calls, constructors = [], [], [], [], []
        protected_buffers = [image]
        def denied(*a, **kw):
            raise AssertionError('coords_scaled owned stage forwarded to original')
        forbidden_codes = {provider.coords_scaled.fget.__code__} if entries else set()
        if entries:
            # _cached's wrapper code is shared; deny only its original image body.
            for cell in provider.image.fget.__closure__ or ():
                body = cell.cell_contents
                if inspect.isfunction(body) and body.__name__ == 'image':
                    forbidden_codes.add(body.__code__)
        # Direct original fget code catches an alias cached before monkeypatching.
        def trace(frame, event, value):
            if event == 'call' and frame.f_code in forbidden_codes:
                denied()
            if event == 'c_return' and value in entries:
                calls.append(entries[value])
            if event == 'c_return' and value is xx._core.get_num_threads and frame.f_code is native_fget.__code__:
                worker_calls.append('c_return')
            if event == 'return' and frame.f_code is native_fget.__code__ and entries:
                protected_buffers.append(frame.f_locals['mask'])
                worker_snapshots.append(frame.f_locals['workers'])
                phase_rows.append({name: frame.f_locals[name] for name in ('index_owned', 'origin_owned', 'multiply_owned', 'add_owned')})
                phase_rows[-1]['native_accepted'] = frame.f_locals['native_result'] is not None
            if event == 'return' and frame.f_code is owned.regionprops.__code__:
                assert type(value) is list and value and all(type(region) is owned.RegionProperties for region in value)
                protected_buffers.extend(array for region in value for array in (region._spacing, region._offset))
                constructors.append([type(region).__module__+'.'+type(region).__qualname__ for region in value])
        saved = []
        if entries:
            saved = [(public, 'regionprops', public.regionprops), (original, 'regionprops', original.regionprops),
                     (owned, '_provider_regionprops', owned._provider_regionprops), (provider, 'coords_scaled', provider.coords_scaled),
                     (np, 'argwhere', np.argwhere)]
        previous = sys.getprofile()
        try:
            for obj, name, value in saved:
                setattr(obj, name, property(denied) if isinstance(value, property) else denied)
            sys.setprofile(trace)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                native_operation = make_operation(native, image, options, spec)
                actual, actual_again = native_operation(), native_operation()
        finally:
            sys.setprofile(previous)
            for obj, name, value in saved:
                setattr(obj, name, value)
        assert not notices
        snapshot = checked_output(actual, expected, protected_buffers+[actual_again, expected_again], np)
        checked_output(actual_again, expected, protected_buffers+[actual, expected_again], np)
        if entries:
            assert len(phase_rows) == 2 and worker_snapshots == [1, 1] and len(worker_calls) == 2
            assert len(constructors) == (2 if spec['operation'] == 'constructor' else 1)
            integer = spec['native_partition'] == 'integer-provider'
            assert calls == list(STAGES[:3] if integer else STAGES)*2
            for row in phase_rows:
                assert row == dict(index_owned=True, origin_owned=True, multiply_owned=not integer,
                                   add_owned=None if integer else True, native_accepted=not integer)
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        assert xx.get_num_threads() == 1 and not _compat._fallback_enabled
        assert input_state(image) == state and image.tobytes(order='C') == pixels
        # All spies/checks restored: actual public allocation/destruction only.
        timing = measure(reference_operation, native_operation, 7, 100, journal=journal,
                         case_index=index, case_id=spec['id'], pass_index=args.pass_index, deadline=args.deadline_utc)
        assert np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer
        assert xx.get_num_threads() == 1 and input_state(image) == state and image.tobytes(order='C') == pixels
        rows.append(dict(case_id=spec['id'], parameters=spec, actual_C_returns=calls, phases=phase_rows,
                         native_accepted=bool(entries) and spec['native_partition']=='f64',
                         worker_snapshots=worker_snapshots, constructor_types=constructors, validation=snapshot, **timing))
    assert sum(sum(map(len, row['samples_ms'])) for row in rows) == 84
    assert source_fingerprint(args.source_root) == before and CONTRACT.read_bytes() == raw_contract
    assert hashlib.sha256(core.read_bytes()).hexdigest() == core_sha
    report = dict(metadata=dict(source_revision=args.source_revision, backend_role=args.backend_role,
        pass_index=args.pass_index, deadline_utc=args.deadline_utc, threads=1, repeat=7, target_ms=100,
        warmup_calibration_min_ms=250, raw_samples=84, reference_pool_limits=pools, cpu_model=cpu,
        logical_cpu_count=os.cpu_count(), avx2_available=bool(xx.has_avx2()), python=sys.version,
        numpy=np.__version__, scipy=scipy.__version__, reference=skimage.__version__, **before,
        core_sha256=core_sha, original_binary_sha256=binary_sha, input_manifest_sha256=args.input_manifest_sha256,
        method_sha256=contract['method_sha256'], fallback_enabled=False,
        contract_sha256=hashlib.sha256(raw_contract).hexdigest(), scope=contract['scope']), results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# coords_scaled property workloads', '', contract['scope'], '', '| Case | Original ms | Backend ms | Original/backend |', '|---|---:|---:|---:|']
    lines += [f"| {row['case_id']} | {row['reference_ms']:.9f} | {row['cpp_ms']:.9f} | {row['speedup']:.9f}x |" for row in rows]
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
