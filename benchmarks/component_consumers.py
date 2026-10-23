"""Fixed public consumer parity and isolated allocation-inclusive measurements.

Reference-only authoring/checks never exercise native calls or timers. Raster,
output-reset and metadata/error cases retain separate classifications. Returned
ArrayMap observations inspect sparse vectors/state and never densify maps.
"""
try:
    from .run import measure
    from .provenance import source_fingerprint
    from .resize import save, cpu_metadata, source_revision
except ImportError:
    from run import measure
    from provenance import source_fingerprint
    from resize import save, cpu_metadata, source_revision
import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import traceback
import warnings
import zipfile
import numpy as np
import scipy
import skimage
import skimagexx as xx
from skimagexx.coverage import native_functions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ci_measurement_artifacts as artifacts

MANIFEST = ROOT / 'coverage/contracts/component-consumers-benchmark.json'
APIS = ('morphology.remove_small_objects', 'morphology.remove_small_holes',
        'segmentation.clear_border', 'segmentation.relabel_sequential')
DTYPES = ('bool', 'u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'u8', 'i8', 'f2', 'f4', 'f8')
LAYOUTS = ('C', 'F', 'reverse', 'step', 'endian', 'broadcast', 'readonly', 'unaligned', 'tagged')
VERSION = 'component-consumers-benchmark-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_text_sha256(path):
    """Fingerprint source text independently of Git checkout line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


@dataclass(frozen=True)
class Case:
    api: str
    id: str
    fixture: dict
    timed: bool = False
    kind: str = 'contract-only'
    source: str = ''


def corpus():
    shapes = [(17, 19), (512, 513), (7, 11, 13), (17, 129, 131)]
    for a, api in enumerate(APIS):
        types = ('u1', 'i2', 'u8') if a == 3 else ('bool', 'i2', 'f4') if a == 2 else ('bool', 'u1', 'i2')
        for s, shape in enumerate(shapes):
            for d, dtype in enumerate(types):
                for p, pattern in enumerate(('dense', 'stripes', 'checkerboard', 'multivalued')):
                    options = {'max_size': (3, 64)[p % 2], 'connectivity': 1 if p % 2 else len(shape)} if a < 2 else {'buffer_size': p % 2, 'bgval': 0} if a == 2 else {'offset': (1, 2, 127, 256)[p]}
                    yield Case(api, f'raster/{s}/{d}/{p}', dict(shape=shape, dtype=dtype, pattern=pattern,
                        layout=LAYOUTS[(s + 2*d + p) % len(LAYOUTS)], options=options,
                        mask='sparse' if a == 2 and p == 3 else None), True, 'raster', 'test_large_inputs_cross_worker_threshold + real_dtype_and_layout_contract')
        for dtype in DTYPES:
            for layout in LAYOUTS:
                yield Case(api, f'dtype-layout/{dtype}/{layout}', dict(dtype=dtype, layout=layout), source='test_real_dtype_and_layout_contract')
        for dtype in ('bool', 'i2', 'f4'):
            for i, shape in enumerate(((), (0,), (0, 3), (2, 0, 3), (1,), (7,), (2, 3, 4), (2, 1, 3), (2, 2, 2, 2))):
                yield Case(api, f'rank/{dtype}/{i}', dict(dtype=dtype, shape=shape), source='test_rank_empty_scalar_and_singleton')
        for i in range(12):
            yield Case(api, f'parameter/{i}', dict(parameter=i), source='test_threshold_offset_connectivity_and_validation_order')
        for i in range(8):
            yield Case(api, f'validation/{i}', dict(validation=i), source='test_negative_nonfinite_and_joint_error_precedence')
        if a < 3:
            for size, shape in enumerate(((5, 7), (512, 513))):
                for out in ('self', 'copy', 'F', 'strided', 'bool', 'float', 'different-shape', 'readonly'):
                    yield Case(api, f'out/{size}/{out}', dict(shape=shape, dtype='bool' if a == 1 else 'i2', out=out),
                        out in ('self', 'copy', 'F', 'strided'), 'raster-mutation', 'test_out_identity_casting_partial_mutation_and_errors')
        else:
            for dtype in ('i1', 'u1', 'u8'):
                for offset in (1, 2, 127, 128, 255, 256):
                    yield Case(api, f'map-state/{dtype}/{offset}', dict(dtype=dtype, options={'offset': offset}), source='test_relabel_sorted_values_and_map_aliases + returned_ArrayMap_scope')


def array_state(value):
    if isinstance(value, np.ndarray):
        return dict(type=type(value).__name__, dtype=value.dtype.str, metadata=repr(value.dtype.metadata),
            shape=list(value.shape), strides=list(value.strides), owned=bool(value.flags.owndata),
            c=bool(value.flags.c_contiguous), f=bool(value.flags.f_contiguous),
            writeable=bool(value.flags.writeable), aligned=bool(value.flags.aligned), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return dict(type=type(value).__name__, dtype=value.dtype.str, bytes=value.tobytes().hex())
    return dict(type=type(value).__name__, value=repr(value))


def attempt(function):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try: result, error = array_state(function()), None
        except Exception as caught: result, error = None, dict(type=type(caught).__name__, message=str(caught))
    return dict(result=result, error=error, warnings=[dict(type=type(n.message).__name__, message=str(n.message)) for n in notices])


def describe(value, tracked):
    arrays = list(tracked.values())
    if isinstance(value, np.ndarray):
        return dict(array=array_state(value), identity=[value is a for a in arrays], alias=[bool(np.shares_memory(value, a)) for a in arrays])
    if isinstance(value, tuple):
        result = {'tuple': [describe(v, tracked) for v in value]}
        if len(value) == 3 and type(value[1]).__name__ == 'ArrayMap':
            forward, inverse = value[1:]
            result['map_vector_aliases'] = [forward.in_values is inverse.out_values, forward.out_values is inverse.in_values]
        return result
    if type(value).__name__ == 'ArrayMap':
        return dict(type='ArrayMap', in_values=array_state(value.in_values), out_values=array_state(value.out_values),
            dtype=value.dtype.str, dtype_metadata=repr(value.dtype.metadata), direct_len=attempt(value.__len__),
            builtin_len=attempt(lambda: len(value)), repr=repr(value), str=str(value),
            max_str_lines=value._max_str_lines, lazy=value._array is None)
    return array_state(value)


def make_case(case):
    f = case.fixture; api = case.api; a = APIS.index(api)
    shape = tuple(f.get('shape', (5, 7))); dtype = np.dtype(f.get('dtype', 'i2'))
    if f.get('layout') == 'tagged': dtype = np.dtype(dtype, metadata={'unit': 'labels'})
    count = math.prod(shape)
    values = np.arange(count).reshape(shape)
    pattern = f.get('pattern', 'contract')
    if pattern == 'dense': values = np.ones(shape, np.int16)
    elif pattern == 'stripes': values = values // max(shape[-1] if shape else 1, 1) % 3 == 1
    elif pattern == 'checkerboard': values = sum(np.indices(shape)) % 2
    elif pattern == 'multivalued': values = (values // 7 + values % 3) % 5
    else: values = np.resize(np.array([0, 1, 1, 0, 2, 2, 2, 3, 0, 3, 4, 4], np.int16), shape)
    image = values.astype(dtype); options = dict(f.get('options', {})); tracked = {}
    layout = f.get('layout', 'C')
    if layout == 'F': image = np.asfortranarray(image)
    elif layout == 'reverse': image = image[tuple(slice(None, None, -1) for _ in shape)]
    elif layout == 'step':
        owner = np.full(shape[:-1] + (2*shape[-1],), 9, dtype); owner[..., ::2] = image; image = owner[..., ::2]; tracked['image_owner'] = owner
    elif layout == 'endian': image = image.astype(image.dtype.newbyteorder('>'))
    elif layout == 'broadcast': image = np.broadcast_to(image[:1], image.shape)
    elif layout == 'readonly': image.flags.writeable = False
    elif layout == 'unaligned':
        original = image; image = np.ndarray(shape, dtype, buffer=bytearray(image.nbytes+1), offset=1); image[...] = original
    if f.get('mask'): options['mask'] = values % 7 != 0
    if 'parameter' in f:
        i = f['parameter']
        if a < 2:
            old = 'min_size' if a == 0 else 'area_threshold'
            options = [dict(max_size=0), dict(max_size=-1), dict(max_size=2), dict(max_size=2.5), {old: 2},
                {old: 2, 'max_size': 2}, dict(connectivity=0), dict(connectivity=9), dict(connectivity=1.5),
                dict(max_size=None), dict(max_size='bad'), dict(max_size=0)][i]
            if i in (6, 7, 8): image = image.astype(bool)
            if i in (5, 11): image = np.array([-1, 0, 2], np.int16)
        elif a == 2:
            options = [dict(buffer_size=0), dict(buffer_size=1), dict(buffer_size=-1), dict(buffer_size=5), dict(buffer_size=.5),
                dict(buffer_size='bad'), dict(bgval=7), dict(bgval=.5), dict(mask=np.ones(image.shape, bool)),
                dict(mask=np.ones((2, 3), bool)), dict(mask=np.ones(image.shape, np.uint8)), dict(buffer_size=999, mask=np.ones(image.shape, bool))][i]
        else:
            options = {'offset': [0, -1, 1, 2, 127, 128, 255, 256, .5, 1.5, None, '1'][i]}
            if i in (0, 1): image = np.array([-1, 2], np.int16)
    if 'validation' in f:
        i = f['validation']; image = [np.array([-1, 0, 2], np.int16), np.array([0, np.nan, np.inf]), np.array(np.nan),
            np.array([], np.float32), np.array([0, 1], np.float32), np.array([0, 1], np.int16), np.array([0, 1], np.int16), np.array([0, 1], np.int16)][i]
        if a < 2:
            old = 'min_size' if a == 0 else 'area_threshold'
            options = [{}, dict(connectivity='bad'), dict(max_size=0), {}, {old: 2, 'max_size': 2}, {old: 2}, dict(out=np.zeros(2, np.float32)), dict(max_size=0)][i]
        elif a == 2: options = [{}, dict(buffer_size='bad'), dict(mask=np.ones((), np.uint8)), {}, dict(buffer_size=999, mask=np.ones((1,), np.uint8)), dict(mask=np.ones((2,), np.uint8)), dict(mask=np.ones((1,), bool)), dict(buffer_size=-1)][i]
        else: options = {'offset': [1, 0, 1, 0, .5, np.nan, np.inf, True][i]}
    out = f.get('out')
    if out:
        if out == 'self': output = image
        elif out == 'F': output = np.asfortranarray(np.full(image.shape, 9, image.dtype))
        elif out == 'strided':
            owner = np.full(image.shape[:-1] + (2*image.shape[-1],), 9, image.dtype); output = owner[..., ::2]; tracked['out_owner'] = owner
        elif out == 'different-shape': output = np.full((3, 4), 9, image.dtype)
        else:
            output = np.full(image.shape, 9, bool if out == 'bool' else np.float32 if out == 'float' else image.dtype)
            if out == 'readonly': output.flags.writeable = False
        options['out'] = output
    tracked['image'] = image
    tracked.update({k: v for k, v in options.items() if isinstance(v, np.ndarray)})
    return image, options, tracked


def resolve(api, original=False):
    module, name = api.split('.')
    return getattr(importlib.import_module(('skimage' if original else 'skimagexx') + '.' + module), name)


def observe_case(function, case, native=False):
    image, options, tracked = make_case(case)
    before = {k: array_state(v) for k, v in tracked.items()}
    pointers = {k: v.ctypes.data for k, v in tracked.items()}
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            output = function(image, **options); error = None
        except Exception as caught: result, error = None, dict(type=type(caught).__name__, message=str(caught))
        if error is None:
            if native and case.api == APIS[3]:
                assert all(not type(v).__module__.startswith('skimage.') for v in output[1:]), 'forwarded original ArrayMap'
            # Returned-state observation failures are harness failures, not
            # an excuse to reclassify a successful raster as a timing error.
            result = describe(output, tracked)
    return dict(result=result, error=error, warnings=[dict(type=type(w.message).__name__, message=str(w.message)) for w in notices],
        before=before, after={k: array_state(v) for k, v in tracked.items()},
        pointer_stable={k: v.ctypes.data == pointers[k] for k, v in tracked.items()})


def case_spec(case):
    image, options, tracked = make_case(case)
    return dict(api=case.api, case=case.id, fixture=case.fixture, timed_candidate=case.timed, kind=case.kind,
        source_contract=case.source, inputs={k: array_state(v) for k, v in tracked.items()},
        scalar_options={k: repr(v) for k, v in options.items() if not isinstance(v, np.ndarray)})


def timing_callback(function, case):
    image, options, tracked = make_case(case)
    if not case.fixture.get('out'):
        return lambda: function(image, **options), tracked
    def call():
        owned = []
        opts = dict(options)
        source = image
        if case.fixture['out'] == 'self':
            source = image.copy(); opts['out'] = source; owned.append(source)
        elif 'out_owner' in tracked:
            owner = tracked['out_owner'].copy(); opts['out'] = owner[..., ::2]; owned.append(owner)
        else:
            opts['out'] = options['out'].copy(order='K'); owned.append(opts['out'])
        # Retain all reset buffers until measure() destroys the returned value.
        return function(source, **opts), owned
    return call, tracked


def profile():
    assert platform.system() in ('Windows', 'Linux'), 'only the three frozen runtime profiles are supported'
    return ('windows' if platform.system() == 'Windows' else 'linux') + ('-minimum' if np.__version__.startswith('1.') else '-modern')


def seal(capsule):
    manifest = capsule / 'consumer-artifacts.json'; assert not manifest.exists(), 'choose a fresh capsule'
    artifacts.freeze(ROOT, capsule / 'wheels', capsule / 'consumer-runtime.json')
    info = {'schema': 'consumer-exact-offline-v1', 'runtime_manifest_sha256': artifacts.sha(capsule / 'consumer-runtime.json'),
            'corpus_manifest_sha256': artifacts.sha(MANIFEST), 'product_source': source_fingerprint(ROOT)}
    manifest.write_text(json.dumps(info, indent=2)+'\n')
    return verify(capsule)


def verify(capsule):
    manifest = capsule / 'consumer-artifacts.json'; info = json.loads(manifest.read_text())
    assert info['schema'] == 'consumer-exact-offline-v1'
    assert info['runtime_manifest_sha256'] == artifacts.sha(capsule / 'consumer-runtime.json')
    assert info['corpus_manifest_sha256'] == artifacts.sha(MANIFEST) and info['product_source'] == source_fingerprint(ROOT)
    runtime = artifacts.verify(ROOT, capsule / 'wheels', capsule / 'consumer-runtime.json')
    assert {'include/skimagexx/component_consumers.hpp', 'include/skimagexx/components.hpp'} <= set(runtime['runtime']['sdk_sha256'])
    for wheel in (capsule / 'wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if 'skimagexx/__init__.py' not in names: continue
            assert {n for n in names if n.startswith('include/skimagexx/') and not n.endswith('/')} == {
                p.relative_to(ROOT).as_posix() for p in (ROOT / 'include/skimagexx').rglob('*') if p.is_file()}, 'complete SDK header set differs from source'
            python = {n.removeprefix('skimagexx/'): archive.read(n).replace(b'\r\n', b'\n') for n in names if n.startswith('skimagexx/') and n.endswith('.py')}
            assert python == {p.relative_to(ROOT / 'python/skimagexx').as_posix(): p.read_bytes().replace(b'\r\n', b'\n') for p in (ROOT / 'python/skimagexx').rglob('*.py')}
            for name in names:
                if name.startswith('include/skimagexx/') and not name.endswith('/'):
                    assert archive.read(name).replace(b'\r\n', b'\n') == (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
    return dict(manifest_sha256=artifacts.sha(manifest), **info)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', type=Path); parser.add_argument('--prepare-capsule', action='store_true')
    parser.add_argument('--threads', type=int, nargs='+', default=[1]); parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=15.); parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true'); parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args(argv)
    if min(args.threads) < 1 or 1 not in args.threads or len(set(args.threads)) != len(args.threads) or args.repeat < 1 or not math.isfinite(args.target_ms) or args.target_ms <= 0: parser.error('positive unique workers/repeat/target-ms required; include worker1')
    if args.reference_only and (not args.check_only or args.capsule or args.prepare_capsule): parser.error('reference-only requires check-only without capsule')
    if not args.reference_only and args.capsule is None: parser.error('strict public validation requires an exact --capsule')
    if args.prepare_capsule and not args.check_only: parser.error('capsule preparation requires check-only')
    if not args.check_only and (args.output is None or os.environ.get('GITHUB_ACTIONS') != 'true'): parser.error('timing requires isolated Actions host/output; use --check-only locally')
    report = {'metadata': dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, scikit_image=skimage.__version__,
        platform=platform.platform(), **cpu_metadata(), source_revision=source_revision(ROOT), **source_fingerprint(ROOT),
        avx2=xx.has_avx2(), repeat=args.repeat, target_ms=args.target_ms, sample_order_seed=81, cooldown_ms=20,
        corpus_version=VERSION, corpus_manifest_sha256=artifacts.sha(MANIFEST), requested_workers=args.threads, primary_workers=1,
        reference_only=args.reference_only, check_only=args.check_only, public_api_count=4, returned_class='sparse ArrayMap state only; no dense/class speedup claim',
        timing_scope='Public allocations/conversions/result destruction included; mutating/out calls clone/reset working buffers inside timed callback and retain owners through destruction. Nonmutating input construction excluded.',
        warning_policy='Compare all warnings/errors during preflight; suppress warning display symmetrically in measurements', reference_pools=1),
        'status': 'preflight', 'checks': [], 'results': [], 'failures': []}
    save(report, args.output); old_workers = xx.get_num_threads(); old_policy = xx.set_fallback_enabled(False)
    try:
        assert skimage.__version__ == '0.26.0'
        matrix = json.loads(MANIFEST.read_text()); frozen = matrix['profiles'].get(profile(), {})
        if frozen: assert (np.__version__, scipy.__version__) == (frozen['numpy'], frozen['scipy']), 'wrong frozen numerical runtime'
        if not args.reference_only:
            assert set(APIS) <= set(native_functions()), 'public native consumer APIs are not registered'
            assert old_workers == 1, 'default worker1 required'
            from skimagexx import _component_consumers as implementation
            assert implementation._core is xx._core and implementation._core.set_num_threads is xx.set_num_threads
            assert implementation._core.component_consumers_interface_version == 1
            for api in APIS:
                assert resolve(api) is not resolve(api, True) and resolve(api).__module__.startswith('skimagexx.'), 'upstream-forwarded consumer'
            assert frozen, 'reference profiles must be frozen before native acceptance'
            report['metadata']['artifacts'] = seal(args.capsule) if args.prepare_capsule else verify(args.capsule)
        cases = list(corpus()); specs = [case_spec(c) for c in cases]
        assert [(c.api, c.id) for c in cases] == [(c['api'], c['case']) for c in matrix['cases']], 'case IDs differ from frozen source-backed inventory'
        if frozen: assert digest(specs) == frozen['case_spec_sha256'], 'frozen fixture data/layout/parameter contract changed'
        report['metadata'].update(case_count=len(cases), case_spec_sha256=digest(specs), case_specs=specs, profile=profile(),
            reference_functions_sha256={name: artifacts.sha(importlib.import_module(name).__file__) for name in matrix['reference_modules']})
        report['status'] = 'checking'
        for index, case in enumerate(cases):
            row = dict(case_index=index, **specs[index])
            try:
                expected = observe_case(resolve(case.api, True), case)
                classification = 'validation-error' if expected['error'] else case.kind if case.timed else 'metadata-contract'
                row.update(reference=expected, classification=classification, timing_eligible=case.timed and expected['error'] is None)
                if frozen:
                    original = frozen['cases'][index]
                    assert original['id'] == [case.api, case.id] and original['classification'] == classification
                    assert original['observation_sha256'] == digest(expected), 'original differs from frozen reference observation'
                if not args.reference_only:
                    row['by_workers'] = {}
                    for worker in args.threads:
                        xx.set_num_threads(worker); assert xx.get_num_threads() == worker
                        actual = observe_case(resolve(case.api), case, True)
                        assert actual == expected, (actual, expected)
                        row['by_workers'][str(worker)] = digest(actual)
                row['status'] = 'passed'
            except Exception as error:
                row.update(status='failed', exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc()); report['failures'].append(row)
            report['checks'].append(row); save(report, args.output)
            print(f'CHECK {index+1}/{len(cases)} {case.api} {case.id}: {row["status"]}', flush=True)
        if report['failures']: report['status'] = 'failed-parity'; return 1
        report['metadata']['timing_eligible_case_count'] = sum(row['timing_eligible'] for row in report['checks'])
        if not args.check_only:
            report['status'] = 'measuring'
            rng = random.Random(81); orders = []
            for _ in range(args.repeat):
                order = ['scikit-image', 'current']; rng.shuffle(order); orders.append(order)
            report['metadata']['paired_sample_order'] = orders
            for index, case in enumerate(cases):
                if not report['checks'][index]['timing_eligible']: continue
                for worker in args.threads:
                    xx.set_num_threads(worker)
                    left, left_arrays = timing_callback(resolve(case.api, True), case); right, right_arrays = timing_callback(resolve(case.api), case)
                    left_before = {k: array_state(v) for k, v in left_arrays.items()}; right_before = {k: array_state(v) for k, v in right_arrays.items()}
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore'); timing = measure(left, right, args.repeat, args.target_ms)
                    assert {k: array_state(v) for k, v in left_arrays.items()} == left_before
                    assert {k: array_state(v) for k, v in right_arrays.items()} == right_before
                    report['results'].append(dict(case_index=index, api=case.api, case=case.id, kind=case.kind,
                        threads=worker, primary=worker == 1, slower_than_original=timing['speedup'] < 1, **timing)); save(report, args.output)
            report['summary_by_api'] = {}
            for api in APIS:
                report['summary_by_api'][api] = {}
                for worker in args.threads:
                    rows = [r for r in report['results'] if r['api'] == api and r['threads'] == worker]
                    expected_count = sum(r['api'] == api and r['timing_eligible'] for r in report['checks'])
                    assert len(rows) == expected_count
                    report['summary_by_api'][api][str(worker)] = dict(case_count=len(rows), primary=worker == 1,
                        geometric_mean_speedup=math.exp(math.fsum(math.log(r['speedup']) for r in rows)/len(rows)),
                        regressions=[dict(case_index=r['case_index'], speedup=r['speedup']) for r in rows if r['speedup'] < 1])
        if not args.reference_only: assert verify(args.capsule) == report['metadata']['artifacts'], 'source/runtime/package/SDK/corpus seal changed'
        report['status'] = 'complete-reference-check' if args.reference_only else 'complete-parity-check' if args.check_only else 'complete-measurement'
        return 0
    except Exception as error:
        report['failures'].append(dict(phase=report['status'], exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        report['status'] = 'failed'; print(f'FAILED {type(error).__name__}: {error}', file=sys.stderr, flush=True); return 1
    finally:
        xx.set_num_threads(old_workers); xx.set_fallback_enabled(old_policy); save(report, args.output)


if __name__ == '__main__': raise SystemExit(main())
