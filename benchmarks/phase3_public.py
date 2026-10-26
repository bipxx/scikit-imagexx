"""Eight public APIs versus scikit-image 0.26.0, one worker and seven paired rounds.

Use --check-only locally. Measurements require an isolated artifact-fed Actions
job, --prepared-source and --wheel. Calls include allocation, provider work and
output destruction; fixture creation is excluded. This measures owned Python
orchestration plus C++ stages and NumPy providers, not an isolated kernel.
Importing this module or listing case_specs() needs no scientific packages.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys
from types import SimpleNamespace
import traceback
import warnings
import zipfile

try:
    from .provenance import source_fingerprint
except ImportError:
    from provenance import source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
FOOTPRINTS = ('ellipse', 'octagon', 'octahedron', 'star', 'footprint_from_sequence')
COLOCALIZATION = ('intersection_coeff', 'manders_coloc_coeff', 'manders_overlap_coeff')
APIS = tuple('morphology.' + name for name in FOOTPRINTS) + tuple('measure.' + name for name in COLOCALIZATION)
ROUNDS = 7
TARGET_MS = 15
DEFAULT_POLICY = dict(divide='warn', over='warn', under='ignore', invalid='warn')
POOL_VARIABLES = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')
STAGES = ('footprint_shapes_ellipse', 'footprint_shapes_hull', 'footprint_shapes_l1',
          'footprint_shapes_atoms', 'footprint_shapes_cross_plan', 'footprint_shapes_cross',
          'footprint_shapes_dilate', 'footprint_shapes_full_mask', 'coloc_binary',
          'coloc_count', 'coloc_joint_count', 'coloc_min_negative', 'coloc_gather',
          'coloc_sum', 'coloc_product')
PROVIDERS = {
    'footprint': 'NumPy allocation/dtype, mgrid/complex magnitude and cross sums; C++ geometry, raster, fill and dilation stages.',
    'colocalization': 'NumPy scalar arithmetic plus explicit unsupported array-stage providers; C++ admitted array stages. No full-C++ or leaf-causal speedup claim.',
}


def _factory(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / (name + '_cases.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FOOTPRINT_FACTORY = _factory('footprint_shapes')
COLOC_FACTORY = _factory('colocalization')


def case_specs():
    """Fixed prospective workload definitions; no outputs or tuning decisions."""
    rows = []
    sizes = {
        'ellipse': ((3, 2), (63, 47), (511, 383)),
        'octagon': ((3, 2), (65, 32), (511, 256)),
        'octahedron': ((2,), (15,), (47,)),
        'star': ((3,), (43,), (341,)),
    }
    for name, arguments in sizes.items():
        modes = (None,) if name == 'star' else (None, 'crosses' if name == 'ellipse' else 'sequence')
        for tier, values in zip(('tiny', 'cache', 'large'), arguments):
            for dtype in ('uint8', 'float64'):
                for mode in modes:
                    options = dict(dtype=dtype)
                    if name != 'star': options['decomposition'] = mode
                    rows.append(dict(id=f'morphology.{name}/{tier}/{dtype}/{mode}', api='morphology.' + name,
                        family='footprint', tier=tier, dtype=dtype, layout='generated',
                        work_kind='raster' if mode is None else 'decomposition_plan',
                        fixture=dict(api=name, args=list(values), kwargs=options)))
    for tier, repeats in (('tiny', 2), ('cache', 64), ('large', 128)):
        for layout in ('C', 'F', 'reverse'):
            rows.append(dict(id=f'morphology.footprint_from_sequence/{tier}/{layout}',
                api='morphology.footprint_from_sequence', family='footprint', tier=tier,
                dtype='uint8', layout=layout, work_kind='sequence_expansion',
                fixture=dict(api='footprint_from_sequence', args=[],
                    kwargs=dict(pattern='cross', layout=layout, future_repetitions=repeats))))
    rows.append(dict(id='morphology.octagon/tiny/even-sequence-warning', api='morphology.octagon',
        family='footprint', tier='tiny', dtype='uint8', layout='generated', work_kind='decomposition_plan',
        fixture=dict(api='octagon', args=[4, 3], kwargs=dict(dtype='uint8', decomposition='sequence'))))
    coloc = COLOC_FACTORY.benchmark_cases()
    # Keep all eighteen original prospective specs, adding one cache/F case per API.
    coloc += [dict(row, id=row['id'] + '-F', layout='F') for row in coloc if row['name'] == 'cache']
    for spec in coloc:
        rows.append(dict(id='measure.' + spec['id'], api='measure.' + spec['api'],
            family='colocalization', tier=spec['name'], dtype=spec['dtype'], layout=spec['layout'],
            work_kind='array_metric', fixture=spec))
    return rows


def _runtime():
    for name in POOL_VARIABLES: os.environ[name] = '1'
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
    import numpy as np
    import scipy
    import skimage
    import skimagexx as xx
    from skimagexx import _compat, _footprint_shapes, _colocalization
    from skimagexx.coverage import native_functions
    from threadpoolctl import threadpool_info, threadpool_limits
    try:
        from . import resize
        from .run import measure
    except ImportError:
        import resize
        from run import measure
    # Existing stdlib transport helper; no new artifact schema is introduced.
    transport_spec = importlib.util.spec_from_file_location(
        'phase3_artifact_transport', ROOT / 'tools/ci_measurement_artifacts.py')
    transport = importlib.util.module_from_spec(transport_spec)
    transport_spec.loader.exec_module(transport)
    if skimage.__version__ != '0.26.0': raise RuntimeError('reference must be exactly scikit-image 0.26.0')
    if _footprint_shapes._core is not xx._core or _colocalization._core is not xx._core:
        raise RuntimeError('public families do not share the ordinary core')
    if not set(APIS) <= set(native_functions()): raise RuntimeError('all eight public APIs must be registered')
    functions = {}
    traps = []
    for api in APIS:
        module, name = api.split('.')
        original_module = importlib.import_module('skimage.' + module)
        public_module = importlib.import_module('skimagexx.' + module)
        original, native = getattr(original_module, name), getattr(public_module, name)
        owned = _footprint_shapes if module == 'morphology' else _colocalization
        if native is not getattr(owned, name) or inspect.signature(native) != inspect.signature(original):
            raise RuntimeError('public alias/signature differs: ' + api)
        functions[api] = (original, native)
        traps += [(original_module, name), (importlib.import_module(original.__module__), name)]
    traps += [(importlib.import_module('skimage.morphology.convex_hull'), 'convex_hull_image'),
              (importlib.import_module('skimage.morphology.gray'), 'dilation')]
    traps = list(dict.fromkeys(traps))
    for name in STAGES:
        function = getattr(xx._core, name)
        if not inspect.isbuiltin(function) or function.__module__ != xx._core.__name__:
            raise RuntimeError('stage is not an actual core builtin: ' + name)
    return SimpleNamespace(np=np, scipy=scipy, skimage=skimage, xx=xx, compat=_compat,
        helpers=resize, measure=measure, functions=functions, traps=traps,
        pool_info=threadpool_info, pool_limits=threadpool_limits, package_files=transport.package_files)


def _pools(runtime):
    pools = runtime.pool_info()
    if not pools or any(row['num_threads'] != 1 for row in pools):
        raise RuntimeError('loaded upstream BLAS/OpenMP pools must report one thread')
    if runtime.xx.get_num_threads() != 1 or runtime.compat._fallback_enabled:
        raise RuntimeError('native worker/fallback policy changed')
    if runtime.np.geterr() != DEFAULT_POLICY or runtime.np.geterrcall() is not None:
        raise RuntimeError('benchmark requires the ordinary NumPy floating policy')
    return pools


def _verify_wheel(runtime, path):
    with zipfile.ZipFile(path) as archive:
        expected = {name.removeprefix('skimagexx/'): hashlib.sha256(archive.read(name)).hexdigest()
                    for name in archive.namelist() if name.startswith('skimagexx/') and not name.endswith('/')}
    installed = runtime.package_files(Path(runtime.xx.__file__).resolve().parent)
    if not expected or installed != expected: raise RuntimeError('installed complete package differs from exact wheel')
    return dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), package_files=len(expected))


def _fixture(runtime, spec):
    if spec['family'] == 'footprint':
        return FOOTPRINT_FACTORY.fixture(spec['fixture'], runtime.np)
    args, kwargs, _ = COLOC_FACTORY.fixture(spec['fixture'], runtime.np, [])
    return args, kwargs


def _arrays(value, np):
    if isinstance(value, np.ndarray): return [value]
    if isinstance(value, dict): return [a for item in value.values() for a in _arrays(item, np)]
    if isinstance(value, (tuple, list)): return [a for item in value for a in _arrays(item, np)]
    return []


def _snapshot(value, np, inputs=()):
    if isinstance(value, np.ndarray):
        return dict(type=type(value).__name__, dtype=value.dtype.str, shape=list(value.shape),
            strides=list(value.strides), own=bool(value.flags.owndata), writeable=bool(value.flags.writeable),
            raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest(),
            input_aliases=[[value is a, bool(np.shares_memory(value, a))] for a in inputs])
    if isinstance(value, np.generic): return dict(type=type(value).__name__, dtype=value.dtype.str, raw=value.tobytes().hex())
    if isinstance(value, np.dtype): return dict(dtype=value.str)
    if isinstance(value, (tuple, list)):
        return dict(type=type(value).__name__, items=[_snapshot(v, np, inputs) for v in value])
    if isinstance(value, dict): return {k: _snapshot(v, np, inputs) for k, v in value.items()}
    return dict(type=type(value).__name__, value=repr(value))


@contextmanager
def _native_probe(runtime, witness):
    """Untimed actual returns and original-algorithm traps, restored before timing."""
    saved = [(module, name, getattr(module, name)) for module, name in runtime.traps]
    stages = {name: getattr(runtime.xx._core, name) for name in STAGES}
    def denied(*args, **kwargs):
        witness['forward_attempts'] += 1
        raise RuntimeError('original algorithm forwarding')
    def tracked(name, function):
        def call(*args, **kwargs):
            value = function(*args, **kwargs)
            witness['stages'].append(dict(name=name, admitted=not name.startswith('coloc_') or value is not None,
                validation_only=name == 'coloc_binary' and value is not None and value[0] is False))
            return value
        return call
    def profile(frame, event, value):
        if event == 'c_return' and getattr(value, '__name__', None) in stages and getattr(value, '__module__', None) == runtime.xx._core.__name__:
            witness['c_returns'].append(value.__name__)
    if sys.getprofile() is not None: raise RuntimeError('an external profiler is active')
    try:
        for module, name, _ in saved: setattr(module, name, denied)
        for name, function in stages.items(): setattr(runtime.xx._core, name, tracked(name, function))
        sys.setprofile(profile)
        yield
    finally:
        sys.setprofile(None)
        for name, function in stages.items(): setattr(runtime.xx._core, name, function)
        for module, name, function in saved: setattr(module, name, function)


def _check_case(runtime, spec, row):
    np = runtime.np
    args, options = _fixture(runtime, spec)
    inputs = _arrays((args, options), np)
    before = _snapshot((args, options), np)
    original, native = runtime.functions[spec['api']]
    policy, handler, buffer = np.geterr(), np.geterrcall(), np.getbufsize()
    expected, notices, error = runtime.helpers.observe(lambda: original(*args, **options))
    row.update(input=before, reference=_snapshot(expected, np, inputs), reference_warnings=notices,
               reference_error=None if error is None else [type(error).__name__, str(error)])
    if error is not None: raise RuntimeError('prospective benchmark case unexpectedly raises in the original') from error
    if _snapshot((args, options), np) != before: raise RuntimeError('original mutated benchmark input')
    if np.geterr() != policy or np.geterrcall() is not handler or np.getbufsize() != buffer:
        raise RuntimeError('original changed floating policy/handler/buffer')
    witness = dict(stages=[], c_returns=[], forward_attempts=0)
    row['native_witness'] = witness
    with _native_probe(runtime, witness):
        actual, actual_notices, actual_error = runtime.helpers.observe(lambda: native(*args, **options))
    row.update(current=_snapshot(actual, np, inputs), current_warnings=actual_notices,
               current_error=None if actual_error is None else [type(actual_error).__name__, str(actual_error)])
    if actual_error is not None: raise actual_error
    if row['reference'] != row['current'] or notices != actual_notices:
        raise RuntimeError('exact public result/structure/raw bits/warnings differ')
    if _snapshot((args, options), np) != before: raise RuntimeError('candidate mutated benchmark input')
    if np.geterr() != policy or np.geterrcall() is not handler or np.getbufsize() != buffer:
        raise RuntimeError('floating policy/handler/buffer changed')
    if witness['forward_attempts'] or not witness['c_returns']:
        raise RuntimeError('no genuine compiled return or original forwarding detected')
    if [s['name'] for s in witness['stages']] != witness['c_returns']:
        raise RuntimeError('stage observer and real builtin returns differ')
    row['admitted_stage_returns'] = sum(s['admitted'] and not s['validation_only'] for s in witness['stages'])
    row['provider_frontier'] = PROVIDERS[spec['family']]


def _save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--prepared-source', type=Path)
    parser.add_argument('--wheel', '--seal-wheel', dest='wheel', type=Path,
                        help='Exact product wheel to verify; no new seal is created')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error('choose a fresh output path; prior results are retained')
    if bool(args.prepared_source) != bool(args.wheel): parser.error('--prepared-source and --wheel must be supplied together')
    if not args.check_only and (not args.prepared_source or os.environ.get('GITHUB_ACTIONS') != 'true'):
        parser.error('timing requires an artifact-fed isolated Actions job, --prepared-source and --wheel')
    if sys.flags.optimize: parser.error('assertions must be enabled')
    report = dict(schema='phase3-public-v1', status='preflight', metadata=dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv if argv is None else list(argv),
        threads=1, rounds=ROUNDS, target_ms=TARGET_MS, sample_order_seed=81,
        scope='Allocation-inclusive public calls including output destruction; fixture creation excluded.',
        ratio_scope='Owned orchestration+C++ stages+NumPy providers, not isolated kernels or full native API coverage.',
        warning_policy='Compare warnings before timing; suppress display symmetrically during timing.',
        check_only=args.check_only), checks=[], results=[], failures=[])
    _save(report, args.output)
    runtime = None
    previous_workers = previous_fallback = None
    try:
        runtime = _runtime()
        xx = runtime.xx
        previous_workers = xx.get_num_threads()
        previous_fallback = xx.set_fallback_enabled(False)
        xx.set_num_threads(1)
        with runtime.pool_limits(limits=1):
            pools = _pools(runtime)
            helpers = runtime.helpers
            report['metadata'].update(platform=platform.platform(), python=sys.version,
                numpy=runtime.np.__version__, scipy=runtime.scipy.__version__, scikit_image=runtime.skimage.__version__,
                avx2=xx.has_avx2(), cpu=helpers.cpu_metadata(), source_revision=helpers.source_revision(ROOT),
                **source_fingerprint(ROOT), current_binary_sha256=helpers.binary_hashes(),
                current_python_sha256=helpers.python_hashes(), thread_pools=pools,
                numpy_policy=runtime.np.geterr(), numpy_buffer=runtime.np.getbufsize(),
                github_run_id=os.environ.get('GITHUB_RUN_ID'), github_job=os.environ.get('GITHUB_JOB'))
            if args.prepared_source:
                report['metadata']['prepared_source'] = helpers.verify_prepared(ROOT, args.prepared_source)
                report['metadata']['wheel'] = _verify_wheel(runtime, args.wheel)
            specs = case_specs()
            if len(specs) != 73 or len({s['id'] for s in specs}) != 73 or {s['api'] for s in specs} != set(APIS):
                raise RuntimeError('fixed complete workload set differs')
            report['metadata']['case_specs'] = specs
            report['metadata']['case_spec_sha256'] = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
            for spec in specs:
                row = dict(id=spec['id'], api=spec['api'], status='checking')
                try:
                    _pools(runtime)
                    _check_case(runtime, spec, row)
                    row['status'] = 'passed'
                except Exception as error:
                    row.update(status='failed', error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                    report['failures'].append(row)
                report['checks'].append(row)
                _save(report, args.output)
            if report['failures']:
                report['status'] = 'failed-parity'
                return 1
            if args.check_only:
                report['status'] = 'complete-parity-check'
                return 0
            report['status'] = 'measuring'
            rng = random.Random(81)
            orders = []
            for _ in range(ROUNDS):
                order = ['scikit-image', 'current']; rng.shuffle(order); orders.append(order)
            report['metadata']['paired_sample_order'] = orders
            for index, spec in enumerate(specs):
                _pools(runtime)
                args_, options = _fixture(runtime, spec)
                if _snapshot((args_, options), runtime.np) != report['checks'][index]['input']:
                    raise RuntimeError('actual timed fixture differs from full preflight')
                original, native = runtime.functions[spec['api']]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    timing = runtime.measure(lambda: original(*args_, **options),
                                             lambda: native(*args_, **options), ROUNDS, TARGET_MS)
                row = dict(id=spec['id'], api=spec['api'], work_kind=spec['work_kind'], threads=1,
                    provider_frontier=PROVIDERS[spec['family']],
                    absolute_overhead_ms=timing['cpp_ms'] - timing['reference_ms'],
                    slower_than_original=timing['speedup'] < 1, **timing)
                report['results'].append(row)
                _save(report, args.output)
                _pools(runtime)
                if runtime.np.getbufsize() != report['metadata']['numpy_buffer']:
                    raise RuntimeError('timed public calls changed the NumPy buffer size')
                if _snapshot((args_, options), runtime.np) != report['checks'][index]['input']:
                    raise RuntimeError('timed public calls mutated inputs')
                print(f"{spec['id']}: {timing['reference_ms']:.6f}/{timing['cpp_ms']:.6f} ms, {timing['speedup']:.3f}x", flush=True)
            if helpers.verify_prepared(ROOT, args.prepared_source) != report['metadata']['prepared_source']:
                raise RuntimeError('prepared product changed during timing')
            if _verify_wheel(runtime, args.wheel) != report['metadata']['wheel']:
                raise RuntimeError('wheel/installed package changed during timing')
            report['slower_cases_by_absolute_overhead'] = sorted(
                [dict(id=r['id'], absolute_overhead_ms=r['absolute_overhead_ms'], speedup=r['speedup'])
                 for r in report['results'] if r['slower_than_original']],
                key=lambda row: row['absolute_overhead_ms'], reverse=True)
            report['status'] = 'complete-measurement'
            return 0
    except Exception as error:
        report['failures'].append(dict(phase=report['status'], type=type(error).__name__,
                                       message=str(error), traceback=traceback.format_exc()))
        report['status'] = 'failed'
        return 1
    finally:
        if runtime is not None and previous_workers is not None:
            runtime.xx.set_num_threads(previous_workers)
            if previous_fallback is not None: runtime.xx.set_fallback_enabled(previous_fallback)
        _save(report, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
