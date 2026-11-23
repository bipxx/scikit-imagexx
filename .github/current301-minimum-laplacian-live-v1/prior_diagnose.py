"""Compare one exact live stock pyramid expansion with its hosted candidate."""

import argparse
from base64 import urlsafe_b64encode
import csv
import ctypes
from hashlib import sha256
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import warnings

HERE = Path(__file__).resolve().parent


def digest(raw):
    return sha256(raw).hexdigest()


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def prove_inputs(source, stock, candidate, extraction, pins):
    assert sys.platform == 'linux' and sys.version_info[:3] == (3, 11, 17)
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert len(os.sched_getaffinity(0)) == 1
    assert not any(name in os.environ for name in ('GH_TOKEN', 'PYTHONPATH', 'PYTHONHOME', 'NPY_DISABLE_CPU_FEATURES',
                                                  'LD_PRELOAD', 'SKIMAGEXX_DISABLE_SIMD'))
    assert (source / '.git').exists()
    git = ['git', '--no-optional-locks', '-C', str(source)]
    assert subprocess.check_output([*git, 'rev-parse', 'HEAD', 'HEAD^{tree}'], text=True).splitlines() == [pins['source'], pins['tree']]
    assert not subprocess.check_output([*git, 'status', '--porcelain=v1', '--untracked-files=all'])
    for name, pin in pins['source_paths'].items():
        assert digest(subprocess.check_output([*git, 'show', 'HEAD:' + name])) == pin['git_raw_sha256']
        assert digest((source / name).read_bytes().replace(b'\r\n', b'\n')) == pin['normalized_lf_sha256']
    x = json.loads(extraction.read_bytes())
    assert (x['source'], x['artifact'], x['core_sha256'], x['wheel_sha256'], x['members']) == (
        pins['source'], pins['artifact'], pins['candidate_core_sha256'], pins['candidate_wheel']['sha256'], 265)
    assert digest((candidate / x['record_path']).read_bytes()) == x['record_sha256']
    assert digest((candidate / x['core_member']).read_bytes()) == x['core_sha256']
    for name in ('filters.py', '_pyramid_spatial.py'):
        actual = (candidate / 'skimagexx' / name).read_bytes()
        assert digest(actual) == pins['source_paths']['python/skimagexx/' + name]['git_raw_sha256']

    versions = {dist.metadata['Name'].lower().replace('_', '-'): dist for dist in metadata.distributions(path=[str(stock)])}
    assert {name: versions[name].version for name in pins['stock_versions']} == pins['stock_versions']
    records = {}
    for name, pin in pins['stock_members'].items():
        module = name.split('/')[0]
        key = 'scikit-image' if module == 'skimage' else module
        dist = versions[key]
        if key not in records:
            path = Path(dist.locate_file(next(file for file in dist.files if file.name == 'RECORD')))
            with path.open(encoding='utf-8', newline='') as stream:
                records[key] = {row[0]: row[1:] for row in csv.reader(stream)}
        member = stock / name
        assert Path(dist.locate_file(name)) == member and member.stat().st_size == pin['bytes']
        code = urlsafe_b64encode(bytes.fromhex(pin['sha256'])).decode().rstrip('=')
        assert records[key][name] == ['sha256=' + code, str(pin['bytes'])]
        assert digest(member.read_bytes()) == pin['sha256']
    return x


def cpu_state(umath):
    wanted = {'vendor_id', 'cpu family', 'model', 'model name', 'stepping', 'flags', 'Microcode'}
    affinity = sorted(os.sched_getaffinity(0))
    blocks = ({k.strip(): v.strip() for k, _, v in (line.partition(':') for line in block.splitlines())}
              for block in Path('/proc/cpuinfo').read_text().split('\n\n'))
    processor, = (block for block in blocks if block.get('processor') == str(affinity[0]))
    fields = {key: value for key, value in processor.items() if key in wanted or key == 'microcode'}
    return {'uname': list(platform.uname()), 'os_release': platform.freedesktop_os_release(),
            'libc': platform.libc_ver(), 'cpuinfo': fields,
            'affinity': affinity, 'numpy_cpu_features': umath.__cpu_features__,
            'numpy_cpu_baseline': umath.__cpu_baseline__, 'numpy_cpu_dispatch': umath.__cpu_dispatch__}


def compare(np, left, right):
    assert left.dtype == right.dtype == np.dtype('float64') and left.shape == right.shape
    assert left.size <= 126 * 178 * 3 and np.isfinite(left).all() and np.isfinite(right).all()
    a, b = left.view(np.uint64), right.view(np.uint64)
    assert not ((a | b) >> 63).any()
    locations = np.flatnonzero(a.ravel() != b.ravel())
    first = []
    for i in locations[:8]:
        y = np.unravel_index(i, left.shape)
        first.append(dict(index=[int(k) for k in y], stock_word=f'{int(a[y]):016x}', candidate_word=f'{int(b[y]):016x}'))
    return {'different_words': len(locations), 'first': first,
            'max_ulp': int((np.maximum(a, b) - np.minimum(a, b)).max()),
            'max_absolute': float(np.abs(left - right).max())}


def diagnose(source, stock, candidate, out, pins, extraction):
    proof = prove_inputs(source, stock, candidate, extraction, pins)
    sys.path[:0] = [str(candidate), str(stock)]
    import numpy as np
    from numpy.core import _multiarray_umath as umath
    import scipy
    from scipy import ndimage
    from scipy.ndimage import _filters, _nd_image, _interpolation, _ni_support
    import skimage
    from skimage import transform
    from skimage.transform import pyramids, _warps
    from skimage._shared import filters as stock_filters
    from skimage.util import dtype
    import skimagexx as xx
    from skimagexx import _core, _pyramid_spatial, filters

    modules = [umath, _filters, _nd_image, _interpolation, _ni_support, pyramids, _warps, stock_filters, dtype]
    assert {Path(m.__file__).relative_to(stock).as_posix() for m in modules} == set(pins['stock_members'])
    assert Path(_core.__file__) == candidate / proof['core_member']
    assert Path(filters.__file__) == candidate / 'skimagexx/filters.py'
    assert Path(_pyramid_spatial.__file__) == candidate / 'skimagexx/_pyramid_spatial.py'
    assert _filters._nd_image is _nd_image and ndimage.gaussian_filter1d is _filters.gaussian_filter1d
    assert stock_filters.ndi is ndimage and pyramids.gaussian is stock_filters.gaussian
    stock_api, candidate_api = transform.pyramid_expand, xx.transform.pyramid_expand
    assert stock_api is pyramids.pyramid_expand and candidate_api is _pyramid_spatial.pyramid_expand
    assert inspect.signature(stock_api) == inspect.signature(candidate_api)
    fixture = runpy.run_path(str(source / 'tests/pyramid_spatial_cases.py'))
    spec, = [item for item in fixture['FIXED'] if item['id'] == pins['case']]
    image = fixture['fixture'](spec['shape'], spec['dtype'], np)
    state = fixture['array_state']
    assert state(image)['sha256'] == pins['frozen_input_sha256'] and image.shape == (63, 89, 3)
    fenv = ctypes.CDLL('libm.so.6').fegetround
    fenv.restype, fenv.argtypes = ctypes.c_int, []
    fenv_before = {'fegetround': fenv(), 'numpy_geterr': np.geterr(), 'numpy_bufsize': np.getbufsize()}
    cpu = cpu_state(umath)
    xx.set_num_threads(1)
    previous = xx.set_fallback_enabled(False)
    values = {'stock': {}, 'candidate': {}}
    stages = {
        pyramids.convert_to_float.__code__: ('stock', 'convert'),
        _pyramid_spatial._convert.__code__: ('candidate', 'convert'),
        pyramids.resize.__code__: ('stock', 'resize'),
        _pyramid_spatial.resize.__code__: ('candidate', 'resize'),
        pyramids._smooth.__code__: ('stock', 'gaussian'),
        _pyramid_spatial._smooth.__code__: ('candidate', 'gaussian'),
    }
    stock_passes, candidate_args, c_returns = [], {}, []

    def trace(frame, event, value):
        code = frame.f_code
        if event == 'c_return' and (value is _core.gaussian_nd or value is _core.resize_zoom_typed):
            c_returns.append(value.__name__)
        if event != 'return' or value is None:
            return
        if code in stages:
            side, stage = stages[code]
            caller = stock_api.__code__ if side == 'stock' else _pyramid_spatial._pyramid.__code__
            if frame.f_back.f_code is caller:
                assert stage not in values[side]
                values[side][stage] = value
        if code is _filters.gaussian_filter1d.__code__:
            local = frame.f_locals
            weights = np.asarray(local['weights'], dtype=np.float64).copy()
            stock_passes.append((local['axis'], weights, value.copy()))
        if code is filters._gaussian_nd.__code__:
            local = frame.f_locals
            candidate_args.update({name: local[name] for name in ('kernels', 'axes', 'codes', 'constant', 'preserve', 'target')})

    def run(api):
        prior = sys.getprofile()
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:
                sys.setprofile(trace)
                result = api(image, **spec['options'])
            finally:
                sys.setprofile(prior)
        return result, [[item.category.__name__, str(item.message)] for item in notices]

    try:
        stock_out, stock_warnings = run(stock_api)
        candidate_out, candidate_warnings = run(candidate_api)
    finally:
        xx.set_fallback_enabled(previous)
    assert state(image)['sha256'] == pins['frozen_input_sha256']
    assert c_returns == ['gaussian_nd', 'resize_zoom_typed', 'gaussian_nd']
    assert stock_passes and [p[0] for p in stock_passes] == candidate_args['axes'] == [0, 1]
    assert candidate_args['target'] == np.dtype('float64') and candidate_out.shape == stock_out.shape == (126, 178, 3)
    stage_states = {side: {stage: state(array) for stage, array in rows.items()} for side, rows in values.items()}
    arrays_equal = state(stock_out)['sha256'] == state(candidate_out)['sha256']
    frozen = pins['frozen_output_sha256']
    reproduced = state(candidate_out)['sha256'] == pins['hosted_observed_candidate_sha256']
    frozen_resize = pins['frozen_stages_sha256']['resize']
    same_resize = all(stage_states[side]['resize']['sha256'] == frozen_resize for side in ('stock', 'candidate'))
    coeffs = lambda a: [f'{int(word):016x}' for word in a.view(np.uint64)]
    kernels = [{'axis': axis, 'stock': coeffs(weights), 'candidate': coeffs(candidate_args['kernels'][index])}
               for index, (axis, weights, _) in enumerate(stock_passes)]
    assert all(len(k['stock']) == len(k['candidate']) == 7 for k in kernels)
    axis_witnesses = []
    matched_axis_input_and_weights = None
    if not arrays_equal and same_resize:
        current = values['candidate']['resize']
        for index, (axis, _, expected) in enumerate(stock_passes):
            current, flags = _core.gaussian_nd(current, [candidate_args['kernels'][index]], [axis],
                [candidate_args['codes'][index]], candidate_args['constant'],
                candidate_args['preserve'], candidate_args['target'])
            axis_witnesses.append(dict(axis=axis, flags=flags, output=state(current), stock=state(expected),
                                       **compare(np, expected, current)))
        assert state(current)['sha256'] == state(candidate_out)['sha256']
        first = next(index for index, row in enumerate(axis_witnesses) if row['different_words'])
        if kernels[first]['stock'] != kernels[first]['candidate']:
            axis, weights, stock_axis_output = stock_passes[first]
            stock_axis_input = values['stock']['resize'] if first == 0 else stock_passes[first - 1][2]
            same, flags = _core.gaussian_nd(stock_axis_input, [weights], [axis], [candidate_args['codes'][first]],
                candidate_args['constant'], candidate_args['preserve'], candidate_args['target'])
            matched_axis_input_and_weights = dict(axis=axis, flags=flags, native_output=state(same),
                stock_output=state(stock_axis_output), **compare(np, stock_axis_output, same))
    if not same_resize:
        explanation = 'upstream_stage_changed'
    elif arrays_equal and reproduced:
        explanation = 'live_stock_candidate_match_original_host_mismatch_from_frozen'
    elif arrays_equal and state(stock_out)['sha256'] == frozen:
        explanation = 'frozen_matches_both_failed_suite_mismatch_not_reproduced'
    elif arrays_equal:
        explanation = 'live_stock_candidate_match_new_unfrozen_words'
    else:
        explanation = 'live_candidate_stock_diverge'
    report = dict(source=pins['source'], case=pins['case'], extraction=proof, state=explanation,
        candidate_reproduced_failed_hash=reproduced, resized_frozen_input_match=same_resize,
        versions={'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-image': skimage.__version__},
        native_libm_image='libm.so.6', floating_before=fenv_before,
        floating_after={'fegetround': fenv(), 'numpy_geterr': np.geterr(), 'numpy_bufsize': np.getbufsize()},
        cpu=cpu, stages=stage_states, stock_output=state(stock_out), candidate_output=state(candidate_out),
        frozen_output_sha256=frozen, expected_failed_candidate_sha256=pins['hosted_observed_candidate_sha256'],
        kernels=kernels, first_axis_witnesses=axis_witnesses, matched_axis_input_and_weights=matched_axis_input_and_weights,
        whole=compare(np, stock_out, candidate_out), original_calls=1, candidate_calls=1,
        extra_native_axis_calls=len(axis_witnesses) + int(matched_axis_input_and_weights is not None), timings=0,
        warnings={'stock': stock_warnings, 'candidate': candidate_warnings}, c_returns=c_returns)
    for side, array in (('stock', stock_out), ('candidate', candidate_out)):
        with (out / (side + '-output.f64')).open('xb') as stream:
            stream.write(array.tobytes())
    write(out / 'diagnostic.json', report)
    print(explanation, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-only', action='store_true')
    for name in ('source', 'stock', 'candidate', 'out', 'extraction'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    gate = runpy.run_path(str(HERE / 'extract_candidate.py'))
    pins = json.loads(gate['pinned'](HERE / 'pins.json', gate['PINS_SHA']))
    if args.entry_only:
        assert not any(getattr(args, key) for key in ('source', 'stock', 'candidate', 'out', 'extraction'))
        assert not {'numpy', 'scipy', 'skimage', 'skimagexx'}.intersection(sys.modules)
        print(json.dumps({'case': pins['case'], 'source': pins['source'], 'pinned_files': len(pins['source_paths'])}))
        raise SystemExit(0)
    diagnose(*(getattr(args, key).resolve() for key in ('source', 'stock', 'candidate', 'out')),
             pins, args.extraction.resolve())
