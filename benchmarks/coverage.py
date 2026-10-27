"""Every native public function versus original scikit-image, strict dispatch."""
from run import measure
import argparse
import fnmatch
from datetime import datetime, timezone
from provenance import source_fingerprint
import importlib
import json
from pathlib import Path
import platform
import warnings
import numpy as np
import skimage
import skimagexx as xx
from skimagexx.coverage import euler_default_error
from skimagexx.coverage import native_functions, native_dtypes, execution_kind, needs_color_input, default_call, parity_tolerance, MOMENTS_FUNCTIONS, DERIVED_MOMENTS_FUNCTIONS, COORDS_MOMENTS_FUNCTIONS, moments_default_error, shannon_default_error, block_default_error, crofton_default_error, perimeter_default_error


def select_apis(apis, modules=None, patterns=None):
    apis = [api for api in apis if modules is None or api.rsplit('.', 1)[0] in modules]
    if patterns:
        selected = set()
        for pattern in patterns:
            matches = [api for api in apis if fnmatch.fnmatchcase(api, pattern)]
            if not matches:
                raise ValueError(f'API pattern {pattern!r} matches no registered entry points in selected modules')
            selected.update(matches)
        apis = [api for api in apis if api in selected]
    if not apis:
        raise ValueError('module selection matches no registered entry points')
    return apis


def dtype_metadata(value):
    if isinstance(value,(tuple,list)):return [dtype_metadata(item) for item in value]
    if type(value).__name__=='ArrayMap':
        return {field:dtype_metadata(getattr(value,field)) for field in ('in_values','out_values')}
    dtype=getattr(value,'dtype',None)
    metadata=getattr(dtype,'metadata',None)
    return None if metadata is None else dict(metadata)


def comparison_error(actual,expected,rtol,atol):
    if isinstance(expected,(tuple,list)):
        assert type(actual) is type(expected) and len(actual)==len(expected)
        return max((comparison_error(a,e,rtol,atol) for a,e in zip(actual,expected)),default=0.)
    if type(expected).__name__=='ArrayMap':
        assert type(actual).__name__==type(expected).__name__
        return max(comparison_error(getattr(actual,field),getattr(expected,field),rtol,atol)
                   for field in ('in_values','out_values'))
    a,e=np.asarray(actual),np.asarray(expected)
    assert a.dtype==e.dtype and a.shape==e.shape
    np.testing.assert_allclose(a,e,rtol=rtol,atol=atol)
    return float(np.max(np.abs(a.astype(float)-e.astype(float)))) if a.size else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype-metadata',action='store_true',help='Attach and verify custom NumPy dtype tags')
    parser.add_argument('--sizes', nargs='+', type=int, default=[256, 1024])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--modules', nargs='+', help='Benchmark only these registry modules; default covers all')
    parser.add_argument('--api', action='append', help='Repeatable API glob intersected with --modules; every pattern must match')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/function-coverage.json'))
    args = parser.parse_args()
    if min(args.sizes+args.threads+[args.repeat]) < 1:
        parser.error('sizes, threads, and repeat must be positive')
    xx.set_fallback_enabled(False)
    # Binary morphology deprecation text is tested separately before timing.
    warnings.filterwarnings('ignore',message=r'`binary_.*` is deprecated since version 0.26',category=FutureWarning)
    try:
        apis = select_apis(native_functions(), args.modules, args.api)
    except ValueError as error:
        parser.error(str(error))
    if 'restoration.denoise_tv_bregman' in apis and min(args.sizes) < 4:
        parser.error('Bregman original-library benchmarks require both spatial axes >=4; unsafe geometry remains untimed intake metadata')
    max_tree_paths={'morphology.'+name for name in ('max_tree','max_tree_local_maxima',
                    'area_opening','area_closing','diameter_opening','diameter_closing')}
    if max_tree_paths.intersection(apis) and min(args.sizes) < 4:
        parser.error('max-tree original-library benchmarks require both spatial axes >=4; unsafe geometry remains untimed intake metadata')
    lpi_apis = {'filters.filter_forward', 'filters.filter_inverse', 'filters.wiener'}
    reference_fft_workers = None
    if (lpi_apis | {'filters.butterworth'}).intersection(apis):
        import scipy.fft as reference_fft
        reference_fft_workers = {'before': reference_fft.get_workers()}
        assert reference_fft_workers['before'] == 1
    lpi_scope = ('Python Gaussian impulse callback and NumPy coordinates, casting, padding, products, '
                 'gain/Wiener arithmetic, permutation, cropping and absolute value; owned C++ FFT only. '
                 'Each timed call includes the default callback and construction of a new filter and transfer cache.')
    window_scope = ('SciPy signal.get_window and NumPy radial coordinates and clipping remain providers; '
                    'owned C++ order-0/1 sampling. Default Hann/order1, shape-only with no pixel input; '
                    'the dtype label denotes the float64 output, not an input-pixel dtype.')
    moments_scope = ('NumPy dtype promotion, coordinate powers and layout preparation, and the literal '
                     'analytical central-moment and centroid-ratio Python/NumPy shell remain providers; '
                     'owned typed C++ per-axis projections capture floating-point flags and use the shared global workers.')
    derived_scope = 'Fixed owning 4x4 moment tensors are prepared outside timing; dtype is the actual mu/nu input dtype, not an image label. Normalization uses bounded native batch arithmetic for assessed contiguous 4x4 real defaults; its scalar fallback retains NumPy powers, first division, promotion and assignment with an owned final divide. Hu keeps dtype conversion and output wrapping around owned C++ polynomial arithmetic. Each timed public call includes its allocation, eligibility checks and conversions. Integer normalization has exact source-error coverage and no timing row.'
    inertia_scope = ('Automatic central moments use owned per-axis projections; NumPy coordinate powers, '
                     'analytical preparation, diagonal summation and indexing remain providers. Finite real tensor '
                     'assembly is owned C++; exceptional arithmetic retains local NumPy expressions. Both timed '
                     'public calls include moments, tensor allocation and preparation; genuine original central '
                     'moments are separately prepared outside timing only for the frozen per-cell checker.')
    eigvals_scope = ('Default image calls include owned moments, inertia assembly and the existing C++ symmetric eigensolver; '
                     'NumPy preparation, clipping and scalar-list wrapping remain providers. Unsupported solver domains '
                     'retain NumPy eigvalsh primitives without native solver credit. Genuine original central moments '
                     'and tensor are prepared outside timing only for the frozen solver-plus-Weyl checker.')
    coords_scope = ('Coordinate inputs are owning N-by-2 point clouds prepared outside timing from the first '
                    '2*N seeded gray values; size denotes points and dtype is the actual coordinate dtype. '
                    'NumPy tuple/center/cast/power preparation remains explicit; ordinary finite real typed '
                    'products and reduction are owned C++. K1, complex and exceptional domains use NumPy '
                    'primitives without reduction credit. Genuine reference monomials are captured outside '
                    'timing for the frozen 2*gamma_(N+D)*S policy.')
    shannon_scope = ('Ordinary real sorted unique counts and binary64 entropy arithmetic are owned C++. '
                     'NumPy array preparation and unsupported unique types remain explicit providers; '
                     'NumPy scalar boxing and Python logarithmic base conversion retain source semantics. '
                     'Each timed default image call includes counts, entropy and allocation. Genuine reference '
                     'nats are captured outside timing for the frozen entropy-only and base-division policy.')
    perimeter_scope = ('Literal uint8 casting, source coefficient construction and scalar boxing remain explicit providers. '
                       'Binary erosion, byte border subtraction, modulo-256 stencil counts and the 50-term sum are owned C++. '
                       'Default cases use the existing seed 2026 gray image thresholded above 127 for uint8 or above 0.5 '
                       'for floats, then cast back to the same dtype: binary masks with roughly half the pixels foreground, not arbitrary labels. '
                       'Mask construction stays outside timing; public casting, allocation, erosion, counts and sum stay inside. '
                       'The frozen exact-rational 2*gamma_100*S checker runs outside timing.')
    crofton_scope = ('Ordinary real thresholding, zero-boundary 2x2 stencil histogram and the 16-term weighted sum '
                     'are owned C++. NumPy/Python coefficient construction, scalar boxing and input/storage '
                     'preparation remain explicit; special input types retain NumPy/SciPy count providers. '
                     'Timed default directions4 calls include all preparation, histogram, weighted arithmetic '
                     'and allocation. The frozen exact-rational 2*gamma_32*S checker runs outside timing.')
    block_scope = ('Ordinary default sums own typed block arithmetic and output partitioning. '
                   'NumPy padding, block-view metadata, storage/layout preparation and allocation remain providers; '
                   'custom functions/kwargs, complex padding and exceptional/type-frontier sums use literal '
                   'primitive reducers without native sum credit. Half sums require original padded native/aligned C '
                   'storage and block volume<=np.getbufsize(); other half layouts or buffered reductions remain '
                   'literal providers and open native-coverage work. Timed default image calls include the entire '
                   'padding, summation and allocation path; the frozen per-block 2*gamma_m*S checker runs outside timing.')
    euler_scope = ('NumPy binary conversion, padding, LUT selection and scalar boxing remain providers; '
                   'configuration histograms and bounded exact integer LUT arithmetic are owned C++. '
                   'Default image preparation is outside timing; each public call includes its binary/pad '
                   'and private-storage preparation and allocation. The shared Snum<=2**53 exact checker '
                   'runs before timing; the canonical 2D/3D cohort is benchmarks/euler_number.py.')
    provider_frontiers = {api: lpi_scope for api in apis if api in lpi_apis}
    if 'measure.euler_number' in apis:
        provider_frontiers['measure.euler_number'] = euler_scope
    if 'measure.perimeter' in apis:
        provider_frontiers['measure.perimeter'] = perimeter_scope
    if 'measure.perimeter_crofton' in apis:
        provider_frontiers['measure.perimeter_crofton'] = crofton_scope
    if 'measure.block_reduce' in apis:
        provider_frontiers['measure.block_reduce'] = block_scope
    if 'measure.shannon_entropy' in apis:
        provider_frontiers['measure.shannon_entropy'] = shannon_scope
    provider_frontiers.update({api: moments_scope for api in apis if api in MOMENTS_FUNCTIONS})
    provider_frontiers.update({api: derived_scope for api in apis if api in DERIVED_MOMENTS_FUNCTIONS})
    provider_frontiers.update({api: coords_scope for api in apis if api in COORDS_MOMENTS_FUNCTIONS})
    if 'measure.inertia_tensor' in apis:
        provider_frontiers['measure.inertia_tensor'] = inertia_scope
    if 'measure.inertia_tensor_eigvals' in apis:
        provider_frontiers['measure.inertia_tensor_eigvals'] = eigvals_scope
    if 'filters.window' in apis:
        provider_frontiers['filters.window'] = window_scope
    if 'filters.butterworth' in apis:
        provider_frontiers['filters.butterworth'] = (
            'NumPy padding, frequency-grid/mask arithmetic, spectrum multiplication and cropping; owned C++ FFT only')
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    results = []
    for size_index, size in enumerate(args.sizes):
        for dtype in (np.uint8, np.float32, np.float64):
            rng = np.random.default_rng(2026)
            gray = rng.integers(0, 256, (size, size), dtype=np.uint8)
            rgb = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            if dtype != np.uint8:
                gray = (gray/255).astype(dtype); rgb = (rgb/255).astype(dtype)
            if args.dtype_metadata:
                tagged=np.dtype(dtype,metadata={'units':'linear','source':'benchmark'})
                gray=gray.view(tagged);rgb=rgb.view(tagged)
            for api in apis:
                # Derived moments have fixed 4x4 inputs, independent of image size.
                if api in DERIVED_MOMENTS_FUNCTIONS and size_index != 0:
                    continue
                # Window consumes only shape, so fixture pixel dtypes are not distinct cases.
                if api == 'filters.window' and dtype != np.float64:
                    continue
                module, name = api.rsplit('.', 1)
                if np.dtype(dtype).name not in native_dtypes(api):
                    continue
                native = getattr(importlib.import_module('skimagexx.'+module), name)
                reference = getattr(importlib.import_module('skimage.'+module), name)
                data = rgb if needs_color_input(api) else gray
                call_args,options = default_call(api,data)
                if args.dtype_metadata and call_args and isinstance(call_args[0],np.ndarray) and call_args[0].dtype.metadata is None:
                    first=call_args[0].view(np.dtype(call_args[0].dtype,metadata={'units':'linear','source':'benchmark'}))
                    call_args=(first,*call_args[1:])
                case_size = 4 if api in DERIVED_MOMENTS_FUNCTIONS else call_args[0].shape[0] if api in COORDS_MOMENTS_FUNCTIONS else size
                case_dtype = call_args[0].dtype.name if api in DERIVED_MOMENTS_FUNCTIONS or api in COORDS_MOMENTS_FUNCTIONS else np.dtype(dtype).name
                expected = reference(*call_args,**options)
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    assert xx.get_num_threads() == threads
                    actual = native(*call_args,**options)
                    assert xx.get_num_threads() == threads
                    if api == 'measure.euler_number':
                        error = euler_default_error(call_args[0], actual, expected)
                    elif api == 'measure.perimeter':
                        error = perimeter_default_error(call_args[0], actual, expected)
                    elif api == 'measure.perimeter_crofton':
                        error = crofton_default_error(call_args[0], actual, expected)
                    elif api == 'measure.block_reduce':
                        error = block_default_error(call_args[0], actual, expected)
                    elif api == 'measure.shannon_entropy':
                        error = shannon_default_error(call_args[0], actual, expected)
                    elif api in COORDS_MOMENTS_FUNCTIONS:
                        error = moments_default_error(api, call_args[0], actual, expected)
                    elif api == 'measure.inertia_tensor_eigvals':
                        error = moments_default_error(api, data, actual, expected)
                    elif api == 'measure.inertia_tensor':
                        error = moments_default_error(api, data, actual, expected)
                    elif api in MOMENTS_FUNCTIONS:
                        error = moments_default_error(api, data, actual, expected)
                    elif api in DERIVED_MOMENTS_FUNCTIONS:
                        error = moments_default_error(api, call_args[0], actual, expected)
                    else:
                        rtol,atol = parity_tolerance(api,np.dtype(dtype),image=data,expected=expected)
                        error=comparison_error(actual,expected,rtol,atol)
                    if args.dtype_metadata:assert dtype_metadata(actual)==dtype_metadata(expected)
                    timing = measure(lambda: reference(*call_args,**options), lambda: native(*call_args,**options), args.repeat, 15)
                    assert xx.get_num_threads() == threads
                    backend = 'identity' if call_args and actual is call_args[0] else execution_kind(api)
                    results.append({'api': api, 'size': case_size, 'dtype': case_dtype,
                                    **({'input_shape': list(call_args[0].shape)} if api in DERIVED_MOMENTS_FUNCTIONS or api in COORDS_MOMENTS_FUNCTIONS else {}),
                                    **({'size_unit': 'points'} if api in COORDS_MOMENTS_FUNCTIONS else {}),
                                    'threads': threads, 'backend': backend, 'layout':'dtype-metadata' if args.dtype_metadata else 'standard',
                                    'output_dtype_metadata':dtype_metadata(actual), 'max_abs_error':
                                    error,
                                    **timing})
                    print(f'{api:24} {case_size} {case_dtype} t={threads}: {timing["speedup"]:.2f}x',flush=True)
    if reference_fft_workers is not None:
        reference_fft_workers['after'] = reference_fft.get_workers()
        assert reference_fft_workers['after'] == 1
    assert set(r['api'] for r in results) == set(apis)
    report = {'metadata': {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                           'cpu': platform.processor(), 'reference': skimage.__version__, 'numpy': np.__version__,
                           **fingerprint, 'fallback_enabled': False, 'avx2': xx.has_avx2(),
                           'selected_apis': apis, 'dtype_metadata':args.dtype_metadata,
                           'reference_fft_workers': reference_fft_workers,
                           'provider_frontiers': provider_frontiers,
                           'special_default_arguments': {'color.deltaE_cie76':'second input is first[..., ::-1]; raw Lab values',
                               'color.deltaE_ciede94':'second input is first[..., ::-1]; raw Lab; graphic-arts weights',
                               'color.deltaE_ciede2000':'second input is first[..., ::-1]; raw Lab; kL=kC=kH=1',
                               'color.deltaE_cmc':'second input is first[..., ::-1]; raw Lab; kL=kC=1',
                               'color.convert_colorspace':"fromspace='RGB', tospace='HSV'",
                               'color.separate_stains':'conv_matrix=hed_from_rgb',
                               'color.combine_stains':'conv_matrix=rgb_from_hed'},
                           'repeat': args.repeat, 'timing': 'allocation-inclusive, warmed Python calls; original reference pools limited to one thread'},
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    lines = ['# Native function performance versus scikit-image'+(' with dtype metadata' if args.dtype_metadata else ''), '',
             'Every native public function is checked before timing. Fallback is disabled; '
             'forwarding to scikit-image cannot count as native performance. Reference thread pools use one thread. '
             'Numbers are medians of warmed allocation-inclusive calls on seeded synthetic images. '
             'Identity calls return the input unchanged; metadata calls return small values/arrays without processing pixels. '
             'Speedup is reference/current; below 1 means slower. This does not prove full API parity or universal performance.'
             + (' ' + lpi_scope if lpi_apis.intersection(apis) else '')
             + (' ' + window_scope if 'filters.window' in apis else '')
             + (' ' + moments_scope if set(MOMENTS_FUNCTIONS).intersection(apis) else '')
             + (' ' + inertia_scope if 'measure.inertia_tensor' in apis else '')
             + (' ' + eigvals_scope if 'measure.inertia_tensor_eigvals' in apis else '')
             + (' ' + derived_scope if set(DERIVED_MOMENTS_FUNCTIONS).intersection(apis) else '')
             + (' ' + coords_scope if set(COORDS_MOMENTS_FUNCTIONS).intersection(apis) else '')
             + (' ' + shannon_scope if 'measure.shannon_entropy' in apis else '')
             + (' ' + euler_scope if 'measure.euler_number' in apis else '')
             + (' ' + perimeter_scope if 'measure.perimeter' in apis else '')
             + (' ' + crofton_scope if 'measure.perimeter_crofton' in apis else '')
             + (' ' + block_scope if 'measure.block_reduce' in apis else ''), '',
             '| API | Size | Dtype | Workers | Execution | Reference ms | Current ms | Speedup |', '|---|---:|---|---:|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["size"]} | {r["dtype"]} | {r["threads"]} | {r["backend"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
