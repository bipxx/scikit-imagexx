"""Untimed diagnostics for host-dependent ridge-Hessian zero decisions."""
import json
import math
import platform
import numpy as np
from skimage import feature, filters
import skimagexx as xx
from skimagexx import _core


def main():
    xx.set_fallback_enabled(False)
    xx.set_num_threads(1)
    numpy_core = np.core if np.__version__.startswith('1.') else np._core
    report = {'platform': platform.platform(), 'numpy': np.__version__,
              'cpu_features': numpy_core._multiarray_umath.__cpu_features__, 'cases': []}
    for dtype in (np.bool_, np.float64):
        for black in (True, False):
            rng = np.random.default_rng(4691)
            shape = (7, 9, 11)
            if dtype is np.float64:
                a = rng.normal(size=shape)
            else:
                a = (rng.bit_generator.random_raw(np.prod(shape)).reshape(shape) & 1).astype(dtype)
            working = a.astype(float)
            if not black:
                working = -working
            options = dict(mode='mirror', cval=-.3, use_gaussian_derivatives=True)
            original = feature.hessian_matrix(working, 1, **options)
            native = xx.feature.hessian_matrix(working, 1, **options)
            eigen = feature.hessian_matrix_eigvals(original)
            eigen = np.take_along_axis(eigen, abs(eigen).argsort(0), 0)
            second, third = np.maximum(eigen[1:], 1e-10)
            ra = second/third
            rb = abs(eigen[0])/np.sqrt(second*third)
            norm = np.sqrt((eigen**2).sum(0))
            exponents = [-ra**2/.5, -rb**2/.5, -norm**2/450.]
            numpy_exp = [np.exp(x) for x in exponents]
            libm_exp = [np.array([math.exp(float(v)) for v in x.flat]).reshape(shape) for x in exponents]
            def response(values):
                return (1-values[0])*values[1]*(1-values[2])
            expected = response(numpy_exp)
            scalar = response(libm_exp)
            supplied, _ = _core.frangi_accumulate(original, None, np.dtype('float64'),
                                                 .5, .5, 450., False, False, False, False)
            result = xx.filters.hessian(a, sigmas=[1], mode='mirror', cval=-.3, black_ridges=black)
            upstream = filters.hessian(a, sigmas=[1], mode='mirror', cval=-.3, black_ridges=black)
            bad = ((expected == 0) != (supplied == 0)) | ((expected == 0) != (scalar == 0)) | (abs(result-upstream) > 2e-13)
            points = []
            for index in zip(*np.nonzero(bad)):
                points.append(dict(index=list(map(int, index)),
                    exponents=[float(x[index]).hex() for x in exponents],
                    numpy_exp=[float(x[index]).hex() for x in numpy_exp],
                    libm_exp=[float(x[index]).hex() for x in libm_exp],
                    reference_response=float(expected[index]), libm_response=float(scalar[index]),
                    native_supplied_response=float(supplied[index]), result=float(result[index]),
                    upstream=float(upstream[index])))
            report['cases'].append(dict(dtype=np.dtype(dtype).name, black=black,
                hessian_bit_equal=[bool(np.array_equal(x.view(np.uint64), y.view(np.uint64))) for x,y in zip(original,native)],
                hessian_max_abs=[float(np.max(abs(x-y))) for x,y in zip(original,native)], points=points))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
