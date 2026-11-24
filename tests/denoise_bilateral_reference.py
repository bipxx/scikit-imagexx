"""Freeze safe bilateral reference endpoints on the two pinned profile entries."""
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch
import warnings

import numpy as np
import scipy
import skimage
from skimage.restoration import _denoise, _denoise_cy, denoise_bilateral
from skimage._shared import utils
from skimage.util import dtype


VERSIONS = dict(windows=('3.13.14', '2.5.4', '1.18.1', '0.26.0'),
                minimum=('3.11.14', '1.24.4', '1.15.3', '0.26.0'))
CYTHON = dict(windows='0136d13d55789de97b0cdc3df00c7d72334da56293d9461c7969017194539ea1',
              minimum='d662cc8dd8b13ef6fb680d8ca71d61277a87fca3a90f8994507c935699dfcd90')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cases(benchmark):
    for dtype in ('float32', 'float64'):
        gray, = benchmark.make_case(dict(shape=(8, 11), dtype=dtype, kwargs={}))[0]
        rgb, = benchmark.make_case(dict(shape=(9, 13, 3), dtype=dtype, kwargs={}))[0]
        impulse = np.zeros((7, 9), dtype=dtype)
        impulse[1, 1] = 0.75
        impulse[3, 5] = 0.25
        yield dtype+'-gray-default', gray, {}, True
        yield dtype+'-gray-border', gray, dict(win_size=5, sigma_spatial=1.25, sigma_color=0.125,
                                              bins=7, cval=0.21875), True
        yield dtype+'-gray-impulse', impulse, dict(win_size=15, sigma_spatial=2.25, sigma_color=0.11,
                                                  bins=32, cval=0.125), True
        yield dtype+'-rgb-default', rgb, dict(channel_axis=-1), True
        yield dtype+'-rgb-odd-bins', rgb, dict(channel_axis=-1, win_size=5, bins=7, sigma_color=0.125,
                                              sigma_spatial=1.25, cval=0.75), True
        yield dtype+'-rgb-three-bins-witness', rgb, dict(channel_axis=-1, win_size=5, bins=4,
                                                        sigma_color=0.125, cval=0.75), True
        yield dtype+'-zero-total', gray, dict(win_size=25, sigma_spatial=0.25, sigma_color=1/256,
                                             cval=0., bins=10000), True
        yield dtype+'-constant-identity', np.full((7, 8), 0.5, dtype=dtype), {}, True
        yield dtype+'-constant-invalid-mode', np.full((7, 8), 0.5, dtype=dtype), dict(mode='bad'), False
        yield dtype+'-invalid-mode', gray, dict(mode='bad'), False
        yield dtype+'-invalid-gray-rank', rgb, {}, False
        yield dtype+'-invalid-channel-rank', gray, dict(channel_axis=-1), False
        yield dtype+'-empty', np.zeros((0, 7), dtype=dtype), {}, False
        yield dtype+'-safe-strided-gray', gray[:, ::2], {}, False


def snapshot(image):
    return dict(shape=list(image.shape), dtype=image.dtype.str, strides=list(image.strides),
                data_sha256=sha(image.tobytes()), owns=bool(image.flags.owndata),
                nan_count=int(np.count_nonzero(np.isnan(image))))


def observe(target, image, options, captures):
    initial = snapshot(image)
    status = dict(input=initial.copy())
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            result = target(image, **options)
        except Exception as error:
            status['error'] = [type(error).__module__, type(error).__name__, str(error)]
        else:
            status['output'] = snapshot(result)
            status['identity'] = result is image
            status['aliases'] = bool(np.shares_memory(result, image))
    assert snapshot(image) == initial
    status['warnings'] = [[w.category.__module__, w.category.__name__, str(w.message)] for w in notices]
    status['prepared'] = list(captures)
    return status


def load(path):
    spec = spec_from_file_location('bilateral_cases', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    profile, out, benchmark_path = sys.argv[1:]
    observed_version = (platform.python_version(), np.__version__, scipy.__version__, skimage.__version__)
    assert observed_version == VERSIONS[profile]
    assert sha(Path(_denoise_cy.__file__).read_bytes()) == CYTHON[profile]
    assert sha(Path(_denoise.__file__).read_bytes().replace(b'\r\n', b'\n')) == '88481e09372f298e8c08f8297edff7e17f4bd6f168307076439f69f471577c7f'
    assert Path(inspect.getsourcefile(inspect.unwrap(denoise_bilateral))) == Path(_denoise.__file__)
    paths = (_denoise, _denoise_cy, utils, dtype)
    sources = {module.__name__: sha(Path(module.__file__).read_bytes()) for module in paths}
    original_kernel = _denoise._denoise_bilateral
    captures = []

    def record(image, maximum, window, sigma_color, sigma_spatial, bins, mode, cval,
               color_lut, spatial_lut, scratch, output):
        captures.append(dict(field_sha256=sha(image.tobytes()), maximum=float(maximum).hex(),
                             window=window, bins=bins, mode=mode, cval=float(cval).hex(),
                             sigma_color=float(sigma_color).hex(), sigma_spatial=float(sigma_spatial).hex(),
                             color=sha(color_lut.tobytes()), spatial=sha(spatial_lut.tobytes()),
                             spatial_count=int(spatial_lut.size)))
        return original_kernel(image, maximum, window, sigma_color, sigma_spatial, bins, mode, cval,
                               color_lut, spatial_lut, scratch, output)

    workbook = load(Path(benchmark_path))
    rows, benchmarks = [], []
    with patch.object(_denoise, '_denoise_bilateral', record):
        for name, image, options, native in cases(workbook):
            captures.clear()
            rows.append(dict(id=name, native=native, **observe(denoise_bilateral, image, options, captures)))
        for spec in workbook.case_specs():
            captures.clear()
            (image,), options = workbook.make_case(spec)
            benchmarks.append(dict(id=spec['id'], **observe(denoise_bilateral, image, options, captures)))
    assert all('output' in row for row in rows if row['native'])
    assert all('output' in row and row['output']['nan_count'] == 0 and not row['warnings'] for row in benchmarks)
    packet = dict(profile=profile, versions=observed_version, sources=sources,
                  observer_sha256=sha(Path(__file__).read_bytes()),
                  benchmark_sha256=sha(Path(benchmark_path).read_bytes()), rows=rows, benchmark=benchmarks)
    with Path(out).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(packet, stream, indent=2)
        stream.write('\n')
    print(f'Frozen {len(rows)} contract cases and {len(benchmarks)} fixed benchmark cases on {profile}.')


if __name__ == '__main__':
    main()
