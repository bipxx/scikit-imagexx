"""One-thread public call timings for the I/O suite against scikit-image 0.26."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from timeit import Timer

for _variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_variable] = '1'


def run(native, scratch=None):
    import numpy as np
    from skimage import __version__, io as stock
    from skimagexx import get_num_threads, set_num_threads
    from tifffile import imwrite

    if __version__ != '0.26.0':
        raise RuntimeError(f'Expected scikit-image 0.26.0, found {__version__}')
    previous = get_num_threads()
    set_num_threads(1)
    with ExitStack() as cleanup:
        cleanup.callback(set_num_threads, previous)

        def path(suffix, data=b''):
            with NamedTemporaryFile(dir=scratch, prefix='io_suite_', suffix=suffix, delete=False) as stream:
                result = Path(stream.name)
                cleanup.callback(result.unlink, missing_ok=True)
                stream.write(data)
            return str(result)

        image = np.arange(128 * 192, dtype='u1').reshape(128, 192)
        small = [image[i:i + 4, :32].copy() for i in range(64)] * 4
        strided = [image[i:i + 8:2, :32] for i in range(64)] * 4
        pgm = path('.pgm', b'P5\n192 128\n255\n' + image.tobytes())
        ppm = path('.ppm')
        rgb = np.repeat(image[..., None], 3, axis=2)
        sift = path('.key', b'512 128\n' + (b'12.25 4.5 3.125 -1.5\n' + b'123 ' * 128 + b'\n') * 512)
        surf = path('.surf', b'65\n512\n' + (b'12.25 4.5 3.125 -1.5 2.0 1\n' + b'.123 ' * 64 + b'\n') * 512)
        tiff = path('.tif')
        imwrite(tiff, image)
        stock.imread(pgm)
        stock.imsave(ppm, rgb, check_contrast=False)
        stack = stock.image_stack
        position = len(stack)
        cleanup.callback(lambda: stack.__delitem__(slice(position, None)))

        def load_image(i):
            return small[i]

        original_ic = stock.ImageCollection(list(range(len(small))), False, load_image)
        native_ic = native.ImageCollection(list(range(len(small))), False, load_image)
        files = [tiff] * len(small)
        original_multi = stock.MultiImage(files, False)
        native_multi = native.MultiImage(files, False)
        for collection in (original_ic, native_ic, original_multi, native_multi):
            # Time concatenation separately from the filesystem and codec work.
            collection.data[:] = small

        def case(module, name):
            if name == 'concatenate contiguous':
                return lambda: module.concatenate_images(small)
            if name == 'concatenate strided':
                return lambda: module.concatenate_images(strided)
            if name == 'ImageCollection.concatenate':
                return (native_ic if module is native else original_ic).concatenate
            if name == 'MultiImage.concatenate':
                return (native_multi if module is native else original_multi).concatenate
            if name == 'imread PGM':
                return lambda: module.imread(pgm)
            if name == 'imsave PPM':
                return lambda: module.imsave(ppm, rgb, check_contrast=False)
            if name == 'imread_collection':
                return lambda: module.imread_collection(pgm)
            if name == 'imread_collection_wrapper':
                return lambda: module.imread_collection_wrapper(load_image)(range(len(small))).concatenate()
            if name == 'load_sift':
                return lambda: module.load_sift(sift)
            if name == 'load_surf':
                return lambda: module.load_surf(surf)
            if name == 'push':
                return lambda: module.push(image)
            return module.pop

        names = (
            'concatenate contiguous', 'concatenate strided', 'ImageCollection.concatenate',
            'MultiImage.concatenate', 'imread PGM', 'imsave PPM', 'imread_collection',
            'imread_collection_wrapper', 'load_sift', 'load_surf', 'push', 'pop',
        )
        rows = []
        for name in names:
            original, candidate = (case(module, name) for module in (stock, native))
            repeats = 9
            calls = 50_000 if name in ('push', 'pop') else 30 if name in ('load_sift', 'load_surf') else 250

            def sample(function):
                if name == 'pop':
                    stack.extend([image] * calls)
                seconds = Timer(function).timeit(calls) / calls
                if name == 'push':
                    del stack[position:]
                return seconds

            timings = {original: [], candidate: []}
            for trial in range(repeats):
                for function in (original, candidate)[::1 if trial % 2 == 0 else -1]:
                    timings[function].append(sample(function))
            old, new = (median(timings[function]) for function in (original, candidate))
            rows.append((name, old, new))
            print(f'{name:29} {old * 1e6:9.3f} -> {new * 1e6:9.3f} us  ({old / new:.3f}x)', flush=True)
        return rows


if __name__ == '__main__':
    from skimagexx import _io_suite

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', help='Existing directory for temporary input and output files')
    options = parser.parse_args()
    raise SystemExit(any(old < new for _, old, new in run(_io_suite, options.scratch)))
