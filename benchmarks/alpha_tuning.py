"""Fixed 48-case one-worker grayscale expansion tuning workload."""
from run import measure
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import numpy as np
import skimage
from skimage import color
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    xx.set_num_threads(1)
    xx.set_fallback_enabled(False)
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    rows = []
    for dtype in ('uint8', 'int8', 'uint16', 'int16', 'float16', '>u2', '>f4', '>f8'):
        image = np.random.default_rng(315).integers(0, 128, (512, 512), dtype=np.uint8).astype(dtype)
        for layout, view in [('contiguous', image), ('reversed', image[::-1, ::-1]),
                             ('strided', np.tile(image, (2, 2))[::2, ::2])]:
            for name in ('gray2rgb', 'gray2rgba'):
                reference, native = getattr(color, name), getattr(xx.color, name)
                expected, actual = reference(view), native(view)
                assert expected.dtype == actual.dtype and expected.shape == actual.shape
                np.testing.assert_array_equal(actual, expected)
                timing = measure(lambda: reference(view), lambda: native(view), 7, 15)
                rows.append({'api': 'color.'+name, 'dtype': dtype, 'shape': [512, 512],
                             'layout': layout, 'threads': 1, **timing})
                print(name, dtype, layout, f'{timing["cpp_ms"]:.6f}ms', flush=True)
    assert fingerprint == source_fingerprint(root)
    metric = math.exp(statistics.mean(math.log(row['cpp_ms']) for row in rows))
    result = {'metadata': {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                          'reference': skimage.__version__, 'numpy': np.__version__,
                          'platform': platform.platform(), 'cpu': platform.processor(),
                          'avx2': xx.has_avx2(), 'fallback_enabled': False,
                          'reference_thread_pools': 1, 'repeat': 7, **fingerprint},
              'metric': 'geometric mean of 48 allocation-inclusive one-worker C++ median latencies in milliseconds; lower is better',
              'cpp_geomean_ms': metric, 'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('cpp_geomean_ms='+str(metric), flush=True)


if __name__ == '__main__':
    main()
