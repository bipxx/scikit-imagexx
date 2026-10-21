"""Non-default color dtypes/layouts, with native parity required before timing."""
from run import measure
import argparse
from datetime import datetime, timezone
from provenance import source_fingerprint
import json
from pathlib import Path
import platform
import sys
import numpy as np
import skimage
from skimage import color as reference
import skimagexx as xx
from skimagexx.coverage import parity_tolerance, default_call

FUNCTIONS = ['rgb2yuv','yuv2rgb','rgb2yiq','yiq2rgb','rgb2ypbpr','ypbpr2rgb',
             'rgb2ycbcr','ycbcr2rgb','rgb2ydbdr','ydbdr2rgb','rgb2rgbcie',
             'rgbcie2rgb','rgb2xyz','xyz2rgb','rgb2hsv','hsv2rgb','xyz2lab',
             'lab2xyz','rgb2lab','lab2rgb','lab2lch','lch2lab',
             'xyz2luv','luv2xyz','rgb2luv','luv2rgb','separate_stains','combine_stains','rgb2hed','hed2rgb']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--threads',type=int,nargs='+',default=[1])
    parser.add_argument('--output',type=Path,default=Path('benchmarks/color-layouts.json'))
    args = parser.parse_args()
    if min([args.size]+args.threads) < 1:
        parser.error('size and workers must be positive')
    xx.set_fallback_enabled(False)
    results = []
    for dtype in (np.uint16,np.float32):
        full = np.random.default_rng(102).integers(0,65536,(args.size,2*args.size,3),dtype=np.uint16)
        if dtype == np.float32:
            full = full.astype(np.float32)/np.float32(65535)
        strided = full[:,::2]
        layouts = [('strided',strided,-1),('channel-first',np.ascontiguousarray(np.moveaxis(strided,-1,0)),0)]
        for layout,a,axis in layouts:
            for name in FUNCTIONS:
                original, current = getattr(reference,name), getattr(xx.color,name)
                call_args,options=default_call('color.'+name,a)
                options['channel_axis']=axis
                expected = original(*call_args,**options)
                for threads in args.threads:
                    xx.set_num_threads(threads)
                    actual = current(*call_args,**options)
                    assert actual.dtype == expected.dtype
                    rtol,atol = parity_tolerance('color.'+name,np.dtype(dtype))
                    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
                    timing = measure(lambda: original(*call_args,**options),lambda: current(*call_args,**options),7,15)
                    result = {'api':'color.'+name,'dtype':np.dtype(dtype).name,'size':args.size,
                              'threads':threads,'layout':layout,'channel_axis':axis,'backend':'native',
                              'max_abs_error':float(np.max(np.abs(actual-expected))),**timing}
                    results.append(result)
                    print(f'{name:16} {np.dtype(dtype).name:7} {layout:13} t={threads}: {timing["speedup"]:.2f}x',flush=True)
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    report = {'metadata':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
                          'cpu':platform.processor(),'reference':skimage.__version__,'numpy':np.__version__,
                          **fingerprint,'avx2':xx.has_avx2(),'fallback_enabled':False,
                          'reference_thread_pools':1,'command':sys.argv},'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines = ['# Color dtype and layout performance','',
             'Every call is checked against original scikit-image before timing. '
             'Fallback is disabled. Seven warmed allocation-inclusive samples include '
             'channel movement, contiguous copies and integer normalization. The reference '
             'uses one thread. Speedup = reference/current; slower cases remain visible.','',
             '| API | Input | Layout | Workers | Original ms | C++ ms | Speedup |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["api"]} | {r["dtype"]} | {r["layout"]} | {r["threads"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
