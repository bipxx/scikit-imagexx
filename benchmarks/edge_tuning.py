"""Fixed four-case one-worker Roberts column experiment; lower latency is better."""
from run import measure
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import platform
import statistics
import numpy as np
import skimage
from skimage import filters
import skimagexx as xx
from provenance import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    xx.set_num_threads(1);xx.set_fallback_enabled(False)
    root=Path(__file__).resolve().parents[1];fingerprint=source_fingerprint(root)
    rows=[]
    for dtype in (np.float32,np.float64):
        image=np.random.default_rng(418).integers(0,256,(512,512),dtype=np.uint8).astype(dtype)
        image/=dtype(255);column=image[:,:1]
        for name in ('roberts_pos_diag','roberts_neg_diag'):
            original,native=getattr(filters,name),getattr(xx.filters,name)
            expected=original(column);actual=native(column)
            assert actual.shape==expected.shape and actual.dtype==expected.dtype
            tolerance=2e-6 if dtype==np.float32 else 2e-13
            np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)
            timing=measure(lambda:original(column),lambda:native(column),11,25)
            rows.append({'api':'filters.'+name,'dtype':str(column.dtype),'shape':[512,1],
                         'strides':list(column.strides),'threads':1,**timing})
            print(name,str(column.dtype),f'{timing["cpp_ms"]:.7f} ms',flush=True)
    assert fingerprint==source_fingerprint(root)
    metric=math.exp(statistics.mean(math.log(row['cpp_ms']) for row in rows))
    metadata={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'reference':skimage.__version__,
              'numpy':np.__version__,'avx2':xx.has_avx2(),'fallback_enabled':False,
              'platform':platform.platform(),'cpu':platform.processor(),'reference_thread_pools':1,
              'repeat':11,**fingerprint}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'metadata':metadata,'metric':'geometric mean allocation-inclusive C++ median latency in milliseconds; lower is better',
                                      'cpp_geomean_ms':metric,'results':rows},indent=2)+'\n',encoding='utf-8')
    print('cpp_geomean_ms='+str(metric),flush=True)


if __name__=='__main__':main()
