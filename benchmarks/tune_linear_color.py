"""Fixed one-worker linear-branch workloads for the color kernels."""
from run import measure
import argparse
import json
import math
from pathlib import Path
import numpy as np
from skimage import color as original
import skimagexx as xx
from skimagexx.coverage import parity_tolerance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    xx.set_num_threads(1); xx.set_fallback_enabled(False)
    cases=[('rgb2xyz',[.01,.03,.04],512),('rgb2lab',[.01,.03,.04],512),
           ('rgb2luv',[.01,.03,.04],512),('xyz2lab',[.001,.003,.006],512),
           ('xyz2luv',[.001,.003,.006],512),('xyz2rgb',[.001,.003,.006],512),
           ('lab2rgb',[1,0,0],256),('luv2rgb',[1,0,0],256)]
    rows=[]
    for dtype in (np.float32,np.float64):
        for name,values,size in cases:
            image=np.broadcast_to(np.array(values,dtype=dtype),(size,size,3))
            reference,native=getattr(original,name),getattr(xx.color,name)
            expected,actual=reference(image),native(image)
            assert actual.dtype==expected.dtype
            rtol,atol=parity_tolerance('color.'+name,np.dtype(dtype))
            np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
            rows.append({'api':name,'dtype':np.dtype(dtype).name,'size':size,
                         **measure(lambda:reference(image),lambda:native(image),7,15)})
    metric=math.exp(sum(math.log(r['cpp_ms']) for r in rows)/len(rows))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'metric':'geometric mean native ms; lower is better',
                                     'workers':1,'cases':16,'geomean_ms':metric,'results':rows},indent=2)+'\n',encoding='utf-8')
    print(f'geomean_ms={metric:.6f}')


if __name__=='__main__': main()
