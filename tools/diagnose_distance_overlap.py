"""Check whether NumPy 1.x color results depend on temporary allocation layout."""
import hashlib
import json
import math
from pathlib import Path
import runpy
import numpy as np
from skimage.color import deltaE_ciede2000
import skimagexx as xx


def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def overlap(src,dst):
    # NumPy 1.24 loops_utils.h.src uses inclusive endpoints and stride*len.
    p=src.ctypes.data;q=dst.ctypes.data
    p0,p1=sorted((p,p+src.strides[0]*src.size))
    q0,q1=sorted((q,q+dst.strides[0]*dst.size))
    return not ((p0==q0 and p1==q1) or p0>q1 or q0>p1)


def main():
    generator=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tests/test_distance_math_provider.py'))['_antipodal_colors']
    colors=generator(np.float64)
    original=colors[(np.arange(128)*23+124)%len(colors)]
    other=original.copy();other[:,1:]*=-1
    result={'numpy':np.__version__,'svml':xx.color._SVML_DOUBLE_ANGLES,'cases':[]}
    xx.set_fallback_enabled(False)
    for layout in ('contiguous','strided'):
        a,b=original,other
        if layout=='strided':a=np.repeat(a,2,axis=0)[::2];b=np.repeat(b,3,axis=0)[::3]
        outputs={};plain=[]
        for attempt in range(128):
            pressure=[np.empty(128,np.float64) for _ in range(attempt%17)]
            actual=xx.color.deltaE_ciede2000(a,b)
            expected=deltaE_ciede2000(a,b)
            key=digest(expected);outputs.setdefault(key,expected.copy())
            plain.append({'attempt':attempt,'reference_sha256':key,'native_sha256':digest(actual),
                          'large_error_indices':np.flatnonzero(abs(actual-expected)>.001).tolist()})
        captures=[];ufunc=np.arctan2
        def traced(y,x,*args,**kwargs):
            out=ufunc(y,x,*args,**kwargs)
            scalar=np.array([math.atan2(float(v),float(w)) for v,w in zip(y,x)])
            captures.append({'input_y':digest(y),'input_x':digest(x),'output':digest(out),
                             'y_stride':y.strides[0],'x_stride':x.strides[0],
                             'y_out_overlap':overlap(y,out),'x_out_overlap':overlap(x,out),
                             'different_from_libm':np.flatnonzero(out!=scalar).tolist()})
            return out
        np.arctan2=traced
        try:
            for attempt in range(32):
                pressure=[np.empty(128,np.float64) for _ in range(attempt%17)]
                expected=deltaE_ciede2000(a,b)
                outputs.setdefault(digest(expected),expected.copy())
        finally:np.arctan2=ufunc
        result['cases'].append({'layout':layout,'input_a':digest(a),'input_b':digest(b),'plain_calls':plain,
                                'distinct_reference_outputs':{key:value.tolist() for key,value in outputs.items()},
                                'atan2_calls':captures})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
