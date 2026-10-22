"""Record layout-dependent NumPy math without relaxing parity budgets."""
import json
import platform
import numpy as np
from skimage.color import deltaE_ciede2000
import skimagexx as xx


def stages(a,b):
    a1,b1=a[:,1],a[:,2];a2,b2=b[:,1],b[:,2]
    first=np.hypot(a1,b1);second=np.hypot(a2,b2)
    cbar=.5*(first+second);c7=cbar**7
    scale=1+.5*(1-np.sqrt(c7/(c7+25**7)))
    x1=a1*scale;x2=a2*scale
    h1=np.arctan2(b1,x1);h2=np.arctan2(b2,x2)
    h1[h1<0]+=2*np.pi;h2[h2<0]+=2*np.pi
    return {'initial_hypot':first,'power7':c7,'scale':scale,
            'scaled_hypot':np.hypot(x1,b1),'h1':h1,'h2':h2,'h_diff':h2-h1,
            'reference':deltaE_ciede2000(a,b),'native':xx.color.deltaE_ciede2000(a,b)}


def main():
    result={'numpy':np.__version__,'platform':platform.platform(),
            'dispatch':str(np.lib.introspect.opt_func_info(func_name='arctan2|hypot|power')) if hasattr(np.lib,'introspect') else None,
            'svml_probe':xx.color._SVML_DOUBLE_ANGLES,'cases':[]}
    for dtype in (np.float32,np.float64):
        h=np.array([0,164,345,180,360],dtype=dtype)*dtype(np.pi/180)
        h=np.concatenate((h,np.nextafter(h,dtype(-np.inf)),np.nextafter(h,dtype(np.inf))))
        lab=np.stack((np.resize(np.array([0,15.999,16,50,100],dtype=dtype),h.shape),20*np.cos(h),20*np.sin(h)),axis=-1)
        variants=[]
        for item in lab:
            for dx in range(-3,4):
                for dy in range(-3,4):
                    value=item.copy()
                    for dim,n in [(1,dx),(2,dy)]:
                        for _ in range(abs(n)):value[dim]=np.nextafter(value[dim],dtype(np.inf if n>0 else -np.inf))
                    variants.append(value)
        original=np.array(variants);other=original.copy();other[:,1:]*=-1
        for layout in ('contiguous','reversed','planar'):
            a,b=original,other
            if layout=='reversed':a=a[::-1];b=b[::-1]
            if layout=='planar':a=np.ascontiguousarray(a.T).T;b=np.ascontiguousarray(b.T).T
            actual=stages(a,b);copied=stages(np.ascontiguousarray(a),np.ascontiguousarray(b))
            row={'dtype':np.dtype(dtype).name,'layout':layout,'strides':a.strides,'stages':{}}
            for name,values in actual.items():
                expected=copied[name];different=np.flatnonzero(values!=expected)
                row['stages'][name]={'different_from_copy':len(different),
                    'examples':[{'index':int(i),'actual':float(values[i]).hex(),'copy':float(expected[i]).hex()} for i in different[:8]]}
            different=np.flatnonzero(abs(actual['native']-actual['reference'])>.001)
            row['native_mismatch_indices']=different.tolist()
            result['cases'].append(row)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
