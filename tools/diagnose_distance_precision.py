"""Compare NumPy float32 primitives with the platform math library."""
import ctypes
import ctypes.util
import json
import platform
import subprocess
import numpy as np


def difference(a,b):
    a,b=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
    ulps=np.abs(a.view(np.int32).astype(np.int64)-b.view(np.int32).astype(np.int64))
    return {'different':int(np.count_nonzero(a!=b)),'max_ulp':int(ulps.max()),
            'max_abs':float(np.max(np.abs(a-b)))}


def main():
    lib=ctypes.CDLL('ucrtbase' if platform.system()=='Windows' else ctypes.util.find_library('m'))
    def function(name):
        f=getattr(lib,name,None) or getattr(lib,'_'+name)
        f.argtypes=[ctypes.c_float,ctypes.c_float];f.restype=ctypes.c_float
        return f
    hypotf,atan2f,powf=map(function,('hypotf','atan2f','powf'))
    a=(np.random.default_rng(32).random((67,3))*200-100).astype(np.float32)
    x,y=a[:,1],a[:,2]
    c=np.hypot(x,y)
    core=getattr(np,'_core',None) or np.core
    cpu=core._multiarray_umath.__cpu_features__
    result={'numpy':np.__version__,'platform':platform.platform(),
            'cpu_features':[name for name,on in cpu.items() if on and ('AVX' in name or name.startswith('X86_V'))],
            'hypotf':difference(c,[hypotf(xx,yy) for xx,yy in zip(x,y)]),
            'atan2f':difference(np.arctan2(y,x),[atan2f(yy,xx) for xx,yy in zip(x,y)]),
            'power7':difference(c**7,[powf(cc,7) for cc in c]),
            'array_vs_scalar_hypot':difference(c,[np.hypot(xx,yy) for xx,yy in zip(x,y)])}
    if platform.system()=='Linux':
        try:
            dl=ctypes.CDLL(ctypes.util.find_library('dl'))
            dl.dlvsym.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_char_p];dl.dlvsym.restype=ctypes.c_void_p
            address=dl.dlvsym(lib._handle,b'hypotf',b'GLIBC_2.2.5')
            if address:
                old=ctypes.CFUNCTYPE(ctypes.c_float,ctypes.c_float,ctypes.c_float)(address)
                result['legacy_hypotf']=difference(c,[old(xx,yy) for xx,yy in zip(x,y)])
            binary=core._multiarray_umath.__file__
            symbols=subprocess.check_output(['readelf','-Ws',binary],text=True)
            result['hypot_symbols']=sorted({line.split()[-1] for line in symbols.splitlines() if 'hypot' in line})
        except (OSError,subprocess.SubprocessError):pass
    # The 180-degree CIEDE2000 branch amplifies one-ulp angle differences.
    # Include double primitives and exact representations on every CI host.
    import math
    from pathlib import Path
    from skimage.color import deltaE_ciede2000
    import skimagexx as xx
    data=np.loadtxt(Path(__file__).parent.parent/'tests/data/ciede2000-sharma.txt')
    first,second=data[:,:3],data[:,3:6]
    radius=.5*(np.hypot(first[:,1],first[:,2])+np.hypot(second[:,1],second[:,2]))
    c7=radius**7
    scale=1+.5*(1-np.sqrt(c7/(c7+25**7)))
    for lab in (first,second):
        x,y=lab[:,1]*scale,lab[:,2]
        array=np.arctan2(y,x)
        scalar=np.array([math.atan2(yy,xxx) for xxx,yy in zip(x,y)])
        result.setdefault('double_atan2',[]).append([
            {'index':int(i),'x':float(x[i]).hex(),'y':float(y[i]).hex(),
             'numpy':float(array[i]).hex(),'libm':float(scalar[i]).hex()}
            for i in np.flatnonzero(array!=scalar)])
    result['sharma_native']=xx.color.deltaE_ciede2000(first,second).tolist()
    result['sharma_original']=deltaE_ciede2000(first,second).tolist()
    h=np.array([0,164,345,180,360],dtype=np.float64)*(np.pi/180)
    h=np.concatenate((h,np.nextafter(h,-np.inf),np.nextafter(h,np.inf)))
    lab=np.stack((np.resize([0.,15.999,16.,50.,100.],h.shape),20*np.cos(h),20*np.sin(h)),axis=-1)
    other=lab.copy();other[:,1:]*=-1
    radius=np.hypot(lab[:,1],lab[:,2]);c7=radius**7
    scale=1+.5*(1-np.sqrt(c7/(c7+25**7)))
    high=np.maximum(abs(lab[:,1]),abs(lab[:,2]));low=np.minimum(abs(lab[:,1]),abs(lab[:,2]))
    approx_radius=high*np.sqrt(1+(low/high)**2)
    r2=approx_radius*approx_radius;r4=r2*r2;approx_c7=(r4*r2)*approx_radius
    approx_scale=1+.5*(1-np.sqrt(approx_c7/(approx_c7+25**7)))
    def hues(factor):
        first=np.arctan2(lab[:,2],lab[:,1]*factor);first[first<0]+=2*np.pi
        second=np.arctan2(other[:,2],other[:,1]*factor);second[second<0]+=2*np.pi
        return abs(second-first)
    original_hues,approx_hues=hues(scale),hues(approx_scale)
    native=xx.color.deltaE_ciede2000(np.tile(lab,(19,1)),np.tile(other,(19,1)))[:15]
    expected=deltaE_ciede2000(np.tile(lab,(19,1)),np.tile(other,(19,1)))[:15]
    result['antipodes']=[{'index':i,'input':[float(v).hex() for v in lab[i]],
        'radius':float(radius[i]).hex(),'approx_radius':float(approx_radius[i]).hex(),
        'c7':float(c7[i]).hex(),'approx_c7':float(approx_c7[i]).hex(),
        'scale':float(scale[i]).hex(),'approx_scale':float(approx_scale[i]).hex(),
        'diff':float(original_hues[i]).hex(),'approx_diff':float(approx_hues[i]).hex(),
        'native':float(native[i]),'original':float(expected[i])} for i in range(15)]
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
