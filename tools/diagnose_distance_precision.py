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
    cpu=np._core._multiarray_umath.__cpu_features__
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
            binary=np._core._multiarray_umath.__file__
            symbols=subprocess.check_output(['readelf','-Ws',binary],text=True)
            result['hypot_symbols']=sorted({line.split()[-1] for line in symbols.splitlines() if 'hypot' in line})
        except (OSError,subprocess.SubprocessError):pass
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
