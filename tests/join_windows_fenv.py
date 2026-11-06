"""JOIN test-only Windows UCRT flags/rounding; no image computation.

Windows SDK ucrt/fenv.h defines fexcept_t as unsigned long (LLP64:4bytes).
Only fegetexceptflag creates saved flag state; never reconstruct it from bits.
"""
import ctypes
from functools import lru_cache
import sys

class WindowsFEnv:
    mask=0x1f # SDK FE_ALL_EXCEPT
    def __init__(self,library=None):
        assert sys.platform=='win32'and ctypes.sizeof(ctypes.c_ulong)==4
        self.library=ctypes.CDLL('ucrtbase.dll')if library is None else library
        signatures={'fegetexceptflag':[ctypes.POINTER(ctypes.c_ulong),ctypes.c_int],
                    'fesetexceptflag':[ctypes.POINTER(ctypes.c_ulong),ctypes.c_int],
                    'fetestexcept':[ctypes.c_int], 'fegetround':[],
                    'fesetround':[ctypes.c_int]}
        for name,args in signatures.items():
            function=getattr(self.library,name)
            function.argtypes=args;function.restype=ctypes.c_int
            setattr(self,name,function)
    def state(self):
        return self.fetestexcept(self.mask),self.fegetround()
    def capture(self):
        flags=ctypes.c_ulong()
        assert self.fegetexceptflag(ctypes.byref(flags),self.mask)==0
        return flags,self.state()
    def restore(self,flags,before):
        assert isinstance(flags,ctypes.c_ulong)
        assert self.fesetexceptflag(ctypes.byref(flags),self.mask)==0
        assert self.fesetround(before[1])==0
        assert self.state()==before,'Windows JOIN exception flags/rounding restoration'

@lru_cache(maxsize=1)
def backend():
    return WindowsFEnv()
