"""Virtual padding metadata must never cause a NumPy read from the input."""
import subprocess
import sys
import textwrap


def test_mean_layout_allocation_does_not_read_image_pixels():
    # A child protects the test runner if a future iterator change accidentally
    # initializes a buffer or advances over the virtual padded input.
    script=textwrap.dedent(r'''
        import ctypes
        import mmap
        import os
        import numpy as np
        from skimage import transform as original
        from skimagexx.transform import _local_mean_output

        size=mmap.PAGESIZE
        if os.name=='nt':
            allocate=ctypes.windll.kernel32.VirtualAlloc
            allocate.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_ulong,ctypes.c_ulong]
            allocate.restype=ctypes.c_void_p
            address=allocate(None,size,0x3000,0x01)  # committed PAGE_NOACCESS
            assert address
        else:
            page=mmap.mmap(-1,size)
            address=ctypes.addressof(ctypes.c_char.from_buffer(page))
            protect=ctypes.CDLL(None).mprotect
            protect.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
            protect.restype=ctypes.c_int
            assert protect(address,size,0)==0
        storage=(ctypes.c_char*size).from_address(address)

        real=(np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,
              np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64)
        for dtype in real:
            item=np.dtype(dtype).itemsize
            cases=[
                (np.zeros((5,7),dtype=dtype,order='F'),(5,4)),
                (np.broadcast_to(np.zeros((1,7,11),dtype=dtype),(5,7,11)),(1,1,1)),
                (np.zeros((5,7),dtype=dtype)[::-1,::-1],(1,1)),
                (np.zeros((5,7,11),dtype=dtype).transpose(1,0,2),(1,1,1)),
            ]
            for readable,blocks in cases:
                expected=original.downscale_local_mean(readable,blocks)
                offset=sum((length-1)*-stride for length,stride in zip(readable.shape,readable.strides) if stride<0)
                protected=np.ndarray(readable.shape,dtype=dtype,buffer=storage,
                    offset=offset,strides=readable.strides)
                after=tuple((-length)%factor for length,factor in zip(protected.shape,blocks))
                shape=tuple((length+width)//factor for length,width,factor in zip(protected.shape,after,blocks))
                actual=_local_mean_output(protected,blocks,after,shape,expected.dtype)
                assert actual.shape==expected.shape
                assert actual.dtype==expected.dtype
                assert actual.strides==expected.strides,(dtype,readable.strides,actual.strides,expected.strides)
        print('48 protected-input allocation cases passed')
    ''')
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert '48 protected-input allocation cases passed' in result.stdout
