"""Raw layout normalization preserves every byte, including NaN payloads."""
import importlib
import os
import numpy as np
import pytest

core=importlib.import_module(os.environ.get('SKIMAGEXX_COPY_BACKEND','skimagexx._core'))
TYPES=['bool','u1','i1','u2','i2','u4','i4','u8','i8','f2','f4','f8']


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    old=core.get_num_threads();core.set_num_threads(request.param)
    yield
    core.set_num_threads(old)


@pytest.mark.parametrize('dtype',TYPES)
@pytest.mark.parametrize('layout',['C','F','reverse','step','transpose','broadcast-row','broadcast-column','unaligned','empty'])
@pytest.mark.parametrize('opposite',[False,True])
def test_raw_storage_copy(dtype,layout,opposite):
    descriptor=np.dtype(dtype)
    if opposite and descriptor.itemsize>1:descriptor=descriptor.newbyteorder('S')
    image=np.empty((17,67),descriptor)
    # Fill storage directly: float cases deliberately include arbitrary NaN
    # encodings and bool cases include noncanonical truth bytes.
    image.view(np.uint8).reshape(-1)[:]=np.resize(np.arange(256,dtype=np.uint8),image.nbytes)
    if descriptor.kind=='f' and descriptor.itemsize in (4,8):
        words=([0x7f800001,0x7fc00001,0xff800123,0x80000000,0] if descriptor.itemsize==4
               else [0x7ff0000000000001,0x7ff8000000000001,0xfff0000000000123,0x8000000000000000,0])
        pattern=np.array(words,dtype=np.dtype('u'+str(descriptor.itemsize)).newbyteorder(descriptor.byteorder))
        image.view(np.uint8).reshape(-1)[:pattern.nbytes]=pattern.view(np.uint8)
    if layout=='F':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='step':image=image[::2,::3]
    elif layout=='transpose':image=image.T
    elif layout=='broadcast-row':image=np.broadcast_to(image[:1],image.shape)
    elif layout=='broadcast-column':image=np.broadcast_to(image[:,:1],image.shape)
    elif layout=='unaligned':
        other=np.ndarray(image.shape,descriptor,buffer=bytearray(image.nbytes+1),offset=1)
        other.view(np.uint8).reshape(-1)[:]=image.view(np.uint8).reshape(-1);image=other
    elif layout=='empty':image=image[:0]
    original=image.tobytes(order='C');expected=original
    if not descriptor.isnative and descriptor.itemsize>1:
        expected=np.frombuffer(original,np.uint8).reshape(-1,descriptor.itemsize)[:,::-1].tobytes()
    if descriptor.kind=='f' and descriptor.itemsize==2:
        actual=core.resize_half_identity(image,False,0)[0]
    else:
        actual=core.resize_zoom_typed(image,list(image.shape),0,3,0.,False,True,True)
    assert actual.dtype==np.dtype(descriptor.type) and actual.shape==image.shape
    assert actual.flags.c_contiguous and actual.flags.owndata
    assert actual.tobytes()==expected
    assert image.tobytes(order='C')==original


@pytest.mark.parametrize('layout',['reverse','step','broadcast-row','broadcast-column'])
def test_partial_parallel_rows_and_singleton_axes(layout):
    image=np.arange(3*129*1029,dtype=np.uint64).reshape(3,129,1029)
    if layout=='reverse':image=image[::-1,:,::-1]
    elif layout=='step':image=image[:,::2,::2]
    elif layout=='broadcast-row':image=np.broadcast_to(image[:,:1],image.shape)
    else:image=np.broadcast_to(image[:,:,:1],image.shape)
    image=image[:,None,:,:]
    actual=core.resize_zoom_typed(image,list(image.shape),0,3,0.,False,True,True)
    assert actual.tobytes()==image.tobytes(order='C')
