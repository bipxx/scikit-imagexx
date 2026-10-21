"""Raw strided integer prefixes, singleton axes and exact parallel carries."""
import numpy as np
import pytest
import skimagexx as xx
from test_integral import sample,check


@pytest.fixture(autouse=True,params=[1,4])
def configuration(request):
    previous=xx.set_fallback_enabled(False);xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous);xx.set_num_threads(1)


def line(dtype,count,layout):
    base=sample(dtype,(count*3,))
    if layout=='contiguous':return base[:count]
    if layout=='reversed':return base[:count][::-1]
    if layout=='stepped':return base[::3]
    if layout=='endian':return base.astype(np.dtype(dtype).newbyteorder('>'))[::3]
    if layout=='tagged':return base.view(np.dtype(dtype,metadata={'units':'raw','source':'singleton'}))[::3]
    if layout=='zero-stride':return np.broadcast_to(base[0] if count else dtype(0),(count,))
    if layout=='readonly':
        result=base[::3];result.flags.writeable=False;return result
    raw=np.empty(base.nbytes+1,dtype=np.uint8)
    result=np.ndarray(base.shape,dtype=base.dtype,buffer=raw,offset=1);result[:]=base
    return result[::3]


@pytest.mark.parametrize('dtype',[np.int64,np.uint64])
@pytest.mark.parametrize('layout',['contiguous','reversed','stepped','endian','tagged','zero-stride','readonly','unaligned'])
@pytest.mark.parametrize('pattern',[(0,),(1,0),(0,1),(1,1,0,1),(1,0,1,1)])
@pytest.mark.parametrize('length',[17,262145])
def test_strided_singleton_axes(dtype,layout,pattern,length):
    image=line(dtype,length,layout).reshape(tuple(length if n==0 else n for n in pattern))
    snapshot=image.copy()
    for target in (None,np.int64,np.uint64):check(image,dtype=target)
    np.testing.assert_array_equal(image,snapshot)


@pytest.mark.parametrize('dtype',[np.int64,np.uint64])
@pytest.mark.parametrize('layout',['stepped','reversed','zero-stride','endian'])
@pytest.mark.parametrize('length',[0,1,2,3,4,7,8,15,16,17,31,32,33,65])
def test_simd_prefix_tails(dtype,layout,length):
    image=line(dtype,length,layout).reshape(length,1)
    for target in (None,np.int64,np.uint64):check(image,dtype=target)


@pytest.mark.parametrize('dtype',[np.int64,np.uint64])
@pytest.mark.parametrize('shape',[(0,7),(3,0,5),(1,1),(1,1,1,1),(1,17,1),(17,1)])
def test_empty_singleton_and_tagged_output(dtype,shape):
    image=sample(dtype,shape)
    for target in (np.dtype(np.int64,metadata={'sum':'raw'}),np.dtype(np.uint64,metadata={'sum':'modular'}),np.dtype('>i8'),np.dtype('>u8')):
        check(image,dtype=target)
