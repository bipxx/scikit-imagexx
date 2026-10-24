"""Byte/energy-trace regression for strided TV input address transport.

Default is the real installed core. An explicit private selector is isolated
implementation evidence, never a public namespace/capability admission.
"""
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
import skimagexx as xx
from skimage.restoration._denoise import _denoise_tv_chambolle_nd

BACKEND=os.environ.get('SKIMAGEXX_TV_CHAMBOLLE_STAGE_CORE')
if BACKEND:
    spec=importlib.util.spec_from_file_location('_tv_stage_core',Path(BACKEND).resolve(strict=True))
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
else:
    from skimagexx import _core as core


@pytest.fixture(params=(1,4))
def active(request):
    assert core.tv_chambolle_finite_interface=='tv-chambolle-finite-v1'
    assert core.tv_chambolle_finite_capabilities==3 and inspect.isbuiltin(core.tv_chambolle_finite_run)
    assert core.tv_chambolle_finite_run.__module__==core.__name__
    if os.environ.get('SKIMAGEXX_REQUIRE_GIL_ASSERTIONS')=='1':
        assert getattr(core,'gil_assertions' if BACKEND else '_consumer_gil_assertions') is True
    previous=core.get_num_threads()
    if request.param==4 and os.environ.get('SKIMAGEXX_EXPECT_OPENMP_OFF')=='1':
        with pytest.raises(ValueError,match='this build has no OpenMP support'):core.set_num_threads(4)
        assert core.get_num_threads()==previous==1
        pytest.skip('explicit noOpenMP worker4 rejection; serial-only configuration')
    core.set_num_threads(request.param)
    try:yield core
    finally:core.set_num_threads(previous)


def fixture(shape,dtype,layout):
    value=((np.arange(np.prod(shape)).reshape(shape)*13)%29-14).astype(dtype)/dtype(16)
    value.flat[::31]=-dtype(0)
    if layout=='F':value=np.asfortranarray(value)
    elif layout=='step':
        owner=np.zeros(tuple(2*n for n in shape),dtype);view=owner[tuple(slice(None,None,2) for _ in shape)];view[...] =value;value=view
    elif layout=='reverse':value=value[tuple(slice(None,None,-1) if a%2==0 else slice(None) for a in range(len(shape)))]
    elif layout=='broadcast':value=np.broadcast_to(value.take([0],axis=-1),shape)
    elif layout=='unaligned':
        owner=np.zeros(value.nbytes+1,np.uint8);view=np.ndarray(shape,dtype,buffer=owner,offset=1);view[...] =value;value=view
    value.setflags(write=False)
    return value


def source_reference():
    source=inspect.getsource(_denoise_tv_chambolle_nd)
    assert hashlib.sha256(source.encode()).hexdigest()=='16e4edb3ed653ad3ce967c9b2121bb18b6a4997db01f5387747577d1e7cf141e'
    trace=[];final={}
    def bits(value):
        a=np.asarray(value);return int(a.view(np.uint32 if a.dtype.itemsize==4 else np.uint64))
    def capture(i,s,n,E,previous,initial,eps):
        difference=np.asarray(0,dtype=np.asarray(E).dtype)[()] if i==0 else np.abs(previous-E)
        threshold=np.asarray(0,dtype=np.asarray(E).dtype)[()] if i==0 else eps*initial
        trace.append((bits(s),bits(n),bits(E),bits(difference),bits(threshold),np.asarray(E).dtype.itemsize==4,bool(i and difference<threshold)))
    modified=source.replace('def _denoise_tv_chambolle_nd(', 'def reference(').replace('E = (d**2).sum()', 'E = (d**2).sum()\n        square_scalar=E.copy()').replace('E += weight * norm.sum()', 'norm_scalar=norm.sum()\n        E += weight*norm_scalar').replace('E /= float(image.size)', 'E /= float(image.size)\n        capture(i,square_scalar,norm_scalar,E,None if i==0 else E_previous,None if i==0 else E_init,eps)').replace('    return out','    final.update(p=p.copy(),g=g.copy(),d=d.copy())\n    return out')
    namespace=dict(np=np,capture=capture,final=final);exec(compile(modified,'pinned-original-TV-trace-observer','exec'),namespace)
    return namespace['reference'],trace,final


@pytest.mark.parametrize('dtype',(np.float32,np.float64))
@pytest.mark.parametrize('shape',((1,),(9,),(3,17),(4,1,7),(2,1,3,9)))
@pytest.mark.parametrize('layout',('C','F','step','reverse','broadcast','unaligned'))
@pytest.mark.parametrize('iterations',(1,2,8))
def test_strided_output_and_every_energy_word_match_original(active,dtype,shape,layout,iterations):
    image=fixture(shape,dtype,layout);before=image.tobytes();metadata=(image.shape,image.strides,image.flags.writeable,id(image.base))
    p=np.zeros((image.ndim,)+image.shape,image.dtype);g=np.zeros_like(p);d=np.zeros(image.shape,image.dtype)
    reference,traces,final=source_reference()
    expected=reference(image,weight=.5,eps=0.,max_num_iter=iterations)
    actual,count,alias,trace=active.tv_chambolle_finite_run(image,p,g,d,.5,0.,iterations,np.__version__,np.getbufsize(),True)
    assert actual.dtype==expected.dtype and actual.shape==expected.shape and actual.tobytes()==expected.tobytes()
    assert count==iterations and alias==(iterations==1) and [tuple(r) for r in trace]==traces
    assert (actual is image)==alias
    if not alias:assert actual.flags.owndata and actual.flags.c_contiguous and actual.base is None
    for left,right in ((p,final['p']),(g,final['g']),(d,final['d'])):assert left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes()
    assert before==image.tobytes() and metadata==(image.shape,image.strides,image.flags.writeable,id(image.base))


@pytest.mark.parametrize('layout',('F','step','reverse','broadcast','unaligned'))
@pytest.mark.parametrize('dtype',(np.float32,np.float64))
def test_strided_last_coordinate_rejection_has_no_partial_writes(active,layout,dtype):
    image=fixture((3,17),dtype,layout);image.setflags(write=True)
    image[-1,-1]=np.inf;image.setflags(write=False)
    p=np.zeros((image.ndim,)+image.shape,dtype);g=np.zeros_like(p);d=np.zeros(image.shape,dtype);before=[a.tobytes() for a in (p,g,d)]
    with pytest.raises(NotImplementedError,match='finite'):active.tv_chambolle_finite_run(image,p,g,d,.5,0.,2,np.__version__,np.getbufsize(),True)
    assert before==[a.tobytes() for a in (p,g,d)]
