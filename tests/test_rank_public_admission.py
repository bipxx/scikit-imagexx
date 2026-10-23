"""Compatibility admission is distinct from native overlapping-buffer parity."""
import warnings
import numpy as np
import pytest
import skimagexx as xx
from skimage.filters import rank as reference
from skimagexx.filters import rank
from skimagexx.filters.rank import generic
from skimagexx import _rank


@pytest.fixture(autouse=True,params=[1,4])
def workers(request):
    previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads()
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(threads);xx.set_fallback_enabled(previous)


def inputs(ndim,kind,dtype=np.uint8):
    shape=(5,7) if ndim==2 else (3,4,5)
    image=(np.arange(int(np.prod(shape)))%23).astype(dtype).reshape(shape)
    fp=np.ones((3,)*ndim,np.uint8);options={};owners=[]
    if kind=='view':options['out']=image.view();owners=[image]
    elif kind=='offset':
        owner=(np.arange(image.size+1)%23).astype(dtype)
        image=owner[:-1].reshape(shape);options['out']=owner[1:].reshape(shape);owners=[owner]
    elif kind=='mask':
        mask=(np.arange(image.size).reshape(shape)%3!=0).astype(np.uint8)
        options.update(mask=mask,out=mask.view());owners=[image,mask]
    return image,fp,options,owners


@pytest.mark.parametrize('surface',[rank,generic],ids=['canonical','generic'])
@pytest.mark.parametrize('name',['gradient','median','modal'])
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('kind',['view','offset','mask'])
def test_proven_alias_compatibility_output_warnings_and_mutations(surface,name,ndim,kind):
    expected_image,fp,expected_options,expected_owners=inputs(ndim,kind)
    image,footprint,options,owners=inputs(ndim,kind)
    with warnings.catch_warnings(record=True) as original_warnings:
        warnings.simplefilter('always')
        expected=getattr(reference,name)(expected_image,fp,**expected_options)
    previous=xx.set_fallback_enabled(True)
    try:
        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter('always')
            actual=getattr(surface,name)(image,footprint,**options)
    finally:xx.set_fallback_enabled(previous)
    fallback=[w for w in actual_warnings if isinstance(w.message,xx.FallbackWarning)]
    assert len(fallback)==1
    assert str(fallback[0].message)==f'skimage.filters.rank.{name} executed by scikit-image fallback'
    assert [(type(w.message),str(w.message)) for w in actual_warnings if not isinstance(w.message,xx.FallbackWarning)]==[(type(w.message),str(w.message)) for w in original_warnings]
    assert actual.dtype==expected.dtype and actual.strides==expected.strides
    assert actual.tobytes()==expected.tobytes()
    assert [a.tobytes() for a in owners]==[a.tobytes() for a in expected_owners]
    assert np.shares_memory(actual,options['out'])==np.shares_memory(expected,expected_options['out'])


@pytest.mark.parametrize('name',['gradient','median','modal'])
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('kind',['view','offset','mask'])
def test_strict_proven_alias_rejection_is_before_writes(name,ndim,kind):
    image,fp,options,owners=inputs(ndim,kind)
    before=[a.tobytes() for a in owners]
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        with pytest.raises(NotImplementedError,match='fallback is disabled'):
            getattr(rank,name)(image,fp,**options)
    assert not notices and before==[a.tobytes() for a in owners]


@pytest.mark.parametrize('name',['gradient','median','modal','entropy'])
@pytest.mark.parametrize('enabled',[False,True])
def test_unproved_wide_and_entropy_aliases_never_forward(name,enabled,monkeypatch):
    image,fp,options,owners=inputs(2,'view',np.uint16 if name!='entropy' else np.uint8)
    before=[a.tobytes() for a in owners]
    def forbidden(*args,**kwargs):raise AssertionError('unproved overlap forwarded')
    monkeypatch.setattr(reference,name,forbidden)
    previous=xx.set_fallback_enabled(enabled)
    try:
        with pytest.raises(NotImplementedError,match='compatibility is pending'):
            getattr(rank,name)(image,fp,**options)
    finally:xx.set_fallback_enabled(previous)
    assert before==[a.tobytes() for a in owners]


@pytest.mark.parametrize('name',['gradient','median','modal','entropy'])
@pytest.mark.parametrize('kind',['short-mask','short-out','zero-depth'])
def test_unsafe_alias_extent_rejection_never_forwards(name,kind,monkeypatch):
    image=np.arange(35,dtype=np.uint8).reshape(5,7);fp=np.ones((3,3),np.uint8)
    if kind=='short-out':options={'out':image[:1,:].view()}
    elif kind=='short-mask':options={'out':image.view(),'mask':np.ones((1,1),np.uint8)}
    else:options={'out':np.zeros((5,7,0),np.uint8)}
    def forbidden(*args,**kwargs):raise AssertionError('unsafe buffer forwarded')
    monkeypatch.setattr(reference,name,forbidden)
    previous=xx.set_fallback_enabled(True)
    try:
        with pytest.raises(ValueError):getattr(rank,name)(image,fp,**options)
    finally:xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('name',['gradient','median','modal'])
def test_compatibility_delegation_preserves_argument_objects(name,monkeypatch):
    image,fp,options,_=inputs(2,'mask')
    sentinel=object();calls=[]
    def delegated(module,operation,*args,**kwargs):
        calls.append((module,operation,args,kwargs));return sentinel
    monkeypatch.setattr(_rank,'fallback',delegated)
    assert getattr(rank,name)(image,fp,**options) is sentinel
    module,operation,args,kwargs=calls[0]
    assert module=='filters.rank' and operation==name
    assert args[0] is image and args[1] is fp
    assert kwargs['out'] is options['out'] and kwargs['mask'] is options['mask']
    assert kwargs['shift_x']==kwargs['shift_y']==kwargs['shift_z']==0
