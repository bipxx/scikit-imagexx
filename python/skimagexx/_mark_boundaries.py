"""Owned MARK conversion/cubic/mask/paint; allocation and scalar metadata only."""
from functools import lru_cache
import importlib.metadata
import math
import warnings
import numpy as np
from . import _core


@lru_cache(maxsize=1)
def _modern_zoom():
    return np.lib.NumpyVersion(importlib.metadata.version('scipy'))>='1.16.0'


def _shape_check(image,mask):
    if mask.ndim>image.ndim:
        raise IndexError(f'too many indices for array: array is {image.ndim}-dimensional, but {mask.ndim} were indexed')
    for axis,(actual,expected) in enumerate(zip(image.shape,mask.shape)):
        if actual!=expected:
            if np.lib.NumpyVersion(np.__version__)>='2.0.0':
                raise IndexError(f'boolean index did not match indexed array along axis {axis}; size of axis is {actual} but size of corresponding boolean axis is {expected}')
            raise IndexError(f'boolean index did not match indexed array along dimension {axis}; dimension is {actual} but corresponding boolean dimension is {expected}')


def _paint(image,mask,color):
    _shape_check(image,mask)
    if mask.ndim==image.ndim:raise NotImplementedError('mark non-channel-bearing assignment is unassessed')
    # Conversion is exactly the source assignment's callback phase. Pixels and
    # mask application remain owned C++; NumPy sees only the small RHS palette.
    values=_core.mark_boundaries_color(color,image.dtype)
    if values.dtype.kind!='f' or values.dtype.itemsize not in(4,8) or not values.dtype.isnative:values=np.asarray(values,dtype=image.dtype)
    selected=int(_core.mark_boundaries_count(mask))
    target=(selected,)+image.shape[mask.ndim:]
    try:np.broadcast_shapes(values.shape,target)
    except ValueError:
        a=str(values.shape).replace(' ','');b=str(target).replace(' ','')
        raise ValueError(f'shape mismatch: value array of shape {a} could not be broadcast to indexing result of shape {b}')from None
    # Source mapping.c's advanced assignment has a dtype-transfer/status phase
    # after FORCECAST. Execute that existing NumPy compatibility boundary on
    # the RHS palette itself, and consume its actual payload in native scatter.
    # No image pixels or boundary/dilation/cubic work are given to NumPy.
    palette=np.empty((1,)+(values.shape or(1,)),dtype=image.dtype)
    palette[np.ones(1,dtype=bool)]=values
    values=palette[0].reshape(values.shape)
    return _core.mark_boundaries_paint(image,mask,np.require(values,requirements=('C','A')))


def mark_boundaries(image,label_img,color=(1,1,0),outline_color=None,mode='outer',background_label=0):
    dtype=np.dtype(image.dtype)
    if dtype.kind=='c':raise ValueError('complex valued input is not supported')
    if dtype.kind=='f' and dtype.itemsize not in(2,4,8):raise NotImplementedError('mark extended floating precision is unassessed')
    if dtype.kind not in 'biuf':raise ValueError(f'Cannot convert from {dtype} to float64.')
    array=np.asarray(image)
    if array.ndim>4:raise NotImplementedError('mark image rank>4 is unassessed')
    modern=_modern_zoom()
    subpixel=mode=='subpixel'
    if subpixel:
        # Source computes its factors after conversion/gray replication. A zero
        # extent requires no pixel arithmetic; preserve literal division error.
        for size in array.shape[:2]if array.ndim==2 else array.shape[:-1]:2-1/size
    workers=_core.get_num_threads()
    prepared=_core.mark_boundaries_prepare(array,subpixel,modern,workers)
    # Source reads labels only after image conversion and optional cubic zoom.
    labels_dtype=label_img.dtype
    if type(label_img)is not np.ndarray or not labels_dtype.isnative or labels_dtype.kind not in 'biuf':
        raise NotImplementedError('mark label carrier/swapped partition is unassessed')
    selected=3 if subpixel else 1 if mode=='inner'else 2 if mode=='outer'else 0
    if labels_dtype.kind=='f' and selected in(2,3):raise ValueError("Invalid integer data type 'f'.")
    if labels_dtype.kind=='f' and selected==1 and background_label!=0:raise NotImplementedError('mark float-inner nonzero background promotion is unassessed')
    bits=labels_dtype.itemsize*8
    signed=labels_dtype.kind=='i'
    low=-(1<<(bits-1))if signed else 0;high=(1<<(bits-1))-1 if signed else (1<<bits)-1
    representable=False;word=0
    if type(background_label)in(int,bool,float):
        if isinstance(background_label,(int,bool))or math.isfinite(background_label):
            representable=low<=background_label<=high and int(background_label)==background_label
            if representable:word=int(background_label)%(1<<64)
    elif background_label is not None and not isinstance(background_label,str):
        raise NotImplementedError('mark legacy/background protocol partition is unassessed')
    legacy_string=isinstance(background_label,str)and np.lib.NumpyVersion(np.__version__)<'2.0.0'
    if legacy_string and selected!=2:
        raise NotImplementedError('mark legacy/string background mode partition is unassessed')
    try:mask=_core.mark_boundaries_mask(label_img,selected,word,representable,workers)
    except ValueError as error:
        if 'frontier'in str(error):raise NotImplementedError(str(error))from None
        raise
    if legacy_string:
        # NumPy1 numeric-array == string returns scalar False (FutureWarning).
        # Source ~False is Python -1; adjacent bool & -1 promotes to default
        # integer, then boundaries &= integer fails its same_kind output cast.
        warnings.warn('elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison',FutureWarning,stacklevel=2)
        exceptions=__import__('numpy.core._exceptions',fromlist=['_UFuncOutputCastingError'])
        raise exceptions._UFuncOutputCastingError(np.bitwise_and,'same_kind',np.result_type(np.dtype('bool'),-1),np.dtype('bool'),2)
    if outline_color is not None:
        if mask.ndim!=2:raise IndexError('tuple index out of range')
        outline=_core.mark_boundaries_outline(mask)
        _paint(prepared,outline,outline_color)
    native_result=_paint(prepared,mask,color)
    assert native_result is True
    return prepared
