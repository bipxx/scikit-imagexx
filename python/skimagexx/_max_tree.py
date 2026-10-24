"""Owned max-tree wrappers with prepared C++ fast and source-provider stages.

Adapted intake/metadata orchestration from scikit-image0.26 (BSD-3-Clause;
source and license retained in the reference contract). Numerical max-tree,
attribute, filtering and label traversals execute owned C++. Observable sort
providers are honored on the staged route; pristine sorting is wholly C++.
This module alone is not public registration or full protocol acceptance.
"""
import inspect
import types
import numpy as np
from scipy import ndimage as ndi
from . import _core

_NP_NAMES = ('ones','zeros','empty','argsort','array','asarray','subtract','nonzero',
             'stack','cumprod','sqrt','sum','sort','unique','isfinite','maximum','minimum','max')
_NP_ORIGINAL = {name:getattr(np,name) for name in _NP_NAMES}
_NO_FAST = object()
_STAGE_ENTRIES = ('max_tree_build_stage','max_tree_area_stage','max_tree_extension_stage',
                  'max_tree_filter_stage','max_tree_maxima_stage')


def _require_core():
    if (getattr(_core,'max_tree_interface',None)!='max-tree-prepared-v1'
        or type(getattr(_core,'max_tree_type_capabilities',None)) is not int
        or _core.max_tree_type_capabilities!=1023
        or type(getattr(_core,'max_tree_operation_capabilities',None)) is not int
        or _core.max_tree_operation_capabilities!=63
        or getattr(_core,'max_tree_parallel_traversal',None) is not False
        or getattr(_core,'max_tree_stage_interface',None)!='max-tree-stages-v1'
        or type(getattr(_core,'max_tree_stage_capabilities',None)) is not int
        or _core.max_tree_stage_capabilities!=31):
        raise NotImplementedError('the prepared/staged max-tree native interface is unavailable')
    for name in ('max_tree_run',)+_STAGE_ENTRIES:
        function=getattr(_core,name,None)
        if not inspect.isbuiltin(function) or function.__module__!=_core.__name__ or function.__name__!=name:
            raise NotImplementedError('max-tree requires its owned native builtins')
    return _core


def invert(image):
    if image.dtype=='bool': return ~image
    if np.issubdtype(image.dtype,np.unsignedinteger):
        return np.subtract(np.iinfo(image.dtype).max,image,dtype=image.dtype)
    if np.issubdtype(image.dtype,np.signedinteger):
        return np.subtract(-1,image,dtype=image.dtype)
    return np.subtract(1,image,dtype=image.dtype)


def _validate_connectivity(image_dim,connectivity,offset):
    if connectivity is None: connectivity=1
    if np.isscalar(connectivity):
        connectivity=ndi.generate_binary_structure(image_dim,connectivity)
    else:
        connectivity=np.array(connectivity,bool)
        if connectivity.ndim!=image_dim:
            raise ValueError('Connectivity dimension must be same as image')
    if offset is None:
        if any(x%2==0 for x in connectivity.shape):
            raise ValueError('Connectivity array must have an unambiguous center')
        offset=np.array(connectivity.shape)//2
    return connectivity,offset


def _offsets_to_raveled_neighbors(image_shape,footprint,center,order='C'):
    ndim=len(image_shape)
    if not footprint.ndim==ndim==len(center):
        raise ValueError("number of dimensions in image shape, footprint and itscenter index does not match")
    offsets=np.stack([(idx-c) for idx,c in zip(np.nonzero(footprint),center)],axis=-1)
    if order=='F': offsets=offsets[:,::-1];image_shape=image_shape[::-1]
    elif order!='C': raise ValueError("order must be 'C' or 'F'")
    ravel_factors=image_shape[1:]+(1,)
    ravel_factors=np.cumprod(ravel_factors[::-1])[::-1]
    raveled_offsets=(offsets*ravel_factors).sum(axis=1)
    spacing=np.ones(ndim)
    weighted_offsets=offsets*spacing
    distances=np.sqrt(np.sum(weighted_offsets**2,axis=1))
    sorted_offsets=raveled_offsets[np.argsort(distances,kind='stable')]
    np.sort(distances,kind='stable')  # source also computes the unused distance result
    if any(x<y for x,y in zip(image_shape,footprint.shape)):
        _,indices=np.unique(sorted_offsets,return_index=True)
        indices=np.sort(indices,kind='stable');sorted_offsets=sorted_offsets[indices]
    return sorted_offsets[1:]


def _build_stage(image,mask,structure,offset,shape,parent,traverser):
    roots=parent.copy()
    points=np.zeros((structure.shape[0],len(shape)),dtype=np.int32)
    shift=np.ravel_multi_index(offset,shape)
    for i,step in enumerate(structure):
        current=np.unravel_index(step+shift,shape)
        for d in range(len(shape)): points[i,d]=current[d]-offset[d]
    _require_core().max_tree_build_stage(image,mask,structure,shape,points,parent,traverser,roots)


def _area_stage(image,parent,traverser):
    area=np.ones(len(image),dtype=np.float64)
    _require_core().max_tree_area_stage(image,parent,traverser,area)
    return area


def _extension_stage(image,shape,parent,traverser):
    extension=np.ones(len(image),dtype=np.float64)
    high=np.array(np.unravel_index(np.arange(len(image)),shape),dtype=np.float64).T
    low=np.array(np.unravel_index(np.arange(len(image)),shape),dtype=np.float64).T
    # Arbitrary replacement arithmetic ufunc callbacks are a distinct protocol
    # scope. Never silently claim that a C++ stage invoked such callbacks.
    if any(getattr(np,name) is not _NP_ORIGINAL[name] for name in ('maximum','minimum','max')):
        raise NotImplementedError('custom coordinate arithmetic providers are unassessed')
    _require_core().max_tree_extension_stage(image,shape,parent,traverser,extension,low,high)
    return extension


def _filter_stage(image,output,parent,traverser,attribute,threshold):
    _require_core().max_tree_filter_stage(image,output,parent,traverser,attribute,threshold)


def _maxima_stage(image,output,parent,traverser):
    _require_core().max_tree_maxima_stage(image,output,parent,traverser)


_max_tree=types.SimpleNamespace(_max_tree=_build_stage,_compute_area=_area_stage,
    _compute_extension=_extension_stage,_direct_filter=_filter_stage,_max_tree_local_maxima=_maxima_stage)
_STAGE_ORIGINAL=dict(vars(_max_tree))


def _try_fast(image,operation,threshold=0,connectivity=1,parent=None,tree_traverser=None):
    if type(image) is not np.ndarray: return _NO_FAST
    if any(getattr(np,name) is not function for name,function in _NP_ORIGINAL.items()): return _NO_FAST
    if any(getattr(_max_tree,name) is not function for name,function in _STAGE_ORIGINAL.items()): return _NO_FAST
    if any(globals()[name] is not function for name,function in _MODULE_ORIGINAL.items()): return _NO_FAST
    if (image.ndim not in (2,3) or any(dim<4 for dim in image.shape)
        or not image.dtype.isnative or image.dtype.kind not in 'iuf'
        or image.dtype.str.lstrip('<>=|') not in ('u1','u2','u4','u8','i1','i2','i4','i8','f4','f8')
        or not image.flags.aligned or not image.flags.writeable): return _NO_FAST
    reuse=parent is not None and tree_traverser is not None
    if reuse:
        if any(type(value) is not np.ndarray or value.dtype!=np.dtype('i8') or not value.flags.writeable
               or not value.flags.aligned or value.size!=image.size for value in (parent,tree_traverser)):
            return _NO_FAST
        if tree_traverser.ndim!=1: return _NO_FAST
    elif type(connectivity) is not int or not 1<=connectivity<=image.ndim: return _NO_FAST
    if type(threshold) not in (int,float) or (type(threshold) is int and abs(threshold)>2**53): return _NO_FAST
    prepared=image.ravel().reshape(image.shape)
    if operation==0: output=None
    elif operation==1: output=np.ones(image.shape,dtype=np.uint64)
    else: output=image.copy()
    if reuse: p=parent.ravel();s=tree_traverser
    else: p=np.zeros(image.shape,dtype=np.int64);s=np.empty(image.size,dtype=np.int64)
    try: _require_core().max_tree_run(prepared,output,p,s,operation,connectivity,threshold,reuse)
    except NotImplementedError: return _NO_FAST
    return (p,s) if operation==0 else output


def max_tree(image, connectivity=1):
    _require_core()
    fast=_try_fast(image,0,connectivity=connectivity)
    if fast is not _NO_FAST: return fast
    mask=np.ones(image.shape)
    for k in range(len(image.shape)):
        np.moveaxis(mask,k,0)[0]=0
        np.moveaxis(mask,k,0)[-1]=0
    neighbors,offset=_validate_connectivity(image.ndim,connectivity,offset=None)
    parent=np.zeros(image.shape,dtype=np.int64)
    flat=_offsets_to_raveled_neighbors(image.shape,neighbors,offset).astype(np.int32)
    traverser=np.argsort(image.ravel(),kind='stable').astype(np.int64)
    _max_tree._max_tree(image.ravel(),mask.ravel().astype(np.uint8),flat,
        offset.astype(np.int32),np.array(image.shape,dtype=np.int32),parent.ravel(),traverser)
    return parent,traverser


def area_opening(image, area_threshold=64, connectivity=1, parent=None, tree_traverser=None):
    _require_core()
    fast=_try_fast(image,2,area_threshold,connectivity,parent,tree_traverser)
    if fast is not _NO_FAST: return fast
    output=image.copy()
    if parent is None or tree_traverser is None: parent,tree_traverser=max_tree(image,connectivity)
    area=_max_tree._compute_area(image.ravel(),parent.ravel(),tree_traverser)
    _max_tree._direct_filter(image.ravel(),output.ravel(),parent.ravel(),tree_traverser,area,area_threshold)
    return output


def diameter_opening(image, diameter_threshold=8, connectivity=1, parent=None, tree_traverser=None):
    _require_core()
    fast=_try_fast(image,4,diameter_threshold,connectivity,parent,tree_traverser)
    if fast is not _NO_FAST: return fast
    output=image.copy()
    if parent is None or tree_traverser is None: parent,tree_traverser=max_tree(image,connectivity)
    diam=_max_tree._compute_extension(image.ravel(),np.array(image.shape,dtype=np.int32),parent.ravel(),tree_traverser)
    _max_tree._direct_filter(image.ravel(),output.ravel(),parent.ravel(),tree_traverser,diam,diameter_threshold)
    return output


def area_closing(image, area_threshold=64, connectivity=1, parent=None, tree_traverser=None):
    _require_core()
    fast=_try_fast(image,3,area_threshold,connectivity,parent,tree_traverser)
    if fast is not _NO_FAST: return fast
    image_inv=invert(image);output=image_inv.copy()
    if parent is None or tree_traverser is None: parent,tree_traverser=max_tree(image_inv,connectivity)
    area=_max_tree._compute_area(image_inv.ravel(),parent.ravel(),tree_traverser)
    _max_tree._direct_filter(image_inv.ravel(),output.ravel(),parent.ravel(),tree_traverser,area,area_threshold)
    return invert(output)


def diameter_closing(image, diameter_threshold=8, connectivity=1, parent=None, tree_traverser=None):
    _require_core()
    fast=_try_fast(image,5,diameter_threshold,connectivity,parent,tree_traverser)
    if fast is not _NO_FAST: return fast
    image_inv=invert(image);output=image_inv.copy()
    if parent is None or tree_traverser is None: parent,tree_traverser=max_tree(image_inv,connectivity)
    diam=_max_tree._compute_extension(image_inv.ravel(),np.array(image_inv.shape,dtype=np.int32),parent.ravel(),tree_traverser)
    _max_tree._direct_filter(image_inv.ravel(),output.ravel(),parent.ravel(),tree_traverser,diam,diameter_threshold)
    return invert(output)


def max_tree_local_maxima(image, connectivity=1, parent=None, tree_traverser=None):
    _require_core()
    fast=_try_fast(image,1,connectivity=connectivity,parent=parent,tree_traverser=tree_traverser)
    if fast is not _NO_FAST: return fast
    output=np.ones(image.shape,dtype=np.uint64)
    if parent is None or tree_traverser is None: parent,tree_traverser=max_tree(image,connectivity)
    _max_tree._max_tree_local_maxima(image.ravel(),output.ravel(),parent.ravel(),tree_traverser)
    return output


_MODULE_ORIGINAL={name:globals()[name] for name in ('max_tree','invert','_validate_connectivity','_offsets_to_raveled_neighbors')}
