"""Native component consumers and returned ArrayMap contract, interface1.

Parameter/state metadata follows scikit-image0.26 (BSD-3-Clause; see
third_party/scikit-image-LICENSE.txt). Pixel scans live in native stages.
"""
from functools import lru_cache,wraps
import warnings
import numpy as np
from . import _core
from ._components import label as _label

try:
    # Identity of the deprecated sentinel is scalar API metadata, not an
    # upstream consumer implementation. Its explicit use remains compatible.
    from skimage._shared.utils import DEPRECATED as _DEPRECATED
except ImportError:
    class _Deprecated:
        def __repr__(self):return '<DEPRECATED>'
    _DEPRECATED=_Deprecated()


def _size_deprecation(old):
    def decorate(function):
        @wraps(function)
        def call(*args,**kwargs):
            previous=_DEPRECATED;replacement=_DEPRECATED
            if len(args)>1:
                previous=args[1];args=args[:1]+(_DEPRECATED,)+args[2:]
            if old in kwargs:
                previous=kwargs[old];kwargs[old]=_DEPRECATED
            if len(args)>3:replacement=args[3]
            if 'max_size' in kwargs:replacement=kwargs['max_size']
            if previous is not _DEPRECATED:
                warnings.warn(f'Parameter `{old}` is deprecated since version 0.26.0 and will be removed in 2.0.0 (or later). '
                              'To avoid this warning, please use the parameter `max_size` instead. For more details, '
                              f'see the documentation of `{function.__name__}`. Note that the new threshold removes objects '
                              'smaller than **or equal to** its value, while the previous parameter only removed smaller ones.',FutureWarning,stacklevel=2)
                if replacement is not _DEPRECATED:
                    raise ValueError(f'Both deprecated parameter `{old}` and new parameter `max_size` are used. Use only the latter to avoid conflicting values.')
                kwargs['max_size']=previous
            return function(*args,**kwargs)
        return call
    return decorate


def _check_dtype(ar):
    if not (ar.dtype==bool or np.issubdtype(ar.dtype,np.integer)):
        raise TypeError(f'Only bool or integer image types are supported. Got {ar.dtype}.')


@lru_cache(maxsize=128)
def _count_admission(dtype):
    # Empty metadata input reproduces NumPy's dtype admission and exact error
    # category/message; all actual counts are computed by our C++ kernel.
    np.bincount(np.empty(0,dtype=dtype))


def _threshold(value,empty=False):
    metadata=np.empty(0,dtype=np.intp) if empty else np.array([0],dtype=np.intp)
    metadata<=value
    if empty:return 0.
    if isinstance(value,(list,tuple,np.ndarray)):
        metadata=np.asarray(value)
        if metadata.size==1:value=metadata.reshape(())[()]
    try:return float(value)
    except OverflowError:return float('inf') if value>0 else -float('inf')


def _binary_labels(ar,connectivity):
    if connectivity<1:connectivity=1
    keys=_core.consumer_bool_keys(ar)
    if ar.ndim==0:return _core.consumer_indices(keys)
    effective=sum(bool(step<=connectivity) for step in range(1,ar.ndim+1))
    # An all-false SciPy footprint (e.g. NaN connectivity) isolates every
    # foreground pixel. Public label's connectivity0 rejection must not leak.
    if effective==0:return _core.consumer_isolated_labels(keys)
    return _core.components_label(keys,0,effective,True,np.dtype(np.intp))[0]


def _remove_objects(ar,max_size,connectivity,out):
    _check_dtype(ar)
    if out is None:out=_core.consumer_copy(ar,None,False)
    else:_core.consumer_assign(out,ar)
    if max_size==0:return out
    if out.dtype==bool:components=_binary_labels(ar,connectivity)
    else:
        _count_admission(out.dtype)
        components=_core.consumer_indices(out)
    try:counts=_core.consumer_count(components)
    except ValueError:
        raise ValueError('Negative value labels are not supported. Try relabeling the input with `scipy.ndimage.label` or `skimage.morphology.label`.') from None
    if not counts.empty and counts.maximum==1 and out.dtype!=bool:
        warnings.warn('Only one label was provided to `remove_small_objects`. Did you mean to use a boolean array?',UserWarning,stacklevel=3)
    return _core.consumer_filter(components,counts,_threshold(max_size,counts.empty),out)


def _invert(ar,out):
    if not out.flags.writeable:raise ValueError('output array is read-only')
    if ar.shape!=out.shape:
        text=lambda shape:'('+','.join(str(n) for n in shape)+')'
        try:combined=np.broadcast_shapes(ar.shape,out.shape)
        except ValueError:
            raise ValueError(f'operands could not be broadcast together with shapes {text(ar.shape)} {text(out.shape)} ') from None
        if combined!=out.shape:
            raise ValueError(f"non-broadcastable output operand with shape {text(out.shape)} doesn't match the broadcast shape {text(combined)}")
        ar=np.broadcast_to(ar,out.shape)
    return _core.consumer_invert(ar,out)


@_size_deprecation('min_size')
def remove_small_objects(ar,min_size=_DEPRECATED,connectivity=1,*,max_size=64,out=None):
    return _remove_objects(ar,max_size,connectivity,out)


@_size_deprecation('area_threshold')
def remove_small_holes(ar,area_threshold=_DEPRECATED,connectivity=1,*,max_size=64,out=None):
    _check_dtype(ar)
    if ar.dtype!=bool:
        warnings.warn('Any labeled images will be returned as a boolean array. Did you mean to use a boolean array?',UserWarning,stacklevel=2)
    if out is not None:
        if out.dtype!=bool:raise TypeError('out dtype must be bool')
    else:out=_core.consumer_copy(ar,np.dtype(bool),True)
    _invert(ar,out)
    _remove_objects(out,max_size,connectivity,out)
    _invert(out,out)
    return out


def clear_border(labels,buffer_size=0,bgval=0,mask=None,*,out=None):
    if any(buffer_size>=length for length in labels.shape) and mask is None:
        raise ValueError('buffer size may not be greater than labels size')
    if out is None:out=_core.consumer_copy(labels,None,False)
    if mask is not None:
        if out.shape!=mask.shape:raise TypeError('exceptions must derive from BaseException')
        if mask.dtype!=bool:raise TypeError('mask should be of type bool.')
        borders=_core.consumer_empty(out.shape,np.dtype(bool))
        _invert(mask,borders)
    else:
        ext=buffer_size+1
        stops=[slice(None,ext).indices(length)[1] for length in out.shape]
        starts=[slice(-ext,None).indices(length)[0] for length in out.shape]
        borders=_core.consumer_border(out.shape,stops,starts)
    components,number=_label(out,background=0,return_num=True)
    return _core.consumer_clear(_core.consumer_indices(components),borders,number,out,bgval)


@lru_cache(maxsize=1)
def _shape_warnings():
    # Profile the native NumPy metadata setter, whose warning changed in2.5.
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');metadata=np.empty(1).view();metadata.shape=(-1,)
    return [(type(w.message),str(w.message)) for w in notices]


def _mapping_notice():
    for category,message in _shape_warnings():warnings.warn(message,category,stacklevel=3)


def relabel_sequential(label_field,offset=1):
    if offset<=0:raise ValueError('Offset must be strictly positive.')
    if _core.consumer_min_negative(label_field):raise ValueError('Cannot relabel array that contains negative values.')
    offset=int(offset)
    if label_field.dtype.kind not in 'iu':raise TypeError('label_field must have an integer dtype')
    old=_core.consumer_unique(label_field)
    zero=bool(old[0]==0)
    last=np.int64(0 if zero and len(old)==1 else offset+len(old)-int(zero)-1)
    required=np.min_scalar_type(last);input_type=label_field.dtype
    if input_type.itemsize<required.itemsize:output_type=required
    else:output_type=input_type if last<np.iinfo(input_type).max else required
    values=_core.consumer_sequence(len(old),offset,zero,np.dtype(output_type))
    output=_core.consumer_empty(label_field.shape,np.dtype(output_type))
    _mapping_notice()
    _core.consumer_map(label_field,old,values,output)
    return output,ArrayMap(old,values),ArrayMap(values,old)


class ArrayMap:
    def __init__(self,in_values,out_values):
        self.in_values=in_values;self.out_values=out_values
        self._max_str_lines=4;self._array=None

    def __len__(self):
        kind=getattr(getattr(self.in_values,'dtype',None),'kind',None)
        maximum=_core.consumer_max(self.in_values) if kind in ('i','u') else _core.consumer_dense_max(self.in_values)
        return maximum+1

    def __array__(self,dtype=None,copy=None):
        dtype=self.out_values.dtype if dtype is None else dtype
        kind=getattr(getattr(self.in_values,'dtype',None),'kind',None)
        maximum=_core.consumer_max(self.in_values) if kind in ('i','u') else _core.consumer_dense_max(self.in_values)
        # The scalar size (and overflow warning) precedes dtype admission.
        return _core.consumer_dense(self.in_values,self.out_values,maximum+1,dtype)

    @property
    def dtype(self):return self.out_values.dtype

    def __repr__(self):return f'ArrayMap({self.in_values!r}, {self.out_values!r})'

    def __str__(self):
        if len(self.in_values)<=self._max_str_lines+1:rows=list(range(len(self.in_values)))
        else:rows=list(range(self._max_str_lines//2))+[None]+list(range(-self._max_str_lines//2,0))
        return '\n'.join(['ArrayMap:']+[('  ...' if i is None else f'  {self.in_values[i]} \u2192 {self.out_values[i]}') for i in rows])

    def __call__(self,arr):return self.__getitem__(arr)

    def __getitem__(self,index):
        scalar=np.isscalar(index)
        if scalar:index=np.array([index])
        elif isinstance(index,slice):
            index=_core.consumer_arange(index.start or 0,index.stop if index.stop is not None else len(self),index.step)
        if index.dtype==bool:index=_core.consumer_compact(index)[0]
        if index.dtype.kind not in 'iu':raise TypeError('The dtype of an array to be remapped should be integer.')
        _mapping_notice()
        out=_core.consumer_map(index,self.in_values,self.out_values)
        return out[0] if scalar else out

    def __setitem__(self,indices,values):
        if self._array is None:self._array=self.__array__()
        _core.consumer_scatter(self._array,indices,values)
        self.in_values,self.out_values=_core.consumer_compact(self._array)
