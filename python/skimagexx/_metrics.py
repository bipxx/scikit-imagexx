"""Raw-value pairwise image metrics, backed by fused native reductions.

Default FP policies use a fused kernel. Stateful call/log/print policies
materialize native stages so callbacks observe upstream evaluation order.
"""
import warnings
import numpy as np
from . import _core
from ._compat import fallback

_REAL={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64}
_RANGES={np.bool_:(False,True),np.float16:(-1,1),np.float32:(-1,1),np.float64:(-1,1)}
for _type in (np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64):
    _info=np.iinfo(_type);_RANGES[_type]=(int(_info.min),int(_info.max))
with warnings.catch_warnings(record=True) as _mean_probe:
    warnings.simplefilter('always',RuntimeWarning)
    with np.errstate(all='ignore'):np.mean(np.empty(0))
_EMPTY_MEAN_MESSAGE=str(_mean_probe[0].message)
_SCALAR_POWER_DOUBLE=((np.asarray(np.float32(.5))-np.asarray(np.float32(0)))**2).dtype==np.float64


def _prepare(first,second):
    if first.shape!=second.shape:raise ValueError('Input images must have the same dimensions.')
    if not (type(first) in (np.ndarray,np.memmap) or isinstance(first,np.generic)):return None
    if not (type(second) in (np.ndarray,np.memmap) or isinstance(second,np.generic)):return None
    if first.dtype.kind=='c' or second.dtype.kind=='c':raise ValueError('complex valued input is not supported')
    if first.dtype.type not in _REAL or second.dtype.type not in _REAL:return None
    return np.float32 if first.dtype.char in 'ef' and second.dtype.char in 'ef' else np.float64


def _emit(flags,operation,dtype):
    # Tiny metadata probes reproduce NumPy's exact warning/error policy and
    # wording. Image subtraction/products/reductions are exclusively native.
    if not flags:return
    typ=np.dtype(dtype).type;maximum=np.finfo(dtype).max
    if flags&1:
        with np.errstate(invalid='ignore',under='ignore'):
            if operation=='subtract':np.subtract(np.array([maximum],dtype),np.array([-maximum],dtype))
            elif operation in ('square','multiply'):
                value=np.array([maximum],dtype)
                np.square(value) if operation=='square' else np.multiply(value,value)
            else:np.add.reduce(np.array([maximum,maximum],dtype),dtype=dtype)
    if flags&2:
        with np.errstate(over='ignore',invalid='ignore'):
            small=np.array([np.finfo(dtype).tiny],dtype)
            if operation=='subtract':np.subtract(small,small*typ(.5))
            elif operation=='square':np.square(small)
            elif operation=='multiply':np.multiply(small,small)
    if flags&4:
        with np.errstate(over='ignore',under='ignore'):
            if operation=='subtract':np.subtract(np.array([np.inf],dtype),np.array([np.inf],dtype))
            elif operation in ('square','multiply'):
                value=np.array([np.inf],dtype);zero=np.array([0],dtype)
                np.multiply(value,zero) if operation=='multiply' else np.square(np.array([np.nan],dtype))
            else:np.add.reduce(np.array([np.inf,-np.inf],dtype),dtype=dtype)


def _mean(total,size,dtype):
    if size==0:warnings.warn(_EMPTY_MEAN_MESSAGE,RuntimeWarning,stacklevel=3)
    value=np.dtype(dtype).type(total)
    return np.dtype(dtype).type(value/np.intp(size))


def _casts(result):
    for key in ('cast0_flags','cast1_flags'):
        if result[key]&4:
            np.array([0x7f800001],np.uint32).view(np.float32).astype(np.float64)


def _error(result,size,dtype,scalar=False):
    _emit(result['subtract_flags'],'subtract',dtype)
    if scalar and result['square_flags']:
        # Zero-dimensional subtraction produces a NumPy scalar; its power
        # reports 'scalar power' rather than the array ufunc's 'square'.
        flags=result['square_flags'];typ=np.dtype(dtype).type
        if flags&1:
            with np.errstate(invalid='ignore',under='ignore'):typ(np.finfo(dtype).max)**2
        if flags&2:
            with np.errstate(over='ignore',invalid='ignore'):typ(np.finfo(dtype).tiny)**2
        if flags&4:
            bits=np.array([0x7f800001],np.uint32) if dtype is np.float32 else np.array([0x7ff0000000000001],np.uint64)
            with np.errstate(over='ignore',under='ignore'):bits.view(dtype)[0]**2
    else:_emit(result['square_flags'],'square',dtype)
    _emit(result['error_reduce_flags'],'reduce',np.float64)
    return _mean(result['error'],size,np.float64)


def _reduce(first,second,dtype,mode):
    a=np.asarray(first) if isinstance(first,np.generic) else first
    b=np.asarray(second) if isinstance(second,np.generic) else second
    return _core.metrics_reduce(a,b,dtype is np.float32,mode,np.geterr()['under']!='ignore',
                                bool(_SCALAR_POWER_DOUBLE and dtype is np.float32 and first.ndim==0))


def _stateful():
    return any(policy not in ('warn','raise','ignore') for policy in np.geterr().values())


def _stage_emit(flags,operation,dtype,scalar=False):
    # One tiny compiled probe reproduces the *combined* status mask. No
    # temporary errstate: a callback may replace the caller's policy/handler.
    if not flags:return
    typ=np.dtype(dtype).type;info=np.finfo(dtype)
    if operation=='cast':
        np.array([0x7f800001],np.uint32).view(np.float32).astype(np.float64)
        return
    if scalar:
        value=typ(info.max if flags&1 else info.tiny)
        if flags&4:
            bits=np.array([0x7f800001],np.uint32) if dtype is np.float32 else np.array([0x7ff0000000000001],np.uint64)
            value=bits.view(dtype)[0]
        value**2
        return
    if operation=='reduce':
        values=([info.max,info.max] if flags&1 else [])+([np.inf,-np.inf] if flags&4 else [])
        np.add.reduce(np.array(values,dtype),dtype=dtype)
        return
    if operation=='subtract':
        first=([info.max] if flags&1 else [])+([np.inf] if flags&4 else [])
        second=([-info.max] if flags&1 else [])+([np.inf] if flags&4 else [])
        np.subtract(np.array(first,dtype),np.array(second,dtype))
        return
    values=([info.max] if flags&1 else [])+([info.tiny] if flags&2 else [])+([0] if flags&4 else [])
    probe=np.array(values,dtype)
    if flags&4:
        if dtype is np.float32:probe.view(np.uint32)[-1]=0x7f800001
        else:probe.view(np.uint64)[-1]=0x7ff0000000000001
    np.square(probe) if operation=='square' else np.multiply(probe,probe)


def _stage_cast(image,dtype):
    image=np.asarray(image) if isinstance(image,np.generic) else image
    if image.dtype==np.dtype(dtype):return image
    output,flags=_core.metrics_stage(image,image,dtype is np.float32,0)
    _stage_emit(flags,'cast',dtype)
    return output


def _stage_pair(first,second,dtype):
    return _stage_cast(first,dtype),_stage_cast(second,dtype)


def _stage_mean(image,dtype):
    size=image.size
    total,flags=_core.metrics_stage_sum(image,dtype is np.float32)
    _stage_emit(flags,'reduce',dtype)
    return _mean(total,size,dtype)


def _stage_error(first,second,dtype):
    delta,flags=_core.metrics_stage(first,second,dtype is np.float32,1)
    _stage_emit(flags,'subtract',dtype)
    power_dtype=np.float64 if _SCALAR_POWER_DOUBLE and dtype is np.float32 and delta.ndim==0 else dtype
    squared,flags=_core.metrics_stage(delta,delta,power_dtype is np.float32,2)
    _stage_emit(flags,'square',power_dtype,scalar=delta.ndim==0)
    return _stage_mean(squared,np.float64)


def _stage_nrmse(first,second,dtype,normalization):
    first,second=_stage_pair(first,second,dtype)
    normalization=normalization.lower()
    if normalization=='euclidean':
        product,flags=_core.metrics_stage(first,first,dtype is np.float32,3)
        _stage_emit(flags,'multiply',dtype)
        denominator=np.sqrt(_stage_mean(product,np.float64))
    elif normalization=='mean':denominator=_stage_mean(first,dtype)
    elif normalization=='min-max':
        high=_core.metrics_stage_extremum(first,True)
        low=_core.metrics_stage_extremum(first,False)
        denominator=high-low
    else:raise ValueError('Unsupported norm_type')
    return np.sqrt(mean_squared_error(first,second))/denominator


def mean_squared_error(image0,image1):
    dtype=_prepare(image0,image1)
    if dtype is None:return fallback('metrics','mean_squared_error',image0,image1)
    if _stateful():return _stage_error(*_stage_pair(image0,image1,dtype),dtype)
    data=_reduce(image0,image1,dtype,0)
    _casts(data)
    return _error(data,image0.size,dtype,image0.ndim==0)


def normalized_root_mse(image_true,image_test,*,normalization='euclidean'):
    dtype=_prepare(image_true,image_test)
    if dtype is None:return fallback('metrics','normalized_root_mse',image_true,image_test,normalization=normalization)
    if _stateful():return _stage_nrmse(image_true,image_test,dtype,normalization)
    modes={'euclidean':1,'min-max':3,'mean':2}
    if type(normalization) not in (str,np.str_) or normalization.lower() not in modes:
        # Upstream converts both arrays before inspecting this parameter.
        # Preserve possible signaling-NaN cast warnings even on an error path.
        probe=_reduce(image_true,image_test,dtype,0)
        _casts(probe)
    normalization=normalization.lower()
    if normalization=='euclidean':code=1
    elif normalization=='min-max':code=3
    elif normalization=='mean':code=2
    else:raise ValueError('Unsupported norm_type')
    if normalization=='min-max' and not image_true.size:raise ValueError('zero-size array to reduction operation maximum which has no identity')
    data=_reduce(image_true,image_test,dtype,code)
    _casts(data)
    if normalization=='euclidean':
        _emit(data['energy_flags'],'multiply',dtype)
        _emit(data['energy_reduce_flags'],'reduce',np.float64)
        denominator=np.sqrt(_mean(data['energy'],image_true.size,np.float64))
    elif normalization=='mean':
        _emit(data['mean_reduce_flags'],'reduce',dtype)
        denominator=_mean(data['mean'],image_true.size,dtype)
    else:
        low=high=np.dtype(dtype).type(np.nan) if data['nan'] else None
        if 'minmax_scalars' in data:low,high=data['minmax_scalars']
        if not data['nan']:low=np.dtype(dtype).type(data['low']);high=np.dtype(dtype).type(data['high'])
        denominator=high-low
    return np.sqrt(_error(data,image_true.size,dtype,image_true.ndim==0))/denominator


def peak_signal_noise_ratio(image_true,image_test,*,data_range=None):
    if image_true.shape!=image_test.shape:raise ValueError('Input images must have the same dimensions.')
    known=lambda image:(type(image) in (np.ndarray,np.memmap) or isinstance(image,np.generic))
    if not known(image_true) or not known(image_test):
        return fallback('metrics','peak_signal_noise_ratio',image_true,image_test,data_range=data_range)
    if any(image.dtype.type not in _REAL and image.dtype.kind!='c' for image in (image_true,image_test)):
        return fallback('metrics','peak_signal_noise_ratio',image_true,image_test,data_range=data_range)
    stateful=_stateful()
    if data_range is None:
        if image_true.dtype!=image_test.dtype:
            warnings.warn('Inputs have mismatched dtype.  Setting data_range based on image_true.',UserWarning,stacklevel=2)
        dmin,dmax=_RANGES[image_true.dtype.type]
        if stateful:
            prepared=np.asarray(image_true) if isinstance(image_true,np.generic) else image_true
            true_min=_core.metrics_stage_extremum(prepared,False)
            true_max=_core.metrics_stage_extremum(prepared,True)
        else:
            low,high,nan=_core.metrics_raw_range(np.asarray(image_true) if isinstance(image_true,np.generic) else image_true)
            true_min,true_max=(np.nan,np.nan) if nan else (low[()],high[()])
        if true_max>dmax or true_min<dmin:
            raise ValueError('image_true has intensity values outside the range expected for its data type. Please manually specify the data_range.')
        data_range=dmax if true_min>=0 else dmax-dmin
    dtype=_prepare(image_true,image_test)
    if dtype is None:return fallback('metrics','peak_signal_noise_ratio',image_true,image_test,data_range=data_range)
    if stateful:
        first,second=_stage_pair(image_true,image_test,dtype)
        error=mean_squared_error(first,second)
    else:
        data=_reduce(image_true,image_test,dtype,0)
        _casts(data)
        error=_error(data,image_true.size,dtype,image_true.ndim==0)
    data_range=float(data_range)
    return 10*np.log10((data_range**2)/error)


__all__=['mean_squared_error','normalized_root_mse','peak_signal_noise_ratio']
