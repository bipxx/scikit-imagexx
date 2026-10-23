"""Raw-value pairwise image metrics, backed by fused native reductions.

Native FP policies are warn, raise and ignore. Stateful call/log/print
policies use explicit compatibility fallback until stagewise execution exists.
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
    if any(policy not in ('warn','raise','ignore') for policy in np.geterr().values()):return None
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


def mean_squared_error(image0,image1):
    dtype=_prepare(image0,image1)
    if dtype is None:return fallback('metrics','mean_squared_error',image0,image1)
    data=_reduce(image0,image1,dtype,0)
    _casts(data)
    return _error(data,image0.size,dtype,image0.ndim==0)


def normalized_root_mse(image_true,image_test,*,normalization='euclidean'):
    dtype=_prepare(image_true,image_test)
    if dtype is None:return fallback('metrics','normalized_root_mse',image_true,image_test,normalization=normalization)
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
    if any(policy not in ('warn','raise','ignore') for policy in np.geterr().values()):
        return fallback('metrics','peak_signal_noise_ratio',image_true,image_test,data_range=data_range)
    if data_range is None:
        if image_true.dtype!=image_test.dtype:
            warnings.warn('Inputs have mismatched dtype.  Setting data_range based on image_true.',UserWarning,stacklevel=2)
        dmin,dmax=_RANGES[image_true.dtype.type]
        low,high,nan=_core.metrics_raw_range(np.asarray(image_true) if isinstance(image_true,np.generic) else image_true)
        true_min,true_max=(np.nan,np.nan) if nan else (low[()],high[()])
        if true_max>dmax or true_min<dmin:
            raise ValueError('image_true has intensity values outside the range expected for its data type. Please manually specify the data_range.')
        data_range=dmax if true_min>=0 else dmax-dmin
    dtype=_prepare(image_true,image_test)
    if dtype is None:return fallback('metrics','peak_signal_noise_ratio',image_true,image_test,data_range=data_range)
    data=_reduce(image_true,image_test,dtype,0)
    _casts(data)
    error=_error(data,image_true.size,dtype,image_true.ndim==0)
    data_range=float(data_range)
    return 10*np.log10((data_range**2)/error)


__all__=['mean_squared_error','normalized_root_mse','peak_signal_noise_ratio']
