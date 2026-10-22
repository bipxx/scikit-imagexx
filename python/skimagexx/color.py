"""Color API compatible with scikit-image."""
import numpy as np
import warnings
from functools import lru_cache as _lru_cache
from . import _core
from ._whitepoints import ILLUMINANTS as _ILLUMINANTS
from ._stain_matrices import MATRICES as _STAIN_MATRICES
from ._compat import retain_dtype_metadata, fallback, module_attribute, native_image
from . import _rgb_colors as _LABEL_COLORS

_LABEL_COLOR_DICT={k:v for k,v in vars(_LABEL_COLORS).items() if isinstance(v,tuple)}
_LABEL_DEFAULT_COLORS=('red','blue','yellow','magenta','green','indigo','darkorange','cyan','pink','yellowgreen')
_LABEL_TYPES={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64}


def _label_rgb_vector(value):
    if type(value) is str:value=_LABEL_COLOR_DICT[value]
    if type(value) not in (list,tuple,np.ndarray):return None
    if type(value) is np.ndarray:
        if value.ndim!=1 or value.dtype.kind not in 'biuf' or value.dtype.itemsize>8:return None
    elif any(type(x) not in (int,float,bool,np.float16,np.float32,np.float64,np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64) for x in value):return None
    result=np.array(value[:3])
    if result.shape!=(3,) or result.dtype.kind not in 'biuf' or result.dtype.itemsize>8:return None
    if not np.isfinite(result).all():return None
    return result


def label2rgb(label,image=None,colors=None,alpha=0.3,bg_label=0,bg_color=(0,0,0),
              image_alpha=1,kind='overlay',*,saturation=0,channel_axis=-1):
    def delegated():
        return fallback('color','label2rgb',label,image,colors,alpha,bg_label,bg_color,image_alpha,kind,
                        saturation=saturation,channel_axis=channel_axis)
    if (type(label) is not np.ndarray or label.dtype.type not in _LABEL_TYPES
            or type(bg_label) is not int or not 0<=bg_label<=2**64-1
            or type(channel_axis) is not int or type(kind) is not str):return delegated()
    if image is not None and (type(image) is not np.ndarray or image.dtype.type not in (np.uint8,np.float32,np.float64)
                               or not image.dtype.isnative):return delegated()
    if np.geterr()['under']!='ignore':return delegated()
    overlay=kind=='overlay'
    if kind not in ('overlay','avg'):
        if image is not None:np.moveaxis(image,channel_axis,-1)
        raise ValueError("`kind` must be either 'overlay' or 'avg'.")
    vectors=None
    if overlay:
        if type(saturation) not in (int,float,bool) or not 0<=saturation<=1:return delegated()
        choices=_LABEL_DEFAULT_COLORS if colors is None else colors
        if type(choices) not in (list,tuple,np.ndarray):return delegated()
        if type(choices) is np.ndarray and (choices.ndim!=2 or choices.dtype.kind not in 'biuf'):return delegated()
        vectors=[_label_rgb_vector(c) for c in choices]
        if not vectors or any(v is None for v in vectors):return delegated()
    if image is not None:
        moved=np.moveaxis(image,channel_axis,-1)
        if overlay:
            if moved.shape[:label.ndim]!=label.shape or moved.ndim>label.ndim+1:
                raise ValueError('`image` and `label` must be the same shape')
            if moved.ndim==label.ndim+1 and moved.shape[-1]!=3:
                raise ValueError('`image` must be RGB (image.shape[-1] must be 3).')
            if any(type(x) not in (int,float,bool) or not 0<=x<=1 for x in (alpha,image_alpha)):return delegated()
        elif moved.shape not in (label.shape,label.shape+(3,)):return delegated()
        if not moved.size and overlay:np.empty(0,dtype=moved.dtype).min()
        if not _core.label_image_valid(moved,overlay):return delegated()
    elif not overlay:return delegated()
    # Upstream's 0D nonzero/indexing path differs between NumPy1 and NumPy2;
    # retain that scalar/error contract before grouping any pixels.
    if not overlay and label.ndim==0:return delegated()
    if not label.size and overlay:np.empty(0,dtype=label.dtype).min()
    mapped=_core.label_ranks(label,bg_label,overlay)
    if mapped is None:return delegated()
    ranks,regions,background,present=mapped
    if not overlay:
        probe=np.zeros((1,3),dtype=moved.dtype)
        if present:probe[...]=bg_color
        result=_core.label_average(ranks,regions,moved,moved.ndim==label.ndim+1,probe,present)
        return np.moveaxis(result,-1,channel_axis)
    if image is not None and moved.ndim==label.ndim+1:
        hsv=rgb2hsv(moved)
        moved=hsv2rgb(_core.label_saturation(hsv,float(saturation)))
    background_vector=_label_rgb_vector((0,0,0) if bg_color is None else bg_color)
    if background_vector is None:return delegated()
    needed=max(0,regions-1)
    palette=np.stack([background_vector,*vectors[:min(needed,len(vectors))]])
    effective_alpha=1 if image is None else alpha
    palette=palette*effective_alpha
    if palette.shape[0]==1:palette=np.concatenate((palette,palette),axis=0)
    dtype=np.result_type(palette.dtype,np.float64 if image is None or moved.dtype==np.uint8 else moved.dtype)
    if dtype not in (np.dtype('float32'),np.dtype('float64')):return delegated()
    result=_core.label_overlay(ranks,None if image is None else moved,palette,
        image is not None and moved.ndim==label.ndim+1,
        1 if image is None else image_alpha,0 if image is None else 1-image_alpha,
        0 if image is None else 1-alpha,bg_color is None and present,background,dtype)
    return np.moveaxis(result,-1,channel_axis)

__all__ = ['rgb2gray', 'gray2rgb', 'rgb2yuv', 'yuv2rgb', 'rgb2yiq', 'yiq2rgb',
           'rgb2ypbpr', 'ypbpr2rgb', 'rgb2ycbcr', 'ycbcr2rgb', 'rgb2ydbdr',
           'ydbdr2rgb', 'rgb2rgbcie', 'rgbcie2rgb', 'rgb2xyz', 'xyz2rgb']
__all__ += ['rgb2hsv','hsv2rgb','xyz2lab','lab2xyz','rgb2lab','lab2rgb',
            'lab2lch','lch2lab','xyz_tristimulus_values']
__all__ += ['xyz2luv','luv2xyz','rgb2luv','luv2rgb']
__all__ += ['separate_stains','combine_stains','rgb2hed','hed2rgb']
__all__ += ['deltaE_cie76','convert_colorspace']
__all__ += ['deltaE_ciede94','deltaE_ciede2000','deltaE_cmc']
globals().update(_STAIN_MATRICES)
_STAIN_LOG_ADJUST=-np.log(1e-6)


def rgb2gray(rgb, *, channel_axis=-1):
    if isinstance(channel_axis, (int, np.integer)) and isinstance(rgb, np.ndarray) and rgb.ndim == 3:
        moved = np.moveaxis(rgb, channel_axis, -1)
        if native_image(moved, rgb=True):
            return retain_dtype_metadata(_core.rgb2gray(moved),moved)
    return fallback('color', 'rgb2gray', rgb, channel_axis=channel_axis)


def _stack_axis(axis,ndim):
    try:from numpy._core.multiarray import normalize_axis_index
    except ImportError:from numpy.core.multiarray import normalize_axis_index
    return normalize_axis_index(axis,ndim)


def _pixel_order(arr,trailing_channel=False):
    dimensions=arr.ndim-int(trailing_channel)
    order=tuple(sorted(range(dimensions),key=lambda i:abs(arr.strides[i]),reverse=True))
    if order==tuple(range(dimensions)):return arr,None
    axes=order+((dimensions,) if trailing_channel else ())
    return arr.transpose(axes),order


def _restore_pixels(result,order):
    if order is None:return result
    inverse=tuple(order.index(i) for i in range(len(order)))
    return result.transpose(inverse+(len(order),))


def _field_byte_order(source,target):
    if source.itemsize!=target.itemsize:return None
    order=list(range(source.itemsize))
    def visit(a,b,offset):
        if a.fields is not None:
            if b.fields is None or a.names!=b.names:return False
            for name in a.names:
                first,start=a.fields[name][:2];second,end=b.fields[name][:2]
                if start!=end or not visit(first,second,offset+start):return False
            return True
        if a.subdtype is not None:
            if b.subdtype is None:return False
            first,shape=a.subdtype;second,other=b.subdtype
            if shape!=other or first.itemsize!=second.itemsize:return False
            for index in range(int(np.prod(shape))):
                if not visit(first,second,offset+index*first.itemsize):return False
            return True
        if a==b:return True
        if b!=a.newbyteorder('='):return False
        component=a.itemsize//2 if a.kind=='c' else 4 if a.kind=='U' else a.itemsize
        if component not in (2,4,8):return False
        for start in range(0,a.itemsize,component):
            for byte in range(component):order[offset+start+byte]=offset+start+component-1-byte
        return True
    return tuple(order) if visit(source,target,0) else None


def _expansion_format(arr):
    target=np.result_type(arr.dtype,arr.dtype)
    if target==arr.dtype:return target,0
    if arr.dtype.fields is not None:
        order=_field_byte_order(arr.dtype,target)
        return (target,order) if order is not None else (None,0)
    if target!=arr.dtype.newbyteorder('='):return None,0
    width=arr.dtype.itemsize
    component=width//2 if arr.dtype.kind=='c' else 4 if arr.dtype.kind=='U' else width
    if component not in (2,4,8):return None,0
    return target,component


def gray2rgb(image, *, channel_axis=-1):
    if isinstance(image,np.ndarray) and type(image) is not np.ndarray:
        return fallback('color','gray2rgb',image,channel_axis=channel_axis)
    arr=np.asarray(image)
    if arr.dtype.hasobject:return fallback('color','gray2rgb',image,channel_axis=channel_axis)
    target,swap=_expansion_format(arr)
    if target is None:return fallback('color','gray2rgb',image,channel_axis=channel_axis)
    axis=_stack_axis(channel_axis,arr.ndim+1)
    ordered,order=_pixel_order(arr)
    result=_restore_pixels(_core.gray_expand(ordered,None,3,target,swap),order)
    return np.moveaxis(result,-1,axis)


def __getattr__(name):
    return module_attribute('color', name)


_TRANSFORMS = {name: i for i, name in enumerate(__all__[2:16])}
_COLOR_TYPES = {np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
                np.int32, np.uint64, np.int64, np.float32, np.float64}


def _transform(name, image, channel_axis):
    if isinstance(image, np.ndarray) and type(image) is not np.ndarray:
        return fallback('color', name, image, channel_axis=channel_axis)
    a = np.asarray(image)
    axis = channel_axis
    if axis is None:
        axis = -1
    elif not np.isscalar(axis):
        if len(axis) > 1:
            raise ValueError('only a single channel axis is currently supported')
        axis = axis[0]
    if name == 'ycbcr2rgb' and axis == -1 and not isinstance(image, np.ndarray):
        # Upstream copies/indexes before _prepare_colorarray for this function.
        image.copy()[..., 0]
    moved = a if axis == -1 else np.moveaxis(a, axis, -1)
    if name == 'ycbcr2rgb' and (moved.ndim == 0 or moved.shape[-1] < 3 or moved.dtype.type in (np.bool_, np.int8)):
        # Offset subtraction precedes upstream shape/type preparation. Check
        # its indexing and scalar-casting errors without touching image pixels.
        for j, amount in enumerate((16,128,128)):
            moved[...,j]
            if moved.dtype.type in (np.bool_, np.int8):
                probe = np.empty(0, dtype=moved.dtype)
                np.subtract(probe, amount, out=probe)
    if moved.shape[-1] != 3:
        raise ValueError(f'the input array must have size 3 along `channel_axis`, got {moved.shape}')
    if moved.dtype.type not in _COLOR_TYPES or not moved.dtype.isnative:
        return fallback('color', name, image, channel_axis=channel_axis)
    result = retain_dtype_metadata(_core.color_transform(moved, _TRANSFORMS[name]),moved)
    return result if axis == -1 else np.moveaxis(result, -1, axis)


def rgb2yuv(rgb, *, channel_axis=-1):
    return _transform('rgb2yuv', rgb, channel_axis)


def yuv2rgb(yuv, *, channel_axis=-1):
    return _transform('yuv2rgb', yuv, channel_axis)


def rgb2yiq(rgb, *, channel_axis=-1):
    return _transform('rgb2yiq', rgb, channel_axis)


def yiq2rgb(yiq, *, channel_axis=-1):
    return _transform('yiq2rgb', yiq, channel_axis)


def rgb2ypbpr(rgb, *, channel_axis=-1):
    return _transform('rgb2ypbpr', rgb, channel_axis)


def ypbpr2rgb(ypbpr, *, channel_axis=-1):
    return _transform('ypbpr2rgb', ypbpr, channel_axis)


def rgb2ycbcr(rgb, *, channel_axis=-1):
    return _transform('rgb2ycbcr', rgb, channel_axis)


def ycbcr2rgb(ycbcr, *, channel_axis=-1):
    return _transform('ycbcr2rgb', ycbcr, channel_axis)


def rgb2ydbdr(rgb, *, channel_axis=-1):
    return _transform('rgb2ydbdr', rgb, channel_axis)


def ydbdr2rgb(ydbdr, *, channel_axis=-1):
    return _transform('ydbdr2rgb', ydbdr, channel_axis)


def rgb2rgbcie(rgb, *, channel_axis=-1):
    return _transform('rgb2rgbcie', rgb, channel_axis)


def rgbcie2rgb(rgbcie, *, channel_axis=-1):
    return _transform('rgbcie2rgb', rgbcie, channel_axis)


def rgb2xyz(rgb, *, channel_axis=-1):
    return _transform('rgb2xyz', rgb, channel_axis)


def xyz2rgb(xyz, *, channel_axis=-1):
    return _transform('xyz2rgb', xyz, channel_axis)


def xyz_tristimulus_values(*, illuminant, observer, dtype=float):
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return np.asarray(_ILLUMINANTS[illuminant][observer],dtype=dtype)
    except KeyError:
        raise ValueError(f'Unknown illuminant/observer combination (`{illuminant}`, `{observer}`)')


_PERCEPTUAL = {name:i for i,name in enumerate(('rgb2hsv','hsv2rgb','xyz2lab','lab2xyz','rgb2lab','lab2rgb',
                                             'lab2lch','lch2lab','xyz2luv','luv2xyz','rgb2luv','luv2rgb'))}


@_lru_cache(maxsize=64)
def _luv_reference(white,dtype):
    # Three-value metadata only. NumPy's dot rounding affects singular inverse
    # denominators, so preserve its installed BLAS behavior without image work.
    w=np.asarray(white,dtype=dtype)
    denominator=np.array([1,15,3],dtype=dtype)@w
    return float(4*w[0]/denominator),float(9*w[1]/denominator)


def _perceptual(name, image, channel_axis, illuminant='D65', observer='2'):
    cie = name in ('xyz2lab','lab2xyz','rgb2lab','lab2rgb','xyz2luv','luv2xyz')
    options = {'channel_axis':channel_axis}
    if cie:
        options.update(illuminant=illuminant,observer=observer)
    if isinstance(image,np.ndarray) and type(image) is not np.ndarray:
        return fallback('color',name,image,**options)
    axis = channel_axis
    if axis is None:
        axis = -1
    elif not np.isscalar(axis):
        if len(axis) > 1:
            raise ValueError('only a single channel axis is currently supported')
        axis = axis[0]
    if name in ('rgb2hsv','xyz2luv') and axis == -1:
        image.ndim  # Upstream inspects this before coercing array-like input.
    a = np.asarray(image)
    moved = a if axis == -1 else np.moveaxis(a,axis,-1)
    polar = name in ('lab2lch','lch2lab')
    if polar and moved.shape[-1] < 3:
        raise ValueError('Input image has less than 3 channels.')
    if not polar and moved.shape[-1] != 3:
        shape = (1,)+moved.shape if name in ('rgb2hsv','xyz2luv') and moved.ndim == 1 else moved.shape
        raise ValueError(f'the input array must have size 3 along `channel_axis`, got {shape}')
    if moved.dtype.type not in _COLOR_TYPES or not moved.dtype.isnative:
        return fallback('color',name,image,**options)
    white = xyz_tristimulus_values(illuminant=illuminant,observer=observer) if cie or name in ('rgb2luv','luv2rgb') else (1.0,1.0,1.0)
    uv=_luv_reference(tuple(white),np.dtype(np.float32 if moved.dtype==np.float32 else np.float64).str) if 'luv' in name else None
    result,negative_z,unsupported = _core.perceptual_transform(moved,_PERCEPTUAL[name],white,uv)
    if name in ('rgb2hsv','lab2lch','lch2lab'):result=retain_dtype_metadata(result,moved)
    if unsupported:
        return fallback('color',name,image,**options)
    if negative_z:
        target = 'XYZ' if name == 'lab2xyz' else 'sRGB'
        via = '' if name == 'lab2xyz' else ', via XYZ'
        warnings.warn(f'Conversion from CIE-LAB{via} to {target} color space resulted in '
                      f'{negative_z} negative Z values that have been clipped to zero',UserWarning,stacklevel=3)
    return result if axis == -1 else np.moveaxis(result,-1,axis)


def rgb2hsv(rgb, *, channel_axis=-1):
    return _perceptual('rgb2hsv',rgb,channel_axis)


def hsv2rgb(hsv, *, channel_axis=-1):
    return _perceptual('hsv2rgb',hsv,channel_axis)


def xyz2lab(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('xyz2lab',xyz,channel_axis,illuminant,observer)


def lab2xyz(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('lab2xyz',lab,channel_axis,illuminant,observer)


def rgb2lab(rgb, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('rgb2lab',rgb,channel_axis,illuminant,observer)


def lab2rgb(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('lab2rgb',lab,channel_axis,illuminant,observer)


def lab2lch(lab, *, channel_axis=-1):
    return _perceptual('lab2lch',lab,channel_axis)


def lch2lab(lch, *, channel_axis=-1):
    return _perceptual('lch2lab',lch,channel_axis)


def xyz2luv(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('xyz2luv',xyz,channel_axis,illuminant,observer)


def luv2xyz(luv, illuminant='D65', observer='2', *, channel_axis=-1):
    return _perceptual('luv2xyz',luv,channel_axis,illuminant,observer)


def rgb2luv(rgb, *, channel_axis=-1):
    return _perceptual('rgb2luv',rgb,channel_axis)


def luv2rgb(luv, *, channel_axis=-1):
    return _perceptual('luv2rgb',luv,channel_axis)


def _stain(name,image,matrix,channel_axis):
    options={'channel_axis':channel_axis}
    args=(image,matrix) if name in ('separate_stains','combine_stains') else (image,)
    if isinstance(image,np.ndarray) and type(image) is not np.ndarray:
        return fallback('color',name,*args,**options)
    if isinstance(matrix,np.ndarray) and type(matrix) is not np.ndarray:
        return fallback('color',name,*args,**options)
    axis=channel_axis
    if axis is None: axis=-1
    elif not np.isscalar(axis):
        if len(axis)>1: raise ValueError('only a single channel axis is currently supported')
        axis=axis[0]
    a=np.asarray(image)
    moved=a if axis==-1 else np.moveaxis(a,axis,-1)
    if moved.shape[-1]!=3:
        raise ValueError(f'the input array must have size 3 along `channel_axis`, got {moved.shape}')
    mat=np.asarray(matrix)
    if moved.dtype.type not in _COLOR_TYPES or not moved.dtype.isnative or mat.dtype.kind not in 'buif' or mat.ndim!=2 or mat.shape[0]!=3:
        return fallback('color',name,*args,**options)
    prepared=np.float32 if moved.dtype==np.float32 else np.float64
    intermediate=np.result_type(np.empty(0,dtype=prepared),_STAIN_LOG_ADJUST)
    output_dtype=np.result_type(intermediate,mat.dtype)
    if output_dtype not in (np.dtype(np.float32),np.dtype(np.float64)):
        return fallback('color',name,*args,**options)
    weights=np.array(mat.T,dtype=np.float64,order='C',copy=True)
    result=_core.stain_transform(moved,weights,name in ('combine_stains','hed2rgb'),
                                 intermediate==np.float32,output_dtype==np.float32,float(_STAIN_LOG_ADJUST))
    result=retain_dtype_metadata(result,moved)
    return result if axis==-1 else np.moveaxis(result,-1,axis)


def separate_stains(rgb,conv_matrix,*,channel_axis=-1):
    return _stain('separate_stains',rgb,conv_matrix,channel_axis)


def combine_stains(stains,conv_matrix,*,channel_axis=-1):
    return _stain('combine_stains',stains,conv_matrix,channel_axis)


def rgb2hed(rgb,*,channel_axis=-1):
    return _stain('rgb2hed',rgb,_STAIN_MATRICES['hed_from_rgb'],channel_axis)


def hed2rgb(hed,*,channel_axis=-1):
    return _stain('hed2rgb',hed,_STAIN_MATRICES['rgb_from_hed'],channel_axis)


def _distance_metadata(result,a,b,first_scalar,second_scalar,reverse):
    if result.ndim==0 or (a.dtype.metadata is None and b.dtype.metadata is None):return result
    prepared=np.float32 if a.dtype.type is b.dtype.type is np.float32 else np.float64
    def component(image,scalar):
        # Component extraction turns a one-color vector into NumPy scalars,
        # which cannot carry dtype tags. Type promotion also drops old tags.
        dtype=image.dtype if image.dtype.type is prepared else np.dtype(prepared)
        return prepared(0) if scalar else np.empty(0,dtype=dtype)
    first=component(a,first_scalar);second=component(b,second_scalar)
    # The initial lightness term determines the result descriptor. CIE76 and
    # CIEDE2000 subtract comparison-reference; CIE94 and CMC reverse that order.
    descriptor=(second-first if reverse else first-second).dtype
    if descriptor.metadata is not None and descriptor==result.dtype:return result.view(descriptor)
    return result


def deltaE_cie76(lab1,lab2,channel_axis=-1):
    a,b=np.asarray(lab1),np.asarray(lab2)
    first_scalar=a.ndim==1;second_scalar=b.ndim==1
    if any(x.dtype.type not in _COLOR_TYPES or not x.dtype.isnative for x in (a,b)):
        return fallback('color','deltaE_cie76',lab1,lab2,channel_axis=channel_axis)
    a,b=np.moveaxis(a,channel_axis,-1),np.moveaxis(b,channel_axis,-1)
    for x in (a,b):
        if x.shape[-1]<3:
            raise ValueError(f'not enough values to unpack (expected 3, got {x.shape[-1]})')
    try:
        a,b=np.broadcast_arrays(a[...,:3],b[...,:3])
    except ValueError:
        # Ask the same ufunc to diagnose incompatible pixel shapes, using
        # zero-stride scalar views: no image arithmetic or allocation runs.
        scalar=np.empty((),dtype=np.float64)
        np.subtract(np.broadcast_to(scalar,b.shape[:-1]),np.broadcast_to(scalar,a.shape[:-1]))
        raise
    result=_core.delta_e76(a,b)
    result=_distance_metadata(result,a,b,first_scalar,second_scalar,True)
    return result[()] if result.ndim==0 else result


def convert_colorspace(arr,fromspace,tospace,*,channel_axis=-1):
    # Match upstream's route through RGB, including RGB->RGB identity and the
    # intermediate allocation/rounding of non-RGB pairs.
    def identity(image,*,channel_axis):
        return image
    fromdict={'rgb':identity,'hsv':hsv2rgb,'rgb cie':rgbcie2rgb,'xyz':xyz2rgb,
              'yuv':yuv2rgb,'yiq':yiq2rgb,'ypbpr':ypbpr2rgb,'ycbcr':ycbcr2rgb,'ydbdr':ydbdr2rgb}
    todict={'rgb':identity,'hsv':rgb2hsv,'rgb cie':rgb2rgbcie,'xyz':rgb2xyz,
            'yuv':rgb2yuv,'yiq':rgb2yiq,'ypbpr':rgb2ypbpr,'ycbcr':rgb2ycbcr,'ydbdr':rgb2ydbdr}
    fromspace,tospace=fromspace.lower(),tospace.lower()
    if fromspace not in fromdict:
        raise ValueError(f'`fromspace` has to be one of {fromdict.keys()}')
    if tospace not in todict:
        raise ValueError(f'`tospace` has to be one of {todict.keys()}')
    return todict[tospace](fromdict[fromspace](arr,channel_axis=channel_axis),channel_axis=channel_axis)


def _weighted_distance(name,lab1,lab2,channel_axis,parameters,options):
    a,b=np.asarray(lab1),np.asarray(lab2)
    if any(x.dtype.type not in _COLOR_TYPES or not x.dtype.isnative for x in (a,b)):
        return fallback('color',name,lab1,lab2,channel_axis=channel_axis,**options)
    if any(type(v) not in (int,float,np.float32) for v in parameters):
        return fallback('color',name,lab1,lab2,channel_axis=channel_axis,**options)
    try:
        valid=all(np.isfinite(float(v)) for v in parameters)
    except OverflowError:
        valid=False
    if not valid:
        return fallback('color',name,lab1,lab2,channel_axis=channel_axis,**options)
    first_scalar=a.ndim==1;second_scalar=b.ndim==1
    both_scalar=first_scalar and second_scalar
    if name=='deltaE_ciede2000': channel_axis=channel_axis%a.ndim
    a,b=np.moveaxis(a,channel_axis,-1),np.moveaxis(b,channel_axis,-1)
    for x in (a,b):
        if x.shape[-1]<3:
            if name=='deltaE_ciede2000':
                raise ValueError(f'not enough values to unpack (expected 3, got {x.shape[-1]})')
            raise ValueError('Input image has less than 3 channels.')
    single=a.dtype==b.dtype==np.float32
    if single and any(abs(float(v))>np.finfo(np.float32).max for v in parameters):
        return fallback('color',name,lab1,lab2,channel_axis=channel_axis,**options)
    prepared=np.float32 if single else np.float64
    probe=np.empty((),dtype=prepared) if first_scalar and name=='deltaE_cmc' else np.empty(0,dtype=prepared)
    wide=np.result_type(probe,np.deg2rad(30))==np.float64
    promote=single and both_scalar and name=='deltaE_cmc'
    try:
        a,b=np.broadcast_arrays(a[...,:3],b[...,:3])
    except ValueError:
        scalar=np.empty((),dtype=np.float64)
        np.add(np.broadcast_to(scalar,a.shape[:-1]),np.broadcast_to(scalar,b.shape[:-1]))
        raise
    result=_core.distance_metric(a,b,{'deltaE_ciede94':0,'deltaE_ciede2000':1,'deltaE_cmc':2}[name],
                                 parameters,bool(wide),promote)
    result=_distance_metadata(result,a,b,first_scalar,second_scalar,name=='deltaE_ciede2000')
    return result[()] if result.ndim==0 else result


def deltaE_ciede94(lab1,lab2,kH=1,kC=1,kL=1,k1=.045,k2=.015,*,channel_axis=-1):
    return _weighted_distance('deltaE_ciede94',lab1,lab2,channel_axis,(kL,kC,kH,k1,k2),
                              {'kH':kH,'kC':kC,'kL':kL,'k1':k1,'k2':k2})


def deltaE_ciede2000(lab1,lab2,kL=1,kC=1,kH=1,*,channel_axis=-1):
    return _weighted_distance('deltaE_ciede2000',lab1,lab2,channel_axis,(kL,kC,kH,.045,.015),
                              {'kL':kL,'kC':kC,'kH':kH})


def deltaE_cmc(lab1,lab2,kL=1,kC=1,*,channel_axis=-1):
    return _weighted_distance('deltaE_cmc',lab1,lab2,channel_axis,(kL,kC,1,.045,.015),{'kL':kL,'kC':kC})



def _alpha_flags(image):
    if image.dtype.fields is None:return _core.alpha_array_flags(_pixel_order(image)[0])
    flags=0
    for name in image.dtype.names:flags|=_alpha_flags(image[name])
    return flags


def gray2rgba(image,alpha=None,*,channel_axis=-1):
    def forwarded():return fallback('color','gray2rgba',image,alpha,channel_axis=channel_axis)
    if isinstance(image,np.ndarray) and type(image) is not np.ndarray:return forwarded()
    arr=np.asarray(image)
    if arr.dtype.hasobject:return forwarded()
    target,swap=_expansion_format(arr)
    if target is None:return forwarded()
    if alpha is None:
        if arr.dtype.type not in _COLOR_TYPES|{np.float16}:return forwarded()
        from .util import dtype_limits
        _,actual_alpha=dtype_limits(arr,clip_negative=False)
    else:actual_alpha=alpha
    if isinstance(actual_alpha,np.ndarray) and type(actual_alpha) is not np.ndarray:return forwarded()
    source_alpha=np.asarray(actual_alpha)
    if source_alpha.size<=1:
        with np.errstate(over='ignore',under='ignore'):
            alpha_arr=source_alpha.astype(arr.dtype)
        unsafe=not np.array_equal(alpha_arr,actual_alpha)
    else:
        if source_alpha.dtype!=arr.dtype or source_alpha.dtype.hasobject:return forwarded()
        flags=_alpha_flags(source_alpha)
        if flags&6:return forwarded()
        unsafe=bool(flags&1);alpha_arr=source_alpha
    if unsafe:warnings.warn(f'alpha cannot be safely cast to image dtype {arr.dtype.name}',UserWarning,stacklevel=2)
    try:alpha_arr=np.broadcast_to(alpha_arr,arr.shape)
    except ValueError as error:raise ValueError('alpha.shape must match image.shape') from error
    axis=_stack_axis(channel_axis,arr.ndim+1)
    ordered,order=_pixel_order(arr)
    ordered_alpha=alpha_arr.transpose(order) if order is not None else alpha_arr
    result=_restore_pixels(_core.gray_expand(ordered,ordered_alpha,4,target,swap),order)
    return np.moveaxis(result,-1,axis)


def rgba2rgb(rgba,background=(1,1,1),*,channel_axis=-1):
    def forwarded():return fallback('color','rgba2rgb',rgba,background,channel_axis=channel_axis)
    if isinstance(rgba,np.ndarray) and type(rgba) is not np.ndarray:return forwarded()
    arr=np.asanyarray(rgba)
    if not isinstance(channel_axis,int):raise TypeError('channel_axis must be an integer')
    if channel_axis < -arr.ndim or channel_axis>=arr.ndim:
        axis_error=np.exceptions.AxisError if hasattr(np,'exceptions') else np.AxisError
        raise axis_error('channel_axis exceeds array dimensions')
    axis=channel_axis%arr.ndim
    if arr.shape[axis]!=4:raise ValueError(f'the input array must have size 4 along `channel_axis`, got {arr.shape}')
    if arr.dtype.type not in _COLOR_TYPES or not arr.dtype.isnative:return forwarded()
    if arr.size and arr.dtype.kind=='f':
        low,high=_core.intensity_bounds(arr.ravel(order='K'))
        if not (np.isfinite(low) and np.isfinite(high)):return forwarded()
        if max(abs(low),abs(high))>np.sqrt(np.finfo(arr.dtype).max)/2:return forwarded()
    if isinstance(background,np.ndarray) and type(background) is not np.ndarray:return forwarded()
    raw_background=np.asanyarray(background)
    if raw_background.dtype.kind in 'OcV' or _core.alpha_array_flags(raw_background)&6:return forwarded()
    float_dtype=np.float32 if arr.dtype==np.float32 else np.float64
    converted_background=raw_background.ravel().astype(float_dtype)
    if converted_background.size!=3:
        raise ValueError(f'background must be an array-like containing 3 RGB values. Got {converted_background.size} items')
    if np.any(converted_background<0) or np.any(converted_background>1):raise ValueError('background RGB values must be floats between 0 and 1.')
    moved=np.moveaxis(arr,axis,-1)
    ordered,order=_pixel_order(moved,trailing_channel=True)
    result=_restore_pixels(_core.rgba_composite(ordered,converted_background),order)
    return np.moveaxis(result,-1,axis)


__all__ += ['rgba2rgb','gray2rgba']
