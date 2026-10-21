"""Color API compatible with scikit-image."""
import numpy as np
import warnings
from functools import lru_cache as _lru_cache
from . import _core
from ._whitepoints import ILLUMINANTS as _ILLUMINANTS
from ._stain_matrices import MATRICES as _STAIN_MATRICES
from ._compat import fallback, module_attribute, native_image

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
            return _core.rgb2gray(moved)
    return fallback('color', 'rgb2gray', rgb, channel_axis=channel_axis)


def gray2rgb(image, *, channel_axis=-1):
    if native_image(image) and isinstance(channel_axis, (int, np.integer)) and -3 <= channel_axis < 3:
        return np.moveaxis(_core.gray2rgb(image), -1, channel_axis)
    return fallback('color', 'gray2rgb', image, channel_axis=channel_axis)


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
    result = _core.color_transform(moved, _TRANSFORMS[name])
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
    return result if axis==-1 else np.moveaxis(result,-1,axis)


def separate_stains(rgb,conv_matrix,*,channel_axis=-1):
    return _stain('separate_stains',rgb,conv_matrix,channel_axis)


def combine_stains(stains,conv_matrix,*,channel_axis=-1):
    return _stain('combine_stains',stains,conv_matrix,channel_axis)


def rgb2hed(rgb,*,channel_axis=-1):
    return _stain('rgb2hed',rgb,_STAIN_MATRICES['hed_from_rgb'],channel_axis)


def hed2rgb(hed,*,channel_axis=-1):
    return _stain('hed2rgb',hed,_STAIN_MATRICES['rgb_from_hed'],channel_axis)


def deltaE_cie76(lab1,lab2,channel_axis=-1):
    a,b=np.asarray(lab1),np.asarray(lab2)
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
    first_scalar=a.ndim==1
    both_scalar=a.ndim==b.ndim==1
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
    return result[()] if result.ndim==0 else result


def deltaE_ciede94(lab1,lab2,kH=1,kC=1,kL=1,k1=.045,k2=.015,*,channel_axis=-1):
    return _weighted_distance('deltaE_ciede94',lab1,lab2,channel_axis,(kL,kC,kH,k1,k2),
                              {'kH':kH,'kC':kC,'kL':kL,'k1':k1,'k2':k2})


def deltaE_ciede2000(lab1,lab2,kL=1,kC=1,kH=1,*,channel_axis=-1):
    return _weighted_distance('deltaE_ciede2000',lab1,lab2,channel_axis,(kL,kC,kH,.045,.015),
                              {'kL':kL,'kC':kC,'kH':kH})


def deltaE_cmc(lab1,lab2,kL=1,kC=1,*,channel_axis=-1):
    return _weighted_distance('deltaE_cmc',lab1,lab2,channel_axis,(kL,kC,1,.045,.015),{'kL':kL,'kC':kC})
