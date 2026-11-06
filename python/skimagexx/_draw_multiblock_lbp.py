"""Native ordered LBP phases with explicit storage/metadata/policy providers.

NumPy owns coercion/copy representation, dtype metadata and error descriptors.
C++ owns normalized pixels, gray replication, arithmetic and assignment.
User-defined alpha arithmetic remains that argument's named protocol provider.
No original scikit-image target or helper is imported.
"""
import importlib
import warnings
import numpy as np
from . import _core


def _broadcast(a,b):
    result=[]
    for k in range(1,max(len(a),len(b))+1):
        x=a[-k]if k<=len(a)else 1;y=b[-k]if k<=len(b)else 1
        if x!=y and x!=1 and y!=1:
            def fmt(s):return '('+','.join(map(str,s))+(','if len(s)==1 else'')+')'
            raise ValueError(f'operands could not be broadcast together with shapes {fmt(a)} {fmt(b)} ')
        result.append(y if x==1 else x)
    return tuple(reversed(result))


def _deliver(flags,operation):
    policy=np.geterr();handler=np.geterrcall()
    for bit,key,label in ((1,'divide','divide by zero'),(2,'over','overflow'),(4,'under','underflow'),(8,'invalid','invalid value')):
        if not flags&bit:continue
        action=policy[key];message=f'{label} encountered in {operation}'
        if action=='warn':warnings.warn(message,RuntimeWarning,stacklevel=3)
        elif action=='raise':raise FloatingPointError(message)
        elif action=='call':
            if handler is None:raise NameError(f'python callback specified for {label} (in  {operation}) but no function found.')
            handler(label,flags)
        elif action=='print':print(f'Warning: {message}')
        elif action=='log':
            if handler is None:raise NameError(f'log specified for {label} (in {operation}) but no object with write method found.')
            handler.write(f'Warning: {message}\n')


def _binary(a,b,plus=False):
    # Resolve weak Python vs strong NumPy scalar metadata before transport.
    dtype=np.result_type(a,b);aa=np.asarray(a);bb=np.asarray(b)
    shape=_broadcast(aa.shape,bb.shape)
    if dtype.kind!='f':raise NotImplementedError('LBP nonfloating phase provider')
    # Keep floating payload precision/byte order as supplied. The native phase
    # performs its working-precision conversions, rather than NumPy casting
    # an entire patch before the arithmetic loop. Nonnumeric scalar-carrier
    # coercion remains an explicit array representation provider.
    if aa.dtype.kind!='f':aa=np.asarray(a,dtype=dtype)
    if bb.dtype.kind!='f':bb=np.asarray(b,dtype=dtype)
    result,flags=_core._lbp_binary(aa,bb,shape,dtype,plus)
    _deliver(flags,'add'if plus else'multiply')
    return result


def _offset_size(value):
    dtype=np.asarray(value).dtype
    if dtype.kind not in 'biu':
        exceptions=importlib.import_module('numpy._core._exceptions'if np.__version__.split('.')[0]!='1'else'numpy.core._exceptions')
        raise exceptions._UFuncOutputCastingError(np.multiply,'same_kind',np.result_type(np.dtype('intp'),dtype),np.dtype('intp'),2)
    return value


def draw_multiblock_lbp(image,r,c,width,height,lbp_code=0,
                       color_greater_block=(1,1,1),
                       color_less_block=(0,0.69,0.96),alpha=0.5):
    greater=np.asarray(color_greater_block,dtype=np.float64)
    less=np.asarray(color_less_block,dtype=np.float64)
    copied=np.copy(image)
    gray=len(image.shape)<3
    if gray:
        # Source np.stack asks each of its three argument protocols in order;
        # only coercion is provided here, C++ owns actual gray replication.
        planes=[np.asanyarray(image)for _ in range(3)]
        source=planes[0]
    else:source=copied
    if source.dtype.kind not in 'biuf':raise ValueError(f'Cannot convert from {source.dtype.name} to float64.')
    output=_core._lbp_prepare(source)
    if not gray:
        stored=np.empty_like(copied,dtype=output.dtype,order='K',subok=False)
        _core._lbp_assign(stored,output);output=stored
    height=_offset_size(height);width=_offset_size(width)
    central_r=r+height;central_c=c+width
    for element,(dy,dx)in enumerate(((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))):
        rr=central_r+dy*height;cc=central_c+dx*width
        selected=greater if lbp_code&(1<<(7-element))else less
        complement=1-alpha
        patch=output[rr:rr+height,cc:cc+width]
        first=_binary(complement,patch)
        if type(alpha)not in(int,float,bool)and not isinstance(alpha,(np.generic,np.ndarray)):
            # This explicitly named argument provider owns its computation.
            second=alpha*selected
        else:second=_binary(alpha,selected)
        combined=_binary(first,second,True)
        if combined.shape!=patch.shape:
            if _broadcast(combined.shape,patch.shape)!=patch.shape:
                def fmt(s):return '('+','.join(map(str,s))+')'
                raise ValueError(f'could not broadcast input array from shape {fmt(combined.shape)} into shape {fmt(patch.shape)}')
        _core._lbp_assign(patch,combined)
    return output
