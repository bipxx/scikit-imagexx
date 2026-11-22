"""Frozen single-layer pyramid recipes and array observations."""

from hashlib import sha256

FIXED = ({'id': 'reduce-odd-f32-reflect',
  'api': 'pyramid_reduce',
  'shape': [193, 257],
  'dtype': 'float32',
  'options': {}},
 {'id': 'reduce-planar-u8-constant',
  'api': 'pyramid_reduce',
  'shape': [3, 157, 211],
  'dtype': 'uint8',
  'options': {'downscale': 1.7,
              'mode': 'constant',
              'cval': 7,
              'preserve_range': True,
              'channel_axis': 0}},
 {'id': 'expand-odd-f32-wrap',
  'api': 'pyramid_expand',
  'shape': [105, 143],
  'dtype': 'float32',
  'options': {'upscale': 1.6, 'sigma': 0.8, 'mode': 'wrap'}},
 {'id': 'expand-rgb-u8-reflect',
  'api': 'pyramid_expand',
  'shape': [63, 89, 3],
  'dtype': 'uint8',
  'options': {'channel_axis': -1}})


def digest(raw):
    return sha256(raw).hexdigest()


def fixture(shape, dtype, np):
    coords = np.ogrid[tuple(slice(0, n) for n in shape)]
    image = sum((axis + 3) * grid + grid**2 for axis, grid in enumerate(coords)) % 251
    if dtype == 'bool':
        return image % 5 < 2
    if dtype == 'int16':
        return (image - 123).astype(dtype)
    if dtype.startswith('float'):
        return ((image - 123) / 67).astype(dtype)
    return image.astype(dtype)


def cases(plan, np):
    for row in plan['fixed']:
        yield row['id'], row['api'], fixture(row['shape'],row['dtype'],np), row['options']
    basic = fixture((5,7),'uint8',np)
    for api, scale in [('pyramid_reduce','downscale'),('pyramid_expand','upscale')]:
        base = {scale:2.3}
        for key, image, options in [
            ('scaled-u8',basic,base),
            ('bool-order-one',fixture((5,7),'bool',np),{'order':1}),
            ('half',fixture((5,7),'float16',np),base),
            ('signed-preserve',fixture((5,7),'int16',np),{'preserve_range':True,'sigma':0.75}),
            ('middle-strided',fixture((5,7,3),'float32',np).transpose(0,2,1),{**base,'channel_axis':1}),
            ('axis-modulo',fixture((5,7,3),'float64',np),{**base,'channel_axis':5}),
            ('fortran-constant',np.asfortranarray(fixture((5,7),'float32',np)),{'mode':'constant','cval':-1.75}),
            ('wrap-zero-sigma-nearest-order',basic,{'mode':'wrap','sigma':0.,'order':0}),
            ('cubic',basic,{'sigma':0.6,'order':3}),
            ('all-nan-warning',np.full((4,5),np.nan),{'mode':'constant','cval':-1.}),
            ('invalid-scale-before-conversion',basic,{scale:1}),
            ('nan-scale-after-conversion',basic,{scale:float('nan')}),
            ('negative-sigma',basic,{'sigma':-0.5}),
            ('invalid-order',basic,{'order':7}),
            ('edge-mode',basic,{'mode':'edge'}),
            ('symmetric-mode',basic,{'mode':'symmetric'}),
            ('scipy-nearest-mode',basic,{'mode':'nearest'}),
            ('complex-conversion-error',basic.astype('complex128'),{}),
        ]:
            yield api.split('_')[-1]+'-'+key, api, image, options


def array_state(array, with_bytes=True):
    out = dict(shape=list(array.shape), dtype=array.dtype.str,
               dtype_metadata=dict(array.dtype.metadata) if array.dtype.metadata else None,
               strides=list(array.strides), owns=bool(array.flags.owndata), writeable=bool(array.flags.writeable),
               c=bool(array.flags.c_contiguous), f=bool(array.flags.f_contiguous), base=type(array.base).__name__)
    if with_bytes:
        raw = array.tobytes()
        out['sha256'] = digest(raw)
        if array.size <= 180:
            out['hex'] = raw.hex()
    return out
