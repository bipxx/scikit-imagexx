"""Prospective storage closure; unchanged test_color_transforms error budget."""
API='color.rgb2yuv'
STAGE='color_transform'
def cases():
    rows=[]
    def add(name,dtype,layout='C',axis=-1,kind='ordinary',policy=None):
        rows.append(dict(id=name,dtype=dtype,layout=layout,axis=axis,kind=kind,policy=policy or {}))
    for layout in ('C','F','reverse','step','unaligned','readonly','tagged'):
        add('half-'+layout,'float16',layout)
    add('half-swapped','>f2')
    add('f32-swapped','>f4');add('f32-swapped-F','>f4','F')
    add('f32-axis0','>f4',axis=0);add('f64-swapped','>f8')
    add('f64-axis-tuple','>f8',axis=(0,));add('half-axis-none','float16',axis=None)
    add('half-empty','float16',kind='empty');add('half-invalid-shape','float16',kind='invalid-shape')
    add('half-nonfinite-warn','float16',kind='nonfinite',policy={'invalid':'warn'})
    add('half-nonfinite-raise','float16',kind='nonfinite',policy={'invalid':'raise'})
    assert len(rows)==18
    return rows

def fixture(spec,np):
    image=(np.arange(24).reshape(2,4,3)-12).astype(spec['dtype'])/np.array(8,dtype=spec['dtype'])
    # Division may choose native output; restore the declared storage descriptor.
    image=image.astype(spec['dtype'])
    if spec['kind']=='empty':image=np.empty((0,3),dtype=spec['dtype'])
    elif spec['kind']=='invalid-shape':image=np.zeros((2,4),dtype=spec['dtype'])
    elif spec['kind']=='nonfinite':image.flat[:6]=[np.nan,0,1,np.inf,-np.inf,0]
    layout=spec['layout']
    if layout=='F':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1,::-1]
    elif layout=='step':
        owner=np.zeros((4,8,3),dtype=image.dtype);owner[::2,::2]=image;image=owner[::2,::2]
    elif layout=='unaligned':
        owner=np.empty(image.nbytes+1,np.uint8);view=np.ndarray(image.shape,image.dtype,owner,1);view[:]=image;image=view
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='tagged':image=image.view(np.dtype(image.dtype,metadata={'storage':'half'}))
    axis=spec['axis']
    if axis not in(-1,None):image=np.moveaxis(image,-1,axis[0]if isinstance(axis,tuple)else axis)
    return image,dict(channel_axis=axis)

def limits(image,expected,matrix,np):
    tol=2e-6 if expected.dtype==np.float32 else 2e-13
    absolute=tol
    if expected.dtype==np.float32:
        magnitude=np.max(np.abs(image)[np.isfinite(image)],initial=0)
        absolute+=4*np.finfo(np.float32).eps*np.max(np.sum(np.abs(matrix),axis=1))*magnitude
    return tol,absolute
