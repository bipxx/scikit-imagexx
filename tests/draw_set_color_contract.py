"""Bounded source-safe set_color recipes, frozen before original calls."""
IDS=['gray', 'rgb', 'alpha-zero', 'alpha-one', 'alpha-negative', 'alpha-over', 'alpha-vector', 'duplicates', 'clip', 'all-clipped', 'empty', 'image-F', 'image-reverse', 'image-unaligned', 'color-alias', 'alpha-alias', 'readonly-color', 'integer-color', 'scalar-color', 'tagged-image', 'error-color', 'error-readonly', 'error-floatcoords', 'error-alpha-length', 'error-listcoords', 'error-unpack']
CASES=[dict(id=name)for name in IDS]

def fixture(spec,np):
    if spec.get('partition')=='float32':return float32_fixture(spec,np)
    name=spec['id'];assert name in IDS
    image=np.arange(60,dtype=np.float64).reshape(4,5,3)/64
    rr=np.array([0,1,2,3],dtype=np.intp);cc=np.array([1,2,3,4],dtype=np.intp)
    color=np.array([.25,.5,.75]);alpha=.5
    if name in ('gray','scalar-color'):image=image[:,:,0].copy();color=np.array([.25])
    if name=='scalar-color':color=.25
    if name=='alpha-zero':alpha=0.
    if name=='alpha-one':alpha=1.
    if name=='alpha-negative':alpha=-.5
    if name=='alpha-over':alpha=1.5
    if name=='alpha-vector':alpha=np.array([0.,.25,.5,1.])
    if name=='duplicates':rr=np.array([1,1,1,2]);cc=np.array([2,2,2,3]);alpha=np.array([.25,.5,.75,.125])
    if name=='clip':rr=np.array([-1,1,9,3]);cc=np.array([0,2,3,4]);alpha=np.array([.125,.25,.5,.75])
    if name=='all-clipped':rr=np.array([-1,9]);cc=np.array([0,3])
    if name=='empty':rr=cc=np.array([],dtype=np.intp)
    if name=='image-F':image=np.asfortranarray(image)
    if name=='image-reverse':image=image[::-1,::-1,::-1]
    if name=='image-unaligned':
        owner=np.zeros(image.nbytes+1,dtype=np.uint8);v=np.ndarray(image.shape,dtype=np.float64,buffer=owner,offset=1);v[:]=image;image=v
    if name=='color-alias':color=image[1,2]
    if name=='alpha-alias':alpha=image[:,2,0]
    if name=='readonly-color':color.flags.writeable=False
    if name=='integer-color':color=np.array([0,1,0],dtype=np.int64)
    if name=='tagged-image':image=image.view(np.dtype(np.float64,metadata={'tag':7}))
    if name=='error-color':color=np.array([.5,.25])
    if name=='error-readonly':image.flags.writeable=False
    if name=='error-floatcoords':rr=rr.astype(np.float64)
    if name=='error-alpha-length':alpha=np.array([.5,.5])
    if name=='error-listcoords':rr=rr.tolist();cc=cc.tolist()
    coords=(rr,cc) if name!='error-unpack' else (rr,)
    assert image.size<=60 and len(rr)<=4 and image.ndim in (2,3)
    return (image,coords,color),dict(alpha=alpha)

# Newpartition stays separate: no replay or mutation of the original26recipes.
F32_CASES=[{'id': 'f32-gray', 'partition': 'float32'}, {'id': 'f32-rgb-vector', 'partition': 'float32'}, {'id': 'f32-default-integer-palette', 'partition': 'float32'}, {'id': 'f32-integer-zero', 'partition': 'float32'}, {'id': 'f32-floating-zero', 'partition': 'float32'}, {'id': 'f32-duplicates', 'partition': 'float32'}, {'id': 'f32-clipped', 'partition': 'float32'}, {'id': 'f32-F', 'partition': 'float32'}, {'id': 'f32-reverse', 'partition': 'float32'}, {'id': 'f32-unaligned', 'partition': 'float32'}, {'id': 'f32-color-alias', 'partition': 'float32'}, {'id': 'f32-alpha-alias', 'partition': 'float32'}]

def float32_fixture(spec,np):
    assert spec in F32_CASES
    name=spec['id'];image=np.arange(48,dtype=np.float32).reshape(4,4,3)/np.float32(64)
    rr=np.array([0,1,2,3],dtype=np.intp);cc=np.array([1,2,3,0],dtype=np.intp)
    color=np.array([.25,.5,.75],dtype=np.float64);alpha=.5
    if name in('f32-gray','f32-clipped','f32-color-alias','f32-alpha-alias'):
        image=np.arange(16,dtype=np.float32).reshape(4,4)/np.float32(32);color=np.array([.25])
    if name=='f32-rgb-vector':alpha=np.array([0.,.5,1.,.5]);color.flags.writeable=False
    if name=='f32-default-integer-palette':color=np.array([0,1,0],dtype=np.int64);alpha=1
    if name in('f32-integer-zero','f32-floating-zero'):
        image=np.array([[-0.]],dtype=np.float32);rr=cc=np.array([0],dtype=np.intp);color=np.array([-1],dtype=np.int64 if name=='f32-integer-zero'else np.float64);alpha=0
    if name=='f32-duplicates':rr=np.array([1,1,1,2],dtype=np.intp);cc=np.array([2,2,2,3],dtype=np.intp);alpha=np.array([0.,.5,1.,.5])
    if name=='f32-clipped':rr=np.array([-1,1,8,3],dtype=np.intp);cc=np.array([0,2,3,0],dtype=np.intp);alpha=np.array([0.,.5,1.,.5])
    if name=='f32-F':image=np.asfortranarray(image)
    if name=='f32-reverse':image=image[::-1,::-1,::-1]
    if name=='f32-unaligned':
        owner=np.zeros(image.nbytes+1,dtype=np.uint8);view=np.ndarray(image.shape,dtype=np.float32,buffer=owner,offset=1);view[:]=image;image=view
    if name=='f32-color-alias':
        image.view(np.float64)[1,0]=.25;color=image.view(np.float64)[1,:1];color.flags.writeable=False;cc=np.array([1,0,3,0],dtype=np.intp)
    if name=='f32-alpha-alias':
        alpha=image.view(np.float64).reshape(-1)[:4];alpha[:]=[0.,.5,1.,.5]
    # Exactapprovedsource-safety intake, not inferred candidate admission.
    assert type(image)is np.ndarray and image.dtype==np.dtype(np.float32)and image.dtype.isnative and image.ndim in(2,3)and 0<image.size<=48
    assert rr.shape==cc.shape and rr.size<=4 and rr.dtype==cc.dtype==np.dtype(np.intp)
    assert np.isfinite(image).all()and np.isfinite(color).all()and np.max(np.abs(image))<=16 and np.max(np.abs(color))<=16
    assert all(float(x)in(0.,.5,1.)for x in(alpha if type(alpha)is np.ndarray else[alpha]))
    return(image,(rr,cc),color),dict(alpha=alpha)
