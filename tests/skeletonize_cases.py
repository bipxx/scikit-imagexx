"""Frozen skeletonize recipes extracted from the two retained stock profiles."""
from hashlib import sha256


def digest(raw):
    return sha256(raw).hexdigest()


def fixture(spec,np):
    h,w = spec['shape']
    r,c = np.ogrid[:h,:w]
    name = spec['recipe']
    if name=='diamond':
        field = abs(r-h//2) + abs(c-w//2) <= min(h,w)//2-2
    elif name=='islands':
        field = (r%17<3)&(c%23<4)
    elif name=='rect':
        inside = ((r>=h//8)&(r<h-h//8)&(c>=w//8)&(c<w-w//8)) | ((c==w//2)&(r<h//3))
        field = np.where(inside,1+(r+3*c)%251,0)
    else:
        field = (r%17<5)|(c%23<7)
    return field.astype(spec['dtype'])


def snapshot(array):
    return dict(type=type(array).__module__+'.'+type(array).__name__,shape=list(array.shape),strides=list(array.strides),
                dtype=array.dtype.str,metadata=None if array.dtype.metadata is None else dict(array.dtype.metadata),
                c=array.flags.c_contiguous,f=array.flags.f_contiguous,owns=array.flags.owndata,
                aligned=array.flags.aligned,writeable=array.flags.writeable,base_type=type(array.base).__name__,
                sha256=digest(array.tobytes()))


FIXED = ({'id': 'tiny-diamond', 'shape': [17, 19], 'recipe': 'diamond', 'dtype': 'bool', 'method': None}, {'id': 'sparse-islands', 'shape': [384, 512], 'recipe': 'islands', 'dtype': 'bool', 'method': None}, {'id': 'dense-edged-rectangle', 'shape': [149, 187], 'recipe': 'rect', 'dtype': 'uint8', 'method': None}, {'id': 'explicit-zhang-junctions', 'shape': [256, 320], 'recipe': 'junctions', 'dtype': 'bool', 'method': 'zhang'})


def cases(np):
    plan = {"fixed_cases": FIXED}
    for spec in FIXED:
        yield spec["id"], fixture(spec, np), spec["method"]
    image=fixture(plan['fixed_cases'][0],np)
    readonly=image.copy()
    readonly.flags.writeable=False
    asymmetric=np.zeros((8,9),bool)
    asymmetric[0:7,1:7]=True
    asymmetric[4:8,5:9]=True
    controls=[
        ('edge-full',np.ones((4,7),bool),None,True),
        ('order-asymmetric',asymmetric,None,True),
        ('all-zero',np.zeros((7,9),bool),None,True),
        ('one-row',np.ones((1,8),bool),None,True),
        ('one-column',np.ones((8,1),bool),None,True),
        ('strided',image[::2,::-2],None,True),
        ('fortran',np.asfortranarray(image),None,True),
        ('tagged-bool',image.view(np.dtype(bool,metadata={'source':'tagged'})),None,True),
        ('equiv-uint8',image.astype(np.uint8).view(np.dtype('uint8').newbyteorder('=')),'zhang',True),
        ('readonly',readonly,None,True),
        ('empty',np.empty((0,5),bool),None,True),
        ('rank-one',image.ravel().copy(),None,False),
        ('invalid-method',image,'x',False),
        ('zhang-3d',image[None,...].copy(),'zhang',False),
    ]
    for name, image, method, _ in controls:
        yield name, image, method
