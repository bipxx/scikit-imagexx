"""Frozen safe coordinate and integral-image Haar cases."""
import hashlib
import json

TYPES = ('type-2-x','type-2-y','type-3-x','type-3-y','type-4')
TIMED = ('coord-tiny','coord-mixed-large','value-small','value-roi-large','value-subset')
SUBSET = [
    [[(0,0),(1,0)],[(0,1),(1,1)],[(0,2),(1,2)]],
    [[(1,0),(1,1)],[(2,0),(2,1)]],
    [[(0,1),(0,1)],[(0,2),(0,2)],[(1,2),(1,2)],[(1,1),(1,1)]],
    [[(2,1),(2,1)],[(2,2),(2,2)]],
]
SUBSET_TYPES = ['type-3-x','type-2-y','type-4','type-2-x']


def cases():
    out=[]
    def coord(name,w,h,sel=None,**flags):
        out.append(dict(id='coord-'+name,which='coord',width=w,height=h,select=sel,**flags))
    def value(name,w=4,h=4,sel=None,**flags):
        out.append(dict(id='value-'+name,which='value',width=w,height=h,select=sel,**flags))
    coord('tiny',2,2)
    coord('mixed-large',12,9,['type-4','type-2-y','type-3-x'])
    for family in TYPES:
        coord(family,4,3,family)
    coord('row',5,1)
    coord('column',1,5)
    coord('swapped',3,5)
    coord('point',1,1)
    coord('zero-width',0,4)
    coord('zero-height',4,0)
    coord('negative-width',-2,4)
    coord('negative-height',4,-2)
    coord('empty-list',4,4,[])
    coord('mixed-duplicates',4,4,['type-3-y','type-4','type-3-y','type-2-x'])
    coord('tuple',5,4,['type-3-y','type-2-x'],selector_kind='tuple')
    coord('array',5,4,['type-4','type-2-y'],selector_kind='array')
    coord('object-selector-array',5,4,['type-4','type-2-y'],selector_kind='object')
    coord('iterator',4,4,['type-2-y'],selector_kind='iterator')
    coord('selector-bad-str',4,4,'bad')
    coord('selector-mixed-bad',4,4,['type-2-x','bad'])
    coord('selector-bad-scalar',4,4,17)
    coord('selector-nested',4,4,[['type-2-x']])
    coord('float-width',4.8,3,'type-2-x')
    coord('float-height',3,4.8,'type-2-y')
    coord('bool-width',True,3)
    coord('np-width',4,3,'type-4',dimension_kind='numpy')
    coord('str-width','4',3,'type-4')
    coord('intp-overflow',2**80,2,'type-2-x')
    value('small',4,4)
    value('roi-large',20,18,r=2,c=3,shape=[25,28])
    for family in TYPES:
        value(family,4,3,family,r=2,c=1)
    value('asymmetrical-roi',5,4, ['type-4','type-3-y','type-3-x'],r=1,c=2)
    value('negative-origin-valid-slice',3,3,'type-4',r=-6,c=-6)
    value('mixed-duplicates',5,4,['type-2-x','type-3-y','type-2-x'])
    value('selector-array',5,4,['type-3-x','type-2-y'],selector_kind='array')
    value('selector-tuple',5,4,['type-3-x','type-2-y'],selector_kind='tuple')
    value('iterator',4,4,['type-2-x'],selector_kind='iterator')
    value('selector-bad',4,4,'noop')
    value('empty-list',4,4,[])
    value('empty-point',1,1)
    value('empty-width',0,4)
    value('empty-height',4,0)
    for dtype in ('uint8','uint16','uint32','uint64','int8','int16','int32','int64','float32','float64'):
        value('dtype-'+dtype,4,4,dtype=dtype)
        if dtype[0] in 'ui':
            value('wrap-'+dtype,4,4,dtype=dtype,wide=True)
    for dtype in ('bool','float16','complex64','>i2'):
        value('unsupported-'+dtype,4,4,dtype=dtype)
    for layout in ('F','readonly','negative','step'):
        value('storage-'+layout,4,4,'type-3-x',layout=layout,dtype='float32')
    value('precision-float32',4,4,dtype='float32',cancellation=True)
    value('precision-float64',4,4,dtype='float64',cancellation=True)
    value('signed-zero',4,4,dtype='float32',signed_zero=True)
    value('float-nan',4,4,dtype='float64',nan=True)
    value('subset',4,4,subset='normal',r=1,c=1)
    value('subset-reversed',4,4,subset='reverse',dtype='uint16',wide=True)
    value('subset-repeated',4,4,subset='repeated',dtype='float32')
    value('subset-numeric-coords',4,4,subset='numeric')
    value('subset-extra-rectangle-ignored',4,4,subset='extra')
    value('subset-concrete-selector-string-array',4,4,subset='unicode')
    value('subset-empty',4,4,subset='empty')
    value('subset-inconsistent',4,4,subset='mismatch')
    value('subset-list-coords',4,4,subset='listcoords')
    value('subset-list-selector',4,4,subset='listselector')
    value('subset-selector-none',4,4,subset='none')
    value('subset-unknown-later',4,4,subset='unknown-last')
    value('subset-unknown-earlier',4,4,subset='unknown-first')
    value('subset-all-unknown',4,4,subset='unknown-all')
    value('subset-few-rectangles',4,4,subset='short')
    return out


def selector(spec,np):
    select=spec.get('select')
    kind=spec.get('selector_kind')
    if kind=='tuple': return tuple(select)
    if kind=='array': return np.array(select)
    if kind=='object': return np.array(select,dtype=object)
    if kind=='iterator': return iter(select)
    return select


def materialize(spec,np):
    w,h=spec['width'],spec['height']
    if spec.get('dimension_kind')=='numpy': w,h=np.int32(w),np.int64(h)
    selected=selector(spec,np)
    if spec['which']=='coord':
        return (w,h,selected),{}
    shape=tuple(spec.get('shape',(8,9)))
    dtype=np.dtype(spec.get('dtype','uint64'))
    numbers=np.arange(shape[0]*shape[1],dtype=np.int64).reshape(shape)
    if spec.get('wide'):
        bound=np.iinfo(dtype)
        words=[int(bound.max),int(bound.min),int(bound.max)-1,int(bound.min)+1,1,0,17,31]
        image=np.array([words[int(i)%len(words)] for i in numbers.flat],dtype=dtype).reshape(shape)
    else:
        # Exact integral image of a small pattern, produced independently of
        # either project's public integral-image entry point.
        source=(numbers*11+np.arange(shape[0])[:,None]*3)%17
        signed=source-8 if dtype.kind in 'if' else source
        image=np.cumsum(np.cumsum(signed,axis=0),axis=1).astype(dtype)
    if spec.get('cancellation'):
        image=np.where(numbers%3==0,1e18,(numbers%17)/11).astype(dtype)
    if spec.get('signed_zero'):
        image=np.zeros(shape,dtype=dtype)
        image[::2,::2]=-0.0
    if spec.get('nan'):
        image[2,2]=np.nan
    layout=spec.get('layout')
    if layout=='F': image=np.asfortranarray(image)
    if layout=='negative': image=image[::-1]
    if layout=='step': image=np.repeat(image,2,axis=1)[:,::2]
    if layout=='readonly': image.flags.writeable=False

    coords=None
    kind=spec.get('subset')
    if kind:
        import copy
        values=copy.deepcopy(SUBSET)
        families=list(SUBSET_TYPES)
        if kind=='reverse': values.reverse(); families.reverse()
        if kind=='repeated': values+=[values[0],values[1]]; families+=[families[0],families[1]]
        if kind=='numeric': values=[values[1],values[3]]; families=['type-2-y','type-2-x']
        if kind=='extra': values[1].append([(0,0),(1,1)])
        if kind=='empty': values=[]; families=[]
        if kind=='unknown-last': families[-1]='unknown'
        if kind=='unknown-first': families[0]='unknown'
        if kind=='unknown-all': families=['unknown']*len(families)
        if kind=='short': values[0]=values[0][:1]
        coords=np.empty(len(values),dtype=object)
        for i,row in enumerate(values): coords[i]=row
        if kind=='numeric': coords=np.array(values,dtype=np.int64)
        if kind=='listcoords': coords=values
        if kind=='mismatch': families=families[:-1]
        selected=np.array(families,dtype=None if kind=='unicode' else object)
        if kind=='listselector': selected=families
        if kind=='none': selected=None
    return (image,spec.get('r',0),spec.get('c',0),w,h,selected,coords),{}


def fingerprint(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def summary(array,np):
    result = {'shape':list(array.shape),'dtype':array.dtype.str,'strides':list(array.strides),
              'owned':bool(array.flags.owndata),'writeable':bool(array.flags.writeable)}
    if array.dtype==object:
        values=array.tolist()
        def freeze(value):
            if isinstance(value,(tuple,list)):
                return [type(value).__name__,[freeze(item) for item in value]]
            return [type(value).__name__,value]
        value_tree=[freeze(value) for value in values]
        result['sha256']=hashlib.sha256(json.dumps(value_tree,separators=(',',':')).encode()).hexdigest()
        if len(values)<=12: result['values']=value_tree
    else:
        result['sha256']=fingerprint(array)
        if array.size<=7: result['hex_words']=array.tobytes().hex()
    return result