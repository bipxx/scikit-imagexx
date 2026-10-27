"""Exact source-ordered profile sampling contracts, frozen before outputs.

Both SciPy1.15.3/1.18.1 loop over 2D taps(00,01,10,11), multiply each
coefficient by axis0 then axis1 weights, and add sequentially in double.
Order1 computes weight1=1-weight0, NOT direct delta. Integer rounding/clipping
precedes reduction. Require exact pre-callback sampled words/metadata and
public words/metadata/events: arbitrary callbacks can expose even one ULP.
No tolerance or fitted numerical budget is provided. NumPy coordinates,
flip/channel metadata and reducers remain explicit source providers.
Empty spatial axes, nonfinite coordinates entering C, huge coordinate/work
extents and forged storage are excluded from ALL original calls.
"""
import warnings
API='measure.profile_line'
STAGES=('profile_sample',)
UNASSESSED=('zero spatial axes: original-unsafe C sampling',
 'nonfinite coordinates entering C','huge allocation/index extents',
 'forged storage','arbitrary image/coordinate protocols')


class ReducerProbe:
    def __init__(self,mode,events,np):self.mode,self.events,self.np=mode,events,np
    def __repr__(self):return 'ReducerProbe('+self.mode+')'
    def __call__(self,values,axis=None):
        self.events.append(dict(event='reducer.call',axis=axis,dtype=values.dtype.str,
          shape=list(values.shape),strides=list(values.strides),raw=values.tobytes().hex()))
        if self.mode=='raise':raise RuntimeError('profile reducer probe')
        if self.mode=='warn':warnings.warn('profile reducer probe',UserWarning)
        if axis is not None and self.mode in('no-axis','body-typeerror'):
            if self.mode=='body-typeerror':
                values.flat[0]=17
                self.events.append(dict(event='reducer.mutated',raw=values.tobytes().hex()))
            raise TypeError('profile reducer probe TypeError')
        return self.np.sum(values,axis=axis)


def cases():
    rows=[]
    def add(id,kind='geometry',dtype='float64',layout='C',route='native',error=False,policy=None):
        rows.append(dict(id=id,kind=kind,dtype=dtype,layout=layout,route=route,error=error,policy=policy or {}))
    for id in('horizontal','diagonal','reverse-vertical','subpixel','samepoint-width0',
              'channels3-width3','channels1-width2','large-reflect'):
        add(id,dtype='uint8'if id in('horizontal','channels1-width2')else'float32'if id in('diagonal','channels3-width3')else'float64')
    for order in(2,3,4,5):add('order'+str(order),route='provider')
    add('dtype-bool',dtype='bool');add('bool-order1',dtype='bool',route='error',error=True)
    add('dtype-int16',dtype='int16',route='provider');add('dtype-half',dtype='float16',route='error',error=True)
    for layout in('F','reverse','endian','unaligned','readonly','metadata'):
        add('layout-'+layout,kind='layout',dtype='float32'if layout in('F','unaligned','metadata')else'float64',layout=layout)
    for mode in('reflect','mirror','nearest','constant','wrap','grid-constant','grid-wrap'):
        add('mode-'+mode,kind='mode')
    add('nonfinite-image',route='provider',policy={'invalid':'raise'})
    for reducer in('none','sum','max','sqrt','no-axis','body-typeerror','warn','raise'):
        add('reduce-'+reducer,kind='reducer',error=reducer=='raise')
    for id in('list-image','order6','invalid-mode','negative-width','float-width','short-src','nan-endpoint','inf-endpoint'):
        add('error-'+id,kind='error',route='error',error=True)
    assert len(rows)==46 and len({x['id']for x in rows})==46
    return rows


def fixture(spec,np):
    dtype=spec['dtype'];events=[]
    image=(np.arange(99).reshape(9,11)%23+1).astype(dtype)
    src,dst=(1.,1.),(7.,8.);width=3;options={}
    id=spec['id']
    if dtype=='bool':image=(np.arange(99).reshape(9,11)%3==0)
    if id=='horizontal':src,dst=(2.,1.),(2.,8.);width=1
    elif id=='diagonal':src,dst=(.25,.5),(7.25,9.5);width=1
    elif id=='reverse-vertical':src,dst=(7.,2.),(1.,2.);width=1
    elif id=='subpixel':src,dst=(.125,.375),(6.875,9.125);width=2
    elif id=='samepoint-width0':src=dst=(3.,4.);width=0
    elif id.startswith('channels'):
        channels=3 if id=='channels3-width3'else 1
        image=np.stack([image+i for i in range(channels)],axis=-1).astype(dtype)
        width=3 if channels==3 else 2
    elif id=='large-reflect':src,dst=(0.,0.),(8191.,0.);width=33
    elif id.startswith('order')and spec['kind']!='error':options['order']=int(id[5:])
    elif id=='bool-order1':options['order']=1
    elif id.startswith('mode-'):
        src,dst=(-2.25,-1.5),(10.5,12.75);options.update(mode=id[5:],cval=-3.25)
    elif id=='nonfinite-image':
        image[1,1]=np.nan;image[1,2]=np.inf
        options.update(reduce_func=None)
    if spec['kind']=='reducer':
        mode=id[7:]
        options['reduce_func']={'none':None,'sum':np.sum,'max':np.max,'sqrt':np.sqrt}.get(mode)
        if mode in('no-axis','body-typeerror','warn','raise'):options['reduce_func']=ReducerProbe(mode,events,np)
    layout=spec['layout']
    if layout=='F':image=np.asfortranarray(image)
    elif layout=='reverse':image=image[::-1,::-1]
    elif layout=='endian':image=image.astype(image.dtype.newbyteorder('>'))
    elif layout=='unaligned':
        owner=bytearray(image.nbytes+1);view=np.ndarray(image.shape,dtype=image.dtype,buffer=owner,offset=1);view[...]=image;image=view
    elif layout=='readonly':image.flags.writeable=False
    elif layout=='metadata':image=image.astype(np.dtype(image.dtype,metadata={'profile':'frozen'}))
    if spec['kind']=='error':
        error=id[6:]
        if error=='list-image':image=image.tolist()
        elif error=='order6':options['order']=6
        elif error=='invalid-mode':options['mode']='profile-invalid-mode'
        elif error=='negative-width':width=-1
        elif error=='float-width':width=1.5
        elif error=='short-src':src=(1.,)
        elif error=='nan-endpoint':dst=(np.nan,3.)
        elif error=='inf-endpoint':dst=(np.inf,3.)
    options.update(src=src,dst=dst,linewidth=width)
    return image,options,events
