"""Finite prospective inputs; no imports or implicit reference execution."""
API='segmentation.find_boundaries'
SIGNATURE="(label_img, connectivity=1, mode='thick', background=0)"
MAX_INPUT_ELEMENTS=192
MAX_OUTPUT_ELEMENTS=4096


def cases():
    specs=[]
    def add(name,shape,values,dtype='uint8',mode='thick',connectivity=1,background=0,layout='C',**extra):
        specs.append(dict(id='find_boundaries/'+name,shape=list(shape),values=values,
                          dtype=dtype,mode=mode,connectivity=connectivity,
                          background=background,layout=layout,**extra))
    cross=[0,0,0,0,1,2,0,2,2]
    for mode in ('thick','inner','outer','subpixel'):
        add(mode+'-cross',(3,3),cross,mode=mode)
    add('diagonal-full',(2,2),[0,1,1,0],connectivity=2)
    add('inner-negative',(2,3),[-1,-1,2,0,2,-1],dtype='int8',mode='inner',background=-1)
    add('outer-negative',(2,3),[-1,-1,2,0,2,-1],dtype='int16',mode='outer',background=-1)
    add('outer-high-background',(2,3),[7,7,2,7,2,3],mode='outer',background=7)
    add('outer-max-background',(2,2),[255,1,1,2],mode='outer',background=255)
    add('subpixel-max-collision',(2,2),[255,0,1,255],mode='subpixel')
    add('signed-inrange-rounded-words',(2,2),[2**53,2**53+1,2**53+1,2**53+2],dtype='int64')
    add('unsigned-u32-background',(2,2),[0,2**31+1,2**31+2,2**32-1],dtype='uint32',mode='outer',background=2**31+1)
    add('subpixel-u64-max',(2,2),[2**64-1,0,1,2**64-1],dtype='uint64',mode='subpixel')
    add('bool-thick',(2,3),[0,1,0,1,1,0],dtype='bool')
    add('bool-inner-true',(2,3),[0,1,0,1,1,0],dtype='bool',mode='inner',background=True)
    add('bool-noncanonical',(2,3),[0,1,0,1,1,0],dtype='bool',mode='outer',raw_bool=[0,2,255,1,0,2])
    add('bool-subpixel',(2,3),[0,1,0,1,1,0],dtype='bool',mode='subpixel')
    add('fortran-u16',(2,3),[0,1,1,2,2,0],dtype='uint16',mode='outer',connectivity=2,layout='F')
    add('transposed-3d',(2,2,3),[0,1,1,0,2,2,2,0,0,1,2,0],dtype='int32',mode='inner',connectivity=3,layout='transpose',axes=[2,0,1])
    add('reverse-readonly-3d',(2,2,3),[0,1,1,0,2,2,2,0,0,1,2,0],dtype='uint32',connectivity=2,layout='reverse',readonly=True)
    add('stepped-inner',(2,3),[0,1,1,2,2,0],dtype='int16',mode='inner',layout='step')
    add('readonly-1d-subpixel',(4,),[0,1,1,2],dtype='int32',mode='subpixel',readonly=True)
    add('singleton-axes',(1,3,1),[0,1,2],mode='outer',connectivity=3)
    add('rank4',(2,1,2,2),[0,1,0,1,2,2,0,2],connectivity=3)
    add('connectivity-zero',(2,2),[0,1,1,0],connectivity=0)
    add('connectivity-beyond-rank',(2,2),[0,1,1,0],mode='outer',connectivity=7)
    add('unknown-mode-is-thick',(2,2),[0,1,1,0],mode='unrecognised-mode')
    add('empty-subpixel-error',(0,2),[],mode='subpixel')
    add('list-dtype-error',(2,2),[0,1,1,0],layout='list')
    add('connectivity-none-error',(2,2),[0,1,1,0],connectivity=None)
    add('tagged-u16',(2,2),[0,1,1,2],dtype='uint16',metadata={'units':'labels','source':'frozen-reference'})
    add('background-out-of-range',(2,2),[0,1,1,2],mode='outer',background=2**80)
    return specs


def logical_shape(spec):
    shape=spec['shape']
    return [shape[d]for d in spec['axes']]if spec['layout']=='transpose'else shape


def output_bound(spec):
    shape=logical_shape(spec)
    return [2*s-1 for s in shape]if spec['mode']=='subpixel'else shape


def fixture(spec,np):
    if spec['layout']=='list':
        return (list(spec['values']),),dict(connectivity=spec['connectivity'],mode=spec['mode'],background=spec['background']),{}
    dtype=np.dtype(spec['dtype'],metadata=spec['metadata'])if 'metadata'in spec else np.dtype(spec['dtype'])
    labels=np.array(spec['values'],dtype=dtype).reshape(spec['shape'])
    buffers={'labels_owner':labels}
    if 'raw_bool'in spec:labels.view(np.uint8).flat[:]=spec['raw_bool']
    if spec['layout']=='F':labels=np.array(labels,order='F',copy=True);buffers['labels_F_owner']=labels
    elif spec['layout']=='transpose':labels=labels.transpose(spec['axes'])
    elif spec['layout']=='reverse':labels=labels[...,::-1]
    elif spec['layout']=='step':
        shape=list(labels.shape);shape[-1]*=2
        owner=np.zeros(shape,dtype=dtype);owner[...,::2]=labels
        buffers['step_owner']=owner;labels=owner[...,::2]
    if spec.get('readonly'):labels.flags.writeable=False
    buffers['labels']=labels
    return (labels,),dict(connectivity=spec['connectivity'],mode=spec['mode'],background=spec['background']),buffers
