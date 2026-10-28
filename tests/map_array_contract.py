"""Prospective map_array inputs; no originals run until this file is frozen."""
API='util.map_array'
UNASSESSED=('output_vals shorter than input_vals: unchecked Cython read',
            'forged or unaligned storage', 'partially overlapping input/output: prange dependencies',
            'allocation failure and huge dimensions', 'concurrent mutation')


def cases():
    names=('signed8','signed16','signed32','signed64','unsigned8','unsigned16','unsigned32','unsigned64',
           'missing','duplicates-unsorted','cast-wrap','output-int16','output-uint64',
           'output-float32','output-float64','output-complex64','output-complex128',
           'output-bool-error','output-half-error','input-float-error','input-bool-error',
           'scalar','empty-input','empty-mapping','input-fortran','input-reverse','input-step',
           'mapping-step','out-contiguous','out-strided1d','out-strided2d','out-fortran-error',
           'out-shape-error','out-readonly-error','input-readonly-error','keys-readonly-error',
           'values-readonly-error','keys-readonly-cast','values-cast-to-out','input-endian-error',
           'output-endian-error','out-is-input','out-is-values','protocol-order',
           'complex-to-real-warning','object-cast-protocol','cast-invalid-warning',
           'cast-invalid-raise','object-cast-before-readonly-error','readonly-fortran-copy')
    rows=[dict(id=name,api=API,kind=name,policy={'invalid':'raise'} if name=='cast-invalid-raise' else {},
               prospective='pending; implementation admission decided separately') for name in names]
    assert len(rows)==50 and len({r['id'] for r in rows})==50
    return rows


class FloatBox:
    def __init__(self,value,events):self.value,self.events=value,events
    def __float__(self):
        self.events.append(dict(event='value.__float__',value=self.value))
        return self.value
    def __repr__(self):return 'FloatBox('+repr(self.value)+')'


def fixture(spec,np):
    kind=spec['kind'];events=[];buffers={}
    dtype={'signed8':'int8','signed16':'int16','signed32':'int32','signed64':'int64',
           'unsigned8':'uint8','unsigned16':'uint16','unsigned32':'uint32','unsigned64':'uint64'}.get(kind,'uint8')
    image=np.array([[0,1,2,3],[4,0,1,2],[3,4,5,0]],dtype=dtype)
    keys=np.array([3,1,4,0],dtype=dtype);values=np.array([31,11,41,7],np.uint16);out=None
    if kind.startswith('signed'):
        image[2]=[-1,-2,5,0];keys=np.array([3,1,-1,-2],dtype=dtype)
    elif kind=='unsigned64':
        image=np.array([[0,1],[2**63+3,2**64-1]],np.uint64)
        keys=np.array([2**64-1,2**63+3,1,0],np.uint64)
    elif kind=='duplicates-unsorted':keys=np.array([3,1,3,0],np.uint8)
    elif kind=='cast-wrap':keys=np.array([1,257,3,0],np.int16)
    elif kind.startswith('output-') and kind not in ('output-endian-error',):
        target={'output-int16':'int16','output-uint64':'uint64','output-float32':'float32',
                'output-float64':'float64','output-complex64':'complex64','output-complex128':'complex128',
                'output-bool-error':'bool','output-half-error':'float16'}[kind]
        values=np.array([31,11,41,7],dtype=target)
        if kind=='output-uint64':values=np.array([2**64-2,2**63+7,41,0],np.uint64)
        if kind=='output-float32':values=np.array([0.,-0.,np.inf,np.nan],np.float32)
        if kind.startswith('output-complex'):values=np.array([31+.25j,11-0.j,41+2.j,7-3.j],dtype=target)
    elif kind=='input-float-error':image=image.astype(np.float32)
    elif kind=='input-bool-error':image=image.astype(bool)
    elif kind=='scalar':image=np.array(3,np.uint8)
    elif kind=='empty-input':image=np.empty((0,4),np.uint8)
    elif kind=='empty-mapping':keys=np.empty(0,np.uint8);values=np.empty(0,np.uint16)
    elif kind in ('input-fortran','readonly-fortran-copy'):
        image=np.asfortranarray(image)
        if kind=='readonly-fortran-copy':image.flags.writeable=False
    elif kind=='input-reverse':image=image[:,::-1]
    elif kind=='input-step':
        storage=np.repeat(image,2,axis=1);image=storage[:,::2];buffers['input_storage']=storage
    elif kind=='mapping-step':
        key_storage=np.repeat(keys,2);value_storage=np.repeat(values,2)
        keys=key_storage[::2];values=value_storage[::2];buffers.update(key_storage=key_storage,value_storage=value_storage)
    elif kind=='out-contiguous':out=np.full(image.shape,99,np.int16)
    elif kind=='out-strided1d':
        image=image.reshape(-1);storage=np.full(image.size*2,99,np.uint16)
        out=storage[::2];buffers['out_storage']=storage
    elif kind=='out-strided2d':
        storage=np.full((3,8),99,np.uint16);out=storage[:,::2];buffers['out_storage']=storage
    elif kind=='out-fortran-error':out=np.full(image.shape,99,np.uint16,order='F')
    elif kind=='out-shape-error':out=np.full((2,6),99,np.uint16)
    elif kind=='out-readonly-error':out=np.full(image.shape,99,np.uint16);out.flags.writeable=False
    elif kind=='input-readonly-error':image.flags.writeable=False
    elif kind=='keys-readonly-error':keys.flags.writeable=False
    elif kind=='values-readonly-error':values.flags.writeable=False
    elif kind=='keys-readonly-cast':keys=keys.astype(np.uint16);keys.flags.writeable=False
    elif kind=='values-cast-to-out':values=np.array([31.75,11.25,41.5,7.],np.float64);out=np.full(image.shape,99,np.uint8)
    elif kind=='input-endian-error':image=image.astype('>u2');keys=keys.astype('>u2')
    elif kind=='output-endian-error':values=values.astype('>f8')
    elif kind=='out-is-input':out=image
    elif kind=='out-is-values':
        image=np.array([3,2,1,0],np.uint8);keys=np.array([0,1,2,3],np.uint8)
        values=np.array([11,22,33,44],np.uint16);out=values
    elif kind=='complex-to-real-warning':
        values=np.array([31+.25j,11-2.j,41+3.j,7-4.j],np.complex128);out=np.full(image.shape,99.,np.float64)
    elif kind in ('object-cast-protocol','object-cast-before-readonly-error'):
        values=np.array([FloatBox(x,events) for x in (31.25,11.5,41.75,7.)],object)
        out=np.full(image.shape,99.,np.float64)
        if kind=='object-cast-before-readonly-error':image.flags.writeable=False
    elif kind in ('cast-invalid-warning','cast-invalid-raise'):
        values=np.array([np.nan,11.,41.,7.],np.float64);out=np.full(image.shape,99,np.int16)
    elif kind=='protocol-order':
        class TraceArray(np.ndarray):
            def __array_finalize__(self,source):
                self.events=getattr(source,'events',events);self.role=getattr(source,'role','array')
            def reshape(self,*args,**kwargs):
                self.events.append(dict(event=self.role+'.reshape',shape=list(args)))
                return np.ndarray.reshape(self,*args,**kwargs)
            def astype(self,dtype,*args,**kwargs):
                self.events.append(dict(event=self.role+'.astype',dtype=np.dtype(dtype).str,copy=kwargs.get('copy',True)))
                return np.ndarray.astype(self,dtype,*args,**kwargs)
            def view(self,*args,**kwargs):
                self.events.append(dict(event=self.role+'.view'))
                return np.ndarray.view(self,*args,**kwargs)
            def __setattr__(self,name,value):
                if name=='shape' and hasattr(self,'events'):self.events.append(dict(event=self.role+'.shape',shape=list(value)))
                np.ndarray.__setattr__(self,name,value)
        image=image.view(TraceArray);image.role='input'
        keys=keys.view(TraceArray);keys.role='keys'
        values=values.view(TraceArray);values.role='values'
        out=np.full(image.shape,99,np.uint16).view(TraceArray);out.role='out'
    # The original Cython body indexes outval by inval length with unchecked
    # bounds. Only bounded, aligned real NumPy buffers enter these cases.
    assert values.size>=keys.size
    assert all(a.flags.aligned for a in (image,keys,values) if isinstance(a,np.ndarray))
    if out is not None:assert out.flags.aligned
    return (image,keys,values),dict(out=out) if out is not None else {},events,buffers
