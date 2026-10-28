"""Prospective bounded montage fixtures; mean/rescale remain explicit providers."""
API = 'util.montage'
STAGES = ('fixture', 'encode', 'observe')
UNASSESSED = ('allocation failures and huge grids', 'forged storage', 'concurrent mutation', 'padded structured or longdouble bytes')


def cases():
    names = ('gray-square', 'gray-incomplete', 'gray-one', 'gray-strided', 'gray-fortran', 'gray-readonly',
             'grid-row', 'grid-column', 'grid-extra', 'grid-too-small', 'grid-zero', 'grid-negative', 'grid-fractional', 'grid-wrong-length', 'grid-array',
             'padding-one', 'padding-numpy-int', 'padding-negative', 'padding-float',
             'color-last-mean', 'color-middle-mean', 'color-first-mean', 'channel-axis-zero', 'channel-tuple-one', 'channel-tuple-many', 'channel-out-of-range',
             'fill-color-vector', 'fill-color-scalar', 'fill-color-short', 'fill-array', 'fill-nan', 'fill-inf', 'fill-bad-string',
             'rescale-float', 'rescale-uint8', 'rescale-readonly', 'rescale-before-fill-error', 'grid-error-before-rescale',
             'dtype-bool', 'dtype-int8', 'dtype-uint16', 'dtype-int64', 'dtype-float16', 'dtype-complex64', 'dtype-object', 'dtype-bytes', 'dtype-unicode', 'dtype-structured', 'dtype-datetime', 'big-endian', 'dtype-metadata',
             'array-protocol', 'grid-protocol', 'padding-protocol', 'fill-protocol', 'rank-error-before-grid', 'zero-images', 'zero-height')
    return [dict(id=name, api=API, kind=name, policy={}) for name in names]


class ArrayInput:
    def __init__(self, value, events): self.value, self.events = value, events
    def __array__(self, dtype=None, copy=None):
        self.events.append(['input.__array__', None if dtype is None else str(dtype), copy])
        return self.value if dtype is None else self.value.astype(dtype)


class IntValue:
    def __init__(self, value, label, events): self.value, self.label, self.events = value, label, events
    def __int__(self): self.events.append(self.label+'.__int__'); return self.value


class GridValue:
    def __init__(self, events): self.events = events
    def __bool__(self): self.events.append('grid.__bool__'); return True
    def __iter__(self):
        self.events.append('grid.__iter__')
        return iter((IntValue(2, 'rows', self.events), IntValue(2, 'columns', self.events)))


class PaddingValue:
    def __init__(self, value, events): self.value, self.events = value, events
    def __index__(self): self.events.append('padding.__index__'); return self.value
    def __add__(self, other): self.events.append('padding.__add__'); return self.value+other
    def __radd__(self, other): self.events.append('padding.__radd__'); return other+self.value
    def __mul__(self, other): self.events.append('padding.__mul__'); return self.value*other
    def __rmul__(self, other): self.events.append('padding.__rmul__'); return other*self.value


class FillValue:
    def __init__(self, value, events): self.value, self.events = value, events
    def __eq__(self, other): self.events.append(['fill.__eq__', other]); return False
    def __array__(self, dtype=None, copy=None):
        self.events.append(['fill.__array__', None if dtype is None else str(dtype), copy])
        return self.value if dtype is None else self.value.astype(dtype)


def fixture(spec, np):
    kind=spec['kind'];events=[];arr=np.arange(24,dtype=np.uint8).reshape(4,2,3);options={};buffers={}
    if kind=='gray-incomplete': arr=arr[:3];options['fill']=17
    elif kind=='gray-one': arr=arr[:1]
    elif kind=='gray-strided': arr=np.arange(96,dtype=np.float32).reshape(4,4,6)[:,::2,::-2]
    elif kind=='gray-fortran': arr=np.asfortranarray(arr.astype(np.float64))
    elif kind=='gray-readonly': arr.flags.writeable=False;options['fill']=3
    grids={'grid-row':(1,4),'grid-column':(4,1),'grid-extra':(3,2),'grid-too-small':(1,2),'grid-zero':(0,2),'grid-negative':(-1,2),'grid-fractional':(2.9,2.1),'grid-wrong-length':(2,)}
    if kind in grids: options['grid_shape']=grids[kind];options['fill']=5
    elif kind=='grid-array': options['grid_shape']=np.array([2,2]);options['fill']=5
    pads={'padding-one':1,'padding-numpy-int':np.int64(2),'padding-negative':-1,'padding-float':1.5}
    if kind in pads: options['padding_width']=pads[kind];options['fill']=9
    colors=('color-last-mean','color-middle-mean','color-first-mean','channel-axis-zero','channel-tuple-one','channel-tuple-many','channel-out-of-range','fill-color-vector','fill-color-scalar','fill-color-short','fill-array','fill-protocol')
    if kind in colors:
        arr=np.arange(72,dtype=np.uint8).reshape(4,2,3,3);options['channel_axis']=-1
        if kind=='color-middle-mean': arr=np.moveaxis(arr,-1,1);options['channel_axis']=1
        elif kind=='color-first-mean': arr=np.moveaxis(arr,-1,0);options['channel_axis']=0
        elif kind=='channel-axis-zero': options['channel_axis']=0
        elif kind=='channel-tuple-one': options['channel_axis']=(-1,)
        elif kind=='channel-tuple-many': options['channel_axis']=(1,3)
        elif kind=='channel-out-of-range': options['channel_axis']=6
        elif kind=='fill-color-vector': options.update(fill=[1,2,3],padding_width=1)
        elif kind=='fill-color-scalar': options['fill']=4
        elif kind=='fill-color-short': options['fill']=[1,2]
        elif kind=='fill-array': options['fill']=np.array([1.,2.,3.])
        elif kind=='fill-protocol': options['fill']=FillValue(np.array([1.,2.,3.]),events)
    if kind=='fill-nan': options['fill']=float('nan')
    elif kind=='fill-inf': options['fill']=float('inf')
    elif kind=='fill-bad-string': options['fill']='invalid-fill'
    if kind.startswith('rescale') or kind=='grid-error-before-rescale':
        if kind!='rescale-uint8': arr=arr.astype(np.float32)*3-7
        options['rescale_intensity']=True
        if kind=='rescale-readonly': arr.flags.writeable=False
        elif kind=='rescale-before-fill-error': options['fill']='invalid-fill'
        elif kind=='grid-error-before-rescale': options['grid_shape']=(2,)
    dtypes={'dtype-bool':'bool','dtype-int8':'int8','dtype-uint16':'uint16','dtype-int64':'int64','dtype-float16':'float16','dtype-complex64':'complex64','dtype-object':object,'dtype-bytes':'S4','dtype-unicode':'U4','dtype-structured':[('a','u1'),('b','u2')],'dtype-datetime':'datetime64[D]','big-endian':'>u2','dtype-metadata':np.dtype('uint16',metadata={'contract':'montage'})}
    if kind in dtypes:
        arr=arr.astype(dtypes[kind]);options['fill']=0
    if kind=='array-protocol': buffers['input_owner']=arr;arr=ArrayInput(arr,events);options['fill']=0
    elif kind=='grid-protocol': options.update(grid_shape=GridValue(events),fill=0)
    elif kind=='padding-protocol': options.update(padding_width=PaddingValue(1,events),fill=0)
    elif kind=='rank-error-before-grid': arr=arr[0];options['grid_shape']=GridValue(events)
    elif kind=='zero-images': arr=arr[:0];options['fill']=0
    elif kind=='zero-height': arr=arr[:3,:0];options['fill']=0
    if isinstance(arr,np.ndarray): buffers['input']=arr
    return (arr,),options,events,buffers
