"""Pinned rank intensity contracts; reference and pending native facts differ.

Buffers use previously qualified allocation-safe extents. Overlap, undersized
mask/output and zero-depth Cython outputs are excluded, never guessed errors.
"""
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import warnings
import numpy as np
import pytest
import skimage
from skimage.filters import rank as reference
import skimagexx as xx
from skimagexx.coverage import native_functions

spec=importlib.util.spec_from_file_location('rank_intensity_base',Path(__file__).with_name('test_rank_neighborhood_contracts.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
NAMES=('autolevel','equalize','enhance_contrast','threshold')
MATRIX=Path(__file__).parents[1]/'coverage/contracts/rank-intensity.json'
PATTERNS=('flat-zero','flat-seven','flat-u16','tie-midpoint','cdf-fraction','u8-extremes',
          'u16-extremes','u16-low-bins','center-below-window','center-above-window',
          'zero-window','disconnected','masked-center')
CASES=[(name+'/'+key.split('/',1)[1],name,dict(parameters))
       for name in NAMES for key,_,parameters in base.CASES
       if key.startswith('minimum/') and parameters.get('out') not in ('image','view','overlap')
       and parameters.get('special')!='alias-converted']
for name in NAMES:
    for ndim in (2,3):
        for pattern in PATTERNS:
            CASES.append((f'{name}/{ndim}d/intensity={pattern}',name,dict(ndim=ndim,intensity=pattern)))
        for output in ('u1','u2','f4','f8','tagged'):
            CASES.append((f'{name}/{ndim}d/cdf-fraction/out={output}',name,
                          dict(ndim=ndim,intensity='cdf-fraction',intensity_out=output)))
        CASES.append((f'{name}/{ndim}d/missing-footprint',name,dict(ndim=ndim,omitted=True)))


def profile():return 'minimum' if np.__version__.startswith('1.') else 'windows' if os.name=='nt' else 'modern'


def make_case(parameters):
    if 'intensity' not in parameters and not parameters.get('omitted'):return base.make_case(parameters)
    ndim=parameters['ndim'];shape=(5,7) if ndim==2 else (3,4,5);pattern=parameters.get('intensity')
    values=np.asarray([1,5,9],np.uint8)
    if pattern=='flat-zero':values=np.asarray([0],np.uint8)
    elif pattern=='flat-seven':values=np.asarray([7],np.uint8)
    elif pattern=='flat-u16':values=np.asarray([65535],np.uint16)
    elif pattern=='cdf-fraction':values=np.asarray([1,2,2,3,3,3,3],np.uint8)
    elif pattern=='u8-extremes':values=np.asarray([0,255,1,254,127,128],np.uint8)
    elif pattern=='u16-extremes':values=np.asarray([0,65535,1,65534,32767,32768],np.uint16)
    elif pattern=='u16-low-bins':values=np.asarray([0,1,2],np.uint16)
    image=np.resize(values,int(np.prod(shape))).reshape(shape)
    footprint=np.ones((2,4) if ndim==2 else (2,3,4),np.uint8);options={}
    if pattern in ('center-below-window','center-above-window','masked-center'):
        image=np.resize(np.asarray([4,7,10],np.uint8),int(np.prod(shape))).reshape(shape)
        position=tuple(n//2 for n in shape);image[position]=0 if pattern=='center-below-window' else 255
        mask=np.ones(shape,np.uint8);mask[position]=0;options['mask']=mask
    elif pattern=='zero-window':options['mask']=np.zeros(shape,np.uint8)
    elif pattern=='disconnected':footprint.flat[1:-1]=0
    output=parameters.get('intensity_out')
    if output:
        dtype=np.dtype('u1',metadata={'rank':'intensity-out'}) if output=='tagged' else np.dtype(output)
        options['out']=np.full(shape,99,dtype=dtype)
    tracked={'image':image,**({} if parameters.get('omitted') else {'footprint':footprint}),**options}
    return ((image,) if parameters.get('omitted') else (image,footprint)),options,tracked


def describe(value,inputs=()):
    result=base.describe(value,inputs)
    if isinstance(value,np.ndarray) and not value.dtype.hasobject:
        result['raw_bytes_sha256']=hashlib.sha256(value.tobytes()).hexdigest()
        if value.size<=256 and value.dtype.kind in 'buif' and np.isfinite(value).all():result['values']=value.tolist()
    return result


def observe(function,parameters):
    args,options,tracked=make_case(parameters);inputs=[v for v in tracked.values() if isinstance(v,np.ndarray)]
    before={key:describe(value) for key,value in tracked.items()}
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:value,error=describe(function(*args,**options),inputs),None
        except Exception as caught:value,error=None,dict(type=type(caught).__name__,message=str(caught))
    return dict(result=value,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],
                before=before,after={key:describe(value) for key,value in tracked.items()})


def snapshot():
    assert skimage.__version__=='0.26.0'
    return {key:observe(getattr(reference,name),parameters) for key,name,parameters in CASES}


@pytest.fixture(scope='module')
def frozen():return json.loads(MATRIX.read_text(encoding='utf-8'))


def test_frozen_reference_integrity_and_pending_evidence(frozen):
    assert skimage.__version__==frozen['pinned_skimage']=='0.26.0'
    assert len({key for key,_,_ in CASES})==len(CASES)==1012
    assert frozen['native_checks']==0 and frozen['native_pending']==len(CASES)
    helper=Path(__file__).with_name('test_rank_neighborhood_contracts.py')
    assert hashlib.sha256(helper.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==frozen['fixture_helpers']['sha256_posix_lf']
    assert all(parameters.get('out') not in ('image','view','overlap') and parameters.get('special')!='alias-converted' for _,_,parameters in CASES)
    for snapshots in frozen['snapshots'].values():assert set(snapshots)=={key for key,_,_ in CASES}
    for digest,value in frozen['observations'].items():assert hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()==digest


@pytest.mark.parametrize('name',NAMES)
def test_source_signature_and_public_generic_identity(name,frozen):
    generic=importlib.import_module('skimage.filters.rank.generic');function=getattr(reference,name);facts=frozen['functions'][name]
    assert name in reference.__all__ and name in generic.__all__ and getattr(generic,name) is function
    assert str(inspect.signature(function))==facts['signature']
    assert hashlib.sha256(inspect.getsource(function).encode()).hexdigest()==facts['python_source_sha256']
    assert facts['aliases']==['filters.rank.generic.'+name]
    assert facts['native_checks']==0 and facts['native_pending']==sum(n==name for _,n,_ in CASES)


@pytest.mark.parametrize('namespace',['filters.rank','filters.rank.generic'])
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_original_frozen_contract(case,namespace,frozen):
    key,name,parameters=case;function=getattr(importlib.import_module('skimage.'+namespace),name)
    assert observe(function,parameters)==frozen['observations'][frozen['snapshots'][profile()][key]],key


@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('namespace',['filters.rank','filters.rank.generic'])
@pytest.mark.parametrize('case',CASES,ids=[c[0] for c in CASES])
def test_strict_pending_is_separate_from_reference(case,namespace,workers):
    _,name,parameters=case;previous=xx.set_fallback_enabled(False);threads=xx.get_num_threads();xx.set_num_threads(workers)
    try:
        function=getattr(importlib.import_module('skimagexx.'+namespace),name)
        if 'filters.rank.'+name in native_functions():pytest.skip('public native transition requires separate explicit implementation/evidence gate')
        actual=observe(function,parameters)
        assert actual['result'] is None and actual['error']['type']=='NotImplementedError'
        assert 'fallback is disabled' in actual['error']['message'] and not actual['warnings']
        assert actual['before']==actual['after']
    finally:xx.set_num_threads(threads);xx.set_fallback_enabled(previous)


def intensity_oracle(name,image,footprint,mask,center,dtype):
    out=np.empty(image.shape,dtype);bins=256 if image.dtype==np.uint8 else max(3,int(image.max()))+1
    for position in np.ndindex(image.shape):
        values=[]
        for index in np.ndindex(footprint.shape):
            point=tuple(p+i-c for p,i,c in zip(position,index,center))
            if footprint[index] and all(0<=i<n for i,n in zip(point,image.shape)) and mask[point]:values.append(int(image[point]))
        value=0;g=int(image[position])
        if values:
            low,high=min(values),max(values)
            if name=='autolevel' and high>low:
                numerator=(bins-1)*(g-low);value=(abs(numerator)//(high-low))*(-1 if numerator<0 else 1)
            elif name=='equalize':value=((bins-1)*sum(v<=g for v in values))/float(len(values))
            elif name=='enhance_contrast':value=high if high-g<g-low else low
            elif name=='threshold':value=int(g>sum(values)/float(len(values)))
        if np.dtype(dtype).kind=='u':value=int(value)%(1<<(8*np.dtype(dtype).itemsize))
        out[position]=value
    return out


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
@pytest.mark.parametrize('output',[None,np.float32,np.float64])
@pytest.mark.parametrize('pattern',['ties','asymmetric','zero-window','center-below','center-above'])
def test_independent_histogram_intensity_definition(name,ndim,dtype,output,pattern):
    shape=(4,5) if ndim==2 else (3,4,5);image=np.resize(np.array([1,5,9,2,2,7,7],dtype),int(np.prod(shape))).reshape(shape)
    footprint=np.ones((2,4) if ndim==2 else (2,3,4),np.uint8);mask=np.ones(shape,np.uint8)
    if pattern=='asymmetric':footprint.flat[::3]=0;mask.flat[::4]=0
    elif pattern=='zero-window':mask.fill(0)
    elif pattern in ('center-below','center-above'):
        position=tuple(n//2 for n in shape);image[position]=0 if pattern=='center-below' else np.iinfo(dtype).max;mask[position]=0
    sx,sy,sz=(-1,0,0) if ndim==2 else (-1,1,0)
    center=(footprint.shape[0]//2+sy,footprint.shape[1]//2+sx) if ndim==2 else tuple(n//2+s for n,s in zip(footprint.shape,(sx,sy,sz)))
    options=dict(mask=mask,shift_x=sx,shift_y=sy,shift_z=sz);out_dtype=dtype if output is None else output
    if output is not None:options['out']=np.full(shape,99,output)
    expected=intensity_oracle(name,image,footprint,mask,center,out_dtype)
    with warnings.catch_warnings():warnings.simplefilter('ignore');actual=getattr(reference,name)(image,footprint,**options)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('ndim',[2,3])
@pytest.mark.parametrize('output',[np.float32,np.float64])
@pytest.mark.parametrize('direction',['below','above'])
def test_autolevel_signed_wide_quotient_is_observable_in_float_output(ndim,output,direction):
    shape=(5,7) if ndim==2 else (3,4,5)
    levels=np.asarray([1,2] if direction=='above' else [65533,65534],np.uint16)
    image=np.resize(levels,int(np.prod(shape))).reshape(shape);position=tuple(n//2 for n in shape)
    image[position]=65535 if direction=='above' else 0
    footprint=np.ones((3,)*ndim,np.uint8);mask=np.ones(shape,np.uint8);mask[position]=0
    expected=intensity_oracle('autolevel',image,footprint,mask,(1,)*ndim,output)
    assert abs(float(expected[position]))>2**31
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');actual=reference.autolevel(image,footprint,mask=mask,out=np.empty(shape,output))
    np.testing.assert_array_equal(actual,expected)
