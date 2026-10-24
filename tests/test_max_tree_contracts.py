"""Pinned0.26 max-tree reference contracts: six upstream-only APIs, native0.

Unchecked Cython calls require a source-derived certificate. Unsafe intake and
index protocols stop at an explicit entry observer and are never numeric proof.
"""
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
import scipy
import skimage
import skimagexx as xx
from skimagexx.coverage import native_functions

ROOT = Path(__file__).parents[1]
MATRIX = ROOT / 'coverage/contracts/max-tree.json'
original = importlib.import_module('skimage.morphology.max_tree')
NAMES = ('max_tree', 'max_tree_local_maxima', 'area_opening', 'area_closing',
         'diameter_opening', 'diameter_closing')
ENTRIES = ('_max_tree', '_compute_area', '_compute_extension', '_direct_filter', '_max_tree_local_maxima')
DTYPES = ('u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8', 'f4', 'f8')
SOURCE_SHA = 'ed9842dc253e2f8fc64eae3277f9b02cf582d71c3bfae4ff9468b9c12f3a8674'


def profile():
    return 'windows' if os.name == 'nt' else 'minimum' if np.__version__.startswith('1.') else 'modern'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def authenticate():
    assert skimage.__version__ == '0.26.0'
    data = inspect.getsource(original).replace('\r\n', '\n').encode()
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA


def cases():
    rows = []
    for name in NAMES:
        def add(suffix, **options):
            row = dict(id=name + '/' + suffix, function=name, mode='numeric', dtype='u1',
                       shape=[4, 5], pattern='plateau', layout='C', connectivity=1, threshold=3)
            row.update(options); rows.append(row)
        for dtype in DTYPES:
            add('dtype/' + dtype, dtype=dtype, pattern='limits' if np.dtype(dtype).kind in 'iu' else 'signed-zero')
        for shape in ([4, 5], [4, 4, 5]):
            for pattern in ('flat', 'plateau', 'ramp', 'sparse'):
                for connectivity in (1, len(shape)):
                    add(f"geometry/{len(shape)}d/{pattern}/c{connectivity}", shape=shape, pattern=pattern, connectivity=connectivity)
        for layout in ('F', 'reverse', 'step', 'tagged', 'readonly-F'):
            add('layout/' + layout, layout=layout, dtype='f4')
        if name not in ('max_tree', 'max_tree_local_maxima'):
            for threshold in (0, 1, 2.5, 20, 21, -1, 'inf', 'nan'):
                add('threshold/' + str(threshold), threshold=threshold)
            add('threshold/provider', mode='protocol', fault='threshold-provider')
        if name != 'max_tree':
            for reuse in ('both', 'parent-only', 'traverser-only'):
                add('reuse/' + reuse, reuse=reuse)
            add('parent/python-provider', mode='python-reject', reuse='parent-provider')
            for reuse in ('out-of-bounds', 'cycle', 'short-traverser', 'wrong-parent-dtype', 'readonly-tree'):
                add('blocked-reuse/' + reuse, reuse=reuse, mode='entry-blocked')
        for fault in ('empty', 'singleton', 'rank1', 'bool', 'half', 'endian', 'complex', 'object', 'readonly-C', 'wide-connectivity'):
            add('blocked-intake/' + fault, fault=fault, mode='entry-blocked')
        for fault in ('connectivity-dimension', 'connectivity-even', 'unexpected-keyword', 'image-provider'):
            add('python-reject/' + fault, fault=fault, mode='python-reject')
    assert len({row['id'] for row in rows}) == len(rows)
    return tuple(rows)


CASES = cases()


class Tagged(np.ndarray): pass
class ProviderError(RuntimeError): pass
class EntryBlocked(RuntimeError): pass


def array(value):
    a = np.asarray(value)
    return dict(dtype=a.dtype.str, shape=list(a.shape), strides=list(a.strides),
                C=bool(a.flags.c_contiguous), F=bool(a.flags.f_contiguous),
                own=bool(a.flags.owndata), writeable=bool(a.flags.writeable), aligned=bool(a.flags.aligned),
                bytes=None if a.dtype.hasobject else a.tobytes().hex(),
                objects=[scalar(value) for value in a.ravel()] if a.dtype.hasobject else None)


def scalar(value):
    if isinstance(value, np.ndarray): return dict(type=type(value).__name__, array=array(value))
    if isinstance(value, np.generic): return dict(type=type(value).__name__, dtype=value.dtype.str, bytes=value.tobytes().hex())
    if type(value) is float: return dict(type='float', value=value.hex())
    if type(value) in (int, bool, str, type(None)): return dict(type=type(value).__name__, value=value)
    if isinstance(value, (tuple, list)): return [scalar(x) for x in value]
    return dict(type=type(value).__name__)


def neighbors(shape, connectivity):
    offsets = [delta for delta in itertools.product((-1, 0, 1), repeat=len(shape))
               if 0 < sum(x*x for x in delta) <= connectivity]
    return sorted(offsets, key=lambda delta: sum(x*x for x in delta))


def tree_definition(image, connectivity):
    """Small scalar union/find replay with independently enumerated neighbors."""
    shape=image.shape; values=list(np.asarray(image).ravel()); n=len(values)
    order=sorted(range(n), key=lambda index: values[index].item())
    parent=[-1]*n; roots=[-1]*n
    coordinates=list(itertools.product(*(range(s) for s in shape)))
    strides=[math.prod(shape[k+1:]) for k in range(len(shape))]
    for p in reversed(order):
        parent[p]=roots[p]=p
        for delta in neighbors(shape, connectivity):
            position=tuple(a+b for a,b in zip(coordinates[p], delta))
            if any(x<0 or x>=s for x,s in zip(position,shape)): continue
            q=sum(x*s for x,s in zip(position,strides))
            if parent[q]<0: continue
            root=q
            while roots[root]!=roots[roots[root]]:
                roots[root]=roots[roots[root]]
            root=roots[root]
            if root!=p: roots[root]=p; parent[root]=p
    for p in order:
        q=parent[p]
        if values[q]==values[parent[q]]: parent[p]=parent[q]
    return np.array(parent,dtype=np.int64).reshape(shape), np.array(order,dtype=np.int64)


def invert_definition(image):
    a=np.asarray(image)
    if a.dtype.kind=='u': return np.subtract(np.iinfo(a.dtype).max,a,dtype=a.dtype)
    if a.dtype.kind=='i': return np.subtract(-1,a,dtype=a.dtype)
    return np.subtract(1,a,dtype=a.dtype)


def filter_definition(name, image, threshold, connectivity):
    """Source-qualified scalar attributes/filter replay; exact label identities."""
    closing=name.endswith('closing'); work=invert_definition(image) if closing else np.asarray(image)
    parent,order=tree_definition(work,connectivity)
    if name=='max_tree': return parent,order
    values=list(work.ravel()); p=list(parent.ravel()); s=list(order); root=s[0]
    if name=='max_tree_local_maxima':
        out=[1]*len(p); label=1
        for pixel in reversed(s):
            if pixel==root: continue
            q=p[pixel]
            if values[pixel]!=values[q]:
                out[q]=0
                if out[pixel]>0: out[pixel]=label; label+=1
        for pixel in reversed(s):
            if pixel!=root and values[pixel]==values[p[pixel]]: out[pixel]=out[p[pixel]]
        return np.array(out,dtype=np.uint64).reshape(work.shape)
    attribute=[1.]*len(p)
    coordinates=list(itertools.product(*(range(dim) for dim in work.shape)))
    low=[list(c) for c in coordinates]; high=[list(c) for c in coordinates]
    for pixel in reversed(s):
        if pixel==root: continue
        q=p[pixel]
        if name.startswith('area'): attribute[q]+=attribute[pixel]
        else:
            low[q]=[min(a,b) for a,b in zip(low[q],low[pixel])]
            high[q]=[max(a,b) for a,b in zip(high[q],high[pixel])]
            attribute[q]=float(max(b-a for a,b in zip(low[q],high[q]))+1)
    out=[None]*len(p); out[root]=work.dtype.type(0) if attribute[root]<threshold else values[root]
    for pixel in s:
        if pixel==root: continue
        q=p[pixel]
        out[pixel]=out[q] if values[pixel]==values[q] or attribute[pixel]<threshold else values[pixel]
    result=np.array(out,dtype=work.dtype).reshape(work.shape)
    return invert_definition(result) if closing else result


def fixture(spec, actual_provided_tree=False):
    events=[];shape=tuple(spec['shape']);dtype=np.dtype(spec['dtype']);raw=np.arange(math.prod(shape)).reshape(shape)
    pattern=spec['pattern']
    if pattern=='flat': values=np.full(shape,2)
    elif pattern=='plateau': values=(raw//3)%4
    elif pattern=='sparse': values=np.where(raw%11==0,7,np.where(raw%3==0,2,0))
    elif pattern=='ramp': values=raw
    elif pattern=='limits':
        limits=np.iinfo(dtype);values=np.resize(np.array([limits.min,0,limits.max,1],dtype=dtype),shape)
    else: values=np.resize(np.array([-0.,0.,.5,-.5,.125]),shape)
    image=np.array(values,dtype=dtype);owner=image
    layout=spec['layout']
    if layout in ('F','readonly-F'): image=np.array(image,order='F');owner=image
    elif layout=='reverse': image=image[::-1,::-1]
    elif layout=='step':
        owner=np.zeros(tuple(2*s for s in shape),dtype=dtype);image=owner[tuple(slice(None,None,2) for s in shape)];image[...] = values
    elif layout=='tagged': image=image.view(Tagged)
    if layout=='readonly-F': image.flags.writeable=False
    options={'connectivity':spec['connectivity']}
    if spec['function'].startswith(('area','diameter')):
        threshold=spec['threshold'];options['area_threshold' if spec['function'].startswith('area') else 'diameter_threshold']=float(threshold) if threshold in ('inf','nan') else threshold
    reuse=spec.get('reuse');parent=traverser=None
    if reuse:
        work=invert_definition(image) if spec['function'].endswith('closing') else image
        if actual_provided_tree and spec['mode']=='numeric' and reuse=='both':
            # Actual original tree reuse, protected before the unchecked builder.
            certify_image(work.ravel(),work.shape)
            builder=original._max_tree._max_tree
            def checked_builder(*args):
                certify_entry('_max_tree',args,work.shape)
                return builder(*args)
            original._max_tree._max_tree=checked_builder
            try: parent,traverser=original.max_tree(work,spec['connectivity'])
            finally: original._max_tree._max_tree=builder
        else: parent,traverser=tree_definition(work,spec['connectivity'])
        if reuse=='parent-only': options['parent']=parent
        elif reuse=='traverser-only': options['tree_traverser']=traverser
        else: options.update(parent=parent,tree_traverser=traverser)
        if reuse=='out-of-bounds': parent.reshape(-1)[0]=image.size+10
        elif reuse=='cycle': parent.reshape(-1)[:2]=[1,0]
        elif reuse=='short-traverser': options['tree_traverser']=traverser[:1]
        elif reuse=='wrong-parent-dtype': options['parent']=parent.astype(np.int32)
        elif reuse=='readonly-tree': parent.flags.writeable=False;traverser.flags.writeable=False
        elif reuse=='parent-provider':
            class ParentProvider:
                def ravel(self): events.append('parent.ravel');raise ProviderError('parent ravel rejected')
            options['parent']=ParentProvider()
    fault=spec.get('fault')
    if fault=='empty': image=np.empty((0,5),dtype=dtype);owner=image
    elif fault=='singleton': image=np.zeros((1,5),dtype=dtype);owner=image
    elif fault=='rank1': image=np.arange(5,dtype=dtype);owner=image
    elif fault=='bool': image=image.astype(bool);owner=image
    elif fault=='half': image=image.astype('f2');owner=image
    elif fault=='endian': image=image.astype('>f4');owner=image
    elif fault=='complex': image=image.astype('c8');owner=image
    elif fault=='object': image=image.astype(object);owner=image
    elif fault=='readonly-C': image.flags.writeable=False
    elif fault=='wide-connectivity': options['connectivity']=np.ones((5,5),dtype=bool)
    elif fault=='connectivity-dimension': options['connectivity']=np.ones((3,),dtype=bool)
    elif fault=='connectivity-even': options['connectivity']=np.ones((2,2),dtype=bool)
    elif fault=='unexpected-keyword': options['unexpected']=True
    elif fault=='threshold-provider':
        class ThresholdProvider:
            def __float__(self): events.append('threshold.float');raise ProviderError('threshold conversion rejected')
        options['area_threshold' if spec['function'].startswith('area') else 'diameter_threshold']=ThresholdProvider()
    passed=image
    if fault=='image-provider':
        class ImageProvider:
            @property
            def shape(self): events.append('image.shape');raise ProviderError('shape provider rejected')
            def copy(self): events.append('image.copy');raise ProviderError('copy provider rejected')
            @property
            def dtype(self): events.append('image.dtype');raise ProviderError('dtype provider rejected')
        passed=ImageProvider()
    return passed,image,owner,options,events


def certify_image(image, shape):
    assert isinstance(image,np.ndarray) and image.ndim==1 and image.size==math.prod(shape)
    assert image.dtype.str.lstrip('<>=|') in DTYPES and image.dtype.isnative
    assert image.flags.c_contiguous and image.flags.writeable and image.flags.aligned
    assert len(shape) in (2,3) and all(4<=s<=8 for s in shape) and image.size<=512
    assert np.isfinite(image).all()


def certify_tree(image,parent,traverser):
    n=image.size
    assert parent.dtype==traverser.dtype==np.dtype('int64') and parent.size==traverser.size==n
    assert parent.flags.c_contiguous and traverser.flags.c_contiguous
    assert parent.flags.writeable and traverser.flags.writeable
    s=list(map(int,traverser));p=list(map(int,parent));assert sorted(s)==list(range(n))
    position={pixel:i for i,pixel in enumerate(s)}
    assert p[s[0]]==s[0]
    for pixel in s:
        assert 0<=p[pixel]<n
        if pixel!=s[0]: assert position[p[pixel]]<position[pixel]
        assert image[p[pixel]]<=image[pixel]


def certify_entry(name,args,shape):
    image=args[0];certify_image(image,shape)
    if name=='_max_tree':
        _,mask,structure,offset,dimensions,parent,traverser=args
        assert dimensions.dtype==offset.dtype==structure.dtype==np.dtype('int32')
        assert list(dimensions)==list(shape) and list(offset)==[1]*len(shape)
        assert mask.dtype==np.dtype('uint8') and mask.size==image.size
        shaped=mask.reshape(shape)
        for axis in range(len(shape)):
            assert not np.moveaxis(shaped,axis,0)[[0,-1]].any()
        center=sum(math.prod(shape[k+1:]) for k in range(len(shape)))
        for delta in map(int,structure):
            coords=np.unravel_index(center+delta,shape)
            assert all(abs(int(value)-1)<=1 for value in coords)
        assert parent.dtype==traverser.dtype==np.dtype('int64') and parent.size==traverser.size==image.size
        assert sorted(map(int,traverser))==list(range(image.size))
        assert list(traverser)==sorted(range(image.size),key=lambda i:image[i].item())
    elif name=='_compute_area': certify_tree(image,args[1],args[2])
    elif name=='_compute_extension':
        assert list(args[1])==list(shape);certify_tree(image,args[2],args[3])
    elif name=='_direct_filter':
        _,output,parent,traverser,attribute,threshold=args
        certify_tree(image,parent,traverser)
        assert output.dtype==image.dtype and output.size==image.size and output.flags.c_contiguous and output.flags.writeable
        assert attribute.dtype==np.dtype('f8') and attribute.size==image.size
        assert type(threshold) in (int,float)
        assert not np.shares_memory(output,image)
    else:
        _,output,parent,traverser=args;certify_tree(image,parent,traverser)
        assert output.dtype==np.dtype('u8') and output.size==image.size and output.flags.writeable
        assert np.all(output==1)


def observe(spec):
    authenticate()
    passed,image,owner,options,events=fixture(spec,actual_provided_tree=True)
    before=dict(image=array(image),owner=array(owner))
    provided={key:array(value) for key,value in options.items() if key in ('parent','tree_traverser') and isinstance(value,np.ndarray)}
    entries=[];providers=[];saved={};np_saved={}
    def entry_wrapper(name,function):
        def wrapper(*args):
            row={'name':name,'before':[scalar(value) for value in args]};entries.append(row)
            if spec['mode']=='entry-blocked' or (spec['mode']=='protocol' and name=='_direct_filter'):
                raise EntryBlocked('unchecked max-tree entry intentionally blocked')
            certify_entry(name,args,image.shape)
            result=function(*args)
            row['after']=[scalar(value) for value in args]
            row['result']=scalar(result)
            return result
        return wrapper
    def provider_wrapper(name,function):
        def wrapper(*args,**kwargs):
            providers.append({'name':name,'args':[scalar(value) for value in args], 'kwargs':{key:scalar(value) for key,value in kwargs.items()}})
            return function(*args,**kwargs)
        return wrapper
    error=None;result=None;old=np.seterr(all='warn')
    try:
        for name in ENTRIES:
            saved[name]=getattr(original._max_tree,name);setattr(original._max_tree,name,entry_wrapper(name,saved[name]))
        for name in ('max_tree','invert','_validate_connectivity','_offsets_to_raveled_neighbors'):
            saved[name]=getattr(original,name);setattr(original,name,provider_wrapper(name,saved[name]))
        for name in ('ones','zeros','argsort'):
            np_saved[name]=getattr(np,name);setattr(np,name,provider_wrapper('numpy.'+name,np_saved[name]))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                function=saved['max_tree'] if spec['function']=='max_tree' else getattr(original,spec['function'])
                result=function(passed,**options)
            except Exception as exc: error={'type':type(exc).__name__,'message':str(exc)}
            warning_rows=[{'type':type(row.message).__name__,'message':str(row.message)} for row in captured]
    finally:
        for name in ENTRIES: setattr(original._max_tree,name,saved[name])
        for name in ('max_tree','invert','_validate_connectivity','_offsets_to_raveled_neighbors'): setattr(original,name,saved[name])
        for name,function in np_saved.items(): setattr(np,name,function)
        np.seterr(**old)
    def output(value):
        return {'type':type(value).__name__,'array':array(value),'shares_input':bool(np.shares_memory(value,image)),
                'base':array(value.base) if isinstance(value.base,np.ndarray) else None,
                'shares_provided':{key:bool(np.shares_memory(value,options[key])) for key in provided}}
    outcome=None if result is None else [output(value) for value in result] if isinstance(result,tuple) else output(result)
    return {'before':before,'after':dict(image=array(image),owner=array(owner)), 'provided_before':provided,
            'provided_after':{key:array(options[key]) for key in provided},'providers':providers,'entries':entries,
            'result':outcome,'error':error,'warnings':warning_rows,'events':events}


@pytest.fixture(scope='module')
def frozen(): return json.loads(MATRIX.read_text())


def test_source_fixed_ids_safety_and_native0(frozen):
    authenticate()
    assert frozen['native_checks']==0 and list(CASES)==frozen['cases'] and digest(CASES)==frozen['case_contract_sha256']
    assert frozen['profiles'][profile()]['numpy']==np.__version__ and frozen['profiles'][profile()]['scipy']==scipy.__version__
    for name in NAMES: assert str(inspect.signature(getattr(original,name)))==frozen['signatures'][name]
    for row in frozen['sources'].values(): assert hashlib.sha256(row['utf8'].encode()).hexdigest()==row['sha256']
    assert hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==frozen['test_source_canonical_lf_sha256']


@pytest.mark.parametrize('spec',CASES,ids=lambda row:row['id'])
def test_original_exact_repeated_observation(spec,frozen):
    expected=frozen['observations'][frozen['profiles'][profile()]['snapshots'][spec['id']]]
    first=observe(spec);assert first==expected
    assert observe(spec)==first
    assert first['before']==first['after'] and first['provided_before']==first['provided_after']
    if spec['mode']=='numeric': assert first['error'] is None and first['entries'] and first['result'] is not None
    elif spec['mode']=='python-reject': assert first['error'] and first['error']['type']!='EntryBlocked'
    elif spec['mode']=='protocol': assert first['error']['type']=='EntryBlocked' and first['entries'][-1]['name']=='_direct_filter'
    else: assert first['error'] is not None  # Python may reject before the blocked entry.


@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',CASES,ids=lambda row:row['id'])
def test_future_native_pending_rejects_forwarding(spec,workers,monkeypatch):
    name='morphology.'+spec['function'];assert name not in native_functions()
    def forbidden(*args,**kwargs):raise AssertionError('original max-tree kernel forbidden in native-pending call')
    for entry in ENTRIES:monkeypatch.setattr(original._max_tree,entry,forbidden)
    previous=xx.get_num_threads();fallback=xx.set_fallback_enabled(False);xx.set_num_threads(workers)
    try:
        # Only the independent bounded metadata fixture is built; all original entries are forbidden.
        passed,_,_,options,_=fixture(spec)
        with pytest.raises(NotImplementedError):getattr(xx.morphology,spec['function'])(passed,**options)
    finally:xx.set_num_threads(previous);xx.set_fallback_enabled(fallback)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('shape',((4,5),(4,4,5)))
@pytest.mark.parametrize('connectivity',(1,2))
def test_independent_scalar_tree_attributes_and_labels(name,shape,connectivity):
    spec=dict(id='definition',function=name,mode='numeric',dtype='i2',shape=list(shape),pattern='sparse',layout='C',connectivity=connectivity,threshold=3)
    actual=observe(spec);assert actual['error'] is None
    image=fixture(spec)[1];expected=filter_definition(name,image,3,connectivity)
    if name=='max_tree':
        assert [row['array']['bytes'] for row in actual['result']]==[value.tobytes().hex() for value in expected]
    else:assert actual['result']['array']['bytes']==expected.tobytes().hex()
