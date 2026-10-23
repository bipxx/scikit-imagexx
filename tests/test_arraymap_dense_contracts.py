"""Bounded original dense ArrayMap protocols; native observations only locate gaps.

__array__ uses NumPy zeros plus checked ndarray assignment, not unchecked
Cython map_array. No public class registration or native acceptance is implied.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import warnings
import numpy as np
import pytest
import skimage
from skimage.util._map_array import ArrayMap as OriginalArrayMap

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'coverage/contracts/arraymap-dense.json'


def cases():
    keys = [('u8-zero','u1',[0]), ('u8-edge','u1',[0,3,255]), ('u8-reverse','u1',[255,3,0]),
        ('u8-only','u1',[255]), ('i8-edge','i1',[0,3,127]), ('u16-edge','u2',[0,3,255]),
        ('i16-edge','i2',[0,3,255]), ('i64-edge','i8',[0,3,255]), ('u64-edge','u8',[0,3,255]),
        ('negative-valid','i2',[-1,0,2]), ('negative-invalid','i2',[-4,0,2]),
        ('all-negative','i2',[-3,-1]), ('empty','i2',[]), ('duplicates','i2',[0,1,1]),
        ('boolean-mask','bool',[False,True]), ('scalar','i2',3)]
    for name, dtype, values in keys:
        for action in ('direct','direct-copy-false','direct-copy-true','asarray','array-copy-false','array-copy-true','array-copy-none','asarray-copy-false'):
            yield dict(id=f'basic/{name}/{action}', key_dtype=dtype, keys=values, action=action)
    for name, dtype, values in [keys[1], ('ordinary','i2',[0,3,4])]:
        for target in ('default','f4','i2','u1','bool','>i2','tagged','object','c16','U3','invalid'):
            for copy in (None,False,True):
                yield dict(id=f'dtype/{name}/{target}/{copy}', key_dtype=dtype, keys=values, target=target, copy=copy, action='typed')
    for name, dtype, values in [keys[1], ('ordinary','i2',[0,3,4])]:
        for key_layout in ('reverse','step','endian','readonly','tagged'):
            for value_layout in ('C','reverse','step','readonly'):
                yield dict(id=f'layout/{name}/{key_layout}/{value_layout}', key_dtype=dtype, keys=values,
                    key_layout=key_layout, value_layout=value_layout, action='direct')
    for name, dtype, values in [keys[1],keys[8],('ordinary','i2',[0,3,4])]:
        for cache in ('none','cached','readonly','alias-values'):
            for action in ('direct','twice','set-valid','set-invalid','set-mixed-invalid','set-bad-cast','set-duplicates'):
                yield dict(id=f'cache/{name}/{cache}/{action}', key_dtype=dtype, keys=values, cache=cache, action=action)
    for name, dtype, values in [keys[1],keys[4]]:
        for over in ('warn','ignore','raise'):
            for action in ('direct','set-valid'):
                yield dict(id=f'overflow/{name}/{over}/{action}', key_dtype=dtype, keys=values, over=over, action=action)
    for target in ('invalid','object'):
        yield dict(id=f'empty-dtype/{target}', key_dtype='i2', keys=[], target=target, action='typed')
    for dtype, values, offset in [('u1',[0,3,255],1),('u1',[0,3,255],254),('i1',[0,3,127],1),('u8',[0,3,255],1),('u2',[0,3,255],1)]:
        for direction in ('forward','inverse'):
            for action in ('direct','asarray','direct-copy-false','set-valid','set-invalid'):
                yield dict(id=f'returned/{dtype}/{offset}/{direction}/{action}', key_dtype=dtype, keys=values,
                    offset=offset, direction=direction, origin='returned', action=action)


CASES = list(cases())
_NATIVE_CLASS = None


def scalar(value):
    if isinstance(value,np.generic):return dict(type=type(value).__name__,dtype=value.dtype.str,bytes=value.tobytes().hex())
    if isinstance(value,list):return [scalar(v) for v in value]
    if isinstance(value,tuple):return [scalar(v) for v in value]
    if isinstance(value,bytes):return dict(bytes=value.hex())
    return dict(type=type(value).__name__,repr=repr(value))


def array(value):
    if value is None:return None
    if not isinstance(value,np.ndarray):return scalar(value)
    data=scalar(value.tolist()) if value.dtype.hasobject else dict(bytes=value.tobytes().hex())
    return dict(type=type(value).__name__,dtype=value.dtype.str,descr=repr(value.dtype.descr),metadata=repr(value.dtype.metadata),
        shape=list(value.shape),strides=list(value.strides),owned=bool(value.flags.owndata),writeable=bool(value.flags.writeable),
        aligned=bool(value.flags.aligned),c=bool(value.flags.c_contiguous),f=bool(value.flags.f_contiguous),data=data)


def layout(value,kind):
    if value.ndim==0:return value
    if kind=='reverse':return value[::-1]
    if kind=='step':
        owner=np.zeros(value.size*2,value.dtype);owner[::2]=value;return owner[::2]
    if kind=='endian':return value.astype(value.dtype.newbyteorder('>'))
    if kind=='readonly':value.flags.writeable=False
    if kind=='tagged':value=value.view(np.dtype(value.dtype,metadata={'scope':'dense-map'}))
    return value


def construct(spec,native=False):
    keys=layout(np.array(spec['keys'],dtype=spec['key_dtype']),spec.get('key_layout','C'))
    # At most256 dense entries, at most8192 allocated bytes for this corpus.
    assert keys.size==0 or np.max(keys)<=255
    values=np.arange(keys.size,dtype=np.int16).reshape(keys.shape)+2
    values=layout(values,spec.get('value_layout','C'))
    if native:
        import skimagexx as xx
        function=xx.segmentation.relabel_sequential
        assert _NATIVE_CLASS is not None, 'native returned-class setup must occur outside the observed protocol call'
        cls=_NATIVE_CLASS
        assert not cls.__module__.startswith('skimage.'),'actual native returned class required'
    else:
        from skimage.segmentation import relabel_sequential as function
        cls=OriginalArrayMap
    if spec.get('origin')=='returned':
        output,forward,inverse=function(keys,offset=spec['offset'])
        mapping=forward if spec['direction']=='forward' else inverse
        keys,values=mapping.in_values,mapping.out_values
    else:mapping=cls(keys,values)
    cache=spec.get('cache','none')
    if cache=='alias-values':mapping._array=values
    elif cache!='none':
        mapping._array=np.arange(5,dtype=np.int16)
        if cache=='readonly':mapping._array.flags.writeable=False
    return mapping,keys,values,mapping._array


def state(mapping,keys,values,cache):
    return dict(in_values=array(mapping.in_values),out_values=array(mapping.out_values),cache=array(mapping._array),
        same_keys=mapping.in_values is keys,same_values=mapping.out_values is values,same_cache=mapping._array is cache,
        initial_keys=array(keys),initial_values=array(values),initial_cache=array(cache),max_str_lines=mapping._max_str_lines)


def perform(mapping,spec):
    action=spec['action']
    if action=='direct':return mapping.__array__()
    if action=='direct-copy-false':return mapping.__array__(copy=False)
    if action=='direct-copy-true':return mapping.__array__(copy=True)
    if action=='asarray':return np.asarray(mapping)
    if action=='array-copy-false':return np.array(mapping,copy=False)
    if action=='array-copy-true':return np.array(mapping,copy=True)
    if action=='array-copy-none':return np.array(mapping,copy=None)
    if action=='asarray-copy-false':return np.asarray(mapping,copy=False)
    if action=='typed':
        target=spec.get('target','default')
        dtype=None if target=='default' else np.dtype('f4',metadata={'scope':'result'}) if target=='tagged' else 'not-a-dtype' if target=='invalid' else np.dtype(target)
        return mapping.__array__(dtype=dtype,copy=spec.get('copy'))
    if action=='twice':
        first=mapping.__array__(copy=False);first[0]=99;second=mapping.__array__(copy=False)
        return dict(first=array(first),second=array(second),same=first is second,alias=bool(np.shares_memory(first,second)))
    if action=='set-valid':mapping[0]=7
    elif action=='set-invalid':mapping[999]=7
    elif action=='set-mixed-invalid':mapping[np.array([0,999])]=np.array([7,8])
    elif action=='set-bad-cast':mapping[np.array([0,1])]=['7','bad']
    elif action=='set-duplicates':mapping[np.array([0,0])]=np.array([7,8])
    else:raise AssertionError('unknown action')
    return None


def observe(spec,native=False):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        mapping=keys=values=cache=None
        result=error=None;before=after=None
        try:
            mapping,keys,values,cache=construct(spec,native)
            before=state(mapping,keys,values,cache)
            with np.errstate(over=spec.get('over','warn')):value=perform(mapping,spec)
            if isinstance(value,np.ndarray):
                result=dict(array=array(value),alias_keys=bool(np.shares_memory(value,keys)),alias_values=bool(np.shares_memory(value,values)),
                    same_cache=value is mapping._array,alias_cache=bool(cache is not None and np.shares_memory(value,cache)))
            else:result=value
        except Exception as caught:error=dict(type=type(caught).__name__,message=str(caught))
        if mapping is not None:after=state(mapping,keys,values,cache)
    return dict(result=result,error=error,warnings=[dict(type=type(w.message).__name__,message=str(w.message)) for w in notices],before=before,after=after)


def profile():return ('windows' if platform.system()=='Windows' else 'linux')+('-minimum' if np.__version__.startswith('1.') else '-modern')


def test_checked_source_and_bounded_scope():
    source=inspect.getsource(OriginalArrayMap.__array__)
    assert 'np.zeros(np.max(self.in_values) + 1' in source and 'output[self.in_values] = self.out_values' in source
    assert 'map_array(' not in source
    assert len({s['id'] for s in CASES})==len(CASES)
    assert all(not s['keys'] or np.max(np.array(s['keys'],dtype=s['key_dtype']))<=255 for s in CASES if isinstance(s['keys'],list))


@pytest.mark.parametrize('spec',CASES,ids=[s['id'] for s in CASES])
def test_original_dense_protocol(spec):
    matrix=json.loads(MATRIX.read_text());frozen=matrix['profiles'][profile()]
    assert skimage.__version__=='0.26.0' and np.__version__==frozen['numpy']
    assert observe(spec)==matrix['observations'][frozen['observations'][spec['id']]]


def main():
    global _NATIVE_CLASS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-reference',type=Path);parser.add_argument('--audit-native',type=Path)
    args=parser.parse_args();assert bool(args.record_reference)!=bool(args.audit_native)
    report=dict(profile=profile(),python=sys.version,numpy=np.__version__,scikit_image=skimage.__version__,
        source_file=inspect.getsourcefile(OriginalArrayMap),source_sha256=hashlib.sha256(Path(inspect.getsourcefile(OriginalArrayMap)).read_bytes()).hexdigest(),
        methods={name:inspect.getsource(getattr(OriginalArrayMap,name)) for name in ('__len__','__array__','__setitem__')},
        classification='original-reference' if args.record_reference else 'native-gap-audit; never acceptance',cases=[])
    if args.record_reference:
        for spec in CASES:report['cases'].append(dict(spec=spec,reference=observe(spec)))
        output=args.record_reference
    else:
        import skimagexx as xx
        from skimagexx.coverage import native_functions
        assert 'segmentation.relabel_sequential' in native_functions()
        report.update(native_core=str(Path(xx._core.__file__).resolve()),native_core_sha256=hashlib.sha256(Path(xx._core.__file__).read_bytes()).hexdigest())
        previous=xx.get_num_threads();policy=xx.set_fallback_enabled(False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _NATIVE_CLASS=type(xx.segmentation.relabel_sequential(np.array([0,1],np.int16))[1])
            assert not _NATIVE_CLASS.__module__.startswith('skimage.')
            for spec in CASES:
                reference=observe(spec);actual={}
                for worker in (1,4):xx.set_num_threads(worker);actual[str(worker)]=observe(spec,True)
                report['cases'].append(dict(spec=spec,reference=reference,native_by_workers=actual,
                    matches={w:o==reference for w,o in actual.items()}))
        finally:xx.set_num_threads(previous);xx.set_fallback_enabled(policy)
        output=args.audit_native
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['classification'],len(CASES),'cases',sum(not all(r.get('matches',{'reference':True}).values()) for r in report['cases']),'native gaps')


if __name__=='__main__':main()
