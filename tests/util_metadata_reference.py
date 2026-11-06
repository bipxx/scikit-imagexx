"""Bounded saved metadata references; source/request authentication, no originals."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
ROOT=Path(__file__).resolve().parents[1]
PINS={'util.crop':('crop_originals_v1.json','cad3b7f4b5bd21573c30af67758912aecf379c710136f1baec5118dc2552b8f3','558df9e59105586d0b0e374a33836bd447919e9e5d91f21586127b7cafabfbac'),
      'util.regular_grid':('regular_grid_originals_v1.json','e29d10edf77af031674bb2118281976512a08ba95496346e650ed96ddadfd80a','64fcce3556f0cbf8cafd05a54ea3545b65b18bdcabbee25b953a21ba06875581')}


def reference(api,args,options,source_path,np,*,tagged=False):
    name,pin,source_pin=PINS[api]
    raw=(ROOT/'tests/fixtures'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==pin
    assert hashlib.sha256(Path(source_path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==source_pin
    data=json.loads(raw);profile='minimum'if np.__version__.startswith('1.')else'modern'
    assert np.__version__==data['profiles'][profile]['runtime']['numpy']
    assert type(args)is tuple and len(args)==2 and type(options)is dict
    assert type(args[0])is np.ndarray and type(tagged)is bool
    dtype=args[0].dtype
    if tagged:
        assert dtype.metadata is not None and all(type(k)is str and type(v)is str for k,v in dtype.metadata.items())
        assert dict(dtype.metadata)=={'units':'linear','source':'parity'}
    else:assert dtype.metadata is None
    # Only this exact shared-test tag map is admitted. Actual candidate inputs
    # keep tags/dtype; a bare view below authenticates unchanged physical gold.
    bare=np.dtype(dtype.str)
    if api=='util.crop':
        from crop_observer import encode
        assert dtype.name in ('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64','complex64','complex128')
        assert type(args[1])is int and args[1]==0 and set(options)=={'copy','order'}
        assert type(options['copy'])is bool and options['copy']is False
        assert type(options['order'])is str and options['order']=='K'
        row=next(r for r in data['profiles'][profile]['rows']if r['id']=='crop/'+('width-zero'if dtype.name=='int16'else'dtype-'+dtype.name))
        physical_args=(args[0].view(bare),args[1])if tagged else args
        assert encode((physical_args,options),np)=={'container':'tuple','items':row['before']['items'][:2]}
        assert row['error']is None and not row['events'] and not row['warnings'] and not row['FP_events']
        gold=row['result'];payload=zlib.decompress(base64.b64decode(gold['raw_zlib_base64']))
        assert hashlib.sha256(payload).hexdigest()==gold['raw_sha256']
        expected=np.frombuffer(payload,dtype).copy().reshape(gold['shape'])
        expected_metadata=dict(gold)
        if tagged:expected_metadata['dtype_metadata']=dict(dtype.metadata)
        assert encode(expected,np)==expected_metadata
        return expected,('SOURCE_DERIVED_FIXED_TAG_CROP_METADATA_EQUIVALENCE'if tagged else'IDENTICAL_CAPTURED_CROP_REQUEST'),row
    from regular_grid_observer import encode
    assert options=={} and type(args[1])is int
    assert dtype.name in ('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float32','float64')
    literal=np.array([5,7],dtype=dtype)
    if tagged:literal=literal.view(dtype)
    assert encode(args[0],np)==encode(literal,np)
    assert args[1]==(1 if dtype.kind=='b'else 6)
    rows=data['profiles'][profile]['raw_rows']
    row=next(r for r in rows if r['id']==('bool-dimensions'if dtype.kind=='b'else'array-i16'))
    assert row['error']is None and not row['warnings'] and not row['events']
    # Source-derived typed metadata equivalence, not same-input capture: exact
    # finite2vector product/sort is1/True or35/[5,7] in every admitted dtype.
    # np.asanyarray prefix yields this request's dtype; source then float(prod),
    # default-int conversion and axis order. No source/grid body is invoked.
    state=np.geterr(),np.geterrcall(),np.getbufsize()
    assert float(np.prod(args[0]))==(1. if dtype.kind=='b'else 35.)
    assert np.sort(args[0]).tolist()==([True,True]if dtype.kind=='b'else[5,7])
    assert np.argsort(np.argsort(args[0])).tolist()==[0,1]
    assert state==(np.geterr(),np.geterrcall(),np.getbufsize())
    expected=[]
    for item in row['result']['items']:
        if item['start']['type']==['builtins','NoneType']:expected.append(slice(None));continue
        values=[]
        for key in ('start','step'):
            saved=item[key];value=int.from_bytes(bytes.fromhex(saved['raw']),'little',signed=True)
            assert -(2**31)<=value<2**31
            values.append(np.dtype(int).type(value))
        expected.append(slice(values[0],None,values[1]))
    return tuple(expected),('SOURCE_DERIVED_FIXED_TAG_GRID_METADATA_EQUIVALENCE'if tagged else'SOURCE_DERIVED_TYPED_GRID_METADATA_EQUIVALENCE'),row
