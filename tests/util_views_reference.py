"""Exact captured util-view default requests; no original or native calls."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
ROOT=Path(__file__).resolve().parents[1]
APIS=('util.slice_along_axes','util.view_as_blocks','util.view_as_windows')


def reference(api,args,options,source_path,np,*,tagged=False):
    assert api in APIS and type(args)is tuple and len(args)==2 and type(options)is dict
    assert type(args[0])is np.ndarray and type(tagged)is bool
    dtype=args[0].dtype
    assert dtype.name in ('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64','complex64','complex128')
    if tagged:assert dtype.metadata is not None and dict(dtype.metadata)=={'units':'linear','source':'parity'}
    else:assert dtype.metadata is None
    source='1a75717aa0236c350b51cdbb79d8787411d7151bec2dcd335a474823b2be1bac'if api==APIS[0]else'fc9ab1bdc09fa464828b6ee3ce96e610d04606b90c13a3ed58f80138e65c876b'
    assert hashlib.sha256(Path(source_path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==source
    raw=(ROOT/'tests/fixtures/util_views_originals_v1.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='8a6378b76d9ca724e5f228de5a1d1d2b0e4735b61921d9515ef1f07c26fbb3c3'
    data=json.loads(raw);assert np.__version__ in ('1.24.4','2.5.4')
    profile='minimum'if np.__version__=='1.24.4'else'modern'
    family='slice'if api==APIS[0]else'array_views';name=api.split('.')[-1]
    suffix=('ordinary-default'if family=='slice'else'baseline')if dtype.name=='int16'else'dtype-'+dtype.name
    row=next(r for r in data['families'][family][profile]['rows']if r['id']==name+'/'+suffix)
    if family=='slice':
        from slice_along_axes_observer import encode
        assert type(args[1])is list and args[1]==([(1,3)]if dtype.name=='int16'else[(0,2)])and type(args[1][0])is tuple
        assert set(options)=={'axes','copy'}and options['axes']is None and options['copy']is False
    else:
        from array_views_observer import encode
        assert type(args[1])is tuple and args[1]==(2,2)and all(type(v)is int for v in args[1])
        assert options==({}if api==APIS[1]else{'step':1})
        if options:assert type(options['step'])is int
    physical=(args[0].view(np.dtype(dtype.str)),args[1])if tagged else args
    assert encode((physical,options),np)=={'container':'tuple','items':row['before']['items'][:2]}
    assert row['error']is None and not row['events']and not row['warnings']and not row['FP_events']
    gold=dict(row['result']);payload=zlib.decompress(base64.b64decode(gold['raw_zlib_base64']))
    assert hashlib.sha256(payload).hexdigest()==gold['raw_sha256']
    # Separate owned storage, then the source-observed metadata only. No shape
    # algorithm or original API is run to derive an expected answer.
    storage=args[0].ravel(order='C').copy()
    if family=='slice':
        offset=row['aliases']['input_array']['pointer_delta'];assert row['aliases']['input_array']['shares_memory']is True
        expected=np.ndarray(gold['shape'],dtype=dtype,buffer=storage,offset=offset,strides=gold['strides'])
    else:expected=np.lib.stride_tricks.as_strided(storage.reshape(args[0].shape),shape=gold['shape'],strides=gold['strides'])
    if tagged:gold['dtype_metadata']=dict(dtype.metadata)
    assert encode(expected,np)==gold
    assert expected.tobytes(order='C')==payload and not np.shares_memory(expected,args[0])
    return expected,('SOURCE_DERIVED_FIXED_TAG_VIEW_METADATA_EQUIVALENCE'if tagged else'IDENTICAL_CAPTURED_UTIL_VIEW_REQUEST'),row
