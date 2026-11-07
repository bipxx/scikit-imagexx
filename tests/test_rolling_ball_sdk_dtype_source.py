"""Source/transport-only SDK dtype guards, no scientific imports."""
import hashlib,json,struct,base64,zlib
from pathlib import Path
import pytest
from rolling_ball_sdk_dtype_context import ROOT,CONTEXT,historical_sdk_source
H=lambda b:hashlib.sha256(b).hexdigest()
def test_exact_source_dependency_and_historical_projection():
 d=json.loads((ROOT/CONTEXT).read_bytes())
 for name,pin in d['pins'].items():assert H((ROOT/name).read_bytes())==pin,name
 for name,row in d['source_projection'].items():assert H(historical_sdk_source(name,(ROOT/name).read_bytes()))==row['predecessor_raw_sha256']
 assert d['original_calls']==d['timing']==0 and d['state']=='PRIVATE_INSTALLED_SDK_COMPONENT_ONLY'
@pytest.mark.parametrize('name',('include/skimagexx/rolling_ball.hpp','src/rolling_ball.cpp'))
def test_projection_rejects_unassessed_source(name):
 with pytest.raises(AssertionError):historical_sdk_source(name,(ROOT/name).read_bytes()+b' ')
def test_new_complete_operation_and_cast_frontier():
 d=json.loads((ROOT/CONTEXT).read_bytes());s=(ROOT/'src/rolling_ball.cpp').read_text();h=(ROOT/'include/skimagexx/rolling_ball.hpp').read_text()
 assert 'RollingBallResult rolling_ball(RollingBallStorage'in s and 'RollingBallResult rolling_ball(RollingBallStorage'in h
 assert 'detail::decode_half'in s and 'detail::encode_half'in s and 'integer result outside finite representable cast domain'in s
 assert 'std::isfinite(number)'in s and 'std::fegetround()!=FE_TONEAREST'in s
 assert not any(name in s for name in('dtype-bool','dtype-uint8','swapped-f32','ROLLDT1','0c40'))
 assert d['new_saved_records']==11 and d['prior_saved_records']==52
 assert d['genuine_admitted']is False
 assert d['new_original_budget']==0

def test_saved_transport_is_verbatim_admitted_input_and_output():
 d=json.loads((ROOT/CONTEXT).read_bytes());saved=json.loads((ROOT/'tests/fixtures/rolling_ball_original_v1.json').read_bytes())['profiles']
 data=(ROOT/'tests/fixtures/rolling_ball_sdk_dtypes_v1.bin').read_bytes();assert data[:8]==b'ROLLDT1\0'and struct.unpack_from('<Q',data,8)[0]==11
 at=16;names=[]
 for _ in range(11):
  length,code,half,swap,rank,backing,origin,radius,safe=struct.unpack_from('<7QdQ',data,at);at+=72;name=data[at:at+length].decode();at+=length
  shape=struct.unpack_from('<'+'q'*rank,data,at);at+=8*rank;strides=struct.unpack_from('<'+'q'*rank,data,at);at+=8*rank
  row=next(r for r in saved['minimum']['rows']if r['id']==name);other=next(r for r in saved['modern']['rows']if r['id']==name);image=row['before']['items'][0]['items'][0]
  width=(1,1,1,2,2,4,4,8,8,4,8)[code];n=1
  for size in shape:n*=size
  raw=zlib.decompress(base64.b64decode(image['raw_zlib_base64']));expected=zlib.decompress(base64.b64decode(row['result']['raw_zlib_base64']))
  assert H(raw)==image['raw_sha256']and H(expected)==row['result']['raw_sha256']and row['result']==other['result']
  physical=data[at:at+backing];at+=backing;logical=bytearray()
  for i in range(n):
   coords=[];v=i
   for size in reversed(shape):coords.append(v%size);v//=size
   offset=origin+sum(c*s for c,s in zip(reversed(coords),strides));logical+=physical[offset:offset+width]
  assert bytes(logical)==raw and data[at:at+len(expected)]==expected;at+=len(expected);names.append(name)
 assert at==len(data)and names==d['saved_ids']
