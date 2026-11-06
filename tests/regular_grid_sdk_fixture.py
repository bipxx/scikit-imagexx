"""Extract typed SDK inputs/results from admitted rows; stdlib only."""
import hashlib
import json
import math
from pathlib import Path
import struct
ROOT=Path(__file__).resolve().parents[1]


def render():
    path=ROOT/'tests/fixtures/regular_grid_originals_v1.json';raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='e29d10edf77af031674bb2118281976512a08ba95496346e650ed96ddadfd80a'
    def scalar(value):
        if value['type'][0]=='numpy':
            dtype=value['dtype'];data=bytes.fromhex(value['raw'])
            return struct.unpack(('<' if dtype[0]!='>' else '>')+{'i8':'q','f8':'d'}[dtype[1:]],data)[0]
        if 'hex' in value:return float.fromhex(value['hex'])
        return value['value']
    def vector(values):return '{'+','.join(str(v) for v in values)+'}'
    rows=[]
    for profile,packet in json.loads(raw)['profiles'].items():
        for row in packet['raw_rows']:
            shape,point=row['before']['items'];points=scalar(point)
            if points is None or point['type'][0]=='numpy' and points==0:continue
            if 'items' in shape and shape['type'][0]=='builtins':
                values=[scalar(v) for v in shape['items']]
                if any(type(v)is list or type(v)is int and not -(2**63)<=v<2**63 for v in values):continue
                if not values or any(type(v)is float for v in values):dtype='<f8';data=b''.join(struct.pack('<d',v) for v in values)
                elif all(type(v)is bool for v in values):dtype='|b1';data=bytes(values)
                else:dtype='<i8';data=b''.join(struct.pack('<q',v) for v in values)
                count=len(values);stride=int(dtype[2:])
            elif shape['type'][0]=='numpy' and len(shape['shape'])==1:
                dtype=shape['dtype'];data=bytes.fromhex(shape['raw']);count=shape['shape'][0];stride=shape['strides'][0]
            else:continue
            name={'b':'boolean','i':'int','u':'uint','f':'float'}[dtype[1]]
            if name!='boolean':name+=str(int(dtype[2:])*8)
            expected=[]
            if row['error'] is None:
                for item in row['result']['items']:
                    if item['start']['type']==['builtins','NoneType']:expected.append('{skimagexx::RegularGridSliceKind::all,0,1}')
                    else:expected.append('{skimagexx::RegularGridSliceKind::strided,'+str(scalar(item['start']))+','+str(scalar(item['step']))+'}')
            else:assert row['error']==dict(type='ZeroDivisionError',message='float division by zero')
            point_text=('-' if points<0 else '')+'std::numeric_limits<double>::infinity()' if math.isinf(points) else float(points).hex()
            rows.append('{'+','.join([json.dumps(profile+':'+row['id']),'skimagexx::PixelType::'+name,str(dtype[0]=='>').lower(),str(int(dtype[2:])),str(count),str(stride),vector(data),point_text,str(row['error'] is not None).lower(),'{'+','.join(expected)+'}'])+'}')
    return ('// Generated literal admitted inputs and slice words; no reference calls.\n#pragma once\n#include <skimagexx/regular_grid.hpp>\n#include <limits>\n'
        'namespace regular_grid_originals {\nstruct Row {const char* id;skimagexx::PixelType type;bool swap;std::size_t bytes,count;std::ptrdiff_t stride;std::vector<std::uint8_t> input;double points;bool division_error;std::vector<skimagexx::RegularGridSlice> expected;};\n'
        'inline std::vector<Row> rows(){return {\n'+',\n'.join(rows)+'\n};}\n}\n')


if __name__=='__main__':(ROOT/'tests/fixtures/regular_grid_originals_v1.hpp').write_text(render(),encoding='utf-8',newline='\n')
