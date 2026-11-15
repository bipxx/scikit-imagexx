"""Stdlib-only transport of eight admitted observations into SDK literals.

No recipe/native/original/image algorithm is invoked; raw words remain exact.
"""
import base64,hashlib,json,math,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRANSPORT=ROOT/'tests/fixtures/mark_wide_outer_originals_v1.json'
TYPE={'|u1':('uint8',1),'<i8':('int64',8),'<u8':('uint64',8),'<f4':('float32',4),'<f8':('float64',8),'|b1':('boolean',1)}
H=lambda raw:hashlib.sha256(raw).hexdigest()


def payload(value):
    assert value['type']=='ndarray'and value['dtype']in TYPE
    raw=zlib.decompress(base64.b64decode(value['raw_zlib_base64'],validate=True))
    assert len(raw)==math.prod(value['shape'])*TYPE[value['dtype']][1]
    assert H(raw)==value['raw_sha256']
    return raw


def observations(transport=None):
    transport=json.loads(TRANSPORT.read_bytes())if transport is None else transport
    rows=[]
    for profile,entry in transport['profiles'].items():
        assert profile in('minimum','modern')
        raw=zlib.decompress(base64.b64decode(entry['observations_zlib_base64'],validate=True))
        assert H(raw)==entry['observations_sha256']
        saved=json.loads(raw);assert len(saved)==4
        for row in saved:
            assert row['error']is None and row['warnings']==[]
            mark=row['api']=='segmentation.mark_boundaries'
            assert mark or row['api']=='segmentation.find_boundaries'
            args=row['before']['items'][0]['items'];assert len(args)==(2 if mark else 1)
            labels=args[-1];assert labels['shape']==[3,4]and labels['strides']==[32,8]and labels['dtype']in('<i8','<u8')
            inputs=dict(labels=labels,label_bytes=payload(labels),image=args[0]if mark else None,image_bytes=payload(args[0])if mark else b'')
            result=row['result'];raw_result=payload(result)
            assert result['shape']==([3,4,3]if mark else[3,4])and result['own']is True
            width=TYPE[result['dtype']][1];assert all(s%width==0 for s in result['strides'])
            fp={phase:{k:row['target_immediate'][phase][k]for k in('control','status','mxcsr')}for phase in('before','after')}
            assert fp['before']['control']&0x3f==0x3f and fp['before']['control']&0xc00==0
            assert fp['before']['mxcsr']&0x1f80==0x1f80 and fp['before']['mxcsr']&(3<<13)==0
            rows.append(dict(profile=profile,id=row['id'],mark=mark,result=result,result_bytes=raw_result,result_element_strides=[s//width for s in result['strides']],fp=fp,**inputs))
    assert len(rows)==8
    return rows


def literal(values):return '{'+','.join(str(x)for x in values)+'}'


def render():
    lines=['#pragma once','#include "../cmake_consumer/mark_boundaries_consumer.hpp"','#include <cstdint>','#include <utility>',
      'namespace skimagexx_test {',
      'struct WideFP {std::uint16_t control,status;std::uint32_t mxcsr;};',
      'struct WideSaved {MarkSaved data;bool mark;skimagexx::PixelType result_type;WideFP before,after;};',
      'inline std::vector<WideSaved> mark_wide_saved_cases(){std::vector<WideSaved> out;']
    for row in observations():
        lines+=['{WideSaved w;auto& c=w.data;',f'c.id={json.dumps(row["profile"]+"/"+row["id"])};c.modern={str(row["profile"]=="modern").lower()};w.mark={str(row["mark"]).lower()};']
        for field,descriptor,raw in [('labels',row['labels'],row['label_bytes']),('image',row['image'],row['image_bytes'])]:
            if descriptor is None:continue
            cpp,_=TYPE[descriptor['dtype']]
            lines+=[f'c.{field}.type=skimagexx::PixelType::{cpp};c.{field}.half=false;c.{field}.swap=false;c.{field}.shape={literal(descriptor["shape"])};c.{field}.strides={literal(descriptor["strides"])};',f'c.{"label"if field=="labels"else"image"}_bytes=mark_bytes("{raw.hex()}");']
        cpp,_=TYPE[row['result']['dtype']]
        lines+=[f'w.result_type=skimagexx::PixelType::{cpp};c.result_shape={literal(row["result"]["shape"])};c.result_strides={literal(row["result_element_strides"])};',f'c.expected=mark_bytes("{row["result_bytes"].hex()}");',
          'c.parameters.boundary={skimagexx::detail::FindBoundaryMode::outer,1,0,true};c.parameters.outline=false;',
          'c.parameters.color={{1,1,0},{3}};c.parameters.cubic_profile=c.modern?skimagexx::MarkCubicProfile::scipy118:skimagexx::MarkCubicProfile::scipy115;']
        for phase in('before','after'):
            lines+=[f'w.{phase}={literal(row["fp"][phase][k]for k in("control","status","mxcsr"))};']
        lines+=['out.push_back(std::move(w));}']
    return '\n'.join(lines+['return out;}','}'])+'\n'
