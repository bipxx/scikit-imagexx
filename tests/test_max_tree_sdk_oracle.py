"""Authenticate the standalone SDK's original source data; no image calls."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
CLIENT=ROOT/'tests/cmake_consumer'
ORACLE=CLIENT/'max_tree_source_oracle.bin'
MATRIX=ROOT/'coverage/contracts/max-tree.json'
ORACLE_SHA='e7162c6aa8a43434e0c6db0f835912dfc48b64eedca7e79192cab61bc39d93b5'
NAMES=('max_tree','max_tree_local_maxima','area_opening','area_closing','diameter_opening','diameter_closing')
TYPES=('u1','u2','u4','u8','i1','i2','i4','i8','f4','f8')
WIDTH=(1,2,4,8,1,2,4,8,4,8)


def read_oracle():
    raw=ORACLE.read_bytes()
    assert len(raw)==174036 and hashlib.sha256(raw).hexdigest()==ORACLE_SHA
    stream=io.BytesIO(raw)
    def words(fmt):
        size=struct.calcsize('<'+fmt);return struct.unpack('<'+fmt,stream.read(size))
    assert stream.read(8)==b'MTREEV1\0' and words('I')==(233,)
    rows=[]
    for _ in range(233):
        length,=words('I');identifier=stream.read(length).decode('utf-8')
        kind,operation,rank,connectivity=words('4I');shape=words('3Q')
        threshold,reused,count=words('dIQ')
        assert kind<len(TYPES) and operation<len(NAMES) and rank in (2,3) and 0<count<=512
        value=stream.read(count*WIDTH[kind]);parent=stream.read(count*8);traverser=stream.read(count*8)
        output=stream.read(count*(8 if operation==1 else WIDTH[kind])) if operation else None
        rows.append(dict(id=identifier,dtype=TYPES[kind],function=NAMES[operation],rank=rank,shape=shape,
                         connectivity=connectivity,threshold=threshold,reuse=bool(reused),count=count,
                         image=value,parent=parent,traverser=traverser,output=output))
    assert stream.read()==b''
    return rows


@pytest.mark.parametrize('profile',('windows','modern','minimum'))
def test_every_sdk_oracle_word_matches_frozen_original(profile):
    raw=MATRIX.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='45d5999d49f3b1a5cdb8c7ec5767de470e74b158f5fb489f288132722edc6365'
    matrix=json.loads(raw);cases=[c for c in matrix['cases'] if c['mode']=='numeric'];rows=read_oracle()
    assert len(cases)==233 and [r['id'] for r in rows]==[c['id'] for c in cases]
    def data(record):
        return bytes.fromhex(record.get('array',record)['bytes'])
    for row,case in zip(rows,cases):
        original=matrix['observations'][matrix['profiles'][profile]['snapshots'][case['id']]]
        assert row['dtype']==case['dtype'] and row['function']==case['function']
        assert row['rank']==len(case['shape']) and row['shape']==tuple(case['shape']+[1]*(3-len(case['shape'])))
        assert row['connectivity']==case['connectivity']
        # The immutable JSON represents inf/nan parameters as explicit strings.
        assert struct.pack('<d',row['threshold'])==struct.pack('<d',float(case['threshold']))
        # The original rebuilds if either supplied array is missing.
        reused={'parent','tree_traverser'}<=set(original['provided_before'])
        assert row['image']==data(original['before']['image']) and row['reuse']==reused
        if case['function']=='max_tree':
            parent,traverser=original['result']
        elif row['reuse']:
            parent=original['provided_before']['parent'];traverser=original['provided_before']['tree_traverser']
        else:
            builder=next(e for e in original['entries'] if e['name']=='_max_tree')
            parent,traverser=builder['after'][5:7]
        assert row['parent']==data(parent) and row['traverser']==data(traverser)
        if row['output'] is not None:assert row['output']==data(original['result'])


@pytest.mark.parametrize('state,accepted',(('valid',True),('missing',False),('changed',False)))
def test_standalone_cmake_rejects_missing_or_changed_oracle(tmp_path,state,accepted):
    name='cmake.exe' if sys.platform=='win32' else 'cmake'
    own=Path(sys.executable).with_name(name)
    cmake=str(own) if own.exists() else shutil.which('cmake')
    if cmake is None:pytest.skip('CMake is required to exercise its native data guard')
    helper=tmp_path/'max_tree_source_oracle.cmake'
    helper.write_bytes((CLIENT/helper.name).read_bytes())
    if state!='missing':
        raw=ORACLE.read_bytes()
        if state=='changed':raw=raw[:-1]+bytes([raw[-1]^1])
        (tmp_path/ORACLE.name).write_bytes(raw)
    result=subprocess.run([cmake,'-P',str(helper)],capture_output=True,text=True)
    assert (result.returncode==0) is accepted
    if not accepted:assert 'max-tree source oracle' in result.stderr
