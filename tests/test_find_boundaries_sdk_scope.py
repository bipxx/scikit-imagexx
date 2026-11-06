"""Source-only SDK extraction contract; no installed acceptance claim."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_find_boundaries_sdk_source_closure():
    c=json.loads((ROOT/'coverage/contracts/find-boundaries-sdk-v3.json').read_bytes())
    frow=c['f_row_successor'];fraw=(ROOT/frow['contract']).read_bytes()
    assert hashlib.sha256(fraw).hexdigest()==frow['sha256']
    fold=json.loads(fraw)
    assert {k:v for k,v in c.items()if k not in ('source_canonical_pins','f_row_successor')}=={k:v for k,v in fold.items()if k!='source_canonical_pins'}
    binding=c['layout_successor'];raw=(ROOT/binding['contract']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==binding['sha256']
    previous=json.loads(raw)
    assert {k:v for k,v in c.items()if k not in ('source_canonical_pins','layout_successor','f_row_successor')}=={k:v for k,v in previous.items()if k!='source_canonical_pins'}
    assert set(c['source_canonical_pins'])==set(previous['source_canonical_pins'])
    import runpy
    driver=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'))
    for name,pin in c['source_canonical_pins'].items():
        raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        if name in ('src/find_boundaries.cpp','tests/cmake_consumer/find_boundaries_consumer.hpp'):raw=driver['layout_predecessor'](name,raw)
        assert hashlib.sha256(raw).hexdigest()==previous['source_canonical_pins'][name]

        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin,name
    assert c['installed_target']=='skimagexx::skimagexx'
    assert set(c['required_sources'])=={'tests/cmake_consumer/find_boundaries_consumer.hpp','tests/fixtures/find_boundaries_originals_v1.hpp'}
    assert c['proofs']['normal']==c['proofs']['runtime_disabled']==c['proofs']['compiler_scalar']==c['proofs']['sanitize']==532320
    assert c['proofs']['budget1']==c['proofs']['budget4']==266160
    assert c['remaining'] and 'ASSEMBLED' in c['status']


def test_find_boundaries_sdk_helper_owns_saved_rows_and_workers():
    text=(ROOT/'tests/cmake_consumer/find_boundaries_consumer.hpp').read_text()
    for token in ('#include <skimagexx/find_boundaries.hpp>','../fixtures/find_boundaries_originals_v1.hpp','find_boundaries_originals::rows()','get_num_threads()==workers','set_num_threads(workers)','find_boundaries_into(view,row.parameters','check(input==before)','height=513,width=517','~Restore(){set_num_threads(budget);}'):
        assert token in text,token
