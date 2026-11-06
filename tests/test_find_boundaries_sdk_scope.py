"""Source-only SDK extraction contract; no installed acceptance claim."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_find_boundaries_sdk_source_closure():
    c=json.loads((ROOT/'coverage/contracts/find-boundaries-sdk-v1.json').read_bytes())
    for name,pin in c['source_canonical_pins'].items():
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
