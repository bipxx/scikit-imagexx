"""Small source/dependency checks only; no scientific imports or target calls."""
import hashlib
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]


def test_exact_source_dependency_closure():
    contract=json.loads((ROOT/'coverage/contracts/join-segmentations-native-v2.json').read_bytes())
    assert contract['api']=='segmentation.join_segmentations' and contract['API_completion'] is False
    context=json.loads((ROOT/'coverage/contracts/next-util-benchmark-execution-v3.json').read_bytes())
    projection=runpy.run_path(str(ROOT/'benchmarks/next_util_execution.py'))['historical_bytes']
    for group in ('source_raw_sha256','reused_raw_sha256'):
        for name,pin in contract[group].items():
            # Exact admitted inverse authenticates current bytes before restoring historical benchmark bytes.
            raw=projection(ROOT,name)if name in context['sources']and name!='python/skimagexx/_join_segmentations.py'else(ROOT/name).read_bytes()
            if name=='python/skimagexx/_join_segmentations.py':
                body=(ROOT/'coverage/contracts/join-segmentations-unique-reuse-v1.json').read_bytes()
                assert hashlib.sha256(body).hexdigest()=='137cfcd9341a809cafa216701fa020f871fe8f4bb03dbf54f2241db2e2ba3df2'
                reuse=json.loads(body);assert hashlib.sha256(raw).hexdigest()==reuse['current_raw_sha256']
                current,old=reuse['current'].encode(),reuse['predecessor'].encode();assert raw.count(current)==1
                raw=raw.replace(current,old,1);assert hashlib.sha256(raw).hexdigest()==reuse['predecessor_raw_sha256']
            if name=='tests/test_join_segmentations_public.py':
                envelope=(ROOT/'coverage/contracts/join-segmentations-captured-fenv-input-v1.json').read_bytes()
                assert hashlib.sha256(envelope).hexdigest()=='1af50e744744268b18baca3552d593970d43aa82536cb0db9e84e5ee48c3da18'
                record=json.loads(envelope);assert hashlib.sha256(raw).hexdigest()==record['current_raw_sha256']
                text=raw.decode()
                for before,after in reversed(record['inverse_replacements']):
                    assert text.count(after)==1
                    text=text.replace(after,before,1)
                raw=text.encode();assert hashlib.sha256(raw).hexdigest()==record['predecessor_raw_sha256']
                fix=json.loads((ROOT/'coverage/contracts/join-windows-fenv-fixture-v1.json').read_bytes())
                assert hashlib.sha256(raw).hexdigest()==fix['current_raw_sha256']
                text=raw.decode()
                for before,after in reversed(fix['inverse_replacements']):
                    assert text.count(after)==1
                    text=text.replace(after,before)
                raw=text.encode()
                assert hashlib.sha256(raw).hexdigest()==fix['predecessor_raw_sha256']
            assert hashlib.sha256(raw).hexdigest()==pin,name
    assert 'tests/fixtures/join_segmentations_originals_v1.hpp' in contract['source_raw_sha256']
    assert 'tests/fixtures/join_segmentations_originals_v1.json' in contract['reused_raw_sha256']


def test_original_packet_preserved():
    fixture=json.loads((ROOT/'tests/fixtures/join_segmentations_originals_v1.json').read_bytes())
    for name,pin in fixture['frozen_source_raw_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    assert fixture['actual_public_topcalls']==100 and fixture['native_calls']==fixture['timed_batches']==0


def test_mapping_unique_reuse_is_exact_one_fragment_no_kernel_or_benchmark_change():
    reuse=json.loads((ROOT/'coverage/contracts/join-segmentations-unique-reuse-v1.json').read_bytes())
    assert reuse['original_budget']==0 and reuse['native_source_unchanged']and reuse['SDK_source_unchanged']
    assert hashlib.sha256((ROOT/'benchmarks/join_segmentations.py').read_bytes()).hexdigest()==reuse['benchmark_driver_raw_sha256']
    assert hashlib.sha256((ROOT/reuse['benchmark_case_contract']).read_bytes()).hexdigest()==reuse['benchmark_contract_raw_sha256']
    source=(ROOT/reuse['frontend']).read_text()
    assert source.count('labels_j=map_j_to_j_initial.out_values')==1
    assert source.index('if not return_mapping:return j')<source.index('labels_j=map_j_to_j_initial.out_values')
