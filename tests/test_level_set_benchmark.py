"""Frozen workload and exact precheck guards; no scientific imports or timing."""
import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'benchmarks/level_set.py'


def load():
    spec=importlib.util.spec_from_file_location('level_set_benchmark_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_exact_eight_frozen_cases_import_without_scientific_dependencies(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0] not in ('numpy','scipy','skimage','skimagexx')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    module=load();rows=module.case_specs()
    contract=json.loads((ROOT/'coverage/contracts/level-set.json').read_bytes())
    assert [r['case_id'] for r in rows]==contract['future_benchmark']['cases']
    assert len(rows)==len({r['case_id'] for r in rows})==8
    assert {(r['api'],tuple(r['shape'])) for r in rows}=={
        ('segmentation.'+api,shape) for api in ('checkerboard_level_set','disk_level_set')
        for shape in ((256,256),(1024,1024),(64,64,64),(128,64,32))}
    specs={s['id']:s for s in contract['cases']}
    assert all(specs[r['case_id']]['benchmark'] and not specs[r['case_id']]['options'] for r in rows)
    for version,observations in contract['original_observations'].items():
        frozen={row['id']:row for row in observations}
        assert all(frozen[r['case_id']]['result']['dtype']=='|i1' and frozen[r['case_id']]['error'] is None
                   and not frozen[r['case_id']]['warnings'] for r in rows)


class ndarray:
    __module__='numpy'
    def __init__(self,words=b'\x00\x01\x00\x01\x01\x00'):
        self.words=words;self.shape=(2,3);self.strides=(3,1);self.base=None
        self.dtype=SimpleNamespace(str='|i1',metadata=None)
        self.flags=SimpleNamespace(owndata=True,c_contiguous=True,f_contiguous=False,writeable=True)
    def tobytes(self):return self.words


def frozen_output():
    return dict(type='numpy.ndarray',dtype='|i1',dtype_metadata=None,shape=[2,3],strides=[3,1],
        own=True,c=True,f=False,writeable=True,base_type='NoneType',
        raw_sha256=hashlib.sha256(b'\x00\x01\x00\x01\x01\x00').hexdigest())


def test_full_words_and_metadata_checked_then_only_hashes_exported():
    result=load().checked_outputs(ndarray(),ndarray(),frozen_output(),SimpleNamespace(ndarray=ndarray))
    assert result['output_sha256']==result['reference_sha256']==frozen_output()['raw_sha256']
    assert result['output_metadata']=={k:v for k,v in frozen_output().items() if k!='raw_sha256'}
    assert not any(isinstance(v,bytes) for v in result.values())


@pytest.mark.parametrize('fault',('words','dtype','metadata','shape','strides','own','c','f','writeable','base','frozen-hash'))
def test_precheck_refuses_each_storage_or_word_drift(fault):
    actual=ndarray();frozen=frozen_output()
    if fault=='words':actual.words=b'\x01'*6
    elif fault=='dtype':actual.dtype.str='|u1'
    elif fault=='metadata':actual.dtype.metadata={'changed':True}
    elif fault=='shape':actual.shape=(3,2)
    elif fault=='strides':actual.strides=(1,2)
    elif fault in ('own','c','f','writeable'):
        name={'own':'owndata','c':'c_contiguous','f':'f_contiguous','writeable':'writeable'}[fault]
        setattr(actual.flags,name,not getattr(actual.flags,name))
    elif fault=='base':actual.base=object()
    else:frozen['raw_sha256']='0'*64
    with pytest.raises(AssertionError):load().checked_outputs(actual,ndarray(),frozen,SimpleNamespace(ndarray=ndarray))


def test_native_observation_and_exact_precheck_complete_before_timers():
    tree=ast.parse(SOURCE.read_bytes());text=ast.unparse(tree)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    timer=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='measure')
    precheck=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checked_outputs')
    restore=next(n for n in ast.walk(main) if isinstance(n,ast.Try) and any('sys.setprofile(prior)' in ast.unparse(f) for f in n.finalbody))
    assert restore.end_lineno<precheck.lineno<timer.lineno
    assert "returns == [entry] and accepted == [True]" in text
    assert "frame.f_locals.get('native_result') is not None" in text
    assert "frame.f_code is reference_code" in text and 'forwarded to the original whole API' in text
    assert [ast.unparse(a) for a in timer.args[2:]]==['args.repeat','19']
    assert 'source_after == source_before' in text and 'CONTRACT.read_bytes() == contract_bytes' in text


def test_all_samples_losses_one_worker_and_provider_scope_retained():
    tree=ast.parse(SOURCE.read_bytes());text=SOURCE.read_text()
    keywords={n.arg for n in ast.walk(tree) if isinstance(n,ast.keyword)}
    assert {'output_metadata','output_sha256','reference_sha256','frozen_reference_sha256','actual_C_returns',
            'native_accepted','harness_sha256','contract_sha256','source_before','core_sha256',
            'reference_pool_limits','provider_frontier'}<=keywords
    assert 'raw_samples=112' in text and 'repeat=7,target_ms=19' in text
    assert "choices=[1]" in text and "choices=[7]" in text and "set(limits.values())=={'1'}" in text
    assert 'output destruction' in text and 'All raw samples and losses' in text
    assert 'raw_hex' not in text and 'raw_zlib_base64=' not in text
    assert 'for row in rows:' in text and "timing['speedup']" in text
