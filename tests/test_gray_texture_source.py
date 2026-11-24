"""Pure component/source guards: no scientific import or native call."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=json.loads((ROOT/'coverage/contracts/gray-texture-native-v1.json').read_bytes())

@pytest.mark.parametrize('path',list(CONTRACT['files'])+list(CONTRACT['dependencies']))
def test_exact_component_and_dependency_bytes(path):
    expected=CONTRACT['files'].get(path,CONTRACT['dependencies'].get(path))
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected


def test_saved_observations_remain_full_and_errors_unmodified():
    data=json.loads((ROOT/'tests/fixtures/gray_texture_original_v1.json').read_bytes())
    for profile in ('minimum','modern'):
        rows=data['profiles'][profile]
        assert len(rows)==len({r['id']for r in rows})==94
        assert sum(r['error']is None for r in rows)==64
        assert sum(r['error']is not None for r in rows)==30
        assert all(r['supplied_call_attempts']==1 and r['caller_restore']['exact32']for r in rows)
        assert all('qualname'in r['error']for r in rows if r['error'])
    assert CONTRACT['saved']['SDK_excluded']==['graycoprops/type-object']


def test_native_ownership_and_policy_boundaries_are_explicit():
    py=ast.parse((ROOT/'python/skimagexx/_gray_texture.py').read_text(encoding='utf-8-sig'))
    imports=[n.module for n in ast.walk(py)if isinstance(n,ast.ImportFrom)]
    assert not any(n and n.startswith('skimage.')for n in imports)
    source=(ROOT/'src/gray_texture.cpp').read_text()
    assert 'log@GLIBC_2.2.5'in source and 'std::log(x)'in source
    assert 'if(property==GrayProperty::invalid)'in source
    assert source.index('P=transform("divide"')<source.index('if(property==GrayProperty::invalid)')
    assert 'std::isnan'not in source and 'quiet_NaN'not in source
    assert 'fetestexcept(flags))std::feclearexcept(flags)'in source
    assert 'get_num_threads()'in source and 'detail::partition'in source
    assert 'NumPy'not in (ROOT/'include/skimagexx/gray_texture.hpp').read_text()


def test_complete_named_sdk_and_public_regressions_are_exercised():
    text=(ROOT/'tests/test_gray_texture.cpp').read_text(encoding='utf-8-sig')
    assert 'graycomatrix(GrayImageView'in text and 'graycoprops(GrayProbabilityView'in text
    assert 'for(int workers:{0,1,4})'in text
    assert 'words(out.values,c.runs)'in text and 'backing.v==old'in text
    public=(ROOT/'tests/test_gray_texture_public.py').read_text(encoding='utf-8-sig')
    assert 'actual=={k:row[k]for k in actual}'in public
    assert "test_normalization_precedes_invalid_property"in public
    assert "'target_immediate'"in public


def test_benchmark_cases_method_and_authority_stay_prospective():
    module=runpy.run_path(str(ROOT/'benchmarks/gray_texture_native.py'))
    assert module['cases']()==CONTRACT['benchmark']['cases']
    assert len(module['cases']())==8
    assert module['METHOD']==CONTRACT['benchmark']['method']
    assert CONTRACT['benchmark']['raw_samples']==8*2*7
    assert CONTRACT['benchmark']['warmup_groups']==8*2
    assert all(v is None for v in CONTRACT['authority'].values())
    with pytest.raises(SystemExit,match='UNBOUND'):module['main']()
    assert len(CONTRACT['frontiers'])>=7
