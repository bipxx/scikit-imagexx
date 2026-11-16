"""Narrow inert SDK source guards; C++ compilation/execution remain separate."""
import hashlib
from pathlib import Path
import re

import pytest
from vi_public_matmul_source_view import _vi_2270_source_epoch

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'tests/variation_of_information_provider_controls.hpp'
CONSUMER = ROOT / 'tests/cmake_consumer/variation_of_information_enabled_consumer.cpp'
CMAKE = ROOT / 'tests/cmake_consumer/CMakeLists.txt'
FROZEN = {
    'include/skimagexx/variation_of_information.hpp': 'c6b00c5c7fedcd8d9467a9a69262d59e63e32b60242ee9760487bad8a32229d8',
    'src/variation_of_information.cpp': '11a379e1f84531d113fec594b59aadd52e08835d421056c02fd39ff480778535',
    'src/bind_variation_of_information.cpp': '7c843713f3388781d953f18104f8f1ec87a5e662fe555b29af5d6c9f89e49311',
    'python/skimagexx/_variation_of_information.py': 'ab5c17e2a8b4aee353c28269dff9d624bf111a7d71226425258251a210c89f15',
    'tests/variation_of_information_current_counts_controls.hpp': '6035836b02c90a611e76818f4dc195894ca3366e520659cd53d9aea30e8584eb',
    'tests/variation_of_information_large_controls.hpp': '12e8b75e8d01e7c1d04b826e94ea67533ae5966ae3796f319d114f432ced6407',
    'tests/fixtures/variation_of_information_saved_v1.hpp': 'dfd40555fa87a96e7b6d224553dabc5c2b7522ff295072b9c4975730bfd98ec0',
    'coverage/contracts/vi-public-log2-provider-v1.json': '4f30e29469dcaa72594c0482a6a04954f04582b2593a0521b0df2a9b71e46096',
}
digest = lambda b: hashlib.sha256(b).hexdigest()


def remove_block(text, name, comment='//'):
    pattern = r'^[ ]*' + comment + ' ' + name + r'_BEGIN\n.*?^[ ]*' + comment + ' ' + name + r'_END\n'
    result, count = re.subn(pattern, '', text, count=1, flags=re.M | re.S)
    assert count == 1
    return result


@pytest.mark.parametrize('path,pin', tuple(FROZEN.items()))
def test_production_old_controls_golds_and_v1_context_stay_frozen(path, pin):
    assert digest((ROOT / path).read_bytes()) == pin


def test_exact_existing_owning_consumer_is_preserved_once():
    source = CONSUMER.read_text(encoding='utf-8')
    source = source.replace('// VI_SDK_EXTENSION_INCLUDE\n#include "../variation_of_information_provider_controls.hpp"\n', '', 1)
    source = source.replace('int vi_sdk_owning_main() {', 'int main() {', 1)
    for name in ('VI_SDK_EXTENSION_ENTRY', 'VI_SDK_EXTENSION_TAIL', 'VI_SDK_EXTENSION_MAIN'):
        source = remove_block(source, name)
    assert source.endswith('}\n\n')
    source = source[:-1]  # Only the declared separator before appended main.
    assert digest(source.encode('utf-8')) == 'a690e5c6836f5b1d2d5c8da1826ecd97875fa6efa00386f175290bb8f01f63ef'


def test_CMake_delta_is_only_same_target_F1_link_and_platform_definition():
    source = CMAKE.read_text(encoding='utf-8')
    block = source[source.index('# VI_SDK_FENV32_BEGIN'):source.index('# VI_SDK_FENV32_END')]
    assert 'CMAKE_SYSTEM_NAME STREQUAL "Linux"' in block
    assert 'CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$"' in block
    assert 'if(NOT EXISTS "${SKIMAGEXX_FENV_HELPER}")' in block
    assert 'target_link_libraries(variation_of_information_consumer PRIVATE "${SKIMAGEXX_FENV_HELPER}")' in block
    assert 'target_compile_definitions(variation_of_information_consumer PRIVATE SKIMAGEXX_VI_FENV32=1)' in block
    assert digest(remove_block(source, 'VI_SDK_FENV32', '#').encode('utf-8')) == 'f6182e059eaf2b046133b29d0b57a8dab022569cdd387c86ac4d4e10fa8b13f0'


def test_compatible_large_words_and_fixture_recipes_reuse_frozen_data():
    source = HEADER.read_text(encoding='utf-8')
    large = (ROOT / 'tests/variation_of_information_large_controls.hpp').read_text(encoding='utf-8')
    counts = (ROOT / 'tests/variation_of_information_current_counts_controls.hpp').read_text(encoding='utf-8')
    words = r'UINT64_C\(0x[0-9a-f]{16}\)'
    assert re.findall(words, source) == re.findall(words, large)
    for recipe in ('n=c<2?16384:65536,labels=c%2?64:8',
                   'truth[i]=i%labels;test[i]=(i/3+i%7)%labels;'):
        assert recipe in source and recipe in large
    for recipe in ('n:{4096u,4099u}', 'ignored_mode:{0,1,2}',
                   'a[i]=(i/7)%4;b[i]=(i/3)%8;wide_a[i]=a[i]+64;wide_b[i]=b[i]+64;'):
        assert recipe in source and recipe in counts
    compatible = source[source.index('inline int compatible_counts()'):source.index('template<class Input>int failure_cases')]
    assert 'skimagexx::variation_of_information(' not in compatible  # No repeated old default suite.


def test_failure_contract_checks_full_fields_thread_and_primary_identity():
    source = HEADER.read_text(encoding='utf-8')
    same = source[source.index('inline bool same('):source.index('inline VariationInformation sentinel')]
    assert all(field in same for field in ('values.data()', 'normalization_fp_status', 'captured_workers'))
    failure = source[source.index('template<class Input>int failure_cases'):source.index('inline int array_boundaries')]
    assert 'invoke(input,out,{},1)||!same(out,before)' in failure
    assert 'throwing:{false,true}' in failure and 'phase:{1u,2u}' in failure
    assert 'error.identity!=&audit' in failure and 'audit.calls!=phase' in failure
    assert 'caught!=throwing' in failure and '!same(out,before)' in failure
    callback = source[source.index('inline bool log2_batch'):source.index('template<class Input>bool invoke')]
    assert callback.index('std::this_thread::get_id()!=a.caller') < callback.index('std::log2(values[i])')
    assert 'throw ProviderFailure(&a)' in callback


def test_refusal_control_requires_unchanged_entered_full32_before_group_restore():
    source = HEADER.read_text(encoding='utf-8')
    refusal = source[source.index('inline int fenv_refusals'):source.index('inline int run()')]
    assert 'compatible:{false,true}' in refusal and 'use_image:{false,true}' in refusal
    assert 'mode<5' in refusal
    assert 'std::memcmp(&entered,&after,32)' in refusal
    assert refusal.index('std::memcmp(&entered,&after,32)') < refusal.index('caller.finish()')
    assert 'FE_DOWNWARD' in refusal and '(1u<<15)' in refusal and '(1u<<6)' in refusal
    assert 'caller.mxcsr()&(1u<<2)' in refusal and 'caller.status()&FE_DIVBYZERO' in refusal
    assert 'feclearexcept' not in source and 'feraiseexcept' not in source
    assert 'reference_fenv_restore(&before_,32)' in source and 'std::memcmp(&before_,&after,32)' in source
    assert 'UNASSESSED_PLATFORM' in refusal
