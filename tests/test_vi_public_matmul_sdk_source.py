"""Current installed-SDK source guards; never select a historical ROOT.

Root must integrate the owning target/extractor/collector/history/selector first.
These are inert source guards: no C++ compilation or SDK/provider/F1 body runs.
The frozen legacy guard module remains independent in its2270 source epoch.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import re
import shlex
import unittest

import pytest

ROOT = Path(__file__).resolve().parents[1]
THIS_GUARD = 'tests/test_vi_public_matmul_sdk_source.py'
CMAKE = 'tests/cmake_consumer/CMakeLists.txt'
QUALIFIER = 'tools/public_qualification.py'
HOOK = 'tools/mark_lbp_fp_context.py'
WORKFLOW = '.github/workflows/public-qualification.yml'
WRAPPER = 'tests/cmake_consumer/variation_of_information_matmul_consumer_v1.cpp'
ADDITIVE_HEADER = 'tests/variation_of_information_matmul_controls_v1.hpp'
PREDECESSOR = '2270b7fefc9a8e5cb3ba7fe51d2db5759c17634d'
FROZEN = {'tests/cmake_consumer/variation_of_information_enabled_consumer.cpp': 'a3b39369cfeed7fdc4372b3aa7afa26e24cfa862396bb8541a7dbb4a6ee48ef1', 'tests/variation_of_information_provider_controls.hpp': '3479d7490e2d85a8f70e721ecd8d204ccd1369b3949f733bd6d83cf49b566fd2', 'tests/variation_of_information_current_counts_controls.hpp': '6035836b02c90a611e76818f4dc195894ca3366e520659cd53d9aea30e8584eb', 'tests/variation_of_information_large_controls.hpp': '12e8b75e8d01e7c1d04b826e94ea67533ae5966ae3796f319d114f432ced6407', 'coverage/contracts/vi-public-log2-provider-v1.json': '4f30e29469dcaa72594c0482a6a04954f04582b2593a0521b0df2a9b71e46096', 'tests/fixtures/variation_of_information_saved_v1.hpp': 'dfd40555fa87a96e7b6d224553dabc5c2b7522ff295072b9c4975730bfd98ec0'}
CURRENT_PRODUCTION = {'include/skimagexx/variation_of_information.hpp': 'c910703f0b5315dc4cb6751c7b6d7eb90c7a9442f81b440da55bbc10a8658bb4', 'src/variation_of_information.cpp': 'e4131ef47cdb66af20f3a7adcc29d8f95eeacafeb1e89f652d21bb29a661b2e9', 'src/bind_variation_of_information.cpp': '8d3310debba4a10aa59f185fe938b2db48f79fb0ed5afafac081078578dc83a3', 'python/skimagexx/_variation_of_information.py': '8e25cd3f76ecc32e9441e42c0bff4c93817e5b02c1e46f147f04829113abcd0c'}
ADDITIVE_SDK = {'tests/cmake_consumer/variation_of_information_matmul_consumer_v1.cpp': '4e637833781ebb5a8f9aaf6a45598048078101387d7e7cc8365b79f6cdaa8df5', 'tests/variation_of_information_matmul_controls_v1.hpp': 'c99531f32f2e52dee3cc67d033d79b915a10fd6d1cedf08c1f458e458735cf89'}
OLD_API_PREFIXES = {'src/variation_of_information.cpp': (12161, '77f6c20d6920710b6dc1287cb23d31ef959a6b281f047a0d2f8bc6adf045d2d3'), 'include/skimagexx/variation_of_information.hpp': (2827, 'e55e9570f914acb088088b92bf9e352e569cd89d2e0f98cf6f485be7cec55152')}

H = (ROOT / ADDITIVE_HEADER).read_text(encoding='utf-8')
W = (ROOT / WRAPPER).read_text(encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def missing(text):
    # Local source guards for concrete callbacks/refusal/restoration boundaries.
    required = {
        'order': "kind!=(next%2?'L':'D')",
        'owned_input': 'record.left.assign(a,a+n)',
        'owned_right': 'record.right.assign(b,b+n)',
        'thread': 'std::this_thread::get_id()!=audit.caller',
        'context': "&context!=(kind=='L'?&audit.logarithm:&audit.inner)",
        'inner_argument': 'VariationInnerProduct{null_dot?nullptr:inner,&audit.inner}',
        'four_phases': 'for(unsigned phase:{1u,2u,3u,4u})',
        'false_and_throw': 'for(bool throwing:{false,true})',
        'no_later_callback': 'a.records.size()!=phase',
        'primary_identity': 'error.identity!=&a',
        'refusal32': 'std::memcmp(&entered,&after,32)',
        'effect32': 'std::memcmp(&a.callback_effect,&after,32)',
        'explicit_worker': 'out.captured_workers!=1',
    }
    problems = {name for name, term in required.items() if term not in text}
    if text.count('!same(out,before)') < 5:
        problems.add('all_named_output_fields')
    return problems


class SourceControls(unittest.TestCase):
    def test_01_exact_draft_and_frozen_source(self):
        for name, pin in {**FROZEN, **CURRENT_PRODUCTION, **ADDITIVE_SDK}.items():
            raw = (ROOT / name).read_bytes()
            if name in ('tests/cmake_consumer/variation_of_information_enabled_consumer.cpp', 'tests/variation_of_information_current_counts_controls.hpp', 'tests/variation_of_information_matmul_controls_v1.hpp', 'tests/variation_of_information_provider_controls.hpp'):
                from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
                from tools.vi_public_matmul_context import immediate_predecessor_bytes
                raw = immediate_predecessor_bytes(ROOT, name, raw, VI_PUBLIC_MATMUL_CONTEXT_SHA256)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), pin, name)
    def test_02_old_whole_native_and_header_prefixes(self):
        for name, (length, pin) in OLD_API_PREFIXES.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()[:length]).hexdigest(), pin)
    def test_03_frozen_owning_group_once_then_additive(self):
        self.assertIn('#define main vi_frozen_sdk_main_v1', W)
        self.assertIn('#include "variation_of_information_enabled_consumer.cpp"', W)
        self.assertEqual(W.count('vi_frozen_sdk_main_v1()'), 1)
        self.assertLess(W.index('vi_frozen_sdk_main_v1()'), W.index('vi_matmul_sdk_controls::run()'))
        self.assertIn('if(int status=vi_frozen_sdk_main_v1())return status;', W)

    def test_04_actual_signature(self):
        header = (ROOT / 'include/skimagexx/variation_of_information.hpp').read_text(encoding='utf-8')
        self.assertIn('using Function=bool(*)(const double*,const double*,std::size_t,double*,void*);', header)
        self.assertIn('VariationLog2Batch,VariationInnerProduct,int workers=0);', header)
        self.assertIn('static_assert(std::is_same_v<VariationInnerProduct::Function,', H)

    def test_05_complete_contract_boundaries(self):
        self.assertEqual(missing(H), set())

    def test_06_contexts_and_copies_are_owned(self):
        self.assertIn("Context logarithm{this,'L'},inner{this,'D'};", H)
        self.assertIn('char kind;std::vector<double> left,right,output;', H)
        self.assertIn('Audit(const Audit&)=delete', H)
        self.assertIn('if(n){record.left.assign', H)

    def test_07_named_output_sentinel_is_frozen(self):
        text = (ROOT / 'tests/variation_of_information_provider_controls.hpp').read_text(encoding='utf-8')
        self.assertIn('std::memcmp(a.values.data(),b.values.data(),16)==0', text)
        self.assertIn('a.normalization_fp_status==b.normalization_fp_status', text)
        self.assertIn('a.captured_workers==b.captured_workers', text)
        self.assertIn('out.normalization_fp_status=0x1234u;out.captured_workers=91;', text)
    def test_08_worker_capture_and_host_effect(self):
        from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
        from tools.vi_public_matmul_context import immediate_predecessor_bytes
        raw = (ROOT / ADDITIVE_HEADER).read_bytes()
        historical = immediate_predecessor_bytes(ROOT, ADDITIVE_HEADER, raw, VI_PUBLIC_MATMUL_CONTEXT_SHA256, True).decode()
        self.assertIn('for(int workers:{1,4,0})', historical)
        self.assertIn('skimagexx::set_num_threads(4);Audit a;a.change_global=true', historical)
        self.assertIn('if(next==1&&audit.change_global)skimagexx::set_num_threads(2)', historical)
        self.assertIn('skimagexx::get_num_threads()!=2', historical)
        self.assertIn('out.captured_workers!=captured', historical)
        for control in ('extra_budget=4', 'extra_budget=1', 'after_callback=2', 'after_callback=1',
                        'out.captured_workers!=captured', 'get_num_threads()!=after_callback'):
            self.assertIn(control, H)

    def test_09_both_entry_failure_groups(self):
        self.assertIn('failures(table)', H)
        self.assertIn('failures(image)', H)
        self.assertIn('for(bool null_log:{true,false})', H)
        self.assertIn('null_log,!null_log', H)

    def test_10_duplicate_zero_empty_and_logical_vectors(self):
        for term in ('{.25,.75,.375,.625}', '{.25,.375,.75,.625}', '{-1.,-2.}', '{1.,2.}',
                     'VariationTableView{nullptr,nullptr,empty_rows,2,2,0}', 'const double zeros[]={0.,-0.,0.,-0.}',
                     'which==2?66:2,which==2?64:0', 'std::memcmp(result.values.data(),words,16)'):
            self.assertIn(term, H)

    def test_11_inputs_preserved(self):
        for term in ('truth!=original_truth', 'wide_truth!=original_wide_truth',
                     'std::memcmp(columns,original_columns.data(),sizeof(columns))',
                     'std::memcmp(rows,original_rows.data(),sizeof(rows))',
                     'std::memcmp(data,original_data.data(),sizeof(data))'):
            self.assertIn(term, H)

    def test_12_F1_platform_and_no_normalization(self):
        for term in ('#if defined(SKIMAGEXX_VI_FENV32)', 'for(unsigned mode=0;mode<5;++mode)',
                     '(1u<<15)', '(1u<<6)', '~(1u<<9)', 'fldcw %0',
                     'UNASSESSED_PLATFORM', 'if(!process.finish()', 'callback_effect'):
            self.assertIn(term, H)
        for forbidden in ('FE_DFL_ENV', 'feclearexcept', 'fesetenv(', 'dlopen', 'NPY_', 'numpy'):
            self.assertNotIn(forbidden, H)

    def test_13_source_envelope_not_actual_runtime(self):
        self.assertIn('unsigned expected_calls=39;', H)
        self.assertIn('expected_calls+=10;', H)
        self.assertIn('totals.callbacks!=92||totals.logs!=50||totals.dots!=42', H)
        self.assertIn('after.checked32-before_counters.checked32!=29', H)
        self.assertIn('after.reads32-before_counters.reads32!=105', H)
        self.assertLess(H.index('const auto before_counters='), H.index('Caller32 process('))

    def test_14_scalar_provider_disclosure(self):
        self.assertIn('not NumPy/BLAS or original-public parity', H)
        self.assertIn('output[i]=std::log2(input[i])', H)
        self.assertIn('result+=left[i]*right[i]', H)
        self.assertIn('provider=scalar-libm-linear-dot', H)
        self.assertNotIn('std::fma(', H)

    def test_15_refuse_missing_fourth_failure_phase(self):
        self.assertIn('four_phases', missing(H.replace('for(unsigned phase:{1u,2u,3u,4u})', 'for(unsigned phase:{1u,2u,3u})')))

    def test_16_refuse_missing_inner_function(self):
        self.assertIn('inner_argument', missing(H.replace('VariationInnerProduct{null_dot?nullptr:inner,&audit.inner}', 'VariationInnerProduct{}')))

    def test_17_refuse_missing_output_guard(self):
        self.assertIn('all_named_output_fields', missing(H.replace('!same(out,before)', 'false')))

    def test_18_refuse_unbounded_later_callback(self):
        self.assertIn('no_later_callback', missing(H.replace('a.records.size()!=phase', 'false')))

    def test_19_refuse_missing_refusal_raw32(self):
        self.assertIn('refusal32', missing(H.replace('std::memcmp(&entered,&after,32)', '0')))

    def test_20_refuse_missing_primary_identity(self):
        self.assertIn('primary_identity', missing(H.replace('error.identity!=&a', 'false')))

    def test_21_refuse_reversed_callback_order(self):
        self.assertIn('order', missing(H.replace("kind!=(next%2?'L':'D')", "kind!=(next%2?'D':'L')")))

    def test_22_refuse_missing_input_copy(self):
        self.assertIn('owned_input', missing(H.replace('record.left.assign(a,a+n)', 'record.left.clear()')))

    def test_23_refuse_missing_explicit_one(self):
        self.assertIn('explicit_worker', missing(H.replace('out.captured_workers!=1', 'false')))

    def test_24_refuse_missing_callback_effect_raw32(self):
        self.assertIn('effect32', missing(H.replace('std::memcmp(&a.callback_effect,&after,32)', '0')))



def _load_source(name):
    spec = importlib.util.spec_from_file_location('_current_vi_sdk_' + Path(name).stem, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal(tree, name):
    rows = [node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    assert len(rows) == 1, name
    return ast.literal_eval(rows[0])


def _current_history():
    """Reuse root's authenticated earliest source layer on actual current bytes."""
    hook = ast.parse((ROOT / HOOK).read_bytes())
    context_pin = _literal(hook, 'VI_PUBLIC_MATMUL_CONTEXT_SHA256')
    helper_pin = _literal(hook, 'VI_PUBLIC_MATMUL_HELPER_SHA256')
    boundary = next(node for node in hook.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'source_successor_predecessor_bytes')
    imports = [node for node in ast.walk(boundary) if isinstance(node, ast.ImportFrom)
               and node.module.startswith('tools.vi_public_matmul_context')]
    assert len(imports) == 1
    helper_name = imports[0].module.replace('.', '/') + '.py'
    assert sha(ROOT / helper_name) == helper_pin
    helper_tree = ast.parse((ROOT / helper_name).read_bytes())
    context_name = _literal(helper_tree, 'CONTEXT')
    assert context_name.startswith('coverage/contracts/vi-public-matmul-source-history-') and context_name.endswith('.json')
    assert sha(ROOT / context_name) == context_pin
    data = json.loads((ROOT / context_name).read_bytes())
    assert data['helper_sha256'] == helper_pin and data['predecessor_source'] == PREDECESSOR
    assert data['execution_authority'] is None
    # These changed shared files must be authenticated, not pass through the layer.
    assert {CMAKE, QUALIFIER, HOOK, WORKFLOW} <= set(data['sources'])
    body = ast.get_source_segment((ROOT / HOOK).read_text(encoding='utf-8'), boundary)
    assert body.index('raw=matmul_predecessor(') < body.index('raw=current_validation_predecessor(')
    helper = _load_source(helper_name)
    assert helper._context(ROOT, context_pin) == data
    return helper, data, context_pin, helper_name, context_name


def _extractor_predicate():
    _current_history()
    sdk = next(node for node in ast.parse((ROOT / QUALIFIER).read_bytes()).body
               if isinstance(node, ast.FunctionDef) and node.name == 'sdk')
    predicate = next(node.test for node in ast.walk(sdk) if isinstance(node, ast.If)
                     and 'tests/fixtures/bezier_curve_originals_v1.bin' in ast.unparse(node.test))
    return compile(ast.Expression(predicate), '<actual current SDK extractor>', 'eval')


def _selectors(raw):
    lines = [line.strip() for line in raw.decode('utf-8').splitlines()
             if line.strip().startswith('python -m pytest -q --capture=sys')]
    assert len(lines) == 1
    args = shlex.split(lines[0])
    return [arg for arg in args[:args.index('|')] if arg.startswith('tests/')]


def test_current_ROOT_is_the_module_checkout_and_has_no_epoch_fixture():
    assert ROOT == Path(__file__).resolve().parents[1]
    assert (ROOT / THIS_GUARD).resolve() == Path(__file__).resolve()
    tree = ast.parse(Path(__file__).read_bytes())
    assert not any(isinstance(node, ast.ImportFrom) and node.module == 'vi_public_matmul_source_view'
                   for node in ast.walk(tree))


def test_current_owning_target_selects_the_successor_once_with_existing_F1_and_flags():
    _current_history()
    text = (ROOT / CMAKE).read_text(encoding='utf-8')
    assert re.findall(r'add_executable\(variation_of_information_consumer\s+([^)]*)\)', text) == [
        'variation_of_information_matmul_consumer_v1.cpp']
    assert text.count('add_test(NAME installed_variation_of_information_consumer COMMAND variation_of_information_consumer)') == 1
    assert 'target_link_libraries(variation_of_information_consumer PRIVATE skimagexx::skimagexx)' in text
    assert 'target_include_directories(variation_of_information_consumer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")' in text
    block = text[text.index('# VI_SDK_FENV32_BEGIN'):text.index('# VI_SDK_FENV32_END')]
    assert 'CMAKE_SYSTEM_NAME STREQUAL "Linux"' in block and 'CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$"' in block
    assert 'target_link_libraries(variation_of_information_consumer PRIVATE "${SKIMAGEXX_FENV_HELPER}")' in block
    assert 'target_compile_definitions(variation_of_information_consumer PRIVATE SKIMAGEXX_VI_FENV32=1)' in block
    assert 'target_compile_options(variation_of_information_consumer PRIVATE -UNDEBUG -ffp-contract=off -fno-fast-math -frounding-math)' in text


def test_current_archive_extractor_retains_every_included_control_source():
    expression = _extractor_predicate()
    for name in (WRAPPER, ADDITIVE_HEADER, 'tests/cmake_consumer/variation_of_information_enabled_consumer.cpp',
                 'tests/variation_of_information_provider_controls.hpp',
                 'tests/variation_of_information_current_counts_controls.hpp',
                 'tests/variation_of_information_large_controls.hpp',
                 'tests/fixtures/variation_of_information_saved_v1.hpp'):
        assert (ROOT / name).is_file() and eval(expression, {'__builtins__': {}, 'name': name}), name
    for name in (WRAPPER + '/', ADDITIVE_HEADER + '/', 'tests/unassessed_provider.hpp',
                 'python/skimagexx/_variation_of_information.py'):
        assert not eval(expression, {'__builtins__': {}, 'name': name}), name


def test_current_collector_includes_wrapper_header_guard_and_live_history_inputs():
    _, _, _, helper_name, context_name = _current_history()
    qualifier = _load_source(QUALIFIER)
    assert {WRAPPER, ADDITIVE_HEADER, THIS_GUARD, helper_name, context_name} <= set(qualifier.DRAW_SOURCE_PATHS)


def test_current_shared_bytes_and_tampering_are_authenticated_before_old_history():
    helper, data, context_pin, _, _ = _current_history()
    for name in (CMAKE, QUALIFIER, HOOK, WORKFLOW):
        current = (ROOT / name).read_bytes()
        for canonical in (False, True):
            old = helper.predecessor_bytes(ROOT, name, current, context_pin, canonical)
            domain = 'canonical' if canonical else 'raw'
            assert hashlib.sha256(old).hexdigest() == data['sources'][name]['predecessor_' + domain + '_sha256']
            with pytest.raises(AssertionError):
                helper.predecessor_bytes(ROOT, name, current + b' ', context_pin, canonical)
        if name == CMAKE:
            assert hashlib.sha256(old).hexdigest() == 'b4e45d2516285e2b9ebbc30e46c9ca40bf133f40559f820f0abfd058a136e9ff'


def test_current_workflow_keeps_old_selectors_and_selects_this_guard_once():
    helper, _, context_pin, _, _ = _current_history()
    raw = (ROOT / WORKFLOW).read_bytes()
    before = _selectors(helper.predecessor_bytes(ROOT, WORKFLOW, raw, context_pin, True))
    current = _selectors(raw)
    assert len(before) == 173 and len(current) == len(set(current))
    assert current.count(THIS_GUARD) == 1
    assert [name for name in current if name in set(before)] == before
