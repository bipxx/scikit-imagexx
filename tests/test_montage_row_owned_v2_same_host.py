"""Inert v2 transport guards; reuse the sealed v1 gate fixtures and helper."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=ROOT/'.github/workflows/montage-row-owned-v2-same-host.yml'


def legacy():
    spec=importlib.util.spec_from_file_location('sealed_v1_fixture',ROOT/'tests/test_montage_row_owned_same_host.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class V2TransportTests(unittest.TestCase):
    def test_products_contracts_and_explicit_closed_v1(self):
        text=WORKFLOW.read_text()
        for value in ('40790763016','b5bd1df7e27099e528fb4c7db67e2cb0bf35abae',
                      '40778508940','4f69eba77a262056459eaab1e0a39e53610e9305',
                      '6b22e64641a702c92873c376ff650cd8c8764d0d',
                      'eb47b18e695a9f376a004a2490e5a62240933a1418b8f2520c6456b964fac01e',
                      '64c10577317d77e6b39a206904f578b06db68ee2be3745d8a3d646100f184bdb'):
            self.assertIn(value,text)
        for name in ('candidate_artifact_id','candidate_artifact_sha256'):
            self.assertIn(name+':\n        type: string\n        required: true',text)
        self.assertNotIn('default:',text)
        self.assertIn("c['parent_v1']['decision']=='UNRESOLVED_MEASUREMENT'",text)
        self.assertIn("Path('results/closed-v1.json')",text)
        self.assertIn("for field in ('gate','measurement','measurement_variance_rule','deadline_utc','case_ids','target_case','preserved_original_wins'):",text)
        self.assertIn("assert c[field]==parent_contract[field],field",text)

    def test_exact_eight_source_paths_and_parent_inverse(self):
        text=WORKFLOW.read_text()
        self.assertIn("added = {'src/montage_simd.hpp', 'src/simd_montage_avx2.cpp', parent_contract, contract}",text)
        self.assertIn("allowed = added | {'src/montage.cpp', 'CMakeLists.txt', 'tests/test_montage_native.cpp', 'include/skimagexx/montage.hpp'}",text)
        self.assertIn("inverse_leaf == old_leaf",text)
        self.assertIn("if(bytes==1){*start=*fill;return;}",text)
        self.assertIn("if(zero||pixel_bytes==1){std::memset(start,*fill,bytes);return;}",text)
        self.assertIn("inverse_v2_test == (parent/'tests/test_montage_native.cpp').read_text()",text)
        self.assertIn("'', inverse_v2_test, flags=re.S)",text)
        self.assertIn("for name in names[0] - {'src/montage.cpp'}:",text)
        self.assertIn("['include/skimagexx/montage.hpp', 'lib/libskimagexx.a']",text)
        self.assertIn("{'simd_montage_avx2.cpp.o'}",text)
        self.assertIn("== ['montage.cpp.o']",text)
        self.assertIn('len(old_objects) == 163 and len(new_objects) == 164',text)

    def test_helper_and_unchanged_measure_launcher(self):
        self.assertEqual(hashlib.sha256((ROOT/'tools/montage_row_owned_comparison.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                         '66fa2153cc7ea0c3fa930811dd807b9904cb5b6767fe5816e0f70f6ea9f05533')
        def launcher(text):
            lines=text.splitlines();start=next(i for i,line in enumerate(lines)if line.strip()=="cat > results/launch.py <<'PY'")
            end=lines.index('          PY',start+1)
            return '\n'.join(line[10:]for line in lines[start+1:end])
        self.assertEqual(launcher(WORKFLOW.read_text()),launcher((ROOT/'.github/workflows/montage-row-owned-same-host.yml').read_text()))
        self.assertIn('1-baseline 2-candidate 3-candidate 4-baseline 5-candidate 6-baseline 7-baseline 8-candidate',WORKFLOW.read_text())

    def test_all_inline_python_and_retained_closure(self):
        text=WORKFLOW.read_text();lines=text.splitlines();count=0
        for i,line in enumerate(lines):
            if line.strip().endswith("python - <<'PY'"):
                end=lines.index('          PY',i+1)
                ast.parse('\n'.join(row[10:]for row in lines[i+1:end]));count+=1
        self.assertEqual(count,4)
        for value in ('== 176','== 83','== 9346','== 13079',"{'passed': 1447}",'== 247','== 29','if: always()',
                      "('approximate-polygon', 275)","('subdivide-polygon', 279)",
                      'montage-row-owned-v2-same-host-raw'):
            self.assertIn(value,text)
        self.assertNotIn('g++',text)
        self.assertNotIn('cmake --build',text)
        self.assertNotIn('pip wheel',text)

    def test_reused_gate_fixtures_keep_metric_specific_classification(self):
        old=legacy()
        for mode,expected in (('clear','KEEP'),('separated','KEEP'),('direct','UNRESOLVED_MEASUREMENT'),('normalized','UNRESOLVED_MEASUREMENT'),('fixed','DISCARD'),('expired','DISCARD')):
            with self.subTest(mode=mode):
                reports=old.fixture()
                if mode=='separated':old.change(reports,3,cpp=.7)
                elif mode=='direct':old.change(reports,3,cpp=1.04,speed=2.5)
                elif mode=='normalized':old.change(reports,3,speed=1.9)
                elif mode=='fixed':
                    for index in (2,3,5,8):old.change(reports,index,cpp=1.1)
                completed='2026-10-29T01:00:00+00:00'if mode=='expired'else'2026-10-29T00:00:00+00:00'
                self.assertEqual(old.compare(reports,completed)['status'],expected)


if __name__=='__main__':unittest.main()
