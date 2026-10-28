"""Inert tests of actual sealed gate math and artifact-only transport."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/montage-row-owned-same-host.yml'
HELPER = ROOT / 'tools/montage_row_owned_comparison.py'
# Exact gate projection of sealed contract64c105; runtime authenticates its full bytes.
SEALED = json.loads(r'''{"case_ids":["gray-small-u8","gray-incomplete-u8","gray-dense-f32","gray-large-u16","gray-mean-f64","rgb-padded-u8","rgb-first-f32","gray-strided-reverse-f32","gray-fortran-f64"],"target_case":"gray-strided-reverse-f32","preserved_original_wins":["gray-small-u8","gray-incomplete-u8","gray-dense-f32","gray-large-u16","rgb-padded-u8","rgb-first-f32"],"gate":{"overall_direct_GM_gt":1.05,"overall_normalized_GM_gt":1.05,"target_each_B_pass_vs_balanced_A_direct_gt":1.5,"target_each_B_pass_vs_balanced_A_normalized_gt":1.5,"target_each_B_pass_original_gt":1.0,"each_other_eight_balanced_direct_gte":0.97,"each_other_eight_balanced_normalized_gte":0.97,"each_preserved_original_win_each_B_pass_gt":1.0,"rgb_first_each_B_pass_direct_gte":0.97,"rgb_first_each_B_pass_normalized_gte":0.97,"all_source_identity_parity_FP_deadline_required":true},"deadline_utc":"2026-10-29T00:59:41Z","measurement_variance_rule":{"null_ratio_interval":[0.97,1.0309278350515465],"scope":"Each of the eight non-target controls independently, separately for direct and normalized metrics. Inspect both same-product null pairs B2/B3 and A6/A7.","direct_null_ratio":"Earlier CPP median / later CPP median for each same-product pair.","normalized_null_ratio":"Later original/CPP ratio / earlier original/CPP ratio for each same-product pair.","if_either_null_ratio_outside_interval":"Require worst observed A/B gain >=0.97 for THAT control and THAT metric. Otherwise UNRESOLVED_MEASUREMENT. No blanket null-band veto or relaxed floors.","worst_direct":"min(all four A CPP pass medians) / max(all four B CPP pass medians)","worst_normalized":"min(all four B original/CPP ratios) / max(all four A original/CPP ratios)","always_publish":"All direct/reference/normalized null ratios, all A/B pass values and both worst-pair gains for every control.","invariants":"All fixed balanced3% floors, RGB-first per-pass floors, target/global/original gates remain required. Large separated gains may pass despite observed null variance; do not label that measurement stable.","variant_limit":1,"automatic_reruns":0},"measurement":{"order":["A","B","B","A","B","A","A","B"],"same_product_null_pairs":[[2,3],[6,7]]}}''')


def compare(reports, completed='2026-10-29T00:00:00+00:00'):
    spec = importlib.util.spec_from_file_location('actual_montage_gates', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare(reports, deepcopy(SEALED), completed)


def fixture():
    reports = []
    for product in SEALED['measurement']['order']:
        rows = []
        for case_id in SEALED['case_ids']:
            cpp = 1. if product == 'A' else .4 if case_id == SEALED['target_case'] else .8
            rows.append(dict(case_id=case_id, dtype='float32', layout='C', cpp_ms=cpp, reference_ms=2., speedup=2./cpp))
        reports.append(dict(results=rows))
    return reports


def change(reports, pass_index, cpp=None, speed=None, case_id='gray-incomplete-u8'):
    row = next(r for r in reports[pass_index-1]['results'] if r['case_id'] == case_id)
    if cpp is not None:row['cpp_ms'] = cpp
    if speed is not None:row['reference_ms'] = row['cpp_ms'] * speed
    row['speedup'] = row['reference_ms'] / row['cpp_ms']


def control(result):
    return next(r for r in result['rows'] if r['case_id'] == 'gray-incomplete-u8')


class SealedGatesTests(unittest.TestCase):
    def test_clear_separated_gain_passes_without_labeling_variability_stable(self):
        reports=fixture();change(reports,3,cpp=.7)
        result=compare(reports);row=control(result)
        self.assertEqual(result['status'],'KEEP')
        self.assertTrue(row['direct_null_outside_band'])
        self.assertTrue(row['normalized_null_outside_band'])
        self.assertTrue(row['direct_variance_separation_pass'])
        self.assertTrue(row['normalized_variance_separation_pass'])
        self.assertGreaterEqual(row['worst_direct_gain'],.97)
        self.assertEqual(len(row['null_pairs']),2)

    def test_direct_variance_requires_direct_worst_pair_only(self):
        reports=fixture();change(reports,3,cpp=1.04,speed=2.5)
        result=compare(reports);row=control(result)
        self.assertEqual(result['status'],'UNRESOLVED_MEASUREMENT')
        self.assertTrue(result['all_fixed_performance_gates_pass'])
        self.assertTrue(row['direct_null_outside_band'])
        self.assertFalse(row['normalized_null_outside_band'])
        self.assertEqual(result['unresolved_metrics'],[dict(case_id=row['case_id'],metric='direct')])

    def test_normalized_variance_requires_normalized_worst_pair_only(self):
        reports=fixture();change(reports,3,speed=1.9)
        result=compare(reports);row=control(result)
        self.assertEqual(result['status'],'UNRESOLVED_MEASUREMENT')
        self.assertTrue(result['all_fixed_performance_gates_pass'])
        self.assertFalse(row['direct_null_outside_band'])
        self.assertTrue(row['normalized_null_outside_band'])
        self.assertEqual(result['unresolved_metrics'],[dict(case_id=row['case_id'],metric='normalized')])

    def test_exact_point97_separation_boundary_and_below(self):
        for metric in('direct','normalized'):
            for floor in(.97,.969999):
                with self.subTest(metric=metric,floor=floor):
                    reports=fixture()
                    if metric=='direct':
                        # Division by1 makes the exact represented threshold;
                        # reciprocating1/.97 would introduce extra rounding.
                        for index in(1,4,6,7):change(reports,index,cpp=floor)
                        change(reports,3,cpp=1.,speed=2.5)
                    else:change(reports,3,speed=2*floor)
                    result=compare(reports);row=control(result)
                    self.assertAlmostEqual(row['worst_'+metric+'_gain'],floor,places=15)
                    self.assertEqual(result['status'],'KEEP' if floor==.97 else 'UNRESOLVED_MEASUREMENT')

    def test_stable_fixed_control_failure_discards(self):
        reports=fixture()
        for index in(2,3,5,8):change(reports,index,cpp=1.1)
        result=compare(reports)
        self.assertEqual(result['status'],'DISCARD')
        self.assertTrue(result['metric_variance_separation_pass'])
        self.assertFalse(result['fixed_gate_facts']['controls_balanced_direct'])
        self.assertFalse(result['fixed_gate_facts']['controls_balanced_normalized'])

    def test_global_target_original_and_rgb_perpass_gates_are_independent(self):
        for fault in('global','target','original','rgb'):
            with self.subTest(fault=fault):
                reports=fixture()
                if fault=='global':
                    for index in(2,3,5,8):
                        for case_id in SEALED['case_ids']:change(reports,index,cpp=.65 if case_id==SEALED['target_case'] else 1.,case_id=case_id)
                    key='overall_direct'
                elif fault=='target':
                    change(reports,5,cpp=.7,case_id=SEALED['target_case']);key='target_each_candidate_direct'
                elif fault=='original':
                    change(reports,5,speed=.9);key='preserved_original_each_candidate'
                else:
                    change(reports,5,cpp=1.05,case_id='rgb-first-f32');key='rgb_first_each_candidate_direct'
                result=compare(reports)
                self.assertNotEqual(result['status'],'KEEP')
                self.assertFalse(result['fixed_gate_facts'][key])

    def test_deadline_equality_and_expiry_never_pass(self):
        for completed in('2026-10-29T00:59:41+00:00','2026-10-29T01:00:00+00:00'):
            result=compare(fixture(),completed)
            self.assertEqual(result['status'],'DISCARD')
            self.assertFalse(result['completed_before_deadline'])

    def test_small_losses_and_all_pass_facts_remain_exported(self):
        reports=fixture()
        for index in(2,3,5,8):change(reports,index,cpp=1/.98)
        result=compare(reports);row=control(result)
        self.assertEqual(result['status'],'KEEP')
        self.assertIn(row['case_id'],result['direct_losses'])
        self.assertIn(row['case_id'],result['normalized_losses'])
        self.assertEqual(len(row['cpp_ms']),8)
        self.assertEqual(len(row['candidate_direct_passes']),4)
        self.assertEqual(result['raw_samples'],1008)
        self.assertEqual(len(result['rows']),9)


class TransportTests(unittest.TestCase):
    def test_exact_retained_XML_table(self):
        lines=WORKFLOW.read_text().splitlines()
        start=next(i for i,line in enumerate(lines)if line.strip().startswith('ROLE=') and "python - <<'PY'"in line)
        end=lines.index('          PY',start+1)
        tree=ast.parse('\n'.join(line[10:]for line in lines[start+1:end]))
        counts={}
        for node in tree.body:
            if isinstance(node,ast.For)and isinstance(node.target,ast.Tuple)and ast.unparse(node.target)=='(family, count)':
                for family,count in ast.literal_eval(node.iter):
                    if family.endswith('intake')or family=='block-reduce-integer-width':counts[family+'.xml']=count
                    else:
                        for mode in('normal','scalar'):counts[family+'-'+mode+'.xml']=count
        counts['approximate-polygon-runtime-reference-guards.xml']=16
        self.assertEqual(len(counts),79)
        self.assertEqual(sum(counts.values()),13079)
        self.assertEqual(sum(v for n,v in counts.items()if n.endswith(('-normal.xml','-scalar.xml'))and not n.startswith('color-transforms-regression-')),9346)

    def test_fixed_inputs_and_source_closure(self):
        source=WORKFLOW.read_text()
        self.assertIn("candidate_artifact_id:\n        type: string\n        required: true",source)
        self.assertIn("candidate_artifact_sha256:\n        type: string\n        required: true",source)
        self.assertNotIn('default:',source)
        for value in('40778508940','40785369226','12855199919','4f69eba77a262056459eaab1e0a39e53610e9305','6b22e64641a702c92873c376ff650cd8c8764d0d','64c10577317d77e6b39a206904f578b06db68ee2be3745d8a3d646100f184bdb','93a0dc6c08da84a705880d7f01e751a47f11e12728aa3d28c19112da5d236bdd','c28ce23805ab39ad55d806f42a227b6269b30927b36491f7f9a062e2ed4dd513'):
            self.assertIn(value,source)
        for value in('== 176','== 83','== 1447',"{'passed': 1447}",'== 247','== 29','== 9346','== 13079',"'approximate-polygon-runtime-reference-guards.xml'", "('approximate-polygon', 275)","('subdivide-polygon', 279)","['include/skimagexx/montage.hpp', 'lib/libskimagexx.a']", "{'simd_montage_avx2.cpp.o'}"):
            self.assertIn(value,source)
        self.assertIn('new_header.replace(new_comment, old_comment) == old_header',source)
        self.assertIn('len(registrations) == 3',source)
        self.assertIn('inverse == old_test',source)
        self.assertIn("['montage.cpp.o']",source)
        self.assertIn('1-baseline 2-candidate 3-candidate 4-baseline 5-candidate 6-baseline 7-baseline 8-candidate',source)
        self.assertIn("runpy.run_path('controller/tools/montage_row_owned_comparison.py')['compare']",source)
        self.assertIn('if: always()',source)
        self.assertNotIn('g++',source)
        self.assertNotIn('cmake --build',source)
        self.assertNotIn('pip wheel',source)

    def test_all_inline_python_and_launcher_parse(self):
        lines=WORKFLOW.read_text().splitlines();count=0
        for i,line in enumerate(lines):
            if line.strip().endswith("python - <<'PY'"):
                end=lines.index('          PY',i+1)
                ast.parse('\n'.join(row[10:]for row in lines[i+1:end]));count+=1
        self.assertEqual(count,4)
        start=next(i for i,line in enumerate(lines)if line.strip()=="cat > results/launch.py <<'PY'")
        end=lines.index('          PY',start+1)
        launcher='\n'.join(row[10:]for row in lines[start+1:end])
        tree=ast.parse(launcher)
        callback=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='retain_measure')
        delegates=[n for n in ast.walk(callback)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='delegate']
        self.assertEqual(len(delegates),1)
        self.assertLess(launcher.index('os.sched_setaffinity'),launcher.index('import run'))
        self.assertLess(launcher.index("process-before.json"),launcher.index('runpy.run_path'))
        self.assertLess(launcher.index('sealed deadline expired'),launcher.index('import run'))
        self.assertIn('stream.flush()',launcher)
        self.assertIn("status='PENDING_FINAL_HARNESS_VALIDATION'",launcher)
        self.assertIn('finally:',launcher)
        self.assertIn('raise\n',launcher)

    def test_actual_partial_return_journal_survives_rethrown_failure(self):
        lines=WORKFLOW.read_text().splitlines()
        start=next(i for i,line in enumerate(lines)if line.strip()=="cat > results/launch.py <<'PY'")
        end=lines.index('          PY',start+1)
        source='\n'.join(line[10:]for line in lines[start+1:end])
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);product=root/'product';(product/'benchmarks').mkdir(parents=True)
            run_file=product/'benchmarks/run.py';run_file.write_bytes(b'# inert module boundary\n')
            pin='c28ce23805ab39ad55d806f42a227b6269b30927b36491f7f9a062e2ed4dd513'
            self.assertEqual(source.count(pin),1)
            # Only this inert copy substitutes a fake module's boundary hash;
            # the committed controller always authenticates the real c28 module.
            source=source.replace(pin,hashlib.sha256(run_file.read_bytes()).hexdigest())
            contract=dict(case_ids=SEALED['case_ids'],deadline_utc='2100-01-01T00:00:00Z')
            (root/'dispatch-contract.json').write_text(json.dumps(contract))
            output=root/'1-baseline';module=types.ModuleType('run');module.__file__=str(run_file);calls=[]
            timing=dict(samples_ms=[[1.]*7,[2.]*7],loops=[3,4])
            reference,native=object(),object()
            def measure(*args,**kwargs):
                calls.append((args,kwargs))
                if len(calls)==2:raise ValueError('original first-attempt failure')
                return timing
            module.measure=measure
            def harness(path,run_name):
                self.assertTrue((output/'process-before.json').is_file())
                for _ in range(2):module.measure(reference,native,7,19)
            old_path=sys.path[:]
            try:
                with patch.object(sys,'argv',['launch.py',str(product),str(output),'7','1-baseline','baseline']),patch.dict(sys.modules,{'run':module}),patch.object(runpy,'run_path',harness),patch('os.sched_setaffinity',create=True),patch('os.sched_getaffinity',return_value={7},create=True):
                    with self.assertRaisesRegex(ValueError,'original first-attempt failure'):
                        exec(compile(source,'<inert-real-launcher>','exec'),{})
            finally:sys.path[:]=old_path
            journal=[json.loads(line)for line in(output/'returned-measures.jsonl').read_text().splitlines()]
            self.assertEqual(len(journal),1)
            self.assertEqual(journal[0]['timing'],timing)
            self.assertEqual(journal[0]['status'],'PENDING_FINAL_HARNESS_VALIDATION')
            self.assertEqual(calls,[((reference,native,7,19),{})]*2)
            final=json.loads((output/'process-after.json').read_bytes())
            self.assertEqual(final['status'],'FAILED')
            self.assertEqual(final['completed_measures'],1)
            self.assertEqual(final['failure']['type'],'ValueError')


if __name__=='__main__':unittest.main()
