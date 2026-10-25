"""Stdlib-only prospective metadata; no original/candidate image calls."""
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
def metadata_module(name):
    path=HERE/(name+'.py');spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
CASES=metadata_module('colocalization_cases')
REFERENCE=metadata_module('colocalization_reference')


class ProspectiveMetadata(unittest.TestCase):
    def test_fixed_partition(self):
        rows=CASES.cases();self.assertEqual(len(rows),286)
        self.assertEqual(len({r['id'] for r in rows}),286)
        self.assertEqual(Counter(r['api'] for r in rows),dict(intersection_coeff=90,manders_coloc_coeff=98,manders_overlap_coeff=98))

    def test_dtype_and_policy_coverage(self):
        for api in CASES.APIS:
            rows=[r for r in CASES.cases() if r['api']==api]
            self.assertEqual({r['dtype'] for r in rows if r['group']=='dtype'},set(CASES.DTYPES))
            self.assertEqual({r['layout'] for r in rows if r['group']=='layout'},set(CASES.LAYOUTS))
            self.assertEqual({r['policy'] for r in rows},{'warn','ignore','raise','call'})

    def test_all_resource_certificates(self):
        for spec in CASES.cases()+CASES.benchmark_cases():
            row=CASES.certificate(spec)
            self.assertLessEqual(row['byte_bound'],128*1024*1024)
            self.assertFalse(row['unsafe_descriptor_probes'])

    def test_resource_rejections(self):
        for shape in ([-1],[2**40],[True],[1.5]):
            with self.assertRaises(AssertionError):CASES.certificate({'shape':shape})

    def test_whole_call_cohort(self):
        rows=CASES.benchmark_cases();self.assertEqual(len(rows),18)
        for api in CASES.APIS:
            group=[r for r in rows if r['api']==api]
            self.assertEqual(len(group),6)
            self.assertTrue(any(r['shape']==[1024,1024] for r in group))
            self.assertTrue(any(len(r['shape'])==3 for r in group))
            self.assertTrue(all(r['policy']=='default' and not r['trace'] for r in group))

    def test_imports_are_stdlib_only(self):
        for name in ('colocalization_cases','colocalization_reference'):
            tree=ast.parse((HERE/(name+'.py')).read_bytes())
            imports=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):imports.extend(a.name.split('.')[0] for a in node.names)
                elif isinstance(node,ast.ImportFrom):imports.append((node.module or '').split('.')[0])
            self.assertFalse(set(imports)&{'numpy','scipy','skimage','skimagexx','pytest'})

    def test_no_algorithm_or_timer_in_factory(self):
        text=(HERE/'colocalization_cases.py').read_text()
        tree=ast.parse(text)
        forbidden={'perf_counter','process_time','timeit','import_module','reconstruction'}
        called={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertFalse(called&forbidden)
        functions={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(functions,{'cases','benchmark_cases','certificate','fixture'})

    def test_plan_matches_exact_specs(self):
        plan=json.loads((ROOT/'coverage/contracts/colocalization-plan.json').read_bytes())
        self.assertEqual(plan['cases'],CASES.cases())
        self.assertEqual(plan['future_whole_call_cases'],CASES.benchmark_cases())
        self.assertEqual(plan['observations'],None)
        self.assertEqual(plan['native_admission'],'unconditional-pending')
        self.assertEqual(plan['new_native_paths'],0)

    def test_source_pin_names_explicit(self):
        self.assertEqual(len(REFERENCE.SOURCE_CANONICAL_LF_SHA256),64)
        self.assertEqual(len(REFERENCE.UTILS_CANONICAL_LF_SHA256),64)
        self.assertNotEqual(REFERENCE.SOURCE_CANONICAL_LF_SHA256,REFERENCE.UTILS_CANONICAL_LF_SHA256)

    def test_observer_restores_provider_and_state(self):
        tree=ast.parse((HERE/'colocalization_reference.py').read_bytes())
        observer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observe')
        tries=[n for n in observer.body if isinstance(n,ast.Try)]
        self.assertEqual(len(tries),1);self.assertTrue(tries[0].finalbody)
        final=ast.unparse(ast.Module(body=tries[0].finalbody,type_ignores=[]))
        for name in ('setprofile','seterr','seterrcall','original.np','original.as_binary_ndarray','original.check_shape_equality'):
            self.assertIn(name,final)


if __name__=='__main__':unittest.main()
