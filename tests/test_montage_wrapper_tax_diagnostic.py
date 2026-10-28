"""Inert checks for the diagnostic transport; no scientific imports/calls."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'tools/diagnostics/montage_wrapper_tax'

class DiagnosticMetadata(unittest.TestCase):
    def test_lf_source_pins_and_fixed_cohort(self):
        c=json.loads((CODE/'diagnostic-contract.json').read_bytes())
        for name,key in [('adapter.cpp','adapter_sha256'),('run_diagnostic.py','runner_sha256')]:
            raw=(CODE/name).read_bytes()
            self.assertNotIn(b'\r',raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),c[key])
        self.assertEqual((c['threads'],c['rounds'],c['seed'],c['raw_samples']),(1,21,2026,756))
        self.assertEqual(len(c['case_ids']),9)
        self.assertEqual(c['routes'],dict(wrapper=0,generic=1,generic_A=1,generic_B=1))
        self.assertEqual(c['loops'],[1000,1000,400,200,100,500,300,50,30])

    def test_preflight_excludes_every_clock_call(self):
        tree=ast.parse((CODE/'run_diagnostic.py').read_text())
        clocks=[x for x in ast.walk(tree)if isinstance(x,ast.Call)and ast.unparse(x.func)=='time.perf_counter_ns']
        self.assertEqual(len(clocks),2)
        guard=[x for x in ast.walk(tree)if isinstance(x,ast.If)and ast.unparse(x.test)=='not args.preflight_only']
        self.assertEqual(len(guard),1)
        self.assertTrue(all(x in list(ast.walk(guard[0]))for x in clocks))
        source=(CODE/'run_diagnostic.py').read_text()
        self.assertIn("status='PREFLIGHT_ONLY_PASS'",source)
        self.assertIn("timing_calls=0 if args.preflight_only",source)
        self.assertIn('assert output.tobytes()==expected',source)
        self.assertIn('assert adapter.identity(prepared)==identity',source)

    def test_same_core_same_descriptor_null_controls(self):
        source=(CODE/'adapter.cpp').read_text()
        self.assertIn('RTLD_NOW|RTLD_NOLOAD|RTLD_LOCAL',source)
        self.assertIn('info[0].dli_fbase!=info[1].dli_fbase',source)
        self.assertIn('const auto entry=state->entries[route]',source)
        self.assertIn('entry(state->descriptor,state->grid,state->fill.buf,state->destination,1)',source)
        self.assertNotIn('montage.cpp',source)
        self.assertNotIn('perf_counter',source)

    def test_separate_builder_and_artifact_only_timing_host(self):
        source=(ROOT/'.github/workflows/montage-wrapper-tax-diagnostic.yml').read_text()
        builder,timing=source.split('  diagnose:\n',1)
        self.assertIn('g++',builder)
        self.assertNotIn('g++',timing)
        self.assertNotIn('pytest',timing)
        self.assertIn('needs: prepare-adapter',timing)
        self.assertIn("item['workflow_run']['id']==int(os.environ['GITHUB_RUN_ID'])",timing)
        self.assertIn("hashlib.sha256(data).hexdigest()==expected",timing)
        self.assertIn("pin['sha256']==os.environ['ADAPTER_BINARY_SHA']",timing)
        self.assertEqual(source.count("python-version: '3.11.17'"),2)
        self.assertIn('cmp results/identity-before.log results/identity-after.log',timing)
        self.assertIn('if: always()',timing)
        self.assertNotIn('workflow_dispatch:',timing)

if __name__=='__main__':unittest.main()
