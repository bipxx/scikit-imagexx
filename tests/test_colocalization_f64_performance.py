"""Inert protocol/driver/transport checks: no scientific imports or timing."""
import ast,copy,hashlib,importlib.util,json,statistics,unittest,tempfile,sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'tools/colocalization_f64_measure.py'
PROTOCOL=ROOT/'coverage/contracts/colocalization-f64-finite-performance-v2.json'
WORKFLOW=ROOT/'.github/workflows/colocalization-f64-same-host.yml'


def driver():
    spec=importlib.util.spec_from_file_location('coloc_driver_inert',DRIVER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fake_report(contract,pass_index=1):
    c=contract;events=[dict(event='driver_started')];rows=[];checks=[]
    for i,spec in enumerate(c['case_specs']):
        case=spec['id'];orders=[[0,1]if(r+i+pass_index)%2==0 else[1,0]for r in range(7)]
        batch=[];samples=[[],[]];counts=[[],[]]
        events.append(dict(event='case_prechecked',case_index=i))
        for phase,round_id,order,elapsed in [('warmup',-1,[0,1],250000000)]+[('sample',r,o,100000000)for r,o in enumerate(orders)]:
            for ordinal,fn in enumerate(order):
                common=dict(case_index=i,case_id=case,function=fn,round=round_id,ordinal=ordinal,loops=10)
                events.append(dict(common,event='batch_before',requested_phase=phase))
                after=dict(common,event='batch_after',phase=phase,elapsed_ns=elapsed,complete=True,accepted=phase=='sample',failure=None)
                events.append(after);batch.append(after)
                if phase=='sample':samples[fn].append(elapsed/10/1e6);counts[fn].append(10)
        timing=dict(reference_ms=10.0,cpp_ms=10.0,speedup=1.0,samples_ms=samples,loops=counts,sample_labels=['scikit-image','current'],warmup_elapsed_ns=[250000000,250000000],sample_order=orders,batch_records=batch)
        rows.append(dict(id=case,api=spec['api'],threads=1,**timing))
        events.append(dict(event='measure_returned',case_index=i,case_id=case,status='PENDING_FINAL_HARNESS_VALIDATION',timing=timing))
        events.append(dict(event='case_validated',case_index=i))
        checks.append(dict(id=case,status='passed',reference={},current={},reference_warnings=[],current_warnings=[],reference_error=None,current_error=None,native_witness=dict(forward_attempts=0,c_returns=['coloc_sum']),admitted_stage_returns=1))
    events.append(dict(event='driver_complete',cases=21))
    meta=dict(threads=1,repeat=7,target_ms=100,raw_samples=294,pass_index=pass_index,method_sha256=c['measurement']['method_sha256'],case_spec_sha256=c['case_spec_sha256'],helpers_sha256=c['validation']['helpers_canonical_lf_sha256'],protocol_sha256='58456c457bde8ddcac907620bf4fb928b541fc689dd160f154039e9cadd4416c',completed_utc=c['created_utc'],benchmark_tool_before=dict(name='threadpoolctl',version='3.6.0',module_sha256=c['benchmark_tools']['threadpoolctl']['module_sha256'],wheel_sha256=c['benchmark_tools']['threadpoolctl']['sha256']))
    meta['benchmark_tool_after']=meta['benchmark_tool_before']
    return dict(status='COMPLETE',metadata=meta,results=rows,checks=checks),events


class ColocPerformanceTests(unittest.TestCase):
    def setUp(self):self.c=json.loads(PROTOCOL.read_bytes());self.m=driver()

    def test_exact_frozen_cases_helpers_products_and_clock(self):
        c=self.c
        self.assertEqual(hashlib.sha256(PROTOCOL.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),'58456c457bde8ddcac907620bf4fb928b541fc689dd160f154039e9cadd4416c')
        self.assertEqual(len(c['case_specs']),21);self.assertEqual(len(c['target_ids']),4);self.assertEqual(len(c['control_ids']),17);self.assertEqual(len(c['preserved_original_wins']),5)
        self.assertEqual(hashlib.sha256(json.dumps(c['case_specs'],sort_keys=True).encode()).hexdigest(),c['case_spec_sha256'])
        for name,pin in c['validation']['helpers_canonical_lf_sha256'].items():
            # Sparse packet includes both pure case factories and the unchanged main closure.
            if (ROOT/name).exists():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),pin)
        self.assertEqual(c['measurement']['accepted_samples'],21*14*4)
        self.assertEqual(c['benchmark_tools']['threadpoolctl']['sha256'],'43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb')
        old=ROOT/c['supersedes_before_any_timing']['path']
        self.assertEqual(hashlib.sha256(old.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),c['supersedes_before_any_timing']['sha256'])
        self.assertEqual(c['variant_limit'],1);self.assertEqual(c['automatic_reruns'],0)
        self.assertEqual((self.m.datetime.fromisoformat(c['deadline_utc'])-self.m.datetime.fromisoformat(c['created_utc'])).total_seconds(),7200)
        self.assertEqual(c['products']['candidate']['revision'],'d496dd179addc03ad13df30f6df4dde4f2b64328')

    def test_report_validator_accepts_complete_and_rejects_missing_evidence(self):
        report,events=fake_report(self.c);self.m.validate_report(report,events,self.c,1)
        for mutate in ('short_batch','missing_return','wrong_order','no_native_return','wrong_loops'):
            r,e=copy.deepcopy((report,events))
            if mutate=='short_batch':next(x for x in e if x['event']=='batch_after'and x['accepted'])['elapsed_ns']=99999999
            elif mutate=='missing_return':e.remove(next(x for x in e if x['event']=='measure_returned'))
            elif mutate=='wrong_order':r['results'][0]['sample_order'][0].reverse()
            elif mutate=='no_native_return':r['checks'][0]['native_witness']['c_returns']=[]
            else:r['results'][0]['loops'][0][0]=1
            with self.subTest(mutate=mutate),self.assertRaises(AssertionError):self.m.validate_report(r,e,self.c,1)

    def test_individual_repeated_target_and_control_floors(self):
        reports=[fake_report(self.c,i+1)[0]for i in range(4)]
        for i,report in enumerate(reports):
            for row in report['results']:
                row.update(reference_ms=2.0,cpp_ms=1.0 if i in (0,3)else .9,speedup=2.0/(1.0 if i in (0,3)else .9))
        class FixedDateTime(self.m.datetime):
            @classmethod
            def now(cls,tz=None):return cls.fromisoformat(self.c['created_utc'])
        clock=patch.object(self.m,'datetime',FixedDateTime);clock.start();self.addCleanup(clock.stop)
        self.assertEqual(self.m.compare(reports,self.c)['decision'],'KEEP')
        for kind in ('one_target_pass','normalized_target','one_control','old_original_win'):
            r=copy.deepcopy(reports)
            case=self.c['target_ids'][0]if kind in ('one_target_pass','normalized_target')else self.c['control_ids'][0]
            row=next(x for x in r[1]['results']if x['id']==case)
            if kind=='one_target_pass':row['cpp_ms']=1.01
            elif kind=='normalized_target':row['speedup']=1.9
            elif kind=='one_control':
                for p in (1,2):next(x for x in r[p]['results']if x['id']==case)['cpp_ms']=1.04
            else:row['speedup']=.99
            with self.subTest(kind=kind):self.assertEqual(self.m.compare(r,self.c)['decision'],'DISCARD')

    def test_external_driver_preserves_helpers_and_explicit_sampler(self):
        tree=ast.parse(DRIVER.read_text());text=DRIVER.read_text()
        self.assertIn("helpers._check_case(runtime,spec,check)",text)
        self.assertIn('timing=sampler.measure(',text)
        self.assertNotIn('runtime.measure=',text);self.assertNotIn('run.measure=',text)
        self.assertIn('helpers._verify_wheel(runtime,args.wheel)',text)
        self.assertIn("report.update(status='FAILED'",text)
        self.assertFalse(any(isinstance(n,ast.Import)and any(a.name in ('numpy','scipy','skimage','skimagexx')for a in n.names)for n in tree.body))

    def test_benchmark_tool_identity_and_unchanged_v1_gate_fields(self):
        old=json.loads((ROOT/self.c['supersedes_before_any_timing']['path']).read_bytes())
        for key in ('gate','target_ids','control_ids','preserved_original_wins','case_specs','case_spec_sha256','deadline_utc','created_utc','allowed_product_source_delta'):
            self.assertEqual(self.c[key],old[key])
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'threadpoolctl.py';path.write_bytes(b'fixed tool bytes')
            c=copy.deepcopy(self.c);c['benchmark_tools']['threadpoolctl']['module_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            module=SimpleNamespace(__file__=str(path),__version__='3.6.0')
            with patch.dict(sys.modules,threadpoolctl=module):
                actual=self.m.tool_identity(c);self.assertEqual(actual['version'],'3.6.0')
                module.__version__='3.5.0'
                with self.assertRaises(AssertionError):self.m.tool_identity(c)

    def test_workflow_inline_python_pins_and_no_builds(self):
        text=WORKFLOW.read_text();lines=text.splitlines();parsed=0
        for i,line in enumerate(lines):
            if line.strip().endswith("python - <<'PY'")or line.strip()=="cat > results/launch.py <<'PY'":
                end=lines.index('          PY',i+1);ast.parse('\n'.join(row[10:]for row in lines[i+1:end]));parsed+=1
        self.assertEqual(parsed,7)
        pin=hashlib.sha256(DRIVER.read_bytes().replace(b'\r\n',b'\n')).hexdigest();self.assertIn(pin,text)
        self.assertIn('1-baseline 2-candidate 3-candidate 4-baseline',text)
        self.assertIn("if: always()",text)
        for forbidden in ('cmake --build','g++','pytest','pip wheel','worst_pair','measurement_variance_rule'):
            self.assertNotIn(forbidden,text)
        self.assertIn("{'colocalization.cpp.o','simd_colocalization_avx2.cpp.o'}",text)
        self.assertIn('needs: prepare-benchmark-tool',text)
        self.assertIn('results/benchmark-tool/threadpoolctl-3.6.0-py3-none-any.whl',text)
        self.assertIn("meta['driver_sha256']",text)
        self.assertIn("driver['validate_report']",text);self.assertIn("driver['compare']",text)


if __name__=='__main__':unittest.main()
