"""Bounded fake-clock and source/transport guards; no scientific imports."""
import ast,hashlib,importlib.util,inspect,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'tools/montage_sustained.py'
WORKFLOW=ROOT/'.github/workflows/montage-sustained-same-host.yml'
PROTOCOL=ROOT/'coverage/contracts/montage-sustained-measurement.json'
INVERSE_EDITS=json.loads(r'''[[";args=parser.parse_args()",";parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--pass-index',type=int,choices=range(1,9),required=True);args=parser.parse_args()"],["    from run import measure","    ROOT=args.source_root.resolve();CONTRACT=ROOT/'coverage/contracts/montage-benchmark.json'\n    frozen_path=ROOT/'benchmarks/montage.py'\n    assert hashlib.sha256(frozen_path.read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest()=='93a0dc6c08da84a705880d7f01e751a47f11e12728aa3d28c19112da5d236bdd'\n    loader=importlib.util.spec_from_file_location('unchanged_montage_helpers',frozen_path);helpers=importlib.util.module_from_spec(loader);loader.loader.exec_module(helpers)\n    case_specs,fixture,snapshot,oracle_tiles=(getattr(helpers,name)for name in('case_specs','fixture','snapshot','oracle_tiles'))\n    sys.path.insert(0,str(ROOT/'benchmarks'))\n    protocol=json.loads(args.protocol.read_bytes())\n    method_metadata=dict(frozen_harness_sha256='93a0dc6c08da84a705880d7f01e751a47f11e12728aa3d28c19112da5d236bdd',measurement_protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),method_sha256=hashlib.sha256(inspect.getsource(measure).encode()).hexdigest(),warmup_calibration_min_ms=250,interbatch_sleep_ms=0,pass_index=args.pass_index,method='sustained-adaptive-full-public-v1')"],["timing=measure(lambda:reference(image,**options),lambda:native(image,**options),args.repeat,19)","timing=measure(lambda:reference(image,**options),lambda:native(image,**options),args.repeat,100,journal=args.output.with_suffix('.batches.jsonl'),case_index=index,case_id=spec['case_id'],pass_index=args.pass_index,deadline=protocol['deadline_utc'])"],["metadata=dict(timestamp_utc=","metadata=dict(**method_metadata,timestamp_utc="],["target_ms=19,raw_samples=126","target_ms=100,raw_samples=126"]]''')

def module():
    spec=importlib.util.spec_from_file_location('actual_sustained',DRIVER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class Clock:
    def __init__(self):self.ns=0;self.active=False;self.live=0;self.calls=0;self.destroyed=0
    def now(self):
        if self.active:assert self.live==0
        self.active=not self.active
        return self.ns
    def fn(self):
        assert self.active
        self.calls+=1;self.ns+=10000;self.live+=1
        clock=self
        class Output:
            def __del__(self):
                assert clock.active
                clock.ns+=1000;clock.live-=1;clock.destroyed+=1
        return Output()

class SustainedTests(unittest.TestCase):
    def test_honest_hashes_and_exact_full_validation_inverse(self):
        m=module();c=json.loads(PROTOCOL.read_bytes())
        self.assertEqual(hashlib.sha256(DRIVER.read_bytes()).hexdigest(),c['driver_sha256'])
        self.assertEqual(hashlib.sha256(inspect.getsource(m.measure).encode()).hexdigest(),c['method_sha256'])
        original=(ROOT/'benchmarks/montage.py').read_text();self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),c['frozen_harness_sha256'])
        old=original[original.index('def main():'):original.index("\n\nif __name__=='__main__'")]
        new=inspect.getsource(m.main).rstrip()
        for left,right in reversed(INVERSE_EDITS):self.assertEqual(new.count(right),1);new=new.replace(right,left,1)
        self.assertEqual(new,old.rstrip())
        self.assertEqual(c['measurement']['target_ms'],100)
        self.assertEqual(c['measurement']['warmup_calibration_min_ms_per_function'],250)
        self.assertNotIn('sleep(',inspect.getsource(m.measure))

    def test_adaptive_batches_destroy_outputs_and_retain_all_evidence(self):
        m=module();clock=Clock()
        with tempfile.TemporaryDirectory()as tmp,patch.object(m.time,'perf_counter_ns',clock.now):
            path=Path(tmp)/'batches.jsonl'
            r=m.measure(clock.fn,clock.fn,7,100,journal=path,case_index=0,case_id='fixed',pass_index=1,deadline='2100-01-01T00:00:00Z')
            events=[json.loads(line)for line in path.read_text().splitlines()]
        self.assertEqual(clock.calls,clock.destroyed);self.assertEqual(clock.live,0);self.assertFalse(clock.active)
        self.assertTrue(all(n>=250000000 for n in r['warmup_elapsed_ns']))
        self.assertEqual(list(map(len,r['samples_ms'])),[7,7]);self.assertEqual(list(map(len,r['loops'])),[7,7])
        self.assertTrue(all(n>1000 for part in r['loops']for n in part))
        accepted=[e for e in r['batch_records']if e['accepted']]
        self.assertEqual(len(accepted),14);self.assertTrue(all(e['elapsed_ns']>=100000000 for e in accepted))
        self.assertEqual(len([e for e in events if e['event']=='batch_before']),len(r['batch_records']))
        self.assertEqual(len([e for e in events if e['event']=='batch_after']),len(r['batch_records']))
        self.assertEqual(events[-1]['timing'],r)
        self.assertEqual(r['sample_order'],[[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]])
        for fn in(0,1):
            rows=[e for e in accepted if e['function']==fn]
            self.assertEqual([e['elapsed_ns']/e['loops']/1e6 for e in rows],r['samples_ms'][fn])

    def test_underlength_sample_attempt_is_calibration_not_a_sample(self):
        m=module();ns=[0];calls=[0]
        def fn():
            calls[0]+=1;ns[0]+=10000 if calls[0]<=34501 else 5000
        with tempfile.TemporaryDirectory()as tmp,patch.object(m.time,'perf_counter_ns',lambda:ns[0]):
            r=m.measure(fn,fn,7,100,journal=Path(tmp)/'events',case_index=0,case_id='fixed',pass_index=1,deadline='2100-01-01T00:00:00Z')
        short=[e for e in r['batch_records']if e['phase']=='underlength-calibration']
        self.assertTrue(short);self.assertTrue(all(not e['accepted']and e['elapsed_ns']<100000000 for e in short))
        self.assertEqual(sum(e['accepted']for e in r['batch_records']),14)

    def test_failure_and_deadline_keep_evidence_without_replay(self):
        m=module();clock=Clock()
        def fail():clock.ns+=10000;raise RuntimeError('original first attempt')
        with tempfile.TemporaryDirectory()as tmp,patch.object(m.time,'perf_counter_ns',clock.now):
            path=Path(tmp)/'events'
            with self.assertRaisesRegex(RuntimeError,'original first attempt'):m.measure(fail,fail,7,100,journal=path,case_index=0,case_id='fixed',pass_index=1,deadline='2100-01-01T00:00:00Z')
            events=[json.loads(line)for line in path.read_text().splitlines()]
            self.assertEqual(len(events),2);self.assertFalse(events[-1]['complete']);self.assertFalse(events[-1]['accepted'])
            with self.assertRaisesRegex(AssertionError,'deadline expired'):m.measure(fail,fail,7,100,journal=path,case_index=0,case_id='fixed',pass_index=1,deadline='2000-01-01T00:00:00Z')
            self.assertEqual(len(path.read_text().splitlines()),2)

    def test_inline_transport_and_unchanged_gate_hash(self):
        text=WORKFLOW.read_text();lines=text.splitlines();count=0
        for i,line in enumerate(lines):
            if line.strip().endswith("python - <<'PY'"):
                end=lines.index('          PY',i+1);ast.parse('\n'.join(row[10:]for row in lines[i+1:end]));count+=1
        self.assertEqual(count,4)
        start=next(i for i,line in enumerate(lines)if line.strip()=="cat > results/launch.py <<'PY'");end=lines.index('          PY',start+1);ast.parse('\n'.join(row[10:]for row in lines[start+1:end]))
        self.assertNotIn('run.measure=',text)
        self.assertIn("all(event['elapsed_ns']>=100000000 for event in accepted)",text)
        self.assertIn("meta['target_ms']==100",text)
        self.assertIn('if: always()',text)
        self.assertEqual(hashlib.sha256((ROOT/'tools/montage_row_owned_comparison.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),'66fa2153cc7ea0c3fa930811dd807b9904cb5b6767fe5816e0f70f6ea9f05533')
        self.assertNotIn('g++',text);self.assertNotIn('pip wheel',text);self.assertNotIn('cmake --build',text)

if __name__=='__main__':unittest.main()
