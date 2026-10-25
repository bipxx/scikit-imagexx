"""Stdlib-only prospective metadata; no original/candidate image calls."""
import ast
from collections import Counter
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import unittest
import uuid
import zlib

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
def metadata_module(name):
    path=HERE/(name+'.py');spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
CASES=metadata_module('colocalization_cases')
REFERENCE=metadata_module('colocalization_reference')


class ProspectiveMetadata(unittest.TestCase):
    def test_fixed_partition(self):
        rows=CASES.cases();self.assertEqual(len(rows),536)
        self.assertEqual(len({r['id'] for r in rows}),536)
        self.assertEqual(Counter(r['api'] for r in rows),dict(intersection_coeff=168,manders_coloc_coeff=184,manders_overlap_coeff=184))

    def test_dtype_and_policy_coverage(self):
        for api in CASES.APIS:
            rows=[r for r in CASES.cases() if r['api']==api]
            self.assertEqual({r['dtype'] for r in rows if r['group']=='dtype'},set(CASES.DTYPES))
            self.assertEqual({r['layout'] for r in rows if r['group']=='layout'},set(CASES.LAYOUTS))
            self.assertEqual({r['policy'] for r in rows},{'warn','ignore','raise','call','default'})

    def test_all_resource_certificates(self):
        for spec in CASES.cases()+CASES.benchmark_cases():
            row=CASES.certificate(spec)
            self.assertLessEqual(row['byte_bound'],128*1024*1024)
            self.assertLessEqual(row['observer_scratch_bound'],16*1024*1024)
            self.assertLessEqual(row['combined_live_bound'],row['combined_limit'])
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

    def test_default_clones_preserve_every_old_field(self):
        rows=CASES.cases();old=rows[:286];new=rows[286:]
        self.assertEqual(len(new),250)
        self.assertEqual([r['clone_of'] for r in new],[r['id'] for r in old if r['group']!='fp'])
        index={r['id']:r for r in old}
        for row in new:
            expected=dict(index[row['clone_of']],id=row['clone_of']+'/policy-default',policy='default',clone_of=row['clone_of'])
            self.assertEqual(row,expected)

    def test_old_286_specs_have_pinned_digest(self):
        plan=json.loads((ROOT/'coverage/contracts/colocalization-plan.json').read_bytes())
        raw=json.dumps(CASES.cases()[:286],sort_keys=True,separators=(',',':')).encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),plan['predecessor']['286_specs_sha256'])

    def test_default_state_setters_are_guarded(self):
        tree=ast.parse((HERE/'colocalization_reference.py').read_bytes())
        observer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observe')
        # All explicit seterr/seterrcall sites must be descendants of the
        # non-default branch, never unconditional setup/restoration.
        setters=[n for n in ast.walk(observer) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('seterr','seterrcall')]
        self.assertTrue(setters)
        admitted=set()
        for node in ast.walk(observer):
            if isinstance(node,ast.If) and ast.unparse(node.test)=="spec['policy'] != 'default'":
                admitted.update(id(n) for child in node.body for n in ast.walk(child))
            if isinstance(node,ast.If) and ast.unparse(node.test)=="spec['policy'] == 'default'":
                admitted.update(id(n) for child in node.orelse for n in ast.walk(child))
        self.assertTrue(all(id(n) in admitted for n in setters))

    def test_streaming_assets_are_bounded_lossless_deduplicated(self):
        path=ROOT/'build/colocalization-reference/metadata-assets'/('stream-'+str(os.getpid())+'-'+uuid.uuid4().hex)
        store=REFERENCE.RawAssets(path)
        piece=bytes(range(256))*1024
        record=store.chunks(piece for _ in range(10))
        self.assertEqual(record['raw_bytes'],len(piece)*10)
        self.assertLessEqual(store.max_raw_chunk,REFERENCE.CHUNK_BYTES)
        self.assertEqual(len(list(path.glob('*.zlib'))),1)
        digest=hashlib.sha256()
        for row in record['chunks']:
            packed=(path/row['path']).read_bytes();raw=zlib.decompress(packed)
            self.assertEqual(len(raw),row['raw_bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['raw_sha256']);digest.update(raw)
        self.assertEqual(digest.hexdigest(),record['raw_sha256'])

    def test_terminal_buffer_bytes_split_and_preserved(self):
        path=ROOT/'build/colocalization-reference/metadata-assets'/('terminal-'+str(os.getpid())+'-'+uuid.uuid4().hex)
        store=REFERENCE.RawAssets(path);raw=bytearray(b'padding\x00\xff'+b'A'*(REFERENCE.CHUNK_BYTES+3))
        record=store.chunks((memoryview(raw),))
        self.assertEqual(record['raw_bytes'],len(raw))
        self.assertEqual(record['raw_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(record['chunks']),2)

    def test_corrupt_immutable_asset_is_rejected(self):
        path=ROOT/'build/colocalization-reference/metadata-assets'/('corrupt-'+str(os.getpid())+'-'+uuid.uuid4().hex)
        store=REFERENCE.RawAssets(path);record=store.chunks((b'metadata-only fixture',))
        (path/record['chunks'][0]['path']).write_bytes(b'intentional synthetic corruption')
        with self.assertRaises(AssertionError):store.chunks((b'metadata-only fixture',))

    def test_semantic_projection_drops_only_explicit_trace_events(self):
        row=dict(events=[dict(event=x) for x in ('numpy-sum','FP-handler','__array__','object-lt','finite-inallocation-mutation','numpy-unknown')],
                 result={'raw':'80000000'},warnings=['keep'],input_before={'raw_sha256':'keep'},state={'keep':True})
        projected=REFERENCE.semantic_projection(row)
        self.assertEqual([x['event'] for x in projected['events']],['FP-handler','__array__','object-lt','finite-inallocation-mutation','numpy-unknown'])
        self.assertEqual({k:v for k,v in projected.items() if k!='events'},{k:v for k,v in row.items() if k!='events'})
        self.assertEqual(len(row['events']),6)

    def test_no_raster_hex_or_full_array_tobytes(self):
        source=(HERE/'colocalization_reference.py').read_text()
        self.assertNotIn('raw_C_hex',source)
        self.assertNotIn("value.tobytes(order='C')",source)
        self.assertIn('block[start:start+count].tobytes',source)
        self.assertIn("terminal['byte_content']=assets.chunks",source)

    def test_observer_restores_provider_and_state(self):
        tree=ast.parse((HERE/'colocalization_reference.py').read_bytes())
        observer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observe')
        tries=[n for n in observer.body if isinstance(n,ast.Try)]
        self.assertEqual(len(tries),1);self.assertTrue(tries[0].finalbody)
        final=ast.unparse(ast.Module(body=tries[0].finalbody,type_ignores=[]))
        for name in ('setprofile','seterr','seterrcall','original.np','original.as_binary_ndarray','original.check_shape_equality'):
            self.assertIn(name,final)


if __name__=='__main__':unittest.main()
