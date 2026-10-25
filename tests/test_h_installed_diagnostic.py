"""Standard-library H diagnostic controls; no products or scientific imports."""
import copy
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h_fixed_diagnostic_metadata', ROOT / 'tools/h_installed_diagnostic.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


class HMetadataTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(D.INPUTS.read_bytes())

    def test_fixed_four_product_pins_and_versions(self):
        self.assertEqual(D.sha(D.INPUTS), D.INPUTS_SHA256)
        for profile in ('windows', 'minimum', 'modern', 'debug'):
            p = D.validate_inputs(self.data, profile)
            self.assertEqual({v['name']:v['version'] for v in p['runtime_wheels'].values()}, p['runtime_pins'])
            self.assertEqual(p['source_topology']['regular_files'], 1277)
        self.assertNotEqual(self.data['products']['modern']['core']['sha256'], self.data['products']['debug']['core']['sha256'])
        self.assertGreater(self.data['products']['debug']['artifact']['size_in_bytes'], D.LI.OUTER_CAP)

    def test_wrong_source_run_scope_or_profile_rejected(self):
        for key, value in (('failed_source', '0'*40), ('failed_run', 1), ('product_member_count', 118)):
            changed = copy.deepcopy(self.data); changed['products']['modern'][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                D.validate_inputs(changed, 'modern')
        del self.data['products']['debug']
        with self.assertRaises(RuntimeError):
            D.validate_inputs(self.data, 'modern')

    def test_exact_projection_and_corruption_rejected(self):
        gate = copy.deepcopy(self.data['gate'])
        before = b'prefix\n' + gate['old_projection'].encode() + b'\nsuffix'
        after = before.replace(gate['old_projection'].encode(), gate['new_projection'].encode())
        for label, raw in (('before', before), ('after', after)):
            gate[label+'_bytes'] = len(raw); gate[label+'_sha256'] = hashlib.sha256(raw).hexdigest()
        self.assertEqual(D.project_gate(before, gate), after)
        for damaged in (before+b'x', before.replace(b'prefix', b'broken'), before.replace(b'allowed_body', b'anything')):
            with self.assertRaises(RuntimeError):
                D.project_gate(damaged, gate)
        gate['after_sha256'] = '0'*64
        with self.assertRaises(RuntimeError):
            D.project_gate(before, gate)

    def test_archived_raw_crlf_preserved_with_canonical_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'source.zip'; raw = b'raw\r\nbytes\x00'
            with zipfile.ZipFile(archive, 'w') as z:
                z.comment = D.FAILED_SOURCE.encode(); z.writestr('a.py', raw)
            state = {'source':dict(revision=D.FAILED_SOURCE, files_sha256={'a.py':hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()})}
            D.materialize(archive, state, root/'source')
            self.assertEqual((root/'source/a.py').read_bytes(), raw)
            state['source']['files_sha256']['missing'] = '0'*64
            with self.assertRaisesRegex(RuntimeError, 'closure'):
                D.materialize(archive, state, root/'other')

    def test_source_path_escape_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'source.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.comment = D.FAILED_SOURCE.encode(); z.writestr('../escape', b'x')
            state = {'source':dict(revision=D.FAILED_SOURCE, files_sha256={'../escape':hashlib.sha256(b'x').hexdigest()})}
            with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                D.materialize(archive, state, root/'source')
            self.assertFalse((root/'escape').exists())

    def test_child_environment_sanitized(self):
        with mock.patch.dict(os.environ, {'PYTHONOPTIMIZE':'2', 'PYTEST_ADDOPTS':'-k subset', 'PYTEST_PLUGINS':'foreign',
            'SKIMAGEXX_H_EXTREMA_CORE':'foreign', 'SKIMAGEXX_DISABLE_SIMD':'1', 'PYTHONPATH':'foreign'}):
            env = D.environment('normal')
            for name in ('PYTHONOPTIMIZE','PYTEST_ADDOPTS','PYTEST_PLUGINS','SKIMAGEXX_H_EXTREMA_CORE','SKIMAGEXX_DISABLE_SIMD','PYTHONPATH'):
                self.assertNotIn(name, env)
            self.assertEqual(D.environment('scalar')['SKIMAGEXX_DISABLE_SIMD'], '1')
            self.assertEqual(env['OMP_NUM_THREADS'], '1')

    def test_optimized_child_rejected_before_startup(self):
        child = subprocess.run([sys.executable, '-I', '-O', '-B', str(ROOT/'tools/h_installed_diagnostic.py'), '--help'],
            capture_output=True, text=True, env=D.environment())
        self.assertNotEqual(child.returncode, 0)
        self.assertIn('requires unoptimized Python', child.stderr)

    def test_timeout_evidence_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(D.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['child'],1200)):
            self.assertIsNone(D.command(Path(tmp), 'normal', ['child'], {}, 1200))
            record = json.loads((Path(tmp)/'normal-exit.json').read_bytes())
            self.assertTrue(record['timed_out']); self.assertEqual(record['timeout_seconds'], 1200)

    def test_installed_record_accepts_only_archived_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'skimagexx').mkdir(); (root/'x.dist-info').mkdir()
            core='skimagexx/_core.so'; record='x.dist-info/RECORD'; original=b'original binary bytes'
            (root/core).write_bytes(original)
            expected={core:hashlib.sha256(original).hexdigest(),record:'unused-wheel-RECORD-hash'}
            package=dict(members_sha256=expected,sole_core=core)
            def write_record():
                data=(root/core).read_bytes();stream=io.StringIO();writer=csv.writer(stream)
                writer.writerow([core,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(),str(len(data))])
                writer.writerow([record,'','']);(root/record).write_text(stream.getvalue(),newline='')
            write_record();dist=types.SimpleNamespace(locate_file=lambda name:root/name)
            with mock.patch.object(D.importlib.metadata,'distribution',return_value=dist):
                self.assertEqual(D.installed_product(package)['sha256'],expected[core])
                (root/core).write_bytes(b'replaced binary');write_record()
                with self.assertRaisesRegex(RuntimeError,'archived member changed'):
                    D.installed_product(package)

    def reports(self, root):
        ids = ['case-'+str(i) for i in range(886)]; unsafe = set(ids[:14]); core = dict(path='/installed/_core.so', sha256='a'*64)
        output = root/'normal-observations.json'
        summary = dict(logical_ids=886, rows=3544, private_adapter=False, full_API_complete=False, operation_timings=0, policies={})
        for label, fallback in (('strict',False),('default',True)):
            rows = [dict(id=name, workers=w, **(dict(status='source-unsafe-blocked', actual_calls=0) if name in unsafe else
                dict(public_observation_exact=True, intermediate_stages_exact=True, actual={'original_numeric_completed':0}, forwarding_attempts=[])))
                for w in (1,4) for name in ids]
            report = dict(schema='h-extrema-installed-public886-v1', profile='modern', fallback_enabled=fallback, failure=None,
                core=core, default_workers=1, final_workers=1, disable_simd=None, has_avx2=True, private_adapter=False,
                full_API_complete=False, operation_timings=0, observations=rows, guards=[dict(id=str(i),passed=True)for i in range(14)])
            path = root/('normal-observations-'+label+'.json');path.write_text(json.dumps(report))
            path.with_suffix('.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary['policies'][label]=dict(rows=1772,safe=1744,blocked=28,guards=14,fallback_enabled=fallback)
        output.write_text(json.dumps(summary));return output,core,ids,unsafe

    def test_complete_counts_and_journal_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.reports(Path(tmp));D.validate_reports(args[0],'normal',self.data['products']['modern'],*args[1:])

    def test_wrong_policy_subset_relabelled_and_changed_core_rejected(self):
        for kind in ('policy','subset','relabel','core','unsafe','failure'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);args=self.reports(root);path=root/'normal-observations-default.json';r=json.loads(path.read_bytes())
                if kind=='policy':r['fallback_enabled']=False
                if kind=='subset':r['observations'].pop()
                if kind=='relabel':r['observations'][18]['workers']=4
                if kind=='core':r['core']['sha256']='b'*64
                if kind=='unsafe':r['observations'][0]['actual_calls']=1
                if kind=='failure':r['failure']={'type':'AssertionError','message':'retained failure'}
                path.write_text(json.dumps(r));path.with_suffix('.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in r['observations']))
                with self.assertRaises(RuntimeError):D.validate_reports(args[0],'normal',self.data['products']['modern'],*args[1:])


class HelperCheckoutTests(unittest.TestCase):
    def test_exact_raw_helper_declares_text_policy(self):
        helper = ROOT / 'tools/li_installed_diagnostic.py'
        self.assertEqual(D.sha(helper), '3a86441c81f5cc7950353a9048f80d7301741cde32e585198ee8fbbcd5301a68')
        self.assertNotIn(b'\r\n', helper.read_bytes())
        self.assertEqual((ROOT / '.gitattributes').read_bytes().splitlines().count(
            b'tools/li_installed_diagnostic.py -text'), 1)

    def test_real_autocrlf_checkout_of_raw_pinned_helper(self):
        """Exercise the actual dependency bytes; never import the checkout copy."""
        name = 'tools/li_installed_diagnostic.py'
        raw = (ROOT / name).read_bytes()
        expected = '3a86441c81f5cc7950353a9048f80d7301741cde32e585198ee8fbbcd5301a68'
        witnesses = []
        with tempfile.TemporaryDirectory(prefix='h-helper-checkout-', dir=ROOT / 'build') as temporary:
            for declared in (False, True):
                repo = Path(temporary) / ('declared' if declared else 'unruled')
                (repo / 'tools').mkdir(parents=True)
                (repo / name).write_bytes(raw)
                (repo / '.gitattributes').write_bytes((name + ' -text\n').encode() if declared else b'')
                commands = []
                for args in (['-c', 'core.autocrlf=false', 'init', '--quiet'],
                             ['-c', 'core.autocrlf=false', 'add', '--', '.gitattributes', name]):
                    command = ['git', '-C', str(repo), *args]
                    result = subprocess.run(command, check=True, capture_output=True)
                    commands.append(dict(args=command, stdout=result.stdout.decode('utf-8', 'replace'),
                                         stderr=result.stderr.decode('utf-8', 'replace')))
                destination = repo / 'checkout'
                destination.mkdir()
                command = ['git', '-c', 'core.autocrlf=true', '-C', str(repo), 'checkout-index',
                           '--all', '--force', '--prefix=' + str(destination).replace(os.sep, '/') + '/']
                result = subprocess.run(command, check=True, capture_output=True)
                commands.append(dict(args=command, stdout=result.stdout.decode('utf-8', 'replace'),
                                     stderr=result.stderr.decode('utf-8', 'replace')))
                actual = (destination / name).read_bytes()
                digest = hashlib.sha256(actual).hexdigest()
                if declared:
                    self.assertEqual(actual, raw)
                    self.assertEqual(digest, expected)
                else:
                    self.assertEqual(actual, raw.replace(b'\n', b'\r\n'))
                    self.assertNotEqual(digest, expected)
                witnesses.append(dict(declared_text_false=declared, expected_raw_sha256=expected,
                                      actual_raw_sha256=digest, bytes=len(actual), CRLF=actual.count(b'\r\n'),
                                      commands=commands))
        print('H_HELPER_GIT_EOL_WITNESS ' + json.dumps(witnesses, sort_keys=True))

    def test_workflow_profiles_preserve_closed_platform_mapping(self):
        workflow = (ROOT / '.github/workflows/h-installed-diagnostic.yml').read_text(encoding='utf-8')
        choice = workflow.split('      profiles:\n', 1)[1].split('\npermissions:', 1)[0]
        self.assertIn('        type: choice\n', choice)
        self.assertIn('        default: all\n', choice)
        self.assertEqual(re.findall(r'^          - (\w+)$', choice, re.MULTILINE), ['all', 'windows'])
        selector = re.search(r"profile: \$\{\{ fromJSON\(inputs\.profiles == 'windows' && '([^']+)' \|\| '([^']+)'\) \}\}", workflow)
        self.assertIsNotNone(selector)
        self.assertEqual(json.loads(selector[1]), ['windows'])
        self.assertEqual(json.loads(selector[2]), ['minimum', 'modern', 'debug', 'windows'])
        self.assertIn("runs-on: ${{ matrix.profile == 'windows' && 'windows-latest' || 'ubuntu-24.04' }}", workflow)
        self.assertIn("python-version: ${{ matrix.profile == 'minimum' && '3.11.17' || '3.13.16' }}", workflow)


if __name__ == '__main__':
    unittest.main()
