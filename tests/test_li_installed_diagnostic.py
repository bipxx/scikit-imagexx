"""Synthetic/stdlib diagnostic guards; never install or import image packages."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('li_fixed_diagnostic_metadata', ROOT / 'tools/li_installed_diagnostic.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def archive(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as output:
        for name, raw, mode in entries:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = mode << 16
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)  # duplicate synthetic member only
                output.writestr(info, raw)
    stream.seek(0)
    return zipfile.ZipFile(stream)


class SelectedMemberTests(unittest.TestCase):
    def test_exact_regular_member_only(self):
        raw = b'exact selected bytes\x00\xff'
        with archive([('a/member.bin', raw, 0o100644),
                      ('../unselected', b'ignored', 0o100644),
                      ('unselected-link', b'/elsewhere', 0o120777)]) as z:
            self.assertEqual(D.selected_member(z, 'a/member.bin', len(raw), hashlib.sha256(raw).hexdigest()), raw)

    def test_missing_and_duplicate_selected_member(self):
        for entries in ([], [('a', b'v', 0o100644), ('a', b'v', 0o100644)]):
            with self.subTest(entries=len(entries)), archive(entries) as z:
                with self.assertRaisesRegex(RuntimeError, 'missing/duplicate'):
                    D.selected_member(z, 'a', 1, hashlib.sha256(b'v').hexdigest())

    def test_unsafe_names_rejected_before_lookup(self):
        with archive([]) as z:
            for name in ('/absolute', '../parent', 'a/../escape', 'a\\escape', 'C:drive', '//server/share'):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'unsafe selected'):
                    D.selected_member(z, name, 1, '0' * 64)

    def test_selected_link_directory_and_device_rejected(self):
        for name, mode in (('link', 0o120777), ('directory/', 0o040755), ('device', 0o020600)):
            with self.subTest(mode=mode), archive([(name, b'v', mode)]) as z:
                with self.assertRaisesRegex(RuntimeError, 'not regular'):
                    D.selected_member(z, name, 1, hashlib.sha256(b'v').hexdigest())

    def test_wrong_size_zero_and_over_cap_rejected(self):
        with archive([('a', b'v', 0o100644)]) as z:
            for size in (0, 2, 256 * 1024**2 + 1):
                with self.subTest(size=size), self.assertRaisesRegex(RuntimeError, 'size differs'):
                    D.selected_member(z, 'a', size, hashlib.sha256(b'v').hexdigest())

    def test_wrong_hash_rejected(self):
        with archive([('a', b'v', 0o100644)]) as z:
            with self.assertRaisesRegex(RuntimeError, 'hash differs'):
                D.selected_member(z, 'a', 1, '0' * 64)

    def test_crc_corruption_rejected(self):
        raw = b'unique selected payload'
        with archive([('a', raw, 0o100644)]) as original:
            data = original.fp.getvalue()
        self.assertEqual(data.count(raw), 1)
        damaged = data.replace(raw, b'X' + raw[1:], 1)
        with zipfile.ZipFile(io.BytesIO(damaged)) as z:
            with self.assertRaises(zipfile.BadZipFile):
                D.selected_member(z, 'a', len(raw), hashlib.sha256(raw).hexdigest())


class FixedInputsTests(unittest.TestCase):
    def setUp(self):
        self.raw = D.INPUTS.read_bytes()
        self.data = json.loads(self.raw)

    def test_actual_fixed_input_hash_and_profiles(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), D.INPUTS_SHA256)
        self.assertEqual(self.data['schema'], 'li-same-failed-product-fixed-inputs-v2')
        self.assertEqual(self.data['gate_source'], 'controller')
        self.assertEqual(self.data['gate_predecessor_source'], '87b99a57b935e09d64ff569fe2ed900122118603')
        for profile in ('windows', 'modern'):
            product = D.validate_inputs(self.data, profile)
            self.assertEqual(product['source'], '28350015823b36333085ce8ce04e3dcc70065572')
            self.assertEqual(product['run'], 40174473080)
            self.assertEqual(set(product['source_files']), D.SOURCE_FILES)
            self.assertEqual({x['name']: x['version'] for x in product['runtime_wheels'].values()}, product['runtime_pins'])
            self.assertTrue(product['core']['RECORD_verified'])

    def test_predecessor_is_not_relabelled_as_current_controller(self):
        for key, value in (('schema', 'li-same-failed-product-fixed-inputs-v1'),
                           ('gate_source', '87b99a57b935e09d64ff569fe2ed900122118603'),
                           ('gate_predecessor_source', 'f' * 40)):
            changed = copy.deepcopy(self.data)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                D.validate_inputs(changed, 'modern')

    def test_changed_failed_source_or_run_rejected(self):
        for key, value in (('source', 'f' * 40), ('run', 1)):
            data = copy.deepcopy(self.data)
            data['products']['modern'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'original failed product'):
                D.validate_inputs(data, 'modern')

    def test_source_file_missing_or_extra_rejected(self):
        for extra in (False, True):
            data = copy.deepcopy(self.data)
            files = data['products']['modern']['source_files']
            if extra:
                files['tests/alternate.py'] = dict(bytes=1, sha256='0' * 64)
            else:
                files.pop(next(iter(files)))
            with self.subTest(extra=extra), self.assertRaisesRegex(RuntimeError, 'source set differs'):
                D.validate_inputs(data, 'modern')

    def test_no_acceptance_schema_profile_set_and_outer_cap(self):
        for change in ('acceptance', 'schema', 'profile', 'cap'):
            data = copy.deepcopy(self.data)
            if change == 'acceptance':
                data['no_product_acceptance'] = False
            elif change == 'schema':
                data['schema'] = 'alternate'
            elif change == 'profile':
                data['products']['minimum'] = data['products']['modern']
            else:
                data['products']['modern']['artifact']['size_in_bytes'] = D.OUTER_CAP + 1
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                D.validate_inputs(data, 'modern')


class CommandRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='li-diagnostic-metadata-', dir=ROOT / 'build')
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def test_failed_child_keeps_both_streams_and_status(self):
        def failed(args, **kw):
            kw['stdout'].write(b'partial journal announcement\n')
            kw['stderr'].write(b'actual synthetic failure\n')
            return types.SimpleNamespace(returncode=23)
        with mock.patch.object(D.subprocess, 'run', side_effect=failed) as run:
            self.assertEqual(D.command(self.out, 'normal', ['inert-no-process']), 23)
        self.assertEqual(run.call_args.kwargs['timeout'], 180)
        status = json.loads((self.out / 'normal-exit.json').read_bytes())
        self.assertEqual(status, dict(command=['inert-no-process'], exit_code=23, timed_out=False))
        self.assertEqual((self.out / 'normal.stdout').read_bytes(), b'partial journal announcement\n')
        self.assertEqual((self.out / 'normal.stderr').read_bytes(), b'actual synthetic failure\n')

    def test_timeout_retains_status_and_partial_stream(self):
        def expired(args, **kw):
            kw['stdout'].write(b'partial\n')
            raise subprocess.TimeoutExpired(args, kw['timeout'])
        with mock.patch.object(D.subprocess, 'run', side_effect=expired):
            self.assertIsNone(D.command(self.out, 'scalar', ['inert-no-process']))
        status = json.loads((self.out / 'scalar-exit.json').read_bytes())
        self.assertTrue(status['timed_out'])
        self.assertIsNone(status['exit_code'])
        self.assertEqual((self.out / 'scalar.stdout').read_bytes(), b'partial\n')

    def test_command_log_names_cannot_be_reused(self):
        with mock.patch.object(D.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1)):
            D.command(self.out, 'normal', ['first-inert'])
        original = (self.out / 'normal-exit.json').read_bytes()
        with mock.patch.object(D.subprocess, 'run') as run:
            with self.assertRaises(FileExistsError):
                D.command(self.out, 'normal', ['second-inert'])
            run.assert_not_called()
        self.assertEqual((self.out / 'normal-exit.json').read_bytes(), original)


class SourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / 'tools/li_installed_diagnostic.py').read_text(encoding='utf-8')
        self.tree = ast.parse(self.source)
        self.body = ast.unparse(next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == 'diagnostic'))

    def test_preflight_identity_and_gate_before_any_child_call(self):
        interpreter = self.body.index("sha(sys.executable) == product['interpreter']['sha256']")
        install = self.body.index("command(output, 'offline-install'")
        core = self.body.index("'installed archived core differs'")
        admission = self.body.index("save(output / 'admission.json'")
        modes = self.body.index("for mode in ('normal', 'scalar')")
        self.assertLess(interpreter, install)
        self.assertLess(install, core)
        self.assertLess(core, admission)
        self.assertLess(admission, modes)
        self.assertIn("sha(gate) == data['gate_sha256']", self.body[:install])
        gate = self.body.index("sha(gate) == data['gate_sha256']")
        self.assertLess(gate, interpreter)
        self.assertLess(gate, self.body.index("subprocess.check_output(['gh', 'api', endpoint])"))

    def test_offline_binary_hash_install_only(self):
        for flag in ('--no-index', '--no-deps', '--require-hashes', '--only-binary=:all:', '--force-reinstall', '--find-links'):
            self.assertIn(repr(flag), self.body)
        self.assertNotIn("'wheel',", self.body)
        self.assertNotIn('cmake', self.body)
        self.assertNotIn('perf_counter', self.source)

    def test_source_only_gate_overlay_and_isolated_modes(self):
        self.assertEqual(self.body.count('target_gate.write_bytes'), 1)
        self.assertIn("env.pop('PYTHONPATH', None)", self.body)
        self.assertIn("env.pop('SKIMAGEXX_DISABLE_SIMD', None)", self.body)
        self.assertIn("env['SKIMAGEXX_DISABLE_SIMD'] = '1'", self.body)
        self.assertIn("[sys.executable, '-I', '-X', 'utf8', '-B'", self.body)
        self.assertLess(self.body.index("save(output / 'diagnosis.json'"), self.body.index('require(all(journals.values())'))

    def test_workflow_retains_failure_and_has_no_build(self):
        workflow = (ROOT / '.github/workflows/li-installed-diagnostic.yml').read_text(encoding='utf-8')
        self.assertIn('fail-fast: false', workflow)
        self.assertIn('if: always()', workflow)
        self.assertIn('include-hidden-files: true', workflow)
        self.assertIn('set -euo pipefail', workflow)
        self.assertIn("python-version: '3.13.16'", workflow)
        self.assertIn('--controller-revision "$CONTROLLER_REVISION"', workflow)
        metadata = workflow.index('unittest discover -s tests -p test_li_installed_diagnostic.py -v')
        preparation = workflow.index('tools/li_installed_diagnostic.py --profile')
        self.assertLess(metadata, preparation)
        self.assertLess(workflow.index('mkdir -p build'), metadata)
        self.assertIn('build/li-diagnostic-metadata-${{ matrix.profile }}.log', workflow[workflow.index('actions/upload-artifact@v4'):])
        for forbidden in ('cmake --build', 'pip wheel', 'build-whole-product', 'workflow run'):
            self.assertNotIn(forbidden, workflow)


class GateCheckoutTests(unittest.TestCase):
    def test_exact_current_gate_declares_raw_text_policy(self):
        data = json.loads(D.INPUTS.read_bytes())
        raw = (ROOT / 'tests/test_li_u16_pipeline.py').read_bytes()
        self.assertNotIn(b'\r\n', raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), data['gate_sha256'])
        rules = (ROOT / '.gitattributes').read_bytes().splitlines()
        self.assertEqual(rules.count(b'tests/test_li_u16_pipeline.py -text'), 1)

    def test_real_autocrlf_checkout_with_and_without_exact_rule(self):
        """Git only; two new synthetic repos, no old source/archive mutation."""
        raw = (ROOT / 'tests/test_li_u16_pipeline.py').read_bytes()
        expected = json.loads(D.INPUTS.read_bytes())['gate_sha256']
        witnesses = []
        with tempfile.TemporaryDirectory(prefix='li-gate-checkout-', dir=ROOT / 'build') as temporary:
            for declared in (False, True):
                repo = Path(temporary) / ('declared' if declared else 'unruled')
                repo.mkdir()
                (repo / 'tests').mkdir()
                (repo / 'tests/test_li_u16_pipeline.py').write_bytes(raw)
                (repo / '.gitattributes').write_bytes(b'tests/test_li_u16_pipeline.py -text\n' if declared else b'')
                commands = []

                def git(*args):
                    command = ['git', '-c', 'core.autocrlf=false', '-C', str(repo), *args]
                    result = subprocess.run(command, check=True, capture_output=True)
                    commands.append(dict(args=command, stdout=result.stdout.decode('utf-8', 'replace'),
                                         stderr=result.stderr.decode('utf-8', 'replace')))
                    return result.stdout

                git('init', '--quiet')
                git('add', '--', '.gitattributes', 'tests/test_li_u16_pipeline.py')
                destination = repo / 'checkout'
                destination.mkdir()
                command = ['git', '-c', 'core.autocrlf=true', '-C', str(repo), 'checkout-index',
                           '--all', '--force', '--prefix=' + str(destination).replace(os.sep, '/') + '/']
                result = subprocess.run(command, check=True, capture_output=True)
                commands.append(dict(args=command, stdout=result.stdout.decode('utf-8', 'replace'),
                                     stderr=result.stderr.decode('utf-8', 'replace')))
                actual = (destination / 'tests/test_li_u16_pipeline.py').read_bytes()
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
        print('LI_GATE_GIT_EOL_WITNESS ' + json.dumps(witnesses, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
